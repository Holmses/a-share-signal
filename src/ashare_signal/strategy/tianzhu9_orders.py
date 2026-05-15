from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.backtest.full_a_momentum import FullAMomentumBacktestEngine
from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.exit_rules import tiered_trailing_take_profit
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


@dataclass(slots=True)
class Tianzhu9Order:
    action: str
    symbol: str
    name: str
    limit_price: float | None
    quantity: int | None
    rank: int | None
    score: float | None
    reason: str
    entry_price: float | None = None
    last_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    unrealized_return: float | None = None
    holding_days: int | None = None


@dataclass(slots=True)
class Tianzhu9OrderPlan:
    signal_trade_date: str
    planned_trade_date: str
    buy_orders: list[Tianzhu9Order]
    sell_orders: list[Tianzhu9Order]
    hold_orders: list[Tianzhu9Order]
    notes: list[str]
    markdown_path: Path
    json_path: Path


def generate_tianzhu9_order_plan(
    config: AppConfig,
    repository: DataRepository,
    base_dir: Path,
    as_of: date | None = None,
    positions_path: Path | None = None,
    top_n: int = 5,
    hold_days: int = 5,
    max_hold_days: int = 10,
    hard_exit_days: int | None = 23,
    min_avg_amount_yuan: float = 50_000_000.0,
) -> Tianzhu9OrderPlan:
    requested_date = to_compact_date(as_of or date.today())
    signal_trade_date = repository.resolve_trade_date(requested_date)
    planned_trade_date = repository.next_open_trade_date(signal_trade_date)

    cached_dates = repository.complete_daily_cache_dates(end_date=signal_trade_date)
    if signal_trade_date not in cached_dates:
        raise ValueError(f"No complete daily cache is available for {signal_trade_date}.")
    signal_index = cached_dates.index(signal_trade_date)
    required_signal_history = SelectionEventStudyEngine.minimum_signal_history_trade_days()
    if signal_index < required_signal_history:
        cached_start = cached_dates[0]
        suggested_sync_start = to_compact_date(
            SelectionEventStudyEngine.recommended_sync_start_date(
                repository=repository,
                target_date=signal_trade_date,
                prior_trade_days=required_signal_history,
            )
        )
        raise ValueError(
            "Tianzhu9 order generation needs at least "
            f"{required_signal_history} complete trade days before signal date {signal_trade_date} "
            f"for factor warm-up, but only found {signal_index}. "
            f"Current cache starts at {cached_start}. "
            f"Sync from {suggested_sync_start} or earlier and rerun."
        )
    feature_dates = cached_dates[
        max(0, signal_index - SelectionEventStudyEngine.factor_history_trade_days()) : signal_index + 1
    ]
    selection_engine = SelectionEventStudyEngine(
        config=config,
        repository=repository,
        base_dir=base_dir,
        top_n_per_group=top_n,
        min_avg_amount_yuan=min_avg_amount_yuan,
        groups=["main", "chinext", "star"],
        variants=["quality_momentum"],
        horizons=[1],
    )
    engine = FullAMomentumBacktestEngine(
        config=config,
        repository=repository,
        base_dir=base_dir,
        top_n=top_n,
        hold_days=hold_days,
        max_hold_days=max_hold_days,
        max_positions=config.market.max_positions,
        groups=["main", "chinext", "star"],
        selection_variant="quality_momentum",
        min_avg_amount_yuan=min_avg_amount_yuan,
        market_min_breadth=0.50,
        market_min_return_20d=0.0,
        style_min_breadth=0.48,
        style_min_return_20d=-0.01,
    )
    factor_frame = selection_engine._build_factor_frame(feature_dates)
    signal_frame = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_trade_date].copy()
    style_state = engine._market_style_state(signal_frame)
    risk_off = bool(style_state["market_risk_off"])
    eligible_groups = set(style_state["eligible_groups"])
    selected = engine._select_candidates(
        signal_frame=signal_frame,
        eligible_groups=eligible_groups,
        excluded_symbols=set(),
        risk_off=risk_off,
    )
    selected_symbols = {str(row["ts_code"]) for row in selected}
    selected_by_symbol = {str(row["ts_code"]): row for row in selected}

    positions = _load_tianzhu9_positions(positions_path or _default_positions_path(base_dir))
    buy_orders = _build_buy_orders(
        config=config,
        selected=selected,
        held_symbols={position["symbol"] for position in positions},
        max_positions=config.market.max_positions,
    )
    sell_orders, hold_orders = _build_position_orders(
        config=config,
        factor_frame=factor_frame,
        signal_trade_date=signal_trade_date,
        selected_symbols=selected_symbols,
        selected_by_symbol=selected_by_symbol,
        positions=positions,
        cached_dates=cached_dates,
        hold_days=hold_days,
        max_hold_days=max_hold_days,
        hard_exit_days=hard_exit_days,
        risk_off=risk_off,
        eligible_groups=eligible_groups,
    )
    notes = []
    market_breadth = float(style_state["market_breadth"])
    market_return_20d = float(style_state["market_return_20d"])
    if risk_off:
        notes.append(
            "全 A 严格过滤：市场风控未通过，"
            f"breadth={market_breadth:.2%}，20日中位收益={market_return_20d:.2%}，本次不新开仓。"
        )
    else:
        groups_text = "、".join(_format_group_name(group) for group in sorted(eligible_groups)) or "无"
        notes.append(
            "全 A 严格过滤：市场风控通过，"
            f"breadth={market_breadth:.2%}，20日中位收益={market_return_20d:.2%}，"
            f"允许开仓分组：{groups_text}。"
        )
    if not positions:
        notes.append("未发现 Tianzhu9 持仓文件或持仓为空，本次只生成买入计划。")
    if not selected:
        notes.append("今日未选出符合全 A 严格过滤条件的目标，或市场/风格过滤未通过。")
    if hard_exit_days is None:
        notes.append(
            "卖出规则：不使用硬止损/固定持有天数退出，"
            "仅在盈利后触发 8%/4%、12%/6%、20%/8% 分层追踪止盈。"
        )
    else:
        notes.append(
            "卖出规则：不使用硬止损，先执行盈利后分层追踪止盈，"
            f"未触发止盈则持仓满 {hard_exit_days} 个交易日硬卖出。"
        )

    reports_dir = base_dir / config.paths.reports_dir / "tianzhu9-orders"
    reports_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = reports_dir / f"tianzhu9-orders-{signal_trade_date}.md"
    json_path = reports_dir / f"tianzhu9-orders-{signal_trade_date}.json"
    plan = Tianzhu9OrderPlan(
        signal_trade_date=signal_trade_date,
        planned_trade_date=planned_trade_date,
        buy_orders=buy_orders,
        sell_orders=sell_orders,
        hold_orders=hold_orders,
        notes=notes,
        markdown_path=markdown_path,
        json_path=json_path,
    )
    markdown = render_tianzhu9_order_plan(plan)
    markdown_path.write_text(markdown, encoding="utf-8")
    _write_plan_json(plan, json_path)
    return plan


def render_tianzhu9_order_plan(plan: Tianzhu9OrderPlan) -> str:
    lines = [
        "# Tianzhu9 全 A 严格过滤调仓计划",
        "",
        f"- 信号日：{_format_trade_date(plan.signal_trade_date)}",
        f"- 计划交易日：{_format_trade_date(plan.planned_trade_date)}",
        f"- 买入：{len(plan.buy_orders)}",
        f"- 卖出：{len(plan.sell_orders)}",
        f"- 继续持有：{len(plan.hold_orders)}",
        "",
        "## 买入计划",
    ]
    lines.extend(_render_buy_order_table(plan.buy_orders, empty="无买入计划。"))
    lines.extend(["", "## 卖出计划"])
    lines.extend(_render_sell_order_table(plan.sell_orders, empty="无卖出计划。"))
    lines.extend(["", "## 继续持有"])
    lines.extend(_render_hold_order_table(plan.hold_orders, empty="无继续持有。"))
    if plan.notes:
        lines.extend(["", "## 备注"])
        lines.extend(f"- {note}" for note in plan.notes)
    return "\n".join(lines) + "\n"


def plan_to_feishu_text(plan: Tianzhu9OrderPlan) -> str:
    lines = [
        f"Tianzhu9 全 A 严格过滤调仓计划 {_format_trade_date(plan.planned_trade_date)}",
        f"信号日：{_format_trade_date(plan.signal_trade_date)}",
        "",
        f"买入：{len(plan.buy_orders)}  卖出：{len(plan.sell_orders)}  持有：{len(plan.hold_orders)}",
    ]
    for title, orders in (("买入", plan.buy_orders), ("卖出", plan.sell_orders), ("继续持有", plan.hold_orders)):
        lines.append("")
        lines.append(f"{title}:")
        if not orders:
            lines.append("- 无")
            continue
        for order in orders[:8]:
            lines.append(f"- {_render_order_text_line(order)}")
            lines.append(f"  {order.reason}")
    if plan.notes:
        lines.append("")
        lines.append("备注:")
        lines.extend(f"- {note}" for note in plan.notes[:5])
    return "\n".join(lines)


def _build_buy_orders(
    config: AppConfig,
    selected: list[dict],
    held_symbols: set[str],
    max_positions: int,
) -> list[Tianzhu9Order]:
    orders = []
    free_slots = max(int(max_positions) - len(held_symbols), 0)
    for candidate in selected:
        if len(orders) >= free_slots:
            break
        symbol = str(candidate["ts_code"])
        if symbol in held_symbols:
            continue
        close_price = float(candidate["close"])
        limit_price = round(close_price * (1 + config.pricing.buy_markup), 2)
        orders.append(
            Tianzhu9Order(
                action="BUY",
                symbol=symbol,
                name=str(candidate.get("name") or symbol),
                limit_price=limit_price,
                quantity=None,
                rank=int(candidate["rank"]),
                score=float(candidate["score"]),
                reason=f"T-1 排名第 {int(candidate['rank'])}，买入限价 = T-1 收盘价 {close_price:.2f} * (1 + {config.pricing.buy_markup:.3%})。",
            )
        )
    return orders


def _build_position_orders(
    config: AppConfig,
    factor_frame: pd.DataFrame,
    signal_trade_date: str,
    selected_symbols: set[str],
    selected_by_symbol: dict[str, dict],
    positions: list[dict],
    hold_days: int,
    max_hold_days: int,
    cached_dates: list[str] | None = None,
    hard_exit_days: int | None = None,
    risk_off: bool = False,
    eligible_groups: set[str] | None = None,
) -> tuple[list[Tianzhu9Order], list[Tianzhu9Order]]:
    sell_orders: list[Tianzhu9Order] = []
    hold_orders: list[Tianzhu9Order] = []
    for position in positions:
        symbol = str(position["symbol"])
        quantity = int(position["quantity"])
        entry_price = float(position["entry_price"])
        entry_date = date.fromisoformat(_normalize_iso_date(str(position["entry_date"])))
        holding_days = _holding_trade_days(
            cached_dates=cached_dates,
            entry_date=entry_date,
            signal_trade_date=signal_trade_date,
        )
        selected_row = selected_by_symbol.get(symbol)
        rank = int(selected_row["rank"]) if selected_row is not None else None
        score = float(selected_row["score"]) if selected_row is not None else None
        feature = FullAMomentumBacktestEngine._feature_row(factor_frame, signal_trade_date, symbol)
        if feature is None:
            hold_orders.append(
                _position_order(
                    "HOLD",
                    position,
                    None,
                    "持仓标的未出现在今日候选特征中，暂按观察处理。",
                    rank=rank,
                    score=score,
                    holding_days=holding_days,
                )
            )
            continue

        prev_close = float(feature["close"])
        highest_close = float(position.get("highest_close") or max(prev_close, entry_price))
        highest_high = float(position.get("highest_high") or highest_close)
        highest_price = max(highest_close, highest_high)
        market_value = prev_close * quantity
        unrealized_pnl = market_value - entry_price * quantity
        unrealized_return = (prev_close / entry_price - 1.0) if entry_price else None

        reason = None
        exit_check = tiered_trailing_take_profit(
            entry_price=entry_price,
            current_close=prev_close,
            highest_price=highest_price,
        )
        if exit_check.should_exit:
            reason = (
                "分层追踪止盈："
                f"最高浮盈 {exit_check.peak_profit_pct:.2%}，"
                f"从高点回撤 {exit_check.drawdown_from_peak_pct:.2%}，"
                f"触发 {exit_check.trigger_profit_pct:.0%}/{exit_check.trigger_drawdown_pct:.0%} 档。"
            )
        elif hard_exit_days is not None and holding_days >= hard_exit_days:
            reason = f"硬卖出：持仓满 {hard_exit_days} 个交易日。"

        if reason:
            limit_price = round(prev_close * (1 - config.pricing.sell_markdown), 2)
            sell_orders.append(
                _position_order(
                    "SELL",
                    position,
                    limit_price,
                    reason,
                    rank=rank,
                    score=score,
                    last_price=prev_close,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_return=unrealized_return,
                    holding_days=holding_days,
                )
            )
        else:
            if symbol in selected_symbols:
                reason = "今日重复入选，未触发分层追踪止盈。"
            else:
                reason = f"未触发分层追踪止盈，当前持有 {holding_days} 天。"
            hold_orders.append(
                _position_order(
                    "HOLD",
                    position,
                    None,
                    reason,
                    rank=rank,
                    score=score,
                    last_price=prev_close,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_return=unrealized_return,
                    holding_days=holding_days,
                )
            )
    return sell_orders, hold_orders


def _holding_trade_days(
    *,
    cached_dates: list[str] | None,
    entry_date: date,
    signal_trade_date: str,
) -> int:
    if cached_dates and signal_trade_date in cached_dates:
        entry_trade_date = to_compact_date(entry_date)
        eligible_entries = [
            trade_date
            for trade_date in cached_dates
            if entry_trade_date <= trade_date <= signal_trade_date
        ]
        if eligible_entries:
            return cached_dates.index(signal_trade_date) - cached_dates.index(eligible_entries[0]) + 1
    signal_day = parse_compact_date(signal_trade_date)
    return max((signal_day - entry_date).days + 1, 1)


def _load_tianzhu9_positions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    required = {"symbol", "name", "entry_date", "entry_price", "quantity"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Tianzhu9 position columns in {path}: {sorted(missing)}")
    rows = frame.to_dict(orient="records")
    for row in rows:
        row["symbol"] = str(row["symbol"])
        row["name"] = str(row["name"])
        row["entry_date"] = _normalize_iso_date(str(row["entry_date"]))
        row["entry_price"] = float(row["entry_price"])
        row["quantity"] = int(row["quantity"])
        if "highest_close" not in row or pd.isna(row["highest_close"]):
            row["highest_close"] = row["entry_price"]
        if "highest_high" not in row or pd.isna(row["highest_high"]):
            row["highest_high"] = row["highest_close"]
    return rows


def _position_order(
    action: str,
    position: dict,
    limit_price: float | None,
    reason: str,
    *,
    rank: int | None = None,
    score: float | None = None,
    last_price: float | None = None,
    market_value: float | None = None,
    unrealized_pnl: float | None = None,
    unrealized_return: float | None = None,
    holding_days: int | None = None,
) -> Tianzhu9Order:
    return Tianzhu9Order(
        action=action,
        symbol=str(position["symbol"]),
        name=str(position["name"]),
        limit_price=limit_price,
        quantity=int(position["quantity"]),
        rank=rank,
        score=score,
        reason=reason,
        entry_price=float(position["entry_price"]),
        last_price=last_price,
        market_value=market_value,
        unrealized_pnl=unrealized_pnl,
        unrealized_return=unrealized_return,
        holding_days=holding_days,
    )


def _render_buy_order_table(orders: list[Tianzhu9Order], empty: str) -> list[str]:
    if not orders:
        return [empty]
    rows = [
        [
            order.symbol,
            order.name,
            _format_price(order.limit_price, default="观察"),
            _format_rank(order.rank),
            _format_score(order.score),
            order.reason,
        ]
        for order in orders
    ]
    return _render_markdown_table(["代码", "名称", "计划买入价", "rank", "score", "原因"], rows)


def _render_sell_order_table(orders: list[Tianzhu9Order], empty: str) -> list[str]:
    if not orders:
        return [empty]
    rows = [
        [
            order.symbol,
            order.name,
            _format_quantity(order.quantity),
            _format_price(order.entry_price),
            _format_price(order.last_price),
            _format_price(order.limit_price),
            _format_signed_money(order.unrealized_pnl),
            _format_pct(order.unrealized_return),
            _format_holding_days(order.holding_days),
            order.reason,
        ]
        for order in orders
    ]
    return _render_markdown_table(
        ["代码", "名称", "数量", "买入价", "现价", "计划卖价", "浮盈亏", "收益率", "持有天数", "原因"],
        rows,
    )


def _render_hold_order_table(orders: list[Tianzhu9Order], empty: str) -> list[str]:
    if not orders:
        return [empty]
    rows = [
        [
            order.symbol,
            order.name,
            _format_quantity(order.quantity),
            _format_price(order.entry_price),
            _format_price(order.last_price),
            _format_money(order.market_value),
            _format_signed_money(order.unrealized_pnl),
            _format_pct(order.unrealized_return),
            _format_holding_days(order.holding_days),
            _format_rank(order.rank),
            _format_score(order.score),
            order.reason,
        ]
        for order in orders
    ]
    return _render_markdown_table(
        ["代码", "名称", "数量", "买入价", "现价", "市值", "浮盈亏", "收益率", "持有天数", "rank", "score", "原因"],
        rows,
    )


def _render_order_text_line(order: Tianzhu9Order) -> str:
    parts = [f"{order.symbol} {order.name}"]
    if order.action == "BUY":
        parts.append(f"价:{_format_price(order.limit_price, default='观察')}")
    elif order.action == "SELL":
        parts.append(f"卖价:{_format_price(order.limit_price, default='观察')}")
    if order.quantity is not None:
        parts.append(f"数量:{order.quantity}")
    if order.entry_price is not None:
        parts.append(f"买入:{order.entry_price:.2f}")
    if order.last_price is not None:
        parts.append(f"现价:{order.last_price:.2f}")
    if order.unrealized_pnl is not None:
        parts.append(
            f"浮盈亏:{_format_signed_money(order.unrealized_pnl)} ({_format_pct(order.unrealized_return)})"
        )
    if order.holding_days is not None:
        parts.append(f"持有:{order.holding_days}天")
    if order.rank is not None:
        parts.append(f"rank:{order.rank}")
    if order.score is not None:
        parts.append(f"score:{order.score:.4f}")
    return " ".join(parts)


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return lines


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_price(value: float | None, default: str = "-") -> str:
    return default if value is None else f"{value:.2f}"


def _format_money(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def _format_signed_money(value: float | None) -> str:
    return "-" if value is None else f"{value:+,.2f}"


def _format_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:+.2%}"


def _format_quantity(value: int | None) -> str:
    return "-" if value is None else str(value)


def _format_holding_days(value: int | None) -> str:
    return "-" if value is None else f"{value}天"


def _format_rank(value: int | None) -> str:
    return "-" if value is None else str(value)


def _format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _format_group_name(value: str) -> str:
    return {
        "main": "主板",
        "chinext": "创业板",
        "star": "科创板",
        "bse": "北交所",
    }.get(value, value or "-")


def _write_plan_json(plan: Tianzhu9OrderPlan, path: Path) -> None:
    payload = {
        "signal_trade_date": plan.signal_trade_date,
        "planned_trade_date": plan.planned_trade_date,
        "buy_orders": [asdict(order) for order in plan.buy_orders],
        "sell_orders": [asdict(order) for order in plan.sell_orders],
        "hold_orders": [asdict(order) for order in plan.hold_orders],
        "notes": plan.notes,
        "markdown_path": str(plan.markdown_path),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _default_positions_path(base_dir: Path) -> Path:
    return base_dir / "data" / "positions" / "tianzhu9_positions.csv"


def _format_trade_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}" if len(value) == 8 and value.isdigit() else value


def _normalize_iso_date(value: str) -> str:
    clean = value.strip()
    if len(clean) == 8 and clean.isdigit():
        return f"{clean[:4]}-{clean[4:6]}-{clean[6:]}"
    return clean
