from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.backtest.tianzhu9_like import Tianzhu9LikeBacktestEngine
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
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
    top_n: int = 1,
    hold_days: int = 2,
    max_hold_days: int = 4,
    min_avg_amount_yuan: float = 50_000_000.0,
) -> Tianzhu9OrderPlan:
    requested_date = to_compact_date(as_of or date.today())
    signal_trade_date = repository.resolve_trade_date(requested_date)
    planned_trade_date = repository.next_open_trade_date(signal_trade_date)

    cached_dates = repository.complete_daily_cache_dates(end_date=signal_trade_date)
    if signal_trade_date not in cached_dates:
        raise ValueError(f"No complete daily cache is available for {signal_trade_date}.")
    signal_index = cached_dates.index(signal_trade_date)
    feature_dates = cached_dates[max(0, signal_index - 100) : signal_index + 1]
    engine = Tianzhu9LikeBacktestEngine(
        config=config,
        repository=repository,
        base_dir=base_dir,
        top_n=top_n,
        hold_days=hold_days,
        max_hold_days=max_hold_days,
        min_avg_amount_yuan=min_avg_amount_yuan,
        execution_mode="limit-swing",
        extend_on_repeat=True,
    )
    factor_frame = engine._build_factor_frame(feature_dates)
    selected = engine._select_candidates(factor_frame, signal_trade_date)
    selected_symbols = {row["ts_code"] for row in selected}

    positions = _load_tianzhu9_positions(positions_path or _default_positions_path(base_dir))
    buy_orders = _build_buy_orders(
        config=config,
        selected=selected,
        held_symbols={position["symbol"] for position in positions},
    )
    sell_orders, hold_orders = _build_position_orders(
        config=config,
        factor_frame=factor_frame,
        signal_trade_date=signal_trade_date,
        selected_symbols=selected_symbols,
        positions=positions,
        hold_days=hold_days,
        max_hold_days=max_hold_days,
    )
    notes = []
    if not positions:
        notes.append("未发现 Tianzhu9 持仓文件或持仓为空，本次只生成买入计划。")
    if not selected:
        notes.append("今日未选出符合条件的创业板目标。")

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
        "# Tianzhu9 调仓计划",
        "",
        f"- 信号日：{_format_trade_date(plan.signal_trade_date)}",
        f"- 计划交易日：{_format_trade_date(plan.planned_trade_date)}",
        f"- 买入：{len(plan.buy_orders)}",
        f"- 卖出：{len(plan.sell_orders)}",
        f"- 继续持有：{len(plan.hold_orders)}",
        "",
        "## 买入计划",
    ]
    lines.extend(_render_order_lines(plan.buy_orders, empty="无买入计划。"))
    lines.extend(["", "## 卖出计划"])
    lines.extend(_render_order_lines(plan.sell_orders, empty="无卖出计划。"))
    lines.extend(["", "## 继续持有"])
    lines.extend(_render_order_lines(plan.hold_orders, empty="无继续持有。"))
    if plan.notes:
        lines.extend(["", "## 备注"])
        lines.extend(f"- {note}" for note in plan.notes)
    return "\n".join(lines) + "\n"


def plan_to_feishu_text(plan: Tianzhu9OrderPlan) -> str:
    lines = [
        f"Tianzhu9 调仓计划 {_format_trade_date(plan.planned_trade_date)}",
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
            price = "观察" if order.limit_price is None else f"{order.limit_price:.2f}"
            rank = "-" if order.rank is None else str(order.rank)
            score = "-" if order.score is None else f"{order.score:.4f}"
            lines.append(f"- {order.symbol} {order.name} 价:{price} rank:{rank} score:{score}")
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
) -> list[Tianzhu9Order]:
    orders = []
    for candidate in selected:
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
    positions: list[dict],
    hold_days: int,
    max_hold_days: int,
) -> tuple[list[Tianzhu9Order], list[Tianzhu9Order]]:
    sell_orders: list[Tianzhu9Order] = []
    hold_orders: list[Tianzhu9Order] = []
    signal_day = parse_compact_date(signal_trade_date)
    for position in positions:
        symbol = str(position["symbol"])
        feature = Tianzhu9LikeBacktestEngine._feature_row(factor_frame, signal_trade_date, symbol)
        if feature is None:
            hold_orders.append(_position_order("HOLD", position, None, "持仓标的未出现在今日候选特征中，暂按观察处理。"))
            continue

        prev_close = float(feature["close"])
        ma_5 = float(feature["ma_5"])
        ma_10 = float(feature["ma_10"])
        entry_price = float(position["entry_price"])
        highest_close = float(position.get("highest_close") or max(prev_close, entry_price))
        entry_date = date.fromisoformat(_normalize_iso_date(str(position["entry_date"])))
        holding_days = max((signal_day - entry_date).days + 1, 1)
        pnl_pct = prev_close / entry_price - 1.0
        high_profit_pct = highest_close / entry_price - 1.0
        drawdown_from_high = prev_close / highest_close - 1.0 if highest_close else 0.0

        reason = None
        if pnl_pct <= -0.05:
            reason = f"硬止损：按 T-1 收盘价计算收益 {pnl_pct:.2%}。"
        elif high_profit_pct >= 0.08 and drawdown_from_high <= -0.04:
            reason = f"移动止盈：曾盈利 {high_profit_pct:.2%}，从高点回撤 {drawdown_from_high:.2%}。"
        elif holding_days >= max_hold_days:
            reason = f"达到最长持有 {max_hold_days} 天。"
        elif holding_days >= hold_days and symbol not in selected_symbols and prev_close < ma_5:
            reason = f"持仓满 {hold_days} 天、不再入选，且跌破 5 日线。"
        elif holding_days >= hold_days and prev_close < ma_10:
            reason = f"持仓满 {hold_days} 天，且跌破 10 日线。"

        if reason:
            limit_price = round(prev_close * (1 - config.pricing.sell_markdown), 2)
            sell_orders.append(_position_order("SELL", position, limit_price, reason))
        else:
            if symbol in selected_symbols:
                reason = f"重复入选，未达到最长持有 {max_hold_days} 天。"
            else:
                reason = f"未触发止损/止盈/均线退出，当前持有 {holding_days} 天。"
            hold_orders.append(_position_order("HOLD", position, None, reason))
    return sell_orders, hold_orders


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
    return rows


def _position_order(action: str, position: dict, limit_price: float | None, reason: str) -> Tianzhu9Order:
    return Tianzhu9Order(
        action=action,
        symbol=str(position["symbol"]),
        name=str(position["name"]),
        limit_price=limit_price,
        quantity=int(position["quantity"]),
        rank=None,
        score=None,
        reason=reason,
    )


def _render_order_lines(orders: list[Tianzhu9Order], empty: str) -> list[str]:
    if not orders:
        return [empty]
    lines = []
    for order in orders:
        price = "观察" if order.limit_price is None else f"{order.limit_price:.2f}"
        quantity = "-" if order.quantity is None else str(order.quantity)
        rank = "-" if order.rank is None else str(order.rank)
        score = "-" if order.score is None else f"{order.score:.4f}"
        lines.append(f"- {order.symbol} {order.name}，价格：{price}，数量：{quantity}，rank：{rank}，score：{score}")
        lines.append(f"  - {order.reason}")
    return lines


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
