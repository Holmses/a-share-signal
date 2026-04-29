from __future__ import annotations

from datetime import date
from pathlib import Path

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.portfolio.engine import PortfolioState
from ashare_signal.report.render import render_markdown, write_markdown
from ashare_signal.strategy.selector import UniverseSignalSelector
from ashare_signal.strategy.signal_board import SignalStrategy
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


def run_daily_signal_job(
    config: AppConfig,
    base_dir: Path,
    as_of: date,
    holdings_path: Path,
) -> Path:
    repository = DataRepository(config=config, base_dir=base_dir)
    requested_trade_date = to_compact_date(as_of)
    actual_trade_date = repository.resolve_trade_date(requested_trade_date)
    try:
        next_trade_date = repository.next_open_trade_date(actual_trade_date)
        calendar_note = None
    except ValueError:
        fallback = SignalStrategy._next_business_day(parse_compact_date(actual_trade_date))
        next_trade_date = to_compact_date(fallback)
        calendar_note = (
            f"交易日历缓存未覆盖 {actual_trade_date} 之后的开市日，"
            f"生效日期暂按下一个工作日 {fallback.isoformat()} 推断。"
        )

    positions = repository.load_positions(holdings_path)
    portfolio = PortfolioState(
        positions=positions,
        max_positions=config.max_positions,
    )
    sellable_positions = portfolio.sellable_positions(
        as_of=parse_compact_date(actual_trade_date),
        min_holding_days=config.market.min_position_holding_days,
    )

    if not sellable_positions and positions:
        notes = ["当前持仓都不满足最短持有天数约束，卖出候选为空。"]
    else:
        notes = []
    if calendar_note:
        notes.append(calendar_note)

    universe = repository.load_universe_snapshot(actual_trade_date)
    selector = UniverseSignalSelector(
        selection_config=config.selection,
        top_buy_n=config.strategy.buy_top_n,
        top_sell_n=config.strategy.sell_top_n,
    )
    selection = selector.select(universe=universe, positions=sellable_positions or positions)
    notes.extend(selection.notes)

    top_buy = selection.buy_candidates[0] if selection.buy_candidates else None
    top_sell = selection.sell_candidates[0] if selection.sell_candidates else None
    if len(portfolio.positions) >= config.max_positions:
        if selector.should_rotate(top_buy, top_sell):
            notes.append("当前组合已满仓，买卖候选分差满足换仓门槛，建议执行 1 卖 1 买。")
        else:
            notes.append("当前组合已满仓，但买卖候选分差未超过换仓门槛，建议继续持有观望。")
    elif selector.should_open_new_position(top_buy):
        notes.append("买入候选分数达到开仓门槛，可考虑补至目标仓位。")
    elif top_buy is not None:
        notes.append("买入候选已生成，但分数未达到开仓门槛，建议暂不新增仓位。")

    strategy = SignalStrategy(config=config)
    board = strategy.build_board(
        signal_date=as_of,
        trade_date=parse_compact_date(actual_trade_date),
        effective_date=parse_compact_date(next_trade_date),
        holdings_count=len(portfolio.positions),
        buy_candidates=selection.buy_candidates,
        sell_candidates=selection.sell_candidates if sellable_positions else [],
        notes=notes,
    )
    markdown = render_markdown(board)
    output_path = base_dir / config.paths.reports_dir / f"signal-board-{actual_trade_date}.md"
    write_markdown(markdown, output_path)
    return output_path
