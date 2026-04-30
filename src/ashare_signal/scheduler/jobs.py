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
        min_holding_days=config.selection.rotation_min_holding_days,
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
    buy_selection = selector.select(universe=universe, positions=positions)
    sell_selection = selector.select(universe=universe, positions=sellable_positions)
    notes.extend(buy_selection.notes)
    notes.extend(note for note in sell_selection.notes if note not in notes)

    top_buy = buy_selection.buy_candidates[0] if buy_selection.buy_candidates else None
    top_sell = sell_selection.sell_candidates[0] if sell_selection.sell_candidates else None
    executable_buy_candidates = []
    executable_sell_candidates = []
    market_allows_buy = selector.market_allows_buy(universe)
    if not market_allows_buy:
        notes.append("市场宽度未达到买入门槛，今日不生成买入执行信号。")
    if len(portfolio.positions) >= config.max_positions:
        if market_allows_buy and selector.should_rotate(top_buy, top_sell):
            notes.append("当前组合已满仓，买卖候选分差满足换仓门槛，建议执行 1 卖 1 买。")
            executable_buy_candidates = buy_selection.buy_candidates
            executable_sell_candidates = sell_selection.sell_candidates
        else:
            notes.append("当前组合已满仓，但买卖候选分差未超过换仓门槛，建议继续持有观望。")
    elif market_allows_buy and selector.should_open_new_position(top_buy):
        notes.append("当前组合未满仓，买入候选分数达到开仓门槛，建议只补仓不轮动卖出。")
        executable_buy_candidates = buy_selection.buy_candidates
    elif top_buy is not None:
        notes.append("买入候选已生成，但分数未达到开仓门槛，建议暂不新增仓位。")
    if len(portfolio.positions) < config.max_positions:
        notes.append("当前组合未满目标持仓数，卖出候选仅作观察，不生成卖出执行信号。")

    strategy = SignalStrategy(config=config)
    board = strategy.build_board(
        signal_date=as_of,
        trade_date=parse_compact_date(actual_trade_date),
        effective_date=parse_compact_date(next_trade_date),
        holdings_count=len(portfolio.positions),
        buy_candidates=executable_buy_candidates,
        sell_candidates=executable_sell_candidates if sellable_positions else [],
        notes=notes,
    )
    markdown = render_markdown(board)
    output_path = base_dir / config.paths.reports_dir / f"signal-board-{actual_trade_date}.md"
    write_markdown(markdown, output_path)
    return output_path
