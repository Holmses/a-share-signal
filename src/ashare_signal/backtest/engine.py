from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.domain.models import Position
from ashare_signal.features.pipeline import build_universe_snapshot
from ashare_signal.strategy.selector import UniverseSignalSelector
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


@dataclass(slots=True)
class PendingSignal:
    signal_trade_date: str
    buy_candidates: list[object] | None = None
    sell_candidates: list[object] | None = None


@dataclass(slots=True)
class BacktestPosition:
    symbol: str
    name: str
    shares: int
    entry_trade_index: int
    entry_trade_date: str
    entry_price: float
    highest_close: float
    signal_type: str = "trend_pullback"
    signal_low: float | None = None


@dataclass(slots=True)
class BacktestTrade:
    trade_date: str
    action: str
    symbol: str
    name: str
    shares: int
    price: float
    gross_amount: float
    fees: float
    net_amount: float
    signal_trade_date: str
    signal_type: str = ""
    exit_reason: str = ""
    pnl: float | None = None


@dataclass(slots=True)
class BacktestResult:
    start_trade_date: str
    end_trade_date: str
    initial_cash: float
    ending_equity: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    turnover: float
    trade_count: int
    sell_trade_count: int
    win_rate: float
    equity_curve_path: Path
    summary_path: Path
    trade_log_path: Path


class BacktestEngine:
    """A-share daily T+1 backtester driven by cached Tushare data."""

    def __init__(self, config: AppConfig, repository: DataRepository, base_dir: Path) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.selector = UniverseSignalSelector(
            selection_config=config.selection,
            top_buy_n=1000,
            top_sell_n=1000,
        )

    def run(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult:
        all_trade_dates = self.repository.open_trade_dates_between("19000101", "29991231")
        if not all_trade_dates:
            raise ValueError("Trade calendar cache is empty. Run `ashare-signal sync-tushare` first.")

        lookback = max(
            self.config.strategy.lookback_momentum_days,
            self.config.strategy.lookback_short_days,
            self.config.strategy.lookback_vol_days,
            20,
        )
        default_start = parse_compact_date(all_trade_dates[min(lookback, len(all_trade_dates) - 1)])
        default_end = parse_compact_date(all_trade_dates[-1])

        resolved_start = self.repository.resolve_trade_date(to_compact_date(start_date or default_start))
        resolved_end = self.repository.resolve_trade_date(to_compact_date(end_date or default_end))
        trade_dates = self.repository.open_trade_dates_between(resolved_start, resolved_end)
        if len(trade_dates) < 2:
            raise ValueError("Backtest requires at least two trade dates in range.")

        initial_cash = float(self.config.backtest.initial_cash)
        cash = initial_cash
        positions: dict[str, BacktestPosition] = {}
        pending_signal: PendingSignal | None = None
        trades: list[BacktestTrade] = []
        equity_rows: list[dict] = []
        total_traded_value = 0.0

        for trade_index, trade_date in enumerate(trade_dates):
            daily_frame = self.repository.load_daily(trade_date).copy()
            numeric_columns = ["open", "high", "low", "close"]
            for column in numeric_columns:
                daily_frame[column] = pd.to_numeric(daily_frame[column], errors="coerce")
            prices = daily_frame.set_index("ts_code")

            traded_today = 0.0
            if pending_signal is not None:
                cash_box = {"cash": cash}
                traded_today += self._execute_pending_sells(
                    trade_date=trade_date,
                    trade_index=trade_index,
                    prices=prices,
                    pending_signal=pending_signal,
                    positions=positions,
                    trades=trades,
                    cash_ref=cash_box,
                )
                traded_today += self._execute_pending_buys(
                    trade_date=trade_date,
                    trade_index=trade_index,
                    prices=prices,
                    pending_signal=pending_signal,
                    positions=positions,
                    trades=trades,
                    cash_ref=cash_box,
                )
                cash = cash_box["cash"]
            total_traded_value += traded_today

            self._update_position_highs(positions=positions, prices=prices)
            close_equity = self._mark_to_market_equity(cash, positions, prices, price_field="close")
            equity_rows.append(
                {
                    "trade_date": trade_date,
                    "equity": close_equity,
                    "cash": cash,
                    "position_count": len(positions),
                    "pending_buy": ",".join(
                        candidate.symbol
                        for candidate in (pending_signal.buy_candidates or [])
                    )
                    if pending_signal
                    else "",
                    "pending_sell": ",".join(
                        candidate.symbol
                        for candidate in (pending_signal.sell_candidates or [])
                    )
                    if pending_signal
                    else "",
                }
            )

            if trade_index == len(trade_dates) - 1:
                pending_signal = None
                continue

            universe = self._load_or_build_universe(trade_date)
            all_position_models = [
                Position(
                    symbol=position.symbol,
                    name=position.name,
                    entry_date=parse_compact_date(position.entry_trade_date),
                    entry_price=position.entry_price,
                    quantity=position.shares,
                )
                for position in positions.values()
            ]
            sellable_positions = [
                Position(
                    symbol=position.symbol,
                    name=position.name,
                    entry_date=parse_compact_date(position.entry_trade_date),
                    entry_price=position.entry_price,
                    quantity=position.shares,
                )
                for position in positions.values()
                if trade_index - position.entry_trade_index >= self.config.market.min_position_holding_days
            ]
            rotation_positions = [
                Position(
                    symbol=position.symbol,
                    name=position.name,
                    entry_date=parse_compact_date(position.entry_trade_date),
                    entry_price=position.entry_price,
                    quantity=position.shares,
                )
                for position in positions.values()
                if trade_index - position.entry_trade_index >= self.config.selection.rotation_min_holding_days
            ]
            buy_selection = self.selector.select(universe=universe, positions=all_position_models)
            sell_selection = self.selector.select(universe=universe, positions=rotation_positions)
            forced_sells = self._select_forced_sells(
                trade_index=trade_index,
                universe=universe,
                prices=prices,
                positions=positions,
            )

            buy_candidates = []
            sell_candidates = []
            sell_candidates.extend(forced_sells)
            sell_symbols = {candidate.symbol for candidate in sell_candidates}
            if rotation_positions:
                for candidate in sell_selection.sell_candidates:
                    if candidate.symbol in sell_symbols:
                        continue
                    if candidate.score <= self.config.selection.sell_health_exit_threshold:
                        sell_candidates.append(candidate)
                        sell_symbols.add(candidate.symbol)

            expected_rebound_positions = sum(
                1 for position in positions.values() if position.signal_type == "rebound_bottoming"
            )
            for candidate in buy_selection.buy_candidates:
                if candidate.symbol in positions or candidate.symbol in sell_symbols:
                    continue
                if candidate.signal_type == "rebound_bottoming":
                    if expected_rebound_positions >= self.config.selection.rebound_max_positions:
                        continue
                    expected_rebound_positions += 1
                if not self.selector.should_open_new_position(candidate):
                    continue
                if not self.selector.market_allows_buy(universe, signal_type=candidate.signal_type):
                    continue
                buy_candidates.append(candidate)

            pending_signal = PendingSignal(
                signal_trade_date=trade_date,
                buy_candidates=buy_candidates,
                sell_candidates=sell_candidates,
            )

        equity_frame = pd.DataFrame(equity_rows)
        returns = equity_frame["equity"].pct_change().fillna(0.0)
        total_return = equity_frame["equity"].iloc[-1] / initial_cash - 1.0
        annual_return = (
            (equity_frame["equity"].iloc[-1] / initial_cash) ** (252 / max(len(equity_frame), 1)) - 1.0
        )
        cumulative_max = equity_frame["equity"].cummax()
        drawdowns = equity_frame["equity"] / cumulative_max - 1.0
        max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0
        sharpe = 0.0
        if returns.std(ddof=0) > 0:
            sharpe = float((returns.mean() / returns.std(ddof=0)) * math.sqrt(252))
        average_equity = float(equity_frame["equity"].mean()) if not equity_frame.empty else initial_cash
        turnover = float(total_traded_value / average_equity) if average_equity > 0 else 0.0

        sell_trades = [trade for trade in trades if trade.action == "SELL"]
        winning_trades = [trade for trade in sell_trades if trade.pnl is not None and trade.pnl > 0]
        win_rate = float(len(winning_trades) / len(sell_trades)) if sell_trades else 0.0

        reports_dir = self.base_dir / self.config.paths.reports_dir / "backtests"
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_path = reports_dir / f"backtest-summary-{resolved_start}-{resolved_end}.json"
        equity_curve_path = reports_dir / f"backtest-equity-{resolved_start}-{resolved_end}.csv"
        trade_log_path = reports_dir / f"backtest-trades-{resolved_start}-{resolved_end}.csv"

        equity_frame.to_csv(equity_curve_path, index=False)
        trade_columns = list(BacktestTrade.__dataclass_fields__.keys())
        pd.DataFrame([asdict(trade) for trade in trades], columns=trade_columns).to_csv(
            trade_log_path,
            index=False,
        )

        summary_payload = {
            "start_trade_date": resolved_start,
            "end_trade_date": resolved_end,
            "initial_cash": initial_cash,
            "ending_equity": float(equity_frame["equity"].iloc[-1]),
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "turnover": turnover,
            "trade_count": len(trades),
            "sell_trade_count": len(sell_trades),
            "win_rate": win_rate,
            "equity_curve_path": str(equity_curve_path),
            "trade_log_path": str(trade_log_path),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return BacktestResult(
            start_trade_date=resolved_start,
            end_trade_date=resolved_end,
            initial_cash=initial_cash,
            ending_equity=float(equity_frame["equity"].iloc[-1]),
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            turnover=turnover,
            trade_count=len(trades),
            sell_trade_count=len(sell_trades),
            win_rate=win_rate,
            equity_curve_path=equity_curve_path,
            summary_path=summary_path,
            trade_log_path=trade_log_path,
        )

    def _load_or_build_universe(self, trade_date: str) -> pd.DataFrame:
        required_columns = {
            "ma_5",
            "ma_60",
            "close_to_ma_5",
            "close_to_ma_60",
            "ma_20_to_ma_60",
            "ma_60_slope_20d",
            "pullback_from_20d_high",
            "low_to_prev_low",
            "amount_ratio_5d",
            "return_3d",
            "return_10d",
            "drawdown_20d",
            "drawdown_60d",
            "down_days_10d",
            "consecutive_down_days",
            "volume_capitulation_score",
            "industry_member_count",
            "industry_return_3d_median",
            "industry_momentum_20d_median",
            "industry_breadth_20d",
            "industry_rebound_breadth",
        }
        try:
            universe = self.repository.load_universe_snapshot(trade_date)
            if required_columns.issubset(universe.columns):
                return universe
        except FileNotFoundError:
            pass

        universe = build_universe_snapshot(
            config=self.config,
            repository=self.repository,
            trade_date=trade_date,
        )
        self.repository.save_universe_snapshot(trade_date, universe)
        return universe

    def _mark_to_market_equity(
        self,
        cash: float,
        positions: dict[str, BacktestPosition],
        prices: pd.DataFrame,
        price_field: str,
    ) -> float:
        equity = cash
        for position in positions.values():
            if position.symbol not in prices.index:
                continue
            price = float(prices.loc[position.symbol, price_field])
            if math.isnan(price):
                continue
            equity += position.shares * price
        return equity

    def _update_position_highs(
        self,
        positions: dict[str, BacktestPosition],
        prices: pd.DataFrame,
    ) -> None:
        for position in positions.values():
            if position.symbol not in prices.index:
                continue
            close_price = float(prices.loc[position.symbol, "close"])
            if math.isnan(close_price):
                continue
            position.highest_close = max(position.highest_close, close_price)

    def _can_add_buy_candidate(self, candidate, positions: dict[str, BacktestPosition]) -> bool:
        if candidate is None:
            return False
        if candidate.signal_type != "rebound_bottoming":
            return True
        rebound_positions = sum(
            1 for position in positions.values() if position.signal_type == "rebound_bottoming"
        )
        return rebound_positions < self.config.selection.rebound_max_positions

    def _select_forced_sells(
        self,
        trade_index: int,
        universe: pd.DataFrame,
        prices: pd.DataFrame,
        positions: dict[str, BacktestPosition],
    ) -> list[object]:
        if not positions:
            return []

        frame = universe.set_index("ts_code")
        candidates = []
        for position in positions.values():
            if trade_index - position.entry_trade_index < self.config.market.min_position_holding_days:
                continue
            if position.symbol not in prices.index:
                continue

            close_price = float(prices.loc[position.symbol, "close"])
            if math.isnan(close_price):
                continue
            pnl_pct = close_price / position.entry_price - 1.0
            drawdown_from_high = close_price / position.highest_close - 1.0 if position.highest_close else 0.0
            max_profit_pct = position.highest_close / position.entry_price - 1.0 if position.entry_price else 0.0

            reason = None
            exit_reason = None
            priority = 99
            holding_trade_days = trade_index - position.entry_trade_index
            if position.signal_type == "rebound_bottoming":
                if pnl_pct <= -self.config.selection.rebound_stop_loss_pct:
                    priority = 0
                    reason = f"筑底反转快速止损：较入场价 {pnl_pct * 100:.1f}%"
                    exit_reason = "rebound_stop_loss"
                elif (
                    position.signal_low is not None
                    and close_price < position.signal_low
                ):
                    priority = 0
                    reason = f"筑底反转破信号低点：收盘价低于 {position.signal_low:.2f}"
                    exit_reason = "rebound_signal_low_break"
                elif (
                    holding_trade_days >= self.config.selection.rebound_fast_exit_days
                    and close_price < position.entry_price
                ):
                    priority = 1
                    reason = (
                        f"筑底反转未按期转强：持有 {holding_trade_days} 个交易日后仍低于成本"
                    )
                    exit_reason = "rebound_fast_exit"
                elif (
                    max_profit_pct >= self.config.selection.rebound_big_profit_trigger_pct
                    and drawdown_from_high <= -self.config.selection.rebound_big_profit_drawdown_pct
                ):
                    priority = 1
                    reason = f"筑底反转大幅盈利移动止盈：最高收盘后回撤 {drawdown_from_high * 100:.1f}%"
                    exit_reason = "rebound_big_profit_trailing"
                elif (
                    max_profit_pct >= self.config.selection.rebound_profit_lock_trigger_pct
                    and drawdown_from_high <= -self.config.selection.rebound_profit_lock_drawdown_pct
                ):
                    priority = 1
                    reason = f"筑底反转利润锁定：最高收盘后回撤 {drawdown_from_high * 100:.1f}%"
                    exit_reason = "rebound_profit_lock"
                elif (
                    max_profit_pct >= self.config.selection.rebound_breakeven_trigger_pct
                    and pnl_pct <= self.config.selection.rebound_breakeven_floor_pct
                ):
                    priority = 1
                    reason = (
                        f"筑底反转保本止盈：曾盈利 {max_profit_pct * 100:.1f}%，"
                        f"当前回落至 {pnl_pct * 100:.1f}%"
                    )
                    exit_reason = "rebound_breakeven"
            if reason is not None:
                pass
            elif pnl_pct <= -self.config.selection.stop_loss_pct:
                priority = 0
                reason = f"硬止损触发：较入场价 {pnl_pct * 100:.1f}%"
                exit_reason = "stop_loss"
            elif (
                position.highest_close >= position.entry_price * (1 + self.config.selection.take_profit_trigger_pct)
                and drawdown_from_high <= -self.config.selection.trailing_stop_drawdown_pct
            ):
                priority = 1
                reason = f"移动止盈触发：最高收盘后回撤 {drawdown_from_high * 100:.1f}%"
                exit_reason = "trailing_take_profit"
            elif position.symbol in frame.index:
                row = frame.loc[position.symbol]
                close_to_ma20 = pd.to_numeric(row.get("close_to_ma_20"), errors="coerce")
                close_to_ma60 = pd.to_numeric(row.get("close_to_ma_60"), errors="coerce")
                close_to_ma5 = pd.to_numeric(row.get("close_to_ma_5"), errors="coerce")
                momentum_5d = pd.to_numeric(row.get("momentum_5d"), errors="coerce")
                return_1d = pd.to_numeric(row.get("return_1d"), errors="coerce")
                if (
                    position.signal_type == "rebound_bottoming"
                    and pnl_pct >= self.config.selection.rebound_profit_exit_min_pct
                    and close_to_ma5 < 0
                    and return_1d < 0
                ):
                    priority = 1
                    reason = (
                        f"筑底反转盈利后短线转弱：收益 {pnl_pct * 100:.1f}%，"
                        f"较5日均线 {close_to_ma5 * 100:.1f}%"
                    )
                    exit_reason = "rebound_profit_short_weak"
                elif close_to_ma60 < 0:
                    priority = 2
                    reason = f"长期趋势破坏：跌破60日均线 {close_to_ma60 * 100:.1f}%"
                    exit_reason = "ma60_break"
                elif (
                    pnl_pct >= self.config.selection.overheat_min_profit_pct
                    and close_to_ma20 >= self.config.selection.overheat_take_profit_close_to_ma20
                    and (return_1d < 0 or close_to_ma5 < 0)
                ):
                    priority = 2
                    reason = (
                        f"过热止盈：较20日均线 {close_to_ma20 * 100:.1f}%，"
                        f"短线转弱 {return_1d * 100:.1f}%"
                    )
                    exit_reason = "overheat_take_profit"
                elif (
                    trade_index - position.entry_trade_index >= self.config.selection.trend_exit_min_holding_days
                    and close_to_ma20 < 0
                    and momentum_5d < 0
                ):
                    priority = 3
                    reason = (
                        f"趋势破坏：跌破20日均线 {close_to_ma20 * 100:.1f}%，"
                        f"5日动量 {momentum_5d * 100:.1f}%"
                    )
                    exit_reason = "trend_break"

            if reason is not None:
                candidates.append(
                    (
                        priority,
                        pnl_pct,
                        position.symbol,
                        position.name,
                        close_price,
                        reason,
                        exit_reason,
                    )
                )

        return [
            self._make_sell_candidate(
                symbol=symbol,
                name=name,
                score=float(priority + pnl_pct),
                reason=reason,
                last_close=close_price,
                exit_reason=exit_reason,
            )
            for priority, pnl_pct, symbol, name, close_price, reason, exit_reason in sorted(candidates)
        ]

    @staticmethod
    def _make_sell_candidate(symbol: str, name: str, score: float, reason: str, last_close: float, exit_reason: str | None = None):
        from ashare_signal.domain.models import Candidate

        return Candidate(
            symbol=symbol,
            name=name,
            score=score,
            reason=reason,
            last_close=last_close,
            signal_type="risk_exit",
            exit_reason=exit_reason,
        )

    def _execute_pending_sells(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        pending_signal: PendingSignal,
        positions: dict[str, BacktestPosition],
        trades: list[BacktestTrade],
        cash_ref: dict[str, float],
    ) -> float:
        traded_value = 0.0
        for candidate in pending_signal.sell_candidates or []:
            traded_value += self._execute_pending_sell(
                trade_date=trade_date,
                trade_index=trade_index,
                prices=prices,
                candidate=candidate,
                signal_trade_date=pending_signal.signal_trade_date,
                positions=positions,
                trades=trades,
                cash_ref=cash_ref,
            )
        return traded_value

    def _execute_pending_sell(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        candidate: object,
        signal_trade_date: str,
        positions: dict[str, BacktestPosition],
        trades: list[BacktestTrade],
        cash_ref: dict[str, float],
    ) -> float:
        if candidate is None or candidate.symbol not in positions:
            return 0.0

        position = positions[candidate.symbol]
        if trade_index - position.entry_trade_index < self.config.market.min_position_holding_days:
            return 0.0
        if candidate.symbol not in prices.index:
            return 0.0

        row = prices.loc[candidate.symbol]
        day_high = float(row["high"])
        day_open = float(row["open"])
        if math.isnan(day_high) or math.isnan(day_open) or day_high < candidate.last_close * (1 - self.config.pricing.sell_markdown):
            return 0.0

        limit_price = round(candidate.last_close * (1 - self.config.pricing.sell_markdown), 2)
        fill_price = day_open if day_open >= limit_price else limit_price
        gross_amount = fill_price * position.shares
        fees = gross_amount * (self.config.backtest.commission_rate + self.config.backtest.stamp_duty_rate)
        net_amount = gross_amount - fees
        cash_ref["cash"] += net_amount
        pnl = net_amount - position.shares * position.entry_price
        trades.append(
            BacktestTrade(
                trade_date=trade_date,
                action="SELL",
                symbol=position.symbol,
                name=position.name,
                shares=position.shares,
                price=fill_price,
                gross_amount=gross_amount,
                fees=fees,
                net_amount=net_amount,
                signal_trade_date=signal_trade_date,
                signal_type=candidate.signal_type,
                exit_reason=candidate.exit_reason or "",
                pnl=pnl,
            )
        )
        del positions[position.symbol]
        return gross_amount

    def _execute_pending_buys(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        pending_signal: PendingSignal,
        positions: dict[str, BacktestPosition],
        trades: list[BacktestTrade],
        cash_ref: dict[str, float],
    ) -> float:
        traded_value = 0.0
        for candidate in pending_signal.buy_candidates or []:
            traded_value += self._execute_pending_buy(
                trade_date=trade_date,
                trade_index=trade_index,
                prices=prices,
                candidate=candidate,
                signal_trade_date=pending_signal.signal_trade_date,
                positions=positions,
                trades=trades,
                cash_ref=cash_ref,
            )
        return traded_value

    def _execute_pending_buy(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        candidate: object,
        signal_trade_date: str,
        positions: dict[str, BacktestPosition],
        trades: list[BacktestTrade],
        cash_ref: dict[str, float],
    ) -> float:
        if candidate is None or candidate.symbol in positions:
            return 0.0
        if candidate.symbol not in prices.index:
            return 0.0

        row = prices.loc[candidate.symbol]
        day_low = float(row["low"])
        day_open = float(row["open"])
        if math.isnan(day_low) or math.isnan(day_open):
            return 0.0

        limit_price = round(candidate.last_close * (1 + self.config.pricing.buy_markup), 2)
        if day_low > limit_price:
            return 0.0

        fill_price = day_open if day_open <= limit_price else limit_price
        portfolio_value_at_open = self._mark_to_market_equity(
            cash=cash_ref["cash"],
            positions=positions,
            prices=prices,
            price_field="open",
        )
        target_value = portfolio_value_at_open / self.config.max_positions
        if candidate.signal_type == "rebound_bottoming":
            target_value *= self.config.selection.rebound_position_size_multiplier
        raw_shares = int(target_value / fill_price)
        shares = (raw_shares // self.config.backtest.lot_size) * self.config.backtest.lot_size
        max_affordable = int(
            cash_ref["cash"] / (fill_price * (1 + self.config.backtest.commission_rate))
        )
        max_affordable = (max_affordable // self.config.backtest.lot_size) * self.config.backtest.lot_size
        shares = min(shares, max_affordable)
        if shares < self.config.backtest.lot_size:
            return 0.0

        gross_amount = fill_price * shares
        fees = gross_amount * self.config.backtest.commission_rate
        net_amount = gross_amount + fees
        cash_ref["cash"] -= net_amount
        positions[candidate.symbol] = BacktestPosition(
            symbol=candidate.symbol,
            name=candidate.name,
            shares=shares,
            entry_trade_index=trade_index,
            entry_trade_date=trade_date,
            entry_price=fill_price,
            highest_close=fill_price,
            signal_type=candidate.signal_type,
            signal_low=candidate.signal_low,
        )
        trades.append(
            BacktestTrade(
                trade_date=trade_date,
                action="BUY",
                symbol=candidate.symbol,
                name=candidate.name,
                shares=shares,
                price=fill_price,
                gross_amount=gross_amount,
                fees=fees,
                net_amount=net_amount,
                signal_trade_date=signal_trade_date,
                signal_type=candidate.signal_type,
                exit_reason="",
                pnl=None,
            )
        )
        return gross_amount
