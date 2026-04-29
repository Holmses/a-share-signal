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
    buy_candidate: object | None = None
    sell_candidate: object | None = None


@dataclass(slots=True)
class BacktestPosition:
    symbol: str
    name: str
    shares: int
    entry_trade_index: int
    entry_trade_date: str
    entry_price: float


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
            top_buy_n=config.strategy.buy_top_n,
            top_sell_n=config.strategy.sell_top_n,
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
                traded_today += self._execute_pending_sell(
                    trade_date=trade_date,
                    trade_index=trade_index,
                    prices=prices,
                    pending_signal=pending_signal,
                    positions=positions,
                    trades=trades,
                    cash_ref=cash_box,
                )
                traded_today += self._execute_pending_buy(
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

            close_equity = self._mark_to_market_equity(cash, positions, prices, price_field="close")
            equity_rows.append(
                {
                    "trade_date": trade_date,
                    "equity": close_equity,
                    "cash": cash,
                    "position_count": len(positions),
                    "pending_buy": pending_signal.buy_candidate.symbol if pending_signal and pending_signal.buy_candidate else "",
                    "pending_sell": pending_signal.sell_candidate.symbol if pending_signal and pending_signal.sell_candidate else "",
                }
            )

            if trade_index == len(trade_dates) - 1:
                pending_signal = None
                continue

            universe = self._load_or_build_universe(trade_date)
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
            selection = self.selector.select(universe=universe, positions=sellable_positions)

            buy_candidate = selection.buy_candidates[0] if selection.buy_candidates else None
            sell_candidate = selection.sell_candidates[0] if sellable_positions and selection.sell_candidates else None

            if len(positions) >= self.config.max_positions:
                if not self.selector.should_rotate(buy_candidate, sell_candidate):
                    buy_candidate = None
                    sell_candidate = None
            elif not self.selector.should_open_new_position(buy_candidate):
                buy_candidate = None
            if buy_candidate is not None and buy_candidate.symbol in positions:
                buy_candidate = None

            pending_signal = PendingSignal(
                signal_trade_date=trade_date,
                buy_candidate=buy_candidate,
                sell_candidate=sell_candidate,
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
        pd.DataFrame([asdict(trade) for trade in trades]).to_csv(trade_log_path, index=False)

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
        try:
            return self.repository.load_universe_snapshot(trade_date)
        except FileNotFoundError:
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

    def _execute_pending_sell(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        pending_signal: PendingSignal,
        positions: dict[str, BacktestPosition],
        trades: list[BacktestTrade],
        cash_ref: dict[str, float],
    ) -> float:
        candidate = pending_signal.sell_candidate
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
                signal_trade_date=pending_signal.signal_trade_date,
                pnl=pnl,
            )
        )
        del positions[position.symbol]
        return gross_amount

    def _execute_pending_buy(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        pending_signal: PendingSignal,
        positions: dict[str, BacktestPosition],
        trades: list[BacktestTrade],
        cash_ref: dict[str, float],
    ) -> float:
        candidate = pending_signal.buy_candidate
        if candidate is None or candidate.symbol in positions:
            return 0.0
        if len(positions) >= self.config.max_positions:
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
                signal_trade_date=pending_signal.signal_trade_date,
                pnl=None,
            )
        )
        return gross_amount
