from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


@dataclass(slots=True)
class Tianzhu9Position:
    symbol: str
    name: str
    shares: int
    entry_trade_date: str
    signal_trade_date: str
    entry_trade_index: int
    entry_price: float
    entry_cost: float
    highest_close: float
    score: float
    rank: int


@dataclass(slots=True)
class Tianzhu9Trade:
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
    rank: int
    score: float
    pnl: float | None = None


@dataclass(slots=True)
class Tianzhu9BacktestResult:
    start_trade_date: str
    end_trade_date: str
    signal_lag_days: int
    top_n: int
    hold_days: int
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
    execution_mode: str
    extend_on_repeat: bool
    max_hold_days: int
    equity_curve_path: Path
    summary_path: Path
    trade_log_path: Path


class Tianzhu9LikeBacktestEngine:
    """Tushare-only approximation of the BigQuant Tianzhu9 daily rank strategy.

    Signals are built from the prior cached trade day and executed on the next
    trade day. This keeps the run free from same-day close data leakage.
    """

    RETURN_LOOKBACK_TRADE_DAYS = 90
    FACTOR_HISTORY_TRADE_DAYS = 100
    SYNC_WARMUP_CALENDAR_DAYS = 180

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        top_n: int = 5,
        hold_days: int = 1,
        max_position_weight: float = 1.0,
        min_avg_amount_yuan: float = 50_000_000.0,
        execution_mode: str = "intraday",
        extend_on_repeat: bool = False,
        max_hold_days: int | None = None,
        max_positions: int | None = None,
        loss_cooldown_days: int = 3,
        max_return_30d: float = 1.20,
        max_return_90d: float = 3.00,
        min_return_5d: float = -0.08,
        max_return_5d: float = 0.12,
        min_close_to_ma5: float = -0.03,
        min_close_to_ma10: float = -0.05,
        max_close_to_ma10: float = 0.18,
        max_close_to_ma20: float = 0.35,
        max_upper_shadow_pct: float = 0.55,
        stop_loss_pct: float = 0.05,
        take_profit_trigger_pct: float = 0.08,
        trailing_stop_drawdown_pct: float = 0.04,
        lot_size: int | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.top_n = max(int(top_n), 1)
        self.hold_days = max(int(hold_days), 1)
        self.max_position_weight = max(min(float(max_position_weight), 1.0), 0.0)
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.execution_mode = execution_mode
        self.extend_on_repeat = bool(extend_on_repeat)
        self.max_hold_days = max(int(max_hold_days or max(self.hold_days, 5)), self.hold_days)
        config_max_positions = getattr(getattr(config, "market", None), "max_positions", self.top_n)
        self.max_positions = max(int(max_positions or config_max_positions), 1)
        self.loss_cooldown_days = max(int(loss_cooldown_days), 0)
        self.max_return_30d = float(max_return_30d)
        self.max_return_90d = float(max_return_90d)
        self.min_return_5d = float(min_return_5d)
        self.max_return_5d = float(max_return_5d)
        self.min_close_to_ma5 = float(min_close_to_ma5)
        self.min_close_to_ma10 = float(min_close_to_ma10)
        self.max_close_to_ma10 = float(max_close_to_ma10)
        self.max_close_to_ma20 = float(max_close_to_ma20)
        self.max_upper_shadow_pct = float(max_upper_shadow_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_trigger_pct = float(take_profit_trigger_pct)
        self.trailing_stop_drawdown_pct = float(trailing_stop_drawdown_pct)
        self.lot_size = int(lot_size or config.backtest.lot_size)
        if self.execution_mode not in {"intraday", "limit-swing"}:
            raise ValueError("execution_mode must be 'intraday' or 'limit-swing'")

    def run(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> Tianzhu9BacktestResult:
        cached_dates = self.repository.complete_daily_cache_dates()
        if not cached_dates:
            raise ValueError("Daily Tushare cache is empty. Run `ashare-signal sync-tushare` first.")

        resolved_end = self._resolve_cached_end(cached_dates, end_date)
        resolved_start = self._resolve_cached_start(cached_dates, start_date, resolved_end)
        start_index = cached_dates.index(resolved_start)
        end_index = cached_dates.index(resolved_end)
        required_history = self.minimum_backtest_history_trade_days()
        if start_index < required_history:
            cached_start = cached_dates[0]
            suggested_sync_start = to_compact_date(
                self.recommended_sync_start_date(
                    repository=self.repository,
                    target_date=resolved_start,
                    prior_trade_days=required_history,
                )
            )
            raise ValueError(
                "Tianzhu9-like backtest needs at least "
                f"{required_history} complete trade days before start date {resolved_start} "
                f"for factor warm-up, but only found {start_index}. "
                f"Current cache starts at {cached_start}. "
                f"Sync from {suggested_sync_start} or earlier and rerun."
            )
        trade_dates = cached_dates[start_index : end_index + 1]
        if len(trade_dates) < 2:
            raise ValueError("Tianzhu9-like backtest requires at least two cached trade dates.")

        feature_dates = cached_dates[max(0, start_index - self.factor_history_trade_days()) : end_index + 1]
        factor_frame = self._build_factor_frame(feature_dates)
        prices = self._load_price_map(feature_dates)

        initial_cash = float(self.config.backtest.initial_cash)
        cash = initial_cash
        positions: dict[str, Tianzhu9Position] = {}
        trades: list[Tianzhu9Trade] = []
        equity_rows: list[dict] = []
        total_traded_value = 0.0
        loss_cooldown_until: dict[str, int] = {}

        for trade_offset, trade_date in enumerate(trade_dates):
            global_trade_index = start_index + trade_offset
            signal_trade_date = cached_dates[global_trade_index - 1]
            day_prices = prices.get(trade_date, pd.DataFrame())
            if day_prices.empty:
                continue

            open_equity = self._mark_to_market_equity(cash, positions, day_prices, "open")
            cooldown_symbols = {
                symbol
                for symbol, cooldown_until in loss_cooldown_until.items()
                if cooldown_until >= global_trade_index
            }
            selected = self._select_candidates(
                factor_frame,
                signal_trade_date,
                excluded_symbols=cooldown_symbols,
            )
            selected_symbols = {row["ts_code"] for row in selected}
            if self.execution_mode == "limit-swing":
                sell_cash_box = {"cash": cash}
                total_traded_value += self._execute_limit_swing_sells(
                    trade_date=trade_date,
                    trade_index=global_trade_index,
                    prices=day_prices,
                    factor_frame=factor_frame,
                    signal_trade_date=signal_trade_date,
                    positions=positions,
                    selected_symbols=selected_symbols,
                    trades=trades,
                    cash_ref=sell_cash_box,
                    loss_cooldown_until=loss_cooldown_until,
                )
                cash = sell_cash_box["cash"]

                buy_cash_box = {"cash": cash}
                buy_candidates = [
                    row
                    for row in selected
                    if loss_cooldown_until.get(str(row["ts_code"]), -1) < global_trade_index
                ]
                total_traded_value += self._execute_limit_buys(
                    trade_date=trade_date,
                    signal_trade_date=signal_trade_date,
                    trade_index=global_trade_index,
                    prices=day_prices,
                    candidates=buy_candidates,
                    open_equity=open_equity,
                    positions=positions,
                    trades=trades,
                    cash_ref=buy_cash_box,
                )
                cash = buy_cash_box["cash"]
            else:
                cash_box = {"cash": cash}
                total_traded_value += self._execute_open_buys(
                    trade_date=trade_date,
                    signal_trade_date=signal_trade_date,
                    trade_index=global_trade_index,
                    prices=day_prices,
                    candidates=selected,
                    open_equity=open_equity,
                    positions=positions,
                    trades=trades,
                    cash_ref=cash_box,
                )
                cash = cash_box["cash"]

                sell_cash_box = {"cash": cash}
                total_traded_value += self._execute_due_close_sells(
                    trade_date=trade_date,
                    trade_index=global_trade_index,
                    prices=day_prices,
                    positions=positions,
                    selected_symbols=selected_symbols,
                    trades=trades,
                    cash_ref=sell_cash_box,
                    loss_cooldown_until=loss_cooldown_until,
                )
                cash = sell_cash_box["cash"]

            self._update_position_highs(positions=positions, prices=day_prices)

            close_equity = self._mark_to_market_equity(cash, positions, day_prices, "close")
            equity_rows.append(
                {
                    "trade_date": trade_date,
                    "equity": close_equity,
                    "cash": cash,
                    "position_count": len(positions),
                    "signal_trade_date": signal_trade_date,
                    "selected": ",".join(row["ts_code"] for row in selected),
                }
            )

        equity_frame = pd.DataFrame(equity_rows)
        if equity_frame.empty:
            raise ValueError("Tianzhu9-like backtest produced no equity rows.")

        returns = equity_frame["equity"].pct_change().fillna(0.0)
        ending_equity = float(equity_frame["equity"].iloc[-1])
        total_return = ending_equity / initial_cash - 1.0
        annual_return = (ending_equity / initial_cash) ** (252 / max(len(equity_frame), 1)) - 1.0
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
        stem = f"tianzhu9-like-top{self.top_n}-h{self.hold_days}"
        if self.execution_mode == "limit-swing":
            stem += "-limit-swing"
        if self.extend_on_repeat:
            stem += f"-extend{self.max_hold_days}"
        stem += f"-{resolved_start}-{resolved_end}"
        summary_path = reports_dir / f"{stem}-summary.json"
        equity_curve_path = reports_dir / f"{stem}-equity.csv"
        trade_log_path = reports_dir / f"{stem}-trades.csv"

        equity_frame.to_csv(equity_curve_path, index=False)
        trade_columns = list(Tianzhu9Trade.__dataclass_fields__.keys())
        pd.DataFrame([asdict(trade) for trade in trades], columns=trade_columns).to_csv(
            trade_log_path,
            index=False,
        )
        summary_payload = {
            "strategy": "tianzhu9_like",
            "start_trade_date": resolved_start,
            "end_trade_date": resolved_end,
            "signal_lag_days": 1,
            "top_n": self.top_n,
            "hold_days": self.hold_days,
            "execution_mode": self.execution_mode,
            "max_position_weight": self.max_position_weight,
            "max_positions": self.max_positions,
            "min_avg_amount_yuan": self.min_avg_amount_yuan,
            "extend_on_repeat": self.extend_on_repeat,
            "max_hold_days": self.max_hold_days,
            "loss_cooldown_days": self.loss_cooldown_days,
            "selection_filters": {
                "max_return_30d": self.max_return_30d,
                "max_return_90d": self.max_return_90d,
                "min_return_5d": self.min_return_5d,
                "max_return_5d": self.max_return_5d,
                "min_close_to_ma5": self.min_close_to_ma5,
                "min_close_to_ma10": self.min_close_to_ma10,
                "max_close_to_ma10": self.max_close_to_ma10,
                "max_close_to_ma20": self.max_close_to_ma20,
                "max_upper_shadow_pct": self.max_upper_shadow_pct,
            },
            "initial_cash": initial_cash,
            "ending_equity": ending_equity,
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

        return Tianzhu9BacktestResult(
            start_trade_date=resolved_start,
            end_trade_date=resolved_end,
            signal_lag_days=1,
            top_n=self.top_n,
            hold_days=self.hold_days,
            initial_cash=initial_cash,
            ending_equity=ending_equity,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            turnover=turnover,
            trade_count=len(trades),
            sell_trade_count=len(sell_trades),
            win_rate=win_rate,
            execution_mode=self.execution_mode,
            extend_on_repeat=self.extend_on_repeat,
            max_hold_days=self.max_hold_days,
            equity_curve_path=equity_curve_path,
            summary_path=summary_path,
            trade_log_path=trade_log_path,
        )

    @classmethod
    def minimum_signal_history_trade_days(cls) -> int:
        return cls.RETURN_LOOKBACK_TRADE_DAYS

    @classmethod
    def minimum_backtest_history_trade_days(cls) -> int:
        return cls.minimum_signal_history_trade_days() + 1

    @classmethod
    def factor_history_trade_days(cls) -> int:
        return cls.FACTOR_HISTORY_TRADE_DAYS

    @classmethod
    def recommended_sync_start_date(
        cls,
        repository: DataRepository,
        target_date: date | str,
        *,
        prior_trade_days: int,
    ) -> date:
        target_trade_date = to_compact_date(target_date)
        try:
            resolved_target = repository.resolve_trade_date(target_trade_date)
            trade_dates = repository.recent_open_trade_dates(
                resolved_target,
                count=prior_trade_days + 1,
            )
            return parse_compact_date(trade_dates[0])
        except Exception:
            return parse_compact_date(target_trade_date) - timedelta(days=cls.SYNC_WARMUP_CALENDAR_DAYS)

    def _resolve_cached_end(self, cached_dates: list[str], end_date: date | None) -> str:
        if end_date is None:
            return cached_dates[-1]
        requested = to_compact_date(end_date)
        eligible = [value for value in cached_dates if value <= requested]
        if not eligible:
            raise ValueError(f"No cached trade date found on or before {requested}")
        return eligible[-1]

    def _resolve_cached_start(self, cached_dates: list[str], start_date: date | None, end_date: str) -> str:
        if start_date is None:
            end_index = cached_dates.index(end_date)
            return cached_dates[max(1, end_index - 126)]
        requested = to_compact_date(start_date)
        eligible = [value for value in cached_dates if value >= requested and value <= end_date]
        if not eligible:
            raise ValueError(f"No cached trade date found on or after {requested}")
        return eligible[0]

    def _build_factor_frame(self, trade_dates: list[str]) -> pd.DataFrame:
        daily = self.repository.load_daily_for_dates(trade_dates)
        daily = daily.copy()
        for column in ("open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"):
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
        daily["amount_yuan"] = daily["amount"] * 1000.0
        daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        grouped = daily.groupby("ts_code", group_keys=False)
        daily["return_30d"] = grouped["close"].pct_change(periods=30)
        daily["return_90d"] = grouped["close"].pct_change(periods=90)
        daily["return_5d"] = grouped["close"].pct_change(periods=5)
        daily["ma_5"] = grouped["close"].transform(lambda series: series.rolling(window=5, min_periods=5).mean())
        daily["ma_10"] = grouped["close"].transform(lambda series: series.rolling(window=10, min_periods=10).mean())
        daily["ma_20"] = grouped["close"].transform(lambda series: series.rolling(window=20, min_periods=20).mean())
        daily["high_20d"] = grouped["high"].transform(lambda series: series.rolling(window=20, min_periods=20).max())
        daily["avg_amount_20d_yuan"] = grouped["amount_yuan"].transform(
            lambda series: series.rolling(window=20, min_periods=20).mean()
        )
        daily["avg_amount_5d_yuan"] = grouped["amount_yuan"].transform(
            lambda series: series.rolling(window=5, min_periods=5).mean()
        )
        daily["amount_ratio_5d"] = daily["amount_yuan"] / daily["avg_amount_5d_yuan"]
        daily["close_to_ma_5"] = daily["close"] / daily["ma_5"] - 1.0
        daily["close_to_ma_10"] = daily["close"] / daily["ma_10"] - 1.0
        daily["close_to_ma_20"] = daily["close"] / daily["ma_20"] - 1.0
        daily["drawdown_from_20d_high"] = daily["close"] / daily["high_20d"] - 1.0
        candle_range = daily["high"] - daily["low"]
        upper_shadow = daily["high"] - daily[["open", "close"]].max(axis=1)
        daily["upper_shadow_pct"] = (upper_shadow / candle_range.where(candle_range > 0)).clip(lower=0.0)

        daily_basic = pd.concat(
            [self.repository.load_daily_basic(trade_date) for trade_date in trade_dates],
            ignore_index=True,
        )
        daily_basic = daily_basic.copy()
        for column in ("turnover_rate", "volume_ratio", "total_mv", "circ_mv"):
            daily_basic[column] = pd.to_numeric(daily_basic[column], errors="coerce")
        daily_basic["total_mv_yuan"] = daily_basic["total_mv"] * 10000.0

        stock_basic = self.repository.load_stock_basic(list_status="L")
        stock_basic = stock_basic[["ts_code", "name", "market", "exchange", "list_date"]].copy()
        stock_basic["list_date"] = pd.to_datetime(stock_basic["list_date"], format="%Y%m%d", errors="coerce")

        frame = daily.merge(
            daily_basic[
                [
                    "ts_code",
                    "trade_date",
                    "turnover_rate",
                    "volume_ratio",
                    "total_mv_yuan",
                ]
            ],
            on=["ts_code", "trade_date"],
            how="left",
        ).merge(stock_basic, on="ts_code", how="left")
        frame["trade_timestamp"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="coerce")
        frame["listed_days"] = (frame["trade_timestamp"] - frame["list_date"]).dt.days
        frame["is_chinext"] = frame["market"].fillna("").str.contains("创业板", regex=False) | frame[
            "ts_code"
        ].str.startswith(("300", "301"))
        frame["is_st"] = frame["name"].fillna("").str.upper().str.contains("ST")

        candidate_mask = (
            frame["is_chinext"]
            & (~frame["is_st"])
            & (frame["listed_days"].fillna(-1) >= self.config.filters.min_list_days)
            & (frame["close"].fillna(0.0) >= self.config.filters.min_price)
            & (frame["avg_amount_20d_yuan"].fillna(0.0) >= self.min_avg_amount_yuan)
            & frame["return_30d"].notna()
            & frame["return_90d"].notna()
            & (frame["return_30d"] <= self.max_return_30d)
            & (frame["return_90d"] <= self.max_return_90d)
            & (frame["return_5d"] >= self.min_return_5d)
            & (frame["return_5d"] <= self.max_return_5d)
            & frame["open"].notna()
            & frame["close"].notna()
            & frame["ma_5"].notna()
            & frame["ma_10"].notna()
            & frame["ma_20"].notna()
            & (frame["close_to_ma_5"] >= self.min_close_to_ma5)
            & (frame["close_to_ma_10"] >= self.min_close_to_ma10)
            & (frame["close_to_ma_10"] <= self.max_close_to_ma10)
            & (frame["close_to_ma_20"] <= self.max_close_to_ma20)
            & (frame["upper_shadow_pct"].fillna(0.0) <= self.max_upper_shadow_pct)
        )
        frame = frame.loc[candidate_mask].copy()
        if frame.empty:
            return frame

        by_date = frame.groupby("trade_date", group_keys=False)
        frame["return_30d_rank"] = by_date["return_30d"].rank(pct=True)
        frame["return_90d_rank"] = by_date["return_90d"].rank(pct=True)
        frame["turnover_rank"] = by_date["turnover_rate"].rank(pct=True)
        frame["amount_rank"] = by_date["avg_amount_20d_yuan"].rank(pct=True)
        frame["volume_ratio_score"] = (
            1.0 - ((frame["volume_ratio"].fillna(1.0) - 1.0).abs() / 3.0)
        ).clip(lower=0.0, upper=1.0)
        frame["stability_score"] = (
            1.0 - ((frame["return_5d"].fillna(0.0).abs() - 0.02) / 0.18)
        ).clip(lower=0.0, upper=1.0)
        frame["trend_quality_score"] = (
            (frame["close"] >= frame["ma_5"]).astype(float) * 0.35
            + (frame["ma_5"] >= frame["ma_10"]).astype(float) * 0.35
            + (frame["ma_10"] >= frame["ma_20"]).astype(float) * 0.30
        )
        frame["overheat_score"] = (
            1.0 - ((frame["close_to_ma_20"].fillna(0.0) - 0.08) / 0.27)
        ).clip(lower=0.0, upper=1.0)
        frame["score"] = (
            frame["return_30d_rank"].fillna(0.0) * 0.28
            + frame["return_90d_rank"].fillna(0.0) * 0.18
            + frame["amount_rank"].fillna(0.0) * 0.15
            + frame["turnover_rank"].fillna(0.0) * 0.10
            + frame["volume_ratio_score"].fillna(0.0) * 0.05
            + frame["stability_score"].fillna(0.0) * 0.14
            + frame["trend_quality_score"].fillna(0.0) * 0.05
            + frame["overheat_score"].fillna(0.0) * 0.05
        )
        return frame.sort_values(["trade_date", "score", "avg_amount_20d_yuan"], ascending=[True, False, False])

    def _load_price_map(self, trade_dates: list[str]) -> dict[str, pd.DataFrame]:
        price_map = {}
        for trade_date in trade_dates:
            frame = self.repository.load_daily(trade_date).copy()
            for column in ("open", "high", "low", "close"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            price_map[trade_date] = frame.set_index("ts_code")
        return price_map

    def _select_candidates(
        self,
        factor_frame: pd.DataFrame,
        signal_trade_date: str,
        excluded_symbols: set[str] | None = None,
    ) -> list[dict]:
        if factor_frame.empty:
            return []
        daily = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_trade_date].copy()
        if daily.empty:
            return []
        if excluded_symbols:
            daily = daily.loc[~daily["ts_code"].isin(excluded_symbols)]
            if daily.empty:
                return []
        daily = daily.sort_values(["score", "avg_amount_20d_yuan"], ascending=[False, False]).head(self.top_n)
        rows = []
        for rank, row in enumerate(daily.to_dict(orient="records"), start=1):
            row["rank"] = rank
            rows.append(row)
        return rows

    def _mark_to_market_equity(
        self,
        cash: float,
        positions: dict[str, Tianzhu9Position],
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

    def _execute_open_buys(
        self,
        trade_date: str,
        signal_trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        candidates: list[dict],
        open_equity: float,
        positions: dict[str, Tianzhu9Position],
        trades: list[Tianzhu9Trade],
        cash_ref: dict[str, float],
    ) -> float:
        if not candidates or open_equity <= 0:
            return 0.0
        if len(positions) >= self.max_positions:
            return 0.0

        traded_value = 0.0
        target_slot_value = open_equity / self.max_positions
        for candidate in candidates:
            if len(positions) >= self.max_positions:
                break
            symbol = candidate["ts_code"]
            if symbol in positions or symbol not in prices.index:
                continue
            row = prices.loc[symbol]
            open_price = float(row["open"])
            if math.isnan(open_price) or open_price <= 0:
                continue
            target_value = min(
                target_slot_value,
                open_equity * self.max_position_weight,
                cash_ref["cash"],
            )
            raw_shares = int(target_value / open_price)
            shares = (raw_shares // self.lot_size) * self.lot_size
            if shares < self.lot_size:
                continue
            gross_amount = open_price * shares
            fees = max(gross_amount * self.config.backtest.commission_rate, 5.0)
            if gross_amount + fees > cash_ref["cash"]:
                affordable = int(cash_ref["cash"] / (open_price * (1 + self.config.backtest.commission_rate)))
                shares = (affordable // self.lot_size) * self.lot_size
                if shares < self.lot_size:
                    continue
                gross_amount = open_price * shares
                fees = max(gross_amount * self.config.backtest.commission_rate, 5.0)
            net_amount = gross_amount + fees
            cash_ref["cash"] -= net_amount
            positions[symbol] = Tianzhu9Position(
                symbol=symbol,
                name=str(candidate.get("name") or symbol),
                shares=shares,
                entry_trade_date=trade_date,
                signal_trade_date=signal_trade_date,
                entry_trade_index=trade_index,
                entry_price=open_price,
                entry_cost=net_amount,
                highest_close=open_price,
                score=float(candidate["score"]),
                rank=int(candidate["rank"]),
            )
            trades.append(
                Tianzhu9Trade(
                    trade_date=trade_date,
                    action="BUY",
                    symbol=symbol,
                    name=str(candidate.get("name") or symbol),
                    shares=shares,
                    price=open_price,
                    gross_amount=gross_amount,
                    fees=fees,
                    net_amount=net_amount,
                    signal_trade_date=signal_trade_date,
                    rank=int(candidate["rank"]),
                    score=float(candidate["score"]),
                    pnl=None,
                )
            )
            traded_value += gross_amount
        return traded_value

    def _execute_limit_buys(
        self,
        trade_date: str,
        signal_trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        candidates: list[dict],
        open_equity: float,
        positions: dict[str, Tianzhu9Position],
        trades: list[Tianzhu9Trade],
        cash_ref: dict[str, float],
    ) -> float:
        if not candidates or open_equity <= 0:
            return 0.0
        if len(positions) >= self.max_positions:
            return 0.0

        traded_value = 0.0
        target_slot_value = open_equity / self.max_positions
        for candidate in candidates:
            if len(positions) >= self.max_positions:
                break
            symbol = candidate["ts_code"]
            if symbol in positions or symbol not in prices.index:
                continue
            row = prices.loc[symbol]
            day_open = float(row["open"])
            day_low = float(row["low"])
            prev_close = float(candidate["close"])
            if any(math.isnan(value) or value <= 0 for value in (day_open, day_low, prev_close)):
                continue
            limit_price = round(prev_close * (1 + self.config.pricing.buy_markup), 2)
            if day_low > limit_price:
                continue
            fill_price = day_open if day_open <= limit_price else limit_price
            target_value = min(
                target_slot_value,
                open_equity * self.max_position_weight,
                cash_ref["cash"],
            )
            raw_shares = int(target_value / fill_price)
            shares = (raw_shares // self.lot_size) * self.lot_size
            if shares < self.lot_size:
                continue
            gross_amount = fill_price * shares
            fees = max(gross_amount * self.config.backtest.commission_rate, 5.0)
            if gross_amount + fees > cash_ref["cash"]:
                affordable = int(cash_ref["cash"] / (fill_price * (1 + self.config.backtest.commission_rate)))
                shares = (affordable // self.lot_size) * self.lot_size
                if shares < self.lot_size:
                    continue
                gross_amount = fill_price * shares
                fees = max(gross_amount * self.config.backtest.commission_rate, 5.0)
            net_amount = gross_amount + fees
            cash_ref["cash"] -= net_amount
            positions[symbol] = Tianzhu9Position(
                symbol=symbol,
                name=str(candidate.get("name") or symbol),
                shares=shares,
                entry_trade_date=trade_date,
                signal_trade_date=signal_trade_date,
                entry_trade_index=trade_index,
                entry_price=fill_price,
                entry_cost=net_amount,
                highest_close=fill_price,
                score=float(candidate["score"]),
                rank=int(candidate["rank"]),
            )
            trades.append(
                Tianzhu9Trade(
                    trade_date=trade_date,
                    action="BUY",
                    symbol=symbol,
                    name=str(candidate.get("name") or symbol),
                    shares=shares,
                    price=fill_price,
                    gross_amount=gross_amount,
                    fees=fees,
                    net_amount=net_amount,
                    signal_trade_date=signal_trade_date,
                    rank=int(candidate["rank"]),
                    score=float(candidate["score"]),
                    pnl=None,
                )
            )
            traded_value += gross_amount
        return traded_value

    def _execute_limit_swing_sells(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        factor_frame: pd.DataFrame,
        signal_trade_date: str,
        positions: dict[str, Tianzhu9Position],
        selected_symbols: set[str],
        trades: list[Tianzhu9Trade],
        cash_ref: dict[str, float],
        loss_cooldown_until: dict[str, int] | None = None,
    ) -> float:
        traded_value = 0.0
        for symbol in list(positions):
            position = positions[symbol]
            if symbol not in prices.index:
                continue
            holding_days = trade_index - position.entry_trade_index + 1
            feature = self._feature_row(factor_frame, signal_trade_date, symbol)
            if feature is None:
                continue

            prev_close = float(feature["close"])
            ma_5 = float(feature["ma_5"])
            ma_10 = float(feature["ma_10"])
            pnl_pct = prev_close / position.entry_price - 1.0
            high_profit_pct = position.highest_close / position.entry_price - 1.0
            drawdown_from_high = prev_close / position.highest_close - 1.0 if position.highest_close else 0.0
            exit_signal = False
            if pnl_pct <= -self.stop_loss_pct:
                exit_signal = True
            elif (
                high_profit_pct >= self.take_profit_trigger_pct
                and drawdown_from_high <= -self.trailing_stop_drawdown_pct
            ):
                exit_signal = True
            elif holding_days >= self.max_hold_days:
                exit_signal = True
            elif holding_days >= self.hold_days and symbol not in selected_symbols and prev_close < ma_5:
                exit_signal = True
            elif holding_days >= self.hold_days and prev_close < ma_10:
                exit_signal = True

            if not exit_signal:
                continue

            row = prices.loc[symbol]
            day_open = float(row["open"])
            day_high = float(row["high"])
            if any(math.isnan(value) or value <= 0 for value in (day_open, day_high, prev_close)):
                continue
            limit_price = round(prev_close * (1 - self.config.pricing.sell_markdown), 2)
            if day_high < limit_price:
                continue
            fill_price = day_open if day_open >= limit_price else limit_price
            gross_amount = fill_price * position.shares
            fees = max(
                gross_amount * (self.config.backtest.commission_rate + self.config.backtest.stamp_duty_rate),
                5.0,
            )
            net_amount = gross_amount - fees
            cash_ref["cash"] += net_amount
            pnl = net_amount - position.entry_cost
            if loss_cooldown_until is not None and pnl <= 0 and self.loss_cooldown_days > 0:
                loss_cooldown_until[symbol] = trade_index + self.loss_cooldown_days
            trades.append(
                Tianzhu9Trade(
                    trade_date=trade_date,
                    action="SELL",
                    symbol=symbol,
                    name=position.name,
                    shares=position.shares,
                    price=fill_price,
                    gross_amount=gross_amount,
                    fees=fees,
                    net_amount=net_amount,
                    signal_trade_date=position.signal_trade_date,
                    rank=position.rank,
                    score=position.score,
                    pnl=pnl,
                )
            )
            traded_value += gross_amount
            del positions[symbol]
        return traded_value

    def _update_position_highs(
        self,
        positions: dict[str, Tianzhu9Position],
        prices: pd.DataFrame,
    ) -> None:
        for position in positions.values():
            if position.symbol not in prices.index:
                continue
            close_price = float(prices.loc[position.symbol, "close"])
            if math.isnan(close_price):
                continue
            position.highest_close = max(position.highest_close, close_price)

    @staticmethod
    def _feature_row(factor_frame: pd.DataFrame, signal_trade_date: str, symbol: str) -> pd.Series | None:
        if factor_frame.empty:
            return None
        rows = factor_frame.loc[
            (factor_frame["trade_date"].astype(str) == signal_trade_date)
            & (factor_frame["ts_code"] == symbol)
        ]
        if rows.empty:
            return None
        return rows.iloc[0]

    def _execute_due_close_sells(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        positions: dict[str, Tianzhu9Position],
        selected_symbols: set[str],
        trades: list[Tianzhu9Trade],
        cash_ref: dict[str, float],
        loss_cooldown_until: dict[str, int] | None = None,
    ) -> float:
        traded_value = 0.0
        for symbol in list(positions):
            position = positions[symbol]
            holding_days = trade_index - position.entry_trade_index + 1
            if holding_days < self.hold_days or symbol not in prices.index:
                continue
            if (
                self.extend_on_repeat
                and symbol in selected_symbols
                and holding_days < self.max_hold_days
            ):
                continue
            close_price = float(prices.loc[symbol, "close"])
            if math.isnan(close_price) or close_price <= 0:
                continue
            gross_amount = close_price * position.shares
            fees = max(
                gross_amount * (self.config.backtest.commission_rate + self.config.backtest.stamp_duty_rate),
                5.0,
            )
            net_amount = gross_amount - fees
            cash_ref["cash"] += net_amount
            pnl = net_amount - position.entry_cost
            if loss_cooldown_until is not None and pnl <= 0 and self.loss_cooldown_days > 0:
                loss_cooldown_until[symbol] = trade_index + self.loss_cooldown_days
            trades.append(
                Tianzhu9Trade(
                    trade_date=trade_date,
                    action="SELL",
                    symbol=symbol,
                    name=position.name,
                    shares=position.shares,
                    price=close_price,
                    gross_amount=gross_amount,
                    fees=fees,
                    net_amount=net_amount,
                    signal_trade_date=position.signal_trade_date,
                    rank=position.rank,
                    score=position.score,
                    pnl=pnl,
                )
            )
            traded_value += gross_amount
            del positions[symbol]
        return traded_value
