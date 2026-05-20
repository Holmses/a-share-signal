from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.backtest.tianzhu9_like import Tianzhu9BacktestResult, Tianzhu9Position, Tianzhu9Trade
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.exit_rules import TIERED_TRAILING_TAKE_PROFIT_LEVELS
from ashare_signal.strategy.exit_rules import tiered_trailing_take_profit
from ashare_signal.utils.dates import to_compact_date


class FullAMomentumBacktestEngine:
    """Full A-share momentum backtest with market and board-style filters."""

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        top_n: int = 5,
        hold_days: int = 5,
        max_hold_days: int = 10,
        max_positions: int | None = None,
        groups: list[str] | None = None,
        selection_variant: str = "quality_momentum",
        min_avg_amount_yuan: float = 50_000_000.0,
        market_min_breadth: float = 0.50,
        market_min_return_20d: float = 0.0,
        style_min_breadth: float = 0.48,
        style_min_return_20d: float = -0.01,
        style_score_weight: float = 0.06,
        loss_cooldown_days: int = 3,
        stop_loss_pct: float = 0.05,
        take_profit_trigger_pct: float = 0.08,
        trailing_stop_drawdown_pct: float = 0.04,
        hard_exit_days: int | None = 23,
        exit_ma20_break: bool = False,
        exit_failure_days: int | None = 8,
        exit_failure_min_peak_profit_pct: float = 0.03,
        exit_adaptive_trailing: bool = False,
        exit_atr_multiplier: float = 1.5,
        exit_market_risk: bool = False,
        exit_industry_weak: bool = False,
        exit_relative_weak: bool = False,
        exit_relative_weak_5d_pct: float = 0.04,
        exit_relative_weak_20d_pct: float = 0.08,
        exit_volume_stall: bool = True,
        exit_volume_stall_ratio: float = 1.4,
        exit_upper_shadow: bool = False,
        exit_upper_shadow_pct: float = 0.45,
        lot_size: int | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.top_n = max(int(top_n), 1)
        self.hold_days = max(int(hold_days), 1)
        self.max_hold_days = max(int(max_hold_days), self.hold_days)
        self.max_positions = max(int(max_positions or config.market.max_positions), 1)
        self.groups = groups or ["main", "chinext", "star"]
        self.selection_variant = selection_variant
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.market_min_breadth = float(market_min_breadth)
        self.market_min_return_20d = float(market_min_return_20d)
        self.style_min_breadth = float(style_min_breadth)
        self.style_min_return_20d = float(style_min_return_20d)
        self.style_score_weight = float(style_score_weight)
        self.loss_cooldown_days = max(int(loss_cooldown_days), 0)
        self.hard_exit_days = max(int(hard_exit_days), 1) if hard_exit_days is not None else None
        self.exit_ma20_break = bool(exit_ma20_break)
        self.exit_failure_days = max(int(exit_failure_days), 1) if exit_failure_days else None
        self.exit_failure_min_peak_profit_pct = float(exit_failure_min_peak_profit_pct)
        self.exit_adaptive_trailing = bool(exit_adaptive_trailing)
        self.exit_atr_multiplier = float(exit_atr_multiplier)
        self.exit_market_risk = bool(exit_market_risk)
        self.exit_industry_weak = bool(exit_industry_weak)
        self.exit_relative_weak = bool(exit_relative_weak)
        self.exit_relative_weak_5d_pct = float(exit_relative_weak_5d_pct)
        self.exit_relative_weak_20d_pct = float(exit_relative_weak_20d_pct)
        self.exit_volume_stall = bool(exit_volume_stall)
        self.exit_volume_stall_ratio = float(exit_volume_stall_ratio)
        self.exit_upper_shadow = bool(exit_upper_shadow)
        self.exit_upper_shadow_pct = float(exit_upper_shadow_pct)
        self.exit_strategy = self._exit_strategy_name()
        self.lot_size = int(lot_size or config.backtest.lot_size)

    def _exit_strategy_name(self) -> str:
        parts = ["tiered_trailing_take_profit"]
        if self.hard_exit_days is None:
            parts.append("no_time_exit")
        else:
            parts.append(f"hard_exit_{self.hard_exit_days}d")
        if self.exit_ma20_break:
            parts.append("ma20_break")
        if self.exit_failure_days is not None:
            parts.append(f"failure_{self.exit_failure_days}d")
        if self.exit_adaptive_trailing:
            parts.append(f"adaptive_atr_{self.exit_atr_multiplier:g}x")
        if self.exit_market_risk:
            parts.append("market_risk_exit")
        if self.exit_industry_weak:
            parts.append("industry_weak_exit")
        if self.exit_relative_weak:
            parts.append("relative_weak_exit")
        if self.exit_volume_stall:
            parts.append("volume_stall_exit")
        if self.exit_upper_shadow:
            parts.append("upper_shadow_exit")
        return "_".join(parts)

    def _exit_slug(self) -> str:
        slug = f"tiered-trailing-hard{self.hard_exit_days or 'none'}"
        if self.exit_ma20_break:
            slug += "-ma20"
        if self.exit_failure_days is not None:
            failure_pct = _slug_float(self.exit_failure_min_peak_profit_pct)
            slug += f"-fail{self.exit_failure_days}d{failure_pct}"
        if self.exit_adaptive_trailing:
            slug += f"-atr{_slug_float(self.exit_atr_multiplier)}"
        if self.exit_market_risk:
            slug += "-mktrisk"
        if self.exit_industry_weak:
            slug += "-indweak"
        if self.exit_relative_weak:
            slug += (
                f"-relweak{_slug_float(self.exit_relative_weak_5d_pct)}"
                f"x{_slug_float(self.exit_relative_weak_20d_pct)}"
            )
        if self.exit_volume_stall:
            slug += f"-volstall{_slug_float(self.exit_volume_stall_ratio)}"
        if self.exit_upper_shadow:
            slug += f"-shadow{_slug_float(self.exit_upper_shadow_pct)}"
        return slug

    def run(self, start_date: date | None = None, end_date: date | None = None) -> Tianzhu9BacktestResult:
        cached_dates = self.repository.complete_daily_cache_dates()
        if not cached_dates:
            raise ValueError("Daily Tushare cache is empty. Run `ashare-signal sync-tushare` first.")

        resolved_end = self._resolve_cached_end(cached_dates, end_date)
        resolved_start = self._resolve_cached_start(cached_dates, start_date, resolved_end)
        start_index = cached_dates.index(resolved_start)
        end_index = cached_dates.index(resolved_end)
        required_history = SelectionEventStudyEngine.minimum_backtest_history_trade_days()
        if start_index < required_history:
            suggested_sync_start = to_compact_date(
                SelectionEventStudyEngine.recommended_sync_start_date(
                    repository=self.repository,
                    target_date=resolved_start,
                    prior_trade_days=required_history,
                )
            )
            raise ValueError(
                "Full A momentum backtest needs at least "
                f"{required_history} complete trade days before start date {resolved_start}. "
                f"Sync from {suggested_sync_start} or earlier and rerun."
            )
        trade_dates = cached_dates[start_index : end_index + 1]
        if len(trade_dates) < 2:
            raise ValueError("Full A momentum backtest requires at least two cached trade dates.")

        feature_dates = cached_dates[
            max(0, start_index - SelectionEventStudyEngine.factor_history_trade_days()) : end_index + 1
        ]
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=self.top_n,
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=[self.selection_variant],
            horizons=[1],
        )
        factor_frame = study_engine._build_factor_frame(feature_dates)
        price_map = study_engine._load_price_map(trade_dates)

        initial_cash = float(self.config.backtest.initial_cash)
        cash = initial_cash
        positions: dict[str, Tianzhu9Position] = {}
        trades: list[Tianzhu9Trade] = []
        equity_rows: list[dict] = []
        total_traded_value = 0.0
        loss_cooldown_until: dict[str, int] = {}

        for trade_offset, trade_date in enumerate(trade_dates):
            trade_index = start_index + trade_offset
            signal_trade_date = cached_dates[trade_index - 1]
            day_prices = price_map.get(trade_date, pd.DataFrame())
            if day_prices.empty:
                continue

            signal_frame = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_trade_date].copy()
            style_state = self._market_style_state(signal_frame)
            risk_off = bool(style_state["market_risk_off"])
            eligible_groups = set(style_state["eligible_groups"])
            selected = self._select_candidates(
                signal_frame=signal_frame,
                eligible_groups=eligible_groups,
                excluded_symbols={
                    symbol
                    for symbol, cooldown_until in loss_cooldown_until.items()
                    if cooldown_until >= trade_index
                },
                risk_off=risk_off,
            )
            selected_symbols = {row["ts_code"] for row in selected}

            sell_cash_box = {"cash": cash}
            total_traded_value += self._execute_sells(
                trade_date=trade_date,
                trade_index=trade_index,
                prices=day_prices,
                factor_frame=factor_frame,
                signal_trade_date=signal_trade_date,
                positions=positions,
                selected_symbols=selected_symbols,
                eligible_groups=eligible_groups,
                risk_off=risk_off,
                trades=trades,
                cash_ref=sell_cash_box,
                loss_cooldown_until=loss_cooldown_until,
            )
            cash = sell_cash_box["cash"]

            open_equity = self._mark_to_market_equity(cash, positions, day_prices, "open")
            buy_cash_box = {"cash": cash}
            total_traded_value += self._execute_buys(
                trade_date=trade_date,
                signal_trade_date=signal_trade_date,
                trade_index=trade_index,
                prices=day_prices,
                candidates=selected,
                open_equity=open_equity,
                positions=positions,
                trades=trades,
                cash_ref=buy_cash_box,
            )
            cash = buy_cash_box["cash"]

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
                    "market_breadth": style_state["market_breadth"],
                    "market_return_20d": style_state["market_return_20d"],
                    "market_source": style_state["market_source"],
                    "benchmark_close_to_ma20": style_state["benchmark_close_to_ma20"],
                    "eligible_groups": ",".join(sorted(eligible_groups)),
                    "risk_off": risk_off,
                }
            )

        equity_frame = pd.DataFrame(equity_rows)
        if equity_frame.empty:
            raise ValueError("Full A momentum backtest produced no equity rows.")

        returns = equity_frame["equity"].pct_change().fillna(0.0)
        ending_equity = float(equity_frame["equity"].iloc[-1])
        total_return = ending_equity / initial_cash - 1.0
        annual_return = (ending_equity / initial_cash) ** (252 / max(len(equity_frame), 1)) - 1.0
        drawdowns = equity_frame["equity"] / equity_frame["equity"].cummax() - 1.0
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
        groups_slug = "-".join(self.groups)
        filter_slug = (
            f"mb{self.market_min_breadth:.2f}-mr{self.market_min_return_20d:.2f}-"
            f"sb{self.style_min_breadth:.2f}-sr{self.style_min_return_20d:.2f}"
        ).replace("-", "m").replace(".", "p")
        stem = (
            f"full-a-momentum-{self.selection_variant}-top{self.top_n}-"
            f"{groups_slug}-exit-{self._exit_slug()}-filter-"
            f"{filter_slug}-{resolved_start}-{resolved_end}"
        )
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
            "strategy": "full_a_momentum",
            "selection_variant": self.selection_variant,
            "start_trade_date": resolved_start,
            "end_trade_date": resolved_end,
            "signal_lag_days": 1,
            "top_n": self.top_n,
            "groups": self.groups,
            "hold_days": self.hold_days,
            "max_hold_days": self.max_hold_days,
            "exit_strategy": self.exit_strategy,
            "exit_strategy_note": (
                "tiered trailing take-profit; no hard stop-loss, no fixed max-hold exit"
                if self.hard_exit_days is None
                else f"tiered trailing take-profit plus hard exit after {self.hard_exit_days} trade days"
            ),
            "hard_exit_days": self.hard_exit_days,
            "exit_rules": {
                "ma20_break": self.exit_ma20_break,
                "failure_days": self.exit_failure_days,
                "failure_min_peak_profit_pct": self.exit_failure_min_peak_profit_pct,
                "adaptive_trailing": self.exit_adaptive_trailing,
                "atr_multiplier": self.exit_atr_multiplier,
                "market_risk_exit": self.exit_market_risk,
                "industry_weak_exit": self.exit_industry_weak,
                "relative_weak_exit": self.exit_relative_weak,
                "relative_weak_5d_pct": self.exit_relative_weak_5d_pct,
                "relative_weak_20d_pct": self.exit_relative_weak_20d_pct,
                "volume_stall_exit": self.exit_volume_stall,
                "volume_stall_ratio": self.exit_volume_stall_ratio,
                "upper_shadow_exit": self.exit_upper_shadow,
                "upper_shadow_pct": self.exit_upper_shadow_pct,
            },
            "max_positions": self.max_positions,
            "min_avg_amount_yuan": self.min_avg_amount_yuan,
            "market_filter": {
                "market_min_breadth": self.market_min_breadth,
                "market_min_return_20d": self.market_min_return_20d,
                "style_min_breadth": self.style_min_breadth,
                "style_min_return_20d": self.style_min_return_20d,
                "style_score_weight": self.style_score_weight,
            },
            "enhanced_data": {
                "benchmark_index": self.config.market.benchmark,
                "market_source_counts": equity_frame["market_source"].value_counts(dropna=False).to_dict(),
                "sw_industry_rows": int(factor_frame["sw_l1_name"].notna().sum())
                if "sw_l1_name" in factor_frame.columns
                else 0,
                "financial_data_rows": int(factor_frame["financial_data_available"].fillna(False).sum())
                if "financial_data_available" in factor_frame.columns
                else 0,
            },
            "risk_off_days": int(equity_frame["risk_off"].sum()),
            "average_position_count": float(equity_frame["position_count"].mean()),
            "average_invested_ratio": float((1.0 - equity_frame["cash"] / equity_frame["equity"]).mean()),
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
            execution_mode="limit-swing",
            extend_on_repeat=True,
            max_hold_days=self.max_hold_days,
            equity_curve_path=equity_curve_path,
            summary_path=summary_path,
            trade_log_path=trade_log_path,
        )

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
            return cached_dates[max(1, end_index - 252)]
        requested = to_compact_date(start_date)
        eligible = [value for value in cached_dates if value >= requested and value <= end_date]
        if not eligible:
            raise ValueError(f"No cached trade date found on or after {requested}")
        return eligible[0]

    def _market_style_state(self, signal_frame: pd.DataFrame) -> dict:
        if signal_frame.empty:
            return {
                "market_breadth": 0.0,
                "market_return_20d": -1.0,
                "market_risk_off": True,
                "eligible_groups": [],
                "group_scores": {},
                "market_source": "empty",
                "benchmark_close_to_ma20": None,
            }
        market_breadth = float((signal_frame["close"] >= signal_frame["ma_20"]).mean())
        benchmark_return = signal_frame.get("benchmark_return_20d")
        benchmark_close_to_ma20_series = signal_frame.get("benchmark_close_to_ma20")
        benchmark_close_to_ma20 = None
        if benchmark_return is not None and benchmark_return.notna().any():
            market_return_20d = float(benchmark_return.dropna().iloc[-1])
            market_source = "benchmark_index"
            if benchmark_close_to_ma20_series is not None and benchmark_close_to_ma20_series.notna().any():
                benchmark_close_to_ma20 = float(benchmark_close_to_ma20_series.dropna().iloc[-1])
        else:
            market_return_20d = float(signal_frame["return_20d"].median())
            market_source = "stock_median"
        risk_off = market_breadth < self.market_min_breadth or market_return_20d < self.market_min_return_20d
        style_column = "style_group" if "style_group" in signal_frame.columns else "group"
        eligible_groups = []
        group_scores: dict[str, float] = {}
        for group, group_frame in signal_frame.groupby(style_column):
            breadth = float((group_frame["close"] >= group_frame["ma_20"]).mean())
            return_20d = float(group_frame["return_20d"].median())
            momentum_5d = float(group_frame["return_5d"].median())
            style_score = breadth + min(max((return_20d + 0.05) / 0.20, 0.0), 1.0) * 0.5
            style_score += min(max((momentum_5d + 0.03) / 0.12, 0.0), 1.0) * 0.2
            group_scores[str(group)] = style_score
            if breadth >= self.style_min_breadth and return_20d >= self.style_min_return_20d:
                eligible_groups.append(str(group))
        return {
            "market_breadth": market_breadth,
            "market_return_20d": market_return_20d,
            "market_risk_off": risk_off,
            "eligible_groups": [] if risk_off else eligible_groups,
            "group_scores": group_scores,
            "market_source": market_source,
            "benchmark_close_to_ma20": benchmark_close_to_ma20,
        }

    def _select_candidates(
        self,
        signal_frame: pd.DataFrame,
        eligible_groups: set[str],
        excluded_symbols: set[str],
        risk_off: bool,
    ) -> list[dict]:
        if signal_frame.empty or risk_off or not eligible_groups:
            return []
        score_column = f"{self.selection_variant}_score"
        style_column = "style_group" if "style_group" in signal_frame.columns else "group"
        frame = signal_frame.loc[
            signal_frame[style_column].isin(eligible_groups)
            & (~signal_frame["ts_code"].isin(excluded_symbols))
        ].copy()
        if frame.empty:
            return []
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=max(self.top_n, self.max_positions),
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=[self.selection_variant],
            horizons=[1],
        )
        frame = frame.loc[study_engine._variant_mask(frame, self.selection_variant)].copy()
        if frame.empty:
            return []
        style_state = self._market_style_state(signal_frame)
        group_scores = style_state["group_scores"]
        frame["selection_score"] = (
            frame[score_column].fillna(0.0)
            + frame[style_column].map(group_scores).fillna(0.0) * self.style_score_weight
        )
        selected = frame.sort_values(["selection_score", "avg_amount_20d_yuan"], ascending=[False, False]).head(self.top_n)
        rows = []
        for rank, row in enumerate(selected.to_dict(orient="records"), start=1):
            row["rank"] = rank
            row["score"] = float(row["selection_score"])
            rows.append(row)
        return rows

    def _execute_buys(
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
        if not candidates or open_equity <= 0 or len(positions) >= self.max_positions:
            return 0.0
        traded_value = 0.0
        target_slot_value = open_equity / self.max_positions
        for candidate in candidates:
            if len(positions) >= self.max_positions:
                break
            symbol = str(candidate["ts_code"])
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
            target_value = min(target_slot_value, cash_ref["cash"])
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
                highest_high=fill_price,
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

    def _execute_sells(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        factor_frame: pd.DataFrame,
        signal_trade_date: str,
        positions: dict[str, Tianzhu9Position],
        selected_symbols: set[str],
        eligible_groups: set[str],
        risk_off: bool,
        trades: list[Tianzhu9Trade],
        cash_ref: dict[str, float],
        loss_cooldown_until: dict[str, int],
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
            highest_price = max(position.highest_close, position.highest_high or 0.0)
            exit_check = tiered_trailing_take_profit(
                entry_price=position.entry_price,
                current_close=prev_close,
                highest_price=highest_price,
                levels=self._trailing_levels(feature),
            )
            should_exit = exit_check.should_exit
            if not should_exit and self._should_exit_ma20_break(feature, holding_days):
                should_exit = True
            if not should_exit and self._should_exit_failure(feature, position, highest_price, holding_days):
                should_exit = True
            if not should_exit and self._should_exit_market_risk(
                feature=feature,
                holding_days=holding_days,
                eligible_groups=eligible_groups,
                risk_off=risk_off,
            ):
                should_exit = True
            if not should_exit and self._should_exit_industry_weak(feature, holding_days):
                should_exit = True
            if not should_exit and self._should_exit_relative_weak(feature, holding_days):
                should_exit = True
            if not should_exit and self._should_exit_volume_stall(feature, position, highest_price, holding_days):
                should_exit = True
            if not should_exit and self._should_exit_upper_shadow(feature, position, highest_price, holding_days):
                should_exit = True
            if not should_exit and self.hard_exit_days is not None and holding_days >= self.hard_exit_days:
                should_exit = True
            if not should_exit:
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
            if pnl <= 0 and self.loss_cooldown_days > 0:
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

    def _trailing_levels(self, feature: pd.Series) -> tuple[tuple[float, float], ...]:
        if not self.exit_adaptive_trailing:
            return TIERED_TRAILING_TAKE_PROFIT_LEVELS
        atr_pct = _safe_float(feature.get("atr_20d_pct"))
        if atr_pct is None:
            return TIERED_TRAILING_TAKE_PROFIT_LEVELS
        return tuple(
            (profit_pct, max(drawdown_pct, atr_pct * self.exit_atr_multiplier))
            for profit_pct, drawdown_pct in TIERED_TRAILING_TAKE_PROFIT_LEVELS
        )

    def _should_exit_ma20_break(self, feature: pd.Series, holding_days: int) -> bool:
        if not self.exit_ma20_break or holding_days < 3:
            return False
        close = _safe_float(feature.get("close"))
        ma20 = _safe_float(feature.get("ma_20"))
        return close is not None and ma20 is not None and close < ma20

    def _should_exit_failure(
        self,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
    ) -> bool:
        if self.exit_failure_days is None or holding_days < self.exit_failure_days:
            return False
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        if peak_profit >= self.exit_failure_min_peak_profit_pct:
            return False
        close = _safe_float(feature.get("close"))
        ma20 = _safe_float(feature.get("ma_20"))
        return_5d = _safe_float(feature.get("return_5d"))
        return (
            (close is not None and ma20 is not None and close < ma20)
            or (return_5d is not None and return_5d < 0.0)
        )

    def _should_exit_market_risk(
        self,
        *,
        feature: pd.Series,
        holding_days: int,
        eligible_groups: set[str],
        risk_off: bool,
    ) -> bool:
        if not self.exit_market_risk or holding_days < 2:
            return False
        style_group = str(feature.get("style_group") or feature.get("group") or "")
        if not risk_off and (not style_group or style_group in eligible_groups):
            return False
        close = _safe_float(feature.get("close"))
        ma10 = _safe_float(feature.get("ma_10"))
        ma20 = _safe_float(feature.get("ma_20"))
        return_5d = _safe_float(feature.get("return_5d"))
        return (
            (close is not None and ma10 is not None and close < ma10)
            or (close is not None and ma20 is not None and close < ma20)
            or (return_5d is not None and return_5d < 0.0)
        )

    def _should_exit_industry_weak(self, feature: pd.Series, holding_days: int) -> bool:
        if not self.exit_industry_weak or holding_days < 3:
            return False
        style_return_20d = _safe_float(feature.get("style_return_20d_median"))
        style_breadth = _safe_float(feature.get("style_breadth_20d"))
        if style_return_20d is None and style_breadth is None:
            return False
        industry_weak = (
            (style_return_20d is not None and style_return_20d < self.style_min_return_20d)
            or (style_breadth is not None and style_breadth < self.style_min_breadth)
        )
        if not industry_weak:
            return False
        close = _safe_float(feature.get("close"))
        ma10 = _safe_float(feature.get("ma_10"))
        return_5d = _safe_float(feature.get("return_5d"))
        return (
            (close is not None and ma10 is not None and close < ma10)
            or (return_5d is not None and return_5d < 0.0)
        )

    def _should_exit_relative_weak(self, feature: pd.Series, holding_days: int) -> bool:
        if not self.exit_relative_weak or holding_days < 3:
            return False
        relative_5d = _safe_float(feature.get("relative_style_return_5d"))
        relative_20d = _safe_float(feature.get("relative_style_return_20d"))
        return_5d = _safe_float(feature.get("return_5d"))
        close = _safe_float(feature.get("close"))
        ma20 = _safe_float(feature.get("ma_20"))
        short_weak = (
            relative_5d is not None
            and relative_5d <= -self.exit_relative_weak_5d_pct
            and return_5d is not None
            and return_5d < 0.0
        )
        medium_weak = (
            relative_20d is not None
            and relative_20d <= -self.exit_relative_weak_20d_pct
            and close is not None
            and ma20 is not None
            and close < ma20
        )
        return short_weak or medium_weak

    def _should_exit_volume_stall(
        self,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
    ) -> bool:
        if not self.exit_volume_stall or holding_days < 3:
            return False
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        if peak_profit < 0.05:
            return False
        amount_ratio = _safe_float(feature.get("amount_ratio_5d"))
        return_5d = _safe_float(feature.get("return_5d"))
        close = _safe_float(feature.get("close"))
        ma5 = _safe_float(feature.get("ma_5"))
        upper_shadow = _safe_float(feature.get("upper_shadow_pct"))
        return (
            amount_ratio is not None
            and amount_ratio >= self.exit_volume_stall_ratio
            and return_5d is not None
            and return_5d <= 0.02
            and (
                (close is not None and ma5 is not None and close < ma5)
                or (upper_shadow is not None and upper_shadow >= 0.35)
            )
        )

    def _should_exit_upper_shadow(
        self,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
    ) -> bool:
        if not self.exit_upper_shadow or holding_days < 2:
            return False
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        close_to_ma20 = _safe_float(feature.get("close_to_ma_20"))
        if peak_profit < 0.08 and (close_to_ma20 is None or close_to_ma20 < 0.08):
            return False
        upper_shadow = _safe_float(feature.get("upper_shadow_pct"))
        amount_ratio = _safe_float(feature.get("amount_ratio_5d"))
        return (
            upper_shadow is not None
            and upper_shadow >= self.exit_upper_shadow_pct
            and (amount_ratio is None or amount_ratio >= 1.1)
        )

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

    def _update_position_highs(self, positions: dict[str, Tianzhu9Position], prices: pd.DataFrame) -> None:
        for position in positions.values():
            if position.symbol not in prices.index:
                continue
            close_price = float(prices.loc[position.symbol, "close"])
            high_price = float(prices.loc[position.symbol, "high"])
            if not math.isnan(close_price):
                position.highest_close = max(position.highest_close, close_price)
            if not math.isnan(high_price):
                position.highest_high = max(position.highest_high or position.highest_close, high_price)

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


def _safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _slug_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")
