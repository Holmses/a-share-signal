from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.backtest.ranking_event_study import _market_state
from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.ranking import SUPPORTED_RANKING_VARIANTS
from ashare_signal.strategy.ranking import build_ranking_snapshot
from ashare_signal.strategy.sell_reasons import sell_reason_counts, summarize_sell_reasons
from ashare_signal.utils.dates import to_compact_date


@dataclass(slots=True)
class RankingRotationPosition:
    symbol: str
    name: str
    shares: int
    entry_trade_date: str
    signal_trade_date: str
    entry_trade_index: int
    entry_price: float
    entry_cost: float
    score: float
    rank: int


@dataclass(slots=True)
class RankingRotationTrade:
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
    rank: int | None
    score: float | None
    reason: str
    pnl: float | None = None


@dataclass(slots=True)
class RankingRotationBacktestResult:
    start_trade_date: str
    end_trade_date: str
    variant: str
    top_k: int
    candidate_buffer_k: int
    drop_n: int
    rebalance_interval_days: int
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
    risk_off_days: int
    average_position_count: float
    average_invested_ratio: float
    equity_curve_path: Path
    trade_log_path: Path
    summary_path: Path


@dataclass(slots=True)
class SellDecision:
    symbol: str
    reason: str
    rank_position: int | None
    score: float | None


class RankingRotationBacktestEngine:
    """TopK/DropN trial backtest for research-only ranking rotation."""

    DEFAULT_VARIANT = "quality_momentum_rank"
    DEFAULT_GROUPS = ("main", "chinext", "star")

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        variant: str = DEFAULT_VARIANT,
        groups: list[str] | None = None,
        top_k: int = 5,
        candidate_buffer_k: int = 20,
        drop_n: int = 1,
        max_positions: int | None = None,
        min_score_edge: float = 0.02,
        min_holding_days: int = 3,
        rotation_min_holding_days: int = 5,
        rebalance_interval_days: int = 1,
        min_avg_amount_yuan: float = 50_000_000.0,
        market_min_breadth: float = 0.50,
        market_min_return_20d: float = 0.0,
        risk_off_cash_guard: bool = True,
        risk_off_exit: bool = False,
        lot_size: int | None = None,
    ) -> None:
        if variant not in SUPPORTED_RANKING_VARIANTS:
            raise ValueError(f"Unsupported ranking variant: {variant}")
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.variant = variant
        self.groups = groups or list(self.DEFAULT_GROUPS)
        self.top_k = max(int(top_k), 1)
        self.candidate_buffer_k = max(int(candidate_buffer_k), self.top_k)
        self.drop_n = max(int(drop_n), 1)
        self.max_positions = max(int(max_positions or top_k), 1)
        self.min_score_edge = float(min_score_edge)
        self.min_holding_days = max(int(min_holding_days), 0)
        self.rotation_min_holding_days = max(int(rotation_min_holding_days), 0)
        self.rebalance_interval_days = max(int(rebalance_interval_days), 1)
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.market_min_breadth = float(market_min_breadth)
        self.market_min_return_20d = float(market_min_return_20d)
        self.risk_off_cash_guard = bool(risk_off_cash_guard)
        self.risk_off_exit = bool(risk_off_exit)
        self.lot_size = int(lot_size or config.backtest.lot_size)

    def run(self, start_date: date | None = None, end_date: date | None = None) -> RankingRotationBacktestResult:
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
                "Ranking rotation backtest needs at least "
                f"{required_history} complete trade days before start date {resolved_start}. "
                f"Sync from {suggested_sync_start} or earlier and rerun."
            )

        trade_dates = cached_dates[start_index : end_index + 1]
        feature_dates = cached_dates[
            max(0, start_index - SelectionEventStudyEngine.factor_history_trade_days()) : end_index + 1
        ]
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=self.candidate_buffer_k,
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=["quality_momentum"],
            horizons=[1],
        )
        factor_frame = study_engine._build_factor_frame(feature_dates)
        price_map = study_engine._load_price_map(trade_dates)

        initial_cash = float(self.config.backtest.initial_cash)
        cash = initial_cash
        positions: dict[str, RankingRotationPosition] = {}
        trades: list[RankingRotationTrade] = []
        equity_rows: list[dict] = []
        total_traded_value = 0.0

        for trade_offset, trade_date in enumerate(trade_dates):
            trade_index = start_index + trade_offset
            signal_trade_date = cached_dates[trade_index - 1]
            day_prices = price_map.get(trade_date, pd.DataFrame())
            if day_prices.empty:
                continue
            signal_frame = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_trade_date].copy()
            ranking = self._ranking_for_signal(signal_frame)
            market_state = _market_state(
                signal_frame,
                market_min_breadth=self.market_min_breadth,
                market_min_return_20d=self.market_min_return_20d,
            )
            risk_off = market_state["market_state"] == "risk_off"
            feature_index = signal_frame.set_index("ts_code") if not signal_frame.empty else pd.DataFrame()
            is_rebalance_day = trade_offset % self.rebalance_interval_days == 0

            sell_decisions = self._build_sell_decisions(
                positions=positions,
                ranking=ranking,
                trade_index=trade_index,
                risk_off=risk_off,
                is_rebalance_day=is_rebalance_day,
            )
            sell_cash_box = {"cash": cash}
            total_traded_value += self._execute_sells(
                trade_date=trade_date,
                prices=day_prices,
                feature_index=feature_index,
                positions=positions,
                sell_decisions=sell_decisions,
                trades=trades,
                cash_ref=sell_cash_box,
            )
            cash = sell_cash_box["cash"]

            open_equity = self._mark_to_market_equity(cash, positions, day_prices, "open")
            buy_cash_box = {"cash": cash}
            total_traded_value += self._execute_buys(
                trade_date=trade_date,
                signal_trade_date=signal_trade_date,
                trade_index=trade_index,
                prices=day_prices,
                feature_index=feature_index,
                ranking=ranking,
                positions=positions,
                open_equity=open_equity,
                risk_off=risk_off,
                is_rebalance_day=is_rebalance_day,
                trades=trades,
                cash_ref=buy_cash_box,
            )
            cash = buy_cash_box["cash"]

            close_equity = self._mark_to_market_equity(cash, positions, day_prices, "close")
            equity_rows.append(
                {
                    "trade_date": trade_date,
                    "signal_trade_date": signal_trade_date,
                    "equity": close_equity,
                    "cash": cash,
                    "position_count": len(positions),
                    "invested_ratio": 1.0 - cash / close_equity if close_equity > 0 else 0.0,
                    "risk_off": risk_off,
                    "market_breadth": market_state["market_breadth"],
                    "market_return_20d": market_state["market_return_20d"],
                    "top_ranked": ",".join(ranking.head(self.top_k)["ts_code"].astype(str)) if not ranking.empty else "",
                    "holdings": ",".join(sorted(positions)),
                }
            )

        equity_frame = pd.DataFrame(equity_rows)
        if equity_frame.empty:
            raise ValueError("Ranking rotation backtest produced no equity rows.")

        result = self._build_result(
            resolved_start=resolved_start,
            resolved_end=resolved_end,
            equity_frame=equity_frame,
            trades=trades,
            initial_cash=initial_cash,
            total_traded_value=total_traded_value,
        )
        return result

    def _ranking_for_signal(self, signal_frame: pd.DataFrame) -> pd.DataFrame:
        if signal_frame.empty:
            return pd.DataFrame()
        ranking = build_ranking_snapshot(signal_frame, self.config, variant=self.variant)
        ranking = ranking.loc[ranking["is_tradeable"].fillna(False).astype(bool)].copy()
        if ranking.empty:
            return ranking
        ranking["rank_position"] = pd.to_numeric(ranking["rank_position"], errors="coerce")
        ranking["rank_score"] = pd.to_numeric(ranking["rank_score"], errors="coerce")
        return ranking.dropna(subset=["rank_position", "rank_score"]).sort_values(["rank_position", "ts_code"])

    def _build_sell_decisions(
        self,
        *,
        positions: dict[str, RankingRotationPosition],
        ranking: pd.DataFrame,
        trade_index: int,
        risk_off: bool,
        is_rebalance_day: bool = True,
    ) -> list[SellDecision]:
        if not positions:
            return []
        ranking_index = ranking.set_index("ts_code") if not ranking.empty else pd.DataFrame()
        held_symbols = set(positions)
        best_new = self._best_new_candidate(ranking, held_symbols)
        rotation_allowed = is_rebalance_day and (not risk_off or not self.risk_off_cash_guard)
        decisions: list[SellDecision] = []
        for symbol, position in positions.items():
            holding_days = trade_index - position.entry_trade_index + 1
            rank_position = None
            score = None
            if not ranking_index.empty and symbol in ranking_index.index:
                row = ranking_index.loc[symbol]
                rank_position = int(row["rank_position"])
                score = float(row["rank_score"])
            if rank_position is None:
                decisions.append(SellDecision(symbol=symbol, reason="missing_rank", rank_position=None, score=score))
            elif self.risk_off_exit and risk_off and holding_days >= self.min_holding_days:
                decisions.append(
                    SellDecision(
                        symbol=symbol,
                        reason="market_risk_exit",
                        rank_position=rank_position,
                        score=score,
                    )
                )
            elif (
                rotation_allowed
                and rank_position > self.candidate_buffer_k
                and holding_days >= self.min_holding_days
                and best_new is not None
                and score is not None
                and float(best_new["rank_score"]) >= score + self.min_score_edge
            ):
                decisions.append(
                    SellDecision(
                        symbol=symbol,
                        reason="rotation_rank_drop",
                        rank_position=rank_position,
                        score=score,
                    )
                )

        if (
            rotation_allowed
            and len(decisions) < self.drop_n
            and len(positions) >= self.max_positions
            and best_new is not None
        ):
            weakest = self._weakest_holding(positions, ranking_index, trade_index)
            if weakest is not None:
                weakest_position, weakest_rank, weakest_score = weakest
                holding_days = trade_index - weakest_position.entry_trade_index + 1
                already_selling = {decision.symbol for decision in decisions}
                if (
                    weakest_position.symbol not in already_selling
                    and holding_days >= self.rotation_min_holding_days
                    and float(best_new["rank_score"]) >= weakest_score + self.min_score_edge
                ):
                    decisions.append(
                        SellDecision(
                            symbol=weakest_position.symbol,
                            reason="score_edge_rotation",
                            rank_position=weakest_rank,
                            score=weakest_score,
                        )
                    )

        decisions = sorted(
            decisions,
            key=lambda decision: (
                decision.reason != "missing_rank",
                -(decision.rank_position or 10**9),
                decision.score if decision.score is not None else -1.0,
            ),
        )
        return decisions[: self.drop_n]

    def _best_new_candidate(self, ranking: pd.DataFrame, held_symbols: set[str]) -> pd.Series | None:
        if ranking.empty:
            return None
        best_new = ranking.loc[
            (ranking["rank_position"] <= self.candidate_buffer_k)
            & (~ranking["ts_code"].isin(held_symbols))
        ].head(self.top_k)
        if best_new.empty:
            return None
        return best_new.iloc[0]

    def _weakest_holding(
        self,
        positions: dict[str, RankingRotationPosition],
        ranking_index: pd.DataFrame,
        trade_index: int,
    ) -> tuple[RankingRotationPosition, int, float] | None:
        rows = []
        if ranking_index.empty:
            return None
        for symbol, position in positions.items():
            if symbol not in ranking_index.index:
                continue
            row = ranking_index.loc[symbol]
            rows.append((position, int(row["rank_position"]), float(row["rank_score"])))
        if not rows:
            return None
        return sorted(rows, key=lambda item: (item[1], -item[2]), reverse=True)[0]

    def _execute_sells(
        self,
        *,
        trade_date: str,
        prices: pd.DataFrame,
        feature_index: pd.DataFrame,
        positions: dict[str, RankingRotationPosition],
        sell_decisions: list[SellDecision],
        trades: list[RankingRotationTrade],
        cash_ref: dict[str, float],
    ) -> float:
        traded_value = 0.0
        for decision in sell_decisions:
            symbol = decision.symbol
            if symbol not in positions or symbol not in prices.index:
                continue
            position = positions[symbol]
            if symbol not in feature_index.index:
                continue
            prev_close = _safe_float(feature_index.loc[symbol].get("close"))
            row = prices.loc[symbol]
            day_open = _safe_float(row.get("open"))
            day_high = _safe_float(row.get("high"))
            if prev_close is None or day_open is None or day_high is None:
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
            pnl = net_amount - position.entry_cost
            cash_ref["cash"] += net_amount
            trades.append(
                RankingRotationTrade(
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
                    rank=decision.rank_position,
                    score=decision.score,
                    reason=decision.reason,
                    pnl=pnl,
                )
            )
            traded_value += gross_amount
            del positions[symbol]
        return traded_value

    def _execute_buys(
        self,
        *,
        trade_date: str,
        signal_trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        feature_index: pd.DataFrame,
        ranking: pd.DataFrame,
        positions: dict[str, RankingRotationPosition],
        open_equity: float,
        risk_off: bool,
        is_rebalance_day: bool,
        trades: list[RankingRotationTrade],
        cash_ref: dict[str, float],
    ) -> float:
        if ranking.empty or open_equity <= 0 or len(positions) >= self.max_positions:
            return 0.0
        if not is_rebalance_day:
            return 0.0
        if self.risk_off_cash_guard and risk_off:
            return 0.0
        traded_value = 0.0
        target_slot_value = open_equity / self.max_positions
        held_symbols = set(positions)
        candidates = ranking.loc[
            (ranking["rank_position"] <= self.candidate_buffer_k)
            & (~ranking["ts_code"].isin(held_symbols))
        ].head(self.drop_n)
        for _, candidate in candidates.iterrows():
            if len(positions) >= self.max_positions:
                break
            symbol = str(candidate["ts_code"])
            if symbol not in prices.index or symbol not in feature_index.index:
                continue
            prev_close = _safe_float(feature_index.loc[symbol].get("close"))
            row = prices.loc[symbol]
            day_open = _safe_float(row.get("open"))
            day_low = _safe_float(row.get("low"))
            if prev_close is None or day_open is None or day_low is None:
                continue
            limit_price = round(prev_close * (1 + self.config.pricing.buy_markup), 2)
            if day_low > limit_price:
                continue
            fill_price = day_open if day_open <= limit_price else limit_price
            target_value = min(target_slot_value, cash_ref["cash"])
            shares = (int(target_value / fill_price) // self.lot_size) * self.lot_size
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
            positions[symbol] = RankingRotationPosition(
                symbol=symbol,
                name=str(candidate.get("name") or symbol),
                shares=shares,
                entry_trade_date=trade_date,
                signal_trade_date=signal_trade_date,
                entry_trade_index=trade_index,
                entry_price=fill_price,
                entry_cost=net_amount,
                score=float(candidate["rank_score"]),
                rank=int(candidate["rank_position"]),
            )
            trades.append(
                RankingRotationTrade(
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
                    rank=int(candidate["rank_position"]),
                    score=float(candidate["rank_score"]),
                    reason="rank_topk_buy",
                    pnl=None,
                )
            )
            traded_value += gross_amount
        return traded_value

    def _mark_to_market_equity(
        self,
        cash: float,
        positions: dict[str, RankingRotationPosition],
        prices: pd.DataFrame,
        price_field: str,
    ) -> float:
        equity = cash
        for position in positions.values():
            if position.symbol not in prices.index:
                continue
            price = _safe_float(prices.loc[position.symbol].get(price_field))
            if price is None:
                continue
            equity += position.shares * price
        return equity

    def _build_result(
        self,
        *,
        resolved_start: str,
        resolved_end: str,
        equity_frame: pd.DataFrame,
        trades: list[RankingRotationTrade],
        initial_cash: float,
        total_traded_value: float,
    ) -> RankingRotationBacktestResult:
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
        edge_slug = _slug_float(self.min_score_edge)
        risk_slug = ("cashguard" if self.risk_off_cash_guard else "riskbuy") + (
            "-riskexit" if self.risk_off_exit else ""
        )
        stem = (
            f"ranking-rotation-{self.variant.replace('_', '-')}-top{self.top_k}-"
            f"buffer{self.candidate_buffer_k}-drop{self.drop_n}-edge{edge_slug}-"
            f"hold{self.min_holding_days}-rot{self.rotation_min_holding_days}-"
            f"rebalance{self.rebalance_interval_days}-"
            f"{risk_slug}-"
            f"{resolved_start}-{resolved_end}"
        )
        equity_curve_path = reports_dir / f"{stem}-equity.csv"
        trade_log_path = reports_dir / f"{stem}-trades.csv"
        summary_path = reports_dir / f"{stem}-summary.json"
        equity_frame.to_csv(equity_curve_path, index=False)
        pd.DataFrame([asdict(trade) for trade in trades]).to_csv(trade_log_path, index=False)
        reason_counts = sell_reason_counts(sell_trades)
        reason_summary = summarize_sell_reasons(sell_trades)
        summary_payload = {
            "strategy": "ranking_rotation_topk_dropn",
            "variant": self.variant,
            "start_trade_date": resolved_start,
            "end_trade_date": resolved_end,
            "top_k": self.top_k,
            "candidate_buffer_k": self.candidate_buffer_k,
            "drop_n": self.drop_n,
            "max_positions": self.max_positions,
            "min_score_edge": self.min_score_edge,
            "min_holding_days": self.min_holding_days,
            "rotation_min_holding_days": self.rotation_min_holding_days,
            "rebalance_interval_days": self.rebalance_interval_days,
            "risk_off_cash_guard": self.risk_off_cash_guard,
            "risk_off_exit": self.risk_off_exit,
            "market_filter": {
                "market_min_breadth": self.market_min_breadth,
                "market_min_return_20d": self.market_min_return_20d,
            },
            "risk_off_days": int(equity_frame["risk_off"].sum()),
            "average_position_count": float(equity_frame["position_count"].mean()),
            "average_invested_ratio": float(equity_frame["invested_ratio"].mean()),
            "sell_reason_counts": reason_counts,
            "sell_reason_summary": reason_summary,
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
        return RankingRotationBacktestResult(
            start_trade_date=resolved_start,
            end_trade_date=resolved_end,
            variant=self.variant,
            top_k=self.top_k,
            candidate_buffer_k=self.candidate_buffer_k,
            drop_n=self.drop_n,
            rebalance_interval_days=self.rebalance_interval_days,
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
            risk_off_days=int(equity_frame["risk_off"].sum()),
            average_position_count=float(equity_frame["position_count"].mean()),
            average_invested_ratio=float(equity_frame["invested_ratio"].mean()),
            equity_curve_path=equity_curve_path,
            trade_log_path=trade_log_path,
            summary_path=summary_path,
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
            return cached_dates[max(1, end_index - 504)]
        requested = to_compact_date(start_date)
        eligible = [value for value in cached_dates if requested <= value <= end_date]
        if not eligible:
            raise ValueError(f"No cached trade date found on or after {requested}")
        return eligible[0]


def _safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or number <= 0:
        return None
    return number


def _slug_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")
