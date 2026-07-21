from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from ashare_signal.backtest.full_a_momentum import FullAMomentumBacktestEngine
from ashare_signal.backtest.tianzhu9_like import Tianzhu9Position
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository


DEFAULT_POST_EXIT_HORIZONS = (5, 10, 20)


@dataclass(frozen=True, slots=True)
class ExitCandidate:
    name: str
    hard_exit_days: int | None = 23
    ma20_break: bool = False
    market_risk: bool = False
    style_rotation: bool = False
    high_drawdown_pct: float | None = None
    chandelier_atr_multiplier: float | None = None
    trend_decay: bool = False
    winner_bypass_peak_pct: float | None = None
    risk_off_failed_days: int | None = None


EXIT_CANDIDATES: dict[str, ExitCandidate] = {
    candidate.name: candidate
    for candidate in (
        ExitCandidate("baseline_hard23"),
        ExitCandidate("winner_bypass_08", winner_bypass_peak_pct=0.08),
        ExitCandidate("riskoff12", winner_bypass_peak_pct=0.08, risk_off_failed_days=12),
        ExitCandidate("riskoff15", winner_bypass_peak_pct=0.08, risk_off_failed_days=15),
        ExitCandidate("riskoff18", winner_bypass_peak_pct=0.08, risk_off_failed_days=18),
        ExitCandidate("ma20", ma20_break=True),
        ExitCandidate("highdd06", high_drawdown_pct=0.06),
        ExitCandidate("highdd08", high_drawdown_pct=0.08),
        ExitCandidate("highdd10", high_drawdown_pct=0.10),
        ExitCandidate("chandelier2p5", chandelier_atr_multiplier=2.5),
        ExitCandidate("chandelier3p0", chandelier_atr_multiplier=3.0),
        ExitCandidate("chandelier3p5", chandelier_atr_multiplier=3.5),
        ExitCandidate("trend_decay", trend_decay=True),
        ExitCandidate("market_risk", market_risk=True),
        ExitCandidate("style_rotation", style_rotation=True),
        ExitCandidate(
            "riskoff15_chandelier3p0",
            chandelier_atr_multiplier=3.0,
            winner_bypass_peak_pct=0.08,
            risk_off_failed_days=15,
        ),
    )
}
DEFAULT_EXIT_CANDIDATES = tuple(EXIT_CANDIDATES)


@dataclass(slots=True)
class ExitTimingStudyResult:
    start_trade_date: str
    end_trade_date: str
    event_count: int
    candidates: tuple[str, ...]
    events_path: Path
    summary_csv_path: Path
    attribution_path: Path
    markdown_path: Path
    summary_path: Path


class ExitTimingStudyEngine:
    """Research-only exit comparison on a frozen set of baseline entries."""

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        trades_path: Path,
        equity_path: Path,
        candidates: tuple[str, ...] = DEFAULT_EXIT_CANDIDATES,
        max_horizon_days: int = 80,
        post_exit_horizons: tuple[int, ...] = DEFAULT_POST_EXIT_HORIZONS,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.trades_path = trades_path
        self.equity_path = equity_path
        unknown = sorted(set(candidates) - set(EXIT_CANDIDATES))
        if unknown:
            raise ValueError(f"Unknown exit candidate(s): {', '.join(unknown)}")
        self.candidates = tuple(dict.fromkeys(candidates))
        self.max_horizon_days = max(int(max_horizon_days), 2)
        self.post_exit_horizons = tuple(sorted({10, *(max(int(value), 1) for value in post_exit_horizons)}))

    def run(self) -> ExitTimingStudyResult:
        trades = pd.read_csv(self.trades_path)
        equity = pd.read_csv(self.equity_path)
        _validate_inputs(trades, equity)
        trades["trade_date"] = _normalize_dates(trades["trade_date"])
        equity["trade_date"] = _normalize_dates(equity["trade_date"])
        equity = equity.sort_values("trade_date").reset_index(drop=True)
        entries = _frozen_entries(trades)
        if entries.empty:
            raise ValueError("Baseline trade log contains no BUY entries.")

        start_date = str(equity["trade_date"].iloc[0])
        end_date = str(equity["trade_date"].iloc[-1])
        all_dates = self.repository.cached_daily_trade_dates()
        study_dates = [value for value in all_dates if value <= end_date]
        if start_date not in study_dates or end_date not in study_dates:
            raise ValueError("Baseline equity dates are not covered by the local daily cache.")
        first_entry_index = study_dates.index(str(entries["entry_date"].min()))
        load_dates = study_dates[max(0, first_entry_index - 25) :]
        bars = _load_symbol_bars(
            repository=self.repository,
            trade_dates=load_dates,
            symbols=set(entries["symbol"].astype(str)),
        )
        if bars.empty:
            raise ValueError("No daily bars were found for frozen baseline entries.")
        price_map = {
            str(trade_date): frame.set_index("symbol")
            for trade_date, frame in bars.groupby("trade_date", sort=False)
        }
        date_index = {value: index for index, value in enumerate(study_dates)}
        state_by_date = equity.set_index("trade_date")["market_state"].astype(str).to_dict()
        eligible_by_date = {
            str(row.trade_date): _split_groups(row.eligible_groups)
            for row in equity.itertuples()
        }

        engines = {
            name: _candidate_engine(
                candidate=EXIT_CANDIDATES[name],
                config=self.config,
                repository=self.repository,
                base_dir=self.base_dir,
            )
            for name in self.candidates
        }
        event_rows: list[dict] = []
        for entry in entries.to_dict(orient="records"):
            for name in self.candidates:
                event_rows.append(
                    _simulate_frozen_entry(
                        entry=entry,
                        candidate=EXIT_CANDIDATES[name],
                        engine=engines[name],
                        trade_dates=study_dates,
                        date_index=date_index,
                        price_map=price_map,
                        state_by_date=state_by_date,
                        eligible_by_date=eligible_by_date,
                        config=self.config,
                        max_horizon_days=self.max_horizon_days,
                        post_exit_horizons=self.post_exit_horizons,
                    )
                )
        events = pd.DataFrame(event_rows)
        summary = _summarize_candidates(events, self.post_exit_horizons)
        attribution = _build_attribution(events, self.post_exit_horizons)

        reports_dir = self.base_dir / self.config.paths.reports_dir / "exit-timing-study"
        reports_dir.mkdir(parents=True, exist_ok=True)
        source_id = hashlib.sha1(
            f"{self.trades_path.resolve()}|{self.equity_path.resolve()}".encode("utf-8")
        ).hexdigest()[:10]
        stem = f"exit-timing-frozen-{source_id}-{start_date}-{end_date}"
        events_path = reports_dir / f"{stem}-events.csv"
        summary_csv_path = reports_dir / f"{stem}-summary.csv"
        attribution_path = reports_dir / f"{stem}-attribution.csv"
        markdown_path = reports_dir / f"{stem}.md"
        summary_path = reports_dir / f"{stem}-summary.json"
        events.to_csv(events_path, index=False)
        summary.to_csv(summary_csv_path, index=False)
        attribution.to_csv(attribution_path, index=False)

        payload = {
            "strategy": "exit_timing_frozen_entry_study",
            "research_only": True,
            "trades_path": str(self.trades_path),
            "equity_path": str(self.equity_path),
            "start_trade_date": start_date,
            "end_trade_date": end_date,
            "signal_lag_days": 1,
            "entry_count": int(len(entries)),
            "candidate_count": len(self.candidates),
            "candidates": [candidate_to_dict(EXIT_CANDIDATES[name]) for name in self.candidates],
            "max_horizon_days": self.max_horizon_days,
            "post_exit_horizons": list(self.post_exit_horizons),
            "execution_assumptions": {
                "entry_set": "Frozen BUY rows from the baseline trade log.",
                "signal_time": "Exit rules use information available at signal-day close.",
                "execution_time": "Earliest execution is next trade-day open.",
                "sell_fill": "Open or configured sell limit when the daily high reaches that limit.",
                "unfilled_orders": "Suspended or unreachable sell limits remain open and are re-evaluated next day.",
                "fees": "Configured commission, minimum fee, and stamp duty.",
                "terminal_marks": "Unexited events are marked at the study horizon and reported as censored.",
            },
            "timing_labels": {
                "too_early": "10-day post-exit MFE >= 5% and 10-day close return >= 2%.",
                "too_late": "Exit loss <= -5% with holding >= 10 days or peak giveback >= 10 percentage points.",
                "reasonable": "Complete forward sample that is neither too early nor too late.",
                "insufficient_forward_data": "No complete 10-day post-exit path.",
            },
            "exit_priority": [
                "tiered_trailing_take_profit",
                "MA20/failure/high-drawdown/Chandelier/trend-decay",
                "style/market/industry/relative/volume/upper-shadow",
                "risk_off_failed_hard_exit",
                "standard_hard_exit",
            ],
            "summary": summary.to_dict(orient="records"),
            "events_path": str(events_path),
            "summary_csv_path": str(summary_csv_path),
            "attribution_path": str(attribution_path),
            "markdown_path": str(markdown_path),
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(_render_markdown(payload, summary), encoding="utf-8")
        return ExitTimingStudyResult(
            start_trade_date=start_date,
            end_trade_date=end_date,
            event_count=int(len(events)),
            candidates=self.candidates,
            events_path=events_path,
            summary_csv_path=summary_csv_path,
            attribution_path=attribution_path,
            markdown_path=markdown_path,
            summary_path=summary_path,
        )


def candidate_to_dict(candidate: ExitCandidate) -> dict:
    return {
        field: getattr(candidate, field)
        for field in candidate.__dataclass_fields__
    }


def _candidate_engine(
    *,
    candidate: ExitCandidate,
    config: AppConfig,
    repository: DataRepository,
    base_dir: Path,
) -> FullAMomentumBacktestEngine:
    return FullAMomentumBacktestEngine(
        config=config,
        repository=repository,
        base_dir=base_dir,
        hard_exit_days=candidate.hard_exit_days,
        exit_ma20_break=candidate.ma20_break,
        exit_market_risk=candidate.market_risk,
        exit_style_rotation=candidate.style_rotation,
        exit_high_drawdown_pct=candidate.high_drawdown_pct,
        exit_chandelier_atr_multiplier=candidate.chandelier_atr_multiplier,
        exit_trend_decay=candidate.trend_decay,
        exit_winner_hard_exit_bypass_peak_pct=candidate.winner_bypass_peak_pct,
        exit_risk_off_failed_hard_exit_days=candidate.risk_off_failed_days,
        exit_volume_stall=False,
    )


def _frozen_entries(trades: pd.DataFrame) -> pd.DataFrame:
    queues: dict[str, list[int]] = {}
    rows: list[dict] = []
    for _, trade in trades.iterrows():
        symbol = str(trade.get("symbol") or "")
        action = str(trade.get("action") or "").upper()
        if action == "BUY":
            row = {
                "event_id": len(rows) + 1,
                "entry_date": str(trade["trade_date"]),
                "symbol": symbol,
                "name": str(trade.get("name") or symbol),
                "shares": int(trade["shares"]),
                "entry_price": float(trade["price"]),
                "entry_cost": float(trade["net_amount"]),
                "signal_trade_date": str(trade.get("signal_trade_date") or ""),
                "rank": _safe_int(trade.get("rank")),
                "score": _safe_float(trade.get("score")) or 0.0,
                "entry_market_state": str(trade.get("market_state") or "unknown"),
                "style_group": str(trade.get("style_group") or "unknown"),
                "actual_exit_date": None,
                "actual_exit_reason": None,
            }
            rows.append(row)
            queues.setdefault(symbol, []).append(len(rows) - 1)
        elif action == "SELL" and queues.get(symbol):
            index = queues[symbol].pop(0)
            rows[index]["actual_exit_date"] = str(trade["trade_date"])
            rows[index]["actual_exit_reason"] = str(trade.get("exit_reason") or trade.get("reason") or "")
    return pd.DataFrame(rows)


def _load_symbol_bars(
    *,
    repository: DataRepository,
    trade_dates: list[str],
    symbols: set[str],
) -> pd.DataFrame:
    frames = []
    for trade_date in trade_dates:
        try:
            frame = repository.load_daily(trade_date)
        except FileNotFoundError:
            continue
        frame = frame.loc[frame["ts_code"].astype(str).isin(symbols)].copy()
        if frame.empty:
            continue
        columns = [column for column in ("trade_date", "ts_code", "open", "high", "low", "close") if column in frame]
        frames.append(frame[columns])
    if not frames:
        return pd.DataFrame()
    bars = pd.concat(frames, ignore_index=True).rename(columns={"ts_code": "symbol"})
    bars["trade_date"] = _normalize_dates(bars["trade_date"])
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=["open", "high", "low", "close"]).sort_values(["symbol", "trade_date"])
    grouped = bars.groupby("symbol", group_keys=False)
    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    bars["atr_20d"] = true_range.groupby(bars["symbol"]).transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    for days in (5, 10, 20):
        bars[f"ma_{days}"] = grouped["close"].transform(
            lambda series, window=days: series.rolling(window, min_periods=window).mean()
        )
    bars["return_5d"] = grouped["close"].pct_change(5)
    return bars.reset_index(drop=True)


def _simulate_frozen_entry(
    *,
    entry: dict,
    candidate: ExitCandidate,
    engine: FullAMomentumBacktestEngine,
    trade_dates: list[str],
    date_index: dict[str, int],
    price_map: dict[str, pd.DataFrame],
    state_by_date: dict[str, str],
    eligible_by_date: dict[str, set[str]],
    config: AppConfig,
    max_horizon_days: int,
    post_exit_horizons: tuple[int, ...],
) -> dict:
    entry_date = str(entry["entry_date"])
    symbol = str(entry["symbol"])
    entry_index = date_index[entry_date]
    terminal_index = min(entry_index + max_horizon_days - 1, len(trade_dates) - 1)
    position = Tianzhu9Position(
        symbol=symbol,
        name=str(entry["name"]),
        shares=int(entry["shares"]),
        entry_trade_date=entry_date,
        signal_trade_date=str(entry["signal_trade_date"]),
        entry_trade_index=entry_index,
        entry_price=float(entry["entry_price"]),
        entry_cost=float(entry["entry_cost"]),
        highest_close=float(entry["entry_price"]),
        highest_high=float(entry["entry_price"]),
        score=float(entry["score"]),
        rank=int(entry["rank"]),
        market_state=str(entry["entry_market_state"]),
        style_group=str(entry["style_group"]),
    )
    highest_price = position.entry_price
    lowest_price = position.entry_price
    exit_index: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    signal_date: str | None = None
    unfilled_exit_days = 0

    for execution_index in range(entry_index + 1, terminal_index + 1):
        signal_index = execution_index - 1
        signal_trade_date = trade_dates[signal_index]
        execution_date = trade_dates[execution_index]
        signal_prices = price_map.get(signal_trade_date)
        if signal_prices is not None and symbol in signal_prices.index:
            feature = signal_prices.loc[symbol].copy()
        else:
            feature = _latest_symbol_feature(
                price_map=price_map,
                trade_dates=trade_dates,
                signal_index=signal_index,
                symbol=symbol,
            )
        if feature is None:
            continue
        feature["style_group"] = str(entry["style_group"])
        feature["group"] = str(entry["style_group"])
        highest_price = max(highest_price, float(feature["high"]), float(feature["close"]))
        lowest_price = min(lowest_price, float(feature["low"]), float(feature["close"]))
        holding_days = execution_index - entry_index + 1
        market_state = state_by_date.get(execution_date, "unknown")
        reason = engine._exit_reason(
            feature=feature,
            position=position,
            highest_price=highest_price,
            holding_days=holding_days,
            eligible_groups=eligible_by_date.get(execution_date, set()),
            risk_off=market_state == "risk_off",
        )
        if reason is None:
            continue
        execution_prices = price_map.get(execution_date)
        if execution_prices is None or symbol not in execution_prices.index:
            unfilled_exit_days += 1
            continue
        execution_bar = execution_prices.loc[symbol]
        day_open = float(execution_bar["open"])
        day_high = float(execution_bar["high"])
        previous_close = float(feature["close"])
        limit_price = round(previous_close * (1.0 - config.pricing.sell_markdown), 2)
        if any(not math.isfinite(value) or value <= 0 for value in (day_open, day_high, limit_price)):
            unfilled_exit_days += 1
            continue
        if day_high < limit_price:
            unfilled_exit_days += 1
            continue
        exit_index = execution_index
        exit_price = day_open if day_open >= limit_price else limit_price
        exit_reason = reason
        signal_date = signal_trade_date
        break

    censored = exit_index is None
    if censored:
        exit_index = terminal_index
        exit_date = trade_dates[exit_index]
        terminal_prices = price_map.get(exit_date)
        if terminal_prices is None or symbol not in terminal_prices.index:
            available = [
                index
                for index in range(terminal_index, entry_index - 1, -1)
                if price_map.get(trade_dates[index]) is not None
                and symbol in price_map[trade_dates[index]].index
            ]
            if not available:
                raise ValueError(f"No terminal price for {symbol} after {entry_date}")
            exit_index = available[0]
            exit_date = trade_dates[exit_index]
            terminal_prices = price_map[exit_date]
        terminal_bar = terminal_prices.loc[symbol]
        exit_price = float(terminal_bar["close"])
        highest_price = max(highest_price, float(terminal_bar["high"]), exit_price)
        lowest_price = min(lowest_price, float(terminal_bar["low"]), exit_price)
        exit_reason = "study_horizon_mark"
        signal_date = exit_date

    exit_date = trade_dates[exit_index]
    gross_amount = float(entry["shares"]) * float(exit_price)
    fees = max(
        gross_amount * (config.backtest.commission_rate + config.backtest.stamp_duty_rate),
        5.0,
    )
    pnl = gross_amount - fees - float(entry["entry_cost"])
    gross_return = float(exit_price) / float(entry["entry_price"]) - 1.0
    net_return = pnl / float(entry["entry_cost"])
    mfe = highest_price / float(entry["entry_price"]) - 1.0
    mae = lowest_price / float(entry["entry_price"]) - 1.0
    peak_capture = gross_return / mfe if mfe > 0 else None
    peak_giveback = mfe - gross_return
    row = {
        "event_id": int(entry["event_id"]),
        "candidate": candidate.name,
        "entry_date": entry_date,
        "entry_year": entry_date[:4],
        "symbol": symbol,
        "name": str(entry["name"]),
        "entry_market_state": str(entry["entry_market_state"]),
        "style_group": str(entry["style_group"]),
        "entry_price": float(entry["entry_price"]),
        "shares": int(entry["shares"]),
        "exit_signal_date": signal_date,
        "exit_date": exit_date,
        "exit_market_state": state_by_date.get(exit_date, "unknown"),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "holding_days": int(exit_index - entry_index + 1),
        "gross_return": gross_return,
        "net_return": net_return,
        "pnl": pnl,
        "mfe": mfe,
        "mae": mae,
        "peak_capture_ratio": peak_capture,
        "peak_giveback": peak_giveback,
        "winner_reached_8pct": mfe >= 0.08,
        "failed_capital_days": int(exit_index - entry_index + 1) if mfe < 0.08 and net_return <= 0 else 0,
        "unfilled_exit_days": unfilled_exit_days,
        "censored": censored,
        "actual_exit_date": entry.get("actual_exit_date"),
        "actual_exit_reason": entry.get("actual_exit_reason"),
    }
    for horizon in post_exit_horizons:
        metrics = _post_exit_metrics(
            symbol=symbol,
            exit_price=float(exit_price),
            exit_index=exit_index,
            horizon=horizon,
            trade_dates=trade_dates,
            price_map=price_map,
        )
        row.update({f"post_exit_{key}_{horizon}d": value for key, value in metrics.items()})
    row["timing_label"] = _classify_exit_timing(row)
    return row


def _latest_symbol_feature(
    *,
    price_map: dict[str, pd.DataFrame],
    trade_dates: list[str],
    signal_index: int,
    symbol: str,
) -> pd.Series | None:
    for index in range(signal_index, -1, -1):
        prices = price_map.get(trade_dates[index])
        if prices is not None and symbol in prices.index:
            return prices.loc[symbol].copy()
    return None


def _post_exit_metrics(
    *,
    symbol: str,
    exit_price: float,
    exit_index: int,
    horizon: int,
    trade_dates: list[str],
    price_map: dict[str, pd.DataFrame],
) -> dict[str, float | None]:
    target_index = exit_index + horizon
    if target_index >= len(trade_dates):
        return {"return": None, "mfe": None, "mae": None}
    rows = []
    for index in range(exit_index + 1, target_index + 1):
        prices = price_map.get(trade_dates[index])
        if prices is None or symbol not in prices.index:
            return {"return": None, "mfe": None, "mae": None}
        rows.append(prices.loc[symbol])
    return {
        "return": float(rows[-1]["close"]) / exit_price - 1.0,
        "mfe": max(float(row["high"]) for row in rows) / exit_price - 1.0,
        "mae": min(float(row["low"]) for row in rows) / exit_price - 1.0,
    }


def _classify_exit_timing(row: dict) -> str:
    post_return = _safe_float(row.get("post_exit_return_10d"))
    post_mfe = _safe_float(row.get("post_exit_mfe_10d"))
    if post_return is None or post_mfe is None:
        return "insufficient_forward_data"
    if post_mfe >= 0.05 and post_return >= 0.02:
        return "too_early"
    if (
        float(row["gross_return"]) <= -0.05
        and int(row["holding_days"]) >= 10
    ) or float(row["peak_giveback"]) >= 0.10:
        return "too_late"
    return "reasonable"


def _summarize_candidates(events: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for candidate, frame in events.groupby("candidate", sort=False):
        rows.append(_summary_row(frame, candidate=candidate, group_type="candidate", group=candidate, horizons=horizons))
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def _build_attribution(events: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for candidate, candidate_frame in events.groupby("candidate", sort=False):
        for group_type, column in (
            ("exit_reason", "exit_reason"),
            ("entry_market_state", "entry_market_state"),
            ("exit_market_state", "exit_market_state"),
            ("entry_year", "entry_year"),
            ("style_group", "style_group"),
        ):
            for group, frame in candidate_frame.groupby(column, dropna=False):
                rows.append(
                    _summary_row(
                        frame,
                        candidate=candidate,
                        group_type=group_type,
                        group=str(group),
                        horizons=horizons,
                    )
                )
    return pd.DataFrame(rows)


def _summary_row(
    frame: pd.DataFrame,
    *,
    candidate: str,
    group_type: str,
    group: str,
    horizons: tuple[int, ...],
) -> dict:
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").dropna()
    net_returns = pd.to_numeric(frame["net_return"], errors="coerce").dropna()
    profits = pnl.loc[pnl > 0]
    losses = pnl.loc[pnl < 0]
    average_profit = float(profits.mean()) if not profits.empty else None
    average_loss = float(losses.mean()) if not losses.empty else None
    row = {
        "candidate": candidate,
        "group_type": group_type,
        "group": group,
        "events": int(len(frame)),
        "censored_events": int(frame["censored"].fillna(False).sum()),
        "average_net_return": float(net_returns.mean()) if not net_returns.empty else None,
        "median_net_return": float(net_returns.median()) if not net_returns.empty else None,
        "total_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()) if not pnl.empty else None,
        "average_profit": average_profit,
        "average_loss": average_loss,
        "payoff_ratio": (
            float(average_profit / abs(average_loss))
            if average_profit is not None and average_loss not in (None, 0.0)
            else None
        ),
        "profit_factor": (
            float(profits.sum() / abs(losses.sum()))
            if not losses.empty and float(losses.sum()) != 0.0
            else None
        ),
        "average_holding_days": float(frame["holding_days"].mean()),
        "average_mfe": float(frame["mfe"].mean()),
        "average_mae": float(frame["mae"].mean()),
        "winner_peak_capture": _mean_numeric(
            frame.loc[frame["winner_reached_8pct"].fillna(False), "peak_capture_ratio"]
        ),
        "average_peak_giveback": float(frame["peak_giveback"].mean()),
        "too_early_rate": float((frame["timing_label"] == "too_early").mean()),
        "too_late_rate": float((frame["timing_label"] == "too_late").mean()),
        "failed_capital_days": int(frame["failed_capital_days"].sum()),
        "unfilled_exit_days": int(frame["unfilled_exit_days"].sum()),
        "worst_net_return": float(net_returns.min()) if not net_returns.empty else None,
        "bottom_10_total_pnl": float(pnl.nsmallest(10).sum()) if not pnl.empty else 0.0,
    }
    for horizon in horizons:
        row[f"average_post_exit_return_{horizon}d"] = _mean_numeric(frame[f"post_exit_return_{horizon}d"])
        row[f"average_post_exit_mfe_{horizon}d"] = _mean_numeric(frame[f"post_exit_mfe_{horizon}d"])
    return row


def _render_markdown(payload: dict, summary: pd.DataFrame) -> str:
    columns = [
        "candidate",
        "events",
        "average_net_return",
        "win_rate",
        "profit_factor",
        "average_holding_days",
        "average_mfe",
        "average_mae",
        "winner_peak_capture",
        "too_early_rate",
        "too_late_rate",
        "failed_capital_days",
        "bottom_10_total_pnl",
    ]
    lines = [
        "# Frozen-entry exit timing study",
        "",
        f"Period: {payload['start_trade_date']} to {payload['end_trade_date']}",
        "",
        "All candidates use the same baseline BUY events. Signals use close information and execute no earlier than the next trade-day open with the configured sell-limit approximation.",
        "",
        "## Candidate summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in summary.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif column in {
                "average_net_return",
                "win_rate",
                "average_mfe",
                "average_mae",
                "winner_peak_capture",
                "too_early_rate",
                "too_late_rate",
            }:
                values.append(f"{float(value) * 100:.2f}%")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "## Timing definitions",
        "",
        "- Too early: 10-day post-exit MFE is at least 5% and close return is at least 2%.",
        "- Too late: exit loss is at least 5% after 10 days, or peak giveback reaches 10 percentage points.",
        "- Reasonable: a complete forward sample that meets neither condition.",
        "- Incomplete 10-day forward paths are labeled insufficient and are not silently treated as reasonable.",
        "",
        "## Files",
        "",
        f"- Events: `{payload['events_path']}`",
        f"- Summary CSV: `{payload['summary_csv_path']}`",
        f"- Attribution: `{payload['attribution_path']}`",
    ]
    return "\n".join(lines) + "\n"


def _validate_inputs(trades: pd.DataFrame, equity: pd.DataFrame) -> None:
    trade_columns = {"trade_date", "action", "symbol", "shares", "price", "net_amount"}
    equity_columns = {"trade_date", "market_state", "eligible_groups"}
    missing_trades = sorted(trade_columns - set(trades.columns))
    missing_equity = sorted(equity_columns - set(equity.columns))
    if missing_trades:
        raise ValueError(f"Baseline trades are missing columns: {', '.join(missing_trades)}")
    if missing_equity:
        raise ValueError(f"Baseline equity is missing columns: {', '.join(missing_equity)}")
    if equity.empty:
        raise ValueError("Baseline equity is empty.")


def _split_groups(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _normalize_dates(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _safe_int(value: object) -> int:
    numeric = _safe_float(value)
    return int(numeric) if numeric is not None else 0


def _mean_numeric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None
