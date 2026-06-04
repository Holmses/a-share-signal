from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import json
import math
import urllib.error
import urllib.request

import pandas as pd

from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.utils.dates import to_compact_date


@dataclass(slots=True)
class RiskOffStandaloneStudyResult:
    start_entry_date: str
    end_entry_date: str
    start_signal_date: str
    end_signal_date: str
    risk_off_days: int
    event_count: int
    events_path: Path
    daily_path: Path
    summary_csv_path: Path
    data_health_path: Path
    markdown_path: Path
    summary_path: Path


RISK_OFF_TYPES = ("severe", "both_mild", "breadth_only", "return_only")


class RiskOffStandaloneStudyEngine:
    """Standalone risk-off opportunity study.

    This engine deliberately does not read or write live positions, daily plans,
    or scheduler state. It reuses cached market data and emits research reports.
    """

    DEFAULT_GROUPS = ("main", "chinext", "star")
    DEFAULT_HORIZONS = (5, 10, 20)
    DEFAULT_JUNQUANT_SCORE_URL = "https://junquant.com/api/crowding/score-v2/latest"

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        top_n: int = 5,
        min_avg_amount_yuan: float = 50_000_000.0,
        defensive_min_avg_amount_yuan: float = 80_000_000.0,
        groups: list[str] | None = None,
        horizons: list[int] | None = None,
        market_min_breadth: float = 0.50,
        market_min_return_20d: float = 0.0,
        check_external: bool = False,
        external_timeout_seconds: float = 10.0,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.top_n = max(int(top_n), 1)
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.defensive_min_avg_amount_yuan = float(defensive_min_avg_amount_yuan)
        self.groups = groups or list(self.DEFAULT_GROUPS)
        self.horizons = sorted({int(value) for value in (horizons or list(self.DEFAULT_HORIZONS)) if int(value) > 0})
        if not self.horizons:
            raise ValueError("At least one positive horizon is required.")
        self.market_min_breadth = float(market_min_breadth)
        self.market_min_return_20d = float(market_min_return_20d)
        self.check_external = bool(check_external)
        self.external_timeout_seconds = max(float(external_timeout_seconds), 1.0)

    def run(self, start_date: date | None = None, end_date: date | None = None) -> RiskOffStandaloneStudyResult:
        cached_dates = self.repository.complete_daily_cache_dates()
        if not cached_dates:
            raise ValueError("Daily Tushare cache is empty. Run `ashare-signal sync-tushare` first.")

        resolved_end = _resolve_cached_end(cached_dates, end_date)
        resolved_start = _resolve_cached_start(cached_dates, start_date, resolved_end)
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
                "Standalone risk-off study needs at least "
                f"{required_history} complete trade days before start date {resolved_start}. "
                f"Sync from {suggested_sync_start} or earlier and rerun."
            )

        max_horizon = max(self.horizons)
        signal_start_index = start_index - 1
        signal_end_index = min(end_index - 1, len(cached_dates) - 1 - max_horizon)
        if signal_end_index < signal_start_index:
            raise ValueError(
                "Standalone risk-off study has no signal dates with full forward horizon. "
                f"Need {max_horizon} cached trade days after each entry date."
            )

        feature_dates = cached_dates[
            max(0, signal_start_index - SelectionEventStudyEngine.factor_history_trade_days()) : signal_end_index + 1
        ]
        price_dates = cached_dates[start_index : signal_end_index + max_horizon + 1]
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=self.top_n,
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=["quality_momentum"],
            horizons=[1],
        )
        factor_frame = study_engine._build_factor_frame(feature_dates)
        factor_frame = self._merge_daily_basic_research_fields(factor_frame, feature_dates)
        price_map = study_engine._load_price_map(price_dates)

        events: list[dict] = []
        daily_rows: list[dict] = []
        risk_off_days = 0
        for signal_index in range(signal_start_index, signal_end_index + 1):
            signal_trade_date = cached_dates[signal_index]
            entry_trade_date = cached_dates[signal_index + 1]
            day_frame = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_trade_date].copy()
            if day_frame.empty:
                continue
            market_state = _market_state(
                day_frame,
                market_min_breadth=self.market_min_breadth,
                market_min_return_20d=self.market_min_return_20d,
            )
            if not bool(market_state["risk_off"]):
                continue

            risk_off_days += 1
            risk_off_type = _classify_risk_off_type(
                market_breadth=float(market_state["market_breadth"]),
                market_return_20d=float(market_state["market_return_20d"]),
            )
            baseline = self._select_baseline_candidates(day_frame, study_engine)
            defensive = _select_defensive_candidates(
                day_frame,
                top_n=self.top_n,
                min_avg_amount_yuan=self.defensive_min_avg_amount_yuan,
            )
            daily_rows.append(
                {
                    "signal_trade_date": signal_trade_date,
                    "entry_trade_date": entry_trade_date,
                    "risk_off_type": risk_off_type,
                    "market_breadth": market_state["market_breadth"],
                    "market_return_20d": market_state["market_return_20d"],
                    "market_source": market_state["market_source"],
                    "baseline_count": int(len(baseline)),
                    "defensive_count": int(len(defensive)),
                    "candidate_count": int(len(day_frame)),
                }
            )
            events.extend(
                self._build_fixed_horizon_events(
                    selected=baseline,
                    strategy="baseline",
                    signal_trade_date=signal_trade_date,
                    risk_off_type=risk_off_type,
                    entry_index=signal_index + 1,
                    cached_dates=cached_dates,
                    price_map=price_map,
                )
            )
            events.extend(
                self._build_fixed_horizon_events(
                    selected=defensive,
                    strategy="defensive",
                    signal_trade_date=signal_trade_date,
                    risk_off_type=risk_off_type,
                    entry_index=signal_index + 1,
                    cached_dates=cached_dates,
                    price_map=price_map,
                )
            )
            events.extend(
                self._build_elastic_exit_events(
                    selected=baseline,
                    signal_trade_date=signal_trade_date,
                    risk_off_type=risk_off_type,
                    entry_index=signal_index + 1,
                    cached_dates=cached_dates,
                    price_map=price_map,
                )
            )

        events_frame = pd.DataFrame(events)
        daily_frame = pd.DataFrame(daily_rows)
        if events_frame.empty:
            raise ValueError("Standalone risk-off study produced no events.")
        summary_frame = self._summarize(events_frame)
        health_frame = self._build_data_health_frame(
            resolved_start=resolved_start,
            resolved_end=resolved_end,
            complete_dates=cached_dates,
        )

        reports_dir = self.base_dir / self.config.paths.reports_dir / "risk-off-standalone"
        reports_dir.mkdir(parents=True, exist_ok=True)
        horizons_slug = "h" + "-".join(str(value) for value in self.horizons)
        stem = f"risk-off-standalone-top{self.top_n}-{horizons_slug}-{resolved_start}-{resolved_end}"
        events_path = reports_dir / f"{stem}-events.csv"
        daily_path = reports_dir / f"{stem}-daily.csv"
        summary_csv_path = reports_dir / f"{stem}-summary.csv"
        data_health_path = reports_dir / f"{stem}-data-health.csv"
        markdown_path = reports_dir / f"{stem}.md"
        summary_path = reports_dir / f"{stem}-summary.json"

        events_frame.to_csv(events_path, index=False)
        daily_frame.to_csv(daily_path, index=False)
        summary_frame.to_csv(summary_csv_path, index=False)
        health_frame.to_csv(data_health_path, index=False)

        payload = {
            "strategy": "risk_off_standalone",
            "start_entry_date": resolved_start,
            "end_entry_date": cached_dates[signal_end_index + 1],
            "start_signal_date": cached_dates[signal_start_index],
            "end_signal_date": cached_dates[signal_end_index],
            "requested_end_date": resolved_end,
            "top_n": self.top_n,
            "groups": self.groups,
            "horizons": self.horizons,
            "risk_off_days": risk_off_days,
            "event_count": int(len(events_frame)),
            "market_filter": {
                "market_min_breadth": self.market_min_breadth,
                "market_min_return_20d": self.market_min_return_20d,
            },
            "events_path": str(events_path),
            "daily_path": str(daily_path),
            "summary_csv_path": str(summary_csv_path),
            "data_health_path": str(data_health_path),
            "markdown_path": str(markdown_path),
            "summary": summary_frame.to_dict(orient="records"),
            "data_health": health_frame.to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(
            self._render_markdown(
                payload=payload,
                summary_frame=summary_frame,
                health_frame=health_frame,
                daily_frame=daily_frame,
            ),
            encoding="utf-8",
        )

        return RiskOffStandaloneStudyResult(
            start_entry_date=resolved_start,
            end_entry_date=cached_dates[signal_end_index + 1],
            start_signal_date=cached_dates[signal_start_index],
            end_signal_date=cached_dates[signal_end_index],
            risk_off_days=risk_off_days,
            event_count=int(len(events_frame)),
            events_path=events_path,
            daily_path=daily_path,
            summary_csv_path=summary_csv_path,
            data_health_path=data_health_path,
            markdown_path=markdown_path,
            summary_path=summary_path,
        )

    def _merge_daily_basic_research_fields(self, factor_frame: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
        extras = []
        for trade_date in trade_dates:
            try:
                frame = self.repository.load_daily_basic(trade_date)
            except FileNotFoundError:
                continue
            columns = [
                column
                for column in ("ts_code", "trade_date", "dv_ttm", "dv_ratio", "pe_ttm", "pb", "circ_mv")
                if column in frame.columns
            ]
            if {"ts_code", "trade_date"}.issubset(columns):
                extras.append(frame[columns])
        if not extras:
            for column in ("dv_ttm", "dv_ratio", "pe_ttm", "pb", "circ_mv"):
                if column not in factor_frame.columns:
                    factor_frame[column] = pd.NA
            return factor_frame

        extra_frame = pd.concat(extras, ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")
        for column in ("dv_ttm", "dv_ratio", "pe_ttm", "pb", "circ_mv"):
            if column in extra_frame.columns:
                extra_frame[column] = pd.to_numeric(extra_frame[column], errors="coerce")
        return factor_frame.merge(extra_frame, on=["ts_code", "trade_date"], how="left")

    def _select_baseline_candidates(
        self,
        day_frame: pd.DataFrame,
        study_engine: SelectionEventStudyEngine,
    ) -> pd.DataFrame:
        frame = day_frame.loc[study_engine._variant_mask(day_frame, "quality_momentum")].copy()
        if frame.empty:
            return frame
        frame["risk_off_study_score"] = frame["quality_momentum_score"].fillna(0.0)
        return frame.sort_values(["risk_off_study_score", "avg_amount_20d_yuan"], ascending=[False, False]).head(
            self.top_n
        )

    def _build_fixed_horizon_events(
        self,
        *,
        selected: pd.DataFrame,
        strategy: str,
        signal_trade_date: str,
        risk_off_type: str,
        entry_index: int,
        cached_dates: list[str],
        price_map: dict[str, pd.DataFrame],
    ) -> list[dict]:
        rows = []
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            event = _build_base_event(
                row=row,
                strategy=strategy,
                signal_trade_date=signal_trade_date,
                risk_off_type=risk_off_type,
                rank=rank,
                entry_index=entry_index,
                cached_dates=cached_dates,
                price_map=price_map,
            )
            if event is None:
                continue
            if _fill_forward_metrics(event, self.horizons, entry_index, cached_dates, price_map):
                rows.append(event)
        return rows

    def _build_elastic_exit_events(
        self,
        *,
        selected: pd.DataFrame,
        signal_trade_date: str,
        risk_off_type: str,
        entry_index: int,
        cached_dates: list[str],
        price_map: dict[str, pd.DataFrame],
    ) -> list[dict]:
        rows = []
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            event = _build_base_event(
                row=row,
                strategy="elastic_exit",
                signal_trade_date=signal_trade_date,
                risk_off_type=risk_off_type,
                rank=rank,
                entry_index=entry_index,
                cached_dates=cached_dates,
                price_map=price_map,
            )
            if event is None:
                continue
            bars = _load_bars(
                symbol=str(row["ts_code"]),
                entry_index=entry_index,
                horizon=10,
                cached_dates=cached_dates,
                price_map=price_map,
            )
            if bars is None:
                continue
            exit_result = _simulate_elastic_exit(float(event["entry_price"]), bars)
            event.update(exit_result)
            rows.append(event)
        return rows

    def _summarize(self, events_frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        fixed_frame = events_frame.loc[events_frame["strategy"].isin(["baseline", "defensive"])]
        for keys, frame in _summary_groups(fixed_frame):
            strategy, risk_off_type = keys
            for horizon in self.horizons:
                column = f"close_return_{horizon}d"
                if column not in frame.columns:
                    continue
                returns = pd.to_numeric(frame[column], errors="coerce").dropna()
                if returns.empty:
                    continue
                rows.append(
                    {
                        "metric_type": "fixed_horizon",
                        "strategy": strategy,
                        "risk_off_type": risk_off_type,
                        "horizon": horizon,
                        "events": int(len(returns)),
                        "avg_return": float(returns.mean()),
                        "median_return": float(returns.median()),
                        "win_rate": float((returns > 0).mean()),
                        "return_gt_5pct_rate": float((returns >= 0.05).mean()),
                        "return_lt_minus_5pct_rate": float((returns <= -0.05).mean()),
                        "avg_mfe": float(pd.to_numeric(frame[f"mfe_{horizon}d"], errors="coerce").mean()),
                        "avg_mae": float(pd.to_numeric(frame[f"mae_{horizon}d"], errors="coerce").mean()),
                    }
                )

        elastic_frame = events_frame.loc[events_frame["strategy"] == "elastic_exit"].copy()
        for keys, frame in _summary_groups(elastic_frame):
            strategy, risk_off_type = keys
            returns = pd.to_numeric(frame["exit_return_net"], errors="coerce").dropna()
            if returns.empty:
                continue
            rows.append(
                {
                    "metric_type": "elastic_exit",
                    "strategy": strategy,
                    "risk_off_type": risk_off_type,
                    "horizon": 10,
                    "events": int(len(returns)),
                    "avg_return": float(returns.mean()),
                    "median_return": float(returns.median()),
                    "win_rate": float((returns > 0).mean()),
                    "return_gt_5pct_rate": float((returns >= 0.05).mean()),
                    "return_lt_minus_5pct_rate": float((returns <= -0.05).mean()),
                    "avg_mfe": float(pd.to_numeric(frame["exit_mfe"], errors="coerce").mean()),
                    "avg_mae": float(pd.to_numeric(frame["exit_mae"], errors="coerce").mean()),
                    "avg_exit_day": float(pd.to_numeric(frame["exit_day"], errors="coerce").mean()),
                }
            )
        return pd.DataFrame(rows).sort_values(["metric_type", "strategy", "risk_off_type", "horizon"]).reset_index(
            drop=True
        )

    def _build_data_health_frame(
        self,
        *,
        resolved_start: str,
        resolved_end: str,
        complete_dates: list[str],
    ) -> pd.DataFrame:
        rows = []
        in_range_dates = [value for value in complete_dates if resolved_start <= value <= resolved_end]
        try:
            open_dates = self.repository.open_trade_dates_between(resolved_start, resolved_end)
        except (FileNotFoundError, ValueError):
            open_dates = []
        missing_dates = sorted(set(open_dates) - set(in_range_dates)) if open_dates else []
        rows.append(
            {
                "source": "tushare_daily_and_daily_basic_cache",
                "status": "available" if in_range_dates else "missing",
                "rating": "A" if in_range_dates and not missing_dates else "B",
                "start_date": in_range_dates[0] if in_range_dates else None,
                "end_date": in_range_dates[-1] if in_range_dates else None,
                "rows_or_files": len(in_range_dates),
                "missing_dates": len(missing_dates),
                "field_completeness": _field_completeness(
                    self.repository,
                    in_range_dates[-1] if in_range_dates else None,
                    required_fields=("ts_code", "trade_date", "open", "high", "low", "close", "amount"),
                    loader="daily",
                ),
                "note": "Primary local cache used by this standalone backtest.",
            }
        )
        rows.append(
            {
                "source": "tushare_daily_basic_valuation_fields",
                "status": "available" if in_range_dates else "missing",
                "rating": "A" if in_range_dates else "B",
                "start_date": in_range_dates[0] if in_range_dates else None,
                "end_date": in_range_dates[-1] if in_range_dates else None,
                "rows_or_files": len(in_range_dates),
                "missing_dates": len(missing_dates),
                "field_completeness": _field_completeness(
                    self.repository,
                    in_range_dates[-1] if in_range_dates else None,
                    required_fields=("dv_ttm", "pe_ttm", "pb", "total_mv", "circ_mv"),
                    loader="daily_basic",
                ),
                "note": "Required for defensive dividend and valuation screens.",
            }
        )
        rows.append(_path_health_row("tushare_index_member_all_SW2021", self.repository.tushare_root / "index_member_all" / "SW2021.csv", "A"))
        rows.append(_path_health_row("tushare_benchmark_index_daily", self.repository.tushare_root / "index_daily" / f"{self.config.market.benchmark}.csv", "A"))
        rows.append(
            _path_health_row(
                "tushare_moneyflow_optional",
                self.repository.tushare_root / "moneyflow",
                "B",
                note="Optional for future flow studies; not required by the first standalone backtest.",
            )
        )
        rows.extend(self._external_health_rows())
        return pd.DataFrame(rows)

    def _external_health_rows(self) -> list[dict]:
        rows = [
            {
                "source": "yfinance_external",
                "status": "not_used",
                "rating": "B",
                "start_date": None,
                "end_date": None,
                "rows_or_files": 0,
                "missing_dates": None,
                "field_completeness": None,
                "note": "Useful for ETF/index proxies, but must be cached and checked before entering research backtests.",
            },
            {
                "source": "cftc_cboe_external",
                "status": "not_used",
                "rating": "C",
                "start_date": None,
                "end_date": None,
                "rows_or_files": 0,
                "missing_dates": None,
                "field_completeness": None,
                "note": "High-quality public sources, but collection is slower and not required for the first A-share study.",
            },
        ]
        if not self.check_external:
            rows.append(
                {
                    "source": "junquant_score_api",
                    "status": "not_checked",
                    "rating": "C",
                    "start_date": None,
                    "end_date": None,
                    "rows_or_files": 0,
                    "missing_dates": None,
                    "field_completeness": None,
                    "note": "Third-party API is observation-only. Run with --check-external to test current availability.",
                }
            )
            return rows

        started = datetime.utcnow()
        try:
            request = urllib.request.Request(
                self.DEFAULT_JUNQUANT_SCORE_URL,
                headers={"User-Agent": "ashare-signal-risk-off-study/0.1"},
            )
            with urllib.request.urlopen(request, timeout=self.external_timeout_seconds) as response:
                body = response.read().decode("utf-8")
            payload = json.loads(body)
            available = bool(payload.get("available"))
            rows.append(
                {
                    "source": "junquant_score_api",
                    "status": "available" if available else "unavailable",
                    "rating": "C",
                    "start_date": payload.get("date"),
                    "end_date": payload.get("date"),
                    "rows_or_files": 1 if available else 0,
                    "missing_dates": None,
                    "field_completeness": 1.0 if available else 0.0,
                    "note": (
                        "Observation-only third-party source; "
                        f"confirmed_score={payload.get('confirmed_score')}, "
                        f"latency_seconds={(datetime.utcnow() - started).total_seconds():.2f}"
                    ),
                }
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            rows.append(
                {
                    "source": "junquant_score_api",
                    "status": "unavailable",
                    "rating": "C",
                    "start_date": None,
                    "end_date": None,
                    "rows_or_files": 0,
                    "missing_dates": None,
                    "field_completeness": 0.0,
                    "note": f"Observation-only source failed: {error}",
                }
            )
        return rows

    def _render_markdown(
        self,
        *,
        payload: dict,
        summary_frame: pd.DataFrame,
        health_frame: pd.DataFrame,
        daily_frame: pd.DataFrame,
    ) -> str:
        lines = [
            "# Standalone Risk-Off Opportunity Study",
            "",
            "This report is research-only. It does not affect the production strategy, daily plans, or positions.",
            "",
            "## Run Metadata",
            "",
            f"- Entry dates: {payload['start_entry_date']} to {payload['end_entry_date']}",
            f"- Signal dates: {payload['start_signal_date']} to {payload['end_signal_date']}",
            f"- Risk-off days: {payload['risk_off_days']}",
            f"- Events: {payload['event_count']}",
            f"- Horizons: {', '.join(str(value) for value in payload['horizons'])} trade days",
            "",
            "## Risk-Off Day Mix",
            "",
        ]
        if daily_frame.empty:
            lines.append("- No risk-off days.")
        else:
            for risk_off_type, count in daily_frame["risk_off_type"].value_counts().sort_index().items():
                lines.append(f"- {risk_off_type}: {int(count)}")
        lines.extend(["", "## Summary", ""])
        lines.extend(_markdown_table(summary_frame, max_rows=24))
        lines.extend(["", "## Data Health", ""])
        lines.extend(_markdown_table(health_frame, max_rows=12))
        lines.extend(
            [
                "",
                "## Output Files",
                "",
                f"- Events: `{payload['events_path']}`",
                f"- Daily states: `{payload['daily_path']}`",
                f"- Summary CSV: `{payload['summary_csv_path']}`",
                f"- Data health: `{payload['data_health_path']}`",
            ]
        )
        return "\n".join(lines) + "\n"


def _market_state(
    signal_frame: pd.DataFrame,
    *,
    market_min_breadth: float,
    market_min_return_20d: float,
) -> dict:
    if signal_frame.empty:
        return {
            "market_breadth": 0.0,
            "market_return_20d": -1.0,
            "risk_off": True,
            "market_source": "empty",
        }
    market_breadth = float((signal_frame["close"] >= signal_frame["ma_20"]).mean())
    benchmark_return = signal_frame.get("benchmark_return_20d")
    if benchmark_return is not None and benchmark_return.notna().any():
        market_return_20d = float(benchmark_return.dropna().iloc[-1])
        market_source = "benchmark_index"
    else:
        market_return_20d = float(signal_frame["return_20d"].median())
        market_source = "stock_median"
    return {
        "market_breadth": market_breadth,
        "market_return_20d": market_return_20d,
        "risk_off": market_breadth < market_min_breadth or market_return_20d < market_min_return_20d,
        "market_source": market_source,
    }


def _classify_risk_off_type(market_breadth: float, market_return_20d: float) -> str:
    if market_breadth < 0.25 or market_return_20d < -0.05:
        return "severe"
    if market_breadth < 0.50 and market_return_20d < 0.0:
        return "both_mild"
    if market_breadth < 0.50:
        return "breadth_only"
    return "return_only"


def _select_defensive_candidates(
    day_frame: pd.DataFrame,
    *,
    top_n: int,
    min_avg_amount_yuan: float,
) -> pd.DataFrame:
    frame = _add_research_ranks(day_frame)
    if frame.empty:
        return frame
    volatility_cut = frame["volatility_20d"].quantile(0.55)
    mask = (
        (frame["dv_ttm"].fillna(0.0) >= 1.5)
        & (frame["pe_ttm"].fillna(999.0) > 0.0)
        & (frame["pe_ttm"].fillna(999.0) <= 35.0)
        & (frame["pb"].fillna(999.0) <= 4.0)
        & (frame["return_20d"].fillna(-1.0) >= -0.05)
        & (frame["close_to_ma_20"].fillna(-1.0) >= -0.06)
        & (frame["volatility_20d"].fillna(1.0) <= volatility_cut)
        & (frame["avg_amount_20d_yuan"].fillna(0.0) >= float(min_avg_amount_yuan))
    )
    frame = frame.loc[mask].copy()
    if frame.empty:
        return frame
    frame["risk_off_study_score"] = (
        frame["dividend_rank"].fillna(0.0) * 0.30
        + frame["low_vol_score"].fillna(0.0) * 0.26
        + frame["ret20_rank"].fillna(0.0) * 0.14
        + frame["financial_quality_score"].fillna(0.5) * 0.12
        + frame["market_cap_rank"].fillna(0.0) * 0.10
        + frame["pb_score"].fillna(0.0) * 0.08
    )
    return frame.sort_values(["risk_off_study_score", "avg_amount_20d_yuan"], ascending=[False, False]).head(top_n)


def _add_research_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["low_vol_score"] = 1.0 - frame["volatility_20d"].rank(pct=True, ascending=True).fillna(1.0)
    frame["ret20_rank"] = frame["return_20d"].rank(pct=True).fillna(0.0)
    frame["dividend_rank"] = frame["dv_ttm"].rank(pct=True).fillna(0.0)
    frame["pb_score"] = 1.0 - ((frame["pb"].fillna(4.0) - 0.8) / 4.2).clip(lower=0.0, upper=1.0)
    return frame


def _build_base_event(
    *,
    row: pd.Series,
    strategy: str,
    signal_trade_date: str,
    risk_off_type: str,
    rank: int,
    entry_index: int,
    cached_dates: list[str],
    price_map: dict[str, pd.DataFrame],
) -> dict | None:
    symbol = str(row["ts_code"])
    entry_date = cached_dates[entry_index]
    prices = price_map.get(entry_date)
    if prices is None or symbol not in prices.index:
        return None
    entry_price = float(prices.loc[symbol, "open"])
    if not math.isfinite(entry_price) or entry_price <= 0:
        return None
    return {
        "strategy": strategy,
        "risk_off_type": risk_off_type,
        "signal_trade_date": signal_trade_date,
        "entry_trade_date": entry_date,
        "symbol": symbol,
        "name": str(row.get("name") or symbol),
        "style_group": str(row.get("style_group") or row.get("group") or ""),
        "rank": int(rank),
        "score": float(row.get("risk_off_study_score") or row.get("quality_momentum_score") or 0.0),
        "entry_price": entry_price,
        "return_5d_signal": _safe_float(row.get("return_5d")),
        "return_20d_signal": _safe_float(row.get("return_20d")),
        "volatility_20d": _safe_float(row.get("volatility_20d")),
        "dv_ttm": _safe_float(row.get("dv_ttm")),
        "pe_ttm": _safe_float(row.get("pe_ttm")),
        "pb": _safe_float(row.get("pb")),
        "avg_amount_20d_yuan": _safe_float(row.get("avg_amount_20d_yuan")),
        "total_mv_yuan": _safe_float(row.get("total_mv_yuan")),
    }


def _fill_forward_metrics(
    event: dict,
    horizons: list[int],
    entry_index: int,
    cached_dates: list[str],
    price_map: dict[str, pd.DataFrame],
) -> bool:
    symbol = str(event["symbol"])
    entry_price = float(event["entry_price"])
    for horizon in horizons:
        bars = _load_bars(symbol=symbol, entry_index=entry_index, horizon=horizon, cached_dates=cached_dates, price_map=price_map)
        if bars is None:
            return False
        closes = [bar["close"] for bar in bars]
        highs = [bar["high"] for bar in bars]
        lows = [bar["low"] for bar in bars]
        event[f"close_return_{horizon}d"] = closes[-1] / entry_price - 1.0
        event[f"mfe_{horizon}d"] = max(highs) / entry_price - 1.0
        event[f"mae_{horizon}d"] = min(lows) / entry_price - 1.0
    return True


def _load_bars(
    *,
    symbol: str,
    entry_index: int,
    horizon: int,
    cached_dates: list[str],
    price_map: dict[str, pd.DataFrame],
) -> list[dict] | None:
    bars = []
    for trade_date in cached_dates[entry_index : entry_index + horizon]:
        prices = price_map.get(trade_date)
        if prices is None or symbol not in prices.index:
            return None
        row = prices.loc[symbol]
        bar = {
            "trade_date": trade_date,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        if any(not math.isfinite(value) for value in bar.values() if isinstance(value, float)):
            return None
        bars.append(bar)
    if len(bars) != horizon:
        return None
    return bars


def _simulate_elastic_exit(
    entry_price: float,
    bars: list[dict],
    *,
    trigger_profit_pct: float = 0.05,
    trailing_drawdown_pct: float = 0.03,
    profit_floor_pct: float = 0.01,
    hard_stop_pct: float = 0.06,
    cost_pct: float = 0.0016,
) -> dict:
    peak = float(entry_price)
    exit_day = len(bars)
    exit_price = float(bars[-1]["close"])
    exit_reason = f"time{len(bars)}"
    armed = False
    for day_number, bar in enumerate(bars, start=1):
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        hard_stop = entry_price * (1.0 - hard_stop_pct)
        if low <= hard_stop:
            exit_day = day_number
            exit_price = hard_stop
            exit_reason = f"hard_stop_{hard_stop_pct:.0%}"
            break
        peak = max(peak, high)
        if peak >= entry_price * (1.0 + trigger_profit_pct):
            armed = True
        if armed:
            trailing_stop = peak * (1.0 - trailing_drawdown_pct)
            floor_stop = entry_price * (1.0 + profit_floor_pct)
            stop_price = max(trailing_stop, floor_stop)
            if low <= stop_price:
                exit_day = day_number
                exit_price = stop_price
                exit_reason = "profit_trailing"
                break
        if day_number == len(bars):
            exit_day = day_number
            exit_price = close
            exit_reason = f"time{len(bars)}"
            break

    used_bars = bars[:exit_day]
    exit_return_gross = exit_price / entry_price - 1.0
    return {
        "exit_day": int(exit_day),
        "exit_reason": exit_reason,
        "exit_price": float(exit_price),
        "exit_return_gross": float(exit_return_gross),
        "exit_return_net": float(exit_return_gross - cost_pct),
        "exit_mfe": float(max(bar["high"] for bar in used_bars) / entry_price - 1.0),
        "exit_mae": float(min(bar["low"] for bar in used_bars) / entry_price - 1.0),
    }


def _summary_groups(frame: pd.DataFrame):
    if frame.empty:
        return []
    groups = [((strategy, risk_type), group) for (strategy, risk_type), group in frame.groupby(["strategy", "risk_off_type"])]
    groups.extend([((strategy, "ALL"), group) for strategy, group in frame.groupby("strategy")])
    return groups


def _field_completeness(
    repository: DataRepository,
    trade_date: str | None,
    *,
    required_fields: tuple[str, ...],
    loader: str,
) -> float | None:
    if trade_date is None:
        return None
    try:
        frame = repository.load_daily(trade_date) if loader == "daily" else repository.load_daily_basic(trade_date)
    except FileNotFoundError:
        return 0.0
    if frame.empty:
        return 0.0
    available = sum(1 for field in required_fields if field in frame.columns and frame[field].notna().any())
    return float(available / len(required_fields))


def _path_health_row(source: str, path: Path, rating: str, note: str | None = None) -> dict:
    exists = path.exists()
    rows_or_files = 0
    if exists and path.is_file():
        rows_or_files = 1
    elif exists and path.is_dir():
        rows_or_files = len(list(path.glob("*.csv")))
    return {
        "source": source,
        "status": "available" if exists else "missing",
        "rating": rating if exists else "B",
        "start_date": None,
        "end_date": None,
        "rows_or_files": rows_or_files,
        "missing_dates": None,
        "field_completeness": None,
        "note": note or f"Local cache path: {path}",
    }


def _markdown_table(frame: pd.DataFrame, *, max_rows: int) -> list[str]:
    if frame.empty:
        return ["No rows."]
    shown = frame.head(max_rows).copy()
    columns = list(shown.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in shown.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        values = ["..."] + ["" for _ in columns[1:]]
        values[-1] = f"{len(frame) - max_rows} more rows"
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _resolve_cached_end(cached_dates: list[str], end_date: date | None) -> str:
    if end_date is None:
        return cached_dates[-1]
    requested = to_compact_date(end_date)
    eligible = [value for value in cached_dates if value <= requested]
    if not eligible:
        raise ValueError(f"No cached trade date found on or before {requested}")
    return eligible[-1]


def _resolve_cached_start(cached_dates: list[str], start_date: date | None, end_date: str) -> str:
    if start_date is None:
        end_index = cached_dates.index(end_date)
        return cached_dates[max(1, end_index - 252)]
    requested = to_compact_date(start_date)
    eligible = [value for value in cached_dates if requested <= value <= end_date]
    if not eligible:
        raise ValueError(f"No cached trade date found on or after {requested}")
    return eligible[0]


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result
