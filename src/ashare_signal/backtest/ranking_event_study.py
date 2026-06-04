from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json

import pandas as pd

from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.ranking import SUPPORTED_RANKING_VARIANTS
from ashare_signal.strategy.ranking import build_ranking_snapshot
from ashare_signal.utils.dates import to_compact_date


@dataclass(slots=True)
class RankingEventStudyResult:
    start_entry_date: str
    end_entry_date: str
    start_signal_date: str
    end_signal_date: str
    variant: str
    top_ks: list[int]
    horizons: list[int]
    event_count: int
    events_path: Path
    quantiles_path: Path
    daily_path: Path
    summary_csv_path: Path
    markdown_path: Path
    summary_path: Path


class RankingEventStudyEngine:
    """Research-only ranking validation with TopK, quantiles, RankIC, and decay."""

    DEFAULT_VARIANT = "quality_momentum_rank"
    DEFAULT_GROUPS = ("main", "chinext", "star")
    DEFAULT_TOP_KS = (5, 10, 20)
    DEFAULT_HORIZONS = (1, 3, 5, 10)
    DEFAULT_QUANTILES = 5

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        variant: str = DEFAULT_VARIANT,
        groups: list[str] | None = None,
        top_ks: list[int] | None = None,
        horizons: list[int] | None = None,
        quantiles: int = DEFAULT_QUANTILES,
        min_avg_amount_yuan: float = 50_000_000.0,
        market_min_breadth: float = 0.50,
        market_min_return_20d: float = 0.0,
    ) -> None:
        if variant not in SUPPORTED_RANKING_VARIANTS:
            raise ValueError(f"Unsupported ranking variant: {variant}")
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.variant = variant
        self.groups = groups or list(self.DEFAULT_GROUPS)
        self.top_ks = sorted({int(value) for value in (top_ks or list(self.DEFAULT_TOP_KS)) if int(value) > 0})
        self.horizons = sorted({int(value) for value in (horizons or list(self.DEFAULT_HORIZONS)) if int(value) > 0})
        self.quantiles = max(int(quantiles), 2)
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.market_min_breadth = float(market_min_breadth)
        self.market_min_return_20d = float(market_min_return_20d)
        if not self.top_ks:
            raise ValueError("At least one positive top-k value is required.")
        if not self.horizons:
            raise ValueError("At least one positive horizon is required.")

    def run(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> RankingEventStudyResult:
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
                "Ranking event study needs at least "
                f"{required_history} complete trade days before entry date {resolved_start}. "
                f"Sync from {suggested_sync_start} or earlier and rerun."
            )

        max_horizon = max(self.horizons)
        last_entry_index = min(end_index, len(cached_dates) - max_horizon)
        if last_entry_index < start_index:
            raise ValueError(
                "Ranking event study has no entry dates with full forward horizon. "
                f"Need {max_horizon} cached trade days after each entry date."
            )

        signal_start_index = start_index - 1
        signal_end_index = last_entry_index - 1
        ranking_end_index = min(len(cached_dates) - 1, signal_end_index + max_horizon)
        feature_dates = cached_dates[
            max(0, signal_start_index - SelectionEventStudyEngine.factor_history_trade_days()) : ranking_end_index + 1
        ]
        price_dates = cached_dates[start_index : last_entry_index + max_horizon]
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=max(self.top_ks),
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=["quality_momentum"],
            horizons=[1],
        )
        factor_frame = study_engine._build_factor_frame(feature_dates)
        price_map = study_engine._load_price_map(price_dates)

        ranking_by_date = self._build_rankings(
            factor_frame=factor_frame,
            signal_dates=cached_dates[signal_start_index : ranking_end_index + 1],
        )
        events_frame = self._build_events(
            cached_dates=cached_dates,
            signal_start_index=signal_start_index,
            signal_end_index=signal_end_index,
            ranking_by_date=ranking_by_date,
            factor_frame=factor_frame,
            price_map=price_map,
        )
        if events_frame.empty:
            raise ValueError("Ranking event study produced no events.")

        quantiles_frame = _summarize_quantiles(events_frame, horizons=self.horizons)
        daily_frame = _build_daily_metrics(events_frame, horizons=self.horizons, top_ks=self.top_ks)
        summary_frame = pd.concat(
            [
                _summarize_topk(events_frame, top_ks=self.top_ks, horizons=self.horizons),
                _summarize_rank_ic(daily_frame),
                _summarize_rank_decay(events_frame, top_ks=self.top_ks, horizons=self.horizons),
            ],
            ignore_index=True,
        )

        reports_dir = self.base_dir / self.config.paths.reports_dir / "ranking-events"
        reports_dir.mkdir(parents=True, exist_ok=True)
        top_slug = "top" + "-".join(str(value) for value in self.top_ks)
        horizon_slug = "h" + "-".join(str(value) for value in self.horizons)
        variant_slug = self.variant.replace("_", "-")
        stem = f"{variant_slug}-{top_slug}-{horizon_slug}-{resolved_start}-{cached_dates[last_entry_index]}"
        events_path = reports_dir / f"{stem}-events.csv"
        quantiles_path = reports_dir / f"{stem}-quantiles.csv"
        daily_path = reports_dir / f"{stem}-daily.csv"
        summary_csv_path = reports_dir / f"{stem}-summary.csv"
        markdown_path = reports_dir / f"{stem}.md"
        summary_path = reports_dir / f"{stem}-summary.json"

        events_frame.to_csv(events_path, index=False)
        quantiles_frame.to_csv(quantiles_path, index=False)
        daily_frame.to_csv(daily_path, index=False)
        summary_frame.to_csv(summary_csv_path, index=False)

        payload = {
            "strategy": "ranking_event_study",
            "variant": self.variant,
            "groups": self.groups,
            "top_ks": self.top_ks,
            "horizons": self.horizons,
            "quantiles": self.quantiles,
            "start_entry_date": resolved_start,
            "end_entry_date": cached_dates[last_entry_index],
            "start_signal_date": cached_dates[signal_start_index],
            "end_signal_date": cached_dates[signal_end_index],
            "requested_end_date": resolved_end,
            "event_count": int(len(events_frame)),
            "events_path": str(events_path),
            "quantiles_path": str(quantiles_path),
            "daily_path": str(daily_path),
            "summary_csv_path": str(summary_csv_path),
            "markdown_path": str(markdown_path),
            "summary": summary_frame.to_dict(orient="records"),
            "quantiles_summary": quantiles_frame.to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(
            _render_markdown(
                payload=payload,
                summary_frame=summary_frame,
                quantiles_frame=quantiles_frame,
            ),
            encoding="utf-8",
        )

        return RankingEventStudyResult(
            start_entry_date=resolved_start,
            end_entry_date=cached_dates[last_entry_index],
            start_signal_date=cached_dates[signal_start_index],
            end_signal_date=cached_dates[signal_end_index],
            variant=self.variant,
            top_ks=self.top_ks,
            horizons=self.horizons,
            event_count=int(len(events_frame)),
            events_path=events_path,
            quantiles_path=quantiles_path,
            daily_path=daily_path,
            summary_csv_path=summary_csv_path,
            markdown_path=markdown_path,
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
            return cached_dates[max(1, end_index - 120)]
        requested = to_compact_date(start_date)
        eligible = [value for value in cached_dates if requested <= value <= end_date]
        if not eligible:
            raise ValueError(f"No cached trade date found on or after {requested}")
        return eligible[0]

    def _build_rankings(
        self,
        *,
        factor_frame: pd.DataFrame,
        signal_dates: list[str],
    ) -> dict[str, pd.DataFrame]:
        ranking_by_date = {}
        for signal_date in signal_dates:
            day_frame = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_date].copy()
            ranking = build_ranking_snapshot(day_frame, self.config, variant=self.variant)
            ranking = ranking.loc[ranking["is_tradeable"].fillna(False).astype(bool)].copy()
            if not ranking.empty:
                ranking["rank_position"] = pd.to_numeric(ranking["rank_position"], errors="coerce")
                ranking["rank_score"] = pd.to_numeric(ranking["rank_score"], errors="coerce")
                ranking = ranking.dropna(subset=["rank_position", "rank_score"])
                ranking["tradeable_count"] = int(len(ranking))
                ranking["rank_pct"] = ranking["rank_position"] / float(len(ranking))
                ranking["rank_quantile"] = (
                    ((ranking["rank_position"].astype(int) - 1) * self.quantiles) // len(ranking) + 1
                ).astype(int)
            ranking_by_date[signal_date] = ranking
        return ranking_by_date

    def _build_events(
        self,
        *,
        cached_dates: list[str],
        signal_start_index: int,
        signal_end_index: int,
        ranking_by_date: dict[str, pd.DataFrame],
        factor_frame: pd.DataFrame,
        price_map: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        frames = []
        cost_pct = float(self.config.backtest.commission_rate) * 2.0 + float(self.config.backtest.stamp_duty_rate)
        for signal_index in range(signal_start_index, signal_end_index + 1):
            signal_date = cached_dates[signal_index]
            entry_index = signal_index + 1
            entry_date = cached_dates[entry_index]
            ranking = ranking_by_date.get(signal_date, pd.DataFrame()).copy()
            if ranking.empty:
                continue
            day_factors = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_date].copy()
            market_state = _market_state(
                day_factors,
                market_min_breadth=self.market_min_breadth,
                market_min_return_20d=self.market_min_return_20d,
            )
            events = ranking[
                [
                    "trade_date",
                    "ts_code",
                    "name",
                    "universe_group",
                    "industry",
                    "rank_position",
                    "rank_pct",
                    "rank_quantile",
                    "rank_score",
                    "tradeable_count",
                    "signal_type",
                ]
            ].copy()
            events = events.rename(columns={"trade_date": "signal_trade_date"})
            events["entry_trade_date"] = entry_date
            events["variant"] = self.variant
            events["market_state"] = market_state["market_state"]
            events["market_breadth"] = market_state["market_breadth"]
            events["market_return_20d"] = market_state["market_return_20d"]

            entry_prices = price_map.get(entry_date)
            if entry_prices is None or entry_prices.empty:
                continue
            events["entry_price"] = events["ts_code"].map(entry_prices["open"])
            events["entry_price"] = pd.to_numeric(events["entry_price"], errors="coerce")
            events = events.loc[events["entry_price"] > 0].copy()
            if events.empty:
                continue

            for horizon in self.horizons:
                self._merge_forward_metrics(
                    events=events,
                    cached_dates=cached_dates,
                    entry_index=entry_index,
                    signal_index=signal_index,
                    horizon=horizon,
                    ranking_by_date=ranking_by_date,
                    price_map=price_map,
                    cost_pct=cost_pct,
                )
            frames.append(events)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _merge_forward_metrics(
        self,
        *,
        events: pd.DataFrame,
        cached_dates: list[str],
        entry_index: int,
        signal_index: int,
        horizon: int,
        ranking_by_date: dict[str, pd.DataFrame],
        price_map: dict[str, pd.DataFrame],
        cost_pct: float,
    ) -> None:
        horizon_dates = cached_dates[entry_index : entry_index + horizon]
        if len(horizon_dates) != horizon:
            return
        close_prices = price_map[horizon_dates[-1]]["close"]
        high_frame = pd.concat([price_map[trade_date]["high"].rename(trade_date) for trade_date in horizon_dates], axis=1)
        low_frame = pd.concat([price_map[trade_date]["low"].rename(trade_date) for trade_date in horizon_dates], axis=1)
        exit_close = events["ts_code"].map(close_prices)
        max_high = events["ts_code"].map(high_frame.max(axis=1))
        min_low = events["ts_code"].map(low_frame.min(axis=1))
        gross_return = pd.to_numeric(exit_close, errors="coerce") / events["entry_price"] - 1.0
        events[f"close_return_{horizon}d"] = gross_return
        events[f"close_return_net_{horizon}d"] = gross_return - cost_pct
        events[f"mfe_{horizon}d"] = pd.to_numeric(max_high, errors="coerce") / events["entry_price"] - 1.0
        events[f"mae_{horizon}d"] = pd.to_numeric(min_low, errors="coerce") / events["entry_price"] - 1.0

        future_signal_index = signal_index + horizon
        if future_signal_index >= len(cached_dates):
            events[f"rank_position_plus_{horizon}d"] = pd.NA
            events[f"rank_pct_plus_{horizon}d"] = pd.NA
            return
        future_ranking = ranking_by_date.get(cached_dates[future_signal_index], pd.DataFrame())
        if future_ranking.empty:
            events[f"rank_position_plus_{horizon}d"] = pd.NA
            events[f"rank_pct_plus_{horizon}d"] = pd.NA
            return
        future_rank = future_ranking.set_index("ts_code")["rank_position"]
        future_pct = future_ranking.set_index("ts_code")["rank_pct"]
        events[f"rank_position_plus_{horizon}d"] = events["ts_code"].map(future_rank)
        events[f"rank_pct_plus_{horizon}d"] = events["ts_code"].map(future_pct)


def _market_state(
    frame: pd.DataFrame,
    *,
    market_min_breadth: float,
    market_min_return_20d: float,
) -> dict:
    if frame.empty:
        return {"market_state": "risk_off", "market_breadth": 0.0, "market_return_20d": None}
    frame = frame.copy()
    for column in ("close", "ma_20", "return_20d", "benchmark_return_20d"):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    market_breadth = float(((frame["close"] >= frame["ma_20"]) & (frame["return_20d"] >= 0.0)).mean())
    benchmark_return = frame["benchmark_return_20d"].dropna()
    if benchmark_return.empty:
        market_return = frame["return_20d"].dropna().median()
    else:
        market_return = benchmark_return.iloc[-1]
    market_return_value = None if pd.isna(market_return) else float(market_return)
    risk_off = market_breadth < market_min_breadth or (
        market_return_value is not None and market_return_value < market_min_return_20d
    )
    return {
        "market_state": "risk_off" if risk_off else "risk_on",
        "market_breadth": market_breadth,
        "market_return_20d": market_return_value,
    }


def _summarize_topk(events_frame: pd.DataFrame, *, top_ks: list[int], horizons: list[int]) -> pd.DataFrame:
    rows = []
    segments = _segment_groups(events_frame)
    for segment_name, frame in segments:
        for top_k in top_ks:
            top_mask = pd.to_numeric(frame["rank_position"], errors="coerce") <= top_k
            bottom_mask = pd.to_numeric(frame["rank_position"], errors="coerce") > (
                pd.to_numeric(frame["tradeable_count"], errors="coerce") - top_k
            )
            for horizon in horizons:
                column = f"close_return_{horizon}d"
                net_column = f"close_return_net_{horizon}d"
                returns = pd.to_numeric(frame.loc[top_mask, column], errors="coerce").dropna()
                net_returns = pd.to_numeric(frame.loc[top_mask, net_column], errors="coerce").dropna()
                bottom_returns = pd.to_numeric(frame.loc[bottom_mask, column], errors="coerce").dropna()
                if returns.empty:
                    continue
                bottom_avg = float(bottom_returns.mean()) if not bottom_returns.empty else None
                rows.append(
                    {
                        "metric_type": "topk_forward_return",
                        "market_state": segment_name,
                        "top_k": int(top_k),
                        "horizon": int(horizon),
                        "events": int(len(returns)),
                        "avg_close_return": float(returns.mean()),
                        "avg_close_return_net": float(net_returns.mean()),
                        "median_close_return": float(returns.median()),
                        "win_rate": float((returns > 0).mean()),
                        "bottom_k_avg_close_return": bottom_avg,
                        "top_bottom_spread": float(returns.mean() - bottom_avg) if bottom_avg is not None else None,
                        "avg_mfe": float(pd.to_numeric(frame.loc[top_mask, f"mfe_{horizon}d"], errors="coerce").mean()),
                        "avg_mae": float(pd.to_numeric(frame.loc[top_mask, f"mae_{horizon}d"], errors="coerce").mean()),
                    }
                )
    return pd.DataFrame(rows)


def _summarize_quantiles(events_frame: pd.DataFrame, *, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for segment_name, segment in _segment_groups(events_frame):
        for (variant, quantile), frame in segment.groupby(["variant", "rank_quantile"]):
            for horizon in horizons:
                column = f"close_return_{horizon}d"
                net_column = f"close_return_net_{horizon}d"
                returns = pd.to_numeric(frame[column], errors="coerce").dropna()
                net_returns = pd.to_numeric(frame[net_column], errors="coerce").dropna()
                if returns.empty:
                    continue
                rows.append(
                    {
                        "variant": variant,
                        "market_state": segment_name,
                        "rank_quantile": int(quantile),
                        "horizon": int(horizon),
                        "events": int(len(returns)),
                        "avg_close_return": float(returns.mean()),
                        "avg_close_return_net": float(net_returns.mean()),
                        "median_close_return": float(returns.median()),
                        "win_rate": float((returns > 0).mean()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["horizon", "market_state", "rank_quantile"]).reset_index(drop=True)


def _build_daily_metrics(events_frame: pd.DataFrame, *, horizons: list[int], top_ks: list[int]) -> pd.DataFrame:
    rows = []
    for (signal_date, entry_date, market_state), frame in events_frame.groupby(
        ["signal_trade_date", "entry_trade_date", "market_state"]
    ):
        base = {
            "signal_trade_date": signal_date,
            "entry_trade_date": entry_date,
            "market_state": market_state,
            "tradeable_count": int(frame["tradeable_count"].max()),
            "market_breadth": float(pd.to_numeric(frame["market_breadth"], errors="coerce").dropna().iloc[0]),
            "market_return_20d": _first_numeric_or_none(frame["market_return_20d"]),
        }
        for horizon in horizons:
            row = {**base, "horizon": int(horizon)}
            returns = pd.to_numeric(frame[f"close_return_{horizon}d"], errors="coerce")
            scores = pd.to_numeric(frame["rank_score"], errors="coerce")
            row["rank_ic_spearman"] = _safe_corr(scores, returns, method="spearman")
            row["rank_ic_pearson"] = _safe_corr(scores, returns, method="pearson")
            for top_k in top_ks:
                mask = pd.to_numeric(frame["rank_position"], errors="coerce") <= top_k
                top_returns = pd.to_numeric(frame.loc[mask, f"close_return_{horizon}d"], errors="coerce").dropna()
                row[f"top_{top_k}_avg_return"] = float(top_returns.mean()) if not top_returns.empty else None
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_rank_ic(daily_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if daily_frame.empty:
        return pd.DataFrame(rows)
    for segment_name, frame in _segment_groups(daily_frame):
        for horizon, horizon_frame in frame.groupby("horizon"):
            rank_ic = pd.to_numeric(horizon_frame["rank_ic_spearman"], errors="coerce").dropna()
            pearson = pd.to_numeric(horizon_frame["rank_ic_pearson"], errors="coerce").dropna()
            if rank_ic.empty:
                continue
            rows.append(
                {
                    "metric_type": "rank_ic",
                    "market_state": segment_name,
                    "top_k": None,
                    "horizon": int(horizon),
                    "events": int(len(rank_ic)),
                    "avg_rank_ic_spearman": float(rank_ic.mean()),
                    "median_rank_ic_spearman": float(rank_ic.median()),
                    "positive_rank_ic_rate": float((rank_ic > 0).mean()),
                    "avg_rank_ic_pearson": float(pearson.mean()) if not pearson.empty else None,
                }
            )
    return pd.DataFrame(rows)


def _summarize_rank_decay(events_frame: pd.DataFrame, *, top_ks: list[int], horizons: list[int]) -> pd.DataFrame:
    rows = []
    for segment_name, frame in _segment_groups(events_frame):
        for top_k in top_ks:
            top_mask = pd.to_numeric(frame["rank_position"], errors="coerce") <= top_k
            for horizon in horizons:
                future_rank = pd.to_numeric(frame.loc[top_mask, f"rank_position_plus_{horizon}d"], errors="coerce")
                future_rank = future_rank.dropna()
                if future_rank.empty:
                    continue
                rows.append(
                    {
                        "metric_type": "rank_decay",
                        "market_state": segment_name,
                        "top_k": int(top_k),
                        "horizon": int(horizon),
                        "events": int(len(future_rank)),
                        "retention_rate": float((future_rank <= top_k).mean()),
                        "buffer20_retention_rate": float((future_rank <= 20).mean()),
                        "avg_future_rank_position": float(future_rank.mean()),
                        "median_future_rank_position": float(future_rank.median()),
                    }
                )
    return pd.DataFrame(rows)


def _segment_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    segments = [("ALL", frame)]
    if "market_state" in frame.columns:
        segments.extend([(str(name), part) for name, part in frame.groupby("market_state")])
    return segments


def _safe_corr(left: pd.Series, right: pd.Series, *, method: str) -> float | None:
    paired = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(paired) < 3 or paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
        return None
    if method == "spearman":
        left_values = paired["left"].rank(method="average")
        right_values = paired["right"].rank(method="average")
    else:
        left_values = paired["left"]
        right_values = paired["right"]
    value = left_values.corr(right_values)
    if pd.isna(value):
        return None
    return float(value)


def _first_numeric_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _render_markdown(
    *,
    payload: dict,
    summary_frame: pd.DataFrame,
    quantiles_frame: pd.DataFrame,
) -> str:
    lines = [
        f"# Ranking Event Study: {payload['variant']}",
        "",
        "This report is research-only. It does not affect production signals, paper-trading state, or scheduler state.",
        "",
        "## Run Metadata",
        "",
        f"- Entry dates: {payload['start_entry_date']} to {payload['end_entry_date']}",
        f"- Signal dates: {payload['start_signal_date']} to {payload['end_signal_date']}",
        f"- TopK: {', '.join(str(value) for value in payload['top_ks'])}",
        f"- Horizons: {', '.join(str(value) for value in payload['horizons'])} trade days",
        f"- Events: {payload['event_count']}",
        "",
        "## TopK Forward Returns",
        "",
    ]
    lines.extend(
        _markdown_table(
            summary_frame.loc[
                (summary_frame["metric_type"] == "topk_forward_return") & (summary_frame["market_state"] == "ALL")
            ],
            ["top_k", "horizon", "events", "avg_close_return_net", "win_rate", "top_bottom_spread"],
        )
    )
    lines.extend(["", "## RankIC", ""])
    lines.extend(
        _markdown_table(
            summary_frame.loc[(summary_frame["metric_type"] == "rank_ic") & (summary_frame["market_state"] == "ALL")],
            ["horizon", "events", "avg_rank_ic_spearman", "positive_rank_ic_rate", "avg_rank_ic_pearson"],
        )
    )
    lines.extend(["", "## Quantile Returns", ""])
    lines.extend(
        _markdown_table(
            quantiles_frame.loc[quantiles_frame["market_state"] == "ALL"],
            ["rank_quantile", "horizon", "events", "avg_close_return_net", "win_rate"],
        )
    )
    lines.extend(["", "## Rank Decay", ""])
    lines.extend(
        _markdown_table(
            summary_frame.loc[(summary_frame["metric_type"] == "rank_decay") & (summary_frame["market_state"] == "ALL")],
            ["top_k", "horizon", "events", "retention_rate", "buffer20_retention_rate", "avg_future_rank_position"],
        )
    )
    lines.append("")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 80) -> list[str]:
    if frame.empty:
        return ["No rows."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.head(limit).iterrows():
        lines.append("| " + " | ".join(_format_markdown_value(row.get(column)) for column in columns) + " |")
    return lines


def _format_markdown_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "/")
