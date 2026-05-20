from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


@dataclass(slots=True)
class SelectionEventStudyResult:
    start_entry_date: str
    end_entry_date: str
    start_signal_date: str
    end_signal_date: str
    variants: list[str]
    groups: list[str]
    horizons: list[int]
    event_count: int
    summary_path: Path
    events_path: Path
    daily_path: Path


class SelectionEventStudyEngine:
    """Evaluate whether selected symbols rise after the signal day.

    The study uses T-1 factors, enters at the next trade day's open, and
    measures forward close return, maximum favorable excursion, and maximum
    adverse excursion over fixed horizons. It is intentionally independent from
    the live order generator so experimental selection variants do not affect
    production plans.
    """

    RETURN_LOOKBACK_TRADE_DAYS = 90
    FACTOR_HISTORY_TRADE_DAYS = 100
    SYNC_WARMUP_CALENDAR_DAYS = 180
    DEFAULT_GROUPS = ("main", "chinext", "star")
    DEFAULT_VARIANTS = ("legacy", "quality", "quality_momentum", "quality_strict")
    DEFAULT_HORIZONS = (1, 3, 5, 10)
    SW_INDUSTRY_SRC = "SW2021"

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        top_n_per_group: int = 5,
        min_avg_amount_yuan: float = 50_000_000.0,
        groups: list[str] | None = None,
        variants: list[str] | None = None,
        horizons: list[int] | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.top_n_per_group = max(int(top_n_per_group), 1)
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.groups = _dedupe_csv_values(groups or list(self.DEFAULT_GROUPS))
        self.variants = _dedupe_csv_values(variants or list(self.DEFAULT_VARIANTS))
        self.horizons = sorted({int(value) for value in (horizons or list(self.DEFAULT_HORIZONS)) if int(value) > 0})
        if not self.horizons:
            raise ValueError("At least one positive event horizon is required.")
        unknown_groups = set(self.groups) - {"main", "chinext", "star", "bse"}
        if unknown_groups:
            raise ValueError(f"Unsupported groups: {sorted(unknown_groups)}")
        unknown_variants = set(self.variants) - set(self.DEFAULT_VARIANTS)
        if unknown_variants:
            raise ValueError(f"Unsupported variants: {sorted(unknown_variants)}")

    def run(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> SelectionEventStudyResult:
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
                "Selection event study needs at least "
                f"{required_history} complete trade days before entry date {resolved_start} "
                f"for factor warm-up, but only found {start_index}. "
                f"Current cache starts at {cached_start}. "
                f"Sync from {suggested_sync_start} or earlier and rerun."
            )

        max_horizon = max(self.horizons)
        last_entry_index = min(end_index, len(cached_dates) - max_horizon)
        if last_entry_index < start_index:
            raise ValueError(
                "Selection event study has no entry dates with full forward horizon. "
                f"Need {max_horizon} cached trade days after each entry date."
            )

        signal_start_index = start_index - 1
        signal_end_index = last_entry_index - 1
        feature_dates = cached_dates[
            max(0, signal_start_index - self.factor_history_trade_days()) : last_entry_index + max_horizon
        ]
        price_dates = cached_dates[start_index : last_entry_index + max_horizon]
        factor_frame = self._build_factor_frame(feature_dates)
        price_map = self._load_price_map(price_dates)

        events: list[dict] = []
        daily_rows: list[dict] = []
        for signal_index in range(signal_start_index, signal_end_index + 1):
            signal_trade_date = cached_dates[signal_index]
            entry_index = signal_index + 1
            entry_date = cached_dates[entry_index]
            day_frame = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_trade_date]
            if day_frame.empty:
                continue
            for variant in self.variants:
                selected = self._select(day_frame, variant)
                daily_rows.append(
                    {
                        "signal_trade_date": signal_trade_date,
                        "entry_trade_date": entry_date,
                        "variant": variant,
                        "selected_count": len(selected),
                    }
                )
                for row in selected:
                    event = self._build_event(
                        row=row,
                        variant=variant,
                        signal_trade_date=signal_trade_date,
                        entry_index=entry_index,
                        cached_dates=cached_dates,
                        price_map=price_map,
                    )
                    if event is not None:
                        events.append(event)

        events_frame = pd.DataFrame(events)
        if events_frame.empty:
            raise ValueError("Selection event study produced no events.")
        daily_frame = pd.DataFrame(daily_rows)
        summary_frame = self._summarize(events_frame)

        reports_dir = self.base_dir / self.config.paths.reports_dir / "selection-events"
        reports_dir.mkdir(parents=True, exist_ok=True)
        variants_slug = "-".join(self.variants)
        groups_slug = "-".join(self.groups)
        horizon_slug = "h" + "-".join(str(value) for value in self.horizons)
        stem = (
            f"selection-events-top{self.top_n_per_group}-{groups_slug}-{variants_slug}-"
            f"{horizon_slug}-{resolved_start}-{cached_dates[last_entry_index]}"
        )
        events_path = reports_dir / f"{stem}-events.csv"
        daily_path = reports_dir / f"{stem}-daily.csv"
        summary_csv_path = reports_dir / f"{stem}-summary.csv"
        summary_path = reports_dir / f"{stem}-summary.json"

        events_frame.to_csv(events_path, index=False)
        daily_frame.to_csv(daily_path, index=False)
        summary_frame.to_csv(summary_csv_path, index=False)

        payload = {
            "strategy": "selection_event_study",
            "start_entry_date": resolved_start,
            "end_entry_date": cached_dates[last_entry_index],
            "start_signal_date": cached_dates[signal_start_index],
            "end_signal_date": cached_dates[signal_end_index],
            "requested_end_date": resolved_end,
            "top_n_per_group": self.top_n_per_group,
            "groups": self.groups,
            "variants": self.variants,
            "horizons": self.horizons,
            "event_count": int(len(events_frame)),
            "min_avg_amount_yuan": self.min_avg_amount_yuan,
            "summary_csv_path": str(summary_csv_path),
            "events_path": str(events_path),
            "daily_path": str(daily_path),
            "summary": summary_frame.to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return SelectionEventStudyResult(
            start_entry_date=resolved_start,
            end_entry_date=cached_dates[last_entry_index],
            start_signal_date=cached_dates[signal_start_index],
            end_signal_date=cached_dates[signal_end_index],
            variants=self.variants,
            groups=self.groups,
            horizons=self.horizons,
            event_count=int(len(events_frame)),
            summary_path=summary_path,
            events_path=events_path,
            daily_path=daily_path,
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
            return cached_dates[max(1, end_index - 252)]
        requested = to_compact_date(start_date)
        eligible = [value for value in cached_dates if value >= requested and value <= end_date]
        if not eligible:
            raise ValueError(f"No cached trade date found on or after {requested}")
        return eligible[0]

    def _build_factor_frame(self, trade_dates: list[str]) -> pd.DataFrame:
        daily = self.repository.load_daily_for_dates(trade_dates).copy()
        for column in ("open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"):
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
        daily["amount_yuan"] = daily["amount"] * 1000.0
        daily["pct_chg_decimal"] = daily["pct_chg"] / 100.0
        daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        grouped = daily.groupby("ts_code", group_keys=False)
        daily["return_5d"] = grouped["close"].pct_change(periods=5)
        daily["return_10d"] = grouped["close"].pct_change(periods=10)
        daily["return_20d"] = grouped["close"].pct_change(periods=20)
        daily["return_30d"] = grouped["close"].pct_change(periods=30)
        daily["return_60d"] = grouped["close"].pct_change(periods=60)
        daily["return_90d"] = grouped["close"].pct_change(periods=90)
        daily["ma_5"] = grouped["close"].transform(lambda series: series.rolling(window=5, min_periods=5).mean())
        daily["ma_10"] = grouped["close"].transform(lambda series: series.rolling(window=10, min_periods=10).mean())
        daily["ma_20"] = grouped["close"].transform(lambda series: series.rolling(window=20, min_periods=20).mean())
        daily["ma_60"] = grouped["close"].transform(lambda series: series.rolling(window=60, min_periods=60).mean())
        daily["high_20d"] = grouped["high"].transform(lambda series: series.rolling(window=20, min_periods=20).max())
        daily["avg_amount_20d_yuan"] = grouped["amount_yuan"].transform(
            lambda series: series.rolling(window=20, min_periods=20).mean()
        )
        daily["avg_amount_5d_yuan"] = grouped["amount_yuan"].transform(
            lambda series: series.rolling(window=5, min_periods=5).mean()
        )
        daily["volatility_20d"] = grouped["pct_chg_decimal"].transform(
            lambda series: series.rolling(window=20, min_periods=20).std()
        )
        prev_close = grouped["close"].shift(1)
        true_range = pd.concat(
            [
                daily["high"] - daily["low"],
                (daily["high"] - prev_close).abs(),
                (daily["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        daily["atr_20d"] = true_range.groupby(daily["ts_code"]).transform(
            lambda series: series.rolling(window=20, min_periods=20).mean()
        )
        daily["atr_20d_pct"] = daily["atr_20d"] / daily["close"]
        daily["amount_ratio_5d"] = daily["amount_yuan"] / daily["avg_amount_5d_yuan"]
        daily["close_to_ma_5"] = daily["close"] / daily["ma_5"] - 1.0
        daily["close_to_ma_10"] = daily["close"] / daily["ma_10"] - 1.0
        daily["close_to_ma_20"] = daily["close"] / daily["ma_20"] - 1.0
        daily["close_to_ma_60"] = daily["close"] / daily["ma_60"] - 1.0
        daily["drawdown_from_20d_high"] = daily["close"] / daily["high_20d"] - 1.0
        candle_range = daily["high"] - daily["low"]
        upper_shadow = daily["high"] - daily[["open", "close"]].max(axis=1)
        daily["upper_shadow_pct"] = (upper_shadow / candle_range.where(candle_range > 0)).clip(lower=0.0)

        daily_basic = pd.concat(
            [self.repository.load_daily_basic(trade_date) for trade_date in trade_dates],
            ignore_index=True,
        ).copy()
        for column in ("turnover_rate", "volume_ratio", "total_mv", "circ_mv"):
            daily_basic[column] = pd.to_numeric(daily_basic[column], errors="coerce")
        daily_basic["total_mv_yuan"] = daily_basic["total_mv"] * 10000.0

        stock_basic = self.repository.load_stock_basic(list_status="L")
        stock_basic = stock_basic[
            ["ts_code", "name", "market", "exchange", "industry", "list_date"]
        ].copy()
        stock_basic["list_date"] = pd.to_datetime(stock_basic["list_date"], format="%Y%m%d", errors="coerce")
        stock_basic["group"] = stock_basic.apply(_classify_board, axis=1)
        stock_basic = self._merge_sw_industry(stock_basic)

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
        frame = self._merge_index_market_features(frame, trade_dates)
        frame = self._merge_financial_features(frame, trade_dates)
        frame["trade_timestamp"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="coerce")
        frame["listed_days"] = (frame["trade_timestamp"] - frame["list_date"]).dt.days
        frame["is_st"] = frame["name"].fillna("").str.upper().str.contains("ST")
        frame["style_group"] = frame["sw_l1_name"].fillna(frame["industry"]).fillna(frame["group"])

        base_mask = (
            frame["group"].isin(self.groups)
            & (~frame["is_st"])
            & (frame["listed_days"].fillna(-1) >= self.config.filters.min_list_days)
            & (frame["close"].fillna(0.0) >= self.config.filters.min_price)
            & (frame["avg_amount_20d_yuan"].fillna(0.0) >= self.min_avg_amount_yuan)
            & frame["return_30d"].notna()
            & frame["return_90d"].notna()
            & frame["return_5d"].notna()
            & frame["open"].notna()
            & frame["close"].notna()
            & frame["ma_5"].notna()
            & frame["ma_10"].notna()
            & frame["ma_20"].notna()
        )
        frame = frame.loc[base_mask].copy()
        if frame.empty:
            return frame

        by_date_style = frame.groupby(["trade_date", "style_group"], group_keys=False)
        frame["style_return_5d_median"] = by_date_style["return_5d"].transform("median")
        frame["style_return_20d_median"] = by_date_style["return_20d"].transform("median")
        frame["_above_ma20"] = (frame["close"] >= frame["ma_20"]).astype(float)
        frame["style_breadth_20d"] = by_date_style["_above_ma20"].transform("mean")
        frame = frame.drop(columns=["_above_ma20"])
        frame["relative_style_return_5d"] = frame["return_5d"] - frame["style_return_5d_median"]
        frame["relative_style_return_20d"] = frame["return_20d"] - frame["style_return_20d_median"]

        by_date_group = frame.groupby(["trade_date", "group"], group_keys=False)
        frame["return_30d_rank"] = by_date_group["return_30d"].rank(pct=True)
        frame["return_90d_rank"] = by_date_group["return_90d"].rank(pct=True)
        frame["turnover_rank"] = by_date_group["turnover_rate"].rank(pct=True)
        frame["amount_rank"] = by_date_group["avg_amount_20d_yuan"].rank(pct=True)
        frame["market_cap_rank"] = by_date_group["total_mv_yuan"].rank(pct=True)
        frame["volume_ratio_score"] = (
            1.0 - ((frame["volume_ratio"].fillna(1.0) - 1.0).abs() / 3.0)
        ).clip(lower=0.0, upper=1.0)
        frame["stability_score"] = (
            1.0 - ((frame["return_5d"].fillna(0.0).abs() - 0.02) / 0.18)
        ).clip(lower=0.0, upper=1.0)
        frame["strict_stability_score"] = (
            1.0 - ((frame["return_5d"].fillna(0.0).abs() - 0.015) / 0.105)
        ).clip(lower=0.0, upper=1.0)
        frame["trend_quality_score"] = (
            (frame["close"] >= frame["ma_5"]).astype(float) * 0.30
            + (frame["ma_5"] >= frame["ma_10"]).astype(float) * 0.30
            + (frame["ma_10"] >= frame["ma_20"]).astype(float) * 0.25
            + (frame["ma_20"] >= frame["ma_60"]).astype(float) * 0.15
        )
        frame["overheat_score"] = (
            1.0 - ((frame["close_to_ma_20"].fillna(0.0) - 0.08) / 0.27)
        ).clip(lower=0.0, upper=1.0)
        frame["near_high_score"] = (
            1.0 - ((frame["drawdown_from_20d_high"].fillna(-0.20).abs()) / 0.20)
        ).clip(lower=0.0, upper=1.0)
        frame["legacy_score"] = (
            frame["return_30d_rank"].fillna(0.0) * 0.40
            + frame["return_90d_rank"].fillna(0.0) * 0.25
            + frame["amount_rank"].fillna(0.0) * 0.15
            + frame["turnover_rank"].fillna(0.0) * 0.10
            + frame["volume_ratio_score"].fillna(0.0) * 0.05
            + frame["stability_score"].fillna(0.0) * 0.05
        )
        frame["quality_score"] = (
            frame["return_30d_rank"].fillna(0.0) * 0.28
            + frame["return_90d_rank"].fillna(0.0) * 0.18
            + frame["amount_rank"].fillna(0.0) * 0.15
            + frame["turnover_rank"].fillna(0.0) * 0.10
            + frame["volume_ratio_score"].fillna(0.0) * 0.05
            + frame["stability_score"].fillna(0.0) * 0.14
            + frame["trend_quality_score"].fillna(0.0) * 0.05
            + frame["overheat_score"].fillna(0.0) * 0.05
        )
        frame["quality_momentum_score"] = (
            frame["return_30d_rank"].fillna(0.0) * 0.30
            + frame["return_90d_rank"].fillna(0.0) * 0.15
            + frame["amount_rank"].fillna(0.0) * 0.12
            + frame["turnover_rank"].fillna(0.0) * 0.08
            + frame["volume_ratio_score"].fillna(0.0) * 0.04
            + frame["stability_score"].fillna(0.0) * 0.10
            + frame["trend_quality_score"].fillna(0.0) * 0.08
            + frame["overheat_score"].fillna(0.0) * 0.04
            + frame["near_high_score"].fillna(0.0) * 0.05
            + frame["financial_quality_score"].fillna(0.5) * 0.04
        )
        frame["quality_strict_score"] = (
            frame["return_30d_rank"].fillna(0.0) * 0.22
            + frame["return_90d_rank"].fillna(0.0) * 0.14
            + frame["amount_rank"].fillna(0.0) * 0.14
            + frame["turnover_rank"].fillna(0.0) * 0.08
            + frame["volume_ratio_score"].fillna(0.0) * 0.06
            + frame["strict_stability_score"].fillna(0.0) * 0.16
            + frame["trend_quality_score"].fillna(0.0) * 0.10
            + frame["overheat_score"].fillna(0.0) * 0.06
            + frame["near_high_score"].fillna(0.0) * 0.04
        )
        return frame.sort_values(["trade_date", "group", "quality_score"], ascending=[True, True, False])

    def _merge_sw_industry(self, stock_basic: pd.DataFrame) -> pd.DataFrame:
        stock_basic = stock_basic.copy()
        try:
            members = self.repository.load_index_member_all(src=self.SW_INDUSTRY_SRC).copy()
        except (AttributeError, FileNotFoundError):
            members = pd.DataFrame()
        if members.empty:
            stock_basic["sw_l1_code"] = pd.NA
            stock_basic["sw_l1_name"] = pd.NA
            return stock_basic
        if "ts_code" not in members.columns and "con_code" in members.columns:
            members = members.rename(columns={"con_code": "ts_code"})
        if "ts_code" not in members.columns:
            stock_basic["sw_l1_code"] = pd.NA
            stock_basic["sw_l1_name"] = pd.NA
            return stock_basic
        if "is_new" in members.columns:
            is_new = members["is_new"].astype(str).str.upper()
            members = members.loc[is_new.isin(["Y", "1", "TRUE"])]
        members = members.drop_duplicates("ts_code", keep="last")
        if "l1_name" not in members.columns and "index_code" in members.columns:
            try:
                classify = self.repository.load_index_classify(src=self.SW_INDUSTRY_SRC)
            except (AttributeError, FileNotFoundError):
                classify = pd.DataFrame()
            if not classify.empty and {"index_code", "industry_name"}.issubset(classify.columns):
                members = members.merge(classify[["index_code", "industry_name"]], on="index_code", how="left")
                members = members.rename(columns={"industry_name": "l1_name"})
        columns = ["ts_code"]
        rename_map = {}
        if "l1_code" in members.columns:
            columns.append("l1_code")
            rename_map["l1_code"] = "sw_l1_code"
        if "l1_name" in members.columns:
            columns.append("l1_name")
            rename_map["l1_name"] = "sw_l1_name"
        if len(columns) == 1:
            stock_basic["sw_l1_code"] = pd.NA
            stock_basic["sw_l1_name"] = pd.NA
            return stock_basic
        merged = stock_basic.merge(members[columns].rename(columns=rename_map), on="ts_code", how="left")
        if "sw_l1_code" not in merged.columns:
            merged["sw_l1_code"] = pd.NA
        if "sw_l1_name" not in merged.columns:
            merged["sw_l1_name"] = pd.NA
        return merged

    def _merge_index_market_features(self, frame: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
        frame = frame.copy()
        benchmark = self.config.market.benchmark
        try:
            index_daily = self.repository.load_index_daily(benchmark).copy()
        except (AttributeError, FileNotFoundError):
            index_daily = pd.DataFrame()
        if index_daily.empty:
            frame["benchmark_return_20d"] = pd.NA
            frame["benchmark_close_to_ma20"] = pd.NA
            frame["benchmark_pe_ttm"] = pd.NA
            frame["benchmark_pb"] = pd.NA
            return frame
        for column in ("close", "pct_chg", "amount"):
            if column in index_daily.columns:
                index_daily[column] = pd.to_numeric(index_daily[column], errors="coerce")
        index_daily = index_daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        grouped = index_daily.groupby("ts_code", group_keys=False)
        index_daily["benchmark_return_20d"] = grouped["close"].pct_change(periods=20)
        index_daily["benchmark_ma20"] = grouped["close"].transform(
            lambda series: series.rolling(window=20, min_periods=20).mean()
        )
        index_daily["benchmark_close_to_ma20"] = index_daily["close"] / index_daily["benchmark_ma20"] - 1.0
        market = index_daily.loc[
            index_daily["ts_code"] == benchmark,
            ["trade_date", "benchmark_return_20d", "benchmark_close_to_ma20"],
        ].copy()
        try:
            daily_basic = self.repository.load_index_daily_basic_for_dates(trade_dates)
        except AttributeError:
            daily_basic = pd.DataFrame()
        if not daily_basic.empty and "ts_code" in daily_basic.columns:
            daily_basic = daily_basic.loc[daily_basic["ts_code"] == benchmark].copy()
            for column in ("pe_ttm", "pb"):
                if column in daily_basic.columns:
                    daily_basic[column] = pd.to_numeric(daily_basic[column], errors="coerce")
            columns = ["trade_date"]
            rename_map = {}
            if "pe_ttm" in daily_basic.columns:
                columns.append("pe_ttm")
                rename_map["pe_ttm"] = "benchmark_pe_ttm"
            if "pb" in daily_basic.columns:
                columns.append("pb")
                rename_map["pb"] = "benchmark_pb"
            if len(columns) > 1:
                market = market.merge(daily_basic[columns].rename(columns=rename_map), on="trade_date", how="left")
        return frame.merge(market, on="trade_date", how="left")

    def _merge_financial_features(self, frame: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
        frame = frame.copy()
        financial_columns = _financial_columns()
        for column in financial_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        try:
            fina = self.repository.load_fina_indicator_between(end_date=max(trade_dates)).copy()
        except (AttributeError, FileNotFoundError):
            fina = pd.DataFrame()
        if fina.empty or "ts_code" not in fina.columns or "ann_date" not in fina.columns:
            return _add_financial_quality_score(frame)
        fina["ann_timestamp"] = pd.to_datetime(fina["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
        fina = fina.dropna(subset=["ann_timestamp"])
        for column in financial_columns:
            if column in ("ann_date", "end_date"):
                continue
            if column in fina.columns:
                fina[column] = pd.to_numeric(fina[column], errors="coerce")
        frame["trade_timestamp"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        frame["_row_order"] = range(len(frame))
        fina_columns = ["ann_timestamp"] + [column for column in financial_columns if column in fina.columns]
        merged_parts = []
        for symbol, symbol_frame in frame.groupby("ts_code", sort=False):
            symbol_fina = fina.loc[fina["ts_code"] == symbol, fina_columns].sort_values("ann_timestamp")
            if symbol_fina.empty:
                merged_parts.append(symbol_frame)
                continue
            merged_parts.append(
                pd.merge_asof(
                    symbol_frame.sort_values("trade_timestamp"),
                    symbol_fina,
                    left_on="trade_timestamp",
                    right_on="ann_timestamp",
                    direction="backward",
                    suffixes=("", "_fina"),
                )
            )
        merged = pd.concat(merged_parts, ignore_index=True).sort_values("_row_order")
        merged = merged.drop(columns=["_row_order", "ann_timestamp"], errors="ignore")
        for column in financial_columns:
            fina_column = f"{column}_fina"
            if fina_column in merged.columns:
                merged[column] = merged[fina_column].combine_first(merged[column])
                merged = merged.drop(columns=[fina_column])
        return _add_financial_quality_score(merged)

    def _load_price_map(self, trade_dates: list[str]) -> dict[str, pd.DataFrame]:
        price_map = {}
        for trade_date in trade_dates:
            frame = self.repository.load_daily(trade_date).copy()
            for column in ("open", "high", "low", "close"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            price_map[trade_date] = frame.set_index("ts_code")
        return price_map

    def _select(self, day_frame: pd.DataFrame, variant: str) -> list[dict]:
        if day_frame.empty:
            return []
        frame = day_frame.loc[self._variant_mask(day_frame, variant)].copy()
        if frame.empty:
            return []
        score_column = f"{variant}_score"
        selected_frames = []
        for _, group_frame in frame.groupby("group", group_keys=False):
            selected_frames.append(
                group_frame.sort_values([score_column, "avg_amount_20d_yuan"], ascending=[False, False]).head(
                    self.top_n_per_group
                )
            )
        if not selected_frames:
            return []
        selected = pd.concat(selected_frames, ignore_index=True)
        selected = selected.sort_values(["group", score_column, "avg_amount_20d_yuan"], ascending=[True, False, False])
        rows = []
        for group, group_frame in selected.groupby("group", sort=False):
            for rank, row in enumerate(group_frame.to_dict(orient="records"), start=1):
                row["rank"] = rank
                row["variant_score"] = float(row[score_column])
                rows.append(row)
        return rows

    def _variant_mask(self, frame: pd.DataFrame, variant: str) -> pd.Series:
        if variant == "legacy":
            return pd.Series(True, index=frame.index)
        if variant == "quality":
            return (
                (frame["return_30d"] <= 1.20)
                & (frame["return_90d"] <= 3.00)
                & (frame["return_5d"] >= -0.08)
                & (frame["return_5d"] <= 0.12)
                & (frame["close_to_ma_5"] >= -0.03)
                & (frame["close_to_ma_10"] >= -0.05)
                & (frame["close_to_ma_10"] <= 0.18)
                & (frame["close_to_ma_20"] <= 0.35)
                & (frame["upper_shadow_pct"].fillna(0.0) <= 0.55)
            )
        if variant == "quality_momentum":
            return (
                (frame["return_30d"] >= 0.08)
                & (frame["return_30d"] <= 1.10)
                & (frame["return_90d"] >= 0.10)
                & (frame["return_90d"] <= 3.00)
                & (frame["return_5d"] >= -0.06)
                & (frame["return_5d"] <= 0.11)
                & (frame["close_to_ma_5"] >= -0.025)
                & (frame["close_to_ma_10"] >= -0.04)
                & (frame["close_to_ma_10"] <= 0.16)
                & (frame["close_to_ma_20"] >= -0.03)
                & (frame["close_to_ma_20"] <= 0.30)
                & (frame["drawdown_from_20d_high"] >= -0.16)
                & (frame["upper_shadow_pct"].fillna(0.0) <= 0.50)
                & (frame["volume_ratio"].fillna(1.0) <= 3.00)
                & (frame["amount_ratio_5d"].fillna(1.0) >= 0.75)
                & (frame["amount_ratio_5d"].fillna(1.0) <= 2.60)
            )
        return (
            (frame["return_30d"] >= 0.05)
            & (frame["return_30d"] <= 0.90)
            & (frame["return_90d"] >= 0.08)
            & (frame["return_90d"] <= 2.50)
            & (frame["return_5d"] >= -0.04)
            & (frame["return_5d"] <= 0.09)
            & (frame["close_to_ma_5"] >= -0.015)
            & (frame["close_to_ma_5"] <= 0.08)
            & (frame["close_to_ma_10"] >= -0.03)
            & (frame["close_to_ma_10"] <= 0.12)
            & (frame["close_to_ma_20"] >= -0.02)
            & (frame["close_to_ma_20"] <= 0.22)
            & (frame["drawdown_from_20d_high"] >= -0.18)
            & (frame["upper_shadow_pct"].fillna(0.0) <= 0.45)
            & (frame["volume_ratio"].fillna(1.0) <= 2.50)
            & (frame["amount_ratio_5d"].fillna(1.0) >= 0.80)
            & (frame["amount_ratio_5d"].fillna(1.0) <= 2.20)
            & (frame["total_mv_yuan"].fillna(0.0) >= 3_000_000_000.0)
        )

    def _build_event(
        self,
        row: dict,
        variant: str,
        signal_trade_date: str,
        entry_index: int,
        cached_dates: list[str],
        price_map: dict[str, pd.DataFrame],
    ) -> dict | None:
        symbol = str(row["ts_code"])
        entry_date = cached_dates[entry_index]
        entry_prices = price_map.get(entry_date)
        if entry_prices is None or symbol not in entry_prices.index:
            return None
        entry_price = float(entry_prices.loc[symbol, "open"])
        if math.isnan(entry_price) or entry_price <= 0:
            return None

        event = {
            "variant": variant,
            "group": str(row["group"]),
            "signal_trade_date": signal_trade_date,
            "entry_trade_date": entry_date,
            "symbol": symbol,
            "name": str(row.get("name") or symbol),
            "industry": str(row.get("industry") or ""),
            "rank": int(row["rank"]),
            "score": float(row["variant_score"]),
            "entry_price": entry_price,
            "return_5d_signal": float(row["return_5d"]),
            "return_30d_signal": float(row["return_30d"]),
            "return_90d_signal": float(row["return_90d"]),
            "close_to_ma_20_signal": float(row["close_to_ma_20"]),
            "drawdown_from_20d_high_signal": float(row["drawdown_from_20d_high"]),
            "avg_amount_20d_yuan": float(row["avg_amount_20d_yuan"]),
            "total_mv_yuan": float(row["total_mv_yuan"]) if not pd.isna(row["total_mv_yuan"]) else None,
        }
        for horizon in self.horizons:
            horizon_dates = cached_dates[entry_index : entry_index + horizon]
            bars = []
            for trade_date in horizon_dates:
                prices = price_map.get(trade_date)
                if prices is None or symbol not in prices.index:
                    continue
                bars.append(prices.loc[symbol])
            if len(bars) != horizon:
                return None
            closes = [float(bar["close"]) for bar in bars]
            highs = [float(bar["high"]) for bar in bars]
            lows = [float(bar["low"]) for bar in bars]
            if any(math.isnan(value) for value in closes + highs + lows):
                return None
            event[f"close_return_{horizon}d"] = closes[-1] / entry_price - 1.0
            event[f"mfe_{horizon}d"] = max(highs) / entry_price - 1.0
            event[f"mae_{horizon}d"] = min(lows) / entry_price - 1.0
        return event

    def _summarize(self, events_frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        groups = [(keys, frame) for keys, frame in events_frame.groupby(["variant", "group"])]
        groups.extend(
            [((variant, "ALL"), frame) for variant, frame in events_frame.groupby("variant")]
        )
        for (variant, group), frame in groups:
            for horizon in self.horizons:
                close_returns = frame[f"close_return_{horizon}d"]
                mfe = frame[f"mfe_{horizon}d"]
                mae = frame[f"mae_{horizon}d"]
                rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "horizon": horizon,
                        "events": int(len(frame)),
                        "close_win_rate": float((close_returns > 0).mean()),
                        "avg_close_return": float(close_returns.mean()),
                        "median_close_return": float(close_returns.median()),
                        "mfe_gt_2pct_rate": float((mfe >= 0.02).mean()),
                        "mfe_gt_5pct_rate": float((mfe >= 0.05).mean()),
                        "mfe_gt_8pct_rate": float((mfe >= 0.08).mean()),
                        "avg_mfe": float(mfe.mean()),
                        "median_mfe": float(mfe.median()),
                        "mae_lt_minus_5pct_rate": float((mae <= -0.05).mean()),
                        "mae_lt_minus_8pct_rate": float((mae <= -0.08).mean()),
                        "avg_mae": float(mae.mean()),
                        "median_mae": float(mae.median()),
                    }
                )
        return pd.DataFrame(rows).sort_values(["horizon", "variant", "group"]).reset_index(drop=True)


def _classify_board(row: pd.Series) -> str:
    market = str(row.get("market") or "")
    exchange = str(row.get("exchange") or "")
    ts_code = str(row.get("ts_code") or "")
    if market == "创业板" or ts_code.startswith(("300", "301")):
        return "chinext"
    if market == "科创板" or ts_code.startswith("688"):
        return "star"
    if market == "北交所" or exchange == "BSE":
        return "bse"
    return "main"


def _financial_columns() -> list[str]:
    return [
        "ann_date",
        "end_date",
        "roe",
        "roe_waa",
        "roe_dt",
        "roa",
        "roic",
        "grossprofit_margin",
        "netprofit_margin",
        "debt_to_assets",
        "ocf_to_or",
        "ocf_to_profit",
        "basic_eps_yoy",
        "dt_eps_yoy",
        "op_yoy",
        "netprofit_yoy",
        "dt_netprofit_yoy",
        "ocf_yoy",
    ]


def _add_financial_quality_score(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    roe = frame["roe_waa"].combine_first(frame["roe"]).combine_first(frame["roe_dt"])
    growth = frame["dt_netprofit_yoy"].combine_first(frame["netprofit_yoy"]).combine_first(frame["op_yoy"])
    operating_cashflow = frame["ocf_to_or"].combine_first(frame["ocf_to_profit"])
    score_parts = pd.DataFrame(
        {
            "roe_score": ((roe - 3.0) / 12.0).clip(lower=0.0, upper=1.0),
            "gross_margin_score": (frame["grossprofit_margin"] / 35.0).clip(lower=0.0, upper=1.0),
            "debt_score": (1.0 - ((frame["debt_to_assets"] - 35.0) / 45.0)).clip(lower=0.0, upper=1.0),
            "growth_score": ((growth + 20.0) / 80.0).clip(lower=0.0, upper=1.0),
            "cashflow_score": ((operating_cashflow + 5.0) / 25.0).clip(lower=0.0, upper=1.0),
        },
        index=frame.index,
    )
    weights = pd.Series(
        {
            "roe_score": 0.30,
            "gross_margin_score": 0.20,
            "debt_score": 0.20,
            "growth_score": 0.20,
            "cashflow_score": 0.10,
        }
    )
    weighted = score_parts.mul(weights, axis=1)
    available_weight = score_parts.notna().mul(weights, axis=1).sum(axis=1)
    frame["financial_data_available"] = available_weight > 0
    frame["financial_quality_score"] = (weighted.sum(axis=1) / available_weight.where(available_weight > 0)).fillna(0.5)
    return frame


def _dedupe_csv_values(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = str(value).strip().replace("-", "_")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def parse_csv_values(value: str | None, defaults: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(defaults)
    return _dedupe_csv_values(value.split(","))


def parse_horizons(value: str | None, defaults: tuple[int, ...]) -> list[int]:
    if value is None:
        return list(defaults)
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be a comma-separated list of positive integers.")
    return horizons
