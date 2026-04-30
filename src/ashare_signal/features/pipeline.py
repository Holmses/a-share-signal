from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.universe import apply_universe_filters
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


@dataclass(slots=True)
class UniverseBuildResult:
    trade_date: str
    output_path: Path
    total_symbols: int
    candidate_symbols: int


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["amount_yuan"] = frame["amount"] * 1000.0
    return frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def compute_feature_snapshot(
    history: pd.DataFrame,
    daily_basic: pd.DataFrame,
    stock_basic: pd.DataFrame,
    as_of_trade_date: str,
    config: AppConfig,
) -> pd.DataFrame:
    history = _normalize_history(history)
    grouped = history.groupby("ts_code", group_keys=False)

    history["return_1d"] = grouped["close"].pct_change()
    grouped = history.groupby("ts_code", group_keys=False)
    history["momentum_5d"] = grouped["close"].pct_change(
        periods=config.strategy.lookback_short_days
    )
    history["momentum_20d"] = grouped["close"].pct_change(
        periods=config.strategy.lookback_momentum_days
    )
    history["volatility_20d"] = grouped["return_1d"].transform(
        lambda series: series.rolling(
            window=config.strategy.lookback_vol_days,
            min_periods=config.strategy.lookback_vol_days,
        ).std()
    )
    history["avg_amount_20d_yuan"] = grouped["amount_yuan"].transform(
        lambda series: series.rolling(window=20, min_periods=20).mean()
    )
    history["avg_amount_5d_yuan"] = grouped["amount_yuan"].transform(
        lambda series: series.rolling(window=5, min_periods=5).mean()
    )
    history["prev_low"] = grouped["low"].shift(1)
    history["ma_10"] = grouped["close"].transform(
        lambda series: series.rolling(window=10, min_periods=10).mean()
    )
    history["ma_5"] = grouped["close"].transform(
        lambda series: series.rolling(window=5, min_periods=5).mean()
    )
    history["ma_20"] = grouped["close"].transform(
        lambda series: series.rolling(window=20, min_periods=20).mean()
    )
    history["ma_60"] = grouped["close"].transform(
        lambda series: series.rolling(window=60, min_periods=60).mean()
    )
    history["ma_60_lag_20"] = grouped["ma_60"].shift(20)
    history["high_20d"] = grouped["high"].transform(
        lambda series: series.rolling(window=20, min_periods=20).max()
    )
    history["close_to_ma_10"] = history["close"] / history["ma_10"] - 1.0
    history["close_to_ma_5"] = history["close"] / history["ma_5"] - 1.0
    history["close_to_ma_20"] = history["close"] / history["ma_20"] - 1.0
    history["close_to_ma_60"] = history["close"] / history["ma_60"] - 1.0
    history["ma_20_to_ma_60"] = history["ma_20"] / history["ma_60"] - 1.0
    history["ma_60_slope_20d"] = history["ma_60"] / history["ma_60_lag_20"] - 1.0
    history["pullback_from_20d_high"] = history["close"] / history["high_20d"] - 1.0
    history["low_to_prev_low"] = history["low"] / history["prev_low"] - 1.0
    history["amount_ratio_5d"] = history["amount_yuan"] / history["avg_amount_5d_yuan"]

    as_of_timestamp = pd.Timestamp(parse_compact_date(to_compact_date(as_of_trade_date)))
    latest_features = history.loc[history["trade_date"] == as_of_timestamp].copy()
    latest_features["momentum_20d_rank_pct"] = latest_features["momentum_20d"].rank(
        pct=True
    )
    latest_features["volatility_20d_rank_pct"] = latest_features["volatility_20d"].rank(
        pct=True,
        ascending=False,
    )

    basic = stock_basic.copy()
    basic["list_date"] = pd.to_datetime(
        basic["list_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    daily_basic_frame = daily_basic.copy()
    daily_basic_frame["trade_date"] = pd.to_datetime(
        daily_basic_frame["trade_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    numeric_daily_basic_columns = [
        "close",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ]
    for column in numeric_daily_basic_columns:
        daily_basic_frame[column] = pd.to_numeric(
            daily_basic_frame[column],
            errors="coerce",
        )

    daily_basic_frame["total_mv_yuan"] = daily_basic_frame["total_mv"] * 10000.0
    daily_basic_frame["circ_mv_yuan"] = daily_basic_frame["circ_mv"] * 10000.0
    daily_basic_frame = daily_basic_frame.rename(columns={"close": "daily_basic_close"})

    snapshot = basic.merge(
        latest_features[
            [
                "ts_code",
                "trade_date",
                "close",
                "pct_chg",
                "amount_yuan",
                "return_1d",
                "momentum_5d",
                "momentum_20d",
                "momentum_20d_rank_pct",
                "volatility_20d",
                "volatility_20d_rank_pct",
                "avg_amount_5d_yuan",
                "avg_amount_20d_yuan",
                "ma_5",
                "ma_10",
                "ma_20",
                "ma_60",
                "close_to_ma_5",
                "close_to_ma_10",
                "close_to_ma_20",
                "close_to_ma_60",
                "ma_20_to_ma_60",
                "ma_60_slope_20d",
                "pullback_from_20d_high",
                "low_to_prev_low",
                "amount_ratio_5d",
            ]
        ],
        on="ts_code",
        how="left",
    ).merge(
        daily_basic_frame[
            [
                "ts_code",
                "trade_date",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "ps",
                "ps_ttm",
                "dv_ratio",
                "dv_ttm",
                "total_share",
                "float_share",
                "free_share",
                "total_mv",
                "circ_mv",
                "total_mv_yuan",
                "circ_mv_yuan",
            ]
        ],
        on=["ts_code", "trade_date"],
        how="left",
    )

    snapshot["trade_date"] = snapshot["trade_date"].fillna(as_of_timestamp)
    snapshot["listed_days"] = (snapshot["trade_date"] - snapshot["list_date"]).dt.days
    snapshot["is_st"] = snapshot["name"].fillna("").str.upper().str.contains("ST")
    snapshot["is_suspended"] = snapshot["close"].isna()

    return snapshot


class UniverseBuilder:
    def __init__(self, config: AppConfig, repository: DataRepository) -> None:
        self.config = config
        self.repository = repository

    def build(self, as_of: date) -> UniverseBuildResult:
        requested_trade_date = to_compact_date(as_of)
        actual_trade_date = self.repository.resolve_trade_date(requested_trade_date)
        universe = build_universe_snapshot(
            config=self.config,
            repository=self.repository,
            trade_date=actual_trade_date,
        )
        output_path = self.repository.save_universe_snapshot(actual_trade_date, universe)

        return UniverseBuildResult(
            trade_date=actual_trade_date,
            output_path=output_path,
            total_symbols=len(universe),
            candidate_symbols=int(universe["is_candidate"].sum()),
        )


def build_universe_snapshot(
    config: AppConfig,
    repository: DataRepository,
    trade_date: str,
) -> pd.DataFrame:
    actual_trade_date = repository.resolve_trade_date(trade_date)
    lookback = max(
        config.strategy.lookback_momentum_days,
        config.strategy.lookback_short_days,
        config.strategy.lookback_vol_days,
        80,
    )
    trade_dates = repository.recent_open_trade_dates(
        actual_trade_date,
        count=lookback + 1,
    )
    daily_history = repository.load_daily_for_dates(trade_dates)
    daily_basic = repository.load_daily_basic(actual_trade_date)
    stock_basic = repository.load_stock_basic(list_status="L")

    snapshot = compute_feature_snapshot(
        history=daily_history,
        daily_basic=daily_basic,
        stock_basic=stock_basic,
        as_of_trade_date=actual_trade_date,
        config=config,
    )
    return apply_universe_filters(snapshot, config)
