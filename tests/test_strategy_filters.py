from types import SimpleNamespace

import pandas as pd

from ashare_signal.config import SelectionConfig
from ashare_signal.strategy.filters import ELIGIBLE_REASON, HardTradeFilter, StrategyPreferenceFilter


def _filter_config():
    return SimpleNamespace(
        filters=SimpleNamespace(
            min_list_days=60,
            min_price=3.0,
            min_avg_turnover=50_000_000.0,
            exclude_st=True,
            exclude_suspended=True,
        )
    )


def _preference_row(**overrides):
    row = {
        "ts_code": "600036.SH",
        "close_to_ma_60": 0.08,
        "ma_20_to_ma_60": 0.04,
        "ma_60_slope_20d": 0.03,
        "momentum_20d": 0.12,
        "pullback_from_20d_high": -0.08,
        "close_to_ma_20": 0.03,
        "close_to_ma_5": 0.02,
        "return_1d": 0.02,
        "return_3d": 0.03,
        "low_to_prev_low": 0.01,
        "momentum_5d": 0.03,
        "amount_ratio_5d": 1.20,
        "volume_ratio": 1.10,
        "total_mv_yuan": 60_000_000_000.0,
        "drawdown_20d": -0.12,
        "drawdown_60d": -0.20,
        "down_days_10d": 5,
        "consecutive_down_days": 0,
        "large_net_mf_to_amount": 0.01,
        "is_limit_up": False,
        "is_limit_down": False,
    }
    row.update(overrides)
    return row


def test_hard_trade_filter_marks_tradability_reason() -> None:
    snapshot = pd.DataFrame(
        [
            {
                "ts_code": "600036.SH",
                "exchange": "SSE",
                "market": "主板",
                "is_st": False,
                "is_suspended": False,
                "listed_days": 5000,
                "close": 42.0,
                "avg_amount_20d_yuan": 80_000_000.0,
            },
            {
                "ts_code": "000001.SZ",
                "exchange": "SZSE",
                "market": "主板",
                "is_st": True,
                "is_suspended": False,
                "listed_days": 5000,
                "close": 12.0,
                "avg_amount_20d_yuan": 80_000_000.0,
            },
        ]
    )

    result = HardTradeFilter(_filter_config()).apply(snapshot)

    eligible = result.loc[result["ts_code"] == "600036.SH"].iloc[0]
    rejected = result.loc[result["ts_code"] == "000001.SZ"].iloc[0]
    assert bool(eligible["is_tradeable"]) is True
    assert eligible["hard_filter_reason"] == ELIGIBLE_REASON
    assert bool(rejected["is_tradeable"]) is False
    assert rejected["hard_filter_reason"] == "st_stock"
    assert rejected["exclude_reason"] == "st_stock"


def test_trend_pullback_preference_filter_separates_strategy_reason() -> None:
    frame = pd.DataFrame(
        [
            _preference_row(ts_code="600036.SH"),
            _preference_row(ts_code="000001.SZ", close_to_ma_60=-0.02),
        ]
    )

    result = StrategyPreferenceFilter(SelectionConfig()).apply(frame, "trend_pullback_rank")

    eligible = result.loc[result["ts_code"] == "600036.SH"].iloc[0]
    rejected = result.loc[result["ts_code"] == "000001.SZ"].iloc[0]
    assert bool(eligible["passes_strategy_preference_filter"]) is True
    assert eligible["strategy_preference_reason"] == ELIGIBLE_REASON
    assert bool(rejected["passes_strategy_preference_filter"]) is False
    assert rejected["strategy_preference_reason"] == "trend_structure_invalid"


def test_rebound_bottoming_preference_filter_separates_strategy_reason() -> None:
    frame = pd.DataFrame(
        [
            _preference_row(ts_code="300750.SZ"),
            _preference_row(ts_code="002594.SZ", return_3d=-0.04),
        ]
    )

    result = StrategyPreferenceFilter(SelectionConfig()).apply(frame, "rebound_bottoming_rank")

    eligible = result.loc[result["ts_code"] == "300750.SZ"].iloc[0]
    rejected = result.loc[result["ts_code"] == "002594.SZ"].iloc[0]
    assert bool(eligible["passes_strategy_preference_filter"]) is True
    assert eligible["strategy_preference_reason"] == ELIGIBLE_REASON
    assert bool(rejected["passes_strategy_preference_filter"]) is False
    assert rejected["strategy_preference_reason"] == "rebound_stabilization_missing"
