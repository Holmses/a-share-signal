import pandas as pd

from ashare_signal.backtest.ranking_event_study import _build_daily_metrics
from ashare_signal.backtest.ranking_event_study import _market_state
from ashare_signal.backtest.ranking_event_study import _summarize_quantiles
from ashare_signal.backtest.ranking_event_study import _summarize_rank_decay
from ashare_signal.backtest.ranking_event_study import _summarize_topk


def _events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": ["quality_momentum_rank"] * 4,
            "signal_trade_date": ["20260407"] * 4,
            "entry_trade_date": ["20260408"] * 4,
            "market_state": ["risk_on"] * 4,
            "market_breadth": [0.60] * 4,
            "market_return_20d": [0.02] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "rank_position": [1, 2, 3, 4],
            "rank_pct": [0.25, 0.50, 0.75, 1.0],
            "rank_quantile": [1, 1, 2, 2],
            "rank_score": [0.90, 0.80, 0.20, 0.10],
            "tradeable_count": [4, 4, 4, 4],
            "close_return_1d": [0.04, 0.02, -0.01, -0.03],
            "close_return_net_1d": [0.0384, 0.0184, -0.0116, -0.0316],
            "mfe_1d": [0.05, 0.03, 0.00, -0.01],
            "mae_1d": [0.00, -0.01, -0.02, -0.04],
            "rank_position_plus_1d": [1, 3, 2, 4],
            "rank_pct_plus_1d": [0.25, 0.75, 0.50, 1.0],
        }
    )


def test_summarize_topk_reports_top_bottom_spread() -> None:
    summary = _summarize_topk(_events_frame(), top_ks=[2], horizons=[1])
    row = summary.loc[(summary["market_state"] == "ALL") & (summary["top_k"] == 2)].iloc[0]

    assert row["events"] == 2
    assert round(row["avg_close_return"], 4) == 0.03
    assert round(row["bottom_k_avg_close_return"], 4) == -0.02
    assert round(row["top_bottom_spread"], 4) == 0.05


def test_quantiles_and_rank_ic_use_full_ranked_cross_section() -> None:
    events = _events_frame()

    quantiles = _summarize_quantiles(events, horizons=[1])
    top_quantile = quantiles.loc[(quantiles["market_state"] == "ALL") & (quantiles["rank_quantile"] == 1)].iloc[0]
    assert round(top_quantile["avg_close_return"], 4) == 0.03

    daily = _build_daily_metrics(events, horizons=[1], top_ks=[2])
    assert daily["rank_ic_spearman"].iloc[0] > 0.9
    assert round(daily["top_2_avg_return"].iloc[0], 4) == 0.03


def test_summarize_rank_decay_measures_topk_retention() -> None:
    summary = _summarize_rank_decay(_events_frame(), top_ks=[2], horizons=[1])
    row = summary.loc[(summary["market_state"] == "ALL") & (summary["top_k"] == 2)].iloc[0]

    assert row["events"] == 2
    assert row["retention_rate"] == 0.5
    assert row["buffer20_retention_rate"] == 1.0
    assert row["avg_future_rank_position"] == 2.0


def test_market_state_splits_risk_on_and_off() -> None:
    risk_on = _market_state(
        pd.DataFrame(
            {
                "close": [11.0, 12.0, 9.0],
                "ma_20": [10.0, 10.0, 10.0],
                "return_20d": [0.03, 0.02, -0.01],
                "benchmark_return_20d": [0.02, 0.02, 0.02],
            }
        ),
        market_min_breadth=0.50,
        market_min_return_20d=0.0,
    )
    risk_off = _market_state(
        pd.DataFrame(
            {
                "close": [9.0, 8.0, 11.0],
                "ma_20": [10.0, 10.0, 10.0],
                "return_20d": [-0.03, -0.02, 0.01],
                "benchmark_return_20d": [-0.02, -0.02, -0.02],
            }
        ),
        market_min_breadth=0.50,
        market_min_return_20d=0.0,
    )

    assert risk_on["market_state"] == "risk_on"
    assert risk_off["market_state"] == "risk_off"
