import pandas as pd

from ashare_signal.backtest.risk_off_standalone import _classify_risk_off_type
from ashare_signal.backtest.risk_off_standalone import _select_defensive_candidates
from ashare_signal.backtest.risk_off_standalone import _simulate_elastic_exit


def test_classify_risk_off_type_splits_market_states() -> None:
    assert _classify_risk_off_type(0.20, 0.02) == "severe"
    assert _classify_risk_off_type(0.60, -0.06) == "severe"
    assert _classify_risk_off_type(0.40, -0.02) == "both_mild"
    assert _classify_risk_off_type(0.40, 0.02) == "breadth_only"
    assert _classify_risk_off_type(0.60, -0.02) == "return_only"


def test_select_defensive_candidates_requires_dividend_low_vol_and_valuation() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "dv_ttm": [5.0, 0.8, 4.0],
            "pe_ttm": [8.0, 10.0, 50.0],
            "pb": [0.8, 1.0, 1.0],
            "return_20d": [0.02, 0.01, 0.02],
            "close_to_ma_20": [0.01, 0.01, 0.01],
            "volatility_20d": [0.01, 0.012, 0.01],
            "avg_amount_20d_yuan": [100_000_000, 100_000_000, 100_000_000],
            "financial_quality_score": [0.6, 0.6, 0.6],
            "market_cap_rank": [0.8, 0.8, 0.8],
        }
    )

    selected = _select_defensive_candidates(frame, top_n=5, min_avg_amount_yuan=80_000_000)

    assert selected["ts_code"].tolist() == ["000001.SZ"]
    assert selected["risk_off_study_score"].iloc[0] > 0


def test_simulate_elastic_exit_trails_after_profit_trigger() -> None:
    result = _simulate_elastic_exit(
        100.0,
        [
            {"high": 106.0, "low": 103.0, "close": 105.0},
            {"high": 107.0, "low": 101.0, "close": 102.0},
        ],
        cost_pct=0.0,
    )

    assert result["exit_reason"] == "profit_trailing"
    assert result["exit_day"] == 2
    assert round(result["exit_return_net"], 4) == round(107.0 * 0.97 / 100.0 - 1.0, 4)


def test_simulate_elastic_exit_hard_stops_before_profit_tracking() -> None:
    result = _simulate_elastic_exit(
        100.0,
        [
            {"high": 108.0, "low": 93.0, "close": 100.0},
        ],
        cost_pct=0.0,
    )

    assert result["exit_reason"] == "hard_stop_6%"
    assert result["exit_day"] == 1
    assert round(result["exit_return_net"], 4) == -0.06
