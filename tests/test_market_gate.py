import pandas as pd

from ashare_signal.config import SelectionConfig
from ashare_signal.strategy.gates import MarketGate


def test_market_gate_blocks_trend_when_breadth_is_weak() -> None:
    frame = pd.DataFrame(
        {
            "is_candidate": [True, True, True, True],
            "close_to_ma_20": [0.01, -0.02, -0.03, -0.04],
            "momentum_20d": [0.05, -0.01, -0.02, -0.03],
        }
    )

    result = MarketGate(SelectionConfig(market_min_breadth=0.50)).evaluate(
        frame,
        signal_type="trend_pullback",
    )

    assert result.allowed is False
    assert result.gate == "risk_on"
    assert result.state == "risk_off"
    assert result.reason == "market_breadth_below_threshold"


def test_market_gate_allows_rebound_with_lower_breadth_threshold() -> None:
    frame = pd.DataFrame(
        {
            "is_candidate": [True, True, True, True],
            "close_to_ma_20": [0.01, -0.02, -0.03, -0.04],
            "momentum_20d": [0.05, -0.01, -0.02, -0.03],
        }
    )

    result = MarketGate(
        SelectionConfig(
            market_min_breadth=0.50,
            rebound_market_min_breadth=0.25,
        )
    ).evaluate(frame, signal_type="rebound_bottoming")

    assert result.allowed is True
    assert result.gate == "risk_neutral_or_rebound"
    assert result.state == "risk_neutral"
    assert result.breadth == 0.25
