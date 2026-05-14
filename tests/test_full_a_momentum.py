from types import SimpleNamespace

import pandas as pd

from ashare_signal.backtest.full_a_momentum import FullAMomentumBacktestEngine


def _config():
    return SimpleNamespace(
        market=SimpleNamespace(max_positions=5),
        backtest=SimpleNamespace(lot_size=100),
        pricing=SimpleNamespace(buy_markup=0.003, sell_markdown=0.003),
    )


def test_market_style_state_blocks_new_buys_when_market_is_weak(tmp_path) -> None:
    engine = FullAMomentumBacktestEngine(
        config=_config(),
        repository=SimpleNamespace(),
        base_dir=tmp_path,
        market_min_breadth=0.50,
        market_min_return_20d=0.0,
    )
    frame = pd.DataFrame(
        {
            "group": ["main", "main", "chinext", "star"],
            "close": [9.0, 8.0, 12.0, 11.0],
            "ma_20": [10.0, 10.0, 10.0, 10.0],
            "return_20d": [-0.04, -0.03, 0.02, 0.01],
            "return_5d": [-0.02, -0.01, 0.01, 0.02],
        }
    )

    state = engine._market_style_state(frame)

    assert state["market_risk_off"] is True
    assert state["eligible_groups"] == []


def test_market_style_state_keeps_only_strong_board_groups(tmp_path) -> None:
    engine = FullAMomentumBacktestEngine(
        config=_config(),
        repository=SimpleNamespace(),
        base_dir=tmp_path,
        market_min_breadth=0.50,
        market_min_return_20d=0.0,
        style_min_breadth=0.60,
        style_min_return_20d=0.0,
    )
    frame = pd.DataFrame(
        {
            "group": ["main", "main", "chinext", "chinext", "star", "star"],
            "close": [11.0, 12.0, 9.0, 8.0, 12.0, 9.0],
            "ma_20": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "return_20d": [0.05, 0.04, -0.02, -0.03, 0.03, 0.01],
            "return_5d": [0.02, 0.01, -0.01, -0.02, 0.02, 0.00],
        }
    )

    state = engine._market_style_state(frame)

    assert state["market_risk_off"] is False
    assert state["eligible_groups"] == ["main"]
