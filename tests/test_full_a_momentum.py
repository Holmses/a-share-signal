from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_signal.backtest.full_a_momentum import FullAMomentumBacktestEngine
from ashare_signal.backtest.tianzhu9_like import Tianzhu9Position, Tianzhu9Trade


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
    assert state["market_state"] == "risk_off"


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
    assert state["market_state"] == "normal"


def _position() -> Tianzhu9Position:
    return Tianzhu9Position(
        symbol="000001.SZ",
        name="Ping An",
        shares=100,
        entry_trade_date="20250102",
        signal_trade_date="20250101",
        entry_trade_index=1,
        entry_price=10.0,
        entry_cost=1000.0,
        highest_close=10.0,
        score=1.0,
        rank=1,
    )


def test_tianzhu9_trade_diagnostic_fields_keep_legacy_construction() -> None:
    trade = Tianzhu9Trade(
        trade_date="20250102",
        action="BUY",
        symbol="000001.SZ",
        name="Ping An",
        shares=100,
        price=10.0,
        gross_amount=1000.0,
        fees=5.0,
        net_amount=1005.0,
        signal_trade_date="20250101",
        rank=1,
        score=1.0,
    )

    assert trade.pnl is None
    assert trade.entry_recipe is None
    assert trade.exit_reason is None
    assert trade.holding_days is None


def test_default_recipe_candidate_selection_uses_baseline_path(tmp_path, monkeypatch) -> None:
    engine = FullAMomentumBacktestEngine(config=_config(), repository=SimpleNamespace(), base_dir=tmp_path)
    expected = [{"ts_code": "000001.SZ"}]

    def fake_select_candidates(**kwargs):
        return expected

    monkeypatch.setattr(engine, "_select_candidates", fake_select_candidates)

    result = engine._select_candidates_from_recipes(
        signal_frame=pd.DataFrame(),
        style_state={},
        eligible_groups=set(),
        excluded_symbols=set(),
        risk_off=True,
        market_state="risk_off",
    )

    assert result is expected


def test_rebound_bottoming_watch_cannot_be_used_for_production_buys(tmp_path) -> None:
    with pytest.raises(ValueError, match="research-only"):
        FullAMomentumBacktestEngine(
            config=_config(),
            repository=SimpleNamespace(),
            base_dir=tmp_path,
            enabled_recipes=["momentum_core", "rebound_bottoming_watch"],
        )


def test_exit_reason_prefers_trailing_take_profit(tmp_path) -> None:
    engine = FullAMomentumBacktestEngine(config=_config(), repository=SimpleNamespace(), base_dir=tmp_path)
    feature = pd.Series({"close": 11.0, "ma_20": 10.0})

    reason = engine._exit_reason(
        feature=feature,
        position=_position(),
        highest_price=12.0,
        holding_days=5,
        eligible_groups={"main"},
        risk_off=False,
    )

    assert reason == "trailing_take_profit"


def test_exit_reason_records_ma20_break(tmp_path) -> None:
    engine = FullAMomentumBacktestEngine(
        config=_config(),
        repository=SimpleNamespace(),
        base_dir=tmp_path,
        hard_exit_days=None,
        exit_ma20_break=True,
    )
    feature = pd.Series({"close": 9.5, "ma_20": 10.0})

    reason = engine._exit_reason(
        feature=feature,
        position=_position(),
        highest_price=10.2,
        holding_days=3,
        eligible_groups={"main"},
        risk_off=False,
    )

    assert reason == "ma20_break"


def test_exit_reason_records_failure_exit(tmp_path) -> None:
    engine = FullAMomentumBacktestEngine(
        config=_config(),
        repository=SimpleNamespace(),
        base_dir=tmp_path,
        hard_exit_days=None,
        exit_failure_days=3,
        exit_failure_min_peak_profit_pct=0.03,
    )
    feature = pd.Series({"close": 9.5, "ma_20": 10.0, "return_5d": -0.01})

    reason = engine._exit_reason(
        feature=feature,
        position=_position(),
        highest_price=10.1,
        holding_days=3,
        eligible_groups={"main"},
        risk_off=False,
    )

    assert reason == "failure_exit"
