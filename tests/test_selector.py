from datetime import date

import pandas as pd

from ashare_signal.config import SelectionConfig
from ashare_signal.domain.models import Position
from ashare_signal.strategy.selector import UniverseSignalSelector


def _trend_row(
    symbol: str,
    name: str,
    *,
    close: float,
    momentum_rank: float,
    momentum: float,
    ma10: float,
    ma20: float,
    amount: float,
    volatility: float,
    volume_ratio: float,
) -> dict:
    return {
        "ts_code": symbol,
        "name": name,
        "close": close,
        "return_1d": 0.02,
        "return_3d": 0.03,
        "return_10d": 0.04,
        "momentum_20d_rank_pct": momentum_rank,
        "momentum_5d": 0.03,
        "momentum_20d": momentum,
        "close_to_ma_5": 0.02,
        "close_to_ma_10": ma10,
        "close_to_ma_20": ma20,
        "close_to_ma_60": 0.08,
        "ma_20_to_ma_60": 0.04,
        "ma_60_slope_20d": 0.03,
        "pullback_from_20d_high": -0.08,
        "drawdown_20d": -0.08,
        "drawdown_60d": -0.12,
        "low_to_prev_low": 0.01,
        "amount_ratio_5d": 1.10,
        "down_days_10d": 4,
        "consecutive_down_days": 0,
        "avg_amount_20d_yuan": amount,
        "volatility_20d": volatility,
        "volume_capitulation_score": 0.50,
        "volume_ratio": volume_ratio,
        "total_mv_yuan": 60000000000.0,
        "is_candidate": True,
    }


def test_selector_returns_buy_and_sell_candidates() -> None:
    universe = pd.DataFrame(
        [
            _trend_row(
                "600036.SH",
                "招商银行",
                close=42.32,
                momentum_rank=0.95,
                momentum=0.18,
                ma10=0.03,
                ma20=0.08,
                amount=1200000000.0,
                volatility=0.02,
                volume_ratio=1.1,
            ),
            _trend_row(
                "601318.SH",
                "中国平安",
                close=45.96,
                momentum_rank=0.10,
                momentum=-0.06,
                ma10=-0.05,
                ma20=-0.08,
                amount=900000000.0,
                volatility=0.06,
                volume_ratio=0.8,
            ),
            _trend_row(
                "000333.SZ",
                "美的集团",
                close=67.15,
                momentum_rank=0.75,
                momentum=0.11,
                ma10=0.02,
                ma20=0.05,
                amount=800000000.0,
                volatility=0.03,
                volume_ratio=1.0,
            ),
        ]
    )
    positions = [
        Position(
            symbol="601318.SH",
            name="中国平安",
            entry_date=date(2026, 3, 25),
            entry_price=46.80,
            quantity=500,
        )
    ]

    selector = UniverseSignalSelector(selection_config=SelectionConfig())
    result = selector.select(universe=universe, positions=positions)

    assert len(result.buy_candidates) == 1
    assert result.buy_candidates[0].symbol == "600036.SH"
    assert result.buy_candidates[0].signal_type == "trend_pullback"
    assert len(result.sell_candidates) == 1
    assert result.sell_candidates[0].symbol == "601318.SH"


def test_selector_threshold_helpers() -> None:
    selector = UniverseSignalSelector(
        selection_config=SelectionConfig(min_buy_score=0.7, rotation_edge=0.1)
    )

    buy = selector.select(
        universe=pd.DataFrame(
            [
                _trend_row(
                    "600036.SH",
                    "招商银行",
                    close=42.32,
                    momentum_rank=0.95,
                    momentum=0.18,
                    ma10=0.03,
                    ma20=0.08,
                    amount=1200000000.0,
                    volatility=0.02,
                    volume_ratio=1.1,
                )
            ]
        ),
        positions=[],
    ).buy_candidates[0]

    sell = selector.select(
        universe=pd.DataFrame(
            [
                _trend_row(
                    "601318.SH",
                    "中国平安",
                    close=45.96,
                    momentum_rank=0.10,
                    momentum=-0.06,
                    ma10=-0.05,
                    ma20=-0.08,
                    amount=900000000.0,
                    volatility=0.06,
                    volume_ratio=0.8,
                )
            ]
        ),
        positions=[
            Position(
                symbol="601318.SH",
                name="中国平安",
                entry_date=date(2026, 3, 25),
                entry_price=46.80,
                quantity=500,
            )
        ],
    ).sell_candidates[0]

    assert selector.should_open_new_position(buy) is True
    assert selector.should_rotate(buy, sell) is True


def test_selector_can_pick_bottoming_rebound_candidate() -> None:
    rebound = _trend_row(
        "300750.SZ",
        "宁德时代",
        close=190.0,
        momentum_rank=0.25,
        momentum=-0.12,
        ma10=-0.03,
        ma20=-0.06,
        amount=1500000000.0,
        volatility=0.04,
        volume_ratio=1.2,
    )
    rebound.update(
        {
            "return_1d": 0.018,
            "return_3d": 0.03,
            "return_10d": -0.04,
            "close_to_ma_5": 0.01,
            "close_to_ma_60": -0.12,
            "ma_20_to_ma_60": -0.08,
            "ma_60_slope_20d": -0.02,
            "pullback_from_20d_high": -0.12,
            "drawdown_20d": -0.12,
            "drawdown_60d": -0.20,
            "low_to_prev_low": 0.01,
            "amount_ratio_5d": 1.30,
            "down_days_10d": 5,
            "consecutive_down_days": 0,
            "volume_capitulation_score": 0.75,
        }
    )

    selector = UniverseSignalSelector(selection_config=SelectionConfig(rebound_min_score=0.50))
    result = selector.select(universe=pd.DataFrame([rebound]), positions=[])

    assert len(result.buy_candidates) == 1
    assert result.buy_candidates[0].symbol == "300750.SZ"
    assert result.buy_candidates[0].signal_type == "rebound_bottoming"
    assert selector.should_open_new_position(result.buy_candidates[0]) is True


def test_selector_rejects_unstable_falling_rebound_candidate() -> None:
    falling = _trend_row(
        "300750.SZ",
        "宁德时代",
        close=170.0,
        momentum_rank=0.10,
        momentum=-0.25,
        ma10=-0.09,
        ma20=-0.14,
        amount=1500000000.0,
        volatility=0.07,
        volume_ratio=1.5,
    )
    falling.update(
        {
            "return_1d": -0.04,
            "return_3d": -0.08,
            "close_to_ma_5": -0.06,
            "close_to_ma_60": -0.25,
            "ma_20_to_ma_60": -0.12,
            "ma_60_slope_20d": -0.05,
            "drawdown_20d": -0.18,
            "drawdown_60d": -0.42,
            "low_to_prev_low": -0.06,
            "amount_ratio_5d": 2.80,
            "down_days_10d": 9,
            "consecutive_down_days": 6,
        }
    )

    selector = UniverseSignalSelector(selection_config=SelectionConfig(rebound_min_score=0.50))
    result = selector.select(universe=pd.DataFrame([falling]), positions=[])

    assert result.buy_candidates == []
