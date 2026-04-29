from datetime import date

import pandas as pd

from ashare_signal.config import SelectionConfig
from ashare_signal.domain.models import Position
from ashare_signal.strategy.selector import UniverseSignalSelector


def test_selector_returns_buy_and_sell_candidates() -> None:
    universe = pd.DataFrame(
        [
            {
                "ts_code": "600036.SH",
                "name": "招商银行",
                "close": 42.32,
                "momentum_20d_rank_pct": 0.95,
                "momentum_20d": 0.18,
                "close_to_ma_10": 0.03,
                "close_to_ma_20": 0.08,
                "avg_amount_20d_yuan": 1200000000.0,
                "volatility_20d": 0.02,
                "volume_ratio": 1.1,
                "is_candidate": True,
            },
            {
                "ts_code": "601318.SH",
                "name": "中国平安",
                "close": 45.96,
                "momentum_20d_rank_pct": 0.10,
                "momentum_20d": -0.06,
                "close_to_ma_10": -0.05,
                "close_to_ma_20": -0.08,
                "avg_amount_20d_yuan": 900000000.0,
                "volatility_20d": 0.06,
                "volume_ratio": 0.8,
                "is_candidate": True,
            },
            {
                "ts_code": "000333.SZ",
                "name": "美的集团",
                "close": 67.15,
                "momentum_20d_rank_pct": 0.75,
                "momentum_20d": 0.11,
                "close_to_ma_10": 0.02,
                "close_to_ma_20": 0.05,
                "avg_amount_20d_yuan": 800000000.0,
                "volatility_20d": 0.03,
                "volume_ratio": 1.0,
                "is_candidate": True,
            },
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
    assert len(result.sell_candidates) == 1
    assert result.sell_candidates[0].symbol == "601318.SH"


def test_selector_threshold_helpers() -> None:
    selector = UniverseSignalSelector(
        selection_config=SelectionConfig(min_buy_score=0.7, rotation_edge=0.1)
    )

    buy = selector.select(
        universe=pd.DataFrame(
            [
                {
                    "ts_code": "600036.SH",
                    "name": "招商银行",
                    "close": 42.32,
                    "momentum_20d_rank_pct": 0.95,
                    "momentum_20d": 0.18,
                    "close_to_ma_10": 0.03,
                    "close_to_ma_20": 0.08,
                    "avg_amount_20d_yuan": 1200000000.0,
                    "volatility_20d": 0.02,
                    "volume_ratio": 1.1,
                    "is_candidate": True,
                }
            ]
        ),
        positions=[],
    ).buy_candidates[0]

    sell = selector.select(
        universe=pd.DataFrame(
            [
                {
                    "ts_code": "601318.SH",
                    "name": "中国平安",
                    "close": 45.96,
                    "momentum_20d_rank_pct": 0.10,
                    "momentum_20d": -0.06,
                    "close_to_ma_10": -0.05,
                    "close_to_ma_20": -0.08,
                    "avg_amount_20d_yuan": 900000000.0,
                    "volatility_20d": 0.06,
                    "volume_ratio": 0.8,
                    "is_candidate": True,
                }
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
