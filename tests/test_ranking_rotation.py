from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ashare_signal.backtest.ranking_rotation import RankingRotationBacktestEngine
from ashare_signal.backtest.ranking_rotation import RankingRotationPosition


def _config():
    return SimpleNamespace(
        market=SimpleNamespace(max_positions=2),
        backtest=SimpleNamespace(lot_size=100),
        pricing=SimpleNamespace(buy_markup=0.003, sell_markdown=0.003),
    )


def _position(symbol: str, *, entry_index: int = 1, score: float = 0.5, rank: int = 5) -> RankingRotationPosition:
    return RankingRotationPosition(
        symbol=symbol,
        name=symbol,
        shares=100,
        entry_trade_date="20260401",
        signal_trade_date="20260331",
        entry_trade_index=entry_index,
        entry_price=10.0,
        entry_cost=1000.0,
        score=score,
        rank=rank,
    )


def _ranking(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [symbol for symbol, _, _ in rows],
            "rank_position": [rank for _, rank, _ in rows],
            "rank_score": [score for _, _, score in rows],
        }
    )


def test_ranking_rotation_sells_holding_outside_buffer() -> None:
    engine = RankingRotationBacktestEngine(
        config=_config(),
        repository=SimpleNamespace(),
        base_dir=Path("."),
        top_k=2,
        candidate_buffer_k=3,
        drop_n=1,
        min_holding_days=1,
    )
    positions = {
        "AAA": _position("AAA", score=0.80, rank=1),
        "BBB": _position("BBB", score=0.30, rank=4),
    }
    decisions = engine._build_sell_decisions(
        positions=positions,
        ranking=_ranking([("AAA", 1, 0.80), ("CCC", 2, 0.70), ("BBB", 4, 0.30)]),
        trade_index=5,
        risk_off=False,
    )

    assert [(decision.symbol, decision.reason) for decision in decisions] == [("BBB", "rotation_rank_drop")]


def test_ranking_rotation_replaces_weak_holding_when_score_edge_is_large() -> None:
    engine = RankingRotationBacktestEngine(
        config=_config(),
        repository=SimpleNamespace(),
        base_dir=Path("."),
        top_k=2,
        candidate_buffer_k=20,
        drop_n=1,
        min_score_edge=0.05,
        rotation_min_holding_days=3,
    )
    positions = {
        "AAA": _position("AAA", entry_index=1, score=0.72, rank=2),
        "BBB": _position("BBB", entry_index=1, score=0.50, rank=10),
    }
    decisions = engine._build_sell_decisions(
        positions=positions,
        ranking=_ranking([("CCC", 1, 0.60), ("AAA", 2, 0.72), ("BBB", 10, 0.50)]),
        trade_index=6,
        risk_off=False,
    )

    assert [(decision.symbol, decision.reason) for decision in decisions] == [("BBB", "score_edge_rotation")]
