from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_signal.strategy.ranking import build_ranking_snapshot, render_ranking_factor_map


def _config():
    return SimpleNamespace(
        selection=SimpleNamespace(
            buy_min_pullback_from_20d_high=-0.15,
            buy_max_pullback_from_20d_high=-0.05,
            rebound_min_drawdown_20d=-0.08,
            rebound_max_drawdown_60d=-0.35,
            rebound_min_return_3d=-0.01,
        )
    )


def _universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260407", "20260407", "20260407"],
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "name": ["Alpha", "Beta", "Gamma"],
            "market": ["主板", "主板", "创业板"],
            "exchange": ["SZSE", "SZSE", "SZSE"],
            "industry": ["Bank", "Bank", "Tech"],
            "is_candidate": [True, False, True],
            "exclude_reason": ["eligible", "st_stock", "eligible"],
            "momentum_20d_rank_pct": [0.95, 0.20, 0.45],
            "momentum_20d": [0.18, -0.05, 0.04],
            "ma_20_to_ma_60": [0.06, -0.01, 0.01],
            "ma_60_slope_20d": [0.03, -0.02, 0.01],
            "close_to_ma_60": [0.08, -0.05, -0.02],
            "pullback_from_20d_high": [-0.10, -0.12, -0.22],
            "close_to_ma_20": [0.01, -0.02, -0.06],
            "avg_amount_20d_yuan": [150_000_000, 90_000_000, 80_000_000],
            "volatility_20d": [0.02, 0.03, 0.04],
            "large_net_mf_to_amount": [0.02, -0.01, 0.00],
            "net_mf_to_amount": [0.01, -0.01, 0.00],
            "industry_momentum_20d_median": [0.04, 0.04, 0.01],
            "industry_breadth_20d": [0.70, 0.70, 0.40],
            "industry_return_3d_median": [0.02, 0.02, 0.00],
            "industry_rebound_breadth": [0.60, 0.60, 0.55],
            "drawdown_20d": [-0.10, -0.12, -0.24],
            "drawdown_60d": [-0.14, -0.18, -0.30],
            "return_3d": [0.03, -0.01, 0.04],
            "low_to_prev_low": [0.02, -0.03, 0.03],
        }
    )


def test_build_ranking_snapshot_keeps_filter_reasons_and_ranks_tradeable_pool() -> None:
    ranking = build_ranking_snapshot(_universe_frame(), _config())

    assert ranking["ts_code"].tolist()[:2] == ["000001.SZ", "000003.SZ"]
    assert ranking.loc[ranking["ts_code"] == "000003.SZ", "universe_group"].iloc[0] == "chinext"
    assert ranking.loc[ranking["ts_code"] == "000002.SZ", "filter_reason"].iloc[0] == "st_stock"
    assert pd.isna(ranking.loc[ranking["ts_code"] == "000002.SZ", "rank_position"].iloc[0])
    assert set(ranking.loc[ranking["is_tradeable"], "rank_position"].astype(int)) == {1, 2}
    assert ranking.loc[ranking["ts_code"] == "000001.SZ", "rank_score"].iloc[0] > 0
    assert "momentum_rank=" in ranking.loc[ranking["ts_code"] == "000001.SZ", "score_explain"].iloc[0]


def test_build_ranking_snapshot_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unsupported ranking variant"):
        build_ranking_snapshot(_universe_frame(), _config(), variant="unknown")


def test_render_ranking_factor_map_documents_hard_filters_and_scores() -> None:
    content = render_ranking_factor_map()

    assert "exchange_not_supported" in content
    assert "momentum_rank" in content
    assert "rotation_edge" in content
