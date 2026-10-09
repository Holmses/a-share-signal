from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ashare_signal.backtest.recipe_comparison import RecipeComparisonStudyEngine
from ashare_signal.backtest.recipe_comparison import build_recipe_portfolio_curves
from ashare_signal.backtest.recipe_comparison import summarize_recipe_events, summarize_recipe_exposure
from ashare_signal.backtest.recipe_comparison import summarize_recipe_portfolios
from ashare_signal.config import RecipeConfig


def _config():
    return SimpleNamespace(
        recipes=(
            RecipeConfig(name="trend_pullback_rank"),
            RecipeConfig(name="rebound_bottoming_rank"),
        ),
        pricing=SimpleNamespace(buy_markup=0.003, sell_markdown=0.003),
        backtest=SimpleNamespace(commission_rate=0.0003, stamp_duty_rate=0.001),
    )


def _row(symbol: str, **overrides):
    row = {
        "ts_code": symbol,
        "name": symbol,
        "group": "main",
        "industry": "电子",
        "style_group": "电子",
        "avg_amount_20d_yuan": 100_000_000.0,
        "total_mv_yuan": 50_000_000_000.0,
        "return_5d": 0.03,
        "return_30d": 0.20,
        "return_90d": 0.35,
        "return_30d_rank": 0.90,
        "return_90d_rank": 0.80,
        "close_to_ma_5": 0.02,
        "close_to_ma_10": 0.03,
        "close_to_ma_20": 0.04,
        "drawdown_from_20d_high": -0.08,
        "upper_shadow_pct": 0.20,
        "volume_ratio": 1.20,
        "amount_ratio_5d": 1.10,
        "quality_score": 0.80,
        "quality_momentum_score": 0.85,
        "trend_quality_score": 0.90,
        "near_high_score": 0.70,
        "amount_rank": 0.75,
        "stability_score": 0.70,
        "volume_ratio_score": 0.80,
        "financial_quality_score": 0.60,
        "market_cap_rank": 0.55,
        "benchmark_return_20d": 0.02,
        "benchmark_close_to_ma20": 0.01,
    }
    row.update(overrides)
    return row


def test_recipe_comparison_selects_configured_recipes_and_combo() -> None:
    engine = RecipeComparisonStudyEngine(
        config=_config(),
        repository=SimpleNamespace(),
        base_dir=Path("."),
        recipes=["trend_pullback_rank", "rebound_bottoming_rank", "combo_configured"],
        top_n_per_recipe=1,
        horizons=[1],
    )
    frame = pd.DataFrame(
        [
            _row("TREND", return_30d=0.22, drawdown_from_20d_high=-0.06),
            _row(
                "REBOUND",
                return_5d=0.00,
                return_30d=0.02,
                return_90d=0.02,
                drawdown_from_20d_high=-0.18,
                trend_quality_score=0.40,
            ),
        ]
    )

    trend = engine._select_recipe(frame, "trend_pullback_rank")
    rebound = engine._select_recipe(frame, "rebound_bottoming_rank")
    combo = engine._combine_recipe_candidates(
        {
            "trend_pullback_rank": trend,
            "rebound_bottoming_rank": rebound,
        }
    )

    assert [row["ts_code"] for row in trend] == ["TREND"]
    assert [row["ts_code"] for row in rebound] == ["REBOUND"]
    assert {row["source_recipe"] for row in combo} == {"trend_pullback_rank", "rebound_bottoming_rank"}


def test_recipe_comparison_summarizes_performance_and_exposure() -> None:
    events = pd.DataFrame(
        [
            {
                "recipe": "trend_pullback_rank",
                "market_state": "risk_on",
                "close_return_1d": 0.03,
                "close_return_net_1d": 0.02,
                "mfe_1d": 0.04,
                "mae_1d": -0.01,
                "group": "main",
                "industry": "电子",
                "style_group": "电子",
                "market_cap_tier": "mid",
                "source_recipe": "trend_pullback_rank",
            },
            {
                "recipe": "trend_pullback_rank",
                "market_state": "risk_off",
                "close_return_1d": -0.01,
                "close_return_net_1d": -0.02,
                "mfe_1d": 0.01,
                "mae_1d": -0.03,
                "group": "main",
                "industry": "电子",
                "style_group": "电子",
                "market_cap_tier": "mid",
                "source_recipe": "trend_pullback_rank",
            },
        ]
    )

    summary = summarize_recipe_events(events, [1])
    exposure = summarize_recipe_exposure(events)

    all_row = summary.loc[
        (summary["recipe"] == "trend_pullback_rank") & (summary["market_state"] == "ALL")
    ].iloc[0]
    assert all_row["events"] == 2
    assert all_row["win_rate_net"] == 0.5
    assert all_row["avg_close_return_net"] == 0.0
    group_row = exposure.loc[
        (exposure["recipe"] == "trend_pullback_rank")
        & (exposure["exposure_type"] == "group")
        & (exposure["exposure"] == "main")
    ].iloc[0]
    assert group_row["events"] == 2
    assert group_row["weight"] == 1.0


def test_recipe_comparison_builds_equal_weight_portfolio_summary() -> None:
    events = pd.DataFrame(
        [
            {
                "recipe": "trend_pullback_rank",
                "entry_trade_date": "20260506",
                "symbol": "AAA",
                "score": 0.8,
                "close_return_net_3d": 0.03,
            },
            {
                "recipe": "trend_pullback_rank",
                "entry_trade_date": "20260506",
                "symbol": "BBB",
                "score": 0.7,
                "close_return_net_3d": -0.01,
            },
            {
                "recipe": "trend_pullback_rank",
                "entry_trade_date": "20260507",
                "symbol": "CCC",
                "score": 0.9,
                "close_return_net_3d": 0.02,
            },
        ]
    )

    curve = build_recipe_portfolio_curves(events, [3])
    summary = summarize_recipe_portfolios(curve)

    assert list(curve["basket_size"]) == [2, 1]
    assert list(curve["basket_return_net"].round(4)) == [0.01, 0.02]
    row = summary.iloc[0]
    assert row["periods"] == 2
    assert row["win_rate"] == 1.0
    assert round(row["total_return"], 4) == 0.0302
