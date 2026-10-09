from pathlib import Path
import json

import pytest

from ashare_signal.backtest.full_a_momentum import FullAMomentumBacktestEngine
from ashare_signal.config import AppConfig
from ashare_signal.config import BacktestConfig, FilterConfig, MarketConfig, PathConfig
from ashare_signal.config import PricingConfig, RuntimeConfig, SelectionConfig, StrategyConfig
from ashare_signal.config import load_config
from ashare_signal.strategy.recipe import PortfolioRecipe, UniverseRecipe
from ashare_signal.strategy.recipe import configured_recipes_from_app_config
from ashare_signal.strategy.recipe import full_a_momentum_recipe
from ashare_signal.strategy.recipe import ranking_rotation_recipe
from ashare_signal.strategy.recipe import recipe_from_app_config


def _config() -> AppConfig:
    return AppConfig(
        market=MarketConfig(
            name="ashare",
            benchmark="000300.SH",
            max_positions=5,
            min_position_holding_days=1,
        ),
        filters=FilterConfig(
            min_list_days=60,
            min_price=3.0,
            min_avg_turnover=50_000_000.0,
            exclude_st=True,
            exclude_suspended=True,
        ),
        pricing=PricingConfig(
            buy_markup=0.003,
            sell_markdown=0.003,
            cancel_if_gap_exceeds=0.02,
        ),
        strategy=StrategyConfig(
            buy_top_n=1,
            sell_top_n=1,
            lookback_momentum_days=20,
            lookback_short_days=5,
            lookback_vol_days=20,
        ),
        backtest=BacktestConfig(
            initial_cash=1_000_000.0,
            commission_rate=0.0003,
            stamp_duty_rate=0.001,
            lot_size=100,
        ),
        selection=SelectionConfig(
            market_min_breadth=0.50,
            min_buy_score=0.60,
            rotation_edge=0.25,
        ),
        runtime=RuntimeConfig(),
        paths=PathConfig(
            raw_data_dir=Path("data/raw"),
            processed_data_dir=Path("data/processed"),
            reports_dir=Path("reports/generated"),
            logs_dir=Path("logs"),
        ),
        tushare_token=None,
    )


def test_recipe_from_app_config_is_json_safe() -> None:
    recipe = recipe_from_app_config(_config())

    payload = recipe.to_dict()

    assert recipe.slug == "v1-signal-selector"
    assert len(recipe.fingerprint) == 12
    assert payload["universe"]["min_avg_amount_yuan"] == 50_000_000.0
    assert payload["alpha"]["variant"] == "v1_trend_rebound"
    assert payload["portfolio"]["max_positions"] == 5
    assert payload["entry"]["min_score"] == 0.60
    assert payload["risk"]["market_min_breadth"] == 0.50
    assert payload["tags"] == ["production", "v1"]
    json.dumps(payload, ensure_ascii=False)


def test_configured_recipes_load_from_example_config() -> None:
    config = load_config(Path("configs/strategy.toml.example"))

    recipes = configured_recipes_from_app_config(config)

    assert [recipe.recipe_id for recipe in recipes] == ["trend_pullback_rank", "rebound_bottoming_rank"]
    trend, rebound = recipes
    assert trend.alpha.factor_set == ("momentum", "trend", "pullback", "liquidity", "volatility")
    assert trend.portfolio.max_positions == 3
    assert trend.entry.min_score == 0.60
    assert trend.entry.max_daily_buys == 1
    assert trend.exit.sell_rules == ("rotation_rank_drop", "trailing_take_profit", "trend_break")
    assert trend.exit.max_daily_sells == 1
    assert trend.risk.market_gate == "risk_on"
    assert rebound.alpha.factor_set == ("drawdown", "stabilization", "moneyflow", "industry_rebound")
    assert rebound.portfolio.max_positions == 2
    assert rebound.portfolio.sizing == "partial_equal_weight"
    assert rebound.risk.market_gate == "risk_neutral_or_rebound"
    assert json.dumps([recipe.to_dict() for recipe in recipes], ensure_ascii=False)


def test_full_a_momentum_recipe_captures_selection_and_exit_contract() -> None:
    recipe = full_a_momentum_recipe(
        _config(),
        top_n=5,
        max_positions=5,
        selection_variant="quality_momentum",
        market_min_breadth=0.50,
        style_min_breadth=0.48,
        hard_exit_days=23,
    )

    assert recipe.family == "full_a_momentum"
    assert recipe.alpha.score_column == "quality_momentum_score"
    assert recipe.portfolio.target_positions == 5
    assert recipe.portfolio.max_holding_days == 10
    assert recipe.exit.hard_exit_days == 23
    assert recipe.risk.style_min_breadth == 0.48
    assert recipe.research_only is False


def test_full_a_momentum_engine_can_be_created_from_recipe(tmp_path) -> None:
    recipe = full_a_momentum_recipe(
        _config(),
        top_n=3,
        hold_days=4,
        max_hold_days=12,
        max_positions=6,
        market_min_return_20d=0.0,
        style_min_return_20d=-0.02,
        hard_exit_days=23,
    )

    engine = FullAMomentumBacktestEngine.from_recipe(
        config=_config(),
        repository=object(),
        base_dir=tmp_path,
        recipe=recipe,
    )

    assert engine.top_n == 3
    assert engine.hold_days == 4
    assert engine.max_hold_days == 12
    assert engine.max_positions == 6
    assert engine.market_min_return_20d == 0.0
    assert engine.style_min_return_20d == -0.02
    assert engine.hard_exit_days == 23


def test_ranking_rotation_recipe_marks_research_only_controls() -> None:
    recipe = ranking_rotation_recipe(
        _config(),
        top_k=5,
        candidate_buffer_k=20,
        drop_n=1,
        risk_off_cash_guard=True,
    )

    assert recipe.family == "ranking_rotation"
    assert recipe.alpha.ranking_variant == "quality_momentum_rank"
    assert recipe.rotation is not None
    assert recipe.rotation.candidate_buffer_k == 20
    assert recipe.rotation.drop_n == 1
    assert recipe.risk.allow_buy_when_risk_off is False
    assert recipe.research_only is True


def test_recipe_validation_rejects_invalid_position_contracts() -> None:
    with pytest.raises(ValueError, match="universe groups"):
        UniverseRecipe(groups=())

    with pytest.raises(ValueError, match="target_positions"):
        PortfolioRecipe(max_positions=2, target_positions=3)
