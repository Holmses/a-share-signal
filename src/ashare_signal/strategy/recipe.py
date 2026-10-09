from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

from ashare_signal.config import AppConfig, RecipeConfig
from ashare_signal.strategy.exit_rules import DEFAULT_EXIT_PROFILE, DEFAULT_FAILURE_EXIT_DAYS
from ashare_signal.strategy.exit_rules import DEFAULT_FAILURE_EXIT_MIN_PEAK_PROFIT_PCT, DEFAULT_HARD_EXIT_DAYS
from ashare_signal.strategy.exit_rules import DEFAULT_VOLUME_STALL_EXIT, DEFAULT_VOLUME_STALL_RATIO


DEFAULT_RECIPE_GROUPS: tuple[str, ...] = ("main", "chinext", "star")

ScalarValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class RecipeParam:
    name: str
    value: ScalarValue
    unit: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("recipe parameter name must not be empty")
        object.__setattr__(self, "name", self.name.strip())


@dataclass(frozen=True, slots=True)
class UniverseRecipe:
    groups: tuple[str, ...] = DEFAULT_RECIPE_GROUPS
    min_list_days: int = 60
    min_price: float = 3.0
    min_avg_amount_yuan: float = 50_000_000.0
    exclude_st: bool = True
    exclude_suspended: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", _normalize_names(self.groups, "universe groups"))
        _require_positive_int(self.min_list_days, "min_list_days")
        _require_positive_float(self.min_price, "min_price")
        _require_positive_float(self.min_avg_amount_yuan, "min_avg_amount_yuan")


@dataclass(frozen=True, slots=True)
class AlphaRecipe:
    family: str
    variant: str
    factor_set: tuple[str, ...] = ()
    score_column: str | None = None
    ranking_variant: str | None = None
    lookback_momentum_days: int | None = None
    lookback_short_days: int | None = None
    lookback_vol_days: int | None = None
    params: tuple[RecipeParam, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _require_name(self.family, "alpha family"))
        object.__setattr__(self, "variant", _require_name(self.variant, "alpha variant"))
        object.__setattr__(self, "factor_set", _normalize_names(self.factor_set, "factor_set", allow_empty=True))
        object.__setattr__(self, "params", tuple(self.params))
        for field_name in ("lookback_momentum_days", "lookback_short_days", "lookback_vol_days"):
            value = getattr(self, field_name)
            if value is not None:
                _require_positive_int(value, field_name)


@dataclass(frozen=True, slots=True)
class PortfolioRecipe:
    max_positions: int
    target_positions: int | None = None
    sizing: str = "equal_weight"
    min_holding_days: int = 1
    max_holding_days: int | None = None
    lot_size: int = 100
    params: tuple[RecipeParam, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int(self.max_positions, "max_positions")
        _require_positive_int(self.min_holding_days, "min_holding_days")
        _require_positive_int(self.lot_size, "lot_size")
        if self.target_positions is not None:
            _require_positive_int(self.target_positions, "target_positions")
            if self.target_positions > self.max_positions:
                raise ValueError("target_positions must not exceed max_positions")
        if self.max_holding_days is not None:
            _require_positive_int(self.max_holding_days, "max_holding_days")
            if self.max_holding_days < self.min_holding_days:
                raise ValueError("max_holding_days must be greater than or equal to min_holding_days")
        object.__setattr__(self, "sizing", _require_name(self.sizing, "portfolio sizing"))
        object.__setattr__(self, "params", tuple(self.params))


@dataclass(frozen=True, slots=True)
class EntryRecipe:
    top_n: int
    min_score: float | None = None
    signal_lag_days: int = 1
    max_daily_buys: int | None = None
    params: tuple[RecipeParam, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int(self.top_n, "top_n")
        _require_non_negative_int(self.signal_lag_days, "signal_lag_days")
        if self.max_daily_buys is not None:
            _require_positive_int(self.max_daily_buys, "max_daily_buys")
        object.__setattr__(self, "params", tuple(self.params))


@dataclass(frozen=True, slots=True)
class ExitRecipe:
    profile: str = DEFAULT_EXIT_PROFILE
    sell_rules: tuple[str, ...] = ()
    hard_exit_days: int | None = DEFAULT_HARD_EXIT_DAYS
    stop_loss_pct: float | None = None
    take_profit_trigger_pct: float | None = None
    trailing_stop_drawdown_pct: float | None = None
    ma20_break: bool = False
    failure_days: int | None = DEFAULT_FAILURE_EXIT_DAYS
    failure_min_peak_profit_pct: float = DEFAULT_FAILURE_EXIT_MIN_PEAK_PROFIT_PCT
    adaptive_trailing: bool = False
    atr_multiplier: float = 1.5
    market_risk_exit: bool = False
    industry_weak_exit: bool = False
    relative_weak_exit: bool = False
    relative_weak_5d_pct: float = 0.04
    relative_weak_20d_pct: float = 0.08
    volume_stall_exit: bool = DEFAULT_VOLUME_STALL_EXIT
    volume_stall_ratio: float = DEFAULT_VOLUME_STALL_RATIO
    upper_shadow_exit: bool = False
    upper_shadow_pct: float = 0.45
    max_daily_sells: int | None = None
    params: tuple[RecipeParam, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", _require_name(self.profile, "exit profile"))
        object.__setattr__(self, "sell_rules", _normalize_names(self.sell_rules, "sell_rules", allow_empty=True))
        if self.hard_exit_days is not None:
            _require_positive_int(self.hard_exit_days, "hard_exit_days")
        if self.failure_days is not None:
            _require_positive_int(self.failure_days, "failure_days")
        if self.max_daily_sells is not None:
            _require_positive_int(self.max_daily_sells, "max_daily_sells")
        for field_name in (
            "stop_loss_pct",
            "take_profit_trigger_pct",
            "trailing_stop_drawdown_pct",
            "failure_min_peak_profit_pct",
            "atr_multiplier",
            "relative_weak_5d_pct",
            "relative_weak_20d_pct",
            "volume_stall_ratio",
            "upper_shadow_pct",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_positive_float(value, field_name)
        object.__setattr__(self, "params", tuple(self.params))


@dataclass(frozen=True, slots=True)
class RiskRecipe:
    market_gate: str | None = None
    market_min_breadth: float | None = None
    market_min_return_20d: float | None = None
    style_min_breadth: float | None = None
    style_min_return_20d: float | None = None
    style_score_weight: float | None = None
    allow_buy_when_risk_off: bool = False
    force_exit_when_risk_off: bool = False
    loss_cooldown_days: int = 0
    params: tuple[RecipeParam, ...] = ()

    def __post_init__(self) -> None:
        if self.market_gate is not None:
            object.__setattr__(self, "market_gate", _require_name(self.market_gate, "market_gate"))
        _require_non_negative_int(self.loss_cooldown_days, "loss_cooldown_days")
        object.__setattr__(self, "params", tuple(self.params))


@dataclass(frozen=True, slots=True)
class ExecutionRecipe:
    buy_markup: float
    sell_markdown: float
    cancel_if_gap_exceeds: float | None = None
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.001

    def __post_init__(self) -> None:
        for field_name in (
            "buy_markup",
            "sell_markdown",
            "initial_cash",
            "commission_rate",
            "stamp_duty_rate",
        ):
            _require_non_negative_float(getattr(self, field_name), field_name)
        if self.cancel_if_gap_exceeds is not None:
            _require_non_negative_float(self.cancel_if_gap_exceeds, "cancel_if_gap_exceeds")


@dataclass(frozen=True, slots=True)
class RotationRecipe:
    candidate_buffer_k: int
    drop_n: int
    min_score_edge: float = 0.0
    rotation_min_holding_days: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(self.candidate_buffer_k, "candidate_buffer_k")
        _require_positive_int(self.drop_n, "drop_n")
        _require_non_negative_float(self.min_score_edge, "min_score_edge")
        _require_non_negative_int(self.rotation_min_holding_days, "rotation_min_holding_days")
        if self.drop_n > self.candidate_buffer_k:
            raise ValueError("drop_n must not exceed candidate_buffer_k")


@dataclass(frozen=True, slots=True)
class StrategyRecipe:
    recipe_id: str
    name: str
    family: str
    universe: UniverseRecipe
    alpha: AlphaRecipe
    portfolio: PortfolioRecipe
    entry: EntryRecipe
    exit: ExitRecipe
    risk: RiskRecipe
    execution: ExecutionRecipe
    rotation: RotationRecipe | None = None
    research_only: bool = False
    tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe_id", _require_name(self.recipe_id, "recipe_id"))
        object.__setattr__(self, "name", _require_name(self.name, "recipe name"))
        object.__setattr__(self, "family", _require_name(self.family, "recipe family"))
        object.__setattr__(self, "tags", _normalize_names(self.tags, "recipe tags", allow_empty=True))
        object.__setattr__(self, "notes", tuple(str(note).strip() for note in self.notes if str(note).strip()))

    @property
    def slug(self) -> str:
        return _slug(self.recipe_id)

    @property
    def fingerprint(self) -> str:
        payload = _json_safe(asdict(self))
        payload["slug"] = self.slug
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe(asdict(self))
        payload["slug"] = self.slug
        payload["fingerprint"] = self.fingerprint
        return payload


def recipe_from_app_config(
    config: AppConfig,
    *,
    recipe_id: str = "v1_signal_selector",
    name: str = "V1 signal selector",
) -> StrategyRecipe:
    """Build a recipe for the current production selector configuration."""
    return StrategyRecipe(
        recipe_id=recipe_id,
        name=name,
        family="signal_selector",
        universe=UniverseRecipe(
            groups=DEFAULT_RECIPE_GROUPS,
            min_list_days=config.filters.min_list_days,
            min_price=config.filters.min_price,
            min_avg_amount_yuan=config.filters.min_avg_turnover,
            exclude_st=config.filters.exclude_st,
            exclude_suspended=config.filters.exclude_suspended,
        ),
        alpha=AlphaRecipe(
            family="selector_score",
            variant="v1_trend_rebound",
            factor_set=(
                "momentum",
                "trend",
                "pullback",
                "liquidity",
                "volatility",
                "rebound",
                "moneyflow",
                "industry",
            ),
            score_column="buy_score",
            lookback_momentum_days=config.strategy.lookback_momentum_days,
            lookback_short_days=config.strategy.lookback_short_days,
            lookback_vol_days=config.strategy.lookback_vol_days,
            params=(
                RecipeParam("enable_rebound_strategy", config.selection.enable_rebound_strategy),
                RecipeParam("rotation_edge", config.selection.rotation_edge),
            ),
        ),
        portfolio=PortfolioRecipe(
            max_positions=config.market.max_positions,
            target_positions=config.market.max_positions,
            min_holding_days=config.market.min_position_holding_days,
            lot_size=config.backtest.lot_size,
        ),
        entry=EntryRecipe(
            top_n=config.strategy.buy_top_n,
            min_score=config.selection.min_buy_score,
            signal_lag_days=1,
        ),
        exit=ExitRecipe(
            profile="v1_selector_exit",
            sell_rules=(
                "sell_health_exit",
                "hard_stop_loss",
                "trailing_take_profit",
                "trend_break",
                "rotation_edge",
            ),
            stop_loss_pct=config.selection.stop_loss_pct,
            take_profit_trigger_pct=config.selection.take_profit_trigger_pct,
            trailing_stop_drawdown_pct=config.selection.trailing_stop_drawdown_pct,
            params=(
                RecipeParam("sell_top_n", config.strategy.sell_top_n),
                RecipeParam("sell_health_exit_threshold", config.selection.sell_health_exit_threshold),
                RecipeParam("trend_exit_min_holding_days", config.selection.trend_exit_min_holding_days),
            ),
        ),
        risk=RiskRecipe(market_gate="risk_on", market_min_breadth=config.selection.market_min_breadth),
        execution=_execution_from_config(config),
        tags=("production", "v1"),
    )


def configured_recipes_from_app_config(config: AppConfig) -> tuple[StrategyRecipe, ...]:
    """Build enabled StrategyRecipe objects from the optional [[recipes]] config."""
    return tuple(recipe_from_recipe_config(config, recipe_config) for recipe_config in config.recipes if recipe_config.enabled)


def recipe_from_recipe_config(config: AppConfig, recipe_config: RecipeConfig) -> StrategyRecipe:
    if recipe_config.name == "trend_pullback_rank":
        return _trend_pullback_recipe(config, recipe_config)
    if recipe_config.name == "rebound_bottoming_rank":
        return _rebound_bottoming_recipe(config, recipe_config)
    return _generic_configured_recipe(config, recipe_config)


def full_a_momentum_recipe(
    config: AppConfig,
    *,
    recipe_id: str = "full_a_momentum_quality_momentum",
    name: str = "Full A momentum",
    top_n: int = 5,
    hold_days: int = 5,
    max_hold_days: int = 10,
    max_positions: int | None = None,
    groups: tuple[str, ...] = DEFAULT_RECIPE_GROUPS,
    selection_variant: str = "quality_momentum",
    min_avg_amount_yuan: float = 50_000_000.0,
    market_min_breadth: float = 0.50,
    market_min_return_20d: float = 0.0,
    style_min_breadth: float = 0.48,
    style_min_return_20d: float = -0.01,
    style_score_weight: float = 0.06,
    loss_cooldown_days: int = 3,
    stop_loss_pct: float = 0.05,
    take_profit_trigger_pct: float = 0.08,
    trailing_stop_drawdown_pct: float = 0.04,
    hard_exit_days: int | None = DEFAULT_HARD_EXIT_DAYS,
    exit_profile: str = DEFAULT_EXIT_PROFILE,
) -> StrategyRecipe:
    """Build a recipe for the existing Full A momentum backtest family."""
    resolved_max_positions = max_positions or config.market.max_positions
    return StrategyRecipe(
        recipe_id=recipe_id,
        name=name,
        family="full_a_momentum",
        universe=UniverseRecipe(
            groups=groups,
            min_list_days=config.filters.min_list_days,
            min_price=config.filters.min_price,
            min_avg_amount_yuan=min_avg_amount_yuan,
            exclude_st=config.filters.exclude_st,
            exclude_suspended=config.filters.exclude_suspended,
        ),
        alpha=AlphaRecipe(
            family="selection_event_study",
            variant=selection_variant,
            factor_set=("momentum", "quality", "liquidity", "style"),
            score_column=f"{selection_variant}_score",
            lookback_momentum_days=config.strategy.lookback_momentum_days,
            lookback_short_days=config.strategy.lookback_short_days,
            lookback_vol_days=config.strategy.lookback_vol_days,
        ),
        portfolio=PortfolioRecipe(
            max_positions=resolved_max_positions,
            target_positions=top_n,
            min_holding_days=hold_days,
            max_holding_days=max_hold_days,
            lot_size=config.backtest.lot_size,
        ),
        entry=EntryRecipe(top_n=top_n, signal_lag_days=1),
        exit=ExitRecipe(
            profile=exit_profile,
            sell_rules=("tiered_trailing_take_profit", "hard_exit_days"),
            hard_exit_days=hard_exit_days,
            stop_loss_pct=stop_loss_pct,
            take_profit_trigger_pct=take_profit_trigger_pct,
            trailing_stop_drawdown_pct=trailing_stop_drawdown_pct,
        ),
        risk=RiskRecipe(
            market_gate="risk_on",
            market_min_breadth=market_min_breadth,
            market_min_return_20d=market_min_return_20d,
            style_min_breadth=style_min_breadth,
            style_min_return_20d=style_min_return_20d,
            style_score_weight=style_score_weight,
            loss_cooldown_days=loss_cooldown_days,
        ),
        execution=_execution_from_config(config),
        tags=("research", "backtest"),
    )


def ranking_rotation_recipe(
    config: AppConfig,
    *,
    recipe_id: str = "ranking_rotation_quality_momentum_rank_top5_drop1",
    name: str = "Ranking rotation TopK DropN",
    ranking_variant: str = "quality_momentum_rank",
    top_k: int = 5,
    candidate_buffer_k: int = 20,
    drop_n: int = 1,
    max_positions: int | None = None,
    min_score_edge: float = 0.02,
    min_holding_days: int = 3,
    rotation_min_holding_days: int = 5,
    groups: tuple[str, ...] = DEFAULT_RECIPE_GROUPS,
    min_avg_amount_yuan: float = 50_000_000.0,
    market_min_breadth: float = 0.50,
    market_min_return_20d: float = 0.0,
    risk_off_cash_guard: bool = True,
    risk_off_exit: bool = False,
) -> StrategyRecipe:
    """Build a recipe for the research-only ranking rotation experiment."""
    resolved_max_positions = max_positions or top_k
    return StrategyRecipe(
        recipe_id=recipe_id,
        name=name,
        family="ranking_rotation",
        universe=UniverseRecipe(
            groups=groups,
            min_list_days=config.filters.min_list_days,
            min_price=config.filters.min_price,
            min_avg_amount_yuan=min_avg_amount_yuan,
            exclude_st=config.filters.exclude_st,
            exclude_suspended=config.filters.exclude_suspended,
        ),
        alpha=AlphaRecipe(
            family="cross_sectional_rank",
            variant=ranking_variant,
            factor_set=("momentum", "trend", "pullback", "liquidity", "volatility", "industry"),
            score_column="rank_score",
            ranking_variant=ranking_variant,
            lookback_momentum_days=config.strategy.lookback_momentum_days,
            lookback_short_days=config.strategy.lookback_short_days,
            lookback_vol_days=config.strategy.lookback_vol_days,
        ),
        portfolio=PortfolioRecipe(
            max_positions=resolved_max_positions,
            target_positions=top_k,
            min_holding_days=min_holding_days,
            lot_size=config.backtest.lot_size,
        ),
        entry=EntryRecipe(top_n=top_k, signal_lag_days=1),
        exit=ExitRecipe(
            profile="ranking_drop_rotation",
            sell_rules=("rotation_rank_drop", "score_edge_rotation", "market_risk_exit"),
            market_risk_exit=risk_off_exit,
            params=(RecipeParam("candidate_buffer_k", candidate_buffer_k),),
        ),
        risk=RiskRecipe(
            market_gate="cash_guard" if risk_off_cash_guard else "riskbuy",
            market_min_breadth=market_min_breadth,
            market_min_return_20d=market_min_return_20d,
            allow_buy_when_risk_off=not risk_off_cash_guard,
            force_exit_when_risk_off=risk_off_exit,
        ),
        execution=_execution_from_config(config),
        rotation=RotationRecipe(
            candidate_buffer_k=candidate_buffer_k,
            drop_n=drop_n,
            min_score_edge=min_score_edge,
            rotation_min_holding_days=rotation_min_holding_days,
        ),
        research_only=True,
        tags=("research", "ranking"),
    )


def _trend_pullback_recipe(config: AppConfig, recipe_config: RecipeConfig) -> StrategyRecipe:
    max_positions = recipe_config.max_positions or config.market.max_positions
    return StrategyRecipe(
        recipe_id=recipe_config.name,
        name="Trend pullback rank",
        family="configured_recipe",
        universe=_universe_from_config(config),
        alpha=AlphaRecipe(
            family="trend_pullback",
            variant=recipe_config.name,
            factor_set=recipe_config.factor_set or ("momentum", "trend", "pullback", "liquidity", "volatility"),
            score_column="buy_score",
            lookback_momentum_days=config.strategy.lookback_momentum_days,
            lookback_short_days=config.strategy.lookback_short_days,
            lookback_vol_days=config.strategy.lookback_vol_days,
            params=(
                RecipeParam("buy_momentum_weight", config.selection.buy_momentum_weight),
                RecipeParam("buy_trend_weight", config.selection.buy_trend_weight),
                RecipeParam("buy_pullback_weight", config.selection.buy_pullback_weight),
                RecipeParam("buy_liquidity_weight", config.selection.buy_liquidity_weight),
                RecipeParam("buy_volatility_weight", config.selection.buy_volatility_weight),
                RecipeParam("buy_volume_ratio_weight", config.selection.buy_volume_ratio_weight),
                RecipeParam("buy_ma20_weight", config.selection.buy_ma20_weight),
                RecipeParam("buy_ma10_weight", config.selection.buy_ma10_weight),
            ),
        ),
        portfolio=PortfolioRecipe(
            max_positions=max_positions,
            target_positions=max_positions,
            min_holding_days=config.market.min_position_holding_days,
            lot_size=config.backtest.lot_size,
        ),
        entry=EntryRecipe(
            top_n=config.strategy.buy_top_n,
            min_score=_recipe_min_score(recipe_config, config.selection.min_buy_score),
            signal_lag_days=1,
            max_daily_buys=recipe_config.max_daily_buys,
            params=(
                RecipeParam("buy_min_close_to_ma20", config.selection.buy_min_close_to_ma20),
                RecipeParam("buy_max_close_to_ma20", config.selection.buy_max_close_to_ma20),
                RecipeParam("buy_min_pullback_from_20d_high", config.selection.buy_min_pullback_from_20d_high),
                RecipeParam("buy_max_pullback_from_20d_high", config.selection.buy_max_pullback_from_20d_high),
                RecipeParam("buy_min_momentum_5d", config.selection.buy_min_momentum_5d),
                RecipeParam("buy_max_volume_ratio", config.selection.buy_max_volume_ratio),
                RecipeParam("buy_min_amount_ratio_5d", config.selection.buy_min_amount_ratio_5d),
                RecipeParam("buy_max_amount_ratio_5d", config.selection.buy_max_amount_ratio_5d),
            ),
        ),
        exit=ExitRecipe(
            profile="trend_pullback_exit",
            sell_rules=recipe_config.sell_rules,
            stop_loss_pct=config.selection.stop_loss_pct,
            take_profit_trigger_pct=config.selection.take_profit_trigger_pct,
            trailing_stop_drawdown_pct=config.selection.trailing_stop_drawdown_pct,
            max_daily_sells=recipe_config.max_daily_sells,
            params=(
                RecipeParam("sell_health_exit_threshold", config.selection.sell_health_exit_threshold),
                RecipeParam("trend_exit_min_holding_days", config.selection.trend_exit_min_holding_days),
                RecipeParam("rotation_min_holding_days", config.selection.rotation_min_holding_days),
            ),
        ),
        risk=RiskRecipe(
            market_gate=recipe_config.market_gate,
            market_min_breadth=config.selection.market_min_breadth,
        ),
        execution=_execution_from_config(config),
        tags=("configured", "jq1", "trend"),
        notes=recipe_config.notes,
    )


def _rebound_bottoming_recipe(config: AppConfig, recipe_config: RecipeConfig) -> StrategyRecipe:
    max_positions = recipe_config.max_positions or config.selection.rebound_max_positions
    return StrategyRecipe(
        recipe_id=recipe_config.name,
        name="Rebound bottoming rank",
        family="configured_recipe",
        universe=_universe_from_config(config),
        alpha=AlphaRecipe(
            family="rebound_bottoming",
            variant=recipe_config.name,
            factor_set=recipe_config.factor_set or ("drawdown", "stabilization", "moneyflow", "industry_rebound"),
            score_column="buy_score",
            lookback_momentum_days=config.strategy.lookback_momentum_days,
            lookback_short_days=config.strategy.lookback_short_days,
            lookback_vol_days=config.strategy.lookback_vol_days,
            params=(
                RecipeParam("buy_reversal_weight", config.selection.buy_reversal_weight),
                RecipeParam("rebound_min_drawdown_20d", config.selection.rebound_min_drawdown_20d),
                RecipeParam("rebound_min_drawdown_60d", config.selection.rebound_min_drawdown_60d),
                RecipeParam("rebound_max_drawdown_60d", config.selection.rebound_max_drawdown_60d),
                RecipeParam("rebound_min_large_net_mf_to_amount", config.selection.rebound_min_large_net_mf_to_amount),
                RecipeParam(
                    "rebound_prefer_large_net_mf_to_amount",
                    config.selection.rebound_prefer_large_net_mf_to_amount,
                ),
            ),
        ),
        portfolio=PortfolioRecipe(
            max_positions=max_positions,
            target_positions=max_positions,
            sizing="partial_equal_weight",
            min_holding_days=config.market.min_position_holding_days,
            lot_size=config.backtest.lot_size,
            params=(RecipeParam("rebound_position_size_multiplier", config.selection.rebound_position_size_multiplier),),
        ),
        entry=EntryRecipe(
            top_n=config.strategy.buy_top_n,
            min_score=_recipe_min_score(recipe_config, config.selection.rebound_min_score),
            signal_lag_days=1,
            max_daily_buys=recipe_config.max_daily_buys,
            params=(
                RecipeParam("rebound_max_close_to_ma60_below", config.selection.rebound_max_close_to_ma60_below),
                RecipeParam("rebound_max_down_days_10d", config.selection.rebound_max_down_days_10d),
                RecipeParam("rebound_min_return_3d", config.selection.rebound_min_return_3d),
                RecipeParam("rebound_max_return_3d", config.selection.rebound_max_return_3d),
                RecipeParam("rebound_max_close_to_ma5", config.selection.rebound_max_close_to_ma5),
                RecipeParam("rebound_max_close_to_ma20", config.selection.rebound_max_close_to_ma20),
                RecipeParam("rebound_min_amount_ratio_5d", config.selection.rebound_min_amount_ratio_5d),
                RecipeParam("rebound_max_amount_ratio_5d", config.selection.rebound_max_amount_ratio_5d),
            ),
        ),
        exit=ExitRecipe(
            profile="rebound_bottoming_exit",
            sell_rules=recipe_config.sell_rules,
            stop_loss_pct=config.selection.rebound_stop_loss_pct,
            max_daily_sells=recipe_config.max_daily_sells,
            params=(
                RecipeParam("rebound_fast_exit_days", config.selection.rebound_fast_exit_days),
                RecipeParam("rebound_breakeven_trigger_pct", config.selection.rebound_breakeven_trigger_pct),
                RecipeParam("rebound_breakeven_floor_pct", config.selection.rebound_breakeven_floor_pct),
                RecipeParam("rebound_profit_lock_trigger_pct", config.selection.rebound_profit_lock_trigger_pct),
                RecipeParam("rebound_profit_lock_drawdown_pct", config.selection.rebound_profit_lock_drawdown_pct),
                RecipeParam("rebound_big_profit_trigger_pct", config.selection.rebound_big_profit_trigger_pct),
                RecipeParam("rebound_big_profit_drawdown_pct", config.selection.rebound_big_profit_drawdown_pct),
                RecipeParam("rebound_profit_exit_min_pct", config.selection.rebound_profit_exit_min_pct),
            ),
        ),
        risk=RiskRecipe(
            market_gate=recipe_config.market_gate,
            market_min_breadth=config.selection.rebound_market_min_breadth,
        ),
        execution=_execution_from_config(config),
        tags=("configured", "jq1", "rebound"),
        notes=recipe_config.notes,
    )


def _generic_configured_recipe(config: AppConfig, recipe_config: RecipeConfig) -> StrategyRecipe:
    max_positions = recipe_config.max_positions or config.market.max_positions
    return StrategyRecipe(
        recipe_id=recipe_config.name,
        name=recipe_config.name.replace("_", " ").title(),
        family="configured_recipe",
        universe=_universe_from_config(config),
        alpha=AlphaRecipe(
            family="configured_factor_set",
            variant=recipe_config.name,
            factor_set=recipe_config.factor_set,
            score_column="recipe_score",
            lookback_momentum_days=config.strategy.lookback_momentum_days,
            lookback_short_days=config.strategy.lookback_short_days,
            lookback_vol_days=config.strategy.lookback_vol_days,
        ),
        portfolio=PortfolioRecipe(
            max_positions=max_positions,
            target_positions=max_positions,
            min_holding_days=config.market.min_position_holding_days,
            lot_size=config.backtest.lot_size,
        ),
        entry=EntryRecipe(
            top_n=config.strategy.buy_top_n,
            min_score=recipe_config.min_score,
            signal_lag_days=1,
            max_daily_buys=recipe_config.max_daily_buys,
        ),
        exit=ExitRecipe(
            profile="configured_exit",
            sell_rules=recipe_config.sell_rules,
            max_daily_sells=recipe_config.max_daily_sells,
        ),
        risk=RiskRecipe(market_gate=recipe_config.market_gate),
        execution=_execution_from_config(config),
        tags=("configured", "jq1"),
        notes=recipe_config.notes,
    )


def _universe_from_config(config: AppConfig) -> UniverseRecipe:
    return UniverseRecipe(
        groups=DEFAULT_RECIPE_GROUPS,
        min_list_days=config.filters.min_list_days,
        min_price=config.filters.min_price,
        min_avg_amount_yuan=config.filters.min_avg_turnover,
        exclude_st=config.filters.exclude_st,
        exclude_suspended=config.filters.exclude_suspended,
    )


def _recipe_min_score(recipe_config: RecipeConfig, default: float) -> float:
    return float(recipe_config.min_score if recipe_config.min_score is not None else default)


def _execution_from_config(config: AppConfig) -> ExecutionRecipe:
    return ExecutionRecipe(
        buy_markup=config.pricing.buy_markup,
        sell_markdown=config.pricing.sell_markdown,
        cancel_if_gap_exceeds=config.pricing.cancel_if_gap_exceeds,
        initial_cash=config.backtest.initial_cash,
        commission_rate=config.backtest.commission_rate,
        stamp_duty_rate=config.backtest.stamp_duty_rate,
    )


def _normalize_names(values: tuple[str, ...], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_name(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_positive_int(value: int, field_name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_positive_float(value: float, field_name: str) -> None:
    if float(value) <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_non_negative_float(value: float, field_name: str) -> None:
    if float(value) < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "strategy-recipe"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value
