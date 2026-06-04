from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


@dataclass(slots=True)
class MarketConfig:
    name: str
    benchmark: str
    max_positions: int
    min_position_holding_days: int


@dataclass(slots=True)
class FilterConfig:
    min_list_days: int
    min_price: float
    min_avg_turnover: float
    exclude_st: bool
    exclude_suspended: bool


@dataclass(slots=True)
class PricingConfig:
    buy_markup: float
    sell_markdown: float
    cancel_if_gap_exceeds: float


@dataclass(slots=True)
class StrategyConfig:
    buy_top_n: int
    sell_top_n: int
    lookback_momentum_days: int
    lookback_short_days: int
    lookback_vol_days: int


@dataclass(slots=True)
class BacktestConfig:
    initial_cash: float = 1000000.0
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.001
    lot_size: int = 100


@dataclass(slots=True)
class SelectionConfig:
    enable_rebound_strategy: bool = True
    buy_momentum_weight: float = 0.40
    buy_liquidity_weight: float = 0.20
    buy_ma20_weight: float = 0.15
    buy_ma10_weight: float = 0.10
    buy_volatility_weight: float = 0.10
    buy_volume_ratio_weight: float = 0.05
    buy_trend_weight: float = 0.15
    buy_pullback_weight: float = 0.15
    buy_reversal_weight: float = 0.05
    sell_momentum_weight: float = 0.45
    sell_ma10_weight: float = 0.25
    sell_ma20_weight: float = 0.20
    sell_volatility_weight: float = 0.10
    min_buy_score: float = 0.60
    rotation_edge: float = 0.25
    sell_health_exit_threshold: float = 0.35
    market_min_breadth: float = 0.35
    buy_min_close_to_ma20: float = -0.03
    buy_max_close_to_ma20: float = 0.08
    buy_min_pullback_from_20d_high: float = -0.15
    buy_max_pullback_from_20d_high: float = -0.05
    buy_min_momentum_5d: float = -0.02
    buy_max_volume_ratio: float = 3.0
    buy_min_amount_ratio_5d: float = 0.80
    buy_max_amount_ratio_5d: float = 1.80
    buy_max_return_1d: float = 0.035
    buy_max_close_to_ma5: float = 0.035
    buy_max_low_to_prev_low: float = 0.04
    buy_min_total_mv_yuan: float = 30000000000.0
    stop_loss_pct: float = 0.08
    take_profit_trigger_pct: float = 0.10
    trailing_stop_drawdown_pct: float = 0.07
    trend_exit_min_holding_days: int = 5
    rotation_min_holding_days: int = 5
    overheat_take_profit_close_to_ma20: float = 0.18
    overheat_min_profit_pct: float = 0.08
    rebound_min_score: float = 0.60
    rebound_max_close_to_ma60_below: float = 0.18
    rebound_min_drawdown_20d: float = -0.08
    rebound_min_drawdown_60d: float = -0.08
    rebound_max_drawdown_60d: float = -0.35
    rebound_max_down_days_10d: int = 7
    rebound_min_return_3d: float = -0.01
    rebound_max_return_3d: float = 0.16
    rebound_max_close_to_ma5: float = 0.12
    rebound_max_close_to_ma20: float = 0.10
    rebound_min_amount_ratio_5d: float = 0.70
    rebound_max_amount_ratio_5d: float = 2.20
    rebound_max_volume_ratio: float = 3.00
    rebound_ideal_max_return_3d: float = 0.08
    rebound_ideal_max_close_to_ma5: float = 0.06
    rebound_ideal_max_close_to_ma20: float = 0.03
    rebound_max_volatility_20d: float = 0.08
    rebound_min_large_net_mf_to_amount: float = -0.03
    rebound_prefer_large_net_mf_to_amount: float = 0.02
    rebound_min_industry_return_3d: float = -0.01
    rebound_prefer_industry_return_3d: float = 0.03
    rebound_min_industry_breadth_20d: float = 0.25
    rebound_prefer_industry_breadth_20d: float = 0.65
    rebound_min_industry_rebound_breadth: float = 0.25
    rebound_prefer_industry_rebound_breadth: float = 0.65
    rebound_market_min_breadth: float = 0.25
    rebound_max_positions: int = 2
    rebound_position_size_multiplier: float = 0.50
    rebound_stop_loss_pct: float = 0.05
    rebound_fast_exit_days: int = 3
    rebound_breakeven_trigger_pct: float = 0.08
    rebound_breakeven_floor_pct: float = 0.01
    rebound_profit_lock_trigger_pct: float = 0.15
    rebound_profit_lock_drawdown_pct: float = 0.06
    rebound_big_profit_trigger_pct: float = 0.25
    rebound_big_profit_drawdown_pct: float = 0.10
    rebound_profit_exit_min_pct: float = 0.15


@dataclass(slots=True)
class RecipeConfig:
    name: str
    enabled: bool = True
    max_positions: int | None = None
    min_score: float | None = None
    market_gate: str = "risk_on"
    factor_set: tuple[str, ...] = ()
    sell_rules: tuple[str, ...] = ()
    max_daily_buys: int | None = None
    max_daily_sells: int | None = None
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class RuntimeConfig:
    paper_start_date: str | None = None
    daily_run_time: str = "18:30"
    timezone: str = "Asia/Shanghai"
    sync_lookback_days: int = 7
    calendar_ahead_days: int = 14


@dataclass(slots=True)
class PathConfig:
    raw_data_dir: Path
    processed_data_dir: Path
    reports_dir: Path
    logs_dir: Path


@dataclass(slots=True)
class AppConfig:
    market: MarketConfig
    filters: FilterConfig
    pricing: PricingConfig
    strategy: StrategyConfig
    backtest: BacktestConfig
    selection: SelectionConfig
    runtime: RuntimeConfig
    paths: PathConfig
    tushare_token: str | None = None
    recipes: tuple[RecipeConfig, ...] = ()

    @property
    def max_positions(self) -> int:
        return self.market.max_positions


def load_env_file(env_path: str | Path) -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_section(data: dict, name: str) -> dict:
    if name not in data:
        raise ValueError(f"Missing config section: {name}")
    return data[name]


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)

    market = _require_section(data, "market")
    filters = _require_section(data, "filters")
    pricing = _require_section(data, "pricing")
    strategy = _require_section(data, "strategy")
    paths = _require_section(data, "paths")
    backtest = data.get("backtest", {})
    selection = data.get("selection", {})
    runtime = data.get("runtime", {})
    recipes = _load_recipe_configs(data.get("recipes", []))

    return AppConfig(
        market=MarketConfig(**market),
        filters=FilterConfig(**filters),
        pricing=PricingConfig(**pricing),
        strategy=StrategyConfig(**strategy),
        backtest=BacktestConfig(**backtest),
        selection=SelectionConfig(**selection),
        runtime=RuntimeConfig(**runtime),
        paths=PathConfig(
            raw_data_dir=Path(paths["raw_data_dir"]),
            processed_data_dir=Path(paths["processed_data_dir"]),
            reports_dir=Path(paths["reports_dir"]),
            logs_dir=Path(paths["logs_dir"]),
        ),
        tushare_token=os.getenv("TUSHARE_TOKEN"),
        recipes=recipes,
    )


def _load_recipe_configs(raw_recipes: list[dict] | tuple[dict, ...]) -> tuple[RecipeConfig, ...]:
    if not isinstance(raw_recipes, list | tuple):
        raise ValueError("Config section recipes must be a TOML array of tables")
    recipes: list[RecipeConfig] = []
    for raw_recipe in raw_recipes:
        if not isinstance(raw_recipe, dict):
            raise ValueError("Each recipe config must be a table")
        recipe = dict(raw_recipe)
        recipe["factor_set"] = _tuple_config_values(recipe.get("factor_set", ()))
        recipe["sell_rules"] = _tuple_config_values(recipe.get("sell_rules", ()))
        recipe["notes"] = _tuple_config_values(recipe.get("notes", ()))
        recipes.append(RecipeConfig(**recipe))
    return tuple(recipes)


def _tuple_config_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise ValueError("Recipe list fields must be strings or arrays of strings")
