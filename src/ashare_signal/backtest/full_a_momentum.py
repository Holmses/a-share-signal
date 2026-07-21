from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
import hashlib
import json
import math

import pandas as pd

from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.backtest.tianzhu9_like import Tianzhu9BacktestResult, Tianzhu9Position, Tianzhu9Trade
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.exit_rules import DEFAULT_EXIT_PROFILE, DEFAULT_FAILURE_EXIT_DAYS
from ashare_signal.strategy.exit_rules import DEFAULT_FAILURE_EXIT_MIN_PEAK_PROFIT_PCT, DEFAULT_HARD_EXIT_DAYS
from ashare_signal.strategy.exit_rules import DEFAULT_VOLUME_STALL_EXIT, DEFAULT_VOLUME_STALL_RATIO
from ashare_signal.strategy.exit_rules import EXIT_PROFILES, LEGACY_EXIT_PROFILE
from ashare_signal.strategy.exit_rules import SLOW_PROFIT_LOCK_PROFILE, TIERED_TRAILING_TAKE_PROFIT_LEVELS
from ashare_signal.strategy.exit_rules import slow_profit_lock_exit_signal
from ashare_signal.strategy.exit_rules import tiered_trailing_take_profit
from ashare_signal.strategy.filters import StrategyPreferenceFilter
from ashare_signal.strategy.recipe import StrategyRecipe
from ashare_signal.strategy.ranking import build_ranking_snapshot
from ashare_signal.strategy.sell_reasons import normalize_sell_reason, sell_reason_counts, summarize_sell_reasons
from ashare_signal.strategy.theme_alerts import build_strong_theme_alerts
from ashare_signal.utils.dates import to_compact_date


DEFAULT_FULL_A_ENABLED_RECIPES = ("momentum_core",)
SUPPORTED_FULL_A_ENTRY_RECIPES = (
    "momentum_core",
    "trend_pullback_overlay",
    "quality_momentum_filter",
    "rebound_bottoming_watch",
)
RANKING_DIAGNOSTIC_VARIANT = "quality_momentum_rank"
RANKING_FACTOR_COLUMNS = (
    "momentum_rank",
    "trend_rank",
    "pullback_rank",
    "liquidity_rank",
    "volatility_rank",
    "moneyflow_rank",
    "industry_rank",
    "rebound_rank",
)
MARKET_RISK_EXIT_MIN_HOLDING_DAYS = 8
MARKET_RISK_EXIT_MIN_5D_LOSS_PCT = 0.02
MARKET_RISK_EXIT_MIN_POSITION_LOSS_PCT = 0.05
STYLE_ROTATION_EXIT_MIN_HOLDING_DAYS = 8
STYLE_ROTATION_EXIT_MIN_5D_LOSS_PCT = 0.02
STYLE_ROTATION_EXIT_MIN_RELATIVE_5D_PCT = 0.04
STYLE_ROTATION_EXIT_MIN_POSITION_LOSS_PCT = 0.05


class FullAMomentumBacktestEngine:
    """Full A-share momentum backtest with market and board-style filters."""

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        top_n: int = 5,
        hold_days: int = 5,
        max_hold_days: int = 10,
        max_positions: int | None = None,
        groups: list[str] | None = None,
        selection_variant: str = "quality_momentum",
        min_avg_amount_yuan: float = 50_000_000.0,
        market_min_breadth: float = 0.50,
        market_min_return_20d: float = 0.0,
        defensive_market_min_breadth: float | None = None,
        defensive_position_size_multiplier: float | None = None,
        aggressive_position_size_multiplier: float = 1.0,
        entry_market_states: list[str] | None = None,
        style_min_breadth: float = 0.48,
        style_min_return_20d: float = -0.01,
        style_score_weight: float = 0.06,
        style_score_weight_active: float | None = None,
        loss_cooldown_days: int = 3,
        stop_loss_pct: float = 0.05,
        take_profit_trigger_pct: float = 0.08,
        trailing_stop_drawdown_pct: float = 0.04,
        hard_exit_days: int | None = DEFAULT_HARD_EXIT_DAYS,
        exit_ma20_break: bool = False,
        exit_failure_days: int | None = DEFAULT_FAILURE_EXIT_DAYS,
        exit_failure_min_peak_profit_pct: float = DEFAULT_FAILURE_EXIT_MIN_PEAK_PROFIT_PCT,
        exit_adaptive_trailing: bool = False,
        exit_atr_multiplier: float = 1.5,
        exit_market_risk: bool = False,
        exit_style_rotation: bool = False,
        exit_industry_weak: bool = False,
        exit_relative_weak: bool = False,
        exit_relative_weak_5d_pct: float = 0.04,
        exit_relative_weak_20d_pct: float = 0.08,
        exit_volume_stall: bool = DEFAULT_VOLUME_STALL_EXIT,
        exit_volume_stall_ratio: float = DEFAULT_VOLUME_STALL_RATIO,
        exit_upper_shadow: bool = False,
        exit_upper_shadow_pct: float = 0.45,
        exit_high_drawdown_pct: float | None = None,
        exit_chandelier_atr_multiplier: float | None = None,
        exit_trend_decay: bool = False,
        exit_winner_hard_exit_bypass_peak_pct: float | None = None,
        exit_risk_off_failed_hard_exit_days: int | None = None,
        exit_profile: str = DEFAULT_EXIT_PROFILE,
        enabled_recipes: list[str] | None = None,
        overlay_recipes: list[str] | None = None,
        quality_filter_enabled: bool = False,
        quality_filter_min_score: float = 0.40,
        quality_filter_score_bonus: float = 0.02,
        overlay_score_bonus: float = 0.03,
        overlay_max_daily_candidates: int = 2,
        ml_predictions_path: Path | None = None,
        ml_min_trend_prob: float | None = None,
        ml_score_weight: float = 0.0,
        theme_buy_point_overlay: bool = False,
        theme_overlay_max_daily_candidates: int = 1,
        theme_overlay_score_bonus: float = 0.04,
        theme_overlay_position_size_multiplier: float = 0.50,
        lot_size: int | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.top_n = max(int(top_n), 1)
        self.hold_days = max(int(hold_days), 1)
        self.max_hold_days = max(int(max_hold_days), self.hold_days)
        self.max_positions = max(int(max_positions or config.market.max_positions), 1)
        self.groups = groups or ["main", "chinext", "star"]
        self.selection_variant = selection_variant
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.market_min_breadth = float(market_min_breadth)
        self.market_min_return_20d = float(market_min_return_20d)
        config_selection = getattr(config, "selection", None)
        default_defensive_breadth = getattr(
            config_selection, "defensive_market_min_breadth", self.market_min_breadth
        )
        default_defensive_multiplier = getattr(config_selection, "defensive_position_size_multiplier", 0.25)
        self.defensive_market_min_breadth = float(
            default_defensive_breadth if defensive_market_min_breadth is None else defensive_market_min_breadth
        )
        self.defensive_position_size_multiplier = float(
            default_defensive_multiplier
            if defensive_position_size_multiplier is None
            else defensive_position_size_multiplier
        )
        self.aggressive_position_size_multiplier = min(max(float(aggressive_position_size_multiplier), 0.0), 1.0)
        self.entry_market_states = _normalize_market_states(
            entry_market_states,
            default=("normal", "aggressive", "defensive"),
        )
        self.style_min_breadth = float(style_min_breadth)
        self.style_min_return_20d = float(style_min_return_20d)
        self.style_score_weight = float(style_score_weight)
        self.style_score_weight_active = (
            float(style_score_weight_active) if style_score_weight_active is not None else None
        )
        self.loss_cooldown_days = max(int(loss_cooldown_days), 0)
        self.take_profit_trigger_pct = float(take_profit_trigger_pct)
        if exit_profile not in EXIT_PROFILES:
            raise ValueError(f"exit_profile must be one of: {', '.join(EXIT_PROFILES)}")
        self.exit_profile = exit_profile
        self.hard_exit_days = max(int(hard_exit_days), 1) if hard_exit_days is not None else None
        self.exit_ma20_break = bool(exit_ma20_break)
        self.exit_failure_days = max(int(exit_failure_days), 1) if exit_failure_days else None
        self.exit_failure_min_peak_profit_pct = float(exit_failure_min_peak_profit_pct)
        self.exit_adaptive_trailing = bool(exit_adaptive_trailing)
        self.exit_atr_multiplier = float(exit_atr_multiplier)
        self.exit_market_risk = bool(exit_market_risk)
        self.exit_style_rotation = bool(exit_style_rotation)
        self.exit_industry_weak = bool(exit_industry_weak)
        self.exit_relative_weak = bool(exit_relative_weak)
        self.exit_relative_weak_5d_pct = float(exit_relative_weak_5d_pct)
        self.exit_relative_weak_20d_pct = float(exit_relative_weak_20d_pct)
        self.exit_volume_stall = bool(exit_volume_stall)
        self.exit_volume_stall_ratio = float(exit_volume_stall_ratio)
        self.exit_upper_shadow = bool(exit_upper_shadow)
        self.exit_upper_shadow_pct = float(exit_upper_shadow_pct)
        self.exit_high_drawdown_pct = (
            min(max(float(exit_high_drawdown_pct), 0.0), 1.0)
            if exit_high_drawdown_pct is not None
            else None
        )
        self.exit_chandelier_atr_multiplier = (
            max(float(exit_chandelier_atr_multiplier), 0.0)
            if exit_chandelier_atr_multiplier is not None
            else None
        )
        self.exit_trend_decay = bool(exit_trend_decay)
        self.exit_winner_hard_exit_bypass_peak_pct = (
            float(exit_winner_hard_exit_bypass_peak_pct)
            if exit_winner_hard_exit_bypass_peak_pct is not None
            else None
        )
        self.exit_risk_off_failed_hard_exit_days = (
            max(int(exit_risk_off_failed_hard_exit_days), 1)
            if exit_risk_off_failed_hard_exit_days is not None
            else None
        )
        self.exit_strategy = self._exit_strategy_name()
        self.enabled_recipes = _normalize_recipe_names(enabled_recipes, DEFAULT_FULL_A_ENABLED_RECIPES)
        self.overlay_recipes = _normalize_recipe_names(overlay_recipes, ())
        self._validate_candidate_recipes()
        self.quality_filter_enabled = bool(quality_filter_enabled or "quality_momentum_filter" in self.enabled_recipes)
        self.quality_filter_min_score = float(quality_filter_min_score)
        self.quality_filter_score_bonus = float(quality_filter_score_bonus)
        self.overlay_score_bonus = float(overlay_score_bonus)
        self.overlay_max_daily_candidates = max(int(overlay_max_daily_candidates), 0)
        self.ml_predictions_path = ml_predictions_path
        self.ml_min_trend_prob = float(ml_min_trend_prob) if ml_min_trend_prob is not None else None
        self.ml_score_weight = float(ml_score_weight)
        self.ml_prediction_map = _load_ml_prediction_map(ml_predictions_path) if ml_predictions_path else {}
        self.theme_buy_point_overlay = bool(theme_buy_point_overlay)
        self.theme_overlay_max_daily_candidates = max(int(theme_overlay_max_daily_candidates), 0)
        self.theme_overlay_score_bonus = float(theme_overlay_score_bonus)
        self.theme_overlay_position_size_multiplier = min(
            max(float(theme_overlay_position_size_multiplier), 0.0),
            1.0,
        )
        self.lot_size = int(lot_size or config.backtest.lot_size)

    @classmethod
    def from_recipe(
        cls,
        *,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        recipe: StrategyRecipe,
        enabled_recipes: list[str] | None = None,
        overlay_recipes: list[str] | None = None,
        quality_filter_enabled: bool = False,
        quality_filter_min_score: float = 0.40,
        quality_filter_score_bonus: float = 0.02,
        overlay_score_bonus: float = 0.03,
        overlay_max_daily_candidates: int = 2,
        ml_predictions_path: Path | None = None,
        ml_min_trend_prob: float | None = None,
        ml_score_weight: float = 0.0,
        theme_buy_point_overlay: bool = False,
        theme_overlay_max_daily_candidates: int = 1,
        theme_overlay_score_bonus: float = 0.04,
        theme_overlay_position_size_multiplier: float = 0.50,
        exit_market_risk: bool | None = None,
        exit_style_rotation: bool | None = None,
        exit_high_drawdown_pct: float | None = None,
        exit_chandelier_atr_multiplier: float | None = None,
        exit_trend_decay: bool = False,
        exit_winner_hard_exit_bypass_peak_pct: float | None = None,
        exit_risk_off_failed_hard_exit_days: int | None = None,
        style_score_weight_active: float | None = None,
        defensive_market_min_breadth: float | None = None,
        defensive_position_size_multiplier: float | None = None,
        aggressive_position_size_multiplier: float = 1.0,
        entry_market_states: list[str] | None = None,
    ) -> "FullAMomentumBacktestEngine":
        if recipe.family != "full_a_momentum":
            raise ValueError(f"recipe family must be 'full_a_momentum', got {recipe.family!r}")
        return cls(
            config=config,
            repository=repository,
            base_dir=base_dir,
            top_n=recipe.entry.top_n,
            hold_days=recipe.portfolio.min_holding_days,
            max_hold_days=_value_or_default(
                recipe.portfolio.max_holding_days,
                recipe.portfolio.min_holding_days,
            ),
            max_positions=recipe.portfolio.max_positions,
            groups=list(recipe.universe.groups),
            selection_variant=recipe.alpha.variant,
            min_avg_amount_yuan=recipe.universe.min_avg_amount_yuan,
            market_min_breadth=_value_or_default(recipe.risk.market_min_breadth, 0.50),
            market_min_return_20d=_value_or_default(recipe.risk.market_min_return_20d, 0.0),
            defensive_market_min_breadth=defensive_market_min_breadth,
            defensive_position_size_multiplier=defensive_position_size_multiplier,
            aggressive_position_size_multiplier=aggressive_position_size_multiplier,
            entry_market_states=entry_market_states,
            style_min_breadth=_value_or_default(recipe.risk.style_min_breadth, 0.48),
            style_min_return_20d=_value_or_default(recipe.risk.style_min_return_20d, -0.01),
            style_score_weight=_value_or_default(recipe.risk.style_score_weight, 0.06),
            style_score_weight_active=style_score_weight_active,
            loss_cooldown_days=recipe.risk.loss_cooldown_days,
            stop_loss_pct=_value_or_default(recipe.exit.stop_loss_pct, 0.05),
            take_profit_trigger_pct=_value_or_default(recipe.exit.take_profit_trigger_pct, 0.08),
            trailing_stop_drawdown_pct=_value_or_default(recipe.exit.trailing_stop_drawdown_pct, 0.04),
            hard_exit_days=recipe.exit.hard_exit_days,
            exit_ma20_break=recipe.exit.ma20_break,
            exit_failure_days=recipe.exit.failure_days,
            exit_failure_min_peak_profit_pct=recipe.exit.failure_min_peak_profit_pct,
            exit_adaptive_trailing=recipe.exit.adaptive_trailing,
            exit_atr_multiplier=recipe.exit.atr_multiplier,
            exit_market_risk=recipe.exit.market_risk_exit if exit_market_risk is None else exit_market_risk,
            exit_style_rotation=False if exit_style_rotation is None else exit_style_rotation,
            exit_industry_weak=recipe.exit.industry_weak_exit,
            exit_relative_weak=recipe.exit.relative_weak_exit,
            exit_relative_weak_5d_pct=recipe.exit.relative_weak_5d_pct,
            exit_relative_weak_20d_pct=recipe.exit.relative_weak_20d_pct,
            exit_volume_stall=recipe.exit.volume_stall_exit,
            exit_volume_stall_ratio=recipe.exit.volume_stall_ratio,
            exit_upper_shadow=recipe.exit.upper_shadow_exit,
            exit_upper_shadow_pct=recipe.exit.upper_shadow_pct,
            exit_high_drawdown_pct=exit_high_drawdown_pct,
            exit_chandelier_atr_multiplier=exit_chandelier_atr_multiplier,
            exit_trend_decay=exit_trend_decay,
            exit_winner_hard_exit_bypass_peak_pct=exit_winner_hard_exit_bypass_peak_pct,
            exit_risk_off_failed_hard_exit_days=exit_risk_off_failed_hard_exit_days,
            exit_profile=recipe.exit.profile,
            enabled_recipes=enabled_recipes,
            overlay_recipes=overlay_recipes,
            quality_filter_enabled=quality_filter_enabled,
            quality_filter_min_score=quality_filter_min_score,
            quality_filter_score_bonus=quality_filter_score_bonus,
            overlay_score_bonus=overlay_score_bonus,
            overlay_max_daily_candidates=overlay_max_daily_candidates,
            ml_predictions_path=ml_predictions_path,
            ml_min_trend_prob=ml_min_trend_prob,
            ml_score_weight=ml_score_weight,
            theme_buy_point_overlay=theme_buy_point_overlay,
            theme_overlay_max_daily_candidates=theme_overlay_max_daily_candidates,
            theme_overlay_score_bonus=theme_overlay_score_bonus,
            theme_overlay_position_size_multiplier=theme_overlay_position_size_multiplier,
            lot_size=recipe.portfolio.lot_size,
        )

    def _validate_candidate_recipes(self) -> None:
        configured = set(self.enabled_recipes) | set(self.overlay_recipes)
        unsupported = sorted(configured - set(SUPPORTED_FULL_A_ENTRY_RECIPES))
        if unsupported:
            raise ValueError(f"Unsupported Full A candidate recipe(s): {', '.join(unsupported)}")
        if "rebound_bottoming_watch" in configured:
            raise ValueError("rebound_bottoming_watch is research-only and cannot be used for production buys")

    def _exit_strategy_name(self) -> str:
        if self.exit_profile == SLOW_PROFIT_LOCK_PROFILE:
            return SLOW_PROFIT_LOCK_PROFILE
        parts = ["tiered_trailing_take_profit"]
        if self.hard_exit_days is None:
            parts.append("no_time_exit")
        else:
            parts.append(f"hard_exit_{self.hard_exit_days}d")
        if self.exit_ma20_break:
            parts.append("ma20_break")
        if self.exit_failure_days is not None:
            parts.append(f"failure_{self.exit_failure_days}d")
        if self.exit_adaptive_trailing:
            parts.append(f"adaptive_atr_{self.exit_atr_multiplier:g}x")
        if self.exit_market_risk:
            parts.append("market_risk_exit")
        if self.exit_style_rotation:
            parts.append("style_rotation_exit")
        if self.exit_industry_weak:
            parts.append("industry_weak_exit")
        if self.exit_relative_weak:
            parts.append("relative_weak_exit")
        if self.exit_volume_stall:
            parts.append("volume_stall_exit")
        if self.exit_upper_shadow:
            parts.append("upper_shadow_exit")
        if self.exit_high_drawdown_pct is not None:
            parts.append(f"high_drawdown_{self.exit_high_drawdown_pct:g}")
        if self.exit_chandelier_atr_multiplier is not None:
            parts.append(f"chandelier_atr_{self.exit_chandelier_atr_multiplier:g}x")
        if self.exit_trend_decay:
            parts.append("trend_decay_exit")
        if self.exit_winner_hard_exit_bypass_peak_pct is not None:
            parts.append(f"winner_no_hard_{self.exit_winner_hard_exit_bypass_peak_pct:g}")
        if self.exit_risk_off_failed_hard_exit_days is not None:
            parts.append(f"risk_off_failed_hard_{self.exit_risk_off_failed_hard_exit_days}d")
        return "_".join(parts)

    def _exit_slug(self) -> str:
        if self.exit_profile == SLOW_PROFIT_LOCK_PROFILE:
            return "slow-profit-lock"
        slug = f"tiered-trailing-hard{self.hard_exit_days or 'none'}"
        if self.exit_ma20_break:
            slug += "-ma20"
        if self.exit_failure_days is not None:
            failure_pct = _slug_float(self.exit_failure_min_peak_profit_pct)
            slug += f"-fail{self.exit_failure_days}d{failure_pct}"
        if self.exit_adaptive_trailing:
            slug += f"-atr{_slug_float(self.exit_atr_multiplier)}"
        if self.exit_market_risk:
            slug += "-mktrisk"
        if self.exit_style_rotation:
            slug += "-stylerot"
        if self.exit_industry_weak:
            slug += "-indweak"
        if self.exit_relative_weak:
            slug += (
                f"-relweak{_slug_float(self.exit_relative_weak_5d_pct)}"
                f"x{_slug_float(self.exit_relative_weak_20d_pct)}"
            )
        if self.exit_volume_stall:
            slug += f"-volstall{_slug_float(self.exit_volume_stall_ratio)}"
        if self.exit_upper_shadow:
            slug += f"-shadow{_slug_float(self.exit_upper_shadow_pct)}"
        if self.exit_high_drawdown_pct is not None:
            slug += f"-highdd{_slug_float(self.exit_high_drawdown_pct)}"
        if self.exit_chandelier_atr_multiplier is not None:
            slug += f"-chandelier{_slug_float(self.exit_chandelier_atr_multiplier)}"
        if self.exit_trend_decay:
            slug += "-trenddecay"
        if self.exit_winner_hard_exit_bypass_peak_pct is not None:
            slug += f"-winnohard{_slug_float(self.exit_winner_hard_exit_bypass_peak_pct)}"
        if self.exit_risk_off_failed_hard_exit_days is not None:
            slug += f"-roffhard{self.exit_risk_off_failed_hard_exit_days}"
        return slug

    def _candidate_recipe_slug(self) -> str:
        if (
            self.enabled_recipes == DEFAULT_FULL_A_ENABLED_RECIPES
            and not self.overlay_recipes
            and not self.quality_filter_enabled
            and not self.ml_prediction_map
            and self.ml_min_trend_prob is None
            and self.ml_score_weight == 0
            and not self.theme_buy_point_overlay
            and self.entry_market_states == ("normal", "aggressive", "defensive")
            and self.aggressive_position_size_multiplier == 1.0
        ):
            return ""
        parts = ["recipes", "-".join(self.enabled_recipes)]
        if self.overlay_recipes:
            parts.append("overlay-" + "-".join(self.overlay_recipes))
        if self.quality_filter_enabled:
            parts.append("quality")
        if self.ml_predictions_path:
            parts.append(f"mlp{_slug_float(self.ml_min_trend_prob or 0.0)}")
            if self.ml_score_weight:
                parts.append(f"mlw{_slug_float(self.ml_score_weight)}")
        if self.theme_buy_point_overlay:
            parts.append(
                f"themebuy{self.theme_overlay_max_daily_candidates}"
                f"x{_slug_float(self.theme_overlay_position_size_multiplier)}"
            )
        if self.entry_market_states != ("normal", "aggressive", "defensive"):
            parts.append("entry-" + "-".join(self.entry_market_states))
        if self.aggressive_position_size_multiplier != 1.0:
            parts.append(f"aggrx{_slug_float(self.aggressive_position_size_multiplier)}")
        return "-" + "-".join(parts).replace("_", "-")

    def run(self, start_date: date | None = None, end_date: date | None = None) -> Tianzhu9BacktestResult:
        cached_dates = self.repository.complete_daily_cache_dates()
        if not cached_dates:
            raise ValueError("Daily Tushare cache is empty. Run `ashare-signal sync-tushare` first.")

        resolved_end = self._resolve_cached_end(cached_dates, end_date)
        resolved_start = self._resolve_cached_start(cached_dates, start_date, resolved_end)
        start_index = cached_dates.index(resolved_start)
        end_index = cached_dates.index(resolved_end)
        required_history = SelectionEventStudyEngine.minimum_backtest_history_trade_days()
        if start_index < required_history:
            suggested_sync_start = to_compact_date(
                SelectionEventStudyEngine.recommended_sync_start_date(
                    repository=self.repository,
                    target_date=resolved_start,
                    prior_trade_days=required_history,
                )
            )
            raise ValueError(
                "Full A momentum backtest needs at least "
                f"{required_history} complete trade days before start date {resolved_start}. "
                f"Sync from {suggested_sync_start} or earlier and rerun."
            )
        trade_dates = cached_dates[start_index : end_index + 1]
        if len(trade_dates) < 2:
            raise ValueError("Full A momentum backtest requires at least two cached trade dates.")

        feature_dates = cached_dates[
            max(0, start_index - SelectionEventStudyEngine.factor_history_trade_days()) : end_index + 1
        ]
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=self.top_n,
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=[self.selection_variant],
            horizons=[1],
        )
        factor_frame = study_engine._build_factor_frame(feature_dates)
        factor_date_keys = factor_frame["trade_date"].astype(str)
        factor_date_indices = factor_date_keys.groupby(factor_date_keys, sort=False).indices
        price_map = study_engine._load_price_map(trade_dates)

        initial_cash = float(self.config.backtest.initial_cash)
        cash = initial_cash
        positions: dict[str, Tianzhu9Position] = {}
        trades: list[Tianzhu9Trade] = []
        equity_rows: list[dict] = []
        total_traded_value = 0.0
        loss_cooldown_until: dict[str, int] = {}
        exit_diagnostics = {"fallback_feature_uses": 0, "missing_feature_days": 0}

        for trade_offset, trade_date in enumerate(trade_dates):
            trade_index = start_index + trade_offset
            signal_trade_date = cached_dates[trade_index - 1]
            day_prices = price_map.get(trade_date, pd.DataFrame())
            if day_prices.empty:
                continue

            signal_indices = factor_date_indices.get(signal_trade_date)
            signal_frame = (
                factor_frame.iloc[signal_indices].copy()
                if signal_indices is not None
                else factor_frame.iloc[0:0].copy()
            )
            style_state = self._market_style_state(signal_frame)
            risk_off = bool(style_state["market_risk_off"])
            market_state = str(style_state["market_state"])
            eligible_groups = set(style_state["eligible_groups"])
            selected = self._select_candidates_from_recipes(
                signal_frame=signal_frame,
                style_state=style_state,
                eligible_groups=eligible_groups,
                excluded_symbols={
                    symbol
                    for symbol, cooldown_until in loss_cooldown_until.items()
                    if cooldown_until >= trade_index
                },
                risk_off=risk_off,
                market_state=market_state,
            )
            selected_symbols = {row["ts_code"] for row in selected}

            sell_cash_box = {"cash": cash}
            total_traded_value += self._execute_sells(
                trade_date=trade_date,
                trade_index=trade_index,
                prices=day_prices,
                factor_frame=signal_frame,
                signal_trade_date=signal_trade_date,
                positions=positions,
                selected_symbols=selected_symbols,
                eligible_groups=eligible_groups,
                risk_off=risk_off,
                market_state=market_state,
                trades=trades,
                cash_ref=sell_cash_box,
                loss_cooldown_until=loss_cooldown_until,
                fallback_price_map=price_map,
                fallback_trade_dates=trade_dates,
                exit_diagnostics=exit_diagnostics,
            )
            cash = sell_cash_box["cash"]

            open_equity = self._mark_to_market_equity(cash, positions, day_prices, "open")
            buy_cash_box = {"cash": cash}
            total_traded_value += self._execute_buys(
                trade_date=trade_date,
                signal_trade_date=signal_trade_date,
                trade_index=trade_index,
                prices=day_prices,
                candidates=selected,
                open_equity=open_equity,
                positions=positions,
                trades=trades,
                cash_ref=buy_cash_box,
            )
            cash = buy_cash_box["cash"]

            self._update_position_highs(positions=positions, prices=day_prices)
            close_equity = self._mark_to_market_equity(cash, positions, day_prices, "close")
            equity_rows.append(
                {
                    "trade_date": trade_date,
                    "equity": close_equity,
                    "cash": cash,
                    "position_count": len(positions),
                    "signal_trade_date": signal_trade_date,
                    "selected": ",".join(row["ts_code"] for row in selected),
                    "market_breadth": style_state["market_breadth"],
                    "market_return_20d": style_state["market_return_20d"],
                    "market_source": style_state["market_source"],
                    "benchmark_close_to_ma20": style_state["benchmark_close_to_ma20"],
                    "eligible_groups": ",".join(sorted(eligible_groups)),
                    "risk_off": risk_off,
                    "market_state": market_state,
                }
            )

        equity_frame = pd.DataFrame(equity_rows)
        if equity_frame.empty:
            raise ValueError("Full A momentum backtest produced no equity rows.")

        returns = equity_frame["equity"].pct_change().fillna(0.0)
        ending_equity = float(equity_frame["equity"].iloc[-1])
        total_return = ending_equity / initial_cash - 1.0
        annual_return = (ending_equity / initial_cash) ** (252 / max(len(equity_frame), 1)) - 1.0
        drawdowns = equity_frame["equity"] / equity_frame["equity"].cummax() - 1.0
        max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0
        sharpe = 0.0
        if returns.std(ddof=0) > 0:
            sharpe = float((returns.mean() / returns.std(ddof=0)) * math.sqrt(252))
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else None
        average_equity = float(equity_frame["equity"].mean()) if not equity_frame.empty else initial_cash
        turnover = float(total_traded_value / average_equity) if average_equity > 0 else 0.0
        sell_trades = [trade for trade in trades if trade.action == "SELL"]
        winning_trades = [trade for trade in sell_trades if trade.pnl is not None and trade.pnl > 0]
        win_rate = float(len(winning_trades) / len(sell_trades)) if sell_trades else 0.0
        pnl_values = pd.Series(
            [float(trade.pnl) for trade in sell_trades if trade.pnl is not None],
            dtype=float,
        )
        positive_pnl = pnl_values.loc[pnl_values > 0]
        negative_pnl = pnl_values.loc[pnl_values < 0]
        average_profit = float(positive_pnl.mean()) if not positive_pnl.empty else None
        average_loss = float(negative_pnl.mean()) if not negative_pnl.empty else None
        payoff_ratio = (
            float(average_profit / abs(average_loss))
            if average_profit is not None and average_loss not in (None, 0.0)
            else None
        )
        profit_factor = (
            float(positive_pnl.sum() / abs(negative_pnl.sum()))
            if not negative_pnl.empty and float(negative_pnl.sum()) != 0.0
            else None
        )
        worst_trade_pnl = float(pnl_values.min()) if not pnl_values.empty else None
        bottom_10_total_pnl = float(pnl_values.nsmallest(10).sum()) if not pnl_values.empty else 0.0
        loss_capital_days = int(
            sum(
                int(trade.holding_days or 0)
                for trade in sell_trades
                if trade.pnl is not None and trade.pnl <= 0
            )
        )

        reports_dir = self.base_dir / self.config.paths.reports_dir / "backtests"
        reports_dir.mkdir(parents=True, exist_ok=True)
        groups_slug = "-".join(self.groups)
        filter_slug = (
            f"mb{self.market_min_breadth:.2f}-mr{self.market_min_return_20d:.2f}-"
            f"db{self.defensive_market_min_breadth:.2f}-dm{self.defensive_position_size_multiplier:.2f}-"
            f"sb{self.style_min_breadth:.2f}-sr{self.style_min_return_20d:.2f}-"
            f"sw{self.style_score_weight:.2f}"
            + (f"-swa{self.style_score_weight_active:.2f}" if self.style_score_weight_active is not None else "")
        ).replace("-", "m").replace(".", "p")
        stem = (
            f"full-a-momentum-{self.selection_variant}-top{self.top_n}-"
            f"{groups_slug}{self._candidate_recipe_slug()}-exit-{self._exit_slug()}-filter-"
            f"{filter_slug}-{resolved_start}-{resolved_end}"
        )
        stem = _bounded_output_stem(stem, resolved_start=resolved_start, resolved_end=resolved_end)
        summary_path = reports_dir / f"{stem}-summary.json"
        equity_curve_path = reports_dir / f"{stem}-equity.csv"
        trade_log_path = reports_dir / f"{stem}-trades.csv"

        equity_frame.to_csv(equity_curve_path, index=False)
        trade_columns = list(Tianzhu9Trade.__dataclass_fields__.keys())
        trade_frame = pd.DataFrame([asdict(trade) for trade in trades], columns=trade_columns)
        trade_frame.to_csv(trade_log_path, index=False)
        sell_frame = trade_frame.loc[trade_frame["action"] == "SELL"].copy()
        buy_frame = trade_frame.loc[trade_frame["action"] == "BUY"].copy()
        summary_payload = {
            "strategy": "full_a_momentum",
            "selection_variant": self.selection_variant,
            "start_trade_date": resolved_start,
            "end_trade_date": resolved_end,
            "signal_lag_days": 1,
            "top_n": self.top_n,
            "groups": self.groups,
            "hold_days": self.hold_days,
            "max_hold_days": self.max_hold_days,
            "exit_strategy": self.exit_strategy,
            "exit_strategy_note": (
                (
                    "slow profit lock: delayed trailing take-profit, MA20/MA60/style weakness exits, "
                    + (
                        "and no fixed max-hold cap"
                        if self.hard_exit_days is None
                        else f"and {self.hard_exit_days}-day cap"
                    )
                )
                if self.exit_profile == SLOW_PROFIT_LOCK_PROFILE
                else "tiered trailing take-profit; no hard stop-loss, no fixed max-hold exit"
                if self.hard_exit_days is None
                else f"tiered trailing take-profit plus hard exit after {self.hard_exit_days} trade days"
            ),
            "exit_profile": self.exit_profile,
            "hard_exit_days": self.hard_exit_days,
            "exit_rules": {
                "ma20_break": self.exit_ma20_break,
                "failure_days": self.exit_failure_days,
                "failure_min_peak_profit_pct": self.exit_failure_min_peak_profit_pct,
                "adaptive_trailing": self.exit_adaptive_trailing,
                "atr_multiplier": self.exit_atr_multiplier,
                "market_risk_exit": self.exit_market_risk,
                "style_rotation_exit": self.exit_style_rotation,
                "industry_weak_exit": self.exit_industry_weak,
                "relative_weak_exit": self.exit_relative_weak,
                "relative_weak_5d_pct": self.exit_relative_weak_5d_pct,
                "relative_weak_20d_pct": self.exit_relative_weak_20d_pct,
                "volume_stall_exit": self.exit_volume_stall,
                "volume_stall_ratio": self.exit_volume_stall_ratio,
                "upper_shadow_exit": self.exit_upper_shadow,
                "upper_shadow_pct": self.exit_upper_shadow_pct,
                "high_drawdown_pct": self.exit_high_drawdown_pct,
                "chandelier_atr_multiplier": self.exit_chandelier_atr_multiplier,
                "trend_decay_exit": self.exit_trend_decay,
                "winner_hard_exit_bypass_peak_pct": self.exit_winner_hard_exit_bypass_peak_pct,
                "risk_off_failed_hard_exit_days": self.exit_risk_off_failed_hard_exit_days,
            },
            "max_positions": self.max_positions,
            "candidate_recipes": {
                "enabled_recipes": list(self.enabled_recipes),
                "overlay_recipes": list(self.overlay_recipes),
                "quality_filter_enabled": self.quality_filter_enabled,
                "quality_filter_min_score": self.quality_filter_min_score,
                "quality_filter_score_bonus": self.quality_filter_score_bonus,
                "overlay_score_bonus": self.overlay_score_bonus,
                "overlay_max_daily_candidates": self.overlay_max_daily_candidates,
                "ranking_diagnostic_variant": RANKING_DIAGNOSTIC_VARIANT,
                "ml_predictions_path": str(self.ml_predictions_path) if self.ml_predictions_path else None,
                "ml_min_trend_prob": self.ml_min_trend_prob,
                "ml_score_weight": self.ml_score_weight,
                "theme_buy_point_overlay": self.theme_buy_point_overlay,
                "theme_overlay_max_daily_candidates": self.theme_overlay_max_daily_candidates,
                "theme_overlay_score_bonus": self.theme_overlay_score_bonus,
                "theme_overlay_position_size_multiplier": self.theme_overlay_position_size_multiplier,
            },
            "min_avg_amount_yuan": self.min_avg_amount_yuan,
            "market_filter": {
                "market_min_breadth": self.market_min_breadth,
                "market_min_return_20d": self.market_min_return_20d,
                "defensive_market_min_breadth": self.defensive_market_min_breadth,
                "defensive_position_size_multiplier": self.defensive_position_size_multiplier,
                "aggressive_position_size_multiplier": self.aggressive_position_size_multiplier,
                "entry_market_states": list(self.entry_market_states),
                "style_min_breadth": self.style_min_breadth,
                "style_min_return_20d": self.style_min_return_20d,
                "style_score_weight": self.style_score_weight,
                "style_score_weight_active": self.style_score_weight_active,
            },
            "enhanced_data": {
                "benchmark_index": self.config.market.benchmark,
                "market_source_counts": equity_frame["market_source"].value_counts(dropna=False).to_dict(),
                "sw_industry_rows": int(factor_frame["sw_l1_name"].notna().sum())
                if "sw_l1_name" in factor_frame.columns
                else 0,
                "financial_data_rows": int(factor_frame["financial_data_available"].fillna(False).sum())
                if "financial_data_available" in factor_frame.columns
                else 0,
            },
            "risk_off_days": int(equity_frame["risk_off"].sum()),
            "exit_data_diagnostics": exit_diagnostics,
            "sell_reason_counts": sell_reason_counts(sell_trades),
            "sell_reason_summary": summarize_sell_reasons(sell_trades),
            "entry_recipe_counts": _value_counts(buy_frame, "entry_recipe"),
            "entry_recipe_summary": _summarize_trade_groups(trade_frame, "entry_recipe"),
            "market_state_counts": _value_counts(equity_frame, "market_state"),
            "market_state_summary": _summarize_trade_groups(trade_frame, "market_state"),
            "average_holding_days": _mean_numeric(sell_frame.get("holding_days")),
            "average_profit_holding_days": _mean_numeric(
                sell_frame.loc[pd.to_numeric(sell_frame["pnl"], errors="coerce") > 0, "holding_days"]
                if not sell_frame.empty
                else None
            ),
            "average_loss_holding_days": _mean_numeric(
                sell_frame.loc[pd.to_numeric(sell_frame["pnl"], errors="coerce") <= 0, "holding_days"]
                if not sell_frame.empty
                else None
            ),
            "average_position_count": float(equity_frame["position_count"].mean()),
            "average_invested_ratio": float((1.0 - equity_frame["cash"] / equity_frame["equity"]).mean()),
            "initial_cash": initial_cash,
            "ending_equity": ending_equity,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "calmar": calmar,
            "turnover": turnover,
            "trade_count": len(trades),
            "sell_trade_count": len(sell_trades),
            "win_rate": win_rate,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "payoff_ratio": payoff_ratio,
            "profit_factor": profit_factor,
            "worst_trade_pnl": worst_trade_pnl,
            "bottom_10_total_pnl": bottom_10_total_pnl,
            "loss_capital_days": loss_capital_days,
            "equity_curve_path": str(equity_curve_path),
            "trade_log_path": str(trade_log_path),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return Tianzhu9BacktestResult(
            start_trade_date=resolved_start,
            end_trade_date=resolved_end,
            signal_lag_days=1,
            top_n=self.top_n,
            hold_days=self.hold_days,
            initial_cash=initial_cash,
            ending_equity=ending_equity,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            turnover=turnover,
            trade_count=len(trades),
            sell_trade_count=len(sell_trades),
            win_rate=win_rate,
            execution_mode="limit-swing",
            extend_on_repeat=True,
            max_hold_days=self.max_hold_days,
            equity_curve_path=equity_curve_path,
            summary_path=summary_path,
            trade_log_path=trade_log_path,
        )

    def _resolve_cached_end(self, cached_dates: list[str], end_date: date | None) -> str:
        if end_date is None:
            return cached_dates[-1]
        requested = to_compact_date(end_date)
        eligible = [value for value in cached_dates if value <= requested]
        if not eligible:
            raise ValueError(f"No cached trade date found on or before {requested}")
        return eligible[-1]

    def _resolve_cached_start(self, cached_dates: list[str], start_date: date | None, end_date: str) -> str:
        if start_date is None:
            end_index = cached_dates.index(end_date)
            return cached_dates[max(1, end_index - 252)]
        requested = to_compact_date(start_date)
        eligible = [value for value in cached_dates if value >= requested and value <= end_date]
        if not eligible:
            raise ValueError(f"No cached trade date found on or after {requested}")
        return eligible[0]

    def _market_style_state(self, signal_frame: pd.DataFrame) -> dict:
        if signal_frame.empty:
            return {
                "market_breadth": 0.0,
                "market_return_20d": -1.0,
                "market_risk_off": True,
                "eligible_groups": [],
                "group_scores": {},
                "market_source": "empty",
                "benchmark_close_to_ma20": None,
                "market_state": "risk_off",
                "position_size_multiplier": 0.0,
                "defensive_allowed": False,
                "strongest_style_group": None,
            }
        market_breadth = float((signal_frame["close"] >= signal_frame["ma_20"]).mean())
        benchmark_return = signal_frame.get("benchmark_return_20d")
        benchmark_close_to_ma20_series = signal_frame.get("benchmark_close_to_ma20")
        benchmark_close_to_ma20 = None
        if benchmark_return is not None and benchmark_return.notna().any():
            market_return_20d = float(benchmark_return.dropna().iloc[-1])
            market_source = "benchmark_index"
            if benchmark_close_to_ma20_series is not None and benchmark_close_to_ma20_series.notna().any():
                benchmark_close_to_ma20 = float(benchmark_close_to_ma20_series.dropna().iloc[-1])
        else:
            market_return_20d = float(signal_frame["return_20d"].median())
            market_source = "stock_median"
        style_column = "style_group" if "style_group" in signal_frame.columns else "group"
        eligible_groups = []
        group_scores: dict[str, float] = {}
        group_metrics: dict[str, dict[str, float]] = {}
        for group, group_frame in signal_frame.groupby(style_column):
            breadth = float((group_frame["close"] >= group_frame["ma_20"]).mean())
            return_20d = float(group_frame["return_20d"].median())
            momentum_5d = float(group_frame["return_5d"].median())
            style_score = breadth + min(max((return_20d + 0.05) / 0.20, 0.0), 1.0) * 0.5
            style_score += min(max((momentum_5d + 0.03) / 0.12, 0.0), 1.0) * 0.2
            group_scores[str(group)] = style_score
            group_metrics[str(group)] = {
                "breadth": breadth,
                "return_20d": return_20d,
                "momentum_5d": momentum_5d,
                "score": style_score,
            }
            if breadth >= self.style_min_breadth and return_20d >= self.style_min_return_20d:
                eligible_groups.append(str(group))
        market_return_ok = market_return_20d >= self.market_min_return_20d
        normal_market = market_breadth >= self.market_min_breadth and market_return_ok
        defensive_candidate = (
            not normal_market
            and market_return_ok
            and market_breadth >= self.defensive_market_min_breadth
        )
        defensive_allowed = defensive_candidate and bool(eligible_groups)
        strongest_style_group = None
        if defensive_allowed and eligible_groups:
            strongest_style_group = max(eligible_groups, key=lambda group: group_scores.get(group, 0.0))
            eligible_groups = [strongest_style_group]
        risk_off = not normal_market and not defensive_allowed
        market_state = self._market_state_name(
            market_breadth=market_breadth,
            market_return_20d=market_return_20d,
            eligible_group_count=len(eligible_groups),
            risk_off=risk_off,
            defensive_allowed=defensive_allowed,
        )
        return {
            "market_breadth": market_breadth,
            "market_return_20d": market_return_20d,
            "market_risk_off": risk_off,
            "eligible_groups": [] if risk_off else eligible_groups,
            "group_scores": group_scores,
            "group_metrics": group_metrics,
            "market_source": market_source,
            "benchmark_close_to_ma20": benchmark_close_to_ma20,
            "market_state": market_state,
            "position_size_multiplier": self._position_size_multiplier_for_market(market_state),
            "defensive_allowed": defensive_allowed,
            "strongest_style_group": strongest_style_group,
        }

    def _style_score_weight_for_market(self, market_state: str) -> float:
        if self.style_score_weight_active is not None and market_state in {"normal", "aggressive"}:
            return self.style_score_weight_active
        return self.style_score_weight

    def _position_size_multiplier_for_market(self, market_state: str) -> float:
        if market_state == "defensive":
            return min(max(self.defensive_position_size_multiplier, 0.0), 1.0)
        if market_state == "aggressive":
            return self.aggressive_position_size_multiplier
        if market_state == "risk_off":
            return 0.0
        return 1.0

    def _market_state_name(
        self,
        *,
        market_breadth: float,
        market_return_20d: float,
        eligible_group_count: int,
        risk_off: bool,
        defensive_allowed: bool = False,
    ) -> str:
        if risk_off:
            return "risk_off"
        if defensive_allowed:
            return "defensive"
        if eligible_group_count < max(1, len(self.groups) // 2):
            return "defensive"
        if market_breadth >= self.market_min_breadth + 0.15 and market_return_20d >= 0.03:
            return "aggressive"
        return "normal"

    def _select_candidates_from_recipes(
        self,
        *,
        signal_frame: pd.DataFrame,
        style_state: dict,
        eligible_groups: set[str],
        excluded_symbols: set[str],
        risk_off: bool,
        market_state: str,
    ) -> list[dict]:
        if (
            self.enabled_recipes == DEFAULT_FULL_A_ENABLED_RECIPES
            and not self.overlay_recipes
            and not self.quality_filter_enabled
            and not self.ml_prediction_map
            and self.ml_min_trend_prob is None
            and self.ml_score_weight == 0
            and not self.theme_buy_point_overlay
            and self.entry_market_states == ("normal", "aggressive", "defensive")
            and self.aggressive_position_size_multiplier == 1.0
        ):
            return self._select_candidates(
                signal_frame=signal_frame,
                eligible_groups=eligible_groups,
                excluded_symbols=excluded_symbols,
                risk_off=risk_off,
                market_state=market_state,
            )

        if market_state not in self.entry_market_states:
            return []

        rows: list[dict] = []
        if "momentum_core" in self.enabled_recipes:
            rows.extend(
                self._select_candidates(
                    signal_frame=signal_frame,
                    eligible_groups=eligible_groups,
                    excluded_symbols=excluded_symbols,
                    risk_off=risk_off,
                    market_state=market_state,
                )
            )
        if "trend_pullback_overlay" in set(self.enabled_recipes) | set(self.overlay_recipes):
            rows.extend(
                self._select_trend_pullback_overlay(
                    signal_frame=signal_frame,
                    style_state=style_state,
                    eligible_groups=eligible_groups,
                    excluded_symbols=excluded_symbols,
                    risk_off=risk_off,
                    market_state=market_state,
                )
            )
        if self.theme_buy_point_overlay:
            rows.extend(
                self._select_theme_buy_point_overlay(
                    signal_frame=signal_frame,
                    style_state=style_state,
                    eligible_groups=eligible_groups,
                    excluded_symbols=excluded_symbols,
                    risk_off=risk_off,
                    market_state=market_state,
                )
            )
        if self.quality_filter_enabled:
            rows = self._apply_quality_momentum_filter(rows, signal_frame)
        rows = self._apply_ml_trend_overlay(rows, signal_frame)
        return self._dedupe_candidate_rows(rows)

    def _select_theme_buy_point_overlay(
        self,
        *,
        signal_frame: pd.DataFrame,
        style_state: dict,
        eligible_groups: set[str],
        excluded_symbols: set[str],
        risk_off: bool,
        market_state: str,
    ) -> list[dict]:
        if (
            signal_frame.empty
            or risk_off
            or market_state not in {"normal", "aggressive"}
            or not eligible_groups
            or self.theme_overlay_max_daily_candidates <= 0
        ):
            return []
        ranking = build_ranking_snapshot(signal_frame, self.config, variant=RANKING_DIAGNOSTIC_VARIANT)
        alerts = build_strong_theme_alerts(
            ranking,
            market_breadth=float(style_state.get("market_breadth") or 0.0),
            market_return_20d=float(style_state.get("market_return_20d") or 0.0),
            market_state=market_state,
            signal_frame=signal_frame,
        )
        buy_symbols = []
        for alert in alerts:
            for candidate in alert.buy_point_candidates:
                if candidate.symbol not in buy_symbols:
                    buy_symbols.append(candidate.symbol)
        if not buy_symbols:
            return []
        frame = self._candidate_source_frame(
            signal_frame=signal_frame,
            style_state=style_state,
            eligible_groups=eligible_groups,
            excluded_symbols=excluded_symbols,
        )
        frame = frame.loc[frame["ts_code"].astype(str).isin(buy_symbols)].copy()
        if frame.empty:
            return []
        order = {symbol: idx for idx, symbol in enumerate(buy_symbols)}
        frame["_theme_order"] = frame["ts_code"].astype(str).map(order).fillna(9999)
        frame["selection_score"] = frame["selection_score"].fillna(0.0) + self.theme_overlay_score_bonus
        selected = frame.sort_values(["_theme_order", "selection_score"], ascending=[True, False]).head(
            self.theme_overlay_max_daily_candidates
        )
        detail_map = self._selection_detail_map(signal_frame)
        rows = []
        for rank, row in enumerate(selected.to_dict(orient="records"), start=1):
            row["score"] = float(row["selection_score"])
            row["base_score"] = float(row["selection_score"]) - self.theme_overlay_score_bonus
            row["recipe_bonus"] = self.theme_overlay_score_bonus
            row["entry_recipe"] = "strong_theme_buy_point_overlay"
            row["entry_reason"] = "full_a_momentum:strong_theme_buy_point_overlay"
            row["style_group"] = row.get("style_group") or row.get("group")
            row["market_state"] = market_state
            row["position_size_multiplier"] = self.theme_overlay_position_size_multiplier
            row["score_explain"] = f"strong_theme_buy_point_overlay +{self.theme_overlay_score_bonus:g}"
            self._apply_selection_diagnostics(row, rank=rank, detail_map=detail_map)
            rows.append(row)
        return rows

    def _select_trend_pullback_overlay(
        self,
        *,
        signal_frame: pd.DataFrame,
        style_state: dict,
        eligible_groups: set[str],
        excluded_symbols: set[str],
        risk_off: bool,
        market_state: str,
    ) -> list[dict]:
        if (
            signal_frame.empty
            or risk_off
            or not eligible_groups
            or market_state not in {"normal", "aggressive"}
            or self.overlay_max_daily_candidates <= 0
        ):
            return []
        frame = self._candidate_source_frame(
            signal_frame=signal_frame,
            style_state=style_state,
            eligible_groups=eligible_groups,
            excluded_symbols=excluded_symbols,
        )
        if frame.empty:
            return []
        try:
            frame = StrategyPreferenceFilter(self.config.selection).apply_trend_pullback(frame)
        except AttributeError:
            return []
        frame = frame.loc[frame["passes_strategy_preference_filter"].fillna(False)].copy()
        if frame.empty:
            return []
        frame["selection_score"] = frame["selection_score"].fillna(0.0) + self.overlay_score_bonus
        selected = frame.sort_values(["selection_score", "avg_amount_20d_yuan"], ascending=[False, False]).head(
            self.overlay_max_daily_candidates
        )
        detail_map = self._selection_detail_map(signal_frame)
        rows = []
        for rank, row in enumerate(selected.to_dict(orient="records"), start=1):
            row["score"] = float(row["selection_score"])
            row["base_score"] = float(row["selection_score"]) - self.overlay_score_bonus
            row["recipe_bonus"] = self.overlay_score_bonus
            row["entry_recipe"] = "trend_pullback_overlay"
            row["entry_reason"] = "full_a_momentum:trend_pullback_overlay"
            row["style_group"] = row.get("style_group") or row.get("group")
            row["market_state"] = market_state
            row["position_size_multiplier"] = self._position_size_multiplier_for_market(market_state)
            row["score_explain"] = f"trend_pullback_overlay +{self.overlay_score_bonus:g}"
            self._apply_selection_diagnostics(row, rank=rank, detail_map=detail_map)
            rows.append(row)
        return rows

    def _candidate_source_frame(
        self,
        *,
        signal_frame: pd.DataFrame,
        style_state: dict,
        eligible_groups: set[str],
        excluded_symbols: set[str],
    ) -> pd.DataFrame:
        score_column = f"{self.selection_variant}_score"
        style_column = "style_group" if "style_group" in signal_frame.columns else "group"
        frame = signal_frame.loc[
            signal_frame[style_column].isin(eligible_groups)
            & (~signal_frame["ts_code"].isin(excluded_symbols))
        ].copy()
        if frame.empty:
            return frame
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=max(self.top_n, self.max_positions),
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=[self.selection_variant],
            horizons=[1],
        )
        frame = frame.loc[study_engine._variant_mask(frame, self.selection_variant)].copy()
        if frame.empty:
            return frame
        group_scores = style_state["group_scores"]
        style_weight = self._style_score_weight_for_market(str(style_state.get("market_state") or ""))
        frame["selection_score"] = (
            frame[score_column].fillna(0.0)
            + frame[style_column].map(group_scores).fillna(0.0) * style_weight
        )
        return frame

    def _apply_quality_momentum_filter(self, rows: list[dict], signal_frame: pd.DataFrame) -> list[dict]:
        if not rows:
            return []
        quality_scores = self._quality_score_map(signal_frame)
        filtered = []
        for row in rows:
            symbol = str(row["ts_code"])
            quality_score = quality_scores.get(symbol, 0.5)
            if quality_score < self.quality_filter_min_score:
                continue
            bonus = self.quality_filter_score_bonus * quality_score
            row = dict(row)
            row["quality_score"] = quality_score
            row["quality_bonus"] = bonus
            row["score"] = float(row.get("score") or 0.0) + bonus
            row["selection_score"] = row["score"]
            row["entry_reason"] = f"{row.get('entry_reason') or 'full_a_momentum'}+quality_momentum_filter"
            row["score_explain"] = f"{row.get('score_explain') or 'base'}; quality +{bonus:g}"
            row["selection_score_explain"] = row["score_explain"]
            filtered.append(row)
        return filtered

    def _apply_ml_trend_overlay(self, rows: list[dict], signal_frame: pd.DataFrame) -> list[dict]:
        if not rows or not self.ml_prediction_map:
            return rows
        signal_trade_date = _signal_trade_date_from_frame(signal_frame)
        if not signal_trade_date:
            return rows
        adjusted = []
        for row in rows:
            symbol = str(row.get("ts_code") or "")
            trend_prob = self.ml_prediction_map.get((signal_trade_date, symbol))
            if trend_prob is None:
                if self.ml_min_trend_prob is not None:
                    continue
                trend_prob = 0.5
            if self.ml_min_trend_prob is not None and trend_prob < self.ml_min_trend_prob:
                continue
            row = dict(row)
            row["ml_trend_prob"] = trend_prob
            if self.ml_score_weight:
                bonus = self.ml_score_weight * trend_prob
                row["ml_bonus"] = bonus
                row["score"] = float(row.get("score") or 0.0) + bonus
                row["selection_score"] = row["score"]
                row["entry_reason"] = f"{row.get('entry_reason') or 'full_a_momentum'}+ml_trend"
                row["score_explain"] = f"{row.get('score_explain') or 'base'}; ml +{bonus:g}"
                row["selection_score_explain"] = row["score_explain"]
            adjusted.append(row)
        return adjusted

    def _quality_score_map(self, signal_frame: pd.DataFrame) -> dict[str, float]:
        if signal_frame.empty:
            return {}
        if "financial_quality_score" in signal_frame.columns:
            scores = pd.to_numeric(signal_frame["financial_quality_score"], errors="coerce").fillna(0.5)
        elif f"{self.selection_variant}_score" in signal_frame.columns:
            raw = pd.to_numeric(signal_frame[f"{self.selection_variant}_score"], errors="coerce")
            scores = raw.rank(pct=True).fillna(0.5)
        else:
            scores = pd.Series(0.5, index=signal_frame.index)
        return {str(symbol): float(score) for symbol, score in zip(signal_frame["ts_code"], scores)}

    def _selection_detail_map(self, signal_frame: pd.DataFrame) -> dict[str, dict]:
        if signal_frame.empty:
            return {}
        try:
            ranking = build_ranking_snapshot(signal_frame, self.config, variant=RANKING_DIAGNOSTIC_VARIANT)
        except (AttributeError, KeyError, ValueError):
            return {}
        details = {}
        for row in ranking.to_dict(orient="records"):
            symbol = str(row.get("ts_code") or "")
            if not symbol:
                continue
            factor_scores = {}
            for column in RANKING_FACTOR_COLUMNS:
                value = _json_float(row.get(column))
                if value is not None:
                    factor_scores[column] = value
            details[symbol] = {
                "ranking_variant": RANKING_DIAGNOSTIC_VARIANT,
                "ranking_score": _json_float(row.get("rank_score")),
                "ranking_position": _json_int(row.get("rank_position")),
                "ranking_score_explain": row.get("score_explain"),
                "ranking_factor_scores": json.dumps(factor_scores, ensure_ascii=False, sort_keys=True),
            }
        return details

    def _apply_selection_diagnostics(self, row: dict, *, rank: int, detail_map: dict[str, dict]) -> None:
        row["selection_variant"] = self.selection_variant
        row["selection_score"] = float(row.get("selection_score") or row.get("score") or 0.0)
        row["selection_rank"] = rank
        row["selection_score_explain"] = row.get("score_explain") or row.get("entry_reason")
        details = detail_map.get(str(row.get("ts_code") or ""), {})
        for key in (
            "ranking_variant",
            "ranking_score",
            "ranking_position",
            "ranking_score_explain",
            "ranking_factor_scores",
        ):
            row[key] = details.get(key)

    def _dedupe_candidate_rows(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
        if "avg_amount_20d_yuan" not in frame.columns:
            frame["avg_amount_20d_yuan"] = 0.0
        frame["avg_amount_20d_yuan"] = pd.to_numeric(frame["avg_amount_20d_yuan"], errors="coerce").fillna(0.0)
        frame = frame.sort_values(["score", "avg_amount_20d_yuan"], ascending=[False, False])
        frame = frame.drop_duplicates(subset=["ts_code"], keep="first").head(self.top_n)
        result = []
        for rank, row in enumerate(frame.to_dict(orient="records"), start=1):
            row["rank"] = rank
            row["score"] = float(row["score"])
            row["selection_rank"] = rank
            row["selection_score"] = row["score"]
            result.append(row)
        return result

    def _select_candidates(
        self,
        signal_frame: pd.DataFrame,
        eligible_groups: set[str],
        excluded_symbols: set[str],
        risk_off: bool,
        market_state: str,
    ) -> list[dict]:
        if signal_frame.empty or risk_off or not eligible_groups or market_state not in self.entry_market_states:
            return []
        score_column = f"{self.selection_variant}_score"
        style_column = "style_group" if "style_group" in signal_frame.columns else "group"
        frame = signal_frame.loc[
            signal_frame[style_column].isin(eligible_groups)
            & (~signal_frame["ts_code"].isin(excluded_symbols))
        ].copy()
        if frame.empty:
            return []
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=max(self.top_n, self.max_positions),
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=[self.selection_variant],
            horizons=[1],
        )
        frame = frame.loc[study_engine._variant_mask(frame, self.selection_variant)].copy()
        if frame.empty:
            return []
        style_state = self._market_style_state(signal_frame)
        group_scores = style_state["group_scores"]
        style_weight = self._style_score_weight_for_market(market_state)
        frame["selection_score"] = (
            frame[score_column].fillna(0.0)
            + frame[style_column].map(group_scores).fillna(0.0) * style_weight
        )
        selected = frame.sort_values(["selection_score", "avg_amount_20d_yuan"], ascending=[False, False]).head(self.top_n)
        detail_map = self._selection_detail_map(signal_frame)
        rows = []
        for rank, row in enumerate(selected.to_dict(orient="records"), start=1):
            row["rank"] = rank
            row["score"] = float(row["selection_score"])
            row["entry_recipe"] = "momentum_core"
            row["entry_reason"] = f"full_a_momentum:{self.selection_variant}"
            row["style_group"] = row.get("style_group") or row.get("group")
            row["market_state"] = market_state
            row["position_size_multiplier"] = self._position_size_multiplier_for_market(market_state)
            self._apply_selection_diagnostics(row, rank=rank, detail_map=detail_map)
            rows.append(row)
        return rows

    def _execute_buys(
        self,
        trade_date: str,
        signal_trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        candidates: list[dict],
        open_equity: float,
        positions: dict[str, Tianzhu9Position],
        trades: list[Tianzhu9Trade],
        cash_ref: dict[str, float],
    ) -> float:
        if not candidates or open_equity <= 0 or len(positions) >= self.max_positions:
            return 0.0
        traded_value = 0.0
        target_slot_value = open_equity / self.max_positions
        for candidate in candidates:
            if len(positions) >= self.max_positions:
                break
            symbol = str(candidate["ts_code"])
            if symbol in positions or symbol not in prices.index:
                continue
            row = prices.loc[symbol]
            day_open = float(row["open"])
            day_low = float(row["low"])
            prev_close = float(candidate["close"])
            if any(math.isnan(value) or value <= 0 for value in (day_open, day_low, prev_close)):
                continue
            limit_price = round(prev_close * (1 + self.config.pricing.buy_markup), 2)
            if day_low > limit_price:
                continue
            fill_price = day_open if day_open <= limit_price else limit_price
            position_size_multiplier = min(
                max(float(candidate.get("position_size_multiplier") or 1.0), 0.0),
                1.0,
            )
            target_value = min(target_slot_value * position_size_multiplier, cash_ref["cash"])
            raw_shares = int(target_value / fill_price)
            shares = (raw_shares // self.lot_size) * self.lot_size
            if shares < self.lot_size:
                continue
            gross_amount = fill_price * shares
            fees = max(gross_amount * self.config.backtest.commission_rate, 5.0)
            if gross_amount + fees > cash_ref["cash"]:
                affordable = int(cash_ref["cash"] / (fill_price * (1 + self.config.backtest.commission_rate)))
                shares = (affordable // self.lot_size) * self.lot_size
                if shares < self.lot_size:
                    continue
                gross_amount = fill_price * shares
                fees = max(gross_amount * self.config.backtest.commission_rate, 5.0)
            net_amount = gross_amount + fees
            cash_ref["cash"] -= net_amount
            entry_recipe = str(candidate.get("entry_recipe") or "momentum_core")
            entry_reason = str(candidate.get("entry_reason") or f"full_a_momentum:{self.selection_variant}")
            market_state = str(candidate.get("market_state") or "unknown")
            style_group = str(candidate.get("style_group") or candidate.get("group") or "")
            ranking_position = _json_int(candidate.get("ranking_position"))
            positions[symbol] = Tianzhu9Position(
                symbol=symbol,
                name=str(candidate.get("name") or symbol),
                shares=shares,
                entry_trade_date=trade_date,
                signal_trade_date=signal_trade_date,
                entry_trade_index=trade_index,
                entry_price=fill_price,
                entry_cost=net_amount,
                highest_close=fill_price,
                score=float(candidate["score"]),
                rank=int(candidate["rank"]),
                highest_high=fill_price,
                entry_recipe=entry_recipe,
                entry_reason=entry_reason,
                market_state=market_state,
                style_group=style_group or None,
                selection_variant=candidate.get("selection_variant"),
                selection_score=_safe_float(candidate.get("selection_score")),
                selection_rank=_json_int(candidate.get("selection_rank")),
                selection_score_explain=candidate.get("selection_score_explain"),
                ranking_variant=candidate.get("ranking_variant"),
                ranking_score=_safe_float(candidate.get("ranking_score")),
                ranking_position=ranking_position,
                ranking_score_explain=candidate.get("ranking_score_explain"),
                ranking_factor_scores=candidate.get("ranking_factor_scores"),
            )
            trades.append(
                Tianzhu9Trade(
                    trade_date=trade_date,
                    action="BUY",
                    symbol=symbol,
                    name=str(candidate.get("name") or symbol),
                    shares=shares,
                    price=fill_price,
                    gross_amount=gross_amount,
                    fees=fees,
                    net_amount=net_amount,
                    signal_trade_date=signal_trade_date,
                    rank=int(candidate["rank"]),
                    score=float(candidate["score"]),
                    pnl=None,
                    entry_recipe=entry_recipe,
                    entry_reason=entry_reason,
                    market_state=market_state,
                    style_group=style_group or None,
                    reason=f"position_size_multiplier={position_size_multiplier:g}",
                    selection_variant=candidate.get("selection_variant"),
                    selection_score=_safe_float(candidate.get("selection_score")),
                    selection_rank=_json_int(candidate.get("selection_rank")),
                    selection_score_explain=candidate.get("selection_score_explain"),
                    ranking_variant=candidate.get("ranking_variant"),
                    ranking_score=_safe_float(candidate.get("ranking_score")),
                    ranking_position=ranking_position,
                    ranking_score_explain=candidate.get("ranking_score_explain"),
                    ranking_factor_scores=candidate.get("ranking_factor_scores"),
                )
            )
            traded_value += gross_amount
        return traded_value

    def _execute_sells(
        self,
        trade_date: str,
        trade_index: int,
        prices: pd.DataFrame,
        factor_frame: pd.DataFrame,
        signal_trade_date: str,
        positions: dict[str, Tianzhu9Position],
        selected_symbols: set[str],
        eligible_groups: set[str],
        risk_off: bool,
        market_state: str,
        trades: list[Tianzhu9Trade],
        cash_ref: dict[str, float],
        loss_cooldown_until: dict[str, int],
        fallback_price_map: dict[str, pd.DataFrame] | None = None,
        fallback_trade_dates: list[str] | None = None,
        exit_diagnostics: dict[str, int] | None = None,
    ) -> float:
        traded_value = 0.0
        for symbol in list(positions):
            position = positions[symbol]
            if symbol not in prices.index:
                continue
            holding_days = trade_index - position.entry_trade_index + 1
            feature = self._feature_row(factor_frame, signal_trade_date, symbol)
            if feature is None and fallback_price_map is not None and fallback_trade_dates is not None:
                feature = _fallback_exit_feature(
                    price_map=fallback_price_map,
                    trade_dates=fallback_trade_dates,
                    signal_trade_date=signal_trade_date,
                    symbol=symbol,
                )
                if feature is not None and exit_diagnostics is not None:
                    exit_diagnostics["fallback_feature_uses"] += 1
            if feature is None:
                if exit_diagnostics is not None:
                    exit_diagnostics["missing_feature_days"] += 1
                continue
            if not str(feature.get("style_group") or ""):
                feature = feature.copy()
                feature["style_group"] = position.style_group
            prev_close = float(feature["close"])
            highest_price = max(position.highest_close, position.highest_high or 0.0)
            exit_reason = self._exit_reason(
                feature=feature,
                position=position,
                highest_price=highest_price,
                holding_days=holding_days,
                eligible_groups=eligible_groups,
                risk_off=risk_off,
            )
            if exit_reason is None:
                continue

            row = prices.loc[symbol]
            day_open = float(row["open"])
            day_high = float(row["high"])
            if any(math.isnan(value) or value <= 0 for value in (day_open, day_high, prev_close)):
                continue
            limit_price = round(prev_close * (1 - self.config.pricing.sell_markdown), 2)
            if day_high < limit_price:
                continue
            fill_price = day_open if day_open >= limit_price else limit_price
            gross_amount = fill_price * position.shares
            fees = max(
                gross_amount * (self.config.backtest.commission_rate + self.config.backtest.stamp_duty_rate),
                5.0,
            )
            net_amount = gross_amount - fees
            cash_ref["cash"] += net_amount
            pnl = net_amount - position.entry_cost
            if pnl <= 0 and self.loss_cooldown_days > 0:
                loss_cooldown_until[symbol] = trade_index + self.loss_cooldown_days
            style_group = position.style_group or str(feature.get("style_group") or feature.get("group") or "")
            trades.append(
                Tianzhu9Trade(
                    trade_date=trade_date,
                    action="SELL",
                    symbol=symbol,
                    name=position.name,
                    shares=position.shares,
                    price=fill_price,
                    gross_amount=gross_amount,
                    fees=fees,
                    net_amount=net_amount,
                    signal_trade_date=position.signal_trade_date,
                    rank=position.rank,
                    score=position.score,
                    pnl=pnl,
                    reason=exit_reason,
                    entry_recipe=position.entry_recipe,
                    entry_reason=position.entry_reason,
                    exit_reason=exit_reason,
                    market_state=market_state,
                    style_group=style_group or None,
                    holding_days=holding_days,
                    selection_variant=position.selection_variant,
                    selection_score=position.selection_score,
                    selection_rank=position.selection_rank,
                    selection_score_explain=position.selection_score_explain,
                    ranking_variant=position.ranking_variant,
                    ranking_score=position.ranking_score,
                    ranking_position=position.ranking_position,
                    ranking_score_explain=position.ranking_score_explain,
                    ranking_factor_scores=position.ranking_factor_scores,
                )
            )
            traded_value += gross_amount
            del positions[symbol]
        return traded_value

    def _exit_reason(
        self,
        *,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
        eligible_groups: set[str],
        risk_off: bool,
    ) -> str | None:
        current_close = float(feature["close"])
        if self.exit_profile == SLOW_PROFIT_LOCK_PROFILE:
            profile_exit = slow_profit_lock_exit_signal(
                entry_price=position.entry_price,
                current_close=current_close,
                highest_price=highest_price,
                holding_days=holding_days,
                ma20=_safe_float(feature.get("ma_20")),
                ma60=_safe_float(feature.get("ma_60")),
                return_5d=_safe_float(feature.get("return_5d")),
                style_return_20d=_safe_float(feature.get("style_return_20d_median")),
                style_breadth_20d=_safe_float(feature.get("style_breadth_20d")),
                hard_exit_days=self.hard_exit_days,
            )
            if profile_exit.should_exit:
                return normalize_sell_reason(profile_exit.reason)
            return None

        exit_check = tiered_trailing_take_profit(
            entry_price=position.entry_price,
            current_close=current_close,
            highest_price=highest_price,
            levels=self._trailing_levels(feature),
        )
        if exit_check.should_exit:
            return "trailing_take_profit"
        if self._should_exit_ma20_break(feature, holding_days):
            return "ma20_break"
        if self._should_exit_failure(feature, position, highest_price, holding_days):
            return "failure_exit"
        if self._should_exit_high_drawdown(feature, highest_price, holding_days):
            return "high_drawdown_exit"
        if self._should_exit_chandelier(feature, highest_price, holding_days):
            return "chandelier_exit"
        if self._should_exit_trend_decay(feature, holding_days):
            return "trend_decay_exit"
        if self._should_exit_style_rotation(
            feature=feature,
            position=position,
            highest_price=highest_price,
            holding_days=holding_days,
            eligible_groups=eligible_groups,
        ):
            return "style_rotation_exit"
        if self._should_exit_market_risk(
            feature=feature,
            position=position,
            highest_price=highest_price,
            holding_days=holding_days,
            eligible_groups=eligible_groups,
            risk_off=risk_off,
        ):
            return "market_risk_exit"
        if self._should_exit_industry_weak(feature, holding_days):
            return "industry_weak_exit"
        if self._should_exit_relative_weak(feature, holding_days):
            return "relative_weak_exit"
        if self._should_exit_volume_stall(feature, position, highest_price, holding_days):
            return "volume_stall_exit"
        if self._should_exit_upper_shadow(feature, position, highest_price, holding_days):
            return "upper_shadow_exit"
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        if (
            self.exit_profile == LEGACY_EXIT_PROFILE
            and self.exit_risk_off_failed_hard_exit_days is not None
            and risk_off
            and holding_days >= self.exit_risk_off_failed_hard_exit_days
            and not self._has_winner_peak(peak_profit)
        ):
            return "risk_off_failed_hard_exit"
        if (
            self.exit_profile == LEGACY_EXIT_PROFILE
            and self.hard_exit_days is not None
            and holding_days >= self.hard_exit_days
            and not self._has_winner_peak(peak_profit)
        ):
            return "max_holding_days_exit"
        return None

    def _has_winner_peak(self, peak_profit: float) -> bool:
        return (
            self.exit_winner_hard_exit_bypass_peak_pct is not None
            and peak_profit >= self.exit_winner_hard_exit_bypass_peak_pct
        )

    def _trailing_levels(self, feature: pd.Series) -> tuple[tuple[float, float], ...]:
        if not self.exit_adaptive_trailing:
            return TIERED_TRAILING_TAKE_PROFIT_LEVELS
        atr_pct = _safe_float(feature.get("atr_20d_pct"))
        if atr_pct is None:
            return TIERED_TRAILING_TAKE_PROFIT_LEVELS
        return tuple(
            (profit_pct, max(drawdown_pct, atr_pct * self.exit_atr_multiplier))
            for profit_pct, drawdown_pct in TIERED_TRAILING_TAKE_PROFIT_LEVELS
        )

    def _should_exit_ma20_break(self, feature: pd.Series, holding_days: int) -> bool:
        if not self.exit_ma20_break or holding_days < 3:
            return False
        close = _safe_float(feature.get("close"))
        ma20 = _safe_float(feature.get("ma_20"))
        return close is not None and ma20 is not None and close < ma20

    def _should_exit_failure(
        self,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
    ) -> bool:
        if self.exit_failure_days is None or holding_days < self.exit_failure_days:
            return False
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        if peak_profit >= self.exit_failure_min_peak_profit_pct:
            return False
        close = _safe_float(feature.get("close"))
        ma20 = _safe_float(feature.get("ma_20"))
        return_5d = _safe_float(feature.get("return_5d"))
        return (
            (close is not None and ma20 is not None and close < ma20)
            or (return_5d is not None and return_5d < 0.0)
        )

    def _should_exit_high_drawdown(
        self,
        feature: pd.Series,
        highest_price: float,
        holding_days: int,
    ) -> bool:
        if self.exit_high_drawdown_pct is None or holding_days < 5 or highest_price <= 0:
            return False
        close = _safe_float(feature.get("close"))
        return close is not None and close / highest_price - 1.0 <= -self.exit_high_drawdown_pct

    def _should_exit_chandelier(
        self,
        feature: pd.Series,
        highest_price: float,
        holding_days: int,
    ) -> bool:
        if self.exit_chandelier_atr_multiplier is None or holding_days < 5 or highest_price <= 0:
            return False
        close = _safe_float(feature.get("close"))
        atr_20d = _safe_float(feature.get("atr_20d"))
        if close is None or atr_20d is None or atr_20d <= 0:
            return False
        stop_price = highest_price - self.exit_chandelier_atr_multiplier * atr_20d
        return close <= stop_price

    def _should_exit_trend_decay(self, feature: pd.Series, holding_days: int) -> bool:
        if not self.exit_trend_decay or holding_days < 5:
            return False
        close = _safe_float(feature.get("close"))
        ma5 = _safe_float(feature.get("ma_5"))
        ma10 = _safe_float(feature.get("ma_10"))
        ma20 = _safe_float(feature.get("ma_20"))
        return_5d = _safe_float(feature.get("return_5d"))
        return (
            close is not None
            and ma5 is not None
            and ma10 is not None
            and ma20 is not None
            and return_5d is not None
            and close < ma20
            and ma5 < ma10
            and return_5d < 0.0
        )

    def _should_exit_market_risk(
        self,
        *,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
        eligible_groups: set[str],
        risk_off: bool,
    ) -> bool:
        if not self.exit_market_risk or holding_days < MARKET_RISK_EXIT_MIN_HOLDING_DAYS:
            return False
        style_group = str(feature.get("style_group") or feature.get("group") or "")
        if not risk_off and (not style_group or style_group in eligible_groups):
            return False
        close = _safe_float(feature.get("close"))
        ma20 = _safe_float(feature.get("ma_20"))
        return_5d = _safe_float(feature.get("return_5d"))
        if close is None or ma20 is None or close >= ma20:
            return False
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        if peak_profit >= self.take_profit_trigger_pct:
            return False
        position_return = close / position.entry_price - 1.0 if position.entry_price else 0.0
        return (
            return_5d is not None
            and return_5d <= -MARKET_RISK_EXIT_MIN_5D_LOSS_PCT
        ) or position_return <= -MARKET_RISK_EXIT_MIN_POSITION_LOSS_PCT

    def _should_exit_style_rotation(
        self,
        *,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
        eligible_groups: set[str],
    ) -> bool:
        if not self.exit_style_rotation or holding_days < STYLE_ROTATION_EXIT_MIN_HOLDING_DAYS:
            return False
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        if peak_profit >= self.take_profit_trigger_pct:
            return False
        style_group = str(feature.get("style_group") or feature.get("group") or "")
        style_return_20d = _safe_float(feature.get("style_return_20d_median"))
        style_breadth = _safe_float(feature.get("style_breadth_20d"))
        relative_5d = _safe_float(feature.get("relative_style_return_5d"))
        relative_20d = _safe_float(feature.get("relative_style_return_20d"))
        style_rotated = (
            (bool(style_group) and style_group not in eligible_groups)
            or (style_return_20d is not None and style_return_20d < self.style_min_return_20d)
            or (style_breadth is not None and style_breadth < self.style_min_breadth)
            or (relative_20d is not None and relative_20d <= -self.exit_relative_weak_20d_pct)
        )
        if not style_rotated:
            return False
        close = _safe_float(feature.get("close"))
        ma20 = _safe_float(feature.get("ma_20"))
        return_5d = _safe_float(feature.get("return_5d"))
        if close is None or ma20 is None or close >= ma20:
            return False
        position_return = close / position.entry_price - 1.0 if position.entry_price else 0.0
        return (
            return_5d is not None
            and return_5d <= -STYLE_ROTATION_EXIT_MIN_5D_LOSS_PCT
        ) or (
            relative_5d is not None
            and relative_5d <= -STYLE_ROTATION_EXIT_MIN_RELATIVE_5D_PCT
        ) or position_return <= -STYLE_ROTATION_EXIT_MIN_POSITION_LOSS_PCT

    def _should_exit_industry_weak(self, feature: pd.Series, holding_days: int) -> bool:
        if not self.exit_industry_weak or holding_days < 3:
            return False
        style_return_20d = _safe_float(feature.get("style_return_20d_median"))
        style_breadth = _safe_float(feature.get("style_breadth_20d"))
        if style_return_20d is None and style_breadth is None:
            return False
        industry_weak = (
            (style_return_20d is not None and style_return_20d < self.style_min_return_20d)
            or (style_breadth is not None and style_breadth < self.style_min_breadth)
        )
        if not industry_weak:
            return False
        close = _safe_float(feature.get("close"))
        ma10 = _safe_float(feature.get("ma_10"))
        return_5d = _safe_float(feature.get("return_5d"))
        return (
            (close is not None and ma10 is not None and close < ma10)
            or (return_5d is not None and return_5d < 0.0)
        )

    def _should_exit_relative_weak(self, feature: pd.Series, holding_days: int) -> bool:
        if not self.exit_relative_weak or holding_days < 3:
            return False
        relative_5d = _safe_float(feature.get("relative_style_return_5d"))
        relative_20d = _safe_float(feature.get("relative_style_return_20d"))
        return_5d = _safe_float(feature.get("return_5d"))
        close = _safe_float(feature.get("close"))
        ma20 = _safe_float(feature.get("ma_20"))
        short_weak = (
            relative_5d is not None
            and relative_5d <= -self.exit_relative_weak_5d_pct
            and return_5d is not None
            and return_5d < 0.0
        )
        medium_weak = (
            relative_20d is not None
            and relative_20d <= -self.exit_relative_weak_20d_pct
            and close is not None
            and ma20 is not None
            and close < ma20
        )
        return short_weak or medium_weak

    def _should_exit_volume_stall(
        self,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
    ) -> bool:
        if not self.exit_volume_stall or holding_days < 3:
            return False
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        if peak_profit < 0.05:
            return False
        amount_ratio = _safe_float(feature.get("amount_ratio_5d"))
        return_5d = _safe_float(feature.get("return_5d"))
        close = _safe_float(feature.get("close"))
        ma5 = _safe_float(feature.get("ma_5"))
        upper_shadow = _safe_float(feature.get("upper_shadow_pct"))
        return (
            amount_ratio is not None
            and amount_ratio >= self.exit_volume_stall_ratio
            and return_5d is not None
            and return_5d <= 0.02
            and (
                (close is not None and ma5 is not None and close < ma5)
                or (upper_shadow is not None and upper_shadow >= 0.35)
            )
        )

    def _should_exit_upper_shadow(
        self,
        feature: pd.Series,
        position: Tianzhu9Position,
        highest_price: float,
        holding_days: int,
    ) -> bool:
        if not self.exit_upper_shadow or holding_days < 2:
            return False
        peak_profit = highest_price / position.entry_price - 1.0 if position.entry_price else 0.0
        close_to_ma20 = _safe_float(feature.get("close_to_ma_20"))
        if peak_profit < 0.08 and (close_to_ma20 is None or close_to_ma20 < 0.08):
            return False
        upper_shadow = _safe_float(feature.get("upper_shadow_pct"))
        amount_ratio = _safe_float(feature.get("amount_ratio_5d"))
        return (
            upper_shadow is not None
            and upper_shadow >= self.exit_upper_shadow_pct
            and (amount_ratio is None or amount_ratio >= 1.1)
        )

    def _mark_to_market_equity(
        self,
        cash: float,
        positions: dict[str, Tianzhu9Position],
        prices: pd.DataFrame,
        price_field: str,
    ) -> float:
        equity = cash
        for position in positions.values():
            if position.symbol not in prices.index:
                continue
            price = float(prices.loc[position.symbol, price_field])
            if math.isnan(price):
                continue
            equity += position.shares * price
        return equity

    def _update_position_highs(self, positions: dict[str, Tianzhu9Position], prices: pd.DataFrame) -> None:
        for position in positions.values():
            if position.symbol not in prices.index:
                continue
            close_price = float(prices.loc[position.symbol, "close"])
            high_price = float(prices.loc[position.symbol, "high"])
            if not math.isnan(close_price):
                position.highest_close = max(position.highest_close, close_price)
            if not math.isnan(high_price):
                position.highest_high = max(position.highest_high or position.highest_close, high_price)

    @staticmethod
    def _feature_row(factor_frame: pd.DataFrame, signal_trade_date: str, symbol: str) -> pd.Series | None:
        if factor_frame.empty:
            return None
        rows = factor_frame.loc[
            (factor_frame["trade_date"].astype(str) == signal_trade_date)
            & (factor_frame["ts_code"] == symbol)
        ]
        if rows.empty:
            return None
        return rows.iloc[0]


def _fallback_exit_feature(
    *,
    price_map: dict[str, pd.DataFrame],
    trade_dates: list[str],
    signal_trade_date: str,
    symbol: str,
) -> pd.Series | None:
    try:
        signal_index = trade_dates.index(signal_trade_date)
    except ValueError:
        return None
    rows = []
    for trade_date in trade_dates[max(0, signal_index - 20) : signal_index + 1]:
        prices = price_map.get(trade_date)
        if prices is None or symbol not in prices.index:
            continue
        row = prices.loc[symbol]
        values = {column: _safe_float(row.get(column)) for column in ("open", "high", "low", "close")}
        if any(value is None or value <= 0 for value in values.values()):
            continue
        rows.append(values)
    if not rows:
        return None
    closes = [float(row["close"]) for row in rows]
    current = rows[-1]
    feature = {
        "open": current["open"],
        "high": current["high"],
        "low": current["low"],
        "close": current["close"],
        "ma_5": sum(closes[-5:]) / 5 if len(closes) >= 5 else None,
        "ma_10": sum(closes[-10:]) / 10 if len(closes) >= 10 else None,
        "ma_20": sum(closes[-20:]) / 20 if len(closes) >= 20 else None,
        "return_5d": closes[-1] / closes[-6] - 1.0 if len(closes) >= 6 else None,
    }
    true_ranges = []
    for index, row in enumerate(rows):
        if index == 0:
            true_ranges.append(float(row["high"]) - float(row["low"]))
            continue
        previous_close = float(rows[index - 1]["close"])
        true_ranges.append(
            max(
                float(row["high"]) - float(row["low"]),
                abs(float(row["high"]) - previous_close),
                abs(float(row["low"]) - previous_close),
            )
        )
    feature["atr_20d"] = sum(true_ranges[-20:]) / 20 if len(true_ranges) >= 20 else None
    ma20 = feature["ma_20"]
    feature["close_to_ma_20"] = float(current["close"]) / ma20 - 1.0 if ma20 else None
    return pd.Series(feature)


def _summarize_trade_groups(frame: pd.DataFrame, group_column: str) -> list[dict]:
    if frame.empty or group_column not in frame.columns:
        return []
    work = frame.copy()
    work[group_column] = work[group_column].fillna("").astype(str)
    work.loc[work[group_column].str.strip() == "", group_column] = "unknown"
    rows = []
    for group, group_frame in work.groupby(group_column):
        sell_frame = group_frame.loc[group_frame["action"] == "SELL"].copy()
        pnl = pd.to_numeric(sell_frame.get("pnl"), errors="coerce")
        holding_days = pd.to_numeric(sell_frame.get("holding_days"), errors="coerce")
        score = pd.to_numeric(group_frame.get("score"), errors="coerce")
        rank = pd.to_numeric(group_frame.get("rank"), errors="coerce")
        sell_count = int(len(sell_frame))
        win_count = int((pnl > 0).sum())
        rows.append(
            {
                "group": group,
                "trade_count": int(len(group_frame)),
                "sell_trade_count": sell_count,
                "win_rate": float(win_count / sell_count) if sell_count else 0.0,
                "total_pnl": float(pnl.sum()) if pnl.notna().any() else 0.0,
                "avg_pnl": float(pnl.mean()) if pnl.notna().any() else None,
                "median_pnl": float(pnl.median()) if pnl.notna().any() else None,
                "avg_holding_days": float(holding_days.mean()) if holding_days.notna().any() else None,
                "avg_score": float(score.mean()) if score.notna().any() else None,
                "avg_rank": float(rank.mean()) if rank.notna().any() else None,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["trade_count"]), str(row["group"])))


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    series = frame[column].fillna("").astype(str).str.strip()
    series = series.loc[series != ""]
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _mean_numeric(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _value_or_default(value, default):
    return default if value is None else value


def _normalize_recipe_names(values: list[str] | tuple[str, ...] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        values = default
    names = tuple(str(value).strip() for value in values if str(value).strip())
    return names or default


def _normalize_market_states(values: list[str] | tuple[str, ...] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    allowed = {"normal", "aggressive", "defensive"}
    if values is None:
        values = default
    states = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    unsupported = sorted(set(states) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported entry market state(s): {', '.join(unsupported)}")
    return states or default


def _json_float(value) -> float | None:
    number = _safe_float(value)
    return None if number is None else float(number)


def _json_int(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _slug_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _bounded_output_stem(stem: str, *, resolved_start: str, resolved_end: str) -> str:
    if len(stem.encode("utf-8")) <= 240:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    return f"full-a-momentum-cfg{digest}-{resolved_start}-{resolved_end}"


def _load_ml_prediction_map(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    frame = pd.read_csv(prediction_path)
    required_columns = {"signal_trade_date", "ts_code", "trend_prob"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"ML predictions file is missing required columns: {', '.join(sorted(missing))}")
    frame = frame[["signal_trade_date", "ts_code", "trend_prob"]].copy()
    frame["signal_trade_date"] = (
        frame["signal_trade_date"].fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    )
    frame["ts_code"] = frame["ts_code"].fillna("").astype(str)
    frame["trend_prob"] = pd.to_numeric(frame["trend_prob"], errors="coerce")
    frame = frame.dropna(subset=["trend_prob"])
    return {
        (str(row.signal_trade_date), str(row.ts_code)): float(row.trend_prob)
        for row in frame.itertuples(index=False)
    }


def _signal_trade_date_from_frame(signal_frame: pd.DataFrame) -> str | None:
    if signal_frame.empty or "trade_date" not in signal_frame.columns:
        return None
    values = signal_frame["trade_date"].dropna().astype(str).str.replace(".0", "", regex=False)
    if values.empty:
        return None
    return values.iloc[0].zfill(8)
