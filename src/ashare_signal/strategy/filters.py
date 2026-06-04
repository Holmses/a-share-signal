from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ashare_signal.config import AppConfig, SelectionConfig


ELIGIBLE_REASON = "eligible"


@dataclass(frozen=True, slots=True)
class FilterRule:
    column: str
    reason: str


HARD_FILTER_RULES: tuple[FilterRule, ...] = (
    FilterRule("passes_exchange_filter", "exchange_not_supported"),
    FilterRule("passes_st_filter", "st_stock"),
    FilterRule("passes_suspension_filter", "suspended"),
    FilterRule("passes_listing_age_filter", "listed_days_too_short"),
    FilterRule("passes_price_filter", "price_below_threshold"),
    FilterRule("passes_liquidity_filter", "liquidity_below_threshold"),
)

TREND_PULLBACK_RULES: tuple[FilterRule, ...] = (
    FilterRule("passes_trend_structure_preference", "trend_structure_invalid"),
    FilterRule("passes_pullback_preference", "pullback_range_mismatch"),
    FilterRule("passes_reversal_preference", "reversal_confirmation_missing"),
    FilterRule("passes_volume_preference", "volume_preference_mismatch"),
    FilterRule("passes_market_cap_preference", "market_cap_below_preference"),
)

REBOUND_BOTTOMING_RULES: tuple[FilterRule, ...] = (
    FilterRule("passes_rebound_drawdown_preference", "rebound_drawdown_out_of_range"),
    FilterRule("passes_rebound_position_preference", "rebound_position_too_weak"),
    FilterRule("passes_rebound_stabilization_preference", "rebound_stabilization_missing"),
    FilterRule("passes_rebound_liquidity_preference", "rebound_liquidity_mismatch"),
    FilterRule("passes_rebound_moneyflow_preference", "rebound_moneyflow_too_weak"),
    FilterRule("passes_rebound_limit_preference", "rebound_limit_state_blocked"),
    FilterRule("passes_market_cap_preference", "market_cap_below_preference"),
)


class HardTradeFilter:
    """Tradability filter for conditions that should stay as hard exclusions."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def apply(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        df = snapshot.copy()

        is_a_share = df["exchange"].isin(["SSE", "SZSE"])
        not_beijing = ~df["ts_code"].fillna("").str.endswith(".BJ")
        market_allowed = ~df["market"].fillna("").str.contains("北交", regex=False)

        st_filter = ~df["is_st"]
        if not self.config.filters.exclude_st:
            st_filter = st_filter | df["is_st"]

        suspension_filter = ~df["is_suspended"]
        if not self.config.filters.exclude_suspended:
            suspension_filter = suspension_filter | df["is_suspended"]

        df["passes_exchange_filter"] = is_a_share & not_beijing & market_allowed
        df["passes_st_filter"] = st_filter
        df["passes_suspension_filter"] = suspension_filter
        df["passes_listing_age_filter"] = df["listed_days"].fillna(-1) >= self.config.filters.min_list_days
        df["passes_price_filter"] = df["close"].fillna(0.0) >= self.config.filters.min_price
        df["passes_liquidity_filter"] = (
            df["avg_amount_20d_yuan"].fillna(0.0) >= self.config.filters.min_avg_turnover
        )

        df["is_tradeable"] = _all_pass(df, HARD_FILTER_RULES)
        df["is_candidate"] = df["is_tradeable"]
        df["hard_filter_reason"] = _first_failed_reason(df, HARD_FILTER_RULES)
        df["exclude_reason"] = df["hard_filter_reason"]
        return df


class StrategyPreferenceFilter:
    """Strategy-specific preference filters that should not be hard tradability filters."""

    def __init__(self, selection_config: SelectionConfig) -> None:
        self.selection_config = selection_config

    def apply(self, frame: pd.DataFrame, recipe_name: str) -> pd.DataFrame:
        if recipe_name == "trend_pullback_rank":
            return self.apply_trend_pullback(frame)
        if recipe_name == "rebound_bottoming_rank":
            return self.apply_rebound_bottoming(frame)
        raise ValueError(f"Unsupported strategy preference recipe: {recipe_name}")

    def apply_trend_pullback(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        config = self.selection_config
        df["preference_recipe"] = "trend_pullback_rank"
        df["passes_trend_structure_preference"] = (
            (_num(df, "close_to_ma_60") > 0)
            & (_num(df, "ma_20_to_ma_60") > 0)
            & (_num(df, "ma_60_slope_20d") > 0)
            & (_num(df, "momentum_20d") > 0)
        )
        df["passes_pullback_preference"] = (
            (_num(df, "pullback_from_20d_high") >= config.buy_min_pullback_from_20d_high)
            & (_num(df, "pullback_from_20d_high") <= config.buy_max_pullback_from_20d_high)
            & (_num(df, "close_to_ma_20") >= config.buy_min_close_to_ma20)
            & (_num(df, "close_to_ma_20") <= config.buy_max_close_to_ma20)
            & (_num(df, "close_to_ma_5") > 0)
        )
        df["passes_reversal_preference"] = (
            (_num(df, "return_1d") > 0)
            & (_num(df, "return_1d") <= config.buy_max_return_1d)
            & (_num(df, "low_to_prev_low") >= 0)
            & (_num(df, "low_to_prev_low") <= config.buy_max_low_to_prev_low)
            & (_num(df, "momentum_5d") >= config.buy_min_momentum_5d)
        )
        df["passes_volume_preference"] = (
            (_num(df, "amount_ratio_5d") >= config.buy_min_amount_ratio_5d)
            & (_num(df, "amount_ratio_5d") <= config.buy_max_amount_ratio_5d)
            & (_num(df, "volume_ratio").isna() | (_num(df, "volume_ratio") <= config.buy_max_volume_ratio))
        )
        df["passes_market_cap_preference"] = _num(df, "total_mv_yuan") >= config.buy_min_total_mv_yuan
        df["passes_strategy_preference_filter"] = _all_pass(df, TREND_PULLBACK_RULES)
        df["strategy_preference_reason"] = _first_failed_reason(df, TREND_PULLBACK_RULES)
        return df

    def apply_rebound_bottoming(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        config = self.selection_config
        df["preference_recipe"] = "rebound_bottoming_rank"
        df["passes_rebound_drawdown_preference"] = (
            (_num(df, "drawdown_20d") <= config.rebound_min_drawdown_20d)
            & (_num(df, "drawdown_60d") <= config.rebound_min_drawdown_60d)
            & (_num(df, "drawdown_60d") >= config.rebound_max_drawdown_60d)
        )
        df["passes_rebound_position_preference"] = (
            (_num(df, "close_to_ma_60") >= -config.rebound_max_close_to_ma60_below)
            & (_num(df, "close_to_ma_5") >= 0)
            & (_num(df, "close_to_ma_5") <= config.rebound_max_close_to_ma5)
            & (_num(df, "close_to_ma_20") <= config.rebound_max_close_to_ma20)
        )
        df["passes_rebound_stabilization_preference"] = (
            (_num(df, "down_days_10d") <= config.rebound_max_down_days_10d)
            & (_num(df, "consecutive_down_days") <= 4)
            & (_num(df, "return_1d") > 0)
            & (_num(df, "return_1d") <= config.buy_max_return_1d)
            & (_num(df, "return_3d") > 0)
            & (_num(df, "return_3d") <= config.rebound_max_return_3d)
            & (_num(df, "low_to_prev_low") >= 0)
        )
        df["passes_rebound_liquidity_preference"] = (
            (_num(df, "amount_ratio_5d") >= config.rebound_min_amount_ratio_5d)
            & (_num(df, "amount_ratio_5d") <= config.rebound_max_amount_ratio_5d)
            & (_num(df, "volume_ratio").isna() | (_num(df, "volume_ratio") <= config.rebound_max_volume_ratio))
        )
        df["passes_rebound_moneyflow_preference"] = (
            _num(df, "large_net_mf_to_amount").isna()
            | (_num(df, "large_net_mf_to_amount") >= config.rebound_min_large_net_mf_to_amount)
        )
        df["passes_rebound_limit_preference"] = ~_bool(df, "is_limit_up") & ~_bool(df, "is_limit_down")
        df["passes_market_cap_preference"] = _num(df, "total_mv_yuan") >= config.buy_min_total_mv_yuan
        df["passes_strategy_preference_filter"] = _all_pass(df, REBOUND_BOTTOMING_RULES)
        df["strategy_preference_reason"] = _first_failed_reason(df, REBOUND_BOTTOMING_RULES)
        return df


def _all_pass(frame: pd.DataFrame, rules: tuple[FilterRule, ...]) -> pd.Series:
    if not rules:
        return pd.Series(True, index=frame.index)
    result = pd.Series(True, index=frame.index)
    for rule in rules:
        result = result & frame[rule.column].fillna(False).astype(bool)
    return result


def _first_failed_reason(frame: pd.DataFrame, rules: tuple[FilterRule, ...]) -> pd.Series:
    reasons = pd.Series(ELIGIBLE_REASON, index=frame.index, dtype="object")
    for rule in rules:
        reasons = reasons.mask(
            (reasons == ELIGIBLE_REASON) & (~frame[rule.column].fillna(False).astype(bool)),
            rule.reason,
        )
    return reasons


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)
