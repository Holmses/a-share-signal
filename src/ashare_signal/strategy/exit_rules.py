from __future__ import annotations

from dataclasses import dataclass
import math


TIERED_TRAILING_TAKE_PROFIT_LEVELS: tuple[tuple[float, float], ...] = (
    (0.20, 0.08),
    (0.12, 0.06),
    (0.08, 0.04),
)

SLOW_PROFIT_LOCK_PROFILE = "slow_profit_lock"
LEGACY_EXIT_PROFILE = "legacy"
EXIT_PROFILES = (SLOW_PROFIT_LOCK_PROFILE, LEGACY_EXIT_PROFILE)
DEFAULT_EXIT_PROFILE = LEGACY_EXIT_PROFILE
DEFAULT_HARD_EXIT_DAYS = 23
DEFAULT_FAILURE_EXIT_DAYS: int | None = None
DEFAULT_FAILURE_EXIT_MIN_PEAK_PROFIT_PCT = 0.03
DEFAULT_VOLUME_STALL_EXIT = False
DEFAULT_VOLUME_STALL_RATIO = 1.4

SLOW_PROFIT_LOCK_TRAILING_LEVELS: tuple[tuple[float, float], ...] = (
    (0.30, 0.12),
    (0.20, 0.08),
    (0.12, 0.05),
)
SLOW_PROFIT_LOCK_TRAILING_MIN_DAYS = 8
SLOW_PROFIT_LOCK_MA20_MIN_DAYS = 12
SLOW_PROFIT_LOCK_MA60_MIN_DAYS = 20
SLOW_PROFIT_LOCK_STYLE_MIN_DAYS = 10
SLOW_PROFIT_LOCK_HARD_EXIT_DAYS = 60
SLOW_PROFIT_LOCK_STYLE_MIN_RETURN_20D = -0.01
SLOW_PROFIT_LOCK_STYLE_MIN_BREADTH_20D = 0.48


@dataclass(frozen=True, slots=True)
class TieredTrailingExit:
    should_exit: bool
    peak_profit_pct: float
    drawdown_from_peak_pct: float
    trigger_profit_pct: float | None = None
    trigger_drawdown_pct: float | None = None


@dataclass(frozen=True, slots=True)
class SlowProfitLockExit:
    should_exit: bool
    reason: str
    trailing_exit: TieredTrailingExit | None = None


def tiered_trailing_take_profit(
    *,
    entry_price: float,
    current_close: float,
    highest_price: float | None,
    levels: tuple[tuple[float, float], ...] = TIERED_TRAILING_TAKE_PROFIT_LEVELS,
) -> TieredTrailingExit:
    """Profit-only tiered trailing take-profit signal.

    The rule has no hard stop-loss and no fixed holding-day exit. It only
    exits after the position has reached a configured profit tier and then
    retraced from the peak.
    """
    if _invalid_price(entry_price) or _invalid_price(current_close):
        return TieredTrailingExit(False, 0.0, 0.0)
    peak_price = (
        highest_price
        if highest_price is not None and not _invalid_price(highest_price)
        else entry_price
    )
    peak_price = max(float(peak_price), float(current_close), float(entry_price))
    peak_profit_pct = peak_price / entry_price - 1.0
    drawdown_from_peak_pct = current_close / peak_price - 1.0 if peak_price > 0 else 0.0
    for trigger_profit_pct, trigger_drawdown_pct in levels:
        if peak_profit_pct >= trigger_profit_pct and drawdown_from_peak_pct <= -trigger_drawdown_pct:
            return TieredTrailingExit(
                True,
                peak_profit_pct,
                drawdown_from_peak_pct,
                trigger_profit_pct,
                trigger_drawdown_pct,
            )
    return TieredTrailingExit(False, peak_profit_pct, drawdown_from_peak_pct)


def slow_profit_lock_exit_signal(
    *,
    entry_price: float,
    current_close: float,
    highest_price: float | None,
    holding_days: int,
    ma20: float | None = None,
    ma60: float | None = None,
    return_5d: float | None = None,
    style_return_20d: float | None = None,
    style_breadth_20d: float | None = None,
    hard_exit_days: int | None = SLOW_PROFIT_LOCK_HARD_EXIT_DAYS,
) -> SlowProfitLockExit:
    """Long-hold exit profile selected by the two-year slow profit lock study."""
    if _invalid_price(entry_price) or _invalid_price(current_close):
        return SlowProfitLockExit(False, "")

    if holding_days >= SLOW_PROFIT_LOCK_TRAILING_MIN_DAYS:
        trailing_exit = tiered_trailing_take_profit(
            entry_price=entry_price,
            current_close=current_close,
            highest_price=highest_price,
            levels=SLOW_PROFIT_LOCK_TRAILING_LEVELS,
        )
        if trailing_exit.should_exit:
            return SlowProfitLockExit(True, "slow_profit_lock_trailing", trailing_exit)

    if (
        holding_days >= SLOW_PROFIT_LOCK_MA20_MIN_DAYS
        and _valid_price(ma20)
        and current_close < float(ma20)
        and (return_5d or 0.0) < 0.0
    ):
        return SlowProfitLockExit(True, "slow_profit_lock_ma20_weak")

    if holding_days >= SLOW_PROFIT_LOCK_MA60_MIN_DAYS and _valid_price(ma60) and current_close < float(ma60):
        return SlowProfitLockExit(True, "slow_profit_lock_ma60")

    if (
        holding_days >= SLOW_PROFIT_LOCK_STYLE_MIN_DAYS
        and style_return_20d is not None
        and style_breadth_20d is not None
        and style_return_20d < SLOW_PROFIT_LOCK_STYLE_MIN_RETURN_20D
        and style_breadth_20d < SLOW_PROFIT_LOCK_STYLE_MIN_BREADTH_20D
        and _valid_price(ma20)
        and current_close < float(ma20)
    ):
        return SlowProfitLockExit(True, "slow_profit_lock_style_weak")

    if hard_exit_days is not None and holding_days >= int(hard_exit_days):
        return SlowProfitLockExit(True, "slow_profit_lock_hard60")

    return SlowProfitLockExit(False, "")


def _invalid_price(value: float) -> bool:
    return math.isnan(float(value)) or float(value) <= 0.0


def _valid_price(value: float | None) -> bool:
    return value is not None and not _invalid_price(value)
