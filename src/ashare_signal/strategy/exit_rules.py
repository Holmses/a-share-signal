from __future__ import annotations

from dataclasses import dataclass
import math


TIERED_TRAILING_TAKE_PROFIT_LEVELS: tuple[tuple[float, float], ...] = (
    (0.20, 0.08),
    (0.12, 0.06),
    (0.08, 0.04),
)


@dataclass(frozen=True, slots=True)
class TieredTrailingExit:
    should_exit: bool
    peak_profit_pct: float
    drawdown_from_peak_pct: float
    trigger_profit_pct: float | None = None
    trigger_drawdown_pct: float | None = None


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


def _invalid_price(value: float) -> bool:
    return math.isnan(float(value)) or float(value) <= 0.0
