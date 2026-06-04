from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ashare_signal.config import SelectionConfig


@dataclass(frozen=True, slots=True)
class MarketGateResult:
    gate: str
    allowed: bool
    state: str
    breadth: float
    min_breadth: float
    reason: str


class MarketGate:
    """Market gate shared by selector, research recipes, and later reports."""

    def __init__(self, selection_config: SelectionConfig) -> None:
        self.selection_config = selection_config

    def evaluate(self, universe: pd.DataFrame, signal_type: str | None = None) -> MarketGateResult:
        pool = universe.loc[universe["is_candidate"].fillna(False).astype(bool)].copy()
        gate = self._gate_for_signal(signal_type)
        min_breadth = self._min_breadth_for_gate(gate)
        if pool.empty:
            return MarketGateResult(
                gate=gate,
                allowed=False,
                state="empty",
                breadth=0.0,
                min_breadth=min_breadth,
                reason="candidate_pool_empty",
            )
        breadth = float(((_num(pool, "close_to_ma_20") > 0) & (_num(pool, "momentum_20d") > 0)).mean())
        allowed = breadth >= min_breadth
        if allowed:
            state = "risk_on" if breadth >= self.selection_config.market_min_breadth else "risk_neutral"
            reason = "market_gate_pass"
        else:
            state = "risk_off"
            reason = "market_breadth_below_threshold"
        return MarketGateResult(
            gate=gate,
            allowed=allowed,
            state=state,
            breadth=breadth,
            min_breadth=min_breadth,
            reason=reason,
        )

    def allows_buy(self, universe: pd.DataFrame, signal_type: str | None = None) -> bool:
        return self.evaluate(universe, signal_type=signal_type).allowed

    def _gate_for_signal(self, signal_type: str | None) -> str:
        if signal_type == "rebound_bottoming":
            return "risk_neutral_or_rebound"
        return "risk_on"

    def _min_breadth_for_gate(self, gate: str) -> float:
        if gate == "risk_neutral_or_rebound":
            return float(self.selection_config.rebound_market_min_breadth)
        return float(self.selection_config.market_min_breadth)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")
