from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ashare_signal.domain.models import Position


@dataclass(slots=True)
class PortfolioState:
    positions: list[Position] = field(default_factory=list)
    max_positions: int = 5

    def is_full(self) -> bool:
        return len(self.positions) >= self.max_positions

    def sellable_positions(self, as_of: date, min_holding_days: int) -> list[Position]:
        return [
            position
            for position in self.positions
            if position.holding_days(as_of) >= min_holding_days
        ]

