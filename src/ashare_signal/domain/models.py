from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class Candidate:
    symbol: str
    name: str
    score: float
    reason: str
    last_close: float


@dataclass(slots=True)
class Position:
    symbol: str
    name: str
    entry_date: date
    entry_price: float
    quantity: int = 0

    def holding_days(self, as_of: date) -> int:
        return (as_of - self.entry_date).days


@dataclass(slots=True)
class TradeSignal:
    action: str
    symbol: str
    name: str
    suggested_price: float
    score: float
    reason: str
    last_close: float | None = None


@dataclass(slots=True)
class SignalBoard:
    signal_date: date
    effective_date: date
    trade_date: date
    holdings_count: int
    buy_signal: TradeSignal | None
    sell_signal: TradeSignal | None
    notes: list[str] = field(default_factory=list)
