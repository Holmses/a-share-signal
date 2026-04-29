from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from ashare_signal.config import AppConfig
from ashare_signal.domain.models import Candidate, SignalBoard, TradeSignal


@dataclass(slots=True)
class SignalStrategy:
    config: AppConfig

    @staticmethod
    def _next_business_day(value: date) -> date:
        next_day = value + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

    def pick_buy_signal(self, candidates: list[Candidate]) -> TradeSignal | None:
        if not candidates:
            return None
        chosen = max(candidates, key=lambda item: item.score)
        return TradeSignal(
            action="BUY",
            symbol=chosen.symbol,
            name=chosen.name,
            suggested_price=round(
                chosen.last_close * (1 + self.config.pricing.buy_markup),
                2,
            ),
            score=chosen.score,
            reason=chosen.reason,
            last_close=chosen.last_close,
        )

    def pick_sell_signal(self, candidates: list[Candidate]) -> TradeSignal | None:
        if not candidates:
            return None
        chosen = min(candidates, key=lambda item: item.score)
        return TradeSignal(
            action="SELL",
            symbol=chosen.symbol,
            name=chosen.name,
            suggested_price=round(
                chosen.last_close * (1 - self.config.pricing.sell_markdown),
                2,
            ),
            score=chosen.score,
            reason=chosen.reason,
            last_close=chosen.last_close,
        )

    def build_board(
        self,
        signal_date: date,
        trade_date: date,
        holdings_count: int,
        buy_candidates: list[Candidate],
        sell_candidates: list[Candidate],
        effective_date: date | None = None,
        notes: list[str] | None = None,
    ) -> SignalBoard:
        default_notes = [
            "建议委托价为次交易日限价参考，不保证成交。",
            "若次日股票停牌、涨跌停或跳空过大，应放弃执行。",
        ]
        return SignalBoard(
            signal_date=signal_date,
            trade_date=trade_date,
            effective_date=effective_date or self._next_business_day(signal_date),
            holdings_count=holdings_count,
            buy_signal=self.pick_buy_signal(buy_candidates),
            sell_signal=self.pick_sell_signal(sell_candidates),
            notes=default_notes + (notes or []),
        )
