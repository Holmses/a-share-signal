from pathlib import Path
from types import SimpleNamespace

import pytest

from ashare_signal.backtest.tianzhu9_like import Tianzhu9LikeBacktestEngine
from ashare_signal.scheduler.tianzhu9_daily import _resolve_tianzhu9_sync_start_date
from ashare_signal.strategy.tianzhu9_orders import generate_tianzhu9_order_plan


class TinyRepository:
    def __init__(self, cached_dates: list[str]) -> None:
        self._cached_dates = cached_dates

    def complete_daily_cache_dates(self, end_date=None):
        if end_date is None:
            return list(self._cached_dates)
        return [value for value in self._cached_dates if value <= end_date]

    def resolve_trade_date(self, as_of, exchange="SSE"):
        eligible = [value for value in self._cached_dates if value <= str(as_of)]
        if not eligible:
            raise ValueError(as_of)
        return eligible[-1]

    def next_open_trade_date(self, trade_date, exchange="SSE"):
        future = [value for value in self._cached_dates if value > trade_date]
        if not future:
            raise ValueError(trade_date)
        return future[0]

    def recent_open_trade_dates(self, as_of, count, exchange="SSE"):
        eligible = [value for value in self._cached_dates if value <= str(as_of)]
        if len(eligible) < count:
            raise ValueError((as_of, count))
        return eligible[-count:]


def _config():
    return SimpleNamespace(
        backtest=SimpleNamespace(lot_size=100),
        market=SimpleNamespace(max_positions=5),
        pricing=SimpleNamespace(buy_markup=0.003, sell_markdown=0.003),
        filters=SimpleNamespace(min_list_days=60, min_price=3.0),
        paths=SimpleNamespace(reports_dir="reports/generated"),
        runtime=SimpleNamespace(sync_lookback_days=7),
    )


def test_backtest_raises_when_warmup_history_is_insufficient(tmp_path: Path) -> None:
    cached_dates = [f"202505{day:02d}" for day in range(1, 31)]
    engine = Tianzhu9LikeBacktestEngine(
        config=_config(),
        repository=TinyRepository(cached_dates),
        base_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="needs at least 91 complete trade days before start date"):
        engine.run(start_date="2025-05-15", end_date="2025-05-30")


def test_generate_orders_raises_when_signal_history_is_insufficient(tmp_path: Path) -> None:
    cached_dates = [f"202505{day:02d}" for day in range(1, 31)] + ["20250602"]

    with pytest.raises(ValueError, match="needs at least 90 complete trade days before signal date"):
        generate_tianzhu9_order_plan(
            config=_config(),
            repository=TinyRepository(cached_dates),
            base_dir=tmp_path,
            as_of="2025-05-30",
        )


def test_tianzhu9_daily_sync_start_backfills_when_cache_is_too_short() -> None:
    cached_dates = [f"202505{day:02d}" for day in range(1, 31)]

    sync_start = _resolve_tianzhu9_sync_start_date(
        repository=TinyRepository(cached_dates),
        config=_config(),
        end_date="2025-05-30",
        sync_start_date=None,
    )

    assert str(sync_start) < "2025-05-01"
