import pandas as pd
import pytest

from ashare_signal.data.sync import TushareSyncService
from ashare_signal.data.tushare_client import TushareTransientError


class FakeTushareClient:
    def __init__(self, fail_daily_dates=None, fail_daily_basic_dates=None) -> None:
        self.fail_daily_dates = set(fail_daily_dates or [])
        self.fail_daily_basic_dates = set(fail_daily_basic_dates or [])
        self.daily_calls: list[str] = []
        self.daily_basic_calls: list[str] = []

    def fetch_trade_calendar(self, start_date, end_date, exchange="SSE"):
        return pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20260511", "is_open": 1, "pretrade_date": "20260508"},
                {"exchange": "SSE", "cal_date": "20260512", "is_open": 1, "pretrade_date": "20260511"},
            ]
        )

    def fetch_stock_basic(self, list_status="L"):
        return pd.DataFrame([{"ts_code": "300001.SZ"}])

    def fetch_daily(self, trade_date):
        self.daily_calls.append(trade_date)
        if trade_date in self.fail_daily_dates:
            raise TushareTransientError("daily timeout")
        return pd.DataFrame([{"ts_code": "300001.SZ", "trade_date": trade_date}])

    def fetch_daily_basic(self, trade_date):
        self.daily_basic_calls.append(trade_date)
        if trade_date in self.fail_daily_basic_dates:
            raise TushareTransientError("daily_basic timeout")
        return pd.DataFrame([{"ts_code": "300001.SZ", "trade_date": trade_date}])

    def fetch_moneyflow(self, trade_date):
        return pd.DataFrame([{"ts_code": "300001.SZ", "trade_date": trade_date}])

    def fetch_limit_list(self, trade_date):
        return pd.DataFrame()


class FakeRepository:
    def __init__(self, complete_cache_dates=None) -> None:
        self.complete_cache_dates = set(complete_cache_dates or [])
        self.saved_daily: list[str] = []
        self.saved_daily_basic: list[str] = []
        self.saved_moneyflow: list[str] = []
        self.saved_calendar = False
        self.saved_stock_basic = False

    def save_trade_calendar(self, frame, exchange="SSE"):
        self.saved_calendar = True

    def save_stock_basic(self, frame, list_status="L"):
        self.saved_stock_basic = True

    def daily_cache_exists(self, trade_date):
        return trade_date in self.complete_cache_dates

    def daily_basic_cache_exists(self, trade_date):
        return trade_date in self.complete_cache_dates

    def moneyflow_cache_exists(self, trade_date):
        return False

    def limit_list_cache_exists(self, trade_date):
        return False

    def save_daily(self, trade_date, frame):
        self.saved_daily.append(trade_date)

    def save_daily_basic(self, trade_date, frame):
        self.saved_daily_basic.append(trade_date)

    def save_moneyflow(self, trade_date, frame):
        self.saved_moneyflow.append(trade_date)

    def save_limit_list(self, trade_date, frame):
        raise AssertionError("empty limit list should not be saved")


def test_sync_uses_existing_complete_cache_after_transient_refresh_failure() -> None:
    client = FakeTushareClient(fail_daily_dates={"20260511"})
    repository = FakeRepository(complete_cache_dates={"20260511"})

    result = TushareSyncService(client=client, repository=repository).sync(
        start_date="20260511",
        end_date="20260512",
    )

    assert result.open_trade_days == 2
    assert repository.saved_calendar is True
    assert repository.saved_stock_basic is True
    assert repository.saved_daily == ["20260512"]
    assert repository.saved_daily_basic == ["20260512"]
    assert "20260511" not in client.daily_basic_calls


def test_sync_fails_when_missing_required_daily_basic_keeps_timing_out() -> None:
    client = FakeTushareClient(fail_daily_basic_dates={"20260512"})
    repository = FakeRepository(complete_cache_dates={"20260511"})

    with pytest.raises(TushareTransientError):
        TushareSyncService(client=client, repository=repository).sync(
            start_date="20260511",
            end_date="20260512",
        )
