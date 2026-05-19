from types import SimpleNamespace

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
        self.fina_indicator_calls: list[str] = []

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

    def fetch_index_daily(self, ts_code, start_date, end_date):
        return pd.DataFrame([{"ts_code": ts_code, "trade_date": end_date}])

    def fetch_index_daily_basic(self, trade_date):
        return pd.DataFrame([{"ts_code": "000300.SH", "trade_date": trade_date}])

    def fetch_index_classify(self, src="SW2021"):
        return pd.DataFrame([{"index_code": "801010.SI", "industry_name": "农林牧渔", "level": "L1"}])

    def fetch_index_member_all(self, src="SW2021"):
        return pd.DataFrame([{"ts_code": "300001.SZ", "l1_code": "801010.SI", "l1_name": "农林牧渔", "is_new": "Y"}])

    def fetch_fina_indicator(self, ts_code):
        self.fina_indicator_calls.append(ts_code)
        return pd.DataFrame([{"ts_code": ts_code, "ann_date": "20260430", "roe_waa": 10.0}])


class FakeRepository:
    def __init__(self, complete_cache_dates=None) -> None:
        self.config = SimpleNamespace(market=SimpleNamespace(benchmark="000300.SH"))
        self.complete_cache_dates = set(complete_cache_dates or [])
        self.saved_daily: list[str] = []
        self.saved_daily_basic: list[str] = []
        self.saved_moneyflow: list[str] = []
        self.saved_index_daily = False
        self.saved_index_daily_basic: list[str] = []
        self.saved_index_classify = False
        self.saved_index_member = False
        self.saved_fina_indicator_symbols: list[str] = []
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

    def index_daily_basic_cache_exists(self, trade_date):
        return False

    def fina_indicator_symbol_cache_exists(self, ts_code):
        return False

    def save_daily(self, trade_date, frame):
        self.saved_daily.append(trade_date)

    def save_daily_basic(self, trade_date, frame):
        self.saved_daily_basic.append(trade_date)

    def save_moneyflow(self, trade_date, frame):
        self.saved_moneyflow.append(trade_date)

    def save_limit_list(self, trade_date, frame):
        raise AssertionError("empty limit list should not be saved")

    def save_index_daily(self, index_code, frame):
        self.saved_index_daily = True

    def save_index_daily_basic(self, trade_date, frame):
        self.saved_index_daily_basic.append(trade_date)

    def save_index_classify(self, src, frame):
        self.saved_index_classify = True

    def save_index_member_all(self, src, frame):
        self.saved_index_member = True

    def save_fina_indicator_symbol(self, ts_code, frame):
        self.saved_fina_indicator_symbols.append(ts_code)


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
    assert repository.saved_index_daily is True
    assert repository.saved_index_daily_basic == ["20260511", "20260512"]
    assert repository.saved_index_classify is True
    assert repository.saved_index_member is True
    assert repository.saved_fina_indicator_symbols == []
    assert "20260511" not in client.daily_basic_calls


def test_sync_can_opt_in_to_fina_indicator_symbol_cache() -> None:
    client = FakeTushareClient()
    repository = FakeRepository()

    result = TushareSyncService(client=client, repository=repository).sync(
        start_date="20260511",
        end_date="20260512",
        sync_fina_indicator=True,
        fina_indicator_limit=1,
    )

    assert client.fina_indicator_calls == ["300001.SZ"]
    assert repository.saved_fina_indicator_symbols == ["300001.SZ"]
    assert result.fina_indicator_files == 1


def test_sync_fails_when_missing_required_daily_basic_keeps_timing_out() -> None:
    client = FakeTushareClient(fail_daily_basic_dates={"20260512"})
    repository = FakeRepository(complete_cache_dates={"20260511"})

    with pytest.raises(TushareTransientError):
        TushareSyncService(client=client, repository=repository).sync(
            start_date="20260511",
            end_date="20260512",
        )
