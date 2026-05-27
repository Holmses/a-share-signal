from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_signal.data.sync import TushareSyncService
from ashare_signal.data.tushare_client import TushareTransientError


class FakeTushareClient:
    def __init__(self, fail_daily_dates=None, fail_daily_basic_dates=None) -> None:
        self.fail_daily_dates = set(fail_daily_dates or [])
        self.fail_daily_basic_dates = set(fail_daily_basic_dates or [])
        self.calendar_calls: list[tuple[str, str]] = []
        self.stock_basic_calls: list[str] = []
        self.daily_calls: list[str] = []
        self.daily_basic_calls: list[str] = []
        self.moneyflow_calls: list[str] = []
        self.limit_list_calls: list[str] = []
        self.index_daily_calls: list[tuple[str, str, str]] = []
        self.index_daily_basic_calls: list[str] = []
        self.index_classify_calls: list[str] = []
        self.index_member_all_calls: list[str] = []
        self.fina_indicator_calls: list[str] = []

    def fetch_trade_calendar(self, start_date, end_date, exchange="SSE"):
        self.calendar_calls.append((start_date, end_date))
        return pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20260511", "is_open": 1, "pretrade_date": "20260508"},
                {"exchange": "SSE", "cal_date": "20260512", "is_open": 1, "pretrade_date": "20260511"},
            ]
        )

    def fetch_stock_basic(self, list_status="L"):
        self.stock_basic_calls.append(list_status)
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
        self.moneyflow_calls.append(trade_date)
        return pd.DataFrame([{"ts_code": "300001.SZ", "trade_date": trade_date}])

    def fetch_limit_list(self, trade_date):
        self.limit_list_calls.append(trade_date)
        return pd.DataFrame()

    def fetch_index_daily(self, ts_code, start_date, end_date):
        self.index_daily_calls.append((ts_code, start_date, end_date))
        return pd.DataFrame([{"ts_code": ts_code, "trade_date": end_date}])

    def fetch_index_daily_basic(self, trade_date):
        self.index_daily_basic_calls.append(trade_date)
        return pd.DataFrame([{"ts_code": "000300.SH", "trade_date": trade_date}])

    def fetch_index_classify(self, src="SW2021"):
        self.index_classify_calls.append(src)
        return pd.DataFrame([{"index_code": "801010.SI", "industry_name": "农林牧渔", "level": "L1"}])

    def fetch_index_member_all(self, src="SW2021"):
        self.index_member_all_calls.append(src)
        return pd.DataFrame([{"ts_code": "300001.SZ", "l1_code": "801010.SI", "l1_name": "农林牧渔", "is_new": "Y"}])

    def fetch_fina_indicator(self, ts_code):
        self.fina_indicator_calls.append(ts_code)
        return pd.DataFrame([{"ts_code": ts_code, "ann_date": "20260430", "roe_waa": 10.0}])


class FakeRepository:
    def __init__(
        self,
        complete_cache_dates=None,
        cached_calendar=False,
        cached_stock_basic=False,
        cached_moneyflow_dates=None,
        cached_limit_list_dates=None,
        cached_index_daily=False,
        cached_index_daily_dates=None,
        cached_index_daily_basic_dates=None,
        cached_index_classify=False,
        cached_index_member=False,
    ) -> None:
        self.config = SimpleNamespace(market=SimpleNamespace(benchmark="000300.SH"))
        self.complete_cache_dates = set(complete_cache_dates or [])
        self.cached_calendar = cached_calendar
        self.cached_stock_basic = cached_stock_basic
        self.cached_moneyflow_dates = set(cached_moneyflow_dates or [])
        self.cached_limit_list_dates = set(cached_limit_list_dates or [])
        self.cached_index_daily = cached_index_daily
        self.cached_index_daily_dates = sorted(cached_index_daily_dates or ["20260511", "20260512"])
        self.cached_index_daily_basic_dates = set(cached_index_daily_basic_dates or [])
        self.cached_index_classify = cached_index_classify
        self.cached_index_member = cached_index_member
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

    def load_trade_calendar(self, exchange="SSE"):
        if not self.cached_calendar:
            raise FileNotFoundError
        return pd.DataFrame(
            [
                {"exchange": exchange, "cal_date": "20260511", "is_open": 1, "pretrade_date": "20260508"},
                {"exchange": exchange, "cal_date": "20260512", "is_open": 1, "pretrade_date": "20260511"},
            ]
        )

    def load_stock_basic(self, list_status="L"):
        if not self.cached_stock_basic:
            raise FileNotFoundError
        return pd.DataFrame([{"ts_code": "300001.SZ"}])

    def load_index_daily(self, index_code):
        if not self.cached_index_daily:
            raise FileNotFoundError
        return pd.DataFrame(
            [
                {"ts_code": index_code, "trade_date": trade_date}
                for trade_date in self.cached_index_daily_dates
            ]
        )

    def save_trade_calendar(self, frame, exchange="SSE"):
        self.saved_calendar = True

    def save_stock_basic(self, frame, list_status="L"):
        self.saved_stock_basic = True

    def daily_cache_exists(self, trade_date):
        return trade_date in self.complete_cache_dates

    def daily_basic_cache_exists(self, trade_date):
        return trade_date in self.complete_cache_dates

    def moneyflow_cache_exists(self, trade_date):
        return trade_date in self.cached_moneyflow_dates

    def limit_list_cache_exists(self, trade_date):
        return trade_date in self.cached_limit_list_dates

    def index_daily_cache_exists(self, index_code):
        return self.cached_index_daily

    def index_daily_basic_cache_exists(self, trade_date):
        return trade_date in self.cached_index_daily_basic_dates

    def index_classify_cache_exists(self, src="SW2021"):
        return self.cached_index_classify

    def index_member_all_cache_exists(self, src="SW2021"):
        return self.cached_index_member

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


def test_sync_prefers_repository_cache_before_fetching_api() -> None:
    client = FakeTushareClient()
    repository = FakeRepository(
        complete_cache_dates={"20260511", "20260512"},
        cached_calendar=True,
        cached_stock_basic=True,
        cached_moneyflow_dates={"20260511", "20260512"},
        cached_limit_list_dates={"20260511", "20260512"},
        cached_index_daily=True,
        cached_index_daily_basic_dates={"20260511", "20260512"},
        cached_index_classify=True,
        cached_index_member=True,
    )

    result = TushareSyncService(client=client, repository=repository).sync(
        start_date="20260511",
        end_date="20260512",
    )

    assert result.daily_files == 0
    assert result.daily_basic_files == 0
    assert result.moneyflow_files == 0
    assert result.limit_list_files == 0
    assert result.index_daily_files == 0
    assert result.index_daily_basic_files == 0
    assert result.index_classify_files == 0
    assert result.index_member_files == 0
    assert client.calendar_calls == []
    assert client.stock_basic_calls == []
    assert client.daily_calls == []
    assert client.daily_basic_calls == []
    assert client.moneyflow_calls == []
    assert client.limit_list_calls == []
    assert client.index_daily_calls == []
    assert client.index_daily_basic_calls == []
    assert client.index_classify_calls == []
    assert client.index_member_all_calls == []


def test_sync_fetches_index_daily_when_cache_does_not_cover_required_dates() -> None:
    client = FakeTushareClient()
    repository = FakeRepository(
        complete_cache_dates={"20260511", "20260512"},
        cached_calendar=True,
        cached_stock_basic=True,
        cached_moneyflow_dates={"20260511", "20260512"},
        cached_limit_list_dates={"20260511", "20260512"},
        cached_index_daily=True,
        cached_index_daily_dates={"20260511"},
        cached_index_daily_basic_dates={"20260511", "20260512"},
        cached_index_classify=True,
        cached_index_member=True,
    )

    result = TushareSyncService(client=client, repository=repository).sync(
        start_date="20260511",
        end_date="20260512",
    )

    assert result.index_daily_files == 1
    assert client.index_daily_calls == [("000300.SH", "20260511", "20260512")]


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
