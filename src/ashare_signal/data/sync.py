from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ashare_signal.data.repository import DataRepository
from ashare_signal.data.tushare_client import TushareClient, TushareTransientError
from ashare_signal.utils.dates import to_compact_date


@dataclass(slots=True)
class SyncResult:
    start_date: str
    end_date: str
    calendar_end_date: str
    open_trade_days: int
    stock_count: int
    daily_files: int
    daily_basic_files: int
    moneyflow_files: int
    limit_list_files: int
    index_daily_files: int
    index_daily_basic_files: int
    index_classify_files: int
    index_member_files: int
    fina_indicator_files: int


class TushareSyncService:
    def __init__(self, client: TushareClient, repository: DataRepository) -> None:
        self.client = client
        self.repository = repository

    def sync(
        self,
        start_date: str,
        end_date: str,
        exchange: str = "SSE",
        calendar_end_date: str | None = None,
        sync_fina_indicator: bool = False,
        fina_indicator_limit: int | None = None,
        force_fina_indicator: bool = False,
    ) -> SyncResult:
        normalized_start_date = to_compact_date(start_date)
        data_end_date = to_compact_date(end_date)
        resolved_calendar_end_date = to_compact_date(calendar_end_date or end_date)

        calendar = self._load_or_fetch_trade_calendar(
            start_date=normalized_start_date,
            end_date=resolved_calendar_end_date,
            exchange=exchange,
        )

        stock_basic = self._load_or_fetch_stock_basic(list_status="L")

        calendar_dates = self._normalize_date_series(calendar["cal_date"])
        open_dates = calendar_dates.loc[
            (calendar["is_open"].astype(int) == 1) & (calendar_dates <= data_end_date)
        ].sort_values().tolist()

        primary_benchmark = self.repository.config.market.benchmark
        benchmarks = tuple(
            dict.fromkeys(
                getattr(self.repository.config.market, "benchmarks", ())
                or (primary_benchmark,)
            )
        )
        if primary_benchmark not in benchmarks:
            benchmarks = (primary_benchmark, *benchmarks)
        index_daily_files = 0
        index_daily_basic_files = 0
        index_classify_files = 0
        index_member_files = 0
        fina_indicator_files = 0

        required_index_start = open_dates[0] if open_dates else normalized_start_date
        required_index_end = open_dates[-1] if open_dates else data_end_date
        for benchmark in benchmarks:
            if self._index_daily_cache_covers(
                index_code=benchmark,
                start_date=required_index_start,
                end_date=required_index_end,
            ):
                continue
            try:
                index_daily = self.client.fetch_index_daily(
                    ts_code=benchmark,
                    start_date=normalized_start_date,
                    end_date=data_end_date,
                )
            except Exception:
                index_daily = None
            if index_daily is not None and not index_daily.empty:
                self.repository.save_index_daily(benchmark, index_daily)
                index_daily_files += 1

        if not self.repository.index_classify_cache_exists(src="SW2021"):
            try:
                index_classify = self.client.fetch_index_classify(src="SW2021")
            except Exception:
                index_classify = None
            if index_classify is not None and not index_classify.empty:
                self.repository.save_index_classify("SW2021", index_classify)
                index_classify_files = 1

        if not self.repository.index_member_all_cache_exists(src="SW2021"):
            try:
                index_member_all = self.client.fetch_index_member_all(src="SW2021")
            except Exception:
                index_member_all = None
            if index_member_all is not None and not index_member_all.empty:
                self.repository.save_index_member_all("SW2021", index_member_all)
                index_member_files = 1

        if sync_fina_indicator and "ts_code" in stock_basic.columns:
            symbols = sorted(stock_basic["ts_code"].dropna().astype(str).unique())
            if fina_indicator_limit is not None:
                symbols = symbols[: max(int(fina_indicator_limit), 0)]
            for ts_code in symbols:
                if (
                    not force_fina_indicator
                    and self.repository.fina_indicator_symbol_cache_exists(ts_code)
                ):
                    continue
                try:
                    fina_indicator = self.client.fetch_fina_indicator(ts_code=ts_code)
                except Exception:
                    continue
                if fina_indicator is not None and not fina_indicator.empty:
                    self.repository.save_fina_indicator_symbol(ts_code, fina_indicator)
                    fina_indicator_files += 1

        daily_files = 0
        daily_basic_files = 0
        moneyflow_files = 0
        limit_list_files = 0
        for trade_date in open_dates:
            daily_exists = self.repository.daily_cache_exists(trade_date)
            daily_basic_exists = self.repository.daily_basic_cache_exists(trade_date)
            moneyflow_exists = self.repository.moneyflow_cache_exists(trade_date)
            limit_list_exists = self.repository.limit_list_cache_exists(trade_date)

            if not daily_exists:
                daily = self.client.fetch_daily(trade_date=trade_date)
                if not daily.empty:
                    self.repository.save_daily(trade_date, daily)
                    daily_files += 1

            if not daily_basic_exists:
                daily_basic = self.client.fetch_daily_basic(trade_date=trade_date)
                if not daily_basic.empty:
                    self.repository.save_daily_basic(trade_date, daily_basic)
                    daily_basic_files += 1

            if not moneyflow_exists:
                try:
                    moneyflow = self.client.fetch_moneyflow(trade_date=trade_date)
                except TushareTransientError:
                    moneyflow = None
                if moneyflow is not None and not moneyflow.empty:
                    self.repository.save_moneyflow(trade_date, moneyflow)
                    moneyflow_files += 1

            if not limit_list_exists:
                try:
                    limit_list = self.client.fetch_limit_list(trade_date=trade_date)
                except TushareTransientError:
                    limit_list = self.client.fetch_limit_list(trade_date=trade_date)
                except Exception:
                    limit_list = None
                if limit_list is not None and not limit_list.empty:
                    self.repository.save_limit_list(trade_date, limit_list)
                    limit_list_files += 1

            if not self.repository.index_daily_basic_cache_exists(trade_date):
                try:
                    index_daily_basic = self.client.fetch_index_daily_basic(trade_date=trade_date)
                except Exception:
                    index_daily_basic = None
                if index_daily_basic is not None and not index_daily_basic.empty:
                    self.repository.save_index_daily_basic(trade_date, index_daily_basic)
                    index_daily_basic_files += 1

        return SyncResult(
            start_date=to_compact_date(start_date),
            end_date=data_end_date,
            calendar_end_date=resolved_calendar_end_date,
            open_trade_days=len(open_dates),
            stock_count=len(stock_basic),
            daily_files=daily_files,
            daily_basic_files=daily_basic_files,
            moneyflow_files=moneyflow_files,
            limit_list_files=limit_list_files,
            index_daily_files=index_daily_files,
            index_daily_basic_files=index_daily_basic_files,
            index_classify_files=index_classify_files,
            index_member_files=index_member_files,
            fina_indicator_files=fina_indicator_files,
        )

    def _load_or_fetch_trade_calendar(
        self,
        start_date: str,
        end_date: str,
        exchange: str,
    ) -> "pd.DataFrame":
        try:
            calendar = self.repository.load_trade_calendar(exchange=exchange)
            cached_dates = self._normalize_date_series(calendar["cal_date"])
            if not calendar.empty and cached_dates.min() <= start_date and cached_dates.max() >= end_date:
                return calendar
        except FileNotFoundError:
            pass

        calendar = self.client.fetch_trade_calendar(
            start_date=start_date,
            end_date=end_date,
            exchange=exchange,
        )
        self.repository.save_trade_calendar(calendar, exchange=exchange)
        return calendar

    def _load_or_fetch_stock_basic(self, list_status: str = "L") -> "pd.DataFrame":
        try:
            stock_basic = self.repository.load_stock_basic(list_status=list_status)
            if not stock_basic.empty:
                return stock_basic
        except FileNotFoundError:
            pass

        stock_basic = self.client.fetch_stock_basic(list_status=list_status)
        self.repository.save_stock_basic(stock_basic, list_status=list_status)
        return stock_basic

    def _index_daily_cache_covers(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        try:
            index_daily = self.repository.load_index_daily(index_code)
        except FileNotFoundError:
            return False
        if index_daily.empty or "trade_date" not in index_daily.columns:
            return False
        cached_dates = self._normalize_date_series(index_daily["trade_date"])
        return cached_dates.min() <= start_date and cached_dates.max() >= end_date

    @staticmethod
    def _normalize_date_series(series: "pd.Series") -> "pd.Series":
        normalized = series.fillna("").astype(str).str.replace(".0", "", regex=False)
        return normalized.where(normalized == "", normalized.str.zfill(8))
