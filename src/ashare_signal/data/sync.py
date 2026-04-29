from __future__ import annotations

from dataclasses import dataclass

from ashare_signal.data.repository import DataRepository
from ashare_signal.data.tushare_client import TushareClient
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
    ) -> SyncResult:
        data_end_date = to_compact_date(end_date)
        resolved_calendar_end_date = to_compact_date(calendar_end_date or end_date)
        calendar = self.client.fetch_trade_calendar(
            start_date=start_date,
            end_date=resolved_calendar_end_date,
            exchange=exchange,
        )
        self.repository.save_trade_calendar(calendar, exchange=exchange)

        stock_basic = self.client.fetch_stock_basic(list_status="L")
        self.repository.save_stock_basic(stock_basic, list_status="L")

        calendar_dates = (
            calendar["cal_date"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(8)
        )
        open_dates = calendar_dates.loc[
            (calendar["is_open"].astype(int) == 1) & (calendar_dates <= data_end_date)
        ].sort_values().tolist()

        daily_files = 0
        daily_basic_files = 0
        for trade_date in open_dates:
            daily = self.client.fetch_daily(trade_date=trade_date)
            if not daily.empty:
                self.repository.save_daily(trade_date, daily)
                daily_files += 1

            daily_basic = self.client.fetch_daily_basic(trade_date=trade_date)
            if not daily_basic.empty:
                self.repository.save_daily_basic(trade_date, daily_basic)
                daily_basic_files += 1

        return SyncResult(
            start_date=to_compact_date(start_date),
            end_date=data_end_date,
            calendar_end_date=resolved_calendar_end_date,
            open_trade_days=len(open_dates),
            stock_count=len(stock_basic),
            daily_files=daily_files,
            daily_basic_files=daily_basic_files,
        )
