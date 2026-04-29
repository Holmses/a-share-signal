from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ashare_signal.utils.dates import to_compact_date

if TYPE_CHECKING:
    import pandas as pd


@dataclass(slots=True)
class TushareClient:
    """Thin wrapper around the Tushare Pro client."""

    token: str | None

    def is_configured(self) -> bool:
        return bool(self.token)

    def require_token(self) -> None:
        if not self.token:
            raise RuntimeError("TUSHARE_TOKEN is not configured.")

    def _pro(self):
        import tushare as ts

        self.require_token()
        return ts.pro_api(self.token)

    def fetch_trade_calendar(
        self,
        start_date: str,
        end_date: str,
        exchange: str = "SSE",
    ) -> "pd.DataFrame":
        return self._pro().trade_cal(
            exchange=exchange,
            start_date=to_compact_date(start_date),
            end_date=to_compact_date(end_date),
            fields="exchange,cal_date,is_open,pretrade_date",
        )

    def fetch_stock_basic(
        self,
        list_status: str = "L",
    ) -> "pd.DataFrame":
        return self._pro().stock_basic(
            exchange="",
            list_status=list_status,
            fields=(
                "ts_code,symbol,name,area,industry,fullname,enname,cnspell,"
                "market,exchange,curr_type,list_status,list_date,delist_date,is_hs"
            ),
        )

    def fetch_daily(self, trade_date: str) -> "pd.DataFrame":
        return self._pro().daily(
            trade_date=to_compact_date(trade_date),
            fields=(
                "ts_code,trade_date,open,high,low,close,pre_close,"
                "change,pct_chg,vol,amount"
            ),
        )

    def fetch_daily_basic(self, trade_date: str) -> "pd.DataFrame":
        return self._pro().daily_basic(
            trade_date=to_compact_date(trade_date),
            fields=(
                "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
                "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
                "free_share,total_mv,circ_mv"
            ),
        )
