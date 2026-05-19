from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

from ashare_signal.utils.dates import to_compact_date

if TYPE_CHECKING:
    import pandas as pd


class TushareTransientError(RuntimeError):
    """Raised when a retryable Tushare request keeps failing."""


def _transient_exception_types() -> tuple[type[BaseException], ...]:
    exception_types: list[type[BaseException]] = [TimeoutError]
    try:
        from requests import exceptions as requests_exceptions

        exception_types.extend(
            [
                requests_exceptions.ChunkedEncodingError,
                requests_exceptions.Timeout,
                requests_exceptions.ConnectionError,
            ]
        )
    except Exception:  # pragma: no cover - optional import guard
        pass

    try:
        from urllib3 import exceptions as urllib3_exceptions

        exception_types.extend(
            [
                urllib3_exceptions.TimeoutError,
                urllib3_exceptions.ProtocolError,
            ]
        )
    except Exception:  # pragma: no cover - optional import guard
        pass

    return tuple(exception_types)


@dataclass(slots=True)
class TushareClient:
    """Thin wrapper around the Tushare Pro client."""

    token: str | None
    max_attempts: int = 3
    retry_sleep_seconds: float = 2.0

    def is_configured(self) -> bool:
        return bool(self.token)

    def require_token(self) -> None:
        if not self.token:
            raise RuntimeError("TUSHARE_TOKEN is not configured.")

    def _pro(self):
        import tushare as ts

        self.require_token()
        return ts.pro_api(self.token)

    def _query(self, api_name: str, **kwargs) -> "pd.DataFrame":
        attempts = max(int(self.max_attempts), 1)
        transient_exceptions = _transient_exception_types()
        for attempt in range(1, attempts + 1):
            try:
                return getattr(self._pro(), api_name)(**kwargs)
            except transient_exceptions as error:
                if attempt >= attempts:
                    raise TushareTransientError(
                        f"Tushare {api_name} request failed after {attempts} attempts: {error}"
                    ) from error
                time.sleep(max(float(self.retry_sleep_seconds), 0.0) * attempt)

        raise RuntimeError("unreachable")

    def fetch_trade_calendar(
        self,
        start_date: str,
        end_date: str,
        exchange: str = "SSE",
    ) -> "pd.DataFrame":
        return self._query(
            "trade_cal",
            exchange=exchange,
            start_date=to_compact_date(start_date),
            end_date=to_compact_date(end_date),
            fields="exchange,cal_date,is_open,pretrade_date",
        )

    def fetch_stock_basic(
        self,
        list_status: str = "L",
    ) -> "pd.DataFrame":
        return self._query(
            "stock_basic",
            exchange="",
            list_status=list_status,
            fields=(
                "ts_code,symbol,name,area,industry,fullname,enname,cnspell,"
                "market,exchange,curr_type,list_status,list_date,delist_date,is_hs"
            ),
        )

    def fetch_daily(self, trade_date: str) -> "pd.DataFrame":
        return self._query(
            "daily",
            trade_date=to_compact_date(trade_date),
            fields=(
                "ts_code,trade_date,open,high,low,close,pre_close,"
                "change,pct_chg,vol,amount"
            ),
        )

    def fetch_daily_basic(self, trade_date: str) -> "pd.DataFrame":
        return self._query(
            "daily_basic",
            trade_date=to_compact_date(trade_date),
            fields=(
                "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
                "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
                "free_share,total_mv,circ_mv"
            ),
        )

    def fetch_moneyflow(self, trade_date: str) -> "pd.DataFrame":
        return self._query(
            "moneyflow",
            trade_date=to_compact_date(trade_date),
            fields=(
                "ts_code,trade_date,buy_lg_amount,sell_lg_amount,"
                "buy_elg_amount,sell_elg_amount,net_mf_amount"
            ),
        )

    def fetch_limit_list(self, trade_date: str) -> "pd.DataFrame":
        return self._query(
            "limit_list_d",
            trade_date=to_compact_date(trade_date),
            fields=(
                "trade_date,ts_code,industry,name,pct_chg,amount,limit_amount,"
                "float_mv,total_mv,turnover_ratio,fd_amount,open_times,up_stat,"
                "limit_times,limit"
            ),
        )

    def fetch_index_daily(self, ts_code: str, start_date: str, end_date: str) -> "pd.DataFrame":
        return self._query(
            "index_daily",
            ts_code=ts_code,
            start_date=to_compact_date(start_date),
            end_date=to_compact_date(end_date),
            fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )

    def fetch_index_daily_basic(self, trade_date: str) -> "pd.DataFrame":
        return self._query(
            "index_dailybasic",
            trade_date=to_compact_date(trade_date),
            fields=(
                "ts_code,trade_date,total_mv,float_mv,total_share,float_share,free_share,"
                "turnover_rate,turnover_rate_f,pe,pe_ttm,pb"
            ),
        )

    def fetch_index_classify(self, src: str = "SW2021") -> "pd.DataFrame":
        return self._query("index_classify", src=src)

    def fetch_index_member_all(self, src: str = "SW2021") -> "pd.DataFrame":
        import pandas as pd

        frames = []
        offset = 0
        limit = 3000
        while True:
            frame = self._query("index_member_all", is_new="Y", offset=offset, limit=limit)
            if frame.empty:
                break
            frames.append(frame)
            if len(frame) < limit:
                break
            offset += limit
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fetch_fina_indicator(self, ts_code: str) -> "pd.DataFrame":
        return self._query(
            "fina_indicator",
            ts_code=ts_code,
            fields=(
                "ts_code,ann_date,end_date,roe,roe_waa,roe_dt,roa,roic,"
                "grossprofit_margin,netprofit_margin,debt_to_assets,ocf_to_or,"
                "ocf_to_profit,basic_eps_yoy,dt_eps_yoy,op_yoy,netprofit_yoy,"
                "dt_netprofit_yoy,ocf_yoy"
            ),
        )
