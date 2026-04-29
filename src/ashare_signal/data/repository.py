from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from ashare_signal.config import AppConfig
from ashare_signal.domain.models import Position
from ashare_signal.utils.dates import to_compact_date

if TYPE_CHECKING:
    import pandas as pd


class DataRepository:
    """Owns local project directories and local cache IO."""

    def __init__(self, config: AppConfig, base_dir: Path) -> None:
        self.config = config
        self.base_dir = base_dir

    def ensure_directories(self) -> None:
        for path in (
            self.config.paths.raw_data_dir,
            self.config.paths.processed_data_dir,
            self.config.paths.reports_dir,
            self.config.paths.logs_dir,
        ):
            (self.base_dir / path).mkdir(parents=True, exist_ok=True)

    @property
    def raw_root(self) -> Path:
        return self.base_dir / self.config.paths.raw_data_dir

    @property
    def processed_root(self) -> Path:
        return self.base_dir / self.config.paths.processed_data_dir

    @property
    def tushare_root(self) -> Path:
        return self.raw_root / "tushare"

    def _read_csv(self, path: Path) -> "pd.DataFrame":
        import pandas as pd

        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def _normalize_date_series(self, series):
        normalized = series.fillna("").astype(str).str.replace(".0", "", regex=False)
        return normalized.where(normalized == "", normalized.str.zfill(8))

    def _write_csv(self, frame: "pd.DataFrame", path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path

    def _upsert_csv(
        self,
        frame: "pd.DataFrame",
        path: Path,
        subset: list[str],
        sort_by: list[str] | None = None,
    ) -> Path:
        import pandas as pd

        if path.exists():
            existing = pd.read_csv(path)
            frame = pd.concat([existing, frame], ignore_index=True)

        frame = frame.drop_duplicates(subset=subset, keep="last")
        if sort_by:
            frame = frame.sort_values(sort_by).reset_index(drop=True)
        return self._write_csv(frame, path)

    def save_trade_calendar(
        self,
        frame: "pd.DataFrame",
        exchange: str = "SSE",
    ) -> Path:
        return self._upsert_csv(
            frame=frame,
            path=self.tushare_root / "trade_cal" / f"{exchange}.csv",
            subset=["exchange", "cal_date"],
            sort_by=["cal_date"],
        )

    def load_trade_calendar(self, exchange: str = "SSE") -> "pd.DataFrame":
        frame = self._read_csv(self.tushare_root / "trade_cal" / f"{exchange}.csv")
        frame["cal_date"] = self._normalize_date_series(frame["cal_date"])
        frame["pretrade_date"] = self._normalize_date_series(frame["pretrade_date"])
        frame["is_open"] = frame["is_open"].astype(int)
        return frame

    def save_stock_basic(self, frame: "pd.DataFrame", list_status: str = "L") -> Path:
        return self._upsert_csv(
            frame=frame,
            path=self.tushare_root / "stock_basic" / f"{list_status}.csv",
            subset=["ts_code"],
            sort_by=["ts_code"],
        )

    def load_stock_basic(self, list_status: str = "L") -> "pd.DataFrame":
        frame = self._read_csv(self.tushare_root / "stock_basic" / f"{list_status}.csv")
        frame["list_date"] = self._normalize_date_series(frame["list_date"])
        return frame

    def save_daily(self, trade_date: str, frame: "pd.DataFrame") -> Path:
        return self._write_csv(frame, self.tushare_root / "daily" / f"{trade_date}.csv")

    def save_daily_basic(self, trade_date: str, frame: "pd.DataFrame") -> Path:
        return self._write_csv(frame, self.tushare_root / "daily_basic" / f"{trade_date}.csv")

    def load_daily(self, trade_date: str) -> "pd.DataFrame":
        frame = self._read_csv(self.tushare_root / "daily" / f"{trade_date}.csv")
        frame["trade_date"] = self._normalize_date_series(frame["trade_date"])
        return frame

    def load_daily_basic(self, trade_date: str) -> "pd.DataFrame":
        frame = self._read_csv(self.tushare_root / "daily_basic" / f"{trade_date}.csv")
        frame["trade_date"] = self._normalize_date_series(frame["trade_date"])
        return frame

    def load_daily_for_dates(self, trade_dates: list[str]) -> "pd.DataFrame":
        import pandas as pd

        frames = [self.load_daily(trade_date) for trade_date in trade_dates]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def cached_daily_trade_dates(self) -> list[str]:
        daily_dir = self.tushare_root / "daily"
        daily_basic_dir = self.tushare_root / "daily_basic"
        if not daily_dir.exists() or not daily_basic_dir.exists():
            return []

        daily_dates = {
            path.stem
            for path in daily_dir.glob("*.csv")
            if len(path.stem) == 8 and path.stem.isdigit()
        }
        daily_basic_dates = {
            path.stem
            for path in daily_basic_dir.glob("*.csv")
            if len(path.stem) == 8 and path.stem.isdigit()
        }
        return sorted(daily_dates & daily_basic_dates)

    def complete_daily_cache_dates(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[str]:
        trade_dates = self.cached_daily_trade_dates()
        if start_date is not None:
            trade_dates = [value for value in trade_dates if value >= self.normalize_trade_date(start_date)]
        if end_date is not None:
            trade_dates = [value for value in trade_dates if value <= self.normalize_trade_date(end_date)]

        complete_dates: list[str] = []
        for trade_date in trade_dates:
            try:
                daily = self.load_daily(trade_date)
                daily_basic = self.load_daily_basic(trade_date)
            except FileNotFoundError:
                continue
            if not daily.empty and not daily_basic.empty:
                complete_dates.append(trade_date)
        return complete_dates

    def latest_complete_daily_cache_date(self, end_date: str | None = None) -> str | None:
        complete_dates = self.complete_daily_cache_dates(end_date=end_date)
        if not complete_dates:
            return None
        return complete_dates[-1]

    def recent_open_trade_dates(
        self,
        as_of: str,
        count: int,
        exchange: str = "SSE",
    ) -> list[str]:
        calendar = self.load_trade_calendar(exchange=exchange)
        open_calendar = calendar.loc[calendar["is_open"] == 1, "cal_date"].astype(str)
        trade_dates = [value for value in open_calendar if value <= as_of]
        if len(trade_dates) < count:
            raise ValueError(
                f"Not enough open trade dates in cache for {as_of}: required {count}, got {len(trade_dates)}"
            )
        return trade_dates[-count:]

    def open_trade_dates_between(
        self,
        start_date: str,
        end_date: str,
        exchange: str = "SSE",
    ) -> list[str]:
        calendar = self.load_trade_calendar(exchange=exchange)
        open_calendar = calendar.loc[calendar["is_open"] == 1, "cal_date"].astype(str)
        return [value for value in open_calendar if start_date <= value <= end_date]

    def resolve_trade_date(self, as_of: str, exchange: str = "SSE") -> str:
        calendar = self.load_trade_calendar(exchange=exchange)
        open_calendar = calendar.loc[calendar["is_open"] == 1, "cal_date"].astype(str)
        eligible = [value for value in open_calendar if value <= as_of]
        if not eligible:
            raise ValueError(f"No open trade date found on or before {as_of}")
        return eligible[-1]

    def next_open_trade_date(self, trade_date: str, exchange: str = "SSE") -> str:
        calendar = self.load_trade_calendar(exchange=exchange)
        open_calendar = calendar.loc[calendar["is_open"] == 1, "cal_date"].astype(str).tolist()
        future = [value for value in open_calendar if value > trade_date]
        if not future:
            raise ValueError(f"No open trade date found after {trade_date}")
        return future[0]

    def save_universe_snapshot(self, trade_date: str, frame: "pd.DataFrame") -> Path:
        return self._write_csv(
            frame,
            self.processed_root / "universe" / f"universe_{trade_date}.csv",
        )

    def load_universe_snapshot(self, trade_date: str) -> "pd.DataFrame":
        return self._read_csv(self.processed_root / "universe" / f"universe_{trade_date}.csv")

    def load_positions(self, path: Path) -> list[Position]:
        frame = self._read_csv(path)
        required_columns = {"symbol", "name", "entry_date", "entry_price", "quantity"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"Missing position columns in {path}: {sorted(missing)}")

        frame["entry_date"] = frame["entry_date"].astype(str)
        frame["entry_price"] = frame["entry_price"].astype(float)
        frame["quantity"] = frame["quantity"].astype(int)

        positions: list[Position] = []
        for row in frame.to_dict(orient="records"):
            entry = row["entry_date"]
            normalized = (
                f"{entry[:4]}-{entry[4:6]}-{entry[6:]}"
                if len(entry) == 8 and entry.isdigit()
                else entry
            )
            positions.append(
                Position(
                    symbol=row["symbol"],
                    name=row["name"],
                    entry_date=date.fromisoformat(normalized),
                    entry_price=float(row["entry_price"]),
                    quantity=int(row["quantity"]),
                )
            )
        return positions

    def daily_cache_exists(self, trade_date: str) -> bool:
        return (self.tushare_root / "daily" / f"{trade_date}.csv").exists()

    def daily_basic_cache_exists(self, trade_date: str) -> bool:
        return (self.tushare_root / "daily_basic" / f"{trade_date}.csv").exists()

    def normalize_trade_date(self, value: str) -> str:
        return to_compact_date(value)
