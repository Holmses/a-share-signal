from __future__ import annotations

from datetime import date, datetime


def to_compact_date(value: date | datetime | str) -> str:
    if isinstance(value, str):
        if len(value) == 8 and value.isdigit():
            return value
        return date.fromisoformat(value).strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.date().strftime("%Y%m%d")
    return value.strftime("%Y%m%d")


def parse_compact_date(value: str | int | date | datetime) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value), "%Y%m%d").date()
