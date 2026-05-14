import pandas as pd
import pytest

from ashare_signal.backtest.selection_event_study import _classify_board
from ashare_signal.backtest.selection_event_study import parse_csv_values, parse_horizons


def test_classify_board_uses_market_and_exchange() -> None:
    assert _classify_board(pd.Series({"ts_code": "300001.SZ", "market": "创业板", "exchange": "SZSE"})) == "chinext"
    assert _classify_board(pd.Series({"ts_code": "688001.SH", "market": "科创板", "exchange": "SSE"})) == "star"
    assert _classify_board(pd.Series({"ts_code": "830001.BJ", "market": "北交所", "exchange": "BSE"})) == "bse"
    assert _classify_board(pd.Series({"ts_code": "600000.SH", "market": "主板", "exchange": "SSE"})) == "main"


def test_parse_csv_values_dedupes_and_normalizes() -> None:
    assert parse_csv_values("main,chinext,quality-strict,main", ("star",)) == [
        "main",
        "chinext",
        "quality_strict",
    ]


def test_parse_horizons_requires_positive_values() -> None:
    assert parse_horizons("10,1,5,5", (1, 3)) == [1, 5, 10]
    with pytest.raises(ValueError, match="positive integers"):
        parse_horizons("1,0", (1, 3))
