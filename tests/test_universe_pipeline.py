import pandas as pd

from ashare_signal.config import load_config
from ashare_signal.features.pipeline import compute_feature_snapshot
from ashare_signal.strategy.universe import apply_universe_filters


def _write_config(path) -> None:
    path.write_text(
        """
[market]
name = "ashare"
benchmark = "000300.SH"
max_positions = 5
min_position_holding_days = 1

[filters]
min_list_days = 60
min_price = 3.0
min_avg_turnover = 50000000
exclude_st = true
exclude_suspended = true

[pricing]
buy_markup = 0.003
sell_markdown = 0.003
cancel_if_gap_exceeds = 0.02

[strategy]
buy_top_n = 1
sell_top_n = 1
lookback_momentum_days = 20
lookback_short_days = 5
lookback_vol_days = 20

[paths]
raw_data_dir = "data/raw"
processed_data_dir = "data/processed"
reports_dir = "reports/generated"
logs_dir = "logs"
        """.strip(),
        encoding="utf-8",
    )


def test_compute_feature_snapshot_generates_expected_columns(tmp_path) -> None:
    config_path = tmp_path / "strategy.toml"
    _write_config(config_path)
    config = load_config(config_path)

    trade_dates = pd.date_range("2026-03-02", periods=21, freq="B").strftime("%Y%m%d")
    history_rows = []
    for offset, trade_date in enumerate(trade_dates):
        history_rows.append(
            {
                "ts_code": "600036.SH",
                "trade_date": trade_date,
                "open": 40 + offset,
                "high": 40.5 + offset,
                "low": 39.5 + offset,
                "close": 40 + offset,
                "pre_close": 39 + offset,
                "change": 1.0,
                "pct_chg": 2.0,
                "vol": 100000,
                "amount": 80000,
            }
        )
        history_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 12.0,
                "high": 12.2,
                "low": 11.8,
                "close": 12.0,
                "pre_close": 12.0,
                "change": 0.0,
                "pct_chg": 0.0,
                "vol": 100000,
                "amount": 1000,
            }
        )

    history = pd.DataFrame(history_rows)
    daily_basic = pd.DataFrame(
        [
            {
                "ts_code": "600036.SH",
                "trade_date": trade_dates[-1],
                "close": 60.0,
                "turnover_rate": 1.2,
                "turnover_rate_f": 1.0,
                "volume_ratio": 0.9,
                "pe": 8.0,
                "pe_ttm": 7.5,
                "pb": 1.1,
                "ps": 2.0,
                "ps_ttm": 1.9,
                "dv_ratio": 3.0,
                "dv_ttm": 3.2,
                "total_share": 100.0,
                "float_share": 80.0,
                "free_share": 70.0,
                "total_mv": 500000.0,
                "circ_mv": 400000.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_dates[-1],
                "close": 12.0,
                "turnover_rate": 0.5,
                "turnover_rate_f": 0.4,
                "volume_ratio": 0.7,
                "pe": 6.0,
                "pe_ttm": 6.5,
                "pb": 0.8,
                "ps": 1.2,
                "ps_ttm": 1.1,
                "dv_ratio": 2.0,
                "dv_ttm": 2.1,
                "total_share": 100.0,
                "float_share": 80.0,
                "free_share": 70.0,
                "total_mv": 200000.0,
                "circ_mv": 150000.0,
            },
        ]
    )
    stock_basic = pd.DataFrame(
        [
            {
                "ts_code": "600036.SH",
                "symbol": "600036",
                "name": "招商银行",
                "area": "深圳",
                "industry": "银行",
                "fullname": "招商银行股份有限公司",
                "enname": "CMB",
                "cnspell": "zsyh",
                "market": "主板",
                "exchange": "SSE",
                "curr_type": "CNY",
                "list_status": "L",
                "list_date": "20020409",
                "delist_date": "",
                "is_hs": "N",
            },
            {
                "ts_code": "000001.SZ",
                "symbol": "000001",
                "name": "ST平安",
                "area": "深圳",
                "industry": "银行",
                "fullname": "平安银行股份有限公司",
                "enname": "PAB",
                "cnspell": "payh",
                "market": "主板",
                "exchange": "SZSE",
                "curr_type": "CNY",
                "list_status": "L",
                "list_date": "20260201",
                "delist_date": "",
                "is_hs": "N",
            },
        ]
    )

    snapshot = compute_feature_snapshot(
        history=history,
        daily_basic=daily_basic,
        stock_basic=stock_basic,
        as_of_trade_date=trade_dates[-1],
        config=config,
    )

    assert "momentum_20d" in snapshot.columns
    assert "avg_amount_20d_yuan" in snapshot.columns
    assert "total_mv_yuan" in snapshot.columns
    row = snapshot.loc[snapshot["ts_code"] == "600036.SH"].iloc[0]
    assert bool(row["is_suspended"]) is False
    assert row["avg_amount_20d_yuan"] == 80000000.0


def test_apply_universe_filters_marks_non_candidates(tmp_path) -> None:
    config_path = tmp_path / "strategy.toml"
    _write_config(config_path)
    config = load_config(config_path)

    snapshot = pd.DataFrame(
        [
            {
                "ts_code": "600036.SH",
                "exchange": "SSE",
                "market": "主板",
                "name": "招商银行",
                "is_st": False,
                "is_suspended": False,
                "listed_days": 5000,
                "close": 42.0,
                "avg_amount_20d_yuan": 80000000.0,
                "momentum_20d_rank_pct": 0.9,
            },
            {
                "ts_code": "000001.SZ",
                "exchange": "SZSE",
                "market": "主板",
                "name": "ST平安",
                "is_st": True,
                "is_suspended": False,
                "listed_days": 10,
                "close": 2.5,
                "avg_amount_20d_yuan": 1000000.0,
                "momentum_20d_rank_pct": 0.2,
            },
        ]
    )

    universe = apply_universe_filters(snapshot, config)

    eligible = universe.loc[universe["ts_code"] == "600036.SH"].iloc[0]
    rejected = universe.loc[universe["ts_code"] == "000001.SZ"].iloc[0]

    assert bool(eligible["is_candidate"]) is True
    assert bool(rejected["is_candidate"]) is False
    assert rejected["exclude_reason"] == "st_stock"
