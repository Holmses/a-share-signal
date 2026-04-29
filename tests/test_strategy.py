from datetime import date

from ashare_signal.config import load_config
from ashare_signal.domain.models import Candidate
from ashare_signal.strategy.signal_board import SignalStrategy


def test_strategy_picks_highest_buy_score(tmp_path) -> None:
    config_path = tmp_path / "strategy.toml"
    config_path.write_text(
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

    strategy = SignalStrategy(load_config(config_path))
    signal = strategy.pick_buy_signal(
        [
            Candidate("000001.SZ", "平安银行", 0.8, "A", 12.0),
            Candidate("600036.SH", "招商银行", 0.9, "B", 42.0),
        ]
    )

    assert signal is not None
    assert signal.symbol == "600036.SH"
    assert signal.suggested_price == 42.13


def test_strategy_uses_next_business_day(tmp_path) -> None:
    config_path = tmp_path / "strategy.toml"
    config_path.write_text(
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

    strategy = SignalStrategy(load_config(config_path))
    board = strategy.build_board(
        signal_date=date(2026, 4, 3),
        trade_date=date(2026, 4, 3),
        holdings_count=5,
        buy_candidates=[],
        sell_candidates=[],
    )

    assert board.effective_date.isoformat() == "2026-04-06"
