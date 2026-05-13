from types import SimpleNamespace
import json

import pandas as pd
import pytest

from ashare_signal.portfolio.tianzhu9_simulator import Tianzhu9PaperBroker


class FakeRepository:
    def __init__(self, daily_by_date) -> None:
        self.daily_by_date = daily_by_date

    def load_daily(self, trade_date):
        if trade_date not in self.daily_by_date:
            raise FileNotFoundError(trade_date)
        return self.daily_by_date[trade_date].copy()


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        market=SimpleNamespace(max_positions=5),
        paths=SimpleNamespace(reports_dir="reports/generated"),
        runtime=SimpleNamespace(timezone="Asia/Shanghai"),
        backtest=SimpleNamespace(
            initial_cash=1_000_000.0,
            commission_rate=0.0003,
            stamp_duty_rate=0.001,
            lot_size=100,
        ),
    )


def test_paper_broker_settles_pending_buy_plan_into_positions(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "signal_trade_date": "20260511",
                "planned_trade_date": "20260512",
                "buy_orders": [
                    {
                        "action": "BUY",
                        "symbol": "301396.SZ",
                        "name": "宏景科技",
                        "limit_price": 100.0,
                        "quantity": None,
                        "rank": 1,
                        "score": 0.98,
                    }
                ],
                "sell_orders": [],
                "hold_orders": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repository = FakeRepository(
        {
            "20260512": pd.DataFrame(
                [
                    {
                        "ts_code": "301396.SZ",
                        "trade_date": "20260512",
                        "open": 99.0,
                        "high": 102.0,
                        "low": 98.0,
                        "close": 101.0,
                    }
                ]
            )
        }
    )
    broker = Tianzhu9PaperBroker(
        config=_config(),
        repository=repository,
        base_dir=tmp_path,
        hold_days=2,
    )

    staged = broker.stage_plan(plan_path, as_of_trade_date="20260511")
    settled = broker.settle_pending_plan(as_of_trade_date="20260512")

    positions = pd.read_csv(staged.positions_path)
    trades = pd.read_csv(settled.trades_path)
    state = json.loads(settled.state_path.read_text(encoding="utf-8"))

    assert staged.pending_plan_date == "20260512"
    assert settled.executed_trades == 1
    assert positions.loc[0, "symbol"] == "301396.SZ"
    assert positions.loc[0, "entry_price"] == 99.0
    assert positions.loc[0, "quantity"] == 5000
    assert trades.loc[0, "action"] == "BUY"
    assert state["cash"] == 504851.5
    assert state["equity"] == 1009851.5
    assert settled.initial_cash == 1_000_000.0
    assert settled.positions_market_value == 505000.0
    assert settled.daily_pnl == 9851.5
    assert settled.total_return == pytest.approx(0.0098515)
    assert settled.positions[0].unrealized_pnl == 10000.0
    assert settled.positions[0].unrealized_return == pytest.approx(101.0 / 99.0 - 1.0)
    assert settled.positions[0].holding_days == 1


def test_paper_broker_can_settle_generated_plan_when_pending_file_is_missing(tmp_path) -> None:
    reports_dir = tmp_path / "reports" / "generated" / "tianzhu9-orders"
    reports_dir.mkdir(parents=True)
    (reports_dir / "tianzhu9-orders-20260511.json").write_text(
        json.dumps(
            {
                "signal_trade_date": "20260511",
                "planned_trade_date": "20260512",
                "buy_orders": [
                    {
                        "action": "BUY",
                        "symbol": "301396.SZ",
                        "name": "宏景科技",
                        "limit_price": 100.0,
                        "quantity": None,
                        "rank": 1,
                        "score": 0.98,
                    }
                ],
                "sell_orders": [],
                "hold_orders": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repository = FakeRepository(
        {
            "20260512": pd.DataFrame(
                [
                    {
                        "ts_code": "301396.SZ",
                        "trade_date": "20260512",
                        "open": 99.0,
                        "high": 102.0,
                        "low": 98.0,
                        "close": 101.0,
                    }
                ]
            )
        }
    )
    broker = Tianzhu9PaperBroker(
        config=_config(),
        repository=repository,
        base_dir=tmp_path,
        hold_days=2,
    )

    result = broker.settle_pending_plan(as_of_trade_date="20260512")

    positions = pd.read_csv(result.positions_path)
    assert result.executed_trades == 1
    assert positions.loc[0, "symbol"] == "301396.SZ"
    assert result.total_return == pytest.approx(0.0098515)
    assert (tmp_path / "data" / "positions" / "tianzhu9_settled_plans" / "20260511-20260512.json").exists()
