from types import SimpleNamespace

import pandas as pd

from ashare_signal.strategy.exit_rules import tiered_trailing_take_profit
from ashare_signal.strategy.tianzhu9_orders import _build_position_orders, _format_group_list


def _config():
    return SimpleNamespace(pricing=SimpleNamespace(sell_markdown=0.003))


def test_tiered_trailing_take_profit_ignores_plain_losses() -> None:
    signal = tiered_trailing_take_profit(
        entry_price=100.0,
        current_close=90.0,
        highest_price=100.0,
    )

    assert signal.should_exit is False
    assert signal.peak_profit_pct == 0.0


def test_tiered_trailing_take_profit_triggers_after_profit_retracement() -> None:
    signal = tiered_trailing_take_profit(
        entry_price=100.0,
        current_close=105.0,
        highest_price=112.0,
    )

    assert signal.should_exit is True
    assert signal.trigger_profit_pct == 0.12
    assert signal.trigger_drawdown_pct == 0.06


def test_order_generation_uses_tiered_trailing_without_hard_stop() -> None:
    factor_frame = pd.DataFrame(
        [
            {
                "trade_date": "20260513",
                "ts_code": "301396.SZ",
                "close": 90.0,
                "ma_5": 95.0,
                "ma_10": 98.0,
                "group": "chinext",
            },
            {
                "trade_date": "20260513",
                "ts_code": "688001.SH",
                "close": 105.0,
                "ma_5": 108.0,
                "ma_10": 109.0,
                "group": "star",
            },
        ]
    )
    positions = [
        {
            "symbol": "301396.SZ",
            "name": "宏景科技",
            "entry_date": "2026-05-01",
            "entry_price": 100.0,
            "quantity": 100,
            "highest_close": 100.0,
            "highest_high": 100.0,
        },
        {
            "symbol": "688001.SH",
            "name": "测试科技",
            "entry_date": "2026-05-01",
            "entry_price": 100.0,
            "quantity": 100,
            "highest_close": 112.0,
            "highest_high": 112.0,
        },
    ]

    sell_orders, hold_orders = _build_position_orders(
        config=_config(),
        factor_frame=factor_frame,
        signal_trade_date="20260513",
        selected_symbols=set(),
        selected_by_symbol={},
        positions=positions,
        hold_days=5,
        max_hold_days=5,
        risk_off=True,
        eligible_groups=set(),
    )

    assert [order.symbol for order in sell_orders] == ["688001.SH"]
    assert sell_orders[0].reason.startswith("分层追踪止盈")
    assert [order.symbol for order in hold_orders] == ["301396.SZ"]
    assert "未触发分层追踪止盈" in hold_orders[0].reason


def test_order_generation_can_force_exit_after_trade_days() -> None:
    factor_frame = pd.DataFrame(
        [
            {
                "trade_date": "20260513",
                "ts_code": "301396.SZ",
                "close": 90.0,
                "ma_5": 95.0,
                "ma_10": 98.0,
                "group": "chinext",
            }
        ]
    )
    positions = [
        {
            "symbol": "301396.SZ",
            "name": "宏景科技",
            "entry_date": "2026-05-04",
            "entry_price": 100.0,
            "quantity": 100,
            "highest_close": 100.0,
            "highest_high": 100.0,
        }
    ]

    sell_orders, hold_orders = _build_position_orders(
        config=_config(),
        factor_frame=factor_frame,
        signal_trade_date="20260513",
        selected_symbols=set(),
        selected_by_symbol={},
        positions=positions,
        hold_days=5,
        max_hold_days=10,
        cached_dates=[
            "20260504",
            "20260505",
            "20260506",
            "20260507",
            "20260508",
            "20260511",
            "20260512",
            "20260513",
        ],
        hard_exit_days=8,
    )

    assert hold_orders == []
    assert [order.symbol for order in sell_orders] == ["301396.SZ"]
    assert sell_orders[0].holding_days == 8
    assert sell_orders[0].reason == "硬卖出：持仓满 8 个交易日。"


def test_format_group_list_compacts_many_industries() -> None:
    text = _format_group_list({f"行业{index:02d}" for index in range(20)}, max_items=3)

    assert text.endswith("等 20 个分组")
