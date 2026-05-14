from pathlib import Path

from ashare_signal.portfolio.tianzhu9_simulator import Tianzhu9PositionSnapshot, Tianzhu9SimulationResult
from ashare_signal.scheduler.tianzhu9_daily import render_tianzhu9_simulation_markdown, simulation_to_feishu_text
from ashare_signal.strategy.tianzhu9_orders import Tianzhu9Order, Tianzhu9OrderPlan, plan_to_feishu_text
from ashare_signal.strategy.tianzhu9_orders import render_tianzhu9_order_plan


def test_render_tianzhu9_order_plan_uses_table_for_holdings(tmp_path: Path) -> None:
    plan = Tianzhu9OrderPlan(
        signal_trade_date="20260513",
        planned_trade_date="20260514",
        buy_orders=[],
        sell_orders=[],
        hold_orders=[
            Tianzhu9Order(
                action="HOLD",
                symbol="301396.SZ",
                name="宏景科技",
                limit_price=None,
                quantity=1500,
                rank=1,
                score=0.9832,
                reason="重复入选，未达到最长持有 4 天。",
                entry_price=319.00,
                last_price=338.78,
                market_value=508170.00,
                unrealized_pnl=28170.00,
                unrealized_return=0.0587,
                holding_days=2,
            )
        ],
        notes=[],
        markdown_path=tmp_path / "plan.md",
        json_path=tmp_path / "plan.json",
    )

    markdown = render_tianzhu9_order_plan(plan)
    feishu_text = plan_to_feishu_text(plan)

    assert "| 代码 | 名称 | 数量 | 买入价 | 现价 | 市值 | 浮盈亏 | 收益率 | 持有天数 | rank | score | 原因 |" in markdown
    assert "| 301396.SZ | 宏景科技 | 1500 | 319.00 | 338.78 | 508,170.00 | +28,170.00 | +5.87% | 2天 | 1 | 0.9832 | 重复入选，未达到最长持有 4 天。 |" in markdown
    assert "买入:319.00 现价:338.78 浮盈亏:+28,170.00 (+5.87%) 持有:2天 rank:1 score:0.9832" in feishu_text


def test_render_tianzhu9_simulation_markdown_uses_current_positions_table(tmp_path: Path) -> None:
    result = Tianzhu9SimulationResult(
        positions_path=tmp_path / "positions.csv",
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        pending_plan_path=tmp_path / "pending.json",
        initial_cash=1_000_000.0,
        cash=519_856.0,
        equity=1_028_026.0,
        previous_equity=983_296.0,
        positions_market_value=508_170.0,
        daily_pnl=44_730.0,
        daily_return=0.045489862665972325,
        total_return=0.028026000000000106,
        positions_count=1,
        executed_trades=0,
        pending_plan_date="20260514",
        last_trade_date="20260513",
        updated_at="2026-05-14T10:46:50+08:00",
        positions=[
            Tianzhu9PositionSnapshot(
                symbol="301396.SZ",
                name="宏景科技",
                entry_date="2026-05-12",
                entry_price=319.00,
                quantity=1500,
                last_price=338.78,
                market_value=508170.00,
                cost_basis=478500.00,
                unrealized_pnl=28170.00,
                unrealized_return=0.0587,
                holding_days=2,
            )
        ],
    )

    markdown = render_tianzhu9_simulation_markdown(result)
    feishu_text = simulation_to_feishu_text(result)

    assert "## 当前持仓" in markdown
    assert "| 代码 | 名称 | 买入日 | 数量 | 买入价 | 现价 | 市值 | 浮盈亏 | 收益率 | 持有天数 |" in markdown
    assert "| 301396.SZ | 宏景科技 | 2026-05-12 | 1500 | 319.00 | 338.78 | 508,170.00 | +28,170.00 | +5.87% | 2天 |" in markdown
    assert "买入:319.00 现价:338.78 市值:508,170.00 浮盈亏:+28,170.00 (+5.87%) 持有:2天" in feishu_text
