from datetime import date

from ashare_signal.domain.models import SignalBoard, TradeSignal
from ashare_signal.report.render import render_markdown


def test_render_markdown_contains_main_sections() -> None:
    board = SignalBoard(
        signal_date=date(2026, 4, 3),
        trade_date=date(2026, 4, 3),
        effective_date=date(2026, 4, 6),
        holdings_count=5,
        buy_signal=TradeSignal(
            action="BUY",
            symbol="600036.SH",
            name="招商银行",
            suggested_price=42.45,
            score=0.91,
            reason="趋势向上。",
            last_close=42.32,
        ),
        sell_signal=TradeSignal(
            action="SELL",
            symbol="601318.SH",
            name="中国平安",
            suggested_price=45.82,
            score=0.12,
            reason="相对走弱。",
            last_close=45.96,
        ),
        notes=["测试备注"],
    )

    markdown = render_markdown(board)

    assert "# 每日信号板" in markdown
    assert "## 买入信号" in markdown
    assert "## 卖出信号" in markdown
    assert "600036.SH 招商银行" in markdown
    assert "计算交易日：2026-04-03" in markdown
