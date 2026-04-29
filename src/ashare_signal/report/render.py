from __future__ import annotations

from pathlib import Path

from ashare_signal.domain.models import SignalBoard, TradeSignal


def _render_signal(title: str, signal: TradeSignal | None) -> list[str]:
    lines = [f"## {title}", ""]
    if signal is None:
        lines.append("无候选信号。")
        lines.append("")
        return lines

    lines.extend(
        [
            f"- 动作：{signal.action}",
            f"- 股票：{signal.symbol} {signal.name}",
            f"- 昨收价：{signal.last_close:.2f}" if signal.last_close is not None else "- 昨收价：未知",
            f"- 建议委托价：{signal.suggested_price:.2f}",
            f"- 分数：{signal.score:.4f}",
            f"- 理由：{signal.reason}",
            "",
        ]
    )
    return lines


def render_markdown(board: SignalBoard) -> str:
    lines = [
        "# 每日信号板",
        "",
        f"- 信号日期：{board.signal_date.isoformat()}",
        f"- 计算交易日：{board.trade_date.isoformat()}",
        f"- 生效日期：{board.effective_date.isoformat()}",
        f"- 当前持仓数：{board.holdings_count}",
        "",
    ]
    lines.extend(_render_signal("卖出信号", board.sell_signal))
    lines.extend(_render_signal("买入信号", board.buy_signal))
    lines.append("## 风险备注")
    lines.append("")
    for note in board.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_markdown(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
