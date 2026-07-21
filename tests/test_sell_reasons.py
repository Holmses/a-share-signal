from types import SimpleNamespace

from ashare_signal.strategy.sell_reasons import normalize_sell_reason
from ashare_signal.strategy.sell_reasons import sell_reason_counts, sell_reason_definition
from ashare_signal.strategy.sell_reasons import summarize_sell_reasons


def test_normalize_sell_reason_keeps_standard_codes() -> None:
    assert normalize_sell_reason("rotation_rank_drop") == "rotation_rank_drop"
    assert normalize_sell_reason("scoreEdgeRotation") == "score_edge_rotation"
    assert normalize_sell_reason("") == "unknown"


def test_normalize_sell_reason_maps_legacy_text() -> None:
    assert normalize_sell_reason("硬止损触发：较入场价 -8.2%") == "hard_stop_loss"
    assert normalize_sell_reason("市场 risk-off，退出") == "market_risk_exit"
    assert normalize_sell_reason("相对走弱。") == "relative_weak_exit"


def test_normalize_sell_reason_maps_slow_profit_lock_aliases() -> None:
    assert normalize_sell_reason("slow_profit_lock_trailing") == "trailing_take_profit"
    assert normalize_sell_reason("slow_profit_lock_ma20_weak") == "ma20_break"
    assert normalize_sell_reason("slow_profit_lock_ma60") == "ma20_break"
    assert normalize_sell_reason("slow_profit_lock_style_weak") == "industry_weak_exit"
    assert normalize_sell_reason("slow_profit_lock_hard60") == "max_holding_days_exit"


def test_summarize_sell_reasons_groups_pnl_and_win_rate() -> None:
    records = [
        SimpleNamespace(reason="rotation_rank_drop", pnl=10.0),
        SimpleNamespace(reason="rotation_rank_drop", pnl=-3.0),
        SimpleNamespace(reason="market_risk_exit", pnl=-2.0),
        {"exit_reason": "score_edge_rotation", "pnl": "5.0"},
    ]
    rows = summarize_sell_reasons(records)

    rotation = next(row for row in rows if row["reason"] == "rotation_rank_drop")
    assert rotation["category"] == "rotation"
    assert rotation["count"] == 2
    assert rotation["win_count"] == 1
    assert rotation["win_rate"] == 0.5
    assert rotation["total_pnl"] == 7.0
    assert rotation["avg_pnl"] == 3.5
    assert sell_reason_counts(records) == {
        "rotation_rank_drop": 2,
        "market_risk_exit": 1,
        "score_edge_rotation": 1,
    }
    assert sell_reason_definition("market_risk_exit").category == "risk"
