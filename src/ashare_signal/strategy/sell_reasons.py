from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class SellReasonDefinition:
    code: str
    category: str
    label: str
    description: str


STANDARD_SELL_REASONS: dict[str, SellReasonDefinition] = {
    "rotation_rank_drop": SellReasonDefinition(
        "rotation_rank_drop",
        "rotation",
        "Rank drop",
        "Position fell out of the configured rank buffer.",
    ),
    "score_edge_rotation": SellReasonDefinition(
        "score_edge_rotation",
        "rotation",
        "Score edge rotation",
        "A stronger candidate exceeded the held position by the configured score edge.",
    ),
    "missing_rank": SellReasonDefinition(
        "missing_rank",
        "rotation",
        "Missing rank",
        "Position disappeared from the current ranking universe.",
    ),
    "strategy_invalid": SellReasonDefinition(
        "strategy_invalid",
        "strategy",
        "Strategy invalid",
        "The original strategy condition no longer holds.",
    ),
    "hard_stop_loss": SellReasonDefinition(
        "hard_stop_loss",
        "risk",
        "Hard stop loss",
        "Fixed loss threshold was hit.",
    ),
    "trailing_take_profit": SellReasonDefinition(
        "trailing_take_profit",
        "profit",
        "Trailing take profit",
        "Profit lock or trailing take-profit rule was hit.",
    ),
    "market_risk_exit": SellReasonDefinition(
        "market_risk_exit",
        "risk",
        "Market risk exit",
        "Market gate moved to risk-off and the strategy requires an exit.",
    ),
    "risk_off_failed_hard_exit": SellReasonDefinition(
        "risk_off_failed_hard_exit",
        "risk",
        "Risk-off failed hard exit",
        "Position failed to reach the winner peak threshold and hit the shorter risk-off hard-exit period.",
    ),
    "industry_weak_exit": SellReasonDefinition(
        "industry_weak_exit",
        "risk",
        "Industry weak exit",
        "Industry or board-style strength deteriorated.",
    ),
    "relative_weak_exit": SellReasonDefinition(
        "relative_weak_exit",
        "risk",
        "Relative weak exit",
        "Position became weak relative to the market or its peer group.",
    ),
    "liquidity_deterioration": SellReasonDefinition(
        "liquidity_deterioration",
        "risk",
        "Liquidity deterioration",
        "Liquidity fell below the strategy's tradability preference.",
    ),
    "limit_break_exit": SellReasonDefinition(
        "limit_break_exit",
        "execution",
        "Limit break exit",
        "A limit-up or limit-break condition triggered an exit.",
    ),
    "max_holding_days_exit": SellReasonDefinition(
        "max_holding_days_exit",
        "time",
        "Max holding days",
        "Maximum configured holding period was reached.",
    ),
    "ma20_break": SellReasonDefinition(
        "ma20_break",
        "trend",
        "MA20 break",
        "Price broke below the 20-day moving average.",
    ),
    "failure_exit": SellReasonDefinition(
        "failure_exit",
        "strategy",
        "Failure exit",
        "Position failed to develop after the configured observation period.",
    ),
    "high_drawdown_exit": SellReasonDefinition(
        "high_drawdown_exit",
        "trend",
        "High drawdown exit",
        "Price retraced from the highest observed price by the configured amount.",
    ),
    "chandelier_exit": SellReasonDefinition(
        "chandelier_exit",
        "trend",
        "Chandelier exit",
        "Price fell below the highest price minus the configured ATR multiple.",
    ),
    "trend_decay_exit": SellReasonDefinition(
        "trend_decay_exit",
        "trend",
        "Trend decay exit",
        "Short and medium trend structure weakened below MA20.",
    ),
    "volume_stall_exit": SellReasonDefinition(
        "volume_stall_exit",
        "strategy",
        "Volume stall",
        "Volume expansion did not support further upside.",
    ),
    "upper_shadow_exit": SellReasonDefinition(
        "upper_shadow_exit",
        "strategy",
        "Upper shadow",
        "Upper-shadow reversal risk was detected.",
    ),
    "fast_failure_exit": SellReasonDefinition(
        "fast_failure_exit",
        "strategy",
        "Fast failure",
        "Short-horizon rebound setup failed quickly.",
    ),
    "profit_lock": SellReasonDefinition(
        "profit_lock",
        "profit",
        "Profit lock",
        "Rebound or swing profit lock was triggered.",
    ),
    "signal_low_break": SellReasonDefinition(
        "signal_low_break",
        "trend",
        "Signal low break",
        "Price broke the signal low or stabilization reference.",
    ),
    "unknown": SellReasonDefinition(
        "unknown",
        "unknown",
        "Unknown",
        "No normalized sell reason was available.",
    ),
}

SELL_REASON_ALIASES: dict[str, str] = {
    "slow_profit_lock_trailing": "trailing_take_profit",
    "slow_profit_lock_ma20_weak": "ma20_break",
    "slow_profit_lock_ma60": "ma20_break",
    "slow_profit_lock_style_weak": "industry_weak_exit",
    "slow_profit_lock_hard60": "max_holding_days_exit",
}

_LEGACY_REASON_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile("硬止损|止损|stop.?loss", re.IGNORECASE), "hard_stop_loss"),
    (re.compile("止盈|盈利|profit|trailing", re.IGNORECASE), "trailing_take_profit"),
    (re.compile("市场|risk.?off|market", re.IGNORECASE), "market_risk_exit"),
    (re.compile("行业|板块|industry|style", re.IGNORECASE), "industry_weak_exit"),
    (re.compile("相对走弱|relative", re.IGNORECASE), "relative_weak_exit"),
    (re.compile("均线|ma20", re.IGNORECASE), "ma20_break"),
    (re.compile("最长持有|最大持有|max.?holding|hard\\d+d", re.IGNORECASE), "max_holding_days_exit"),
    (re.compile("快速|failure|失败", re.IGNORECASE), "failure_exit"),
)


def normalize_sell_reason(reason: str | None) -> str:
    if reason is None:
        return "unknown"
    normalized = str(reason).strip()
    if not normalized:
        return "unknown"
    code = _snake_case(normalized)
    if code in SELL_REASON_ALIASES:
        return SELL_REASON_ALIASES[code]
    if code in STANDARD_SELL_REASONS:
        return code
    for pattern, mapped_code in _LEGACY_REASON_PATTERNS:
        if pattern.search(normalized):
            return mapped_code
    return "unknown"


def sell_reason_definition(reason: str | None) -> SellReasonDefinition:
    code = normalize_sell_reason(reason)
    return STANDARD_SELL_REASONS.get(code, STANDARD_SELL_REASONS["unknown"])


def sell_reason_counts(records: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        code = normalize_sell_reason(_record_value(record, "reason", "exit_reason"))
        counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def summarize_sell_reasons(records: Iterable[object]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        code = normalize_sell_reason(_record_value(record, "reason", "exit_reason"))
        pnl = _float_or_none(_record_value(record, "pnl"))
        bucket = buckets.setdefault(
            code,
            {
                "reason": code,
                "category": sell_reason_definition(code).category,
                "label": sell_reason_definition(code).label,
                "count": 0,
                "win_count": 0,
                "loss_count": 0,
                "total_pnl": 0.0,
                "_pnl_count": 0,
            },
        )
        bucket["count"] += 1
        if pnl is None:
            continue
        bucket["_pnl_count"] += 1
        bucket["total_pnl"] += pnl
        if pnl > 0:
            bucket["win_count"] += 1
        else:
            bucket["loss_count"] += 1

    rows = []
    for bucket in buckets.values():
        pnl_count = bucket.pop("_pnl_count")
        count = int(bucket["count"])
        win_count = int(bucket["win_count"])
        bucket["win_rate"] = float(win_count / count) if count else 0.0
        bucket["avg_pnl"] = float(bucket["total_pnl"] / pnl_count) if pnl_count else None
        rows.append(bucket)
    return sorted(rows, key=lambda row: (-int(row["count"]), str(row["reason"])))


def sell_reason_map() -> list[dict[str, str]]:
    return [asdict(reason) for reason in STANDARD_SELL_REASONS.values()]


def _record_value(record: object, *names: str) -> object | None:
    for name in names:
        if isinstance(record, dict) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return None


def _float_or_none(value: object | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return value.strip("_").lower()
