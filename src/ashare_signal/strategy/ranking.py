from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ashare_signal.config import AppConfig


SUPPORTED_RANKING_VARIANTS = ("quality_momentum_rank",)


@dataclass(frozen=True, slots=True)
class RankingFactorSpec:
    column: str
    label: str
    source: str
    weight: float
    category: str
    note: str


QUALITY_MOMENTUM_FACTORS = (
    RankingFactorSpec(
        column="momentum_rank",
        label="20d momentum percentile",
        source="momentum_20d / momentum_20d_rank_pct",
        weight=0.25,
        category="ranking_factor",
        note="Turns the existing positive-momentum preference into cross-sectional ranking.",
    ),
    RankingFactorSpec(
        column="trend_rank",
        label="trend structure",
        source="ma_20_to_ma_60 + ma_60_slope_20d + close_to_ma_60",
        weight=0.18,
        category="ranking_factor",
        note="Rewards MA alignment and longer trend slope instead of hard-gating every trend field.",
    ),
    RankingFactorSpec(
        column="pullback_rank",
        label="controlled pullback",
        source="pullback_from_20d_high, close_to_ma_20",
        weight=0.12,
        category="ranking_factor",
        note="Prefers pullbacks inside the configured trend-entry band and avoids overextended closes.",
    ),
    RankingFactorSpec(
        column="liquidity_rank",
        label="20d liquidity",
        source="avg_amount_20d_yuan",
        weight=0.15,
        category="hard_filter_plus_ranking_factor",
        note="Low liquidity remains a hard filter; remaining symbols are ranked by tradability.",
    ),
    RankingFactorSpec(
        column="volatility_rank",
        label="20d volatility stability",
        source="volatility_20d",
        weight=0.10,
        category="ranking_factor",
        note="Lower realized volatility receives a higher rank.",
    ),
    RankingFactorSpec(
        column="moneyflow_rank",
        label="large-order moneyflow",
        source="large_net_mf_to_amount / net_mf_to_amount",
        weight=0.06,
        category="optional_ranking_factor",
        note="Uses neutral 0.5 when optional moneyflow cache is unavailable.",
    ),
    RankingFactorSpec(
        column="industry_rank",
        label="industry strength",
        source="industry_momentum_20d_median + industry_breadth_20d + industry_return_3d_median",
        weight=0.08,
        category="optional_ranking_factor",
        note="Rewards industry tailwind while keeping missing industry data neutral.",
    ),
    RankingFactorSpec(
        column="rebound_rank",
        label="bottoming rebound quality",
        source="drawdown_20d, drawdown_60d, return_3d, low_to_prev_low",
        weight=0.06,
        category="ranking_factor",
        note="Carries the current rebound-bottoming logic as a soft rank contribution.",
    ),
)


HARD_FILTER_ROWS = (
    (
        "exchange_not_supported",
        "passes_exchange_filter",
        "A-share exchange, excluding unsupported markets such as BSE.",
    ),
    ("st_stock", "passes_st_filter", "ST symbols stay as hard exclusions."),
    ("suspended", "passes_suspension_filter", "Suspended or missing daily bars stay as hard exclusions."),
    ("listed_days_too_short", "passes_listing_age_filter", "New listings stay outside the first ranking pass."),
    ("price_below_threshold", "passes_price_filter", "Very low-priced names stay outside the tradable pool."),
    ("liquidity_below_threshold", "passes_liquidity_filter", "Minimum liquidity remains a hard tradability floor."),
)


def build_ranking_snapshot(
    universe: pd.DataFrame,
    config: AppConfig,
    *,
    variant: str = "quality_momentum_rank",
) -> pd.DataFrame:
    """Build a research-only cross-sectional ranking snapshot.

    The function keeps all universe rows so the output can explain hard-filtered
    names, but it only assigns ranking positions inside the tradable pool.
    """

    if variant not in SUPPORTED_RANKING_VARIANTS:
        raise ValueError(f"Unsupported ranking variant: {variant}")
    if universe.empty:
        return _empty_ranking_frame()

    frame = universe.copy()
    frame = _coerce_columns(frame)
    frame["trade_date"] = _normalize_trade_date_column(frame)
    frame["universe_group"] = _universe_group(frame)
    frame["is_tradeable"] = _tradeable_mask(frame)
    frame["filter_reason"] = _filter_reason(frame)

    eligible_mask = frame["is_tradeable"]
    eligible = frame.loc[eligible_mask].copy()
    for spec in QUALITY_MOMENTUM_FACTORS:
        frame[spec.column] = 0.0

    if not eligible.empty:
        factor_scores = _quality_momentum_factor_scores(eligible, config)
        for column in factor_scores.columns:
            frame.loc[eligible.index, column] = factor_scores[column]
        weighted = pd.Series(0.0, index=eligible.index)
        for spec in QUALITY_MOMENTUM_FACTORS:
            weighted = weighted + factor_scores[spec.column].fillna(0.0) * spec.weight
        frame.loc[eligible.index, "rank_score"] = weighted
    else:
        frame["rank_score"] = 0.0

    frame["rank_score"] = frame["rank_score"].fillna(0.0)
    frame["rank_position"] = pd.NA
    if eligible_mask.any():
        ranks = frame.loc[eligible_mask, "rank_score"].rank(method="first", ascending=False).astype(int)
        frame.loc[eligible_mask, "rank_position"] = ranks

    frame["signal_type"] = _signal_types(frame)
    frame["score_explain"] = [
        _score_explain(row, QUALITY_MOMENTUM_FACTORS) if bool(row["is_tradeable"]) else str(row["filter_reason"])
        for _, row in frame.iterrows()
    ]

    output_columns = [
        "trade_date",
        "ts_code",
        "name",
        "universe_group",
        "industry",
        "is_tradeable",
        "filter_reason",
        "momentum_rank",
        "trend_rank",
        "pullback_rank",
        "liquidity_rank",
        "volatility_rank",
        "moneyflow_rank",
        "industry_rank",
        "rebound_rank",
        "rank_score",
        "rank_position",
        "signal_type",
        "score_explain",
    ]
    for column in output_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return (
        frame[output_columns]
        .sort_values(["is_tradeable", "rank_position", "ts_code"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def render_ranking_factor_map() -> str:
    lines = [
        "# Ranking Factor Map",
        "",
        "This research map is generated from the current V1 selector and the first BigQuant-style ranking path.",
        "",
        "## Hard Trade Filters",
        "",
        "| filter_reason | source_column | treatment |",
        "| --- | --- | --- |",
    ]
    for reason, source, note in HARD_FILTER_ROWS:
        lines.append(f"| {reason} | {source} | {note} |")

    lines.extend(
        [
            "",
            "## Ranking Factors",
            "",
            "| output_column | source | weight | treatment | note |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for spec in QUALITY_MOMENTUM_FACTORS:
        lines.append(f"| {spec.column} | {spec.source} | {spec.weight:.2f} | {spec.category} | {spec.note} |")

    lines.extend(
        [
            "",
            "## Kept As Risk / Exit Controls",
            "",
            "| rule | treatment |",
            "| --- | --- |",
            "| stop_loss_pct | Keep in exit rules; do not convert to buy ranking. |",
            "| take_profit_trigger_pct | Keep in exit rules; rank studies only measure forward path quality. |",
            "| trailing_stop_drawdown_pct | Keep in exit rules until recipe backtests can attribute sell reasons. |",
            "| rotation_edge | Later TopK/DropN experiments should compare rank_score edge against holding health. |",
        ]
    )
    return "\n".join(lines) + "\n"


def _empty_ranking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_date",
            "ts_code",
            "name",
            "universe_group",
            "industry",
            "is_tradeable",
            "filter_reason",
            "momentum_rank",
            "trend_rank",
            "pullback_rank",
            "liquidity_rank",
            "volatility_rank",
            "moneyflow_rank",
            "industry_rank",
            "rebound_rank",
            "rank_score",
            "rank_position",
            "signal_type",
            "score_explain",
        ]
    )


def _coerce_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _copy_alias(frame, "momentum_20d", ("return_20d",))
    _copy_alias(frame, "pullback_from_20d_high", ("drawdown_from_20d_high",))
    _copy_alias(frame, "drawdown_20d", ("drawdown_from_20d_high",))
    numeric_columns = [
        "momentum_20d",
        "momentum_20d_rank_pct",
        "ma_20_to_ma_60",
        "ma_60_slope_20d",
        "close_to_ma_60",
        "pullback_from_20d_high",
        "close_to_ma_20",
        "avg_amount_20d_yuan",
        "volatility_20d",
        "large_net_mf_to_amount",
        "net_mf_to_amount",
        "industry_momentum_20d_median",
        "industry_breadth_20d",
        "industry_return_3d_median",
        "industry_rebound_breadth",
        "drawdown_20d",
        "drawdown_60d",
        "return_3d",
        "low_to_prev_low",
        "total_mv_yuan",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    bool_columns = [
        "is_candidate",
        "passes_exchange_filter",
        "passes_st_filter",
        "passes_suspension_filter",
        "passes_listing_age_filter",
        "passes_price_filter",
        "passes_liquidity_filter",
    ]
    for column in bool_columns:
        if column not in frame.columns:
            continue
        if frame[column].dtype == bool:
            continue
        normalized = frame[column].astype(str).str.strip().str.lower()
        mapped = normalized.map({"true": True, "false": False, "1": True, "0": False})
        frame[column] = mapped.fillna(frame[column]).astype(bool)
    return frame


def _copy_alias(frame: pd.DataFrame, target: str, aliases: tuple[str, ...]) -> None:
    if target in frame.columns:
        return
    for alias in aliases:
        if alias in frame.columns:
            frame[target] = frame[alias]
            return


def _normalize_trade_date_column(frame: pd.DataFrame) -> pd.Series:
    if "trade_date" not in frame.columns:
        return pd.Series(pd.NA, index=frame.index)
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    if dates.notna().any():
        return dates.dt.strftime("%Y%m%d")
    return frame["trade_date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def _tradeable_mask(frame: pd.DataFrame) -> pd.Series:
    if "is_candidate" in frame.columns:
        return frame["is_candidate"].fillna(False).astype(bool)
    required = [
        "passes_exchange_filter",
        "passes_st_filter",
        "passes_suspension_filter",
        "passes_listing_age_filter",
        "passes_price_filter",
        "passes_liquidity_filter",
    ]
    present = [column for column in required if column in frame.columns]
    if not present:
        return pd.Series(True, index=frame.index)
    return frame[present].fillna(False).all(axis=1)


def _filter_reason(frame: pd.DataFrame) -> pd.Series:
    if "exclude_reason" in frame.columns:
        return frame["exclude_reason"].fillna("eligible").astype(str)
    reasons = pd.Series("eligible", index=frame.index)
    for reason, source, _ in HARD_FILTER_ROWS:
        if source in frame.columns:
            reasons = reasons.mask((reasons == "eligible") & (~frame[source].fillna(False).astype(bool)), reason)
    return reasons


def _universe_group(frame: pd.DataFrame) -> pd.Series:
    if "group" in frame.columns:
        groups = frame["group"].fillna("").astype(str).str.strip()
    else:
        groups = pd.Series("", index=frame.index)
    missing = groups.eq("") | groups.eq("unknown")
    if missing.any():
        groups.loc[missing] = frame.loc[missing].apply(_classify_board, axis=1)
    return groups


def _classify_board(row: pd.Series) -> str:
    market = str(row.get("market") or "")
    exchange = str(row.get("exchange") or "")
    ts_code = str(row.get("ts_code") or "")
    if market == "创业板" or ts_code.startswith(("300", "301")):
        return "chinext"
    if market == "科创板" or ts_code.startswith("688"):
        return "star"
    if market == "北交所" or exchange == "BSE" or ts_code.endswith(".BJ"):
        return "bse"
    return "main"


def _quality_momentum_factor_scores(eligible: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    scores = pd.DataFrame(index=eligible.index)
    scores["momentum_rank"] = _rank_or_existing(eligible, "momentum_20d_rank_pct", "momentum_20d")
    scores["trend_rank"] = _rank_series(
        eligible["ma_20_to_ma_60"].fillna(0.0)
        + eligible["ma_60_slope_20d"].fillna(0.0)
        + eligible["close_to_ma_60"].fillna(0.0) * 0.50
    )
    pullback_score = _center_score(
        eligible["pullback_from_20d_high"],
        lower=config.selection.buy_min_pullback_from_20d_high,
        upper=config.selection.buy_max_pullback_from_20d_high,
    )
    ma20_control = (1.0 - _clip_score(eligible["close_to_ma_20"], 0.05, 0.30)).clip(lower=0.0, upper=1.0)
    scores["pullback_rank"] = (pullback_score * 0.70 + ma20_control.fillna(0.0) * 0.30).clip(0.0, 1.0)
    scores["liquidity_rank"] = _rank_series(eligible["avg_amount_20d_yuan"])
    scores["volatility_rank"] = _rank_series(eligible["volatility_20d"], ascending=False)
    moneyflow = eligible["large_net_mf_to_amount"].combine_first(eligible["net_mf_to_amount"])
    scores["moneyflow_rank"] = _rank_series(moneyflow, default=0.50)
    industry_raw = (
        eligible["industry_momentum_20d_median"].fillna(0.0) * 0.45
        + eligible["industry_breadth_20d"].fillna(0.50) * 0.35
        + eligible["industry_return_3d_median"].fillna(0.0) * 0.20
    )
    scores["industry_rank"] = _rank_series(industry_raw, default=0.50)
    scores["rebound_rank"] = _rebound_rank_score(eligible, config)
    return scores.fillna(0.0).clip(lower=0.0, upper=1.0)


def _rank_or_existing(frame: pd.DataFrame, rank_column: str, value_column: str) -> pd.Series:
    existing = pd.to_numeric(frame[rank_column], errors="coerce")
    if existing.notna().any():
        return existing.fillna(0.0).clip(lower=0.0, upper=1.0)
    return _rank_series(frame[value_column])


def _rank_series(series: pd.Series, *, ascending: bool = True, default: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        return pd.Series(default, index=series.index, dtype=float)
    return numeric.rank(pct=True, ascending=ascending).fillna(default).clip(lower=0.0, upper=1.0)


def _clip_score(series: pd.Series, lower: float, upper: float) -> pd.Series:
    if upper == lower:
        return pd.Series(0.0, index=series.index)
    numeric = pd.to_numeric(series, errors="coerce")
    return ((numeric - lower) / (upper - lower)).clip(lower=0.0, upper=1.0).fillna(0.0)


def _center_score(series: pd.Series, *, lower: float, upper: float) -> pd.Series:
    width = abs(upper - lower) / 2.0
    if width == 0:
        return pd.Series(0.0, index=series.index)
    center = (lower + upper) / 2.0
    numeric = pd.to_numeric(series, errors="coerce")
    return (1.0 - ((numeric - center).abs() / width)).clip(lower=0.0, upper=1.0).fillna(0.0)


def _rebound_rank_score(eligible: pd.DataFrame, config: AppConfig) -> pd.Series:
    drawdown_floor = abs(config.selection.rebound_min_drawdown_20d)
    drawdown_ceiling = abs(config.selection.rebound_max_drawdown_60d)
    depth = _clip_score(-eligible["drawdown_20d"], drawdown_floor, drawdown_ceiling)
    depth_control = (1.0 - _clip_score(-eligible["drawdown_60d"], drawdown_ceiling * 0.60, drawdown_ceiling)).clip(
        lower=0.0,
        upper=1.0,
    )
    stabilization = (
        _clip_score(eligible["return_3d"], config.selection.rebound_min_return_3d, 0.05) * 0.40
        + _clip_score(eligible["low_to_prev_low"], -0.02, 0.04) * 0.30
        + _clip_score(eligible["industry_rebound_breadth"], 0.25, 0.65) * 0.30
    )
    raw = depth * 0.35 + depth_control * 0.25 + stabilization * 0.40
    return raw.fillna(0.0).clip(lower=0.0, upper=1.0)


def _signal_types(frame: pd.DataFrame) -> pd.Series:
    signal_type = pd.Series("filtered_out", index=frame.index)
    eligible = frame["is_tradeable"].fillna(False).astype(bool)
    signal_type.loc[eligible] = "quality_momentum_rank"
    rebound_mask = eligible & (frame["rebound_rank"] >= 0.70) & (frame["momentum_rank"] < 0.60)
    trend_mask = eligible & (frame["trend_rank"] >= 0.60) & (frame["pullback_rank"] >= 0.50)
    defensive_mask = eligible & (frame["volatility_rank"] >= 0.70) & (frame["liquidity_rank"] >= 0.70)
    signal_type.loc[defensive_mask] = "defensive_largecap_rank"
    signal_type.loc[rebound_mask] = "rebound_bottoming_rank"
    signal_type.loc[trend_mask] = "trend_pullback_rank"
    return signal_type


def _score_explain(row: pd.Series, specs: Iterable[RankingFactorSpec]) -> str:
    parts = []
    for spec in specs:
        value = row.get(spec.column)
        if pd.isna(value):
            value = 0.0
        parts.append(f"{spec.column}={float(value):.3f}*{spec.weight:.2f}")
    return "; ".join(parts)
