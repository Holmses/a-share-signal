from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class ThemeAlertCandidate:
    symbol: str
    name: str
    industry: str
    rank_position: int
    rank_score: float
    signal_type: str
    momentum_rank: float
    trend_rank: float
    return_5d: float | None = None
    pullback_from_20d_high: float | None = None
    close_to_ma_20: float | None = None
    amount_ratio_5d: float | None = None
    upper_shadow_pct: float | None = None


@dataclass(frozen=True, slots=True)
class StrongThemeAlert:
    industry: str
    member_count: int
    top_ranked_count: int
    top_ranked_share: float
    avg_rank_score: float
    median_momentum_rank: float
    median_trend_rank: float
    median_industry_rank: float
    market_breadth: float
    market_return_20d: float
    market_state: str
    top_candidates: list[ThemeAlertCandidate]
    reason: str
    buy_point_candidates: list[ThemeAlertCandidate] = field(default_factory=list)


def build_strong_theme_alerts(
    ranking: pd.DataFrame,
    *,
    market_breadth: float,
    market_return_20d: float,
    market_state: str,
    signal_frame: pd.DataFrame | None = None,
    max_alerts: int = 3,
    top_rank_window: int = 50,
    min_industry_members: int = 8,
    min_top_ranked_count: int = 3,
    min_top_ranked_share: float = 0.06,
    min_avg_rank_score: float = 0.54,
    min_momentum_rank: float = 0.70,
    min_trend_rank: float = 0.64,
    min_industry_rank: float = 0.50,
) -> list[StrongThemeAlert]:
    """Detect research-only strong theme alerts from a ranking snapshot.

    This layer intentionally does not return buy orders. It highlights industries
    where ranked candidates are concentrated and the whole group has strong
    momentum/trend/industry-tailwind scores.
    """

    if ranking.empty or "industry" not in ranking.columns:
        return []

    frame = _attach_signal_features(ranking.copy(), signal_frame)
    if "is_tradeable" in frame.columns:
        frame = frame.loc[frame["is_tradeable"].fillna(False).astype(bool)].copy()
    if frame.empty:
        return []

    frame["industry"] = frame["industry"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["industry"] != ""].copy()
    if frame.empty:
        return []

    numeric_columns = [
        "rank_position",
        "rank_score",
        "momentum_rank",
        "trend_rank",
        "industry_rank",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["rank_position"].notna()].copy()
    if frame.empty:
        return []

    top_ranked = frame.loc[frame["rank_position"] <= int(top_rank_window)].copy()
    if top_ranked.empty:
        return []

    alerts: list[StrongThemeAlert] = []
    top_window_count = max(len(top_ranked), 1)
    for industry, group in frame.groupby("industry", sort=False):
        member_count = int(len(group))
        if member_count < int(min_industry_members):
            continue
        group_top = top_ranked.loc[top_ranked["industry"] == industry].copy()
        top_ranked_count = int(len(group_top))
        top_ranked_share = top_ranked_count / top_window_count
        avg_rank_score = float(group["rank_score"].mean())
        median_momentum_rank = float(group["momentum_rank"].median())
        median_trend_rank = float(group["trend_rank"].median())
        median_industry_rank = float(group["industry_rank"].median())
        if (
            top_ranked_count < int(min_top_ranked_count)
            or top_ranked_share < float(min_top_ranked_share)
            or avg_rank_score < float(min_avg_rank_score)
            or median_momentum_rank < float(min_momentum_rank)
            or median_trend_rank < float(min_trend_rank)
            or median_industry_rank < float(min_industry_rank)
        ):
            continue

        candidates = [
            _candidate_from_row(row)
            for _, row in group_top.sort_values("rank_position").head(5).iterrows()
        ]
        buy_point_candidates = [
            _candidate_from_row(row)
            for _, row in _buy_point_candidates(group_top).sort_values("rank_position").head(5).iterrows()
        ]
        reason = (
            f"{industry} 在前 {top_rank_window} 名中占 {top_ranked_count} 只"
            f"（{top_ranked_share:.0%}），行业平均排序分 {avg_rank_score:.3f}，"
            f"动量/趋势/行业因子分位 {median_momentum_rank:.0%}/"
            f"{median_trend_rank:.0%}/{median_industry_rank:.0%}。"
        )
        if market_state != "normal":
            reason += " 市场层仍未进入正常开仓状态，仅作为观察预警。"
        if buy_point_candidates:
            reason += f" 其中 {len(buy_point_candidates)} 只代表股处于回踩买点观察区。"
        alerts.append(
            StrongThemeAlert(
                industry=industry,
                member_count=member_count,
                top_ranked_count=top_ranked_count,
                top_ranked_share=top_ranked_share,
                avg_rank_score=avg_rank_score,
                median_momentum_rank=median_momentum_rank,
                median_trend_rank=median_trend_rank,
                median_industry_rank=median_industry_rank,
                market_breadth=float(market_breadth),
                market_return_20d=float(market_return_20d),
                market_state=str(market_state),
                top_candidates=candidates,
                reason=reason,
                buy_point_candidates=buy_point_candidates,
            )
        )

    return sorted(
        alerts,
        key=lambda item: (
            item.top_ranked_count,
            item.top_ranked_share,
            item.avg_rank_score,
            item.median_momentum_rank,
        ),
        reverse=True,
    )[: int(max_alerts)]


def format_theme_alert_brief(alert: StrongThemeAlert) -> str:
    candidates = "、".join(
        f"{item.symbol} {item.name}#{item.rank_position}" for item in alert.top_candidates[:3]
    )
    suffix = f"；代表股：{candidates}" if candidates else ""
    buy_points = "、".join(
        f"{item.symbol} {item.name}#{item.rank_position}" for item in alert.buy_point_candidates[:3]
    )
    buy_point_suffix = f"；买点观察：{buy_points}" if buy_points else ""
    return f"{alert.industry}：{alert.reason}{suffix}{buy_point_suffix}"


def _attach_signal_features(ranking: pd.DataFrame, signal_frame: pd.DataFrame | None) -> pd.DataFrame:
    if signal_frame is None or signal_frame.empty or "ts_code" not in ranking.columns or "ts_code" not in signal_frame.columns:
        return ranking
    feature_frame = signal_frame.copy()
    if "pullback_from_20d_high" not in feature_frame.columns and "drawdown_from_20d_high" in feature_frame.columns:
        feature_frame["pullback_from_20d_high"] = feature_frame["drawdown_from_20d_high"]
    feature_columns = [
        "ts_code",
        "return_5d",
        "pullback_from_20d_high",
        "close_to_ma_20",
        "amount_ratio_5d",
        "upper_shadow_pct",
    ]
    feature_columns = [column for column in feature_columns if column in feature_frame.columns]
    if len(feature_columns) <= 1:
        return ranking
    return ranking.merge(
        feature_frame[feature_columns].drop_duplicates(subset=["ts_code"]),
        on="ts_code",
        how="left",
    )


def _buy_point_candidates(group_top: pd.DataFrame) -> pd.DataFrame:
    if group_top.empty:
        return group_top
    frame = group_top.copy()
    for column in (
        "return_5d",
        "pullback_from_20d_high",
        "close_to_ma_20",
        "amount_ratio_5d",
        "upper_shadow_pct",
    ):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[
        frame["close_to_ma_20"].between(-0.03, 0.08)
        & frame["pullback_from_20d_high"].between(-0.15, -0.03)
        & frame["return_5d"].between(-0.06, 0.11)
        & frame["amount_ratio_5d"].fillna(1.0).between(0.75, 2.60)
        & frame["upper_shadow_pct"].fillna(0.0).le(0.50)
    ].copy()


def _candidate_from_row(row: pd.Series) -> ThemeAlertCandidate:
    return ThemeAlertCandidate(
        symbol=str(row.get("ts_code") or ""),
        name=str(row.get("name") or row.get("ts_code") or ""),
        industry=str(row.get("industry") or ""),
        rank_position=int(row.get("rank_position")),
        rank_score=float(row.get("rank_score") or 0.0),
        signal_type=str(row.get("signal_type") or ""),
        momentum_rank=float(row.get("momentum_rank") or 0.0),
        trend_rank=float(row.get("trend_rank") or 0.0),
        return_5d=_optional_float(row.get("return_5d")),
        pullback_from_20d_high=_optional_float(row.get("pullback_from_20d_high")),
        close_to_ma_20=_optional_float(row.get("close_to_ma_20")),
        amount_ratio_5d=_optional_float(row.get("amount_ratio_5d")),
        upper_shadow_pct=_optional_float(row.get("upper_shadow_pct")),
    )


def _optional_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number
