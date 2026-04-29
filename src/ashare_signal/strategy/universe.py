from __future__ import annotations

from ashare_signal.config import AppConfig


def apply_universe_filters(snapshot, config: AppConfig):
    df = snapshot.copy()

    is_a_share = df["exchange"].isin(["SSE", "SZSE"])
    not_beijing = ~df["ts_code"].fillna("").str.endswith(".BJ")
    market_allowed = ~df["market"].fillna("").str.contains("北交", regex=False)

    st_filter = ~df["is_st"]
    if not config.filters.exclude_st:
        st_filter = st_filter | df["is_st"]

    suspension_filter = ~df["is_suspended"]
    if not config.filters.exclude_suspended:
        suspension_filter = suspension_filter | df["is_suspended"]

    df["passes_exchange_filter"] = is_a_share & not_beijing & market_allowed
    df["passes_st_filter"] = st_filter
    df["passes_suspension_filter"] = suspension_filter
    df["passes_listing_age_filter"] = (
        df["listed_days"].fillna(-1) >= config.filters.min_list_days
    )
    df["passes_price_filter"] = df["close"].fillna(0.0) >= config.filters.min_price
    df["passes_liquidity_filter"] = (
        df["avg_amount_20d_yuan"].fillna(0.0) >= config.filters.min_avg_turnover
    )

    df["is_candidate"] = (
        df["passes_exchange_filter"]
        & df["passes_st_filter"]
        & df["passes_suspension_filter"]
        & df["passes_listing_age_filter"]
        & df["passes_price_filter"]
        & df["passes_liquidity_filter"]
    )
    df["exclude_reason"] = "eligible"

    reason_order = [
        ("exchange_not_supported", ~df["passes_exchange_filter"]),
        ("st_stock", ~df["passes_st_filter"]),
        ("suspended", ~df["passes_suspension_filter"]),
        ("listed_days_too_short", ~df["passes_listing_age_filter"]),
        ("price_below_threshold", ~df["passes_price_filter"]),
        ("liquidity_below_threshold", ~df["passes_liquidity_filter"]),
    ]
    for label, mask in reason_order:
        df.loc[(df["exclude_reason"] == "eligible") & mask, "exclude_reason"] = label

    return df.sort_values(
        ["is_candidate", "momentum_20d_rank_pct", "avg_amount_20d_yuan"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

