from __future__ import annotations

from ashare_signal.config import AppConfig
from ashare_signal.strategy.filters import HardTradeFilter


def apply_universe_filters(snapshot, config: AppConfig):
    df = HardTradeFilter(config).apply(snapshot)

    return df.sort_values(
        ["is_candidate", "momentum_20d_rank_pct", "avg_amount_20d_yuan"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
