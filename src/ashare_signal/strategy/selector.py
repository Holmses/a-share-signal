from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ashare_signal.config import SelectionConfig
from ashare_signal.domain.models import Candidate, Position


@dataclass(slots=True)
class SignalSelectionResult:
    buy_candidates: list[Candidate]
    sell_candidates: list[Candidate]
    notes: list[str]


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "未知"
    return f"{value * 100:.1f}%"


def _format_amount_yi(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "未知"
    return f"{value / 100000000:.2f}亿"


def _coerce_universe(universe: pd.DataFrame) -> pd.DataFrame:
    frame = universe.copy()
    numeric_columns = [
        "close",
        "pct_chg",
        "amount_yuan",
        "return_1d",
        "momentum_5d",
        "momentum_20d",
        "momentum_20d_rank_pct",
        "volatility_20d",
        "volatility_20d_rank_pct",
        "avg_amount_20d_yuan",
        "ma_10",
        "ma_20",
        "close_to_ma_10",
        "close_to_ma_20",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "total_mv_yuan",
        "circ_mv_yuan",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    bool_columns = [
        "is_candidate",
        "is_st",
        "is_suspended",
        "passes_exchange_filter",
        "passes_st_filter",
        "passes_suspension_filter",
        "passes_listing_age_filter",
        "passes_price_filter",
        "passes_liquidity_filter",
    ]
    for column in bool_columns:
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(frame[column])
            )
            if frame[column].dtype != bool:
                frame[column] = frame[column].astype(bool)

    return frame


def _rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending).fillna(0.0)


def _buy_reason(row: pd.Series) -> str:
    return (
        f"20日动量分位 {_format_pct(row['momentum_20d_rank_pct'])}，"
        f"较20日均线 {_format_pct(row['close_to_ma_20'])}，"
        f"20日日均成交额 {_format_amount_yi(row['avg_amount_20d_yuan'])}，"
        f"20日波动率 {_format_pct(row['volatility_20d'])}。"
    )


def _sell_reason(row: pd.Series) -> str:
    return (
        f"20日动量 {_format_pct(row['momentum_20d'])}，"
        f"较10日均线 {_format_pct(row['close_to_ma_10'])}，"
        f"较20日均线 {_format_pct(row['close_to_ma_20'])}，"
        f"20日波动率 {_format_pct(row['volatility_20d'])}。"
    )


class UniverseSignalSelector:
    def __init__(
        self,
        selection_config: SelectionConfig,
        top_buy_n: int = 1,
        top_sell_n: int = 1,
    ) -> None:
        self.selection_config = selection_config
        self.top_buy_n = top_buy_n
        self.top_sell_n = top_sell_n

    def select(self, universe: pd.DataFrame, positions: Iterable[Position]) -> SignalSelectionResult:
        frame = _coerce_universe(universe)
        holdings = list(positions)
        holding_symbols = {position.symbol for position in holdings}
        notes: list[str] = []

        available_holdings = frame.loc[frame["ts_code"].isin(holding_symbols)].copy()
        missing_symbols = sorted(holding_symbols - set(available_holdings["ts_code"]))
        if missing_symbols:
            notes.append(f"以下持仓未出现在当日 universe 快照中，已跳过卖出评分：{', '.join(missing_symbols)}")

        buy_pool = frame.loc[(frame["is_candidate"]) & (~frame["ts_code"].isin(holding_symbols))].copy()
        buy_candidates = self._select_buy_candidates(buy_pool)

        sell_pool = available_holdings.copy()
        sell_candidates = self._select_sell_candidates(sell_pool)

        if buy_pool.empty:
            notes.append("买入池为空：当前快照中没有可买候选。")
        if sell_pool.empty:
            notes.append("卖出池为空：当前持仓没有出现在当日快照中。")

        return SignalSelectionResult(
            buy_candidates=buy_candidates,
            sell_candidates=sell_candidates,
            notes=notes,
        )

    def _select_buy_candidates(self, buy_pool: pd.DataFrame) -> list[Candidate]:
        if buy_pool.empty:
            return []

        buy_pool = buy_pool.copy()
        buy_pool["liq_rank"] = _rank(buy_pool["avg_amount_20d_yuan"])
        buy_pool["ma20_rank"] = _rank(buy_pool["close_to_ma_20"])
        buy_pool["ma10_rank"] = _rank(buy_pool["close_to_ma_10"])
        buy_pool["vol_stability_rank"] = _rank(buy_pool["volatility_20d"], ascending=False)
        buy_pool["volume_ratio_rank"] = _rank(buy_pool["volume_ratio"])
        buy_pool["buy_score"] = (
            buy_pool["momentum_20d_rank_pct"].fillna(0.0) * self.selection_config.buy_momentum_weight
            + buy_pool["liq_rank"] * self.selection_config.buy_liquidity_weight
            + buy_pool["ma20_rank"] * self.selection_config.buy_ma20_weight
            + buy_pool["ma10_rank"] * self.selection_config.buy_ma10_weight
            + buy_pool["vol_stability_rank"] * self.selection_config.buy_volatility_weight
            + buy_pool["volume_ratio_rank"] * self.selection_config.buy_volume_ratio_weight
        )
        ranked = buy_pool.sort_values(
            ["buy_score", "momentum_20d_rank_pct", "avg_amount_20d_yuan"],
            ascending=[False, False, False],
        ).head(self.top_buy_n)
        return [
            Candidate(
                symbol=row["ts_code"],
                name=row["name"],
                score=float(row["buy_score"]),
                reason=_buy_reason(row),
                last_close=float(row["close"]),
            )
            for _, row in ranked.iterrows()
        ]

    def _select_sell_candidates(self, sell_pool: pd.DataFrame) -> list[Candidate]:
        if sell_pool.empty:
            return []

        sell_pool = sell_pool.copy()
        sell_pool["momentum_rank"] = _rank(sell_pool["momentum_20d"])
        sell_pool["ma10_rank"] = _rank(sell_pool["close_to_ma_10"])
        sell_pool["ma20_rank"] = _rank(sell_pool["close_to_ma_20"])
        sell_pool["vol_stability_rank"] = _rank(sell_pool["volatility_20d"], ascending=False)
        sell_pool["sell_health_score"] = (
            sell_pool["momentum_rank"] * self.selection_config.sell_momentum_weight
            + sell_pool["ma10_rank"] * self.selection_config.sell_ma10_weight
            + sell_pool["ma20_rank"] * self.selection_config.sell_ma20_weight
            + sell_pool["vol_stability_rank"] * self.selection_config.sell_volatility_weight
        )
        ranked = sell_pool.sort_values(
            ["sell_health_score", "momentum_20d", "close_to_ma_10"],
            ascending=[True, True, True],
        ).head(self.top_sell_n)
        return [
            Candidate(
                symbol=row["ts_code"],
                name=row["name"],
                score=float(row["sell_health_score"]),
                reason=_sell_reason(row),
                last_close=float(row["close"]),
            )
            for _, row in ranked.iterrows()
        ]

    def should_open_new_position(self, buy_candidate: Candidate | None) -> bool:
        if buy_candidate is None:
            return False
        return buy_candidate.score >= self.selection_config.min_buy_score

    def should_rotate(
        self,
        buy_candidate: Candidate | None,
        sell_candidate: Candidate | None,
    ) -> bool:
        if buy_candidate is None or sell_candidate is None:
            return False
        return buy_candidate.score >= (
            sell_candidate.score + self.selection_config.rotation_edge
        ) and self.should_open_new_position(buy_candidate)
