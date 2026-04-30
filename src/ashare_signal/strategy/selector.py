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
        "avg_amount_5d_yuan",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_60",
        "close_to_ma_5",
        "close_to_ma_10",
        "close_to_ma_20",
        "close_to_ma_60",
        "ma_20_to_ma_60",
        "ma_60_slope_20d",
        "pullback_from_20d_high",
        "low_to_prev_low",
        "amount_ratio_5d",
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


def _clip_score(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return ((series - lower) / (upper - lower)).clip(lower=0.0, upper=1.0).fillna(0.0)


def _buy_reason(row: pd.Series) -> str:
    return (
        f"MA20/MA60 {_format_pct(row['ma_20_to_ma_60'])}，"
        f"MA60斜率 {_format_pct(row['ma_60_slope_20d'])}，"
        f"20日高点回撤 {_format_pct(row['pullback_from_20d_high'])}，"
        f"较20日均线 {_format_pct(row['close_to_ma_20'])}，"
        f"较5日均线 {_format_pct(row['close_to_ma_5'])}，"
        f"20日动量分位 {_format_pct(row['momentum_20d_rank_pct'])}，"
        f"20日日均成交额 {_format_amount_yi(row['avg_amount_20d_yuan'])}，"
        f"5日量能比 {row['amount_ratio_5d']:.2f}。"
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
        buy_pool = buy_pool.loc[
            (buy_pool["close_to_ma_60"] > 0)
            & (buy_pool["ma_20_to_ma_60"] > 0)
            & (buy_pool["ma_60_slope_20d"] > 0)
            & (buy_pool["momentum_20d"] > 0)
            & (buy_pool["pullback_from_20d_high"] >= self.selection_config.buy_min_pullback_from_20d_high)
            & (buy_pool["pullback_from_20d_high"] <= self.selection_config.buy_max_pullback_from_20d_high)
            & (buy_pool["close_to_ma_20"] >= self.selection_config.buy_min_close_to_ma20)
            & (buy_pool["close_to_ma_20"] <= self.selection_config.buy_max_close_to_ma20)
            & (buy_pool["close_to_ma_5"] > 0)
            & (buy_pool["return_1d"] > 0)
            & (buy_pool["return_1d"] <= self.selection_config.buy_max_return_1d)
            & (buy_pool["close_to_ma_5"] <= self.selection_config.buy_max_close_to_ma5)
            & (buy_pool["low_to_prev_low"] >= 0)
            & (buy_pool["low_to_prev_low"] <= self.selection_config.buy_max_low_to_prev_low)
            & (buy_pool["momentum_5d"] >= self.selection_config.buy_min_momentum_5d)
            & (buy_pool["amount_ratio_5d"] >= self.selection_config.buy_min_amount_ratio_5d)
            & (buy_pool["amount_ratio_5d"] <= self.selection_config.buy_max_amount_ratio_5d)
            & (buy_pool["total_mv_yuan"] >= self.selection_config.buy_min_total_mv_yuan)
            & (
                buy_pool["volume_ratio"].isna()
                | (buy_pool["volume_ratio"] <= self.selection_config.buy_max_volume_ratio)
            )
        ].copy()
        if buy_pool.empty:
            return []

        buy_pool["liq_rank"] = _rank(buy_pool["avg_amount_20d_yuan"])
        buy_pool["ma20_rank"] = _rank(buy_pool["close_to_ma_20"])
        buy_pool["ma10_rank"] = _rank(buy_pool["close_to_ma_10"])
        buy_pool["vol_stability_rank"] = _rank(buy_pool["volatility_20d"], ascending=False)
        buy_pool["volume_ratio_rank"] = _rank(buy_pool["volume_ratio"])
        buy_pool["trend_rank"] = _rank(buy_pool["ma_60_slope_20d"] + buy_pool["ma_20_to_ma_60"])
        pullback_center = (
            self.selection_config.buy_min_pullback_from_20d_high
            + self.selection_config.buy_max_pullback_from_20d_high
        ) / 2
        pullback_width = abs(
            self.selection_config.buy_max_pullback_from_20d_high
            - self.selection_config.buy_min_pullback_from_20d_high
        ) / 2
        buy_pool["pullback_score"] = (
            1.0 - ((buy_pool["pullback_from_20d_high"] - pullback_center).abs() / pullback_width)
        ).clip(lower=0.0, upper=1.0)
        buy_pool["reversal_score"] = (
            _clip_score(buy_pool["return_1d"], 0.0, 0.05) * 0.45
            + _clip_score(buy_pool["close_to_ma_5"], 0.0, 0.04) * 0.35
            + _clip_score(buy_pool["low_to_prev_low"], 0.0, 0.03) * 0.20
        )
        buy_pool["buy_score"] = (
            buy_pool["momentum_20d_rank_pct"].fillna(0.0) * self.selection_config.buy_momentum_weight
            + buy_pool["liq_rank"] * self.selection_config.buy_liquidity_weight
            + buy_pool["ma20_rank"] * self.selection_config.buy_ma20_weight
            + buy_pool["ma10_rank"] * self.selection_config.buy_ma10_weight
            + buy_pool["vol_stability_rank"] * self.selection_config.buy_volatility_weight
            + buy_pool["volume_ratio_rank"] * self.selection_config.buy_volume_ratio_weight
            + buy_pool["trend_rank"] * self.selection_config.buy_trend_weight
            + buy_pool["pullback_score"] * self.selection_config.buy_pullback_weight
            + buy_pool["reversal_score"] * self.selection_config.buy_reversal_weight
        )
        ranked = buy_pool.sort_values(
            ["buy_score", "reversal_score", "pullback_score", "momentum_20d_rank_pct", "avg_amount_20d_yuan"],
            ascending=[False, False, False, False, False],
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
        sell_pool["momentum_health"] = _clip_score(sell_pool["momentum_20d"], -0.20, 0.20)
        sell_pool["ma10_health"] = _clip_score(sell_pool["close_to_ma_10"], -0.10, 0.10)
        sell_pool["ma20_health"] = _clip_score(sell_pool["close_to_ma_20"], -0.15, 0.15)
        sell_pool["vol_stability_health"] = (1.0 - (sell_pool["volatility_20d"] / 0.08)).clip(
            lower=0.0,
            upper=1.0,
        ).fillna(0.0)
        sell_pool["sell_health_score"] = (
            sell_pool["momentum_health"] * self.selection_config.sell_momentum_weight
            + sell_pool["ma10_health"] * self.selection_config.sell_ma10_weight
            + sell_pool["ma20_health"] * self.selection_config.sell_ma20_weight
            + sell_pool["vol_stability_health"] * self.selection_config.sell_volatility_weight
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

    def market_allows_buy(self, universe: pd.DataFrame) -> bool:
        frame = _coerce_universe(universe)
        pool = frame.loc[frame["is_candidate"]].copy()
        if pool.empty:
            return False
        breadth = (
            (pool["close_to_ma_20"] > 0)
            & (pool["momentum_20d"] > 0)
        ).mean()
        return float(breadth) >= self.selection_config.market_min_breadth

    def should_rotate(
        self,
        buy_candidate: Candidate | None,
        sell_candidate: Candidate | None,
    ) -> bool:
        if buy_candidate is None or sell_candidate is None:
            return False
        if sell_candidate.score > self.selection_config.sell_health_exit_threshold:
            return False
        return buy_candidate.score >= (
            sell_candidate.score + self.selection_config.rotation_edge
        ) and self.should_open_new_position(buy_candidate)
