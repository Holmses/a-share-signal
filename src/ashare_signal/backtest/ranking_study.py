from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.features.pipeline import build_universe_snapshot
from ashare_signal.strategy.ranking import SUPPORTED_RANKING_VARIANTS
from ashare_signal.strategy.ranking import build_ranking_snapshot, render_ranking_factor_map
from ashare_signal.utils.dates import to_compact_date


@dataclass(slots=True)
class RankingStudyResult:
    trade_date: str
    variant: str
    total_symbols: int
    tradeable_symbols: int
    top_n: int
    ranking_path: Path
    markdown_path: Path
    factor_map_path: Path


class RankingStudyEngine:
    """Research-only cross-sectional ranking snapshot.

    The engine deliberately writes only ranking and research reports. It does
    not touch daily signal boards, paper-trading state, or live positions.
    """

    DEFAULT_VARIANT = "quality_momentum_rank"

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        variant: str = DEFAULT_VARIANT,
        top_n: int = 20,
    ) -> None:
        if variant not in SUPPORTED_RANKING_VARIANTS:
            raise ValueError(f"Unsupported ranking variant: {variant}")
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.variant = variant
        self.top_n = max(int(top_n), 1)

    def run(self, as_of: date) -> RankingStudyResult:
        requested_trade_date = to_compact_date(as_of)
        trade_date = self.repository.resolve_trade_date(requested_trade_date)
        universe = build_universe_snapshot(
            config=self.config,
            repository=self.repository,
            trade_date=trade_date,
        )
        ranking = build_ranking_snapshot(universe, self.config, variant=self.variant)
        if ranking.empty:
            raise ValueError(f"Ranking study produced no rows for {trade_date}.")

        ranking_dir = self.base_dir / self.config.paths.processed_data_dir / "ranking"
        ranking_dir.mkdir(parents=True, exist_ok=True)
        ranking_path = ranking_dir / f"{trade_date}-ranking-{self.variant}.csv"
        ranking.to_csv(ranking_path, index=False)

        research_dir = self.base_dir / self.config.paths.reports_dir / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        factor_map_path = research_dir / "ranking-factor-map.md"
        factor_map_path.write_text(render_ranking_factor_map(), encoding="utf-8")

        markdown_path = research_dir / f"{trade_date}-ranking-{self.variant}-top{self.top_n}.md"
        markdown_path.write_text(
            self._render_markdown(
                trade_date=trade_date,
                ranking=ranking,
                ranking_path=ranking_path,
                factor_map_path=factor_map_path,
            ),
            encoding="utf-8",
        )

        return RankingStudyResult(
            trade_date=trade_date,
            variant=self.variant,
            total_symbols=int(len(ranking)),
            tradeable_symbols=int(ranking["is_tradeable"].fillna(False).astype(bool).sum()),
            top_n=self.top_n,
            ranking_path=ranking_path,
            markdown_path=markdown_path,
            factor_map_path=factor_map_path,
        )

    def _render_markdown(
        self,
        *,
        trade_date: str,
        ranking: pd.DataFrame,
        ranking_path: Path,
        factor_map_path: Path,
    ) -> str:
        tradeable = ranking.loc[ranking["is_tradeable"].fillna(False).astype(bool)].copy()
        top = tradeable.sort_values(["rank_position", "ts_code"]).head(self.top_n)
        filtered_counts = ranking["filter_reason"].fillna("unknown").value_counts().sort_index()
        signal_counts = tradeable["signal_type"].fillna("unknown").value_counts().sort_index()

        lines = [
            f"# Cross-Section Ranking Study: {self.variant}",
            "",
            "This report is research-only and does not affect generate-signal, paper-trade, or scheduler state.",
            "",
            "## Run Metadata",
            "",
            f"- Trade date: {trade_date}",
            f"- Variant: {self.variant}",
            f"- Total symbols: {len(ranking)}",
            f"- Tradeable symbols: {len(tradeable)}",
            f"- Ranking CSV: {ranking_path}",
            f"- Factor map: {factor_map_path}",
            "",
            "## Filter Mix",
            "",
        ]
        for reason, count in filtered_counts.items():
            lines.append(f"- {reason}: {int(count)}")

        lines.extend(["", "## Signal Type Mix", ""])
        if signal_counts.empty:
            lines.append("- No tradeable symbols.")
        else:
            for signal_type, count in signal_counts.items():
                lines.append(f"- {signal_type}: {int(count)}")

        lines.extend(
            [
                "",
                f"## Top {self.top_n}",
                "",
                "| rank | ts_code | name | group | industry | score | signal_type | explanation |",
                "| ---: | --- | --- | --- | --- | ---: | --- | --- |",
            ]
        )
        if top.empty:
            lines.append("|  |  |  |  |  |  |  | No tradeable ranking rows. |")
        else:
            for _, row in top.iterrows():
                lines.append(
                    "| "
                    f"{_format_rank(row['rank_position'])} | "
                    f"{_cell(row['ts_code'])} | "
                    f"{_cell(row['name'])} | "
                    f"{_cell(row['universe_group'])} | "
                    f"{_cell(row['industry'])} | "
                    f"{float(row['rank_score']):.4f} | "
                    f"{_cell(row['signal_type'])} | "
                    f"{_cell(row['score_explain'])} |"
                )
        lines.append("")
        return "\n".join(lines)


def _format_rank(value) -> str:
    if pd.isna(value):
        return ""
    return str(int(value))


def _cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "/")
