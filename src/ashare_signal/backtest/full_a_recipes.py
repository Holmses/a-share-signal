from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.backtest.full_a_momentum import FullAMomentumBacktestEngine
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.recipe import full_a_momentum_recipe


DEFAULT_FULL_A_RECIPE_STUDY_RECIPES = (
    "momentum_core",
    "trend_pullback_overlay",
    "quality_momentum_filter",
    "combo_v2",
)


@dataclass(slots=True)
class FullARecipeStudyResult:
    start_trade_date: str
    end_trade_date: str
    recipes: list[str]
    comparison_csv_path: Path
    comparison_json_path: Path
    markdown_path: Path


class FullARecipeStudyEngine:
    """Compare Full A momentum recipe combinations with baseline admission checks."""

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        recipes: list[str] | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.recipes = _normalize_study_recipes(recipes)

    def run(self, start_date: date | None = None, end_date: date | None = None) -> FullARecipeStudyResult:
        rows = []
        for recipe_name in self.recipes:
            rows.append(self._run_recipe(recipe_name, start_date=start_date, end_date=end_date))
        if not rows:
            raise ValueError("No Full A recipes were selected for comparison")

        baseline = rows[0]
        for row in rows:
            decision = _admission_decision(row, baseline)
            row["beats_baseline"] = decision["beats_baseline"]
            row["admission_status"] = decision["status"]
            row["admission_notes"] = decision["notes"]

        start_trade_date = str(rows[0]["start_trade_date"])
        end_trade_date = str(rows[0]["end_trade_date"])
        output_dir = self.base_dir / self.config.paths.reports_dir / "full-a-recipes"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{start_trade_date}-{end_trade_date}"
        comparison_csv_path = output_dir / f"{stem}-comparison.csv"
        comparison_json_path = output_dir / f"{stem}-comparison.json"
        markdown_path = output_dir / f"{stem}.md"

        pd.DataFrame([_csv_row(row) for row in rows]).to_csv(comparison_csv_path, index=False)
        comparison_json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(_render_markdown(rows), encoding="utf-8")
        return FullARecipeStudyResult(
            start_trade_date=start_trade_date,
            end_trade_date=end_trade_date,
            recipes=[str(row["recipe"]) for row in rows],
            comparison_csv_path=comparison_csv_path,
            comparison_json_path=comparison_json_path,
            markdown_path=markdown_path,
        )

    def _run_recipe(self, recipe_name: str, start_date: date | None, end_date: date | None) -> dict:
        controls = _recipe_controls(recipe_name)
        recipe = full_a_momentum_recipe(self.config, recipe_id=f"full_a_momentum_{recipe_name}", name=recipe_name)
        result = FullAMomentumBacktestEngine.from_recipe(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            recipe=recipe,
            enabled_recipes=controls["enabled_recipes"],
            overlay_recipes=controls["overlay_recipes"],
            quality_filter_enabled=controls["quality_filter_enabled"],
        ).run(start_date=start_date, end_date=end_date)
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        max_drawdown = float(summary["max_drawdown"])
        annual_return = float(summary["annual_return"])
        calmar = annual_return / abs(max_drawdown) if max_drawdown else math.inf
        return {
            "recipe": recipe_name,
            "entry_recipes": list(summary.get("entry_recipe_counts", {}).keys()),
            "start_trade_date": summary["start_trade_date"],
            "end_trade_date": summary["end_trade_date"],
            "ending_equity": float(summary["ending_equity"]),
            "total_return": float(summary["total_return"]),
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "sharpe": float(summary["sharpe"]),
            "turnover": float(summary["turnover"]),
            "trade_count": int(summary["trade_count"]),
            "sell_trade_count": int(summary["sell_trade_count"]),
            "win_rate": float(summary["win_rate"]),
            "risk_off_days": int(summary["risk_off_days"]),
            "average_position_count": float(summary["average_position_count"]),
            "average_invested_ratio": float(summary["average_invested_ratio"]),
            "sell_reason_counts": summary.get("sell_reason_counts", {}),
            "entry_recipe_counts": summary.get("entry_recipe_counts", {}),
            "market_state_counts": summary.get("market_state_counts", {}),
            "summary_path": str(result.summary_path),
            "trade_log_path": str(result.trade_log_path),
        }


def _recipe_controls(recipe_name: str) -> dict:
    if recipe_name == "momentum_core":
        return {"enabled_recipes": ["momentum_core"], "overlay_recipes": [], "quality_filter_enabled": False}
    if recipe_name == "trend_pullback_overlay":
        return {
            "enabled_recipes": ["momentum_core"],
            "overlay_recipes": ["trend_pullback_overlay"],
            "quality_filter_enabled": False,
        }
    if recipe_name == "quality_momentum_filter":
        return {"enabled_recipes": ["momentum_core"], "overlay_recipes": [], "quality_filter_enabled": True}
    if recipe_name == "combo_v2":
        return {
            "enabled_recipes": ["momentum_core"],
            "overlay_recipes": ["trend_pullback_overlay"],
            "quality_filter_enabled": True,
        }
    raise ValueError(f"Unsupported Full A recipe study variant: {recipe_name}")


def _admission_decision(row: dict, baseline: dict) -> dict:
    if row["recipe"] == baseline["recipe"]:
        return {"beats_baseline": False, "status": "baseline", "notes": "baseline"}
    checks = {
        "total_return": float(row["total_return"]) > float(baseline["total_return"]),
        "max_drawdown": abs(float(row["max_drawdown"])) <= abs(float(baseline["max_drawdown"])) * 1.10,
        "calmar": float(row["calmar"]) > float(baseline["calmar"]),
        "turnover": float(row["turnover"]) <= float(baseline["turnover"]) * 1.30,
        "trade_count": int(row["trade_count"]) <= max(int(baseline["trade_count"]) * 2, int(baseline["trade_count"]) + 10),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "beats_baseline": not failed,
        "status": "candidate" if not failed else "research_only",
        "notes": "passed production candidate checks" if not failed else "failed: " + ",".join(failed),
    }


def _normalize_study_recipes(recipes: list[str] | None) -> list[str]:
    if not recipes:
        return list(DEFAULT_FULL_A_RECIPE_STUDY_RECIPES)
    names = []
    for recipe in recipes:
        name = str(recipe).strip()
        if name and name not in names:
            names.append(name)
    if "momentum_core" not in names:
        names.insert(0, "momentum_core")
    return names


def _csv_row(row: dict) -> dict:
    result = dict(row)
    for key in ("entry_recipes", "sell_reason_counts", "entry_recipe_counts", "market_state_counts"):
        result[key] = json.dumps(result.get(key), ensure_ascii=False, sort_keys=True)
    return result


def _render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Full A Recipe Comparison",
        "",
        "| recipe | status | total_return | max_drawdown | calmar | turnover | trades | win_rate | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {recipe} | {status} | {total_return:.4f} | {max_drawdown:.4f} | {calmar:.4f} | "
            "{turnover:.4f} | {trade_count} | {win_rate:.4f} | {notes} |".format(
                recipe=row["recipe"],
                status=row["admission_status"],
                total_return=float(row["total_return"]),
                max_drawdown=float(row["max_drawdown"]),
                calmar=float(row["calmar"]),
                turnover=float(row["turnover"]),
                trade_count=int(row["trade_count"]),
                win_rate=float(row["win_rate"]),
                notes=row["admission_notes"],
            )
        )
    lines.extend(
        [
            "",
            "Production candidate checks: total return beats baseline, absolute max drawdown is within 110% of baseline,",
            "Calmar beats baseline, turnover is within 130% of baseline, and trade count does not abnormally increase.",
            "",
        ]
    )
    return "\n".join(lines)
