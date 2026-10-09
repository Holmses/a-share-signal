from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json

import pandas as pd

from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository


@dataclass(slots=True)
class RecipeComparisonResult:
    start_entry_date: str
    end_entry_date: str
    start_signal_date: str
    end_signal_date: str
    recipes: list[str]
    horizons: list[int]
    event_count: int
    summary_path: Path
    summary_csv_path: Path
    events_path: Path
    daily_path: Path
    exposure_path: Path
    portfolio_path: Path
    portfolio_summary_path: Path


class RecipeComparisonStudyEngine:
    """Research-only multi-recipe comparison on a shared event-study surface."""

    DEFAULT_HORIZONS = (1, 3, 5, 10)
    DEFAULT_GROUPS = ("main", "chinext", "star")
    QUALITY_MOMENTUM_RECIPE = "quality_momentum_rank"
    BASELINE_RECIPE = "baseline_current_selector"
    COMBO_RECIPE = "combo_configured"

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        recipes: list[str] | None = None,
        groups: list[str] | None = None,
        top_n_per_recipe: int = 5,
        horizons: list[int] | None = None,
        min_avg_amount_yuan: float = 50_000_000.0,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.groups = _dedupe(groups or list(self.DEFAULT_GROUPS))
        self.top_n_per_recipe = max(int(top_n_per_recipe), 1)
        self.horizons = sorted({int(value) for value in (horizons or list(self.DEFAULT_HORIZONS)) if int(value) > 0})
        if not self.horizons:
            raise ValueError("At least one positive horizon is required.")
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.configured_recipe_ids = [
            str(recipe.name) for recipe in getattr(config, "recipes", ()) if getattr(recipe, "enabled", True)
        ]
        default_recipes = self.configured_recipe_ids + [self.QUALITY_MOMENTUM_RECIPE, self.COMBO_RECIPE]
        self.recipes = _dedupe(recipes or default_recipes)
        unknown = set(self.recipes) - set(default_recipes + [self.BASELINE_RECIPE])
        if unknown:
            raise ValueError(f"Unsupported recipe comparison recipes: {sorted(unknown)}")
        if self.COMBO_RECIPE in self.recipes and not self.configured_recipe_ids:
            raise ValueError("combo_configured requires at least one enabled [[recipes]] config entry.")
        self.roundtrip_cost = (
            float(config.pricing.buy_markup)
            + float(config.pricing.sell_markdown)
            + float(config.backtest.commission_rate) * 2
            + float(config.backtest.stamp_duty_rate)
        )

    def run(self, start_date: date | None = None, end_date: date | None = None) -> RecipeComparisonResult:
        cached_dates = self.repository.complete_daily_cache_dates()
        if not cached_dates:
            raise ValueError("Daily Tushare cache is empty. Run `ashare-signal sync-tushare` first.")

        helper = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=max(self.top_n_per_recipe, 5),
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=["quality_momentum"],
            horizons=self.horizons,
        )
        resolved_end = helper._resolve_cached_end(cached_dates, end_date)
        resolved_start = helper._resolve_cached_start(cached_dates, start_date, resolved_end)
        start_index = cached_dates.index(resolved_start)
        end_index = cached_dates.index(resolved_end)
        required_history = helper.minimum_backtest_history_trade_days()
        if start_index < required_history:
            raise ValueError(
                "Recipe comparison needs at least "
                f"{required_history} complete trade days before entry date {resolved_start}."
            )

        max_horizon = max(self.horizons)
        last_entry_index = min(end_index, len(cached_dates) - max_horizon)
        if last_entry_index < start_index:
            raise ValueError(
                "Recipe comparison has no entry dates with full forward horizon. "
                f"Need {max_horizon} cached trade days after each entry date."
            )

        signal_start_index = start_index - 1
        signal_end_index = last_entry_index - 1
        feature_dates = cached_dates[
            max(0, signal_start_index - helper.factor_history_trade_days()) : last_entry_index + max_horizon
        ]
        price_dates = cached_dates[start_index : last_entry_index + max_horizon]
        factor_frame = helper._build_factor_frame(feature_dates)
        price_map = helper._load_price_map(price_dates)

        events: list[dict] = []
        daily_rows: list[dict] = []
        comparison_recipes = [recipe for recipe in self.recipes if recipe != self.COMBO_RECIPE]
        for signal_index in range(signal_start_index, signal_end_index + 1):
            signal_trade_date = cached_dates[signal_index]
            entry_index = signal_index + 1
            entry_date = cached_dates[entry_index]
            day_frame = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_trade_date]
            selected_by_recipe: dict[str, list[dict]] = {}
            if day_frame.empty:
                continue
            for recipe in comparison_recipes:
                selected = self._select_recipe(day_frame, recipe)
                selected_by_recipe[recipe] = selected
                daily_rows.append(
                    {
                        "signal_trade_date": signal_trade_date,
                        "entry_trade_date": entry_date,
                        "recipe": recipe,
                        "selected_count": len(selected),
                    }
                )
                events.extend(
                    self._build_events(
                        helper=helper,
                        selected=selected,
                        recipe=recipe,
                        source_recipe=recipe,
                        signal_trade_date=signal_trade_date,
                        entry_index=entry_index,
                        cached_dates=cached_dates,
                        price_map=price_map,
                    )
                )
            if self.COMBO_RECIPE in self.recipes:
                combo_selected = self._combine_recipe_candidates(selected_by_recipe)
                daily_rows.append(
                    {
                        "signal_trade_date": signal_trade_date,
                        "entry_trade_date": entry_date,
                        "recipe": self.COMBO_RECIPE,
                        "selected_count": len(combo_selected),
                    }
                )
                events.extend(
                    self._build_events(
                        helper=helper,
                        selected=combo_selected,
                        recipe=self.COMBO_RECIPE,
                        source_recipe=None,
                        signal_trade_date=signal_trade_date,
                        entry_index=entry_index,
                        cached_dates=cached_dates,
                        price_map=price_map,
                    )
                )

        events_frame = pd.DataFrame(events)
        if events_frame.empty:
            raise ValueError("Recipe comparison produced no events.")
        daily_frame = pd.DataFrame(daily_rows)
        summary_frame = summarize_recipe_events(events_frame, self.horizons)
        exposure_frame = summarize_recipe_exposure(events_frame)
        portfolio_frame = build_recipe_portfolio_curves(events_frame, self.horizons)
        portfolio_summary_frame = summarize_recipe_portfolios(portfolio_frame)

        reports_dir = self.base_dir / self.config.paths.reports_dir / "recipe-comparison"
        reports_dir.mkdir(parents=True, exist_ok=True)
        recipes_slug = "-".join(recipe.replace("_", "-") for recipe in self.recipes)
        horizon_slug = "h" + "-".join(str(value) for value in self.horizons)
        stem = f"recipe-comparison-{recipes_slug}-{horizon_slug}-{resolved_start}-{cached_dates[last_entry_index]}"
        events_path = reports_dir / f"{stem}-events.csv"
        daily_path = reports_dir / f"{stem}-daily.csv"
        summary_csv_path = reports_dir / f"{stem}-summary.csv"
        exposure_path = reports_dir / f"{stem}-exposure.csv"
        portfolio_path = reports_dir / f"{stem}-portfolio.csv"
        portfolio_summary_path = reports_dir / f"{stem}-portfolio-summary.csv"
        summary_path = reports_dir / f"{stem}-summary.json"

        events_frame.to_csv(events_path, index=False)
        daily_frame.to_csv(daily_path, index=False)
        summary_frame.to_csv(summary_csv_path, index=False)
        exposure_frame.to_csv(exposure_path, index=False)
        portfolio_frame.to_csv(portfolio_path, index=False)
        portfolio_summary_frame.to_csv(portfolio_summary_path, index=False)

        payload = {
            "strategy": "recipe_comparison_study",
            "start_entry_date": resolved_start,
            "end_entry_date": cached_dates[last_entry_index],
            "start_signal_date": cached_dates[signal_start_index],
            "end_signal_date": cached_dates[signal_end_index],
            "requested_end_date": resolved_end,
            "recipes": self.recipes,
            "configured_recipes": self.configured_recipe_ids,
            "groups": self.groups,
            "top_n_per_recipe": self.top_n_per_recipe,
            "horizons": self.horizons,
            "roundtrip_cost": self.roundtrip_cost,
            "event_count": int(len(events_frame)),
            "summary_csv_path": str(summary_csv_path),
            "events_path": str(events_path),
            "daily_path": str(daily_path),
            "exposure_path": str(exposure_path),
            "portfolio_path": str(portfolio_path),
            "portfolio_summary_path": str(portfolio_summary_path),
            "summary": summary_frame.to_dict(orient="records"),
            "exposure": exposure_frame.to_dict(orient="records"),
            "portfolio_summary": portfolio_summary_frame.to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return RecipeComparisonResult(
            start_entry_date=resolved_start,
            end_entry_date=cached_dates[last_entry_index],
            start_signal_date=cached_dates[signal_start_index],
            end_signal_date=cached_dates[signal_end_index],
            recipes=self.recipes,
            horizons=self.horizons,
            event_count=int(len(events_frame)),
            summary_path=summary_path,
            summary_csv_path=summary_csv_path,
            events_path=events_path,
            daily_path=daily_path,
            exposure_path=exposure_path,
            portfolio_path=portfolio_path,
            portfolio_summary_path=portfolio_summary_path,
        )

    def _select_recipe(self, day_frame: pd.DataFrame, recipe: str) -> list[dict]:
        frame = day_frame.copy()
        frame = frame.loc[frame["group"].isin(self.groups)]
        frame = frame.loc[frame["avg_amount_20d_yuan"].fillna(0.0) >= self.min_avg_amount_yuan]
        if frame.empty:
            return []
        if recipe == self.QUALITY_MOMENTUM_RECIPE:
            return _select_top_by_group(
                frame.loc[_quality_momentum_mask(frame)].copy(),
                recipe=recipe,
                score_column="quality_momentum_score",
                top_n=self.top_n_per_recipe,
            )
        if recipe == self.BASELINE_RECIPE:
            return _select_top_by_group(
                frame,
                recipe=recipe,
                score_column="quality_score",
                top_n=self.top_n_per_recipe,
            )
        if recipe == "trend_pullback_rank":
            selected = frame.loc[_trend_pullback_mask(frame)].copy()
            selected["trend_pullback_rank_score"] = (
                selected["return_30d_rank"].fillna(0.0) * 0.30
                + selected["return_90d_rank"].fillna(0.0) * 0.18
                + selected["trend_quality_score"].fillna(0.0) * 0.18
                + selected["near_high_score"].fillna(0.0) * 0.14
                + selected["amount_rank"].fillna(0.0) * 0.12
                + selected["stability_score"].fillna(0.0) * 0.08
            )
            return _select_top_by_group(
                selected,
                recipe=recipe,
                score_column="trend_pullback_rank_score",
                top_n=self.top_n_per_recipe,
            )
        if recipe == "rebound_bottoming_rank":
            selected = frame.loc[_rebound_bottoming_mask(frame)].copy()
            depth = (1.0 - ((selected["drawdown_from_20d_high"].fillna(-0.30).abs() - 0.10) / 0.25)).clip(
                lower=0.0,
                upper=1.0,
            )
            selected["rebound_bottoming_rank_score"] = (
                depth.fillna(0.0) * 0.30
                + selected["stability_score"].fillna(0.0) * 0.20
                + selected["amount_rank"].fillna(0.0) * 0.18
                + selected["volume_ratio_score"].fillna(0.0) * 0.12
                + selected["financial_quality_score"].fillna(0.5) * 0.10
                + (1.0 - selected["market_cap_rank"].fillna(0.5)).clip(lower=0.0, upper=1.0) * 0.10
            )
            return _select_top_by_group(
                selected,
                recipe=recipe,
                score_column="rebound_bottoming_rank_score",
                top_n=self.top_n_per_recipe,
            )
        raise ValueError(f"Unsupported recipe: {recipe}")

    def _combine_recipe_candidates(self, selected_by_recipe: dict[str, list[dict]]) -> list[dict]:
        rows: dict[str, dict] = {}
        for source_recipe, selected in selected_by_recipe.items():
            for row in selected:
                symbol = str(row["ts_code"])
                candidate = dict(row)
                candidate["source_recipe"] = source_recipe
                candidate["combo_score"] = float(row["variant_score"])
                if symbol not in rows or candidate["combo_score"] > rows[symbol]["combo_score"]:
                    rows[symbol] = candidate
        combined = sorted(
            rows.values(),
            key=lambda row: (float(row.get("combo_score") or 0.0), float(row.get("avg_amount_20d_yuan") or 0.0)),
            reverse=True,
        )[: self.top_n_per_recipe * max(len(selected_by_recipe), 1)]
        for rank, row in enumerate(combined, start=1):
            row["rank"] = rank
            row["variant_score"] = float(row["combo_score"])
        return combined

    def _build_events(
        self,
        *,
        helper: SelectionEventStudyEngine,
        selected: list[dict],
        recipe: str,
        source_recipe: str | None,
        signal_trade_date: str,
        entry_index: int,
        cached_dates: list[str],
        price_map: dict[str, pd.DataFrame],
    ) -> list[dict]:
        events = []
        for row in selected:
            event = helper._build_event(
                row=row,
                variant=recipe,
                signal_trade_date=signal_trade_date,
                entry_index=entry_index,
                cached_dates=cached_dates,
                price_map=price_map,
            )
            if event is None:
                continue
            event["recipe"] = event.pop("variant")
            event["source_recipe"] = source_recipe or str(row.get("source_recipe") or recipe)
            event["market_state"] = _market_state(row)
            event["market_cap_tier"] = _market_cap_tier(row.get("total_mv_yuan"))
            event["style_group"] = str(row.get("style_group") or row.get("group") or "")
            for horizon in self.horizons:
                event[f"close_return_net_{horizon}d"] = event[f"close_return_{horizon}d"] - self.roundtrip_cost
            events.append(event)
        return events


def summarize_recipe_events(events_frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    groups = [(keys, frame) for keys, frame in events_frame.groupby(["recipe", "market_state"])]
    groups.extend([((recipe, "ALL"), frame) for recipe, frame in events_frame.groupby("recipe")])
    for (recipe, market_state), frame in groups:
        for horizon in horizons:
            gross_returns = frame[f"close_return_{horizon}d"]
            net_returns = frame[f"close_return_net_{horizon}d"]
            mfe = frame[f"mfe_{horizon}d"]
            mae = frame[f"mae_{horizon}d"]
            rows.append(
                {
                    "recipe": recipe,
                    "market_state": market_state,
                    "horizon": horizon,
                    "events": int(len(frame)),
                    "avg_close_return": float(gross_returns.mean()),
                    "avg_close_return_net": float(net_returns.mean()),
                    "median_close_return_net": float(net_returns.median()),
                    "win_rate_net": float((net_returns > 0).mean()),
                    "avg_mfe": float(mfe.mean()),
                    "avg_mae": float(mae.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["recipe", "market_state", "horizon"]).reset_index(drop=True)


def summarize_recipe_exposure(events_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exposure_type, column in (
        ("group", "group"),
        ("industry", "industry"),
        ("style_group", "style_group"),
        ("market_cap_tier", "market_cap_tier"),
        ("source_recipe", "source_recipe"),
    ):
        for (recipe, exposure), frame in events_frame.groupby(["recipe", column]):
            total = len(events_frame.loc[events_frame["recipe"] == recipe])
            rows.append(
                {
                    "recipe": recipe,
                    "exposure_type": exposure_type,
                    "exposure": str(exposure),
                    "events": int(len(frame)),
                    "weight": float(len(frame) / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["recipe", "exposure_type", "events"], ascending=[True, True, False])


def build_recipe_portfolio_curves(events_frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for recipe, recipe_frame in events_frame.groupby("recipe"):
        for horizon in horizons:
            return_column = f"close_return_net_{horizon}d"
            daily_returns = (
                recipe_frame.groupby("entry_trade_date")
                .agg(
                    basket_return_net=(return_column, "mean"),
                    basket_size=("symbol", "count"),
                    avg_score=("score", "mean"),
                )
                .reset_index()
                .sort_values("entry_trade_date")
            )
            equity = 1.0
            peak = 1.0
            for row in daily_returns.to_dict(orient="records"):
                basket_return = float(row["basket_return_net"])
                equity *= 1.0 + basket_return
                peak = max(peak, equity)
                rows.append(
                    {
                        "recipe": recipe,
                        "horizon": horizon,
                        "entry_trade_date": str(row["entry_trade_date"]),
                        "basket_size": int(row["basket_size"]),
                        "avg_score": float(row["avg_score"]),
                        "basket_return_net": basket_return,
                        "equity": equity,
                        "drawdown": equity / peak - 1.0 if peak > 0 else 0.0,
                    }
                )
    return pd.DataFrame(rows).sort_values(["recipe", "horizon", "entry_trade_date"]).reset_index(drop=True)


def summarize_recipe_portfolios(portfolio_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if portfolio_frame.empty:
        return pd.DataFrame(rows)
    for (recipe, horizon), frame in portfolio_frame.groupby(["recipe", "horizon"]):
        frame = frame.sort_values("entry_trade_date")
        ending_equity = float(frame["equity"].iloc[-1])
        periods = len(frame)
        rows.append(
            {
                "recipe": recipe,
                "horizon": int(horizon),
                "periods": int(periods),
                "avg_basket_size": float(frame["basket_size"].mean()),
                "total_return": ending_equity - 1.0,
                "annual_return": ending_equity ** (252 / max(periods, 1)) - 1.0,
                "max_drawdown": float(frame["drawdown"].min()),
                "win_rate": float((frame["basket_return_net"] > 0).mean()),
                "avg_basket_return_net": float(frame["basket_return_net"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["recipe", "horizon"]).reset_index(drop=True)


def _select_top_by_group(frame: pd.DataFrame, *, recipe: str, score_column: str, top_n: int) -> list[dict]:
    if frame.empty or score_column not in frame.columns:
        return []
    selected_frames = []
    for _, group_frame in frame.groupby("group", group_keys=False):
        selected_frames.append(
            group_frame.sort_values([score_column, "avg_amount_20d_yuan"], ascending=[False, False]).head(top_n)
        )
    if not selected_frames:
        return []
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values(["group", score_column, "avg_amount_20d_yuan"], ascending=[True, False, False])
    rows = []
    for group, group_frame in selected.groupby("group", sort=False):
        for rank, row in enumerate(group_frame.to_dict(orient="records"), start=1):
            row["rank"] = rank
            row["recipe"] = recipe
            row["source_recipe"] = recipe
            row["variant_score"] = float(row[score_column])
            rows.append(row)
    return rows


def _quality_momentum_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["return_30d"] >= 0.08)
        & (frame["return_30d"] <= 1.10)
        & (frame["return_90d"] >= 0.10)
        & (frame["return_90d"] <= 3.00)
        & (frame["return_5d"] >= -0.06)
        & (frame["return_5d"] <= 0.11)
        & (frame["close_to_ma_5"] >= -0.025)
        & (frame["close_to_ma_10"] >= -0.04)
        & (frame["close_to_ma_10"] <= 0.16)
        & (frame["close_to_ma_20"] >= -0.03)
        & (frame["close_to_ma_20"] <= 0.30)
        & (frame["drawdown_from_20d_high"] >= -0.16)
        & (frame["upper_shadow_pct"].fillna(0.0) <= 0.50)
        & (frame["volume_ratio"].fillna(1.0) <= 3.00)
        & (frame["amount_ratio_5d"].fillna(1.0) >= 0.75)
        & (frame["amount_ratio_5d"].fillna(1.0) <= 2.60)
    )


def _trend_pullback_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["return_30d"] >= 0.05)
        & (frame["return_30d"] <= 1.10)
        & (frame["return_90d"] >= 0.08)
        & (frame["return_90d"] <= 3.00)
        & (frame["return_5d"] >= -0.06)
        & (frame["return_5d"] <= 0.12)
        & (frame["close_to_ma_5"] >= -0.02)
        & (frame["close_to_ma_10"] >= -0.04)
        & (frame["close_to_ma_20"] >= -0.03)
        & (frame["close_to_ma_20"] <= 0.24)
        & (frame["drawdown_from_20d_high"] >= -0.18)
        & (frame["drawdown_from_20d_high"] <= -0.01)
        & (frame["trend_quality_score"].fillna(0.0) >= 0.70)
        & (frame["volume_ratio"].fillna(1.0) <= 3.00)
        & (frame["amount_ratio_5d"].fillna(1.0) >= 0.75)
        & (frame["amount_ratio_5d"].fillna(1.0) <= 2.50)
    )


def _rebound_bottoming_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["drawdown_from_20d_high"] <= -0.08)
        & (frame["drawdown_from_20d_high"] >= -0.35)
        & (frame["return_5d"] >= -0.08)
        & (frame["return_5d"] <= 0.08)
        & (frame["return_30d"] >= -0.35)
        & (frame["return_30d"] <= 0.35)
        & (frame["close_to_ma_5"] >= -0.05)
        & (frame["close_to_ma_20"] <= 0.10)
        & (frame["volume_ratio"].fillna(1.0) <= 3.00)
        & (frame["amount_ratio_5d"].fillna(1.0) >= 0.70)
        & (frame["amount_ratio_5d"].fillna(1.0) <= 2.40)
    )


def _market_state(row: dict | pd.Series) -> str:
    benchmark_return = _safe_float(row.get("benchmark_return_20d"))
    benchmark_close_to_ma20 = _safe_float(row.get("benchmark_close_to_ma20"))
    if benchmark_return is not None and benchmark_return < 0:
        return "risk_off"
    if benchmark_close_to_ma20 is not None and benchmark_close_to_ma20 < 0:
        return "risk_off"
    return "risk_on"


def _market_cap_tier(value: object) -> str:
    market_cap = _safe_float(value)
    if market_cap is None:
        return "unknown"
    if market_cap >= 100_000_000_000:
        return "large"
    if market_cap >= 30_000_000_000:
        return "mid"
    return "small"


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result
