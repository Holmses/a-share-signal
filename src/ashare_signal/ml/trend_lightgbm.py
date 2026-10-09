from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import math

import pandas as pd

from ashare_signal.backtest.ranking_event_study import _market_state
from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.backtest.selection_event_study import parse_horizons
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.ranking import build_ranking_snapshot
from ashare_signal.utils.dates import to_compact_date


NUMERIC_FEATURES = [
    "rank_score",
    "rank_position",
    "rank_pct",
    "momentum_rank",
    "trend_rank",
    "pullback_rank",
    "liquidity_rank",
    "volatility_rank",
    "moneyflow_rank",
    "industry_rank",
    "rebound_rank",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_30d",
    "return_60d",
    "return_90d",
    "close_to_ma_5",
    "close_to_ma_10",
    "close_to_ma_20",
    "close_to_ma_60",
    "ma_20_to_ma_60",
    "ma_60_slope_20d",
    "drawdown_from_20d_high",
    "amount_ratio_5d",
    "avg_amount_20d_yuan",
    "turnover_rate",
    "volume_ratio",
    "volatility_20d",
    "atr_20d_pct",
    "upper_shadow_pct",
    "large_net_mf_to_amount",
    "net_mf_to_amount",
    "industry_member_count",
    "industry_return_3d_median",
    "industry_momentum_20d_median",
    "industry_breadth_20d",
    "industry_rebound_breadth",
    "market_breadth",
    "market_return_20d",
]

CATEGORICAL_FEATURES = ["industry", "universe_group", "signal_type", "market_state"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True, slots=True)
class LightGBMTrendStudyResult:
    train_start_date: str
    train_end_date: str
    backtest_start_date: str
    backtest_end_date: str
    evaluated_backtest_end_date: str
    horizon: int
    train_rows: int
    backtest_rows: int
    model_path: Path
    predictions_path: Path
    summary_csv_path: Path
    rolling_portfolio_summary_path: Path
    rolling_portfolio_equity_path: Path
    feature_importance_path: Path
    markdown_path: Path
    summary_path: Path


class LightGBMTrendStudyEngine:
    """Research-only LightGBM trend continuation study."""

    DEFAULT_TOP_KS = (5, 10, 20)

    def __init__(
        self,
        *,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        horizon: int = 10,
        train_trade_days: int = 504,
        backtest_trade_days: int = 504,
        top_ks: list[int] | None = None,
        max_rank_position: int = 300,
        min_avg_amount_yuan: float = 50_000_000.0,
        groups: list[str] | None = None,
        positive_return_threshold: float = 0.05,
        positive_max_drawdown: float = -0.08,
        negative_return_threshold: float = 0.0,
        negative_max_drawdown: float = -0.10,
        allow_short_train: bool = False,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.horizon = max(int(horizon), 1)
        self.train_trade_days = max(int(train_trade_days), 1)
        self.backtest_trade_days = max(int(backtest_trade_days), 1)
        self.top_ks = sorted({int(value) for value in (top_ks or list(self.DEFAULT_TOP_KS)) if int(value) > 0})
        self.max_rank_position = max(int(max_rank_position), 1)
        self.min_avg_amount_yuan = float(min_avg_amount_yuan)
        self.groups = groups or ["main", "chinext", "star"]
        self.positive_return_threshold = float(positive_return_threshold)
        self.positive_max_drawdown = float(positive_max_drawdown)
        self.negative_return_threshold = float(negative_return_threshold)
        self.negative_max_drawdown = float(negative_max_drawdown)
        self.allow_short_train = bool(allow_short_train)

    def run(
        self,
        *,
        backtest_start_date: date | None = None,
        backtest_end_date: date | None = None,
    ) -> LightGBMTrendStudyResult:
        lgb = _import_lightgbm()
        cached_dates = self.repository.complete_daily_cache_dates()
        if not cached_dates:
            raise ValueError("Daily Tushare cache is empty. Run `ashare-signal sync-tushare` first.")

        windows = self._resolve_windows(cached_dates, backtest_start_date, backtest_end_date)
        feature_dates = cached_dates[
            max(0, windows["train_signal_start_index"] - SelectionEventStudyEngine.factor_history_trade_days()) :
            windows["backtest_signal_end_index"] + 1
        ]
        study_engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=max(self.top_ks),
            min_avg_amount_yuan=self.min_avg_amount_yuan,
            groups=self.groups,
            variants=["quality_momentum"],
            horizons=[1],
        )
        factor_frame = study_engine._build_factor_frame(feature_dates)
        price_dates = cached_dates[
            windows["train_entry_start_index"] : windows["backtest_entry_end_index"] + self.horizon
        ]
        price_map = study_engine._load_price_map(price_dates)

        train_frame = self._build_dataset(
            cached_dates=cached_dates,
            signal_start_index=windows["train_signal_start_index"],
            signal_end_index=windows["train_signal_end_index"],
            factor_frame=factor_frame,
            price_map=price_map,
        )
        backtest_frame = self._build_dataset(
            cached_dates=cached_dates,
            signal_start_index=windows["backtest_signal_start_index"],
            signal_end_index=windows["backtest_signal_end_index"],
            factor_frame=factor_frame,
            price_map=price_map,
        )
        train_labeled = train_frame.loc[train_frame["label"].notna()].copy()
        if train_labeled.empty:
            raise ValueError("LightGBM trend study produced no labeled training rows.")
        if train_labeled["label"].nunique() < 2:
            raise ValueError("LightGBM trend study needs both positive and negative labels in the training window.")
        if backtest_frame.empty:
            raise ValueError("LightGBM trend study produced no backtest rows.")

        X_train = _model_matrix(train_labeled)
        y_train = train_labeled["label"].astype(int)
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=250,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

        predictions = backtest_frame.copy()
        predictions["trend_prob"] = model.predict_proba(_model_matrix(predictions))[:, 1]
        predictions["combined_score"] = predictions["rank_score"].fillna(0.0) * 0.70 + predictions["trend_prob"] * 0.30
        summary_frame = summarize_predictions(predictions, top_ks=self.top_ks, horizon=self.horizon)
        metrics = _classification_metrics(predictions)
        if metrics:
            metrics_frame = pd.DataFrame([{"metric_type": "classification", **metrics}])
            summary_frame = pd.concat([summary_frame, metrics_frame], ignore_index=True)

        reports_dir = self.base_dir / self.config.paths.reports_dir / "ml-trend"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"lightgbm-trend-h{self.horizon}-train{windows['train_start_date']}-{windows['train_end_date']}"
            f"-bt{windows['backtest_start_date']}-{windows['evaluated_backtest_end_date']}"
        )
        model_path = reports_dir / f"{stem}.txt"
        predictions_path = reports_dir / f"{stem}-predictions.csv"
        summary_csv_path = reports_dir / f"{stem}-summary.csv"
        rolling_portfolio_summary_path = reports_dir / f"{stem}-rolling-portfolio-summary.csv"
        rolling_portfolio_equity_path = reports_dir / f"{stem}-rolling-portfolio-equity.csv"
        feature_importance_path = reports_dir / f"{stem}-feature-importance.csv"
        markdown_path = reports_dir / f"{stem}.md"
        summary_path = reports_dir / f"{stem}-summary.json"

        portfolio_price_dates = cached_dates[
            windows["backtest_entry_start_index"] : windows["backtest_entry_end_index"] + self.horizon
        ]
        portfolio_prices = self.repository.load_daily_for_dates(portfolio_price_dates)
        rolling_summary, rolling_equity = simulate_rolling_portfolios(
            predictions,
            portfolio_prices,
            top_ks=self.top_ks,
            horizon=self.horizon,
            commission_rate=float(self.config.backtest.commission_rate),
            stamp_duty_rate=float(self.config.backtest.stamp_duty_rate),
        )

        model.booster_.save_model(str(model_path))
        predictions.to_csv(predictions_path, index=False)
        summary_frame.to_csv(summary_csv_path, index=False)
        rolling_summary.to_csv(rolling_portfolio_summary_path, index=False)
        rolling_equity.to_csv(rolling_portfolio_equity_path, index=False)
        feature_importance = pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        feature_importance.to_csv(feature_importance_path, index=False)
        payload = {
            "strategy": "lightgbm_trend_continuation",
            "horizon": self.horizon,
            "label": {
                "positive_return_threshold": self.positive_return_threshold,
                "positive_max_drawdown": self.positive_max_drawdown,
                "negative_return_threshold": self.negative_return_threshold,
                "negative_max_drawdown": self.negative_max_drawdown,
            },
            "train_start_date": windows["train_start_date"],
            "train_end_date": windows["train_end_date"],
            "backtest_start_date": windows["backtest_start_date"],
            "requested_backtest_end_date": windows["backtest_end_date"],
            "evaluated_backtest_end_date": windows["evaluated_backtest_end_date"],
            "train_rows": int(len(train_labeled)),
            "train_positive_rate": float(train_labeled["label"].mean()),
            "backtest_rows": int(len(predictions)),
            "top_ks": self.top_ks,
            "max_rank_position": self.max_rank_position,
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
            "summary_csv_path": str(summary_csv_path),
            "rolling_portfolio_summary_path": str(rolling_portfolio_summary_path),
            "rolling_portfolio_equity_path": str(rolling_portfolio_equity_path),
            "feature_importance_path": str(feature_importance_path),
            "summary": summary_frame.to_dict(orient="records"),
            "rolling_portfolio_summary": rolling_summary.to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(
            _render_markdown(payload, summary_frame, rolling_summary, feature_importance),
            encoding="utf-8",
        )

        return LightGBMTrendStudyResult(
            train_start_date=windows["train_start_date"],
            train_end_date=windows["train_end_date"],
            backtest_start_date=windows["backtest_start_date"],
            backtest_end_date=windows["backtest_end_date"],
            evaluated_backtest_end_date=windows["evaluated_backtest_end_date"],
            horizon=self.horizon,
            train_rows=int(len(train_labeled)),
            backtest_rows=int(len(predictions)),
            model_path=model_path,
            predictions_path=predictions_path,
            summary_csv_path=summary_csv_path,
            rolling_portfolio_summary_path=rolling_portfolio_summary_path,
            rolling_portfolio_equity_path=rolling_portfolio_equity_path,
            feature_importance_path=feature_importance_path,
            markdown_path=markdown_path,
            summary_path=summary_path,
        )

    def _resolve_windows(
        self,
        cached_dates: list[str],
        backtest_start_date: date | None,
        backtest_end_date: date | None,
    ) -> dict:
        requested_end = to_compact_date(backtest_end_date) if backtest_end_date else cached_dates[-1]
        eligible_ends = [value for value in cached_dates if value <= requested_end]
        if not eligible_ends:
            raise ValueError(f"No cached trade date found on or before {requested_end}")
        backtest_end = eligible_ends[-1]
        backtest_end_index = cached_dates.index(backtest_end)
        backtest_signal_end_index = backtest_end_index - self.horizon
        if backtest_signal_end_index < 1:
            raise ValueError(f"Need at least {self.horizon} trade days after each signal date.")

        if backtest_start_date is None:
            backtest_entry_start_index = max(1, backtest_end_index - self.backtest_trade_days + 1)
        else:
            requested_start = to_compact_date(backtest_start_date)
            eligible_starts = [idx for idx, value in enumerate(cached_dates) if value >= requested_start]
            if not eligible_starts:
                raise ValueError(f"No cached trade date found on or after {requested_start}")
            backtest_entry_start_index = max(1, eligible_starts[0])
        backtest_signal_start_index = backtest_entry_start_index - 1

        train_entry_end_index = backtest_entry_start_index - 1
        train_entry_start_index = train_entry_end_index - self.train_trade_days + 1
        if train_entry_start_index < 1:
            if not self.allow_short_train:
                available_start = cached_dates[1] if len(cached_dates) > 1 else cached_dates[0]
                raise ValueError(
                    "LightGBM trend study needs two full training years before the backtest window. "
                    f"Requested train_trade_days={self.train_trade_days}, but local cache only covers "
                    f"{available_start} to {cached_dates[train_entry_end_index]} before backtest start "
                    f"{cached_dates[backtest_entry_start_index]}. "
                    f"Sync older data from about {self._suggest_sync_start(cached_dates, backtest_entry_start_index)} "
                    "or rerun with --allow-short-train for a local smoke study."
                )
            train_entry_start_index = 1
        train_signal_start_index = train_entry_start_index - 1
        train_signal_end_index = train_entry_end_index - 1
        if train_signal_end_index < train_signal_start_index:
            raise ValueError("Training window is empty.")

        return {
            "train_entry_start_index": train_entry_start_index,
            "train_entry_end_index": train_entry_end_index,
            "train_signal_start_index": train_signal_start_index,
            "train_signal_end_index": train_signal_end_index,
            "backtest_entry_start_index": backtest_entry_start_index,
            "backtest_entry_end_index": backtest_signal_end_index + 1,
            "backtest_signal_start_index": backtest_signal_start_index,
            "backtest_signal_end_index": backtest_signal_end_index,
            "train_start_date": cached_dates[train_entry_start_index],
            "train_end_date": cached_dates[train_entry_end_index],
            "backtest_start_date": cached_dates[backtest_entry_start_index],
            "backtest_end_date": backtest_end,
            "evaluated_backtest_end_date": cached_dates[backtest_signal_end_index + 1],
        }

    def _suggest_sync_start(self, cached_dates: list[str], backtest_entry_start_index: int) -> str:
        missing_days = self.train_trade_days - max(backtest_entry_start_index - 1, 0)
        first_cached = pd.Timestamp(cached_dates[0])
        suggested = first_cached - pd.tseries.offsets.BDay(missing_days + 80)
        return suggested.strftime("%Y%m%d")

    def _build_dataset(
        self,
        *,
        cached_dates: list[str],
        signal_start_index: int,
        signal_end_index: int,
        factor_frame: pd.DataFrame,
        price_map: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        frames = []
        for signal_index in range(signal_start_index, signal_end_index + 1):
            signal_date = cached_dates[signal_index]
            entry_index = signal_index + 1
            entry_date = cached_dates[entry_index]
            day_factors = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_date].copy()
            if day_factors.empty:
                continue
            ranking = build_ranking_snapshot(day_factors, self.config, variant="quality_momentum_rank")
            ranking = ranking.loc[ranking["is_tradeable"].fillna(False).astype(bool)].copy()
            if ranking.empty:
                continue
            ranking["rank_position"] = pd.to_numeric(ranking["rank_position"], errors="coerce")
            ranking = ranking.dropna(subset=["rank_position"])
            ranking = ranking.loc[ranking["rank_position"] <= self.max_rank_position].copy()
            if ranking.empty:
                continue
            ranking["rank_pct"] = ranking["rank_position"] / float(len(ranking))
            market = _market_state(day_factors, market_min_breadth=0.50, market_min_return_20d=0.0)
            features = _signal_feature_frame(day_factors)
            events = ranking.merge(features, on=["ts_code", "trade_date"], how="left")
            events = events.rename(columns={"trade_date": "signal_trade_date"})
            events["entry_trade_date"] = entry_date
            events["market_state"] = market["market_state"]
            events["market_breadth"] = market["market_breadth"]
            events["market_return_20d"] = market["market_return_20d"]

            entry_prices = price_map.get(entry_date)
            if entry_prices is None or entry_prices.empty:
                continue
            events["entry_price"] = events["ts_code"].map(entry_prices["open"])
            events["entry_price"] = pd.to_numeric(events["entry_price"], errors="coerce")
            events = events.loc[events["entry_price"] > 0].copy()
            if events.empty:
                continue
            self._merge_forward_label(events, cached_dates, entry_index, price_map)
            frames.append(events)
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
        for column in NUMERIC_FEATURES:
            if column not in frame.columns:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in CATEGORICAL_FEATURES:
            if column not in frame.columns:
                frame[column] = "unknown"
            frame[column] = frame[column].fillna("unknown").astype(str)
        return frame

    def _merge_forward_label(
        self,
        events: pd.DataFrame,
        cached_dates: list[str],
        entry_index: int,
        price_map: dict[str, pd.DataFrame],
    ) -> None:
        horizon_dates = cached_dates[entry_index : entry_index + self.horizon]
        if len(horizon_dates) != self.horizon:
            events["future_return"] = pd.NA
            events["future_mfe"] = pd.NA
            events["future_mae"] = pd.NA
            events["label"] = pd.NA
            return
        close_prices = price_map[horizon_dates[-1]]["close"]
        high_frame = pd.concat([price_map[trade_date]["high"].rename(trade_date) for trade_date in horizon_dates], axis=1)
        low_frame = pd.concat([price_map[trade_date]["low"].rename(trade_date) for trade_date in horizon_dates], axis=1)
        exit_close = events["ts_code"].map(close_prices)
        max_high = events["ts_code"].map(high_frame.max(axis=1))
        min_low = events["ts_code"].map(low_frame.min(axis=1))
        events["future_return"] = pd.to_numeric(exit_close, errors="coerce") / events["entry_price"] - 1.0
        events["future_mfe"] = pd.to_numeric(max_high, errors="coerce") / events["entry_price"] - 1.0
        events["future_mae"] = pd.to_numeric(min_low, errors="coerce") / events["entry_price"] - 1.0
        positive = (events["future_return"] >= self.positive_return_threshold) & (
            events["future_mae"] >= self.positive_max_drawdown
        )
        negative = (events["future_return"] < self.negative_return_threshold) | (
            events["future_mae"] <= self.negative_max_drawdown
        )
        events["label"] = pd.NA
        events.loc[positive, "label"] = 1
        events.loc[negative, "label"] = 0


def summarize_predictions(predictions: pd.DataFrame, *, top_ks: list[int], horizon: int) -> pd.DataFrame:
    rows = []
    for segment, frame in _segments(predictions):
        for method, score_column in (
            ("baseline_rank_score", "rank_score"),
            ("lightgbm_trend_prob", "trend_prob"),
            ("combined_70_rule_30_ml", "combined_score"),
        ):
            for top_k in top_ks:
                selected = (
                    frame.sort_values(["signal_trade_date", score_column], ascending=[True, False])
                    .groupby("signal_trade_date", group_keys=False)
                    .head(top_k)
                )
                returns = pd.to_numeric(selected["future_return"], errors="coerce").dropna()
                if returns.empty:
                    continue
                daily_returns = (
                    selected.assign(_return=pd.to_numeric(selected["future_return"], errors="coerce"))
                    .groupby("signal_trade_date")["_return"]
                    .mean()
                    .dropna()
                )
                rows.append(
                    {
                        "metric_type": "topk_forward_return",
                        "segment": segment,
                        "method": method,
                        "top_k": int(top_k),
                        "horizon": int(horizon),
                        "events": int(len(returns)),
                        "signal_days": int(selected["signal_trade_date"].nunique()),
                        "avg_return": float(returns.mean()),
                        "median_return": float(returns.median()),
                        "win_rate": float((returns > 0).mean()),
                        "avg_daily_portfolio_return": float(daily_returns.mean()) if not daily_returns.empty else None,
                        "median_daily_portfolio_return": float(daily_returns.median()) if not daily_returns.empty else None,
                    }
                )
    return pd.DataFrame(rows)


def simulate_rolling_portfolios(
    predictions: pd.DataFrame,
    price_frame: pd.DataFrame,
    *,
    top_ks: list[int],
    horizon: int,
    commission_rate: float = 0.0003,
    stamp_duty_rate: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate fixed-horizon rolling TopK portfolios from model predictions."""

    if predictions.empty or price_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    pred = predictions.copy()
    for column in ("signal_trade_date", "entry_trade_date", "ts_code", "market_state"):
        if column in pred.columns:
            pred[column] = pred[column].fillna("").astype(str)
    prices = price_frame.copy()
    prices["trade_date"] = prices["trade_date"].fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    prices["ts_code"] = prices["ts_code"].fillna("").astype(str)
    prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["trade_date", "ts_code", "open", "close"])
    prices = prices.loc[(prices["open"] > 0) & (prices["close"] > 0)].copy()
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    trade_dates = sorted(prices["trade_date"].unique().tolist())
    date_index = {trade_date: idx for idx, trade_date in enumerate(trade_dates)}
    price_lookup = {
        (row.trade_date, row.ts_code): (float(row.open), float(row.close))
        for row in prices[["trade_date", "ts_code", "open", "close"]].itertuples(index=False)
    }

    summary_rows = []
    equity_rows = []
    for entry_filter, filtered in _portfolio_entry_filters(pred):
        if filtered.empty:
            continue
        for method, score_column in (
            ("baseline_rank_score", "rank_score"),
            ("lightgbm_trend_prob", "trend_prob"),
            ("combined_70_rule_30_ml", "combined_score"),
        ):
            if score_column not in filtered.columns:
                continue
            scored = filtered.copy()
            scored[score_column] = pd.to_numeric(scored[score_column], errors="coerce")
            scored = scored.dropna(subset=[score_column])
            if scored.empty:
                continue
            for top_k in top_ks:
                selected = (
                    scored.sort_values(["signal_trade_date", score_column], ascending=[True, False])
                    .groupby("signal_trade_date", group_keys=False)
                    .head(int(top_k))
                )
                if selected.empty:
                    continue
                portfolio = _simulate_selected_portfolio(
                    selected,
                    trade_dates=trade_dates,
                    date_index=date_index,
                    price_lookup=price_lookup,
                    horizon=int(horizon),
                    commission_rate=float(commission_rate),
                    stamp_duty_rate=float(stamp_duty_rate),
                )
                if portfolio["trade_count"] == 0:
                    continue
                summary_rows.append(
                    {
                        "metric_type": "rolling_portfolio",
                        "entry_filter": entry_filter,
                        "method": method,
                        "top_k": int(top_k),
                        "horizon": int(horizon),
                        **portfolio["summary"],
                    }
                )
                for equity_row in portfolio["equity_rows"]:
                    equity_rows.append(
                        {
                            "entry_filter": entry_filter,
                            "method": method,
                            "top_k": int(top_k),
                            **equity_row,
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(equity_rows)


def _portfolio_entry_filters(predictions: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    filters = [("all", predictions)]
    if "market_state" in predictions.columns:
        filters.append(("risk_on_only", predictions.loc[predictions["market_state"] == "risk_on"]))
        filters.append(("risk_off_only", predictions.loc[predictions["market_state"] == "risk_off"]))
    return filters


def _simulate_selected_portfolio(
    selected: pd.DataFrame,
    *,
    trade_dates: list[str],
    date_index: dict[str, int],
    price_lookup: dict[tuple[str, str], tuple[float, float]],
    horizon: int,
    commission_rate: float,
    stamp_duty_rate: float,
) -> dict:
    daily_gross: dict[str, list[float]] = {trade_date: [] for trade_date in trade_dates}
    daily_net: dict[str, list[float]] = {trade_date: [] for trade_date in trade_dates}
    trade_net_returns = []
    trade_gross_returns = []
    valid_signal_dates = set()

    for row in selected.itertuples(index=False):
        ts_code = str(getattr(row, "ts_code"))
        signal_trade_date = str(getattr(row, "signal_trade_date"))
        entry_trade_date = str(getattr(row, "entry_trade_date"))
        entry_index = date_index.get(entry_trade_date)
        if entry_index is None:
            continue
        hold_dates = trade_dates[entry_index : entry_index + horizon]
        if len(hold_dates) != horizon:
            continue
        daily_values: list[tuple[str, float, float]] = []
        valid = True
        entry_open = 0.0
        exit_close = 0.0
        previous_close = None
        for offset, trade_date in enumerate(hold_dates):
            price = price_lookup.get((trade_date, ts_code))
            if price is None:
                valid = False
                break
            open_price, close_price = price
            if offset == 0:
                entry_open = open_price
                gross_return = close_price / entry_open - 1.0
            else:
                if previous_close is None or previous_close <= 0:
                    valid = False
                    break
                gross_return = close_price / previous_close - 1.0
            net_return = gross_return
            if offset == 0:
                net_return -= commission_rate
            if offset == horizon - 1:
                net_return -= commission_rate + stamp_duty_rate
                exit_close = close_price
            daily_values.append((trade_date, gross_return, net_return))
            previous_close = close_price
        if not valid or entry_open <= 0 or exit_close <= 0:
            continue
        valid_signal_dates.add(signal_trade_date)
        trade_gross_returns.append(exit_close / entry_open - 1.0)
        trade_net_returns.append(exit_close / entry_open - 1.0 - commission_rate * 2.0 - stamp_duty_rate)
        for trade_date, gross_return, net_return in daily_values:
            daily_gross[trade_date].append(gross_return)
            daily_net[trade_date].append(net_return)

    if not trade_net_returns:
        return {"trade_count": 0, "summary": {}, "equity_rows": []}

    gross_equity = 1.0
    net_equity = 1.0
    gross_daily_returns = []
    net_daily_returns = []
    active_counts = []
    equity_rows = []
    for trade_date in trade_dates:
        active = len(daily_net[trade_date])
        gross_return = float(pd.Series(daily_gross[trade_date]).mean()) if active else 0.0
        net_return = float(pd.Series(daily_net[trade_date]).mean()) if active else 0.0
        gross_equity *= 1.0 + gross_return
        net_equity *= 1.0 + net_return
        gross_daily_returns.append(gross_return)
        net_daily_returns.append(net_return)
        active_counts.append(active)
        equity_rows.append(
            {
                "trade_date": trade_date,
                "gross_daily_return": gross_return,
                "net_daily_return": net_return,
                "gross_equity": gross_equity,
                "net_equity": net_equity,
                "active_positions": int(active),
            }
        )

    summary = {
        "trade_count": int(len(trade_net_returns)),
        "signal_days": int(len(valid_signal_dates)),
        "gross_total_return": float(gross_equity - 1.0),
        "net_total_return": float(net_equity - 1.0),
        "gross_annual_return": _annual_return(gross_equity, len(gross_daily_returns)),
        "net_annual_return": _annual_return(net_equity, len(net_daily_returns)),
        "gross_max_drawdown": _max_drawdown([row["gross_equity"] for row in equity_rows]),
        "net_max_drawdown": _max_drawdown([row["net_equity"] for row in equity_rows]),
        "gross_sharpe": _sharpe(gross_daily_returns),
        "net_sharpe": _sharpe(net_daily_returns),
        "trade_avg_gross_return": float(pd.Series(trade_gross_returns).mean()),
        "trade_avg_net_return": float(pd.Series(trade_net_returns).mean()),
        "trade_win_rate": float((pd.Series(trade_net_returns) > 0).mean()),
        "avg_active_positions": float(pd.Series(active_counts).mean()),
        "median_active_positions": float(pd.Series(active_counts).median()),
        "max_active_positions": int(max(active_counts)),
        "cash_days": int(sum(1 for value in active_counts if value == 0)),
    }
    return {"trade_count": len(trade_net_returns), "summary": summary, "equity_rows": equity_rows}


def _annual_return(final_equity: float, day_count: int) -> float | None:
    if day_count <= 0 or final_equity <= 0:
        return None
    return float(final_equity ** (252.0 / day_count) - 1.0)


def _max_drawdown(equity_values: list[float]) -> float | None:
    if not equity_values:
        return None
    peak = equity_values[0]
    max_drawdown = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = min(max_drawdown, value / peak - 1.0)
    return float(max_drawdown)


def _sharpe(daily_returns: list[float]) -> float | None:
    series = pd.Series(daily_returns, dtype="float64")
    if series.empty:
        return None
    std = float(series.std(ddof=1))
    if std <= 0 or not math.isfinite(std):
        return None
    return float(series.mean() / std * math.sqrt(252.0))


def _signal_feature_frame(day_factors: pd.DataFrame) -> pd.DataFrame:
    frame = day_factors.copy()
    if "drawdown_from_20d_high" not in frame.columns and "pullback_from_20d_high" in frame.columns:
        frame["drawdown_from_20d_high"] = frame["pullback_from_20d_high"]
    columns = ["trade_date", "ts_code"] + [
        column
        for column in set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        if column in frame.columns and column not in {"trade_date", "ts_code", "rank_score", "rank_position"}
    ]
    return frame[columns].drop_duplicates(subset=["trade_date", "ts_code"])


def _model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame.copy()
    for column in MODEL_FEATURES:
        if column not in matrix.columns:
            matrix[column] = pd.NA
    matrix = matrix[MODEL_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].fillna("unknown").astype("category")
    return matrix


def _segments(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    segments = [("ALL", frame)]
    if "market_state" in frame.columns:
        segments.extend((f"market_state={name}", part) for name, part in frame.groupby("market_state"))
    return segments


def _classification_metrics(predictions: pd.DataFrame) -> dict:
    labeled = predictions.loc[predictions["label"].notna()].copy()
    if labeled.empty or labeled["label"].nunique() < 2:
        return {}
    try:
        from sklearn.metrics import log_loss, roc_auc_score
    except ModuleNotFoundError:
        return {}
    y_true = labeled["label"].astype(int)
    y_score = pd.to_numeric(labeled["trend_prob"], errors="coerce").clip(1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score)),
        "labeled_events": int(len(labeled)),
        "positive_rate": float(y_true.mean()),
    }


def _render_markdown(
    payload: dict,
    summary_frame: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> str:
    lines = [
        "# LightGBM Trend Continuation Study",
        "",
        "Research-only. The model output is not connected to production order generation.",
        "",
        "## Windows",
        "",
        f"- Train: {payload['train_start_date']} to {payload['train_end_date']}",
        f"- Backtest: {payload['backtest_start_date']} to {payload['evaluated_backtest_end_date']}",
        f"- Horizon: {payload['horizon']} trade days",
        f"- Train rows: {payload['train_rows']}",
        f"- Backtest rows: {payload['backtest_rows']}",
        "",
        "## TopK Forward Returns",
        "",
    ]
    lines.extend(
        _markdown_table(
            summary_frame.loc[summary_frame["metric_type"] == "topk_forward_return"],
            [
                "segment",
                "method",
                "top_k",
                "events",
                "avg_return",
                "median_return",
                "win_rate",
                "avg_daily_portfolio_return",
            ],
        )
    )
    lines.extend(["", "## Classification Metrics", ""])
    lines.extend(
        _markdown_table(
            summary_frame.loc[summary_frame["metric_type"] == "classification"],
            ["auc", "log_loss", "labeled_events", "positive_rate"],
        )
    )
    lines.extend(["", "## Rolling Portfolio", ""])
    lines.extend(
        _markdown_table(
            rolling_summary,
            [
                "entry_filter",
                "method",
                "top_k",
                "trade_count",
                "net_total_return",
                "net_annual_return",
                "net_max_drawdown",
                "net_sharpe",
                "trade_win_rate",
                "avg_active_positions",
            ],
        )
    )
    lines.extend(["", "## Feature Importance", ""])
    lines.extend(_markdown_table(feature_importance, ["feature", "importance"], limit=30))
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Model: `{payload['model_path']}`",
            f"- Predictions: `{payload['predictions_path']}`",
            f"- Summary CSV: `{payload['summary_csv_path']}`",
            f"- Rolling portfolio summary: `{payload['rolling_portfolio_summary_path']}`",
            f"- Rolling portfolio equity: `{payload['rolling_portfolio_equity_path']}`",
            f"- Feature importance: `{payload['feature_importance_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 80) -> list[str]:
    if frame.empty:
        return ["No rows."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.head(limit).iterrows():
        lines.append("| " + " | ".join(_format_markdown_value(row.get(column)) for column in columns) + " |")
    return lines


def _format_markdown_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return ""
    return str(value).replace("|", "/")


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "LightGBM is required for this study. Install it with `.venv/bin/pip install lightgbm scikit-learn`."
        ) from error
    except OSError as error:
        raise RuntimeError(
            "LightGBM is installed but its native runtime dependency is missing. "
            "On macOS install OpenMP with `brew install libomp` and rerun."
        ) from error
    return lgb


def parse_top_ks(value: str | None) -> list[int]:
    return parse_horizons(value, list(LightGBMTrendStudyEngine.DEFAULT_TOP_KS))
