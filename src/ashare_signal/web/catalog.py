from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

from ashare_signal.web.storage import DashboardStore, json_text


METRIC_FIELDS = (
    "total_return",
    "annual_return",
    "max_drawdown",
    "sharpe",
    "calmar",
    "turnover",
    "win_rate",
    "profit_factor",
    "average_holding_days",
    "average_invested_ratio",
    "sell_trade_count",
)


class ResultCatalog:
    def __init__(self, base_dir: Path, reports_dir: Path, store: DashboardStore) -> None:
        self.base_dir = base_dir.resolve()
        self.reports_dir = (base_dir / reports_dir).resolve()
        self.store = store
        self.last_audit: dict[str, object] = {
            "scanned": 0,
            "indexed": 0,
            "failures": 0,
            "issues": [],
        }

    def rebuild(self) -> dict[str, int]:
        scanned = 0
        indexed = 0
        failures = 0
        issues = []
        if not self.reports_dir.exists():
            self.last_audit = {
                "scanned": 0,
                "indexed": 0,
                "failures": 0,
                "issues": [],
            }
            return {"scanned": 0, "indexed": 0, "failures": 0}
        for path in self.reports_dir.rglob("*-summary.json"):
            scanned += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    failures += 1
                    issues.append(
                        {
                            "path": self._relative(path),
                            "reason": "summary 顶层不是 JSON 对象",
                        }
                    )
                    continue
                self.store.upsert_result(self._result_payload(path, payload))
                indexed += 1
            except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as error:
                failures += 1
                issues.append(
                    {
                        "path": self._relative(path),
                        "reason": str(error) or error.__class__.__name__,
                    }
                )
        audit = {"scanned": scanned, "indexed": indexed, "failures": failures}
        self.last_audit = {**audit, "issues": issues}
        return audit

    def index_summary(self, summary_path: Path, *, source: str = "task", command: str | None = None) -> dict:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Research summary must contain a JSON object.")
        record = self._result_payload(summary_path, payload, source=source, command=command)
        self.store.upsert_result(record)
        result = self.store.get_result(record["id"])
        if result is None:
            raise RuntimeError("Indexed result could not be loaded from SQLite.")
        return result

    def _result_payload(
        self,
        path: Path,
        payload: dict,
        *,
        source: str = "imported",
        command: str | None = None,
    ) -> dict:
        relative = self._relative(path)
        result_id = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
        existing = self.store.get_result(result_id)
        if existing and existing.get("source") == "task" and source == "imported":
            source = "task"
            command = existing.get("command") or command
        artifacts = self._artifacts(path, payload)
        metrics = _extract_metrics(payload)
        strategy = str(payload.get("strategy") or _kind_from_path(path))
        start_date = _date_value(payload, "start_trade_date", "start_date", "train_start_date")
        end_date = _date_value(payload, "end_trade_date", "end_date", "test_end_date")
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        title = _title(payload, path, strategy, start_date, end_date)
        parameters = _extract_parameters(payload)
        return {
            "id": result_id,
            "title": title,
            "kind": _kind_from_path(path),
            "strategy": strategy,
            "source": source,
            "status": "completed",
            "start_date": start_date,
            "end_date": end_date,
            "summary_path": relative,
            "equity_path": artifacts.get("equity"),
            "trades_path": artifacts.get("trades"),
            "metrics_json": json_text(metrics),
            "parameters_json": json_text(parameters),
            "artifacts_json": json_text(artifacts),
            "command": command,
            "protected": int(_is_protected_baseline(payload, path)),
            "archived": 0,
            "created_at": modified,
            "updated_at": modified,
        }

    def _artifacts(self, summary_path: Path, payload: dict) -> dict[str, str]:
        artifacts: dict[str, str] = {"summary": self._relative(summary_path)}
        aliases = {
            "equity_curve_path": "equity",
            "equity_path": "equity",
            "rolling_portfolio_equity_path": "equity",
            "trade_log_path": "trades",
            "trades_path": "trades",
            "events_path": "events",
            "attribution_path": "attribution",
            "markdown_path": "report",
        }
        for key, name in aliases.items():
            value = payload.get(key)
            if value:
                resolved = self._resolve_artifact(value)
                if resolved and resolved.exists():
                    artifacts[name] = self._relative(resolved)

        stem = summary_path.name.removesuffix("-summary.json")
        for suffix, name in (("-equity.csv", "equity"), ("-trades.csv", "trades")):
            candidate = summary_path.with_name(stem + suffix)
            if name not in artifacts and candidate.exists():
                artifacts[name] = self._relative(candidate)
        return artifacts

    def resolve(self, relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        path = (self.base_dir / relative_path).resolve()
        if not path.is_relative_to(self.base_dir):
            return None
        return path

    def load_frame(self, relative_path: str | None) -> pd.DataFrame:
        path = self.resolve(relative_path)
        if path is None or not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            return pd.DataFrame()

    def _resolve_artifact(self, value: object) -> Path | None:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.base_dir / path
        path = path.resolve()
        return path if path.is_relative_to(self.base_dir) else None

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.base_dir))


def _extract_metrics(payload: dict) -> dict:
    metrics = {field: _number(payload.get(field)) for field in METRIC_FIELDS}
    if metrics["average_holding_days"] is None:
        metrics["average_holding_days"] = _number(payload.get("avg_holding_days"))
    if metrics["average_invested_ratio"] is None:
        metrics["average_invested_ratio"] = _number(payload.get("avg_invested_ratio"))
    return metrics


def _extract_parameters(payload: dict) -> dict:
    preferred = (
        "selection_variant",
        "top_n",
        "max_positions",
        "groups",
        "hold_days",
        "max_hold_days",
        "exit_profile",
        "hard_exit_days",
        "exit_rules",
        "market_filter",
        "candidate_recipes",
        "signal_lag_days",
    )
    return {key: payload[key] for key in preferred if key in payload}


def _date_value(payload: dict, *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value).replace("-", "")
    return None


def _kind_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "backtests" in parts:
        return "backtest"
    if "ml-trend" in parts:
        return "lightgbm"
    if "exit-timing-study" in parts or "exit-timing-research" in parts:
        return "exit_research"
    if "risk-off-standalone" in parts or "risk-off-defensive-sleeve" in parts:
        return "risk_off_research"
    if "recipe-comparison" in parts or "recipe-study" in parts:
        return "recipe_research"
    return "research"


def _title(payload: dict, path: Path, strategy: str, start_date: str | None, end_date: str | None) -> str:
    variant = payload.get("selection_variant") or payload.get("entry_filter")
    label = str(variant or strategy).replace("_", " ")
    rules = payload.get("exit_rules") or {}
    exit_tags = []
    if rules.get("risk_off_failed_hard_exit_days") is not None:
        exit_tags.append(f"riskoff{rules['risk_off_failed_hard_exit_days']}")
    if rules.get("winner_hard_exit_bypass_peak_pct") is not None:
        exit_tags.append(f"winner>{float(rules['winner_hard_exit_bypass_peak_pct']):.0%}")
    if rules.get("high_drawdown_pct") is not None:
        exit_tags.append(f"highdd{float(rules['high_drawdown_pct']):.0%}")
    if rules.get("chandelier_atr_multiplier") is not None:
        exit_tags.append(f"chandelier{rules['chandelier_atr_multiplier']}x")
    if rules.get("trend_decay_exit"):
        exit_tags.append("trend-decay")
    if exit_tags:
        label += " · " + "/".join(exit_tags)
    elif _is_protected_baseline(payload, path):
        label += " · baseline"
    elif strategy == "full_a_momentum":
        label += " · custom"
    period = f"{start_date or '?'} - {end_date or '?'}"
    if path.parent.name == "backtests":
        return f"{label} · {period}"
    return f"{path.parent.name.replace('-', ' ')} · {period}"


def _is_protected_baseline(payload: dict, path: Path) -> bool:
    if payload.get("strategy") != "full_a_momentum":
        return False
    market_filter = payload.get("market_filter") or {}
    candidate_recipes = payload.get("candidate_recipes") or {}
    rules = payload.get("exit_rules") or {}
    research_exit_enabled = any(
        rules.get(key) not in (None, False)
        for key in (
            "high_drawdown_pct",
            "chandelier_atr_multiplier",
            "trend_decay_exit",
            "winner_hard_exit_bypass_peak_pct",
            "risk_off_failed_hard_exit_days",
        )
    )
    return bool(
        not research_exit_enabled
        and "full-a-momentum" in path.name
        and payload.get("selection_variant") == "quality_momentum"
        and _number(payload.get("top_n")) == 5
        and _number(payload.get("max_positions")) == 5
        and sorted(payload.get("groups") or []) == ["chinext", "main", "star"]
        and payload.get("exit_profile") == "legacy"
        and _number(payload.get("hard_exit_days")) == 23
        and _number(market_filter.get("market_min_breadth")) == 0.5
        and _number(market_filter.get("market_min_return_20d")) == 0
        and _number(market_filter.get("aggressive_position_size_multiplier")) == 0.5
        and sorted(market_filter.get("entry_market_states") or []) == ["aggressive", "normal"]
        and candidate_recipes.get("enabled_recipes") == ["momentum_core"]
        and not candidate_recipes.get("overlay_recipes")
        and not candidate_recipes.get("ml_predictions_path")
        and not candidate_recipes.get("theme_buy_point_overlay")
    )


def _number(value: object) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number
