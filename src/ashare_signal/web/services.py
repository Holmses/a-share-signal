from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.ranking import build_ranking_snapshot
from ashare_signal.web.catalog import ResultCatalog
from ashare_signal.web.storage import DashboardStore


BENCHMARKS = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
}


class DashboardDataService:
    def __init__(
        self,
        *,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        store: DashboardStore,
        catalog: ResultCatalog,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.store = store
        self.catalog = catalog

    def dashboard(self) -> dict:
        state_path = self.base_dir / "data" / "positions" / "tianzhu9_state.json"
        state = _read_json(state_path, {})
        plan_path = self._latest_plan_path()
        plan = _read_json(plan_path, {}) if plan_path else {}
        signal_date = _date_text(plan.get("signal_trade_date"))
        planned_date = _date_text(plan.get("planned_trade_date"))
        market_state = _plan_market_state(plan)
        candidates = self._top_candidates(signal_date, plan, market_state)
        data_health = self.data_health()
        positions = [_clean_mapping(value) for value in state.get("positions", [])]
        warnings = list(data_health["warnings"])
        state_date = _date_text(state.get("last_trade_date"))
        if signal_date and state_date and signal_date != state_date:
            warnings.append(
                f"模拟账户停留在 {state_date}，最新信号计划为 {signal_date}，两者尚未对齐。"
            )
        return _clean_mapping(
            {
                "data_trade_date": data_health["latest_daily_date"],
                "signal_trade_date": signal_date,
                "planned_trade_date": planned_date,
                "updated_at": _latest_iso(
                    data_health.get("daily_updated_at"),
                    data_health.get("state_updated_at"),
                    data_health.get("plan_updated_at"),
                ),
                "market_state": market_state,
                "market_state_label": _market_state_label(market_state),
                "account": {
                    "initial_cash": state.get("initial_cash"),
                    "cash": state.get("cash"),
                    "equity": state.get("equity"),
                    "positions_market_value": state.get("positions_market_value"),
                    "daily_pnl": state.get("daily_pnl"),
                    "daily_return": state.get("daily_return"),
                    "total_return": state.get("total_return"),
                    "cash_ratio": (
                        float(state.get("cash", 0)) / float(state.get("equity", 0))
                        if float(state.get("equity", 0) or 0) > 0
                        else None
                    ),
                    "position_count": len(positions),
                },
                "positions": positions,
                "buy_orders": plan.get("buy_orders", []),
                "sell_orders": plan.get("sell_orders", []),
                "hold_orders": plan.get("hold_orders", []),
                "notes": plan.get("notes", []),
                "candidates": candidates,
                "warnings": list(dict.fromkeys(warnings)),
            }
        )

    def data_health(self) -> dict:
        daily_dates = self.repository.cached_daily_trade_dates()
        latest_daily = daily_dates[-1] if daily_dates else None
        plan_path = self._latest_plan_path()
        state_path = self.base_dir / "data" / "positions" / "tianzhu9_state.json"
        plan = _read_json(plan_path, {}) if plan_path else {}
        state = _read_json(state_path, {})
        reports_dir = self.base_dir / self.config.paths.reports_dir
        index_states = []
        for code, name in BENCHMARKS.items():
            path = self.repository.tushare_root / "index_daily" / f"{code}.csv"
            coverage = _csv_date_coverage(path)
            index_states.append(
                {
                    "code": code,
                    "name": name,
                    "available": path.exists(),
                    "start_date": coverage[0],
                    "end_date": coverage[1],
                    "updated_at": _mtime_iso(path),
                }
            )
        warnings = []
        if latest_daily is None:
            warnings.append("没有可用的完整日线缓存。")
        elif _date_age_days(latest_daily) > 7:
            warnings.append(f"完整日线停留在 {latest_daily}，已超过 7 个自然日未更新。")
        if not state_path.exists():
            warnings.append("模拟盘状态文件不存在。")
        else:
            state_date = _date_text(state.get("last_trade_date"))
            if state_date and _date_age_days(state_date) > 7:
                warnings.append(f"模拟盘状态停留在 {state_date}，已超过 7 个自然日未更新。")
        if plan_path is None:
            warnings.append("没有可用的 Tianzhu9 调仓计划。")
        else:
            signal_date = _date_text(plan.get("signal_trade_date"))
            if signal_date and _date_age_days(signal_date) > 7:
                warnings.append(f"最新策略信号停留在 {signal_date}，已超过 7 个自然日未更新。")
        missing_benchmarks = [row["name"] for row in index_states if not row["available"]]
        if missing_benchmarks:
            warnings.append("基准行情尚未补齐：" + "、".join(missing_benchmarks))
        lagging_benchmarks = [
            row["name"]
            for row in index_states
            if latest_daily and row["available"] and row["end_date"] and row["end_date"] < latest_daily
        ]
        if lagging_benchmarks:
            warnings.append("基准行情落后于主行情：" + "、".join(lagging_benchmarks))
        report_index = self.catalog.last_audit
        if int(report_index.get("failures", 0)) > 0:
            warnings.append(
                f"报告索引跳过 {report_index['failures']} 个旧格式或损坏的 summary 文件，详情见报告索引状态。"
            )
        return {
            "latest_daily_date": latest_daily,
            "daily_file_count": len(daily_dates),
            "daily_updated_at": _latest_mtime_iso(self.repository.tushare_root / "daily"),
            "calendar_updated_at": _mtime_iso(self.repository.tushare_root / "trade_cal" / "SSE.csv"),
            "state_updated_at": _mtime_iso(state_path),
            "plan_updated_at": _mtime_iso(plan_path),
            "reports_updated_at": _latest_mtime_iso(reports_dir, recursive=True),
            "result_count": len(self.store.list_results(include_archived=True)),
            "report_index": report_index,
            "benchmarks": index_states,
            "warnings": warnings,
        }

    def search_stocks(self, query: str, limit: int = 20) -> list[dict]:
        path = self.repository.tushare_root / "stock_basic" / "L.csv"
        if not path.exists():
            return []
        frame = pd.read_csv(path, dtype={"ts_code": str, "symbol": str})
        query = query.strip().lower()
        if query:
            mask = pd.Series(False, index=frame.index)
            for column in ("ts_code", "symbol", "name", "industry"):
                if column in frame.columns:
                    mask |= frame[column].fillna("").astype(str).str.lower().str.contains(query, regex=False)
            frame = frame.loc[mask]
        fields = [column for column in ("ts_code", "symbol", "name", "industry", "market") if column in frame]
        return [_clean_mapping(row) for row in frame[fields].head(limit).to_dict(orient="records")]

    def stock_detail(self, symbol: str, *, result_id: str | None = None, range_name: str = "1y") -> dict:
        normalized_symbol = symbol.strip().upper()
        all_dates = self.repository.cached_daily_trade_dates()
        latest_date = all_dates[-1] if all_dates else ""
        bars = _cached_symbol_bars(str(self.repository.tushare_root), normalized_symbol, latest_date)
        if range_name == "1y" and len(bars) > 252:
            visible_bars = bars[-252:]
        else:
            visible_bars = bars
        events, audit = self._trade_events(normalized_symbol, bars, result_id=result_id)
        name, industry = self._stock_identity(normalized_symbol)
        return {
            "symbol": normalized_symbol,
            "name": name,
            "industry": industry,
            "adjustment": "none",
            "range": range_name,
            "bars": visible_bars,
            "events": events,
            "execution_audit": audit,
            "warnings": [] if bars else ["该股票在当前日线缓存中没有可用行情。"],
        }

    def results(self, *, include_archived: bool = False) -> list[dict]:
        return self.store.list_results(include_archived=include_archived)

    def result_detail(self, result_id: str) -> dict | None:
        result = self.store.get_result(result_id)
        if result is None:
            return None
        warnings = []
        for key in ("summary_path", "equity_path", "trades_path"):
            value = result.get(key)
            if value and not (self.base_dir / value).exists():
                warnings.append(f"结果引用的 {key} 文件不存在：{value}")
        result["warnings"] = warnings
        result["attribution"] = self.result_attribution(result_id)
        return result

    def compare(self, result_ids: list[str], benchmark: str) -> dict:
        selected = [self.store.get_result(value) for value in result_ids]
        selected = [value for value in selected if value]
        if not selected:
            return {"results": [], "series": [], "benchmark": None, "warnings": ["没有可比较的结果。"]}
        if not any(result["protected"] for result in selected):
            baseline = self._matching_baseline(selected[0])
            if baseline:
                selected.insert(0, baseline)
        series = []
        warnings = []
        for result in selected:
            curve = self._normalized_equity(result)
            if not curve:
                warnings.append(f"{result['title']} 缺少可读取的净值曲线。")
            series.append({"id": result["id"], "name": result["title"], "data": curve})
        benchmark_curve = self._benchmark_curve(benchmark, selected)
        if benchmark not in BENCHMARKS:
            warnings.append(f"未知基准：{benchmark}")
        elif not benchmark_curve:
            warnings.append(f"{BENCHMARKS[benchmark]} 行情尚未缓存或不覆盖所选区间。")
        return {
            "results": selected,
            "series": series,
            "benchmark": {
                "code": benchmark,
                "name": BENCHMARKS.get(benchmark, benchmark),
                "data": benchmark_curve,
            },
            "warnings": warnings,
        }

    def result_attribution(self, result_id: str) -> list[dict]:
        result = self.store.get_result(result_id)
        if result is None:
            return []
        trades = self.catalog.load_frame(result.get("trades_path"))
        paired = _paired_sells(trades)
        if paired.empty:
            return []
        rows = []
        dimensions = (
            ("year", "exit_year"),
            ("market_state", "market_state"),
            ("exit_reason", "exit_reason"),
            ("industry_style", "style_group"),
        )
        for dimension, column in dimensions:
            if column not in paired.columns:
                continue
            work = paired.copy()
            work[column] = work[column].fillna("unknown").astype(str)
            for group, frame in work.groupby(column, dropna=False):
                pnl = pd.to_numeric(frame["pnl"], errors="coerce").dropna()
                if pnl.empty:
                    continue
                rows.append(
                    {
                        "dimension": dimension,
                        "group": str(group),
                        "trades": int(len(pnl)),
                        "total_pnl": float(pnl.sum()),
                        "average_pnl": float(pnl.mean()),
                        "win_rate": float((pnl > 0).mean()),
                        "average_holding_days": _finite_mean(frame.get("holding_days")),
                    }
                )
        return _clean_mapping(rows)

    def result_trades(self, result_id: str, limit: int = 100) -> list[dict]:
        result = self.store.get_result(result_id)
        if result is None:
            return []
        trades = self.catalog.load_frame(result.get("trades_path"))
        paired = _paired_sells(trades)
        if paired.empty:
            return []
        columns = [
            value
            for value in (
                "trade_date",
                "symbol",
                "name",
                "pnl",
                "holding_days",
                "exit_reason",
                "market_state",
                "style_group",
            )
            if value in paired.columns
        ]
        work = paired.sort_values("trade_date", ascending=False).head(limit)
        return _clean_mapping(work[columns].to_dict(orient="records"))

    def _top_candidates(self, signal_date: str | None, plan: dict, market_state: str) -> list[dict]:
        if not signal_date:
            return []
        cache_path = self.base_dir / self.config.paths.processed_data_dir / "dashboard" / f"top20-{signal_date}.json"
        cached = _read_json(cache_path, [])
        if isinstance(cached, list) and len(cached) >= 20:
            return cached[:20]
        try:
            candidates = self._build_top_candidates(signal_date, plan, market_state)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
            return candidates
        except Exception:
            return _fallback_candidates(plan, market_state)

    def _build_top_candidates(self, signal_date: str, plan: dict, market_state: str) -> list[dict]:
        cached_dates = self.repository.complete_daily_cache_dates(end_date=signal_date)
        signal_index = cached_dates.index(signal_date)
        feature_dates = cached_dates[
            max(0, signal_index - SelectionEventStudyEngine.factor_history_trade_days()) : signal_index + 1
        ]
        engine = SelectionEventStudyEngine(
            config=self.config,
            repository=self.repository,
            base_dir=self.base_dir,
            top_n_per_group=20,
            min_avg_amount_yuan=50_000_000.0,
            groups=["main", "chinext", "star"],
            variants=["quality_momentum"],
            horizons=[1],
        )
        factor_frame = engine._build_factor_frame(feature_dates)
        signal_frame = factor_frame.loc[factor_frame["trade_date"].astype(str) == signal_date].copy()
        ranking = build_ranking_snapshot(signal_frame, self.config)
        ranking = ranking.loc[ranking["is_tradeable"].fillna(False)].head(20)
        buy_symbols = {str(order.get("symbol")) for order in plan.get("buy_orders", [])}
        positions = {str(value.get("symbol")) for value in _read_json(
            self.base_dir / "data" / "positions" / "tianzhu9_state.json", {}
        ).get("positions", [])}
        rows = []
        for value in ranking.to_dict(orient="records"):
            symbol = str(value["ts_code"])
            if symbol in buy_symbols:
                status = "planned"
                rejected_reason = None
            elif symbol in positions:
                status = "held"
                rejected_reason = "当前已经持有"
            elif market_state == "risk_off":
                status = "blocked"
                rejected_reason = "市场处于 risk_off，主策略禁止新开仓"
            else:
                status = "ranked_out"
                rejected_reason = "超过当日可用仓位或最终买入名额"
            rows.append(
                {
                    "symbol": symbol,
                    "name": value.get("name") or symbol,
                    "industry": value.get("industry") or "未分类",
                    "rank": int(value["rank_position"]),
                    "score": float(value["rank_score"]),
                    "signal_type": value.get("signal_type"),
                    "market_state": market_state,
                    "status": status,
                    "rejected_reason": rejected_reason,
                    "score_explain": value.get("score_explain"),
                }
            )
        return _clean_mapping(rows)

    def _trade_events(self, symbol: str, bars: list[dict], *, result_id: str | None) -> tuple[list[dict], dict]:
        bar_close = {row["trade_date"]: row["close"] for row in bars}
        calendar_dates = _repository_trade_dates(self.repository) or [value["trade_date"] for value in bars]
        if result_id:
            result = self.store.get_result(result_id)
            trades = self.catalog.load_frame(result.get("trades_path") if result else None)
        else:
            trades = _read_trade_sources(self.base_dir)
        if trades.empty or "symbol" not in trades.columns:
            trades = pd.DataFrame()
        else:
            trades = trades.loc[trades["symbol"].astype(str) == symbol].copy()
        signal_map = _paper_signal_map(self.base_dir)
        events = []
        delayed = 0
        invalid_t_plus_one = 0
        entry_signal_date: str | None = None
        for row in trades.to_dict(orient="records"):
            trade_date = _date_text(row.get("trade_date"))
            action = str(row.get("action") or "").upper()
            signal_date = _date_text(row.get("signal_trade_date"))
            if result_id and action == "SELL" and (not signal_date or signal_date == entry_signal_date):
                signal_date = _previous_trade_date(trade_date, calendar_dates)
            if not signal_date:
                signal_date = signal_map.get((trade_date, action, symbol))
            if action == "BUY" and signal_date:
                entry_signal_date = signal_date
            reason = _trade_event_reason(row, action)
            price = _safe_float(row.get("price"))
            if signal_date:
                events.append(
                    {
                        "date": signal_date,
                        "kind": "signal",
                        "action": action,
                        "price": bar_close.get(signal_date),
                        "quantity": _safe_int(row.get("shares") or row.get("quantity")),
                        "reason": reason or "策略信号",
                        "market_state": row.get("market_state"),
                        "execution_date": trade_date,
                    }
                )
                if trade_date and trade_date <= signal_date:
                    invalid_t_plus_one += 1
                elif _trade_day_gap(signal_date, trade_date, calendar_dates) > 1:
                    delayed += 1
            events.append(
                {
                    "date": trade_date,
                    "kind": "execution",
                    "action": action,
                    "price": price,
                    "quantity": _safe_int(row.get("shares") or row.get("quantity")),
                    "reason": reason or "实际成交",
                    "market_state": row.get("market_state"),
                    "pnl": _safe_float(row.get("pnl")),
                    "signal_date": signal_date,
                    "delayed": bool(signal_date and _trade_day_gap(signal_date, trade_date, calendar_dates) > 1),
                }
            )
            if action == "SELL":
                entry_signal_date = None
        if not result_id:
            plan_path = self._latest_plan_path()
            plan = _read_json(plan_path, {}) if plan_path else {}
            signal_date = _date_text(plan.get("signal_trade_date"))
            planned_date = _date_text(plan.get("planned_trade_date"))
            latest_available_date = calendar_dates[-1] if calendar_dates else None
            for action_key, action in (("buy_orders", "BUY"), ("sell_orders", "SELL")):
                for order in plan.get(action_key, []):
                    if str(order.get("symbol")) != symbol:
                        continue
                    has_execution = any(
                        event.get("kind") == "execution"
                        and event.get("action") == action
                        and (
                            event.get("signal_date") == signal_date
                            or event.get("date") == planned_date
                        )
                        for event in events
                    )
                    if has_execution:
                        continue
                    event_kind = (
                        "unfilled"
                        if planned_date and latest_available_date and planned_date <= latest_available_date
                        else "pending"
                    )
                    reason = order.get("reason") or "策略委托"
                    if event_kind == "unfilled":
                        reason = f"{reason}；计划交易日未找到实际成交记录"
                    events.append(
                        {
                            "date": signal_date,
                            "kind": event_kind,
                            "action": action,
                            "price": bar_close.get(signal_date),
                            "limit_price": order.get("limit_price"),
                            "quantity": order.get("quantity"),
                            "reason": reason,
                            "market_state": _plan_market_state(plan),
                            "execution_date": planned_date,
                        }
                    )
            market_state = _plan_market_state(plan)
            rejected = next(
                (
                    candidate
                    for candidate in self._top_candidates(signal_date, plan, market_state)
                    if candidate.get("symbol") == symbol
                    and candidate.get("status") in {"blocked", "ranked_out"}
                ),
                None,
            )
            if rejected:
                events.append(
                    {
                        "date": signal_date,
                        "kind": "rejected",
                        "action": "BUY",
                        "price": bar_close.get(signal_date),
                        "quantity": None,
                        "reason": rejected.get("rejected_reason") or "候选信号未形成委托",
                        "market_state": market_state,
                        "execution_date": _date_text(plan.get("planned_trade_date")),
                    }
                )
        return _clean_mapping(events), {
            "actual_trades": int(len(trades)),
            "signals_with_execution": sum(1 for event in events if event.get("kind") == "signal"),
            "delayed_executions": delayed,
            "invalid_t_plus_one": invalid_t_plus_one,
            "t_plus_one_valid": invalid_t_plus_one == 0,
            "unfilled_orders": sum(1 for event in events if event.get("kind") == "unfilled"),
            "pending_orders": sum(1 for event in events if event.get("kind") == "pending"),
            "rejected_signals": sum(1 for event in events if event.get("kind") == "rejected"),
        }

    def _normalized_equity(self, result: dict) -> list[dict]:
        frame = self.catalog.load_frame(result.get("equity_path"))
        if frame.empty:
            return []
        date_column = next((value for value in ("trade_date", "date") if value in frame.columns), None)
        equity_column = next(
            (value for value in ("equity", "net_equity", "combined_equity", "gross_equity") if value in frame.columns),
            None,
        )
        if date_column is None or equity_column is None:
            return []
        values = pd.to_numeric(frame[equity_column], errors="coerce")
        valid = values.notna() & values.gt(0)
        frame = frame.loc[valid, [date_column]].copy()
        values = values.loc[valid]
        if frame.empty:
            return []
        normalized = values / float(values.iloc[0])
        drawdown = normalized / normalized.cummax() - 1.0
        return [
            {"date": _date_text(date), "value": float(value), "drawdown": float(dd)}
            for date, value, dd in zip(frame[date_column], normalized, drawdown, strict=True)
        ]

    def _benchmark_curve(self, code: str, selected: list[dict]) -> list[dict]:
        if code not in BENCHMARKS:
            return []
        path = self.repository.tushare_root / "index_daily" / f"{code}.csv"
        if not path.exists():
            return []
        frame = pd.read_csv(path)
        if frame.empty or not {"trade_date", "close"}.issubset(frame.columns):
            return []
        starts = [value.get("start_date") for value in selected if value.get("start_date")]
        ends = [value.get("end_date") for value in selected if value.get("end_date")]
        dates = frame["trade_date"].fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(8)
        if starts:
            frame = frame.loc[dates >= min(starts)].copy()
            dates = dates.loc[frame.index]
        if ends:
            frame = frame.loc[dates <= max(ends)].copy()
            dates = dates.loc[frame.index]
        frame = frame.assign(_date=dates).sort_values("_date")
        close = pd.to_numeric(frame["close"], errors="coerce")
        valid = close.notna() & close.gt(0)
        frame = frame.loc[valid]
        close = close.loc[valid]
        if frame.empty:
            return []
        normalized = close / float(close.iloc[0])
        return [
            {"date": value, "value": float(equity)}
            for value, equity in zip(frame["_date"], normalized, strict=True)
        ]

    def _matching_baseline(self, result: dict) -> dict | None:
        candidates = [
            value
            for value in self.store.list_results(include_archived=False)
            if value["protected"]
            and value.get("start_date") == result.get("start_date")
            and value.get("end_date") == result.get("end_date")
        ]
        return candidates[0] if candidates else None

    def _stock_identity(self, symbol: str) -> tuple[str, str | None]:
        path = self.repository.tushare_root / "stock_basic" / "L.csv"
        if not path.exists():
            return symbol, None
        frame = pd.read_csv(path, dtype={"ts_code": str})
        rows = frame.loc[frame["ts_code"] == symbol]
        if rows.empty:
            return symbol, None
        row = rows.iloc[0]
        return str(row.get("name") or symbol), str(row.get("industry")) if pd.notna(row.get("industry")) else None

    def _latest_plan_path(self) -> Path | None:
        directory = self.base_dir / self.config.paths.reports_dir / "tianzhu9-orders"
        paths = sorted(directory.glob("tianzhu9-orders-*.json")) if directory.exists() else []
        return paths[-1] if paths else None


@lru_cache(maxsize=32)
def _cached_symbol_bars(tushare_root: str, symbol: str, latest_date: str) -> list[dict]:
    del latest_date
    daily_dir = Path(tushare_root) / "daily"
    rows = []
    for path in sorted(daily_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"})
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        match = frame.loc[frame["ts_code"].astype(str) == symbol]
        if match.empty:
            continue
        value = match.iloc[0]
        rows.append(
            {
                "trade_date": _date_text(value.get("trade_date") or path.stem),
                "open": _safe_float(value.get("open")),
                "high": _safe_float(value.get("high")),
                "low": _safe_float(value.get("low")),
                "close": _safe_float(value.get("close")),
                "volume": _safe_float(value.get("vol")),
                "amount": _safe_float(value.get("amount")),
            }
        )
    frame = pd.DataFrame(rows).sort_values("trade_date") if rows else pd.DataFrame()
    if frame.empty:
        return []
    for window in (5, 10, 20, 60):
        frame[f"ma{window}"] = pd.to_numeric(frame["close"], errors="coerce").rolling(window).mean()
    return _clean_mapping(frame.to_dict(orient="records"))


def _read_trade_sources(base_dir: Path) -> pd.DataFrame:
    frames = []
    for path in (
        base_dir / "data" / "positions" / "tianzhu9_trades.csv",
        base_dir / "data" / "positions" / "trades.csv",
    ):
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        frame["source"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _paper_signal_map(base_dir: Path) -> dict[tuple[str, str, str], str]:
    mapping = {}
    directory = base_dir / "data" / "positions" / "tianzhu9_settled_plans"
    for path in directory.glob("*.json") if directory.exists() else []:
        plan = _read_json(path, {})
        signal_date = _date_text(plan.get("signal_trade_date"))
        planned_date = _date_text(plan.get("planned_trade_date"))
        for field, action in (("buy_orders", "BUY"), ("sell_orders", "SELL")):
            for order in plan.get(field, []):
                mapping[(planned_date, action, str(order.get("symbol")))] = signal_date
    return mapping


def _paired_sells(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or not {"action", "symbol"}.issubset(trades.columns):
        return pd.DataFrame()
    queues: dict[str, deque[dict]] = defaultdict(deque)
    rows = []
    for trade in trades.to_dict(orient="records"):
        symbol = str(trade.get("symbol"))
        action = str(trade.get("action") or "").upper()
        if action == "BUY":
            queues[symbol].append(trade)
            continue
        if action != "SELL":
            continue
        buy = queues[symbol].popleft() if queues[symbol] else {}
        trade_date = _date_text(trade.get("trade_date"))
        row = dict(trade)
        row["exit_year"] = trade_date[:4]
        row["entry_market_state"] = buy.get("market_state")
        row["entry_style_group"] = buy.get("style_group")
        if _safe_float(row.get("pnl")) is None:
            buy_net = _safe_float(buy.get("net_amount"))
            sell_net = _safe_float(row.get("net_amount"))
            row["pnl"] = sell_net - buy_net if buy_net is not None and sell_net is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def _fallback_candidates(plan: dict, market_state: str) -> list[dict]:
    candidates: dict[str, dict] = {}
    for alert in plan.get("theme_alerts", []):
        for value in alert.get("top_candidates", []):
            symbol = str(value.get("symbol"))
            candidates.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "name": value.get("name") or symbol,
                    "industry": value.get("industry") or alert.get("industry") or "未分类",
                    "rank": value.get("rank_position"),
                    "score": value.get("rank_score"),
                    "signal_type": value.get("signal_type"),
                    "market_state": market_state,
                    "status": "blocked" if market_state == "risk_off" else "ranked_out",
                    "rejected_reason": (
                        "市场处于 risk_off，主策略禁止新开仓"
                        if market_state == "risk_off"
                        else "超过当日可用仓位或最终买入名额"
                    ),
                },
            )
    return sorted(candidates.values(), key=lambda value: value.get("rank") or 999)[:20]


def _plan_market_state(plan: dict) -> str:
    for alert in plan.get("theme_alerts", []):
        value = str(alert.get("market_state") or "")
        if value:
            return value
    notes = " ".join(str(value) for value in plan.get("notes", []))
    if "risk_off" in notes or "风控未通过" in notes:
        return "risk_off"
    if "弱市最强风格" in notes or "分层风控" in notes:
        return "aggressive"
    return "normal" if plan else "unknown"


def _market_state_label(value: str) -> str:
    return {
        "normal": "正常开仓",
        "aggressive": "强风格半仓",
        "defensive": "防御试仓",
        "risk_off": "风险关闭",
    }.get(value, "状态未知")


def _repository_trade_dates(repository: object) -> list[str]:
    loader = getattr(repository, "cached_daily_trade_dates", None)
    if not callable(loader):
        return []
    try:
        return [date for value in loader() if (date := _date_text(value))]
    except (OSError, TypeError, ValueError):
        return []


def _previous_trade_date(trade_date: str | None, dates: list[str]) -> str | None:
    if not trade_date:
        return None
    previous = [value for value in dates if value < trade_date]
    return previous[-1] if previous else None


def _trade_event_reason(row: dict, action: str) -> object:
    if action == "SELL":
        return row.get("exit_reason") or row.get("reason") or row.get("entry_reason")
    return row.get("entry_reason") or row.get("reason") or row.get("exit_reason")


def _trade_day_gap(start: str | None, end: str | None, dates: list[str]) -> int:
    if not start or not end:
        return 0
    between = [value for value in dates if start < value <= end]
    return len(between)


def _date_age_days(value: str | None) -> int:
    if not value:
        return 0
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return 0
    return max(0, (datetime.now().astimezone().date() - parsed).days)


def _read_json(path: Path | None, default: object) -> object:
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _csv_date_coverage(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    try:
        frame = pd.read_csv(path, usecols=["trade_date"])
    except (OSError, ValueError, pd.errors.ParserError):
        return None, None
    if frame.empty:
        return None, None
    dates = frame["trade_date"].fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    return str(dates.min()), str(dates.max())


def _latest_mtime_iso(directory: Path, *, recursive: bool = False) -> str | None:
    if not directory.exists():
        return None
    paths = directory.rglob("*") if recursive else directory.glob("*")
    mtimes = [path.stat().st_mtime for path in paths if path.is_file()]
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes)).astimezone().isoformat(timespec="seconds")


def _latest_iso(*values: str | None) -> str | None:
    present = [value for value in values if value]
    if not present:
        return None
    return max(present, key=lambda value: datetime.fromisoformat(value).timestamp())


def _mtime_iso(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _date_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).replace(".0", "").replace("-", "")
    return text.zfill(8) if text.isdigit() else text


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: object) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _finite_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _clean_mapping(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _clean_mapping(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_mapping(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_mapping(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value
