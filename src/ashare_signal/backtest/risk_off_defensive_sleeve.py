from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository


DEFAULT_ALLOWED_RISK_OFF_TYPES = ("both_mild", "breadth_only", "return_only")


@dataclass(slots=True)
class RiskOffDefensiveSleeveResult:
    total_return: float
    max_drawdown: float
    sharpe: float
    combined_total_return: float
    combined_max_drawdown: float
    combined_sharpe: float
    trade_count: int
    active_days: int
    average_position_count: float
    equity_path: Path
    trades_path: Path
    summary_path: Path


class RiskOffDefensiveSleeveStudyEngine:
    """Research-only defensive sleeve funded by unused baseline cash."""

    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        *,
        events_path: Path,
        baseline_equity_path: Path,
        allowed_risk_off_types: tuple[str, ...] = DEFAULT_ALLOWED_RISK_OFF_TYPES,
        hold_days: int = 10,
        max_positions: int = 2,
        sleeve_fraction: float = 0.20,
        max_industry_weight: float = 0.50,
        exit_on_risk_on: bool = True,
        require_baseline_cash: bool = True,
        cost_pct: float = 0.0016,
        lot_size: int | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.events_path = events_path
        self.baseline_equity_path = baseline_equity_path
        self.allowed_risk_off_types = tuple(allowed_risk_off_types)
        self.hold_days = max(int(hold_days), 1)
        self.max_positions = max(int(max_positions), 1)
        self.sleeve_fraction = min(max(float(sleeve_fraction), 0.0), 1.0)
        self.max_industry_weight = min(max(float(max_industry_weight), 0.0), 1.0)
        self.exit_on_risk_on = bool(exit_on_risk_on)
        self.require_baseline_cash = bool(require_baseline_cash)
        self.cost_pct = max(float(cost_pct), 0.0)
        self.lot_size = max(int(lot_size or config.backtest.lot_size), 1)

    def run(self) -> RiskOffDefensiveSleeveResult:
        events = pd.read_csv(self.events_path)
        baseline = pd.read_csv(self.baseline_equity_path)
        _validate_inputs(events, baseline)
        events = events.loc[events["strategy"] == "defensive"].copy()
        if events.empty:
            raise ValueError("Risk-off events file contains no defensive candidates.")
        events["entry_trade_date"] = _normalize_dates(events["entry_trade_date"])
        baseline["trade_date"] = _normalize_dates(baseline["trade_date"])
        baseline = baseline.sort_values("trade_date").reset_index(drop=True)

        trade_dates = baseline["trade_date"].tolist()
        symbols = set(events["symbol"].dropna().astype(str))
        price_map = self._load_price_map(trade_dates, symbols)
        equity, trades = _simulate_defensive_sleeve(
            events=events,
            baseline=baseline,
            price_map=price_map,
            allowed_risk_off_types=self.allowed_risk_off_types,
            hold_days=self.hold_days,
            max_positions=self.max_positions,
            sleeve_fraction=self.sleeve_fraction,
            max_industry_weight=self.max_industry_weight,
            exit_on_risk_on=self.exit_on_risk_on,
            require_baseline_cash=self.require_baseline_cash,
            cost_pct=self.cost_pct,
            lot_size=self.lot_size,
        )

        sleeve_metrics = _equity_metrics(equity["equity"])
        combined_metrics = _equity_metrics(equity["combined_equity"])
        reports_dir = self.base_dir / self.config.paths.reports_dir / "risk-off-defensive-sleeve"
        reports_dir.mkdir(parents=True, exist_ok=True)
        start_date = str(baseline["trade_date"].iloc[0])
        end_date = str(baseline["trade_date"].iloc[-1])
        allowed_slug = "-".join(self.allowed_risk_off_types)
        baseline_id = hashlib.sha1(str(self.baseline_equity_path).encode("utf-8")).hexdigest()[:10]
        stem = (
            f"risk-off-defensive-sleeve-{allowed_slug}-h{self.hold_days}-"
            f"slots{self.max_positions}-w{_slug_float(self.sleeve_fraction)}-"
            f"ind{_slug_float(self.max_industry_weight)}-base{baseline_id}-{start_date}-{end_date}"
        )
        equity_path = reports_dir / f"{stem}-equity.csv"
        trades_path = reports_dir / f"{stem}-trades.csv"
        summary_path = reports_dir / f"{stem}-summary.json"
        equity.to_csv(equity_path, index=False)
        trades.to_csv(trades_path, index=False)

        payload = {
            "strategy": "risk_off_defensive_sleeve",
            "research_only": True,
            "events_path": str(self.events_path),
            "baseline_equity_path": str(self.baseline_equity_path),
            "baseline_id": baseline_id,
            "start_trade_date": start_date,
            "end_trade_date": end_date,
            "allowed_risk_off_types": list(self.allowed_risk_off_types),
            "hold_days": self.hold_days,
            "max_positions": self.max_positions,
            "sleeve_fraction": self.sleeve_fraction,
            "max_industry_weight": self.max_industry_weight,
            "exit_on_risk_on": self.exit_on_risk_on,
            "require_baseline_cash": self.require_baseline_cash,
            "cost_pct": self.cost_pct,
            "lot_size": self.lot_size,
            "trade_count": int(len(trades)),
            "active_days": int((equity["position_count"] > 0).sum()),
            "average_position_count": float(equity["position_count"].mean()),
            "sleeve_metrics": sleeve_metrics,
            "combined_metrics": combined_metrics,
            "equity_path": str(equity_path),
            "trades_path": str(trades_path),
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return RiskOffDefensiveSleeveResult(
            total_return=sleeve_metrics["total_return"],
            max_drawdown=sleeve_metrics["max_drawdown"],
            sharpe=sleeve_metrics["sharpe"],
            combined_total_return=combined_metrics["total_return"],
            combined_max_drawdown=combined_metrics["max_drawdown"],
            combined_sharpe=combined_metrics["sharpe"],
            trade_count=int(len(trades)),
            active_days=int((equity["position_count"] > 0).sum()),
            average_position_count=float(equity["position_count"].mean()),
            equity_path=equity_path,
            trades_path=trades_path,
            summary_path=summary_path,
        )

    def _load_price_map(self, trade_dates: list[str], symbols: set[str]) -> dict[str, pd.DataFrame]:
        price_map: dict[str, pd.DataFrame] = {}
        for trade_date in trade_dates:
            try:
                frame = self.repository.load_daily(trade_date)
            except FileNotFoundError:
                continue
            frame = frame.loc[frame["ts_code"].astype(str).isin(symbols), ["ts_code", "open", "close"]].copy()
            if frame.empty:
                continue
            frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            price_map[trade_date] = frame.set_index("ts_code")
        return price_map


def _simulate_defensive_sleeve(
    *,
    events: pd.DataFrame,
    baseline: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    allowed_risk_off_types: tuple[str, ...],
    hold_days: int,
    max_positions: int,
    sleeve_fraction: float,
    max_industry_weight: float,
    exit_on_risk_on: bool,
    require_baseline_cash: bool,
    cost_pct: float,
    lot_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline.copy()
    baseline["trade_date"] = _normalize_dates(baseline["trade_date"])
    dates = baseline["trade_date"].tolist()
    date_index = {trade_date: idx for idx, trade_date in enumerate(dates)}
    baseline_rows = baseline.set_index("trade_date")
    initial_equity = float(baseline["equity"].iloc[0])
    sleeve_cash = initial_equity
    per_slot_value = initial_equity * sleeve_fraction / max_positions
    max_same_industry = max(int(math.floor(max_positions * max_industry_weight)), 1)
    candidates = {
        str(trade_date): frame.sort_values(["rank", "score"], ascending=[True, False])
        for trade_date, frame in events.loc[
            events["risk_off_type"].isin(allowed_risk_off_types)
        ].groupby("entry_trade_date")
    }
    positions: list[dict] = []
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []

    for trade_date in dates:
        prices = price_map.get(trade_date)
        baseline_row = baseline_rows.loc[trade_date]
        market_state = str(baseline_row.get("market_state") or "")
        kept: list[dict] = []
        for position in positions:
            should_exit = date_index[trade_date] >= position["exit_index"]
            exit_reason = "max_holding_days"
            if exit_on_risk_on and market_state != "risk_off":
                should_exit = True
                exit_reason = "market_state_recovered"
            if should_exit and prices is not None and position["symbol"] in prices.index:
                exit_price = float(prices.loc[position["symbol"], "open"])
                proceeds = position["shares"] * exit_price * (1.0 - cost_pct / 2.0)
                sleeve_cash += proceeds
                trade_rows.append(
                    {
                        **position,
                        "exit_date": trade_date,
                        "exit_price": exit_price,
                        "exit_reason": exit_reason,
                        "pnl": proceeds - position["cost"],
                        "return": proceeds / position["cost"] - 1.0,
                    }
                )
            else:
                kept.append(position)
        positions = kept

        frame = candidates.get(trade_date)
        if frame is not None and prices is not None and market_state == "risk_off":
            current_value = _positions_value(positions, prices, cost_pct)
            held_symbols = {position["symbol"] for position in positions}
            industry_counts: dict[str, int] = {}
            for position in positions:
                industry_counts[position["style_group"]] = industry_counts.get(position["style_group"], 0) + 1
            for _, candidate in frame.iterrows():
                if len(positions) >= max_positions:
                    break
                symbol = str(candidate["symbol"])
                style_group = str(candidate.get("style_group") or "unknown")
                if symbol in held_symbols or symbol not in prices.index:
                    continue
                if industry_counts.get(style_group, 0) >= max_same_industry:
                    continue
                available = sleeve_cash
                if require_baseline_cash:
                    available = min(available, max(float(baseline_row["cash"]) - current_value, 0.0))
                budget = min(per_slot_value, available)
                if budget < 1_000.0:
                    continue
                entry_price = float(prices.loc[symbol, "open"])
                if not math.isfinite(entry_price) or entry_price <= 0:
                    continue
                buy_cost_rate = cost_pct / 2.0
                shares = int(budget / (entry_price * (1.0 + buy_cost_rate)) / lot_size) * lot_size
                if shares < lot_size:
                    continue
                gross_cost = shares * entry_price
                cost = gross_cost * (1.0 + buy_cost_rate)
                position = {
                    "entry_date": trade_date,
                    "symbol": symbol,
                    "name": str(candidate.get("name") or symbol),
                    "style_group": style_group,
                    "risk_off_type": str(candidate["risk_off_type"]),
                    "entry_price": entry_price,
                    "shares": shares,
                    "cost": cost,
                    "exit_index": min(date_index[trade_date] + hold_days, len(dates) - 1),
                }
                positions.append(position)
                sleeve_cash -= cost
                current_value += cost
                held_symbols.add(symbol)
                industry_counts[style_group] = industry_counts.get(style_group, 0) + 1

        market_value = _positions_value(positions, prices, cost_pct)
        sleeve_equity = sleeve_cash + market_value
        equity_rows.append(
            {
                "trade_date": trade_date,
                "equity": sleeve_equity,
                "cash": sleeve_cash,
                "market_value": market_value,
                "position_count": len(positions),
                "baseline_equity": float(baseline_row["equity"]),
                "baseline_cash": float(baseline_row["cash"]),
                "market_state": market_state,
                "combined_equity": float(baseline_row["equity"]) + sleeve_equity - initial_equity,
            }
        )
    return pd.DataFrame(equity_rows), pd.DataFrame(trade_rows)


def _positions_value(positions: list[dict], prices: pd.DataFrame | None, cost_pct: float) -> float:
    total = 0.0
    for position in positions:
        if prices is not None and position["symbol"] in prices.index:
            total += position["shares"] * float(prices.loc[position["symbol"], "close"]) * (1.0 - cost_pct / 2.0)
        else:
            total += float(position["cost"])
    return total


def _equity_metrics(equity: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    returns = values.pct_change().fillna(0.0)
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / max(len(values), 1)) - 1.0)
    drawdown = values / values.cummax() - 1.0
    sharpe = 0.0
    if returns.std(ddof=0) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(252.0))
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
    }


def _validate_inputs(events: pd.DataFrame, baseline: pd.DataFrame) -> None:
    event_columns = {"strategy", "risk_off_type", "entry_trade_date", "symbol", "rank", "score"}
    baseline_columns = {"trade_date", "equity", "cash", "market_state"}
    missing_events = sorted(event_columns - set(events.columns))
    missing_baseline = sorted(baseline_columns - set(baseline.columns))
    if missing_events:
        raise ValueError(f"Risk-off events file is missing columns: {', '.join(missing_events)}")
    if missing_baseline:
        raise ValueError(f"Baseline equity file is missing columns: {', '.join(missing_baseline)}")
    if baseline.empty:
        raise ValueError("Baseline equity file is empty.")


def _normalize_dates(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def _slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")
