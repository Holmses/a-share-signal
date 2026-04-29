from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json

import pandas as pd

from ashare_signal.backtest.engine import BacktestResult
from ashare_signal.data.repository import DataRepository


@dataclass(slots=True)
class PortfolioSyncResult:
    positions_path: Path
    latest_pnl_path: Path
    state_path: Path
    snapshots_dir: Path
    current_equity: float
    current_cash: float
    holdings_count: int
    last_trade_date: str


class PortfolioManager:
    """Persist a simulated portfolio from deterministic backtest outputs."""

    def __init__(self, base_dir: Path, repository: DataRepository) -> None:
        self.base_dir = base_dir
        self.repository = repository
        self.positions_root = self.base_dir / "data" / "positions"
        self.snapshots_dir = self.positions_root / "snapshots"
        self.positions_path = self.positions_root / "current_positions.csv"
        self.latest_pnl_path = self.positions_root / "latest_pnl.csv"
        self.state_path = self.positions_root / "current_state.json"
        self.trade_log_copy_path = self.positions_root / "trades.csv"

    def ensure_directories(self) -> None:
        self.positions_root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def sync_from_backtest(self, result: BacktestResult) -> PortfolioSyncResult:
        self.ensure_directories()
        trades = pd.read_csv(result.trade_log_path)
        daily_equity = pd.read_csv(result.equity_curve_path)
        trade_dates = self.repository.open_trade_dates_between(
            result.start_trade_date,
            result.end_trade_date,
        )

        trades["trade_date"] = trades["trade_date"].astype(str)
        positions: dict[str, dict] = {}
        cash = float(result.initial_cash)
        latest_snapshot = pd.DataFrame()

        for trade_date in trade_dates:
            day_trades = trades.loc[trades["trade_date"] == trade_date]
            for trade in day_trades.to_dict(orient="records"):
                if trade["action"] == "BUY":
                    existing = positions.get(trade["symbol"])
                    if existing is None:
                        positions[trade["symbol"]] = {
                            "symbol": trade["symbol"],
                            "name": trade["name"],
                            "entry_trade_date": trade["trade_date"],
                            "entry_price": float(trade["price"]),
                            "quantity": int(trade["shares"]),
                        }
                    else:
                        new_quantity = existing["quantity"] + int(trade["shares"])
                        weighted_price = (
                            existing["entry_price"] * existing["quantity"]
                            + float(trade["price"]) * int(trade["shares"])
                        ) / new_quantity
                        existing["entry_price"] = weighted_price
                        existing["quantity"] = new_quantity
                    cash -= float(trade["net_amount"])
                elif trade["action"] == "SELL":
                    positions.pop(trade["symbol"], None)
                    cash += float(trade["net_amount"])

            snapshot = self._build_snapshot_for_date(trade_date=trade_date, positions=positions)
            snapshot.to_csv(self.snapshots_dir / f"{trade_date}.csv", index=False)
            latest_snapshot = snapshot

        current_positions = pd.DataFrame(
            [
                {
                    "symbol": item["symbol"],
                    "name": item["name"],
                    "entry_date": self._to_iso_date(item["entry_trade_date"]),
                    "entry_price": round(float(item["entry_price"]), 4),
                    "quantity": int(item["quantity"]),
                }
                for item in positions.values()
            ]
        )
        if current_positions.empty:
            current_positions = pd.DataFrame(
                columns=["symbol", "name", "entry_date", "entry_price", "quantity"]
            )
        current_positions.to_csv(self.positions_path, index=False)

        if latest_snapshot.empty:
            latest_snapshot = pd.DataFrame(
                columns=[
                    "trade_date",
                    "symbol",
                    "name",
                    "entry_trade_date",
                    "entry_price",
                    "quantity",
                    "current_price",
                    "market_value",
                    "cost_basis",
                    "pnl",
                    "pnl_pct",
                    "holding_days",
                ]
            )
        latest_snapshot.to_csv(self.latest_pnl_path, index=False)
        trades.to_csv(self.trade_log_copy_path, index=False)

        state_payload = {
            "last_trade_date": result.end_trade_date,
            "start_trade_date": result.start_trade_date,
            "current_cash": cash,
            "current_equity": result.ending_equity,
            "holdings_count": int(len(current_positions)),
            "trade_count": int(result.trade_count),
            "trade_log_path": self._project_path(self.trade_log_copy_path),
            "summary_path": self._project_path(result.summary_path),
        }
        self.state_path.write_text(
            json.dumps(state_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return PortfolioSyncResult(
            positions_path=self.positions_path,
            latest_pnl_path=self.latest_pnl_path,
            state_path=self.state_path,
            snapshots_dir=self.snapshots_dir,
            current_equity=result.ending_equity,
            current_cash=cash,
            holdings_count=int(len(current_positions)),
            last_trade_date=result.end_trade_date,
        )

    def _build_snapshot_for_date(self, trade_date: str, positions: dict[str, dict]) -> pd.DataFrame:
        daily_prices = self.repository.load_daily(trade_date).copy()
        for column in ("close",):
            daily_prices[column] = pd.to_numeric(daily_prices[column], errors="coerce")
        prices = daily_prices.set_index("ts_code")

        rows: list[dict] = []
        trade_day = date.fromisoformat(self._to_iso_date(trade_date))
        for item in positions.values():
            current_price = None
            if item["symbol"] in prices.index:
                current_price = float(prices.loc[item["symbol"], "close"])
            cost_basis = float(item["entry_price"]) * int(item["quantity"])
            market_value = (current_price or float(item["entry_price"])) * int(item["quantity"])
            pnl = market_value - cost_basis
            pnl_pct = pnl / cost_basis if cost_basis else 0.0
            entry_day = date.fromisoformat(self._to_iso_date(item["entry_trade_date"]))
            rows.append(
                {
                    "trade_date": self._to_iso_date(trade_date),
                    "symbol": item["symbol"],
                    "name": item["name"],
                    "entry_trade_date": self._to_iso_date(item["entry_trade_date"]),
                    "entry_price": float(item["entry_price"]),
                    "quantity": int(item["quantity"]),
                    "current_price": current_price,
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": (trade_day - entry_day).days,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _to_iso_date(trade_date: str) -> str:
        value = str(trade_date)
        if len(value) == 8 and value.isdigit():
            return f"{value[:4]}-{value[4:6]}-{value[6:]}"
        return value

    def _project_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)
