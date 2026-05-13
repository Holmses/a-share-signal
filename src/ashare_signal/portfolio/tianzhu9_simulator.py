from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import math
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository


@dataclass(slots=True)
class Tianzhu9PositionSnapshot:
    symbol: str
    name: str
    entry_date: str
    entry_price: float
    quantity: int
    last_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_return: float
    holding_days: int


@dataclass(slots=True)
class Tianzhu9SimulationResult:
    positions_path: Path
    state_path: Path
    trades_path: Path
    pending_plan_path: Path
    initial_cash: float
    cash: float
    equity: float
    previous_equity: float
    positions_market_value: float
    daily_pnl: float
    daily_return: float
    total_return: float
    positions_count: int
    executed_trades: int
    pending_plan_date: str | None
    last_trade_date: str
    updated_at: str
    positions: list[Tianzhu9PositionSnapshot]


class Tianzhu9PaperBroker:
    def __init__(
        self,
        config: AppConfig,
        repository: DataRepository,
        base_dir: Path,
        hold_days: int = 2,
        positions_path: Path | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.base_dir = base_dir
        self.hold_days = max(int(hold_days), 1)
        self.positions_path = positions_path or base_dir / "data" / "positions" / "tianzhu9_positions.csv"
        self.positions_dir = self.positions_path.parent
        self.state_path = self.positions_dir / "tianzhu9_state.json"
        self.trades_path = self.positions_dir / "tianzhu9_trades.csv"
        self.pending_plan_path = self.positions_dir / "tianzhu9_pending_plan.json"

    def settle_pending_plan(self, as_of_trade_date: str) -> Tianzhu9SimulationResult:
        self.positions_dir.mkdir(parents=True, exist_ok=True)
        positions = self._load_positions()
        state = self._load_state()
        trades = []

        pending_plans = self._pending_plans_to_settle(as_of_trade_date)
        for pending_plan in pending_plans:
            planned_trade_date = str(pending_plan.get("planned_trade_date", ""))
            if planned_trade_date and planned_trade_date <= as_of_trade_date:
                trades.extend(self._execute_plan(pending_plan, positions, state))
                self._archive_pending_plan(pending_plan, planned_trade_date)
        self.pending_plan_path.unlink(missing_ok=True)

        self._update_highest_close(positions, as_of_trade_date)
        self._save_positions(positions)
        self._append_trades(trades)
        return self._save_and_result(state, positions, trades, as_of_trade_date)

    def stage_plan(self, new_plan_path: Path, as_of_trade_date: str) -> Tianzhu9SimulationResult:
        self.positions_dir.mkdir(parents=True, exist_ok=True)
        positions = self._load_positions()
        state = self._load_state()
        self._write_pending_plan(new_plan_path)
        if "equity" in state:
            return self._result_from_state(state, positions, executed_trades=0)
        return self._save_and_result(state, positions, [], as_of_trade_date)

    def settle_and_stage_plan(self, new_plan_path: Path, as_of_trade_date: str) -> Tianzhu9SimulationResult:
        self.settle_pending_plan(as_of_trade_date)
        return self.stage_plan(new_plan_path, as_of_trade_date)

    def _save_and_result(
        self,
        state: dict,
        positions: list[dict],
        trades: list[dict],
        as_of_trade_date: str,
    ) -> Tianzhu9SimulationResult:
        previous_equity = float(state.get("equity") or state["initial_cash"])
        snapshots = self._position_snapshots(positions, as_of_trade_date)
        positions_market_value = sum(snapshot.market_value for snapshot in snapshots)
        equity = float(state["cash"]) + positions_market_value
        initial_cash = float(state["initial_cash"])
        updated_at = self._now_iso()
        state.update(
            {
                "initial_cash": initial_cash,
                "cash": float(state["cash"]),
                "equity": equity,
                "previous_equity": previous_equity,
                "positions_market_value": positions_market_value,
                "daily_pnl": equity - previous_equity,
                "daily_return": (equity / previous_equity - 1.0) if previous_equity else 0.0,
                "total_return": (equity / initial_cash - 1.0) if initial_cash else 0.0,
                "last_trade_date": as_of_trade_date,
                "updated_at": updated_at,
                "positions": [asdict(snapshot) for snapshot in snapshots],
            }
        )
        self._save_state(state)
        return self._result_from_state(state, positions, executed_trades=len(trades))

    def _result_from_state(
        self,
        state: dict,
        positions: list[dict],
        executed_trades: int,
    ) -> Tianzhu9SimulationResult:
        staged = self._load_pending_plan()
        snapshots = [
            Tianzhu9PositionSnapshot(**snapshot)
            for snapshot in state.get("positions", [])
        ]
        if not snapshots and positions:
            snapshots = self._position_snapshots(positions, str(state.get("last_trade_date") or ""))
        return Tianzhu9SimulationResult(
            positions_path=self.positions_path,
            state_path=self.state_path,
            trades_path=self.trades_path,
            pending_plan_path=self.pending_plan_path,
            initial_cash=float(state["initial_cash"]),
            cash=float(state["cash"]),
            equity=float(state["equity"]),
            previous_equity=float(state.get("previous_equity") or state["initial_cash"]),
            positions_market_value=float(state.get("positions_market_value") or 0.0),
            daily_pnl=float(state.get("daily_pnl") or 0.0),
            daily_return=float(state.get("daily_return") or 0.0),
            total_return=float(state.get("total_return") or 0.0),
            positions_count=len(positions),
            executed_trades=executed_trades,
            pending_plan_date=str(staged["planned_trade_date"]) if staged else None,
            last_trade_date=str(state.get("last_trade_date") or ""),
            updated_at=str(state.get("updated_at") or ""),
            positions=snapshots,
        )

    def _load_state(self) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            state.setdefault("initial_cash", float(self.config.backtest.initial_cash))
            state.setdefault("cash", float(state["initial_cash"]))
            return state
        initial_cash = float(self.config.backtest.initial_cash)
        return {"initial_cash": initial_cash, "cash": initial_cash}

    def _save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def _now_iso(self) -> str:
        timezone = getattr(getattr(self.config, "runtime", None), "timezone", None) or "Asia/Shanghai"
        try:
            return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
        except Exception:
            return datetime.now().isoformat(timespec="seconds")

    def _load_positions(self) -> list[dict]:
        if not self.positions_path.exists():
            return []
        frame = pd.read_csv(self.positions_path)
        if frame.empty:
            return []
        rows = frame.to_dict(orient="records")
        for row in rows:
            row["symbol"] = str(row["symbol"])
            row["name"] = str(row["name"])
            row["entry_date"] = str(row["entry_date"])
            row["entry_price"] = float(row["entry_price"])
            row["quantity"] = int(row["quantity"])
            row["highest_close"] = float(row.get("highest_close") or row["entry_price"])
        return rows

    def _save_positions(self, positions: list[dict]) -> None:
        columns = ["symbol", "name", "entry_date", "entry_price", "quantity", "highest_close"]
        frame = pd.DataFrame(positions, columns=columns)
        frame.to_csv(self.positions_path, index=False)

    def _load_pending_plan(self) -> dict | None:
        if not self.pending_plan_path.exists():
            return None
        return json.loads(self.pending_plan_path.read_text(encoding="utf-8"))

    def _pending_plans_to_settle(self, as_of_trade_date: str) -> list[dict]:
        pending_plan = self._load_pending_plan()
        if pending_plan is not None:
            planned_trade_date = str(pending_plan.get("planned_trade_date") or "")
            if planned_trade_date and planned_trade_date <= as_of_trade_date:
                return [pending_plan]
            return []
        return self._find_unsettled_generated_plans(as_of_trade_date)

    def _find_unsettled_generated_plans(self, as_of_trade_date: str) -> list[dict]:
        reports_dir = self.base_dir / self.config.paths.reports_dir / "tianzhu9-orders"
        if not reports_dir.exists():
            return []
        candidates = []
        for path in reports_dir.glob("tianzhu9-orders-*.json"):
            try:
                plan = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            planned_trade_date = str(plan.get("planned_trade_date") or "")
            signal_trade_date = str(plan.get("signal_trade_date") or "")
            if not planned_trade_date or planned_trade_date > as_of_trade_date:
                continue
            if self._settled_plan_archive_path(signal_trade_date, planned_trade_date).exists():
                continue
            candidates.append((planned_trade_date, signal_trade_date, plan))
        if not candidates:
            return []
        return [plan for _, _, plan in sorted(candidates)]

    def _write_pending_plan(self, plan_path: Path) -> None:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.pending_plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _archive_pending_plan(self, plan: dict, planned_trade_date: str) -> None:
        archive_dir = self.positions_dir / "tianzhu9_settled_plans"
        archive_dir.mkdir(parents=True, exist_ok=True)
        signal_trade_date = str(plan.get("signal_trade_date") or "unknown")
        path = self._settled_plan_archive_path(signal_trade_date, planned_trade_date)
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    def _settled_plan_archive_path(self, signal_trade_date: str, planned_trade_date: str) -> Path:
        return self.positions_dir / "tianzhu9_settled_plans" / f"{signal_trade_date}-{planned_trade_date}.json"

    def _execute_plan(self, plan: dict, positions: list[dict], state: dict) -> list[dict]:
        planned_trade_date = str(plan["planned_trade_date"])
        prices = self.repository.load_daily(planned_trade_date).set_index("ts_code")
        trades = []
        held = {position["symbol"]: position for position in positions}

        for order in plan.get("sell_orders", []):
            symbol = str(order["symbol"])
            if symbol not in held or symbol not in prices.index:
                continue
            limit_price = order.get("limit_price")
            if limit_price is None:
                continue
            row = prices.loc[symbol]
            day_open = float(row["open"])
            day_high = float(row["high"])
            limit_price = float(limit_price)
            if any(math.isnan(value) or value <= 0 for value in (day_open, day_high, limit_price)):
                continue
            if day_high < limit_price:
                continue
            fill_price = day_open if day_open >= limit_price else limit_price
            position = held[symbol]
            shares = int(position["quantity"])
            gross_amount = fill_price * shares
            fees = max(
                gross_amount * (self.config.backtest.commission_rate + self.config.backtest.stamp_duty_rate),
                5.0,
            )
            net_amount = gross_amount - fees
            state["cash"] = float(state["cash"]) + net_amount
            positions.remove(position)
            trades.append(self._trade_row(planned_trade_date, "SELL", order, shares, fill_price, gross_amount, fees, net_amount))

        held = {position["symbol"]: position for position in positions}
        buy_orders = [order for order in plan.get("buy_orders", []) if str(order["symbol"]) not in held]
        for order in buy_orders:
            if len(positions) >= int(self.config.market.max_positions):
                break
            symbol = str(order["symbol"])
            if symbol not in prices.index:
                continue
            limit_price = order.get("limit_price")
            if limit_price is None:
                continue
            row = prices.loc[symbol]
            day_open = float(row["open"])
            day_low = float(row["low"])
            limit_price = float(limit_price)
            if any(math.isnan(value) or value <= 0 for value in (day_open, day_low, limit_price)):
                continue
            if day_low > limit_price:
                continue
            fill_price = day_open if day_open <= limit_price else limit_price
            shares = self._buy_quantity(
                fill_price,
                float(state["cash"]),
                positions=positions,
                order_count=len(buy_orders),
            )
            if shares < int(self.config.backtest.lot_size):
                continue
            gross_amount = fill_price * shares
            fees = max(gross_amount * self.config.backtest.commission_rate, 5.0)
            if gross_amount + fees > float(state["cash"]):
                shares = self._affordable_quantity(fill_price, float(state["cash"]))
                if shares < int(self.config.backtest.lot_size):
                    continue
                gross_amount = fill_price * shares
                fees = max(gross_amount * self.config.backtest.commission_rate, 5.0)
            net_amount = gross_amount + fees
            state["cash"] = float(state["cash"]) - net_amount
            positions.append(
                {
                    "symbol": symbol,
                    "name": str(order.get("name") or symbol),
                    "entry_date": planned_trade_date,
                    "entry_price": round(fill_price, 4),
                    "quantity": shares,
                    "highest_close": round(fill_price, 4),
                }
            )
            trades.append(self._trade_row(planned_trade_date, "BUY", order, shares, fill_price, gross_amount, fees, net_amount))

        return trades

    def _buy_quantity(self, price: float, cash: float, positions: list[dict], order_count: int) -> int:
        equity = cash + sum(float(position["entry_price"]) * int(position["quantity"]) for position in positions)
        remaining_orders = max(order_count, 1)
        target_value = min(equity / self.hold_days, cash / remaining_orders, cash)
        return self._round_lot(int(target_value / price))

    def _affordable_quantity(self, price: float, cash: float) -> int:
        affordable = int(cash / (price * (1 + self.config.backtest.commission_rate)))
        return self._round_lot(affordable)

    def _round_lot(self, shares: int) -> int:
        lot_size = int(self.config.backtest.lot_size)
        return (int(shares) // lot_size) * lot_size

    def _trade_row(
        self,
        trade_date: str,
        action: str,
        order: dict,
        shares: int,
        price: float,
        gross_amount: float,
        fees: float,
        net_amount: float,
    ) -> dict:
        return {
            "trade_date": trade_date,
            "action": action,
            "symbol": str(order["symbol"]),
            "name": str(order.get("name") or order["symbol"]),
            "shares": shares,
            "price": round(price, 4),
            "gross_amount": round(gross_amount, 4),
            "fees": round(fees, 4),
            "net_amount": round(net_amount, 4),
            "rank": order.get("rank"),
            "score": order.get("score"),
        }

    def _append_trades(self, trades: list[dict]) -> None:
        columns = ["trade_date", "action", "symbol", "name", "shares", "price", "gross_amount", "fees", "net_amount", "rank", "score"]
        if not trades and self.trades_path.exists():
            return
        frame = pd.DataFrame(trades, columns=columns)
        if self.trades_path.exists():
            existing = pd.read_csv(self.trades_path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_csv(self.trades_path, index=False)

    def _update_highest_close(self, positions: list[dict], trade_date: str) -> None:
        if not positions:
            return
        try:
            prices = self.repository.load_daily(trade_date).set_index("ts_code")
        except FileNotFoundError:
            return
        for position in positions:
            symbol = position["symbol"]
            if symbol not in prices.index:
                continue
            close_price = float(prices.loc[symbol, "close"])
            if math.isnan(close_price) or close_price <= 0:
                continue
            position["highest_close"] = max(float(position.get("highest_close") or 0.0), close_price)

    def _position_snapshots(self, positions: list[dict], trade_date: str) -> list[Tianzhu9PositionSnapshot]:
        try:
            prices = self.repository.load_daily(trade_date).set_index("ts_code")
        except FileNotFoundError:
            prices = pd.DataFrame()
        snapshots = []
        for position in positions:
            symbol = position["symbol"]
            price = float(position["entry_price"])
            if not prices.empty and symbol in prices.index:
                close_price = float(prices.loc[symbol, "close"])
                if not math.isnan(close_price) and close_price > 0:
                    price = close_price
            quantity = int(position["quantity"])
            entry_price = float(position["entry_price"])
            market_value = price * quantity
            cost_basis = entry_price * quantity
            unrealized_pnl = market_value - cost_basis
            snapshots.append(
                Tianzhu9PositionSnapshot(
                    symbol=symbol,
                    name=str(position["name"]),
                    entry_date=str(position["entry_date"]),
                    entry_price=entry_price,
                    quantity=quantity,
                    last_price=price,
                    market_value=market_value,
                    cost_basis=cost_basis,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_return=(price / entry_price - 1.0) if entry_price else 0.0,
                    holding_days=self._holding_days(str(position["entry_date"]), trade_date),
                )
            )
        return snapshots

    def _holding_days(self, entry_date: str, trade_date: str) -> int:
        if len(entry_date) == 10 and "-" in entry_date:
            compact_entry = entry_date.replace("-", "")
        else:
            compact_entry = entry_date
        try:
            start = datetime.strptime(compact_entry, "%Y%m%d").date()
            end = datetime.strptime(trade_date, "%Y%m%d").date()
        except ValueError:
            return 1
        return max((end - start).days + 1, 1)
