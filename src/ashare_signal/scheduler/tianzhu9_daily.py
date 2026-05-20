from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import os
import time as time_module
from zoneinfo import ZoneInfo

from ashare_signal.config import AppConfig
from ashare_signal.backtest.selection_event_study import SelectionEventStudyEngine
from ashare_signal.data.repository import DataRepository
from ashare_signal.data.sync import SyncResult, TushareSyncService
from ashare_signal.data.tushare_client import TushareClient
from ashare_signal.notify.feishu import FeishuSendResult, FeishuWebhookNotifier
from ashare_signal.portfolio.tianzhu9_simulator import Tianzhu9PaperBroker, Tianzhu9SimulationResult
from ashare_signal.scheduler.daily import _calendar_end_date, _parse_date, _resolve_sync_start_date, _today
from ashare_signal.scheduler.daily import next_run_datetime, parse_run_time
from ashare_signal.strategy.tianzhu9_orders import Tianzhu9OrderPlan, generate_tianzhu9_order_plan
from ashare_signal.strategy.tianzhu9_orders import plan_to_feishu_text, render_tianzhu9_order_plan
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


@dataclass(slots=True)
class Tianzhu9DailyResult:
    sync_result: SyncResult | None
    data_trade_date: str
    plan: Tianzhu9OrderPlan
    simulation_result: Tianzhu9SimulationResult
    notification_result: FeishuSendResult | None


def run_tianzhu9_daily_workflow(
    config: AppConfig,
    repository: DataRepository,
    base_dir: Path,
    end_date: str | None = None,
    sync_start_date: str | None = None,
    skip_sync: bool = False,
    positions_path: str | None = None,
    notify: bool = True,
    top_n: int = 5,
    hold_days: int = 5,
    max_hold_days: int = 10,
    hard_exit_days: int | None = 23,
    failure_exit_days: int | None = 8,
    failure_exit_min_peak_profit_pct: float = 0.03,
    volume_stall_exit: bool = True,
    volume_stall_ratio: float = 1.4,
) -> Tianzhu9DailyResult:
    resolved_end_date = _parse_date(end_date) if end_date else _today(config.runtime.timezone)
    sync_result: SyncResult | None = None
    if not skip_sync:
        resolved_sync_start_date = _resolve_tianzhu9_sync_start_date(
            repository=repository,
            config=config,
            end_date=resolved_end_date,
            sync_start_date=sync_start_date,
        )
        client = TushareClient(token=config.tushare_token)
        sync_result = TushareSyncService(client=client, repository=repository).sync(
            start_date=to_compact_date(resolved_sync_start_date),
            end_date=to_compact_date(resolved_end_date),
            calendar_end_date=to_compact_date(_calendar_end_date(config, resolved_end_date)),
        )

    data_trade_date = repository.latest_complete_daily_cache_date(end_date=to_compact_date(resolved_end_date))
    if data_trade_date is None:
        raise ValueError("No complete daily and daily_basic cache is available for Tianzhu9 orders.")

    broker = Tianzhu9PaperBroker(
        config=config,
        repository=repository,
        base_dir=base_dir,
        hold_days=hold_days,
        positions_path=base_dir / positions_path if positions_path else None,
    )
    simulation_result = broker.settle_pending_plan(as_of_trade_date=data_trade_date)

    plan = generate_tianzhu9_order_plan(
        config=config,
        repository=repository,
        base_dir=base_dir,
        as_of=parse_compact_date(data_trade_date),
        positions_path=base_dir / positions_path if positions_path else None,
        top_n=top_n,
        hold_days=hold_days,
        max_hold_days=max_hold_days,
        hard_exit_days=hard_exit_days,
        failure_exit_days=failure_exit_days,
        failure_exit_min_peak_profit_pct=failure_exit_min_peak_profit_pct,
        volume_stall_exit=volume_stall_exit,
        volume_stall_ratio=volume_stall_ratio,
    )
    broker.stage_plan(new_plan_path=plan.json_path, as_of_trade_date=data_trade_date)
    plan.markdown_path.write_text(
        render_tianzhu9_order_plan(plan) + "\n" + render_tianzhu9_simulation_markdown(simulation_result),
        encoding="utf-8",
    )
    notification_result = None
    if notify:
        notification_result = send_tianzhu9_plan_to_feishu(plan, simulation_result)
    return Tianzhu9DailyResult(
        sync_result=sync_result,
        data_trade_date=data_trade_date,
        plan=plan,
        simulation_result=simulation_result,
        notification_result=notification_result,
    )


def send_tianzhu9_plan_to_feishu(
    plan: Tianzhu9OrderPlan,
    simulation_result: Tianzhu9SimulationResult,
) -> FeishuSendResult | None:
    webhook = os.getenv("FEISHU_WEBHOOK", "").strip()
    if not webhook:
        return None
    secret = os.getenv("FEISHU_SECRET", "").strip() or None
    text = plan_to_feishu_text(plan) + "\n\n" + simulation_to_feishu_text(simulation_result)
    return FeishuWebhookNotifier(webhook_url=webhook, secret=secret).send_text(text)


def _resolve_tianzhu9_sync_start_date(
    repository: DataRepository,
    config: AppConfig,
    end_date: date,
    sync_start_date: str | None,
) -> date:
    if sync_start_date:
        return _parse_date(sync_start_date)

    complete_dates = repository.complete_daily_cache_dates(end_date=to_compact_date(end_date))
    required_history = SelectionEventStudyEngine.minimum_signal_history_trade_days()
    if len(complete_dates) < required_history + 1:
        return SelectionEventStudyEngine.recommended_sync_start_date(
            repository=repository,
            target_date=end_date,
            prior_trade_days=required_history,
        )

    latest_cached = complete_dates[-1]
    return _resolve_sync_start_date(
        repository=repository,
        config=config,
        paper_start_date=parse_compact_date(latest_cached),
        sync_start_date=None,
        end_date=end_date,
    )


def simulation_to_feishu_text(result: Tianzhu9SimulationResult) -> str:
    lines = [
        "模拟账户:",
        f"- 运行时间: {result.updated_at}",
        f"- 数据日: {format_trade_date(result.last_trade_date)}",
        f"- 初始资金: {format_money(result.initial_cash)}",
        f"- 总资产: {format_money(result.equity)}",
        f"- 现金: {format_money(result.cash)}",
        f"- 持仓市值: {format_money(result.positions_market_value)}",
        f"- 当日盈亏: {format_signed_money(result.daily_pnl)} ({format_pct(result.daily_return)})",
        f"- 总收益率: {format_pct(result.total_return)}",
        f"- 持仓数: {result.positions_count}",
        f"- 本次模拟成交: {result.executed_trades}",
    ]
    lines.append("")
    lines.append("持仓浮盈亏:")
    if not result.positions:
        lines.append("- 无")
    for position in result.positions[:8]:
        lines.append(
            f"- {position.symbol} {position.name} "
            f"数量:{position.quantity} 买入:{position.entry_price:.2f} 现价:{position.last_price:.2f} "
            f"市值:{format_money(position.market_value)} "
            f"浮盈亏:{format_signed_money(position.unrealized_pnl)} ({format_pct(position.unrealized_return)}) "
            f"持有:{position.holding_days}天"
        )
    return "\n".join(lines)


def render_tianzhu9_simulation_markdown(result: Tianzhu9SimulationResult) -> str:
    lines = [
        "## 模拟账户",
        "",
        f"- 运行时间：{result.updated_at}",
        f"- 数据日：{format_trade_date(result.last_trade_date)}",
        f"- 初始资金：{format_money(result.initial_cash)}",
        f"- 总资产：{format_money(result.equity)}",
        f"- 现金：{format_money(result.cash)}",
        f"- 持仓市值：{format_money(result.positions_market_value)}",
        f"- 当日盈亏：{format_signed_money(result.daily_pnl)}（{format_pct(result.daily_return)}）",
        f"- 总收益率：{format_pct(result.total_return)}",
        f"- 持仓数：{result.positions_count}",
        f"- 本次模拟成交：{result.executed_trades}",
        "",
        "## 当前持仓",
    ]
    if not result.positions:
        lines.append("无持仓。")
    else:
        lines.extend(
            [
                "| 代码 | 名称 | 买入日 | 数量 | 买入价 | 现价 | 市值 | 浮盈亏 | 收益率 | 持有天数 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for position in result.positions:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown_cell(position.symbol),
                        _escape_markdown_cell(position.name),
                        _escape_markdown_cell(position.entry_date),
                        str(position.quantity),
                        f"{position.entry_price:.2f}",
                        f"{position.last_price:.2f}",
                        format_money(position.market_value),
                        format_signed_money(position.unrealized_pnl),
                        format_pct(position.unrealized_return),
                        f"{position.holding_days}天",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def format_money(value: float) -> str:
    return f"{value:,.2f}"


def format_signed_money(value: float) -> str:
    return f"{value:+,.2f}"


def format_pct(value: float) -> str:
    return f"{value:+.2%}"


def format_trade_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}" if len(value) == 8 and value.isdigit() else value


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def run_tianzhu9_scheduler(
    config: AppConfig,
    repository: DataRepository,
    base_dir: Path,
    run_at: str | None = None,
    timezone: str | None = None,
    sync_start_date: str | None = None,
    skip_sync: bool = False,
    positions_path: str | None = None,
    run_on_start: bool = False,
    notify: bool = True,
    top_n: int = 5,
    hold_days: int = 5,
    max_hold_days: int = 10,
    hard_exit_days: int | None = 23,
    failure_exit_days: int | None = 8,
    failure_exit_min_peak_profit_pct: float = 0.03,
    volume_stall_exit: bool = True,
    volume_stall_ratio: float = 1.4,
) -> None:
    resolved_run_at = parse_run_time(run_at or config.runtime.daily_run_time)
    resolved_timezone = timezone or config.runtime.timezone
    tzinfo = ZoneInfo(resolved_timezone)

    def execute_once() -> None:
        started_at = datetime.now(tzinfo).isoformat(timespec="seconds")
        print(f"Starting Tianzhu9 daily workflow at {started_at}", flush=True)
        result = run_tianzhu9_daily_workflow(
            config=config,
            repository=repository,
            base_dir=base_dir,
            sync_start_date=sync_start_date,
            skip_sync=skip_sync,
            positions_path=positions_path,
            notify=notify,
            top_n=top_n,
            hold_days=hold_days,
            max_hold_days=max_hold_days,
            hard_exit_days=hard_exit_days,
            failure_exit_days=failure_exit_days,
            failure_exit_min_peak_profit_pct=failure_exit_min_peak_profit_pct,
            volume_stall_exit=volume_stall_exit,
            volume_stall_ratio=volume_stall_ratio,
        )
        print(f"data_trade_date={result.data_trade_date}", flush=True)
        print(f"planned_trade_date={result.plan.planned_trade_date}", flush=True)
        print(f"buy_orders={len(result.plan.buy_orders)}", flush=True)
        print(f"sell_orders={len(result.plan.sell_orders)}", flush=True)
        print(f"hold_orders={len(result.plan.hold_orders)}", flush=True)
        print(f"markdown_path={result.plan.markdown_path}", flush=True)
        print(f"sim_positions={result.simulation_result.positions_count}", flush=True)
        print(f"sim_cash={result.simulation_result.cash}", flush=True)
        print(f"sim_equity={result.simulation_result.equity}", flush=True)
        print(f"sim_daily_pnl={result.simulation_result.daily_pnl}", flush=True)
        print(f"sim_total_return={result.simulation_result.total_return}", flush=True)
        print(f"sim_updated_at={result.simulation_result.updated_at}", flush=True)
        print(f"sim_executed_trades={result.simulation_result.executed_trades}", flush=True)
        if result.notification_result is None:
            print("feishu=skipped", flush=True)
        else:
            print(f"feishu_status={result.notification_result.status_code}", flush=True)

    if run_on_start:
        execute_once()

    while True:
        now = datetime.now(tzinfo)
        next_run = next_run_datetime(now, resolved_run_at)
        sleep_seconds = max((next_run - now).total_seconds(), 1.0)
        print(f"Next Tianzhu9 workflow scheduled at {next_run.isoformat(timespec='seconds')}", flush=True)
        time_module.sleep(sleep_seconds)
        try:
            execute_once()
        except Exception as error:
            print(f"Tianzhu9 workflow failed: {error}", flush=True)
