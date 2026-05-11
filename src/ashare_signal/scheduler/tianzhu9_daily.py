from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import os
import time as time_module
from zoneinfo import ZoneInfo

from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.data.sync import SyncResult, TushareSyncService
from ashare_signal.data.tushare_client import TushareClient
from ashare_signal.notify.feishu import FeishuSendResult, FeishuWebhookNotifier
from ashare_signal.scheduler.daily import _calendar_end_date, _parse_date, _resolve_sync_start_date, _today
from ashare_signal.scheduler.daily import next_run_datetime, parse_run_time
from ashare_signal.strategy.tianzhu9_orders import Tianzhu9OrderPlan, generate_tianzhu9_order_plan
from ashare_signal.strategy.tianzhu9_orders import plan_to_feishu_text
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


@dataclass(slots=True)
class Tianzhu9DailyResult:
    sync_result: SyncResult | None
    data_trade_date: str
    plan: Tianzhu9OrderPlan
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
    top_n: int = 1,
    hold_days: int = 2,
    max_hold_days: int = 4,
) -> Tianzhu9DailyResult:
    resolved_end_date = _parse_date(end_date) if end_date else _today(config.runtime.timezone)
    sync_result: SyncResult | None = None
    if not skip_sync:
        latest_cached = repository.latest_complete_daily_cache_date(end_date=to_compact_date(resolved_end_date))
        paper_start = parse_compact_date(latest_cached) if latest_cached else resolved_end_date - timedelta(days=120)
        resolved_sync_start_date = _resolve_sync_start_date(
            repository=repository,
            config=config,
            paper_start_date=paper_start,
            sync_start_date=sync_start_date,
            end_date=resolved_end_date,
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

    plan = generate_tianzhu9_order_plan(
        config=config,
        repository=repository,
        base_dir=base_dir,
        as_of=parse_compact_date(data_trade_date),
        positions_path=base_dir / positions_path if positions_path else None,
        top_n=top_n,
        hold_days=hold_days,
        max_hold_days=max_hold_days,
    )
    notification_result = None
    if notify:
        notification_result = send_tianzhu9_plan_to_feishu(plan)
    return Tianzhu9DailyResult(
        sync_result=sync_result,
        data_trade_date=data_trade_date,
        plan=plan,
        notification_result=notification_result,
    )


def send_tianzhu9_plan_to_feishu(plan: Tianzhu9OrderPlan) -> FeishuSendResult | None:
    webhook = os.getenv("FEISHU_WEBHOOK", "").strip()
    if not webhook:
        return None
    secret = os.getenv("FEISHU_SECRET", "").strip() or None
    return FeishuWebhookNotifier(webhook_url=webhook, secret=secret).send_text(plan_to_feishu_text(plan))


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
    top_n: int = 1,
    hold_days: int = 2,
    max_hold_days: int = 4,
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
        )
        print(f"data_trade_date={result.data_trade_date}", flush=True)
        print(f"planned_trade_date={result.plan.planned_trade_date}", flush=True)
        print(f"buy_orders={len(result.plan.buy_orders)}", flush=True)
        print(f"sell_orders={len(result.plan.sell_orders)}", flush=True)
        print(f"hold_orders={len(result.plan.hold_orders)}", flush=True)
        print(f"markdown_path={result.plan.markdown_path}", flush=True)
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
