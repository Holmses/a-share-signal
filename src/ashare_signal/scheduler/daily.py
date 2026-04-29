from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
import time as time_module
from zoneinfo import ZoneInfo

from ashare_signal.backtest.engine import BacktestEngine, BacktestResult
from ashare_signal.config import AppConfig
from ashare_signal.data.repository import DataRepository
from ashare_signal.data.sync import SyncResult, TushareSyncService
from ashare_signal.data.tushare_client import TushareClient
from ashare_signal.features.pipeline import UniverseBuildResult, UniverseBuilder
from ashare_signal.portfolio.manager import PortfolioManager, PortfolioSyncResult
from ashare_signal.scheduler.jobs import run_daily_signal_job
from ashare_signal.utils.dates import parse_compact_date, to_compact_date


@dataclass(slots=True)
class DailyRunResult:
    sync_result: SyncResult | None
    universe_result: UniverseBuildResult
    backtest_result: BacktestResult
    portfolio_result: PortfolioSyncResult
    signal_path: Path
    paper_start_date: str
    data_trade_date: str


def _parse_date(value: str | date) -> date:
    return parse_compact_date(to_compact_date(value))


def _today(timezone: str) -> date:
    try:
        tzinfo = ZoneInfo(timezone)
    except Exception as error:  # pragma: no cover - depends on platform tzdata
        raise ValueError(f"Invalid timezone: {timezone}") from error
    return datetime.now(tzinfo).date()


def _resolve_paper_start_date(config: AppConfig, value: str | None) -> date:
    resolved = value or config.runtime.paper_start_date
    if not resolved:
        raise ValueError(
            "paper_start_date is required. Set [runtime].paper_start_date "
            "or pass --paper-start-date."
        )
    return _parse_date(resolved)


def _resolve_sync_start_date(
    repository: DataRepository,
    config: AppConfig,
    paper_start_date: date,
    sync_start_date: str | None,
    end_date: date,
) -> date:
    if sync_start_date:
        return _parse_date(sync_start_date)

    latest_cached = repository.latest_complete_daily_cache_date(end_date=to_compact_date(end_date))
    if latest_cached is None:
        return paper_start_date

    lookback_days = max(int(config.runtime.sync_lookback_days), 0)
    resolved = parse_compact_date(latest_cached) - timedelta(days=lookback_days)
    return min(resolved, end_date)


def _calendar_end_date(config: AppConfig, end_date: date) -> date:
    return end_date + timedelta(days=max(int(config.runtime.calendar_ahead_days), 0))


def run_daily_workflow(
    config: AppConfig,
    repository: DataRepository,
    base_dir: Path,
    paper_start_date: str | None = None,
    end_date: str | None = None,
    sync_start_date: str | None = None,
    skip_sync: bool = False,
) -> DailyRunResult:
    resolved_end_date = _parse_date(end_date) if end_date else _today(config.runtime.timezone)
    resolved_paper_start_date = _resolve_paper_start_date(config, paper_start_date)

    sync_result: SyncResult | None = None
    if not skip_sync:
        resolved_sync_start_date = _resolve_sync_start_date(
            repository=repository,
            config=config,
            paper_start_date=resolved_paper_start_date,
            sync_start_date=sync_start_date,
            end_date=resolved_end_date,
        )
        client = TushareClient(token=config.tushare_token)
        sync_result = TushareSyncService(client=client, repository=repository).sync(
            start_date=to_compact_date(resolved_sync_start_date),
            end_date=to_compact_date(resolved_end_date),
            calendar_end_date=to_compact_date(_calendar_end_date(config, resolved_end_date)),
        )

    data_trade_date = repository.latest_complete_daily_cache_date(
        end_date=to_compact_date(resolved_end_date)
    )
    if data_trade_date is None:
        raise ValueError("No complete daily and daily_basic cache is available for paper trading.")

    universe_result = UniverseBuilder(config=config, repository=repository).build(
        as_of=parse_compact_date(data_trade_date)
    )
    backtest_result = BacktestEngine(
        config=config,
        repository=repository,
        base_dir=base_dir,
    ).run(
        start_date=resolved_paper_start_date,
        end_date=parse_compact_date(data_trade_date),
    )
    portfolio_result = PortfolioManager(base_dir=base_dir, repository=repository).sync_from_backtest(
        backtest_result
    )
    signal_path = run_daily_signal_job(
        config=config,
        base_dir=base_dir,
        as_of=parse_compact_date(portfolio_result.last_trade_date),
        holdings_path=portfolio_result.positions_path,
    )

    return DailyRunResult(
        sync_result=sync_result,
        universe_result=universe_result,
        backtest_result=backtest_result,
        portfolio_result=portfolio_result,
        signal_path=signal_path,
        paper_start_date=to_compact_date(resolved_paper_start_date),
        data_trade_date=data_trade_date,
    )


def parse_run_time(value: str) -> datetime_time:
    try:
        parsed = datetime_time.fromisoformat(value)
    except ValueError as error:
        raise ValueError("run time must use HH:MM or HH:MM:SS format") from error
    return parsed.replace(tzinfo=None)


def next_run_datetime(now: datetime, run_at: datetime_time) -> datetime:
    next_run = datetime.combine(now.date(), run_at, tzinfo=now.tzinfo)
    if next_run <= now:
        next_run += timedelta(days=1)
    return next_run


def _print_scheduled_result(result: DailyRunResult) -> None:
    print("Daily workflow completed", flush=True)
    print(f"data_trade_date={result.data_trade_date}", flush=True)
    print(f"paper_start_date={result.paper_start_date}", flush=True)
    print(f"current_equity={result.portfolio_result.current_equity}", flush=True)
    print(f"current_cash={result.portfolio_result.current_cash}", flush=True)
    print(f"holdings_count={result.portfolio_result.holdings_count}", flush=True)
    print(f"signal_path={result.signal_path}", flush=True)


def run_scheduler(
    config: AppConfig,
    repository: DataRepository,
    base_dir: Path,
    paper_start_date: str | None = None,
    run_at: str | None = None,
    timezone: str | None = None,
    sync_start_date: str | None = None,
    skip_sync: bool = False,
    run_on_start: bool = False,
) -> None:
    resolved_run_at = parse_run_time(run_at or config.runtime.daily_run_time)
    resolved_timezone = timezone or config.runtime.timezone
    tzinfo = ZoneInfo(resolved_timezone)

    def execute_once() -> None:
        started_at = datetime.now(tzinfo).isoformat(timespec="seconds")
        print(f"Starting daily workflow at {started_at}", flush=True)
        result = run_daily_workflow(
            config=config,
            repository=repository,
            base_dir=base_dir,
            paper_start_date=paper_start_date,
            sync_start_date=sync_start_date,
            skip_sync=skip_sync,
        )
        _print_scheduled_result(result)

    if run_on_start:
        execute_once()

    while True:
        now = datetime.now(tzinfo)
        next_run = next_run_datetime(now, resolved_run_at)
        sleep_seconds = max((next_run - now).total_seconds(), 1.0)
        print(f"Next daily workflow scheduled at {next_run.isoformat(timespec='seconds')}", flush=True)
        time_module.sleep(sleep_seconds)
        try:
            execute_once()
        except Exception as error:
            print(f"Daily workflow failed: {error}", flush=True)
