from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ashare_signal.backtest.engine import BacktestEngine
from ashare_signal.config import load_config, load_env_file
from ashare_signal.data.repository import DataRepository
from ashare_signal.portfolio.manager import PortfolioManager
from ashare_signal.scheduler.daily import run_daily_workflow, run_scheduler
from ashare_signal.scheduler.jobs import run_daily_signal_job
from ashare_signal.utils.dates import parse_compact_date


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share V1 signal board CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-signal", help="Generate a daily signal board")
    generate.add_argument(
        "--config",
        default="configs/strategy.toml.example",
        help="Path to the TOML config file",
    )
    generate.add_argument(
        "--as-of",
        default=None,
        help="Signal date in ISO format, defaults to today",
    )
    generate.add_argument(
        "--holdings",
        default="configs/current_positions.csv",
        help="Path to the current holdings CSV file",
    )

    sync = subparsers.add_parser("sync-tushare", help="Sync Tushare raw datasets into the local cache")
    sync.add_argument(
        "--config",
        default="configs/strategy.toml.example",
        help="Path to the TOML config file",
    )
    sync.add_argument(
        "--start-date",
        required=True,
        help="Inclusive sync start date in ISO format",
    )
    sync.add_argument(
        "--end-date",
        required=True,
        help="Inclusive sync end date in ISO format",
    )

    build_universe = subparsers.add_parser(
        "build-universe",
        help="Build a filtered universe snapshot from local Tushare cache",
    )
    build_universe.add_argument(
        "--config",
        default="configs/strategy.toml.example",
        help="Path to the TOML config file",
    )
    build_universe.add_argument(
        "--as-of",
        default=None,
        help="Requested date in ISO format, defaults to today",
    )

    paper_trade = subparsers.add_parser(
        "paper-trade",
        help="Run deterministic paper trading to the latest cached trade date and persist simulated positions",
    )
    paper_trade.add_argument(
        "--config",
        default="configs/strategy.toml.example",
        help="Path to the TOML config file",
    )
    paper_trade.add_argument(
        "--start-date",
        default=None,
        help="Inclusive paper-trade start date in ISO format",
    )
    paper_trade.add_argument(
        "--end-date",
        default=None,
        help="Inclusive paper-trade end date in ISO format, defaults to latest cached trade date",
    )
    paper_trade.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Override [backtest].initial_cash for this run",
    )

    backtest = subparsers.add_parser("backtest", help="Run the placeholder backtest")
    backtest.add_argument(
        "--config",
        default="configs/strategy.toml.example",
        help="Path to the TOML config file",
    )
    backtest.add_argument(
        "--start-date",
        default=None,
        help="Inclusive backtest start date in ISO format",
    )
    backtest.add_argument(
        "--end-date",
        default=None,
        help="Inclusive backtest end date in ISO format",
    )
    backtest.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Override [backtest].initial_cash for this run",
    )

    run_daily = subparsers.add_parser(
        "run-daily",
        help="Sync Tushare data, build the latest universe, update paper positions, and write the signal board",
    )
    run_daily.add_argument(
        "--config",
        default="configs/strategy.toml.example",
        help="Path to the TOML config file",
    )
    run_daily.add_argument(
        "--paper-start-date",
        default=None,
        help="Paper trading start date in ISO format, overrides [runtime].paper_start_date",
    )
    run_daily.add_argument(
        "--end-date",
        default=None,
        help="Data sync end date in ISO format, defaults to today in [runtime].timezone",
    )
    run_daily.add_argument(
        "--sync-start-date",
        default=None,
        help="Optional sync start date override in ISO format",
    )
    run_daily.add_argument(
        "--skip-sync",
        action="store_true",
        help="Use existing local cache without calling Tushare",
    )
    run_daily.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Override [backtest].initial_cash for this run",
    )

    scheduler = subparsers.add_parser(
        "run-scheduler",
        help="Run the daily workflow on a fixed local time inside a long-running container",
    )
    scheduler.add_argument(
        "--config",
        default="configs/strategy.toml.example",
        help="Path to the TOML config file",
    )
    scheduler.add_argument(
        "--paper-start-date",
        default=None,
        help="Paper trading start date in ISO format, overrides [runtime].paper_start_date",
    )
    scheduler.add_argument(
        "--run-at",
        default=None,
        help="Daily run time, defaults to [runtime].daily_run_time",
    )
    scheduler.add_argument(
        "--timezone",
        default=None,
        help="Scheduler timezone, defaults to [runtime].timezone",
    )
    scheduler.add_argument(
        "--sync-start-date",
        default=None,
        help="Optional sync start date override in ISO format",
    )
    scheduler.add_argument(
        "--skip-sync",
        action="store_true",
        help="Use existing local cache without calling Tushare",
    )
    scheduler.add_argument(
        "--run-on-start",
        action="store_true",
        help="Run once immediately before waiting for the next scheduled time",
    )
    scheduler.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Override [backtest].initial_cash for this run",
    )
    return parser


def _parse_date(value: str | None) -> date:
    if not value:
        return date.today()
    return date.fromisoformat(value)


def _handle_missing_dependency(parser: argparse.ArgumentParser, error: ModuleNotFoundError) -> None:
    if error.name in {"pandas", "tushare"}:
        parser.exit(1, f"Missing dependency: {error.name}. Run `pip install -e .` first.\n")
    raise error


def _apply_initial_cash_override(config, value: float | None) -> None:
    if value is not None:
        config.backtest.initial_cash = float(value)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    base_dir = Path.cwd()
    load_env_file(base_dir / ".env")
    config = load_config(base_dir / args.config)
    _apply_initial_cash_override(config, getattr(args, "initial_cash", None))
    repository = DataRepository(config=config, base_dir=base_dir)
    repository.ensure_directories()

    if args.command == "generate-signal":
        try:
            output = run_daily_signal_job(
                config=config,
                base_dir=base_dir,
                as_of=_parse_date(args.as_of),
                holdings_path=base_dir / args.holdings,
            )
        except FileNotFoundError as error:
            parser.exit(
                1,
                f"Missing required input file: {error}. "
                "Run `ashare-signal build-universe` and prepare the holdings CSV first.\n",
            )
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        print(f"Signal board written to: {output}")
        return 0

    if args.command == "sync-tushare":
        try:
            from ashare_signal.data.sync import TushareSyncService
            from ashare_signal.data.tushare_client import TushareClient
        except ModuleNotFoundError as error:
            _handle_missing_dependency(parser, error)

        client = TushareClient(token=config.tushare_token)
        try:
            result = TushareSyncService(client=client, repository=repository).sync(
                start_date=args.start_date,
                end_date=args.end_date,
            )
        except ModuleNotFoundError as error:
            _handle_missing_dependency(parser, error)
        except RuntimeError as error:
            parser.exit(1, f"{error}\n")
        print("Tushare sync completed")
        print(f"start_date={result.start_date}")
        print(f"end_date={result.end_date}")
        print(f"calendar_end_date={result.calendar_end_date}")
        print(f"open_trade_days={result.open_trade_days}")
        print(f"stock_count={result.stock_count}")
        print(f"daily_files={result.daily_files}")
        print(f"daily_basic_files={result.daily_basic_files}")
        return 0

    if args.command == "build-universe":
        try:
            from ashare_signal.features.pipeline import UniverseBuilder
        except ModuleNotFoundError as error:
            _handle_missing_dependency(parser, error)

        try:
            result = UniverseBuilder(config=config, repository=repository).build(
                as_of=_parse_date(args.as_of),
            )
        except FileNotFoundError as error:
            parser.exit(1, f"Missing local cache file: {error}. Run `ashare-signal sync-tushare` first.\n")
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        print("Universe snapshot completed")
        print(f"trade_date={result.trade_date}")
        print(f"total_symbols={result.total_symbols}")
        print(f"candidate_symbols={result.candidate_symbols}")
        print(f"output={result.output_path}")
        return 0

    if args.command == "backtest":
        try:
            result = BacktestEngine(
                config=config,
                repository=repository,
                base_dir=base_dir,
            ).run(
                start_date=_parse_date(args.start_date) if args.start_date else None,
                end_date=_parse_date(args.end_date) if args.end_date else None,
            )
        except FileNotFoundError as error:
            parser.exit(1, f"Missing required input file: {error}. Run `ashare-signal sync-tushare` first.\n")
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        print("Backtest completed")
        print(f"start_trade_date={result.start_trade_date}")
        print(f"end_trade_date={result.end_trade_date}")
        print(f"initial_cash={result.initial_cash}")
        print(f"ending_equity={result.ending_equity}")
        print(f"total_return={result.total_return}")
        print(f"annual_return={result.annual_return}")
        print(f"max_drawdown={result.max_drawdown}")
        print(f"sharpe={result.sharpe}")
        print(f"turnover={result.turnover}")
        print(f"trade_count={result.trade_count}")
        print(f"win_rate={result.win_rate}")
        print(f"equity_curve_path={result.equity_curve_path}")
        print(f"summary_path={result.summary_path}")
        return 0

    if args.command == "paper-trade":
        try:
            backtest_result = BacktestEngine(
                config=config,
                repository=repository,
                base_dir=base_dir,
            ).run(
                start_date=_parse_date(args.start_date) if args.start_date else None,
                end_date=_parse_date(args.end_date) if args.end_date else None,
            )
            portfolio_result = PortfolioManager(
                base_dir=base_dir,
                repository=repository,
            ).sync_from_backtest(backtest_result)
            signal_path = run_daily_signal_job(
                config=config,
                base_dir=base_dir,
                as_of=parse_compact_date(portfolio_result.last_trade_date),
                holdings_path=portfolio_result.positions_path,
            )
        except FileNotFoundError as error:
            parser.exit(1, f"Missing required input file: {error}. Run `ashare-signal sync-tushare` first.\n")
        except ValueError as error:
            parser.exit(1, f"{error}\n")

        print("Paper trade sync completed")
        print(f"last_trade_date={portfolio_result.last_trade_date}")
        print(f"current_equity={portfolio_result.current_equity}")
        print(f"current_cash={portfolio_result.current_cash}")
        print(f"holdings_count={portfolio_result.holdings_count}")
        print(f"positions_path={portfolio_result.positions_path}")
        print(f"latest_pnl_path={portfolio_result.latest_pnl_path}")
        print(f"state_path={portfolio_result.state_path}")
        print(f"snapshots_dir={portfolio_result.snapshots_dir}")
        print(f"signal_path={signal_path}")
        return 0

    if args.command == "run-daily":
        try:
            result = run_daily_workflow(
                config=config,
                repository=repository,
                base_dir=base_dir,
                paper_start_date=args.paper_start_date,
                end_date=args.end_date,
                sync_start_date=args.sync_start_date,
                skip_sync=args.skip_sync,
            )
        except ModuleNotFoundError as error:
            _handle_missing_dependency(parser, error)
        except RuntimeError as error:
            parser.exit(1, f"{error}\n")
        except (FileNotFoundError, ValueError) as error:
            parser.exit(1, f"{error}\n")

        print("Daily workflow completed")
        if result.sync_result is not None:
            print(f"sync_start_date={result.sync_result.start_date}")
            print(f"sync_end_date={result.sync_result.end_date}")
            print(f"calendar_end_date={result.sync_result.calendar_end_date}")
            print(f"daily_files={result.sync_result.daily_files}")
            print(f"daily_basic_files={result.sync_result.daily_basic_files}")
        else:
            print("sync=skipped")
        print(f"data_trade_date={result.data_trade_date}")
        print(f"paper_start_date={result.paper_start_date}")
        print(f"initial_cash={result.backtest_result.initial_cash}")
        print(f"current_equity={result.portfolio_result.current_equity}")
        print(f"current_cash={result.portfolio_result.current_cash}")
        print(f"holdings_count={result.portfolio_result.holdings_count}")
        print(f"trade_count={result.backtest_result.trade_count}")
        print(f"total_return={result.backtest_result.total_return}")
        print(f"max_drawdown={result.backtest_result.max_drawdown}")
        print(f"win_rate={result.backtest_result.win_rate}")
        print(f"positions_path={result.portfolio_result.positions_path}")
        print(f"latest_pnl_path={result.portfolio_result.latest_pnl_path}")
        print(f"state_path={result.portfolio_result.state_path}")
        print(f"signal_path={result.signal_path}")
        return 0

    if args.command == "run-scheduler":
        try:
            run_scheduler(
                config=config,
                repository=repository,
                base_dir=base_dir,
                paper_start_date=args.paper_start_date,
                run_at=args.run_at,
                timezone=args.timezone,
                sync_start_date=args.sync_start_date,
                skip_sync=args.skip_sync,
                run_on_start=args.run_on_start,
            )
        except ModuleNotFoundError as error:
            _handle_missing_dependency(parser, error)
        except KeyboardInterrupt:
            return 130
        except (RuntimeError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
