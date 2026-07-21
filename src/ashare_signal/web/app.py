from __future__ import annotations

from contextlib import asynccontextmanager
import csv
from io import StringIO
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
import uvicorn

from ashare_signal.config import load_config, load_env_file
from ashare_signal.data.repository import DataRepository
from ashare_signal.strategy.exit_rules import EXIT_PROFILES
from ashare_signal.web.catalog import ResultCatalog
from ashare_signal.web.services import BENCHMARKS, DashboardDataService
from ashare_signal.web.storage import DashboardStore
from ashare_signal.web.tasks import BASELINE_PARAMETERS, ResearchTaskRunner


class BacktestRequest(BaseModel):
    start_date: str = Field(pattern=r"^\d{8}$")
    end_date: str = Field(pattern=r"^\d{8}$")
    top_n: int = Field(default=5, ge=1, le=20)
    max_positions: int = Field(default=5, ge=1, le=10)
    market_min_breadth: float = Field(default=0.50, ge=0.0, le=1.0)
    market_min_return_20d: float = Field(default=0.0, ge=-0.30, le=0.30)
    aggressive_position_size_multiplier: float = Field(default=0.50, ge=0.05, le=1.0)
    hard_exit_days: int = Field(default=23, ge=1, le=120)
    exit_profile: str = "legacy"
    winner_bypass_peak_pct: float | None = Field(default=None, ge=0.01, le=1.0)
    risk_off_failed_days: int | None = Field(default=None, ge=2, le=60)
    high_drawdown_pct: float | None = Field(default=None, ge=0.01, le=0.50)
    chandelier_atr_multiplier: float | None = Field(default=None, ge=0.5, le=10.0)
    trend_decay: bool = False

    @model_validator(mode="after")
    def validate_dates_and_exit(self) -> "BacktestRequest":
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if self.exit_profile not in EXIT_PROFILES:
            raise ValueError(f"exit_profile must be one of: {', '.join(EXIT_PROFILES)}")
        if self.risk_off_failed_days is not None and self.winner_bypass_peak_pct is None:
            raise ValueError("risk_off_failed_days requires winner_bypass_peak_pct")
        return self


def create_app(
    *,
    base_dir: Path | None = None,
    config_path: str | None = None,
) -> FastAPI:
    resolved_base = (base_dir or Path.cwd()).resolve()
    resolved_config = config_path or os.getenv("ASHARE_CONFIG", "configs/strategy.toml.example")
    load_env_file(resolved_base / ".env")
    config = load_config(resolved_base / resolved_config)
    repository = DataRepository(config=config, base_dir=resolved_base)
    store = DashboardStore(
        resolved_base / config.paths.processed_data_dir / "dashboard" / "dashboard.sqlite3"
    )
    catalog = ResultCatalog(
        base_dir=resolved_base,
        reports_dir=config.paths.reports_dir,
        store=store,
    )
    service = DashboardDataService(
        config=config,
        repository=repository,
        base_dir=resolved_base,
        store=store,
        catalog=catalog,
    )
    runner = ResearchTaskRunner(
        base_dir=resolved_base,
        config_path=resolved_config,
        store=store,
        catalog=catalog,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        catalog.rebuild()
        runner.start()
        yield
        runner.stop()

    app = FastAPI(
        title="A-share Strategy Console",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.base_dir = resolved_base
    app.state.store = store
    app.state.catalog = catalog
    app.state.service = service
    app.state.runner = runner

    @app.get("/api/health")
    def api_health() -> dict:
        return {"status": "ok", "service": "a-share-strategy-console"}

    @app.get("/api/dashboard")
    def api_dashboard() -> dict:
        return service.dashboard()

    @app.get("/api/data-health")
    def api_data_health() -> dict:
        return service.data_health()

    @app.get("/api/stocks/search")
    def api_stock_search(q: str = "", limit: int = Query(default=20, ge=1, le=50)) -> list[dict]:
        return service.search_stocks(q, limit)

    @app.get("/api/stocks/{symbol}")
    def api_stock_detail(
        symbol: str,
        result_id: str | None = None,
        range_name: str = Query(default="1y", alias="range", pattern=r"^(1y|all)$"),
    ) -> dict:
        return service.stock_detail(symbol, result_id=result_id, range_name=range_name)

    @app.get("/api/stocks/{symbol}/export.csv", response_class=PlainTextResponse)
    def api_stock_export(
        symbol: str,
        result_id: str | None = None,
        range_name: str = Query(default="1y", alias="range", pattern=r"^(1y|all)$"),
    ) -> PlainTextResponse:
        detail = service.stock_detail(symbol, result_id=result_id, range_name=range_name)
        output = StringIO()
        fields = ["trade_date", "open", "high", "low", "close", "volume", "amount", "ma5", "ma10", "ma20", "ma60"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail["bars"])
        return PlainTextResponse(
            output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{symbol}-daily.csv"'},
        )

    @app.get("/api/results")
    def api_results(include_archived: bool = False) -> list[dict]:
        return service.results(include_archived=include_archived)

    @app.get("/api/results/{result_id}")
    def api_result_detail(result_id: str) -> dict:
        result = service.result_detail(result_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Research result not found")
        return result

    @app.get("/api/results/{result_id}/trades")
    def api_result_trades(result_id: str, limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
        if store.get_result(result_id) is None:
            raise HTTPException(status_code=404, detail="Research result not found")
        return service.result_trades(result_id, limit=limit)

    @app.get("/api/compare")
    def api_compare(
        ids: str,
        benchmark: str = Query(default="000300.SH"),
    ) -> dict:
        if benchmark not in BENCHMARKS:
            raise HTTPException(status_code=422, detail="Unsupported benchmark")
        result_ids = [value.strip() for value in ids.split(",") if value.strip()]
        return service.compare(result_ids, benchmark)

    @app.post("/api/results/reindex")
    def api_reindex() -> dict:
        return catalog.rebuild()

    @app.post("/api/results/{result_id}/archive")
    def api_archive(result_id: str, archived: bool = True) -> dict:
        if not store.set_result_archived(result_id, archived):
            raise HTTPException(status_code=404, detail="Research result not found")
        return {"id": result_id, "archived": archived}

    @app.delete("/api/results/{result_id}")
    def api_delete_result(result_id: str, confirm: str = "") -> dict:
        result = store.get_result(result_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Research result not found")
        if confirm != result_id:
            raise HTTPException(status_code=409, detail="Deletion confirmation does not match the result id")
        if result["protected"]:
            raise HTTPException(status_code=409, detail="The current baseline is protected")
        if result["source"] != "task":
            raise HTTPException(status_code=409, detail="Imported historical artifacts are read-only")
        reports_root = (resolved_base / config.paths.reports_dir).resolve()
        for value in set(result["artifacts"].values()):
            path = catalog.resolve(value)
            if path and path.exists() and path.is_relative_to(reports_root):
                path.unlink()
        if not store.delete_result(result_id):
            raise HTTPException(status_code=409, detail="Result could not be deleted")
        return {"id": result_id, "deleted": True}

    @app.post("/api/backtests", status_code=202)
    def api_create_backtest(request: BacktestRequest) -> dict:
        parameters = request.model_dump()
        tasks = runner.enqueue_experiment(parameters, include_baseline=True)
        return {
            "tasks": tasks,
            "baseline_parameters": BASELINE_PARAMETERS,
        }

    @app.get("/api/tasks")
    def api_tasks() -> list[dict]:
        return store.list_tasks()

    @app.get("/api/tasks/{task_id}")
    def api_task(task_id: str) -> dict:
        task = store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Research task not found")
        return task

    @app.get("/api/tasks/{task_id}/log", response_class=PlainTextResponse)
    def api_task_log(task_id: str) -> PlainTextResponse:
        log = runner.read_log(task_id)
        if log is None:
            raise HTTPException(status_code=404, detail="Research task not found")
        return PlainTextResponse(log)

    @app.post("/api/tasks/{task_id}/cancel")
    def api_cancel_task(task_id: str) -> dict:
        task = runner.cancel(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Research task not found")
        return task

    frontend_dist = _frontend_dist(resolved_base)
    if frontend_dist and (frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend(full_path: str, request: Request):
        del request
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        if frontend_dist:
            candidate = (frontend_dist / full_path).resolve()
            if candidate.is_relative_to(frontend_dist) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")
        return {
            "service": "a-share-strategy-console",
            "message": "Frontend build not found. Run npm install && npm run dev in frontend/.",
            "api_docs": "/docs",
        }

    return app


def _frontend_dist(base_dir: Path) -> Path | None:
    candidates = [
        Path(os.environ["ASHARE_FRONTEND_DIST"]) if os.getenv("ASHARE_FRONTEND_DIST") else None,
        base_dir / "frontend" / "dist",
        Path(__file__).resolve().parent / "static",
    ]
    for path in candidates:
        if path and (path / "index.html").exists():
            return path.resolve()
    return None


def run() -> None:
    uvicorn.run(
        "ashare_signal.web.app:create_app",
        factory=True,
        host=os.getenv("ASHARE_WEB_HOST", "0.0.0.0"),
        port=int(os.getenv("ASHARE_WEB_PORT", "8787")),
        reload=False,
    )


if __name__ == "__main__":
    run()
