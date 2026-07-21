from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import uuid

from ashare_signal.web.catalog import ResultCatalog
from ashare_signal.web.storage import DashboardStore, json_text


BASELINE_PARAMETERS = {
    "top_n": 5,
    "max_positions": 5,
    "market_min_breadth": 0.50,
    "market_min_return_20d": 0.0,
    "aggressive_position_size_multiplier": 0.50,
    "hard_exit_days": 23,
    "exit_profile": "legacy",
    "winner_bypass_peak_pct": None,
    "risk_off_failed_days": None,
    "high_drawdown_pct": None,
    "chandelier_atr_multiplier": None,
    "trend_decay": False,
}


class ResearchTaskRunner:
    """Runs validated research commands serially without touching scheduler state."""

    def __init__(
        self,
        *,
        base_dir: Path,
        config_path: str,
        store: DashboardStore,
        catalog: ResultCatalog,
    ) -> None:
        self.base_dir = base_dir
        self.config_path = config_path
        self.store = store
        self.catalog = catalog
        self.queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self.store.recover_interrupted_tasks()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, name="research-task-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.queue.put(None)
        with self._lock:
            for process in self._processes.values():
                if process.poll() is None:
                    process.terminate()
        if self._thread:
            self._thread.join(timeout=5.0)

    def enqueue_experiment(self, parameters: dict, *, include_baseline: bool = True) -> list[dict]:
        tasks = []
        if include_baseline and not _is_baseline(parameters) and not self._has_matching_baseline(parameters):
            baseline = {**BASELINE_PARAMETERS, "start_date": parameters["start_date"], "end_date": parameters["end_date"]}
            tasks.append(self.enqueue(baseline))
        tasks.append(self.enqueue(parameters))
        return tasks

    def enqueue(self, parameters: dict) -> dict:
        task_id = uuid.uuid4().hex[:16]
        command = build_backtest_command(parameters, self.config_path)
        logs_dir = self.base_dir / "logs" / "dashboard-tasks"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{task_id}.log"
        payload = {
            "id": task_id,
            "result_id": None,
            "status": "queued",
            "progress": 0,
            "parameters_json": json_text(parameters),
            "command_json": json_text(command),
            "log_path": str(log_path.relative_to(self.base_dir)),
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "cancel_requested": 0,
        }
        self.store.create_task(payload)
        self.queue.put(task_id)
        task = self.store.get_task(task_id)
        if task is None:
            raise RuntimeError("Queued task could not be read from SQLite.")
        return task

    def cancel(self, task_id: str) -> dict | None:
        task = self.store.get_task(task_id)
        if task is None or task["status"] in {"completed", "failed", "cancelled"}:
            return task
        self.store.update_task(task_id, cancel_requested=1)
        with self._lock:
            process = self._processes.get(task_id)
            if process is not None and process.poll() is None:
                process.terminate()
        if task["status"] == "queued":
            self.store.update_task(
                task_id,
                status="cancelled",
                progress=100,
                finished_at=_now(),
            )
        return self.store.get_task(task_id)

    def read_log(self, task_id: str, max_chars: int = 40_000) -> str | None:
        task = self.store.get_task(task_id)
        if task is None:
            return None
        path = self.base_dir / task["log_path"]
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]

    def _worker(self) -> None:
        while True:
            task_id = self.queue.get()
            if task_id is None:
                return
            task = self.store.get_task(task_id)
            if task is None or task["status"] == "cancelled" or task["cancel_requested"]:
                continue
            self._run_task(task)

    def _run_task(self, task: dict) -> None:
        task_id = str(task["id"])
        log_path = self.base_dir / task["log_path"]
        self.store.update_task(task_id, status="running", progress=5, started_at=_now())
        summary_path: Path | None = None
        progress_stop = threading.Event()
        progress_thread: threading.Thread | None = None
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write("$ " + shlex.join(task["command"]) + "\n\n")
                log_handle.flush()
                process = subprocess.Popen(
                    task["command"],
                    cwd=self.base_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self._lock:
                    self._processes[task_id] = process
                progress_thread = threading.Thread(
                    target=self._heartbeat_progress,
                    args=(task_id, progress_stop),
                    name=f"research-progress-{task_id}",
                    daemon=True,
                )
                progress_thread.start()
                if process.stdout is not None:
                    for line in process.stdout:
                        log_handle.write(line)
                        log_handle.flush()
                        if line.startswith("summary_path="):
                            summary_path = Path(line.split("=", 1)[1].strip())
                        self._advance_progress(task_id, step=4, maximum=90)
                return_code = process.wait()
            refreshed = self.store.get_task(task_id)
            cancelled = bool(refreshed and refreshed["cancel_requested"])
            if cancelled:
                self.store.update_task(
                    task_id,
                    status="cancelled",
                    progress=100,
                    finished_at=_now(),
                )
                return
            if return_code != 0:
                raise RuntimeError(f"Research command exited with code {return_code}.")
            if summary_path is None:
                summary_path = _summary_path_from_log(log_path)
            if summary_path is None or not summary_path.exists():
                raise RuntimeError("Research command completed without a readable summary_path.")
            result = self.catalog.index_summary(
                summary_path,
                source="task",
                command=shlex.join(task["command"]),
            )
            self.store.update_task(
                task_id,
                result_id=result["id"],
                status="completed",
                progress=100,
                finished_at=_now(),
            )
        except Exception as error:
            self.store.update_task(
                task_id,
                status="failed",
                progress=100,
                finished_at=_now(),
                error=str(error),
            )
        finally:
            progress_stop.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)
            with self._lock:
                self._processes.pop(task_id, None)

    def _heartbeat_progress(self, task_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(5.0):
            task = self.store.get_task(task_id)
            if task is None or task["status"] != "running":
                return
            self._advance_progress(task_id, step=1, maximum=85)

    def _advance_progress(self, task_id: str, *, step: int, maximum: int) -> None:
        task = self.store.get_task(task_id)
        if task is None or task["status"] != "running":
            return
        self.store.update_task(
            task_id,
            progress=min(int(task["progress"]) + step, maximum),
        )

    def _has_matching_baseline(self, parameters: dict) -> bool:
        for result in self.store.list_results(include_archived=True):
            if not result["protected"]:
                continue
            if result.get("start_date") == parameters["start_date"] and result.get("end_date") == parameters["end_date"]:
                return True
        return False


def build_backtest_command(parameters: dict, config_path: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ashare_signal",
        "backtest-full-a-momentum",
        "--config",
        config_path,
        "--start-date",
        parameters["start_date"],
        "--end-date",
        parameters["end_date"],
        "--selection-variant",
        "quality_momentum",
        "--top-n",
        str(parameters["top_n"]),
        "--max-positions",
        str(parameters["max_positions"]),
        "--groups",
        "main,chinext,star",
        "--enabled-recipes",
        "momentum_core",
        "--entry-market-states",
        "normal,aggressive",
        "--aggressive-position-size-multiplier",
        str(parameters["aggressive_position_size_multiplier"]),
        "--market-min-breadth",
        str(parameters["market_min_breadth"]),
        "--market-min-return-20d",
        str(parameters["market_min_return_20d"]),
        "--defensive-market-min-breadth",
        "0.5",
        "--defensive-position-size-multiplier",
        "0.25",
        "--style-min-breadth",
        "0.48",
        "--style-min-return-20d",
        "-0.01",
        "--style-score-weight",
        "0.06",
        "--exit-profile",
        parameters["exit_profile"],
        "--hard-exit-days",
        str(parameters["hard_exit_days"]),
    ]
    optional_values = (
        ("winner_bypass_peak_pct", "--exit-winner-hard-exit-bypass-peak-pct"),
        ("risk_off_failed_days", "--exit-risk-off-failed-hard-exit-days"),
        ("high_drawdown_pct", "--exit-high-drawdown-pct"),
        ("chandelier_atr_multiplier", "--exit-chandelier-atr-multiplier"),
    )
    for field, flag in optional_values:
        value = parameters.get(field)
        if value is not None:
            command.extend([flag, str(value)])
    if parameters.get("trend_decay"):
        command.append("--exit-trend-decay")
    return command


def _is_baseline(parameters: dict) -> bool:
    return all(parameters.get(key) == value for key, value in BASELINE_PARAMETERS.items())


def _summary_path_from_log(path: Path) -> Path | None:
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("summary_path="):
            return Path(line.split("=", 1)[1].strip())
    return None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
