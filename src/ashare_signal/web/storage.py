from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Iterator


class DashboardStore:
    """Small SQLite index for research artifacts and background jobs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_results (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    summary_path TEXT UNIQUE,
                    equity_path TEXT,
                    trades_path TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    command TEXT,
                    protected INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_tasks (
                    id TEXT PRIMARY KEY,
                    result_id TEXT,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    parameters_json TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(result_id) REFERENCES research_results(id)
                );

                CREATE INDEX IF NOT EXISTS idx_results_updated
                    ON research_results(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_created
                    ON research_tasks(created_at DESC);
                """
            )

    def upsert_result(self, payload: dict) -> None:
        fields = (
            "id",
            "title",
            "kind",
            "strategy",
            "source",
            "status",
            "start_date",
            "end_date",
            "summary_path",
            "equity_path",
            "trades_path",
            "metrics_json",
            "parameters_json",
            "artifacts_json",
            "command",
            "protected",
            "archived",
            "created_at",
            "updated_at",
        )
        values = [payload.get(field) for field in fields]
        placeholders = ", ".join("?" for _ in fields)
        updates = ", ".join(
            f"{field}=excluded.{field}"
            for field in fields
            if field not in {"id", "created_at", "archived"}
        )
        with self.connect() as connection:
            connection.execute(
                f"""
                INSERT INTO research_results ({', '.join(fields)})
                VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET {updates}
                """,
                values,
            )

    def list_results(self, *, include_archived: bool = False) -> list[dict]:
        query = "SELECT * FROM research_results"
        parameters: tuple[object, ...] = ()
        if not include_archived:
            query += " WHERE archived=0"
        query += " ORDER BY end_date DESC, updated_at DESC"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._decode_result(row) for row in rows]

    def get_result(self, result_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_results WHERE id=?",
                (result_id,),
            ).fetchone()
        return self._decode_result(row) if row else None

    def find_result_by_summary(self, summary_path: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_results WHERE summary_path=?",
                (summary_path,),
            ).fetchone()
        return self._decode_result(row) if row else None

    def set_result_archived(self, result_id: str, archived: bool) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE research_results SET archived=?, updated_at=? WHERE id=?",
                (int(archived), _now(), result_id),
            )
        return cursor.rowcount > 0

    def delete_result(self, result_id: str) -> bool:
        with self.connect() as connection:
            connection.execute(
                "UPDATE research_tasks SET result_id=NULL WHERE result_id=?",
                (result_id,),
            )
            cursor = connection.execute(
                "DELETE FROM research_results WHERE id=? AND protected=0 AND source='task'",
                (result_id,),
            )
        return cursor.rowcount > 0

    def create_task(self, payload: dict) -> None:
        fields = (
            "id",
            "result_id",
            "status",
            "progress",
            "parameters_json",
            "command_json",
            "log_path",
            "created_at",
            "started_at",
            "finished_at",
            "error",
            "cancel_requested",
        )
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO research_tasks ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})",
                [payload.get(field) for field in fields],
            )

    def update_task(self, task_id: str, **changes: object) -> None:
        allowed = {
            "result_id",
            "status",
            "progress",
            "started_at",
            "finished_at",
            "error",
            "cancel_requested",
        }
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE research_tasks SET {assignments} WHERE id=?",
                [*values.values(), task_id],
            )

    def list_tasks(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM research_tasks ORDER BY created_at DESC"
            ).fetchall()
        return [self._decode_task(row) for row in rows]

    def get_task(self, task_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_tasks WHERE id=?",
                (task_id,),
            ).fetchone()
        return self._decode_task(row) if row else None

    def recover_interrupted_tasks(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE research_tasks
                SET status='failed', progress=100, finished_at=?,
                    error='Dashboard restarted while this task was active.'
                WHERE status IN ('queued', 'running')
                """,
                (_now(),),
            )

    @staticmethod
    def _decode_result(row: sqlite3.Row) -> dict:
        payload = dict(row)
        for field in ("metrics_json", "parameters_json", "artifacts_json"):
            payload[field.removesuffix("_json")] = _load_json(payload.pop(field), {})
        payload["protected"] = bool(payload["protected"])
        payload["archived"] = bool(payload["archived"])
        return payload

    @staticmethod
    def _decode_task(row: sqlite3.Row) -> dict:
        payload = dict(row)
        payload["parameters"] = _load_json(payload.pop("parameters_json"), {})
        payload["command"] = _load_json(payload.pop("command_json"), [])
        payload["cancel_requested"] = bool(payload["cancel_requested"])
        return payload


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json(value: str | None, default: object) -> object:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
