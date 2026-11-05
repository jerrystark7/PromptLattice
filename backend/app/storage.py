import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("PROMPTLATTICE_DB", Path(__file__).resolve().parents[1] / "data" / "promptlattice.db"))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


@contextmanager
def connection():
    db = connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize() -> None:
    with connection() as db:
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, mode TEXT NOT NULL, initial_prompt TEXT NOT NULL, latest_prompt TEXT NOT NULL, stop_reason TEXT NOT NULL DEFAULT '', config_json TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS rounds (run_id TEXT NOT NULL, round_no INTEGER NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, detail_json TEXT NOT NULL, PRIMARY KEY(run_id, round_no), FOREIGN KEY(run_id) REFERENCES runs(id))")
        # SSE tasks live only in process memory and cannot be resumed after a
        # backend restart. Never present a persisted task as still running.
        mark_running_interrupted(db, "服务重启时任务仍在运行；未完成的供应商请求无法恢复。")


def mark_running_interrupted(db: sqlite3.Connection | None = None, reason: str = "任务连接中断，无法继续恢复。") -> int:
    if db is None:
        with connection() as db:
            return mark_running_interrupted(db, reason)
    rows = db.execute("SELECT id FROM runs WHERE status IN ('running', 'stopping')").fetchall()
    for row in rows:
        run_id = row["id"]
        db.execute("UPDATE runs SET status = 'interrupted', stop_reason = ? WHERE id = ?", (reason, run_id))
        rounds = db.execute("SELECT round_no, detail_json FROM rounds WHERE run_id = ?", (run_id,)).fetchall()
        for round_row in rounds:
            detail = json.loads(round_row["detail_json"])
            if detail.get("status") in ("running", "stopping"):
                detail["status"] = "interrupted"
                detail["stop_reason"] = reason
                db.execute("UPDATE rounds SET status = 'interrupted', detail_json = ? WHERE run_id = ? AND round_no = ?", (json.dumps(detail, ensure_ascii=False), run_id, round_row["round_no"]))
    return len(rows)


def interrupt_run(run_id: str, reason: str = "页面已刷新；原任务无法从新页面恢复。") -> None:
    with connection() as db:
        db.execute("UPDATE runs SET status = 'interrupted', stop_reason = ? WHERE id = ? AND status = 'running'", (reason, run_id))
        rounds = db.execute("SELECT round_no, detail_json FROM rounds WHERE run_id = ?", (run_id,)).fetchall()
        for round_row in rounds:
            detail = json.loads(round_row["detail_json"])
            if detail.get("status") == "running":
                detail["status"] = "interrupted"
                detail["stop_reason"] = reason
                db.execute("UPDATE rounds SET status = 'interrupted', detail_json = ? WHERE run_id = ? AND round_no = ?", (json.dumps(detail, ensure_ascii=False), run_id, round_row["round_no"]))


def request_stop(run_id: str, reason: str = "正在等待当前供应商请求结束；后续步骤和轮次已取消。") -> None:
    with connection() as db:
        db.execute("UPDATE runs SET status = 'stopping', stop_reason = ? WHERE id = ? AND status = 'running'", (reason, run_id))
        rounds = db.execute("SELECT round_no, detail_json FROM rounds WHERE run_id = ?", (run_id,)).fetchall()
        for round_row in rounds:
            detail = json.loads(round_row["detail_json"])
            if detail.get("status") == "running":
                detail["status"] = "stopping"
                detail["stop_reason"] = reason
                db.execute("UPDATE rounds SET status = 'stopping', detail_json = ? WHERE run_id = ? AND round_no = ?", (json.dumps(detail, ensure_ascii=False), run_id, round_row["round_no"]))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scrub_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: scrub_keys(v) for k, v in value.items() if "api_key" not in k.lower() and k.lower() not in {"authorization", "headers", "jev_key", "target_key"}}
    if isinstance(value, list):
        return [scrub_keys(v) for v in value]
    return value


def create_run(run_id: str, mode: str, prompt: str, config: dict[str, Any]) -> None:
    with connection() as db:
        db.execute("INSERT INTO runs VALUES (?, ?, 'running', ?, ?, ?, '', ?)", (run_id, now(), mode, prompt, prompt, json.dumps(scrub_keys(config), ensure_ascii=False)))


def save_round(run_id: str, round_no: int, detail: dict[str, Any]) -> None:
    safe = scrub_keys(detail)
    safe.setdefault("created_at", now())
    with connection() as db:
        db.execute("INSERT OR REPLACE INTO rounds VALUES (?, ?, ?, ?, ?)", (run_id, round_no, safe["created_at"], safe.get("status", "completed"), json.dumps(safe, ensure_ascii=False)))
        db.execute("UPDATE runs SET latest_prompt = ? WHERE id = ?", (safe.get("next_prompt") or safe.get("prompt") or "", run_id))


def finish_run(run_id: str, status: str, stop_reason: str) -> None:
    with connection() as db:
        db.execute("UPDATE runs SET status = ?, stop_reason = ? WHERE id = ?", (status, stop_reason, run_id))


def list_runs() -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 100").fetchall()
        return [{"id": r["id"], "created_at": r["created_at"], "status": r["status"], "mode": r["mode"], "initial_prompt": r["initial_prompt"], "latest_prompt": r["latest_prompt"], "stop_reason": r["stop_reason"], "round_count": db.execute("SELECT COUNT(*) FROM rounds WHERE run_id = ?", (r["id"],)).fetchone()[0]} for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    with connection() as db:
        run = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return None
        rounds = db.execute("SELECT detail_json FROM rounds WHERE run_id = ? ORDER BY round_no", (run_id,)).fetchall()
        return {"id": run["id"], "created_at": run["created_at"], "status": run["status"], "mode": run["mode"], "initial_prompt": run["initial_prompt"], "latest_prompt": run["latest_prompt"], "stop_reason": run["stop_reason"], "config": json.loads(run["config_json"]), "rounds": [json.loads(r[0]) for r in rounds]}
