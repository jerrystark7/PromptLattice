import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("PROMPTLATTICE_DB", Path(__file__).resolve().parents[1] / "data" / "promptlattice.db"))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def initialize() -> None:
    with connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, mode TEXT NOT NULL, initial_prompt TEXT NOT NULL, latest_prompt TEXT NOT NULL, stop_reason TEXT NOT NULL DEFAULT '', config_json TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS rounds (run_id TEXT NOT NULL, round_no INTEGER NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, detail_json TEXT NOT NULL, PRIMARY KEY(run_id, round_no), FOREIGN KEY(run_id) REFERENCES runs(id))")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scrub_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: scrub_keys(v) for k, v in value.items() if "api_key" not in k.lower() and k.lower() not in {"authorization", "headers", "jev_key", "target_key"}}
    if isinstance(value, list):
        return [scrub_keys(v) for v in value]
    return value


def create_run(run_id: str, mode: str, prompt: str, config: dict[str, Any]) -> None:
    with connect() as db:
        db.execute("INSERT INTO runs VALUES (?, ?, 'running', ?, ?, ?, '', ?)", (run_id, now(), mode, prompt, prompt, json.dumps(scrub_keys(config), ensure_ascii=False)))


def save_round(run_id: str, round_no: int, detail: dict[str, Any]) -> None:
    safe = scrub_keys(detail)
    safe.setdefault("created_at", now())
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO rounds VALUES (?, ?, ?, ?, ?)", (run_id, round_no, safe["created_at"], safe.get("status", "completed"), json.dumps(safe, ensure_ascii=False)))
        db.execute("UPDATE runs SET latest_prompt = ? WHERE id = ?", (safe.get("next_prompt") or safe.get("prompt") or "", run_id))


def finish_run(run_id: str, status: str, stop_reason: str) -> None:
    with connect() as db:
        db.execute("UPDATE runs SET status = ?, stop_reason = ? WHERE id = ?", (status, stop_reason, run_id))


def list_runs() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 100").fetchall()
        return [{"id": r["id"], "created_at": r["created_at"], "status": r["status"], "mode": r["mode"], "initial_prompt": r["initial_prompt"], "latest_prompt": r["latest_prompt"], "stop_reason": r["stop_reason"], "round_count": db.execute("SELECT COUNT(*) FROM rounds WHERE run_id = ?", (r["id"],)).fetchone()[0]} for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    with connect() as db:
        run = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return None
        rounds = db.execute("SELECT detail_json FROM rounds WHERE run_id = ? ORDER BY round_no", (run_id,)).fetchall()
        return {"id": run["id"], "created_at": run["created_at"], "status": run["status"], "mode": run["mode"], "initial_prompt": run["initial_prompt"], "latest_prompt": run["latest_prompt"], "stop_reason": run["stop_reason"], "config": json.loads(run["config_json"]), "rounds": [json.loads(r[0]) for r in rounds]}
