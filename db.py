import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone

log     = logging.getLogger("laventra")
DB_PATH = Path(__file__).parent / "laventra.db"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init() -> None:
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS session (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS queue (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                lavvaggio_id  INTEGER NOT NULL,
                device_id     INTEGER,
                plate         TEXT    NOT NULL,
                vehicle_type  TEXT    NOT NULL DEFAULT 'unknown',
                started_at    TEXT    NOT NULL,
                ended_at      TEXT    NOT NULL,
                retry_count   INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT    NOT NULL,
                last_tried_at TEXT
            );
        """)
    log.debug(f"DB ready → {DB_PATH}")


def session_get(key: str, default: str = None):
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT value FROM session WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default
    except Exception as e:
        log.error(f"DB read error ({key}): {e}")
        return default


def session_set(key: str, value) -> None:
    try:
        with _conn() as con:
            con.execute(
                """
                INSERT INTO session (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )
    except Exception as e:
        log.error(f"DB write error ({key}): {e}")


def session_clear() -> None:
    try:
        with _conn() as con:
            con.execute("DELETE FROM session")
        log.info("Session cleared")
    except Exception as e:
        log.error(f"DB clear error: {e}")


def has_session() -> bool:
    return bool(session_get("token")) and bool(session_get("lavvaggio_id"))


def session_summary() -> dict:
    return {
        "api_url":        session_get("api_url",        "—"),
        "email":          session_get("email",           "—"),
        "lavvaggio_id":   session_get("lavvaggio_id",   "—"),
        "lavvaggio_name": session_get("lavvaggio_name", "—"),
        "device_id":      session_get("device_id",      "—"),
        "camera_url":     session_get("camera_url",     "—"),
        "token_set":      "yes" if session_get("token") else "no",
    }


def enqueue(
    lavvaggio_id: int,
    plate: str,
    vehicle_type: str,
    started_at: str,
    ended_at: str,
    device_id: int = None,
) -> None:
    try:
        now = datetime.now(timezone.utc).isoformat()
        with _conn() as con:
            con.execute(
                """
                INSERT INTO queue
                  (lavvaggio_id, device_id, plate, vehicle_type,
                   started_at, ended_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (lavvaggio_id, device_id, plate, vehicle_type,
                 started_at, ended_at, now),
            )
        log.info(f"📥 Queued offline → {plate}")
    except Exception as e:
        log.error(f"Failed to queue event ({plate}): {e}")


def pending(max_retries: int = 10, limit: int = 20) -> list:
    try:
        with _conn() as con:
            return con.execute(
                """
                SELECT * FROM queue
                WHERE retry_count < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (max_retries, limit),
            ).fetchall()
    except Exception as e:
        log.error(f"Failed to read queue: {e}")
        return []


def mark_sent(event_id: int) -> None:
    try:
        with _conn() as con:
            con.execute("DELETE FROM queue WHERE id = ?", (event_id,))
    except Exception as e:
        log.error(f"Failed to delete queued event {event_id}: {e}")


def mark_failed(event_id: int) -> None:
    try:
        now = datetime.now(timezone.utc).isoformat()
        with _conn() as con:
            con.execute(
                """
                UPDATE queue
                SET retry_count   = retry_count + 1,
                    last_tried_at = ?
                WHERE id = ?
                """,
                (now, event_id),
            )
    except Exception as e:
        log.error(f"Failed to update retry count {event_id}: {e}")


def queue_count(max_retries: int = 10) -> int:
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT COUNT(*) as n FROM queue WHERE retry_count < ?",
                (max_retries,),
            ).fetchone()
        return row["n"] if row else 0
    except Exception as e:
        log.error(f"Failed to count queue: {e}")
        return 0