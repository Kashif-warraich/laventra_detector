"""
SQLite persistence for the detector.

Tables
──────
  kv           Generic key/value: api_url, lavvaggio_id, camera_id, ...
               (Renamed from `session` to break the "session = login" mental model
                that came with the old auth flow. Migration handled below.)
  license      The signed JWT, server-issued timestamps, refresh bookkeeping.
  cameras      Last-known camera list from the backend (so we work offline).
  queue        Retry queue for events that didn't post.
  dead_letter  Events the backend permanently rejected (422). Never deleted
               automatically — operator inspects with --show-deadletter.
"""
import sqlite3
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import config

log = logging.getLogger("laventra")
DB_PATH = config.DB_PATH

# One shared connection, serialised by a reentrant lock. See _conn() for why.
_LOCK = threading.RLock()
_SHARED_CON: sqlite3.Connection | None = None
_SHARED_PATH = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _raw_connection() -> sqlite3.Connection:
    # Reconnect if DB_PATH changed since we last opened (no-op in production,
    # where it never changes; matters for tests that point at a temp file and
    # for --deactivate flows that recreate the DB).
    global _SHARED_CON, _SHARED_PATH
    if _SHARED_CON is None or _SHARED_PATH != DB_PATH:
        if _SHARED_CON is not None:
            try:
                _SHARED_CON.close()
            except Exception:
                pass
        con = sqlite3.connect(DB_PATH, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        # Wait up to 5s for a lock instead of raising "database is locked".
        con.execute("PRAGMA busy_timeout=5000")
        _SHARED_CON = con
        _SHARED_PATH = DB_PATH
    return _SHARED_CON


@contextmanager
def _conn():
    """
    Serialised access to a single shared SQLite connection.

    The detector has up to four writer threads (dispatcher, flusher, heartbeat,
    camera-rediscovery). The old code opened a *new* connection per call and the
    `with sqlite3.connect(...)` context manager committed but never *closed* it —
    a slow connection/fd leak over a multi-day run, plus needless lock contention
    that could surface as "database is locked" and silently drop a write.

    Now every operation funnels through one connection under a reentrant lock:
    writers serialise (SQLite allows only one writer anyway), nothing leaks, and
    the per-call connect + PRAGMA overhead is gone. Call sites keep using
    `with _conn() as con:` unchanged; commit/rollback is handled here.
    """
    with _LOCK:
        con = _raw_connection()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise


def init() -> None:
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS license (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                license_jwt     TEXT    NOT NULL,
                license_id      TEXT,
                device_id       INTEGER,
                lavvaggio_id    INTEGER,
                issued_at       TEXT,
                expires_at      TEXT,
                last_refresh_at TEXT,
                revoked         INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS cameras (
                id          INTEGER PRIMARY KEY,
                name        TEXT    NOT NULL,
                stream_url  TEXT    NOT NULL,
                kind        TEXT,
                metadata    TEXT,
                updated_at  TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS queue (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                plate           TEXT    NOT NULL,
                vehicle_type    TEXT    NOT NULL DEFAULT 'unknown',
                started_at      TEXT    NOT NULL,
                ended_at        TEXT    NOT NULL,
                confidence      REAL,
                camera_id       INTEGER,
                client_event_id TEXT,
                retry_count     INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT    NOT NULL,
                last_tried_at   TEXT
            );
            CREATE TABLE IF NOT EXISTS dead_letter (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                plate         TEXT,
                vehicle_type  TEXT,
                started_at    TEXT,
                ended_at      TEXT,
                confidence    REAL,
                camera_id     INTEGER,
                error_code    INTEGER,
                error_body    TEXT,
                payload_json  TEXT,
                created_at    TEXT    NOT NULL
            );
        """)
        _migrate_legacy(con)
        _ensure_column(con, "queue", "client_event_id", "TEXT")
    log.debug(f"DB ready → {DB_PATH}")


def _ensure_column(con: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """Add a column to an existing table if it's missing (idempotent migration)."""
    try:
        cols = {r["name"] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            log.info(f"Migrated: added {table}.{column}")
    except Exception as e:
        log.warning(f"Column migration skipped ({table}.{column}): {e}")


def _migrate_legacy(con: sqlite3.Connection) -> None:
    """
    Migrate from the pre-license schema:
      - Move `session` rows into `kv`, but DROP the password row before it migrates.
      - Drop the old `token` and `device_token` rows (license replaces both).
    Idempotent: safe to run on a fresh DB.
    """
    try:
        has_session = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session'"
        ).fetchone()
        if not has_session:
            return

        # Carry forward only the keys still meaningful under the license model.
        keep_keys = {
            "api_url", "lavvaggio_id", "lavvaggio_name",
            "camera_id", "camera_url", "camera_port",
            "device_serial",
        }
        rows = con.execute("SELECT key, value FROM session").fetchall()
        carried = 0
        for r in rows:
            if r["key"] in keep_keys and r["value"]:
                con.execute(
                    "INSERT INTO kv (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (r["key"], r["value"]),
                )
                carried += 1
        # Plaintext password and old user-JWT must not survive the migration.
        con.execute("DROP TABLE session")
        log.info(f"Migrated {carried} legacy session entries; legacy auth/password rows dropped")
    except Exception as e:
        log.warning(f"Legacy migration skipped: {e}")


# ── kv (generic key/value) ──────────────────────────────────────────────────
def kv_get(key: str, default: str = None):
    try:
        with _conn() as con:
            row = con.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default
    except Exception as e:
        log.error(f"DB read error ({key}): {e}")
        return default


def kv_set(key: str, value) -> None:
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
    except Exception as e:
        log.error(f"DB write error ({key}): {e}")


def kv_delete(key: str) -> None:
    try:
        with _conn() as con:
            con.execute("DELETE FROM kv WHERE key = ?", (key,))
    except Exception as e:
        log.error(f"DB delete error ({key}): {e}")


# ── license ─────────────────────────────────────────────────────────────────
def license_save(jwt_str: str, *, license_id: str, device_id: int,
                 lavvaggio_id: int, issued_at: str, expires_at: str) -> None:
    try:
        with _conn() as con:
            con.execute(
                """
                INSERT INTO license (id, license_jwt, license_id, device_id,
                                     lavvaggio_id, issued_at, expires_at,
                                     last_refresh_at, revoked)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(id) DO UPDATE SET
                    license_jwt     = excluded.license_jwt,
                    license_id      = excluded.license_id,
                    device_id       = excluded.device_id,
                    lavvaggio_id    = excluded.lavvaggio_id,
                    issued_at       = excluded.issued_at,
                    expires_at      = excluded.expires_at,
                    last_refresh_at = excluded.last_refresh_at,
                    revoked         = 0
                """,
                (jwt_str, license_id, device_id, lavvaggio_id,
                 issued_at, expires_at, _utc_now()),
            )
    except Exception as e:
        log.error(f"License save error: {e}")


def license_get() -> dict | None:
    try:
        with _conn() as con:
            row = con.execute("SELECT * FROM license WHERE id = 1").fetchone()
        return dict(row) if row else None
    except Exception as e:
        log.error(f"License read error: {e}")
        return None


def license_mark_refresh() -> None:
    try:
        with _conn() as con:
            con.execute("UPDATE license SET last_refresh_at = ? WHERE id = 1", (_utc_now(),))
    except Exception as e:
        log.error(f"License mark-refresh error: {e}")


def license_mark_revoked() -> None:
    try:
        with _conn() as con:
            con.execute("UPDATE license SET revoked = 1 WHERE id = 1")
    except Exception as e:
        log.error(f"License mark-revoked error: {e}")


def license_clear() -> None:
    try:
        with _conn() as con:
            con.execute("DELETE FROM license")
    except Exception as e:
        log.error(f"License clear error: {e}")


# ── cameras ─────────────────────────────────────────────────────────────────
def cameras_replace(camera_list: list) -> None:
    """Replace the entire cameras table with the given list of dicts."""
    try:
        now = _utc_now()
        with _conn() as con:
            con.execute("DELETE FROM cameras")
            for c in camera_list:
                con.execute(
                    """
                    INSERT INTO cameras (id, name, stream_url, kind, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(c["id"]),
                        c.get("name", f"Camera {c['id']}"),
                        c["stream_url"],
                        c.get("kind"),
                        c.get("metadata_json"),
                        now,
                    ),
                )
    except Exception as e:
        log.error(f"cameras_replace error: {e}")


def cameras_list() -> list:
    try:
        with _conn() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM cameras ORDER BY id ASC"
            ).fetchall()]
    except Exception as e:
        log.error(f"cameras_list error: {e}")
        return []


def cameras_update_url(camera_id: int, stream_url: str) -> None:
    """
    Update one camera's stream_url in the local cache.

    Startup reads the active camera from this table (cameras.selected()), so a
    runtime URL change must land here to survive a restart — updating only the
    kv `camera_url` would be forgotten on next boot.
    """
    try:
        with _conn() as con:
            con.execute(
                "UPDATE cameras SET stream_url = ?, updated_at = ? WHERE id = ?",
                (stream_url, _utc_now(), int(camera_id)),
            )
    except Exception as e:
        log.error(f"cameras_update_url error: {e}")


def cameras_get(camera_id: int) -> dict | None:
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT * FROM cameras WHERE id = ?", (camera_id,)
            ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        log.error(f"cameras_get error: {e}")
        return None


# ── queue (retry on transient failure) ──────────────────────────────────────
def enqueue(*, plate: str, vehicle_type: str, started_at: str, ended_at: str,
            confidence: float = None, camera_id: int = None,
            client_event_id: str = None) -> None:
    try:
        now = _utc_now()
        with _conn() as con:
            con.execute(
                """
                INSERT INTO queue (plate, vehicle_type, started_at, ended_at,
                                   confidence, camera_id, client_event_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (plate, vehicle_type, started_at, ended_at,
                 confidence, camera_id, client_event_id, now),
            )
    except Exception as e:
        # Last-resort durability: if even the offline queue write fails, dump the
        # full payload to the log at ERROR so the event is recoverable by hand
        # rather than silently lost.
        log.error(
            f"enqueue FAILED ({e}) — event not persisted: "
            f"plate={plate} started={started_at} ended={ended_at} "
            f"client_event_id={client_event_id}"
        )


def pending(max_retries: int = config.QUEUE_MAX_RETRIES,
            limit: int = config.QUEUE_BATCH_LIMIT) -> list:
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
        log.error(f"pending error: {e}")
        return []


def mark_sent(event_id: int) -> None:
    try:
        with _conn() as con:
            con.execute("DELETE FROM queue WHERE id = ?", (event_id,))
    except Exception as e:
        log.error(f"mark_sent error {event_id}: {e}")


def mark_failed(event_id: int) -> None:
    try:
        with _conn() as con:
            con.execute(
                "UPDATE queue SET retry_count = retry_count + 1, last_tried_at = ? WHERE id = ?",
                (_utc_now(), event_id),
            )
    except Exception as e:
        log.error(f"mark_failed error {event_id}: {e}")


def queue_clear() -> int:
    try:
        with _conn() as con:
            n = con.execute("SELECT COUNT(*) AS n FROM queue").fetchone()["n"]
            con.execute("DELETE FROM queue")
        return n
    except Exception as e:
        log.error(f"queue_clear error: {e}")
        return 0


def queue_count(max_retries: int = config.QUEUE_MAX_RETRIES) -> int:
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM queue WHERE retry_count < ?",
                (max_retries,),
            ).fetchone()
        return row["n"] if row else 0
    except Exception as e:
        log.error(f"queue_count error: {e}")
        return 0


def sweep_exhausted(max_retries: int = config.QUEUE_MAX_RETRIES) -> int:
    """
    Move queue rows that have exhausted their retries into dead_letter, then
    delete them from the queue.

    Previously a row that hit retry_count >= max was simply excluded from
    pending()/queue_count() and stranded forever — neither sent, nor
    dead-lettered, nor counted, nor visible to --show-deadletter. This closes
    that silent-loss + table-leak hole. Called on flusher startup and after
    each flush so exhausted events become visible (and alertable) instead of
    rotting in the queue table.
    """
    import json
    moved = 0
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT * FROM queue WHERE retry_count >= ?", (max_retries,)
            ).fetchall()
            for r in rows:
                payload = {
                    "plate":           r["plate"],
                    "vehicle_type":    r["vehicle_type"],
                    "started_at":      r["started_at"],
                    "ended_at":        r["ended_at"],
                    "confidence":      r["confidence"],
                    "camera_id":       r["camera_id"],
                    "client_event_id": r["client_event_id"],
                }
                con.execute(
                    """
                    INSERT INTO dead_letter
                      (plate, vehicle_type, started_at, ended_at, confidence,
                       camera_id, error_code, error_body, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (r["plate"], r["vehicle_type"], r["started_at"], r["ended_at"],
                     r["confidence"], r["camera_id"], 0,
                     f"retries exhausted (>= {max_retries})",
                     json.dumps(payload, default=str), _utc_now()),
                )
                con.execute("DELETE FROM queue WHERE id = ?", (r["id"],))
                moved += 1
    except Exception as e:
        log.error(f"sweep_exhausted error: {e}")
    if moved:
        log.warning(f"⚠️  {moved} event(s) exhausted retries → dead_letter")
    return moved


# ── dead_letter (permanent failures) ────────────────────────────────────────
def dead_letter_add(*, payload: dict, error_code: int, error_body: str) -> None:
    import json
    try:
        with _conn() as con:
            con.execute(
                """
                INSERT INTO dead_letter
                  (plate, vehicle_type, started_at, ended_at, confidence,
                   camera_id, error_code, error_body, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("plate"),
                    payload.get("vehicle_type"),
                    payload.get("started_at"),
                    payload.get("ended_at"),
                    payload.get("confidence"),
                    payload.get("camera_id"),
                    error_code,
                    error_body[:2000] if error_body else None,
                    json.dumps(payload, default=str),
                    _utc_now(),
                ),
            )
    except Exception as e:
        log.error(f"dead_letter_add error: {e}")


def dead_letter_requeue() -> int:
    """
    Move every dead-lettered event back into the retry queue (resetting
    retry_count) and clear those rows. Returns the count moved. Preserves
    client_event_id (parsed from payload_json) so a re-sent event still dedupes
    on the backend. Used by `--requeue-deadletter` after a backend-side fix.
    """
    import json
    try:
        with _conn() as con:
            rows = con.execute("SELECT * FROM dead_letter").fetchall()
            moved = 0
            for r in rows:
                # queue requires started_at/ended_at/plate NOT NULL; skip malformed.
                if not r["started_at"] or not r["ended_at"] or not r["plate"]:
                    log.warning(f"Skipping un-requeueable dead_letter row id={r['id']}")
                    continue
                client_event_id = None
                try:
                    client_event_id = (json.loads(r["payload_json"]) or {}).get("client_event_id")
                except Exception:
                    pass
                con.execute(
                    """
                    INSERT INTO queue (plate, vehicle_type, started_at, ended_at,
                                       confidence, camera_id, client_event_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (r["plate"], r["vehicle_type"] or "unknown",
                     r["started_at"], r["ended_at"], r["confidence"],
                     r["camera_id"], client_event_id, _utc_now()),
                )
                con.execute("DELETE FROM dead_letter WHERE id = ?", (r["id"],))
                moved += 1
        return moved
    except Exception as e:
        log.error(f"dead_letter_requeue error: {e}")
        return 0


def dead_letter_count() -> int:
    try:
        with _conn() as con:
            return con.execute("SELECT COUNT(*) AS n FROM dead_letter").fetchone()["n"]
    except Exception as e:
        log.error(f"dead_letter_count error: {e}")
        return 0


def dead_letter_list(limit: int = 50) -> list:
    try:
        with _conn() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM dead_letter ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()]
    except Exception as e:
        log.error(f"dead_letter_list error: {e}")
        return []


# ── Summary for --status ────────────────────────────────────────────────────
def status_summary() -> dict:
    lic = license_get() or {}
    return {
        "api_url":         kv_get("api_url",        "—"),
        "license_id":      lic.get("license_id",    "—"),
        "lavvaggio_id":    lic.get("lavvaggio_id",  "—"),
        "lavvaggio_name":  kv_get("lavvaggio_name", "—"),
        "device_id":       lic.get("device_id",     "—"),
        "camera_id":       kv_get("camera_id",      "—"),
        "license_set":     "yes" if lic.get("license_jwt") else "no",
        "license_expires": lic.get("expires_at",    "—"),
        "license_revoked": "yes" if lic.get("revoked") else "no",
        "queue":           queue_count(),
        "dead_letter":     dead_letter_count(),
    }


def has_license() -> bool:
    lic = license_get()
    return bool(lic and lic.get("license_jwt") and not lic.get("revoked"))
