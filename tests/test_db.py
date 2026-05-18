"""Tests for db.py — schema, kv, queue, dead_letter, and legacy migration."""
import sqlite3
from pathlib import Path

import pytest

import config
import db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect db to a per-test temp file."""
    p = tmp_path / "laventra-test.db"
    monkeypatch.setattr(db, "DB_PATH", p)
    monkeypatch.setattr(config, "DB_PATH", p)
    db.init()
    yield p


# ─── kv ─────────────────────────────────────────────────────────────────────
def test_kv_set_then_get():
    db.kv_set("api_url", "http://localhost:3000")
    assert db.kv_get("api_url") == "http://localhost:3000"


def test_kv_get_default():
    assert db.kv_get("nope", default="fallback") == "fallback"


def test_kv_set_overwrites():
    db.kv_set("k", "v1")
    db.kv_set("k", "v2")
    assert db.kv_get("k") == "v2"


def test_kv_delete():
    db.kv_set("k", "v")
    db.kv_delete("k")
    assert db.kv_get("k") is None


# ─── license ────────────────────────────────────────────────────────────────
def test_license_save_get():
    db.license_save(
        "jwt.token.value",
        license_id="LIC1", device_id=42, lavvaggio_id=7,
        issued_at="2026-01-01T00:00:00Z", expires_at="2027-01-01T00:00:00Z",
    )
    lic = db.license_get()
    assert lic["license_jwt"] == "jwt.token.value"
    assert lic["license_id"] == "LIC1"
    assert lic["device_id"] == 42
    assert lic["lavvaggio_id"] == 7
    assert lic["revoked"] == 0


def test_license_mark_revoked():
    db.license_save("t", license_id="L", device_id=1, lavvaggio_id=1,
                    issued_at="x", expires_at="y")
    db.license_mark_revoked()
    assert db.license_get()["revoked"] == 1


def test_has_license_requires_not_revoked():
    db.license_save("t", license_id="L", device_id=1, lavvaggio_id=1,
                    issued_at="x", expires_at="y")
    assert db.has_license() is True
    db.license_mark_revoked()
    assert db.has_license() is False


# ─── queue ──────────────────────────────────────────────────────────────────
def test_enqueue_then_pending():
    db.enqueue(plate="AB123CD", vehicle_type="car",
               started_at="2026-01-01T00:00:00Z",
               ended_at="2026-01-01T00:00:05Z",
               confidence=95.0, camera_id=1)
    rows = db.pending()
    assert len(rows) == 1
    assert rows[0]["plate"] == "AB123CD"
    assert rows[0]["camera_id"] == 1


def test_mark_sent_removes_row():
    db.enqueue(plate="X", vehicle_type="car", started_at="a", ended_at="b")
    row_id = db.pending()[0]["id"]
    db.mark_sent(row_id)
    assert db.pending() == []


def test_mark_failed_bumps_retry_count():
    db.enqueue(plate="X", vehicle_type="car", started_at="a", ended_at="b")
    row_id = db.pending()[0]["id"]
    db.mark_failed(row_id)
    rows = db.pending()
    assert rows[0]["retry_count"] == 1


def test_queue_max_retries_filters_pending():
    db.enqueue(plate="X", vehicle_type="car", started_at="a", ended_at="b")
    row_id = db.pending()[0]["id"]
    for _ in range(config.QUEUE_MAX_RETRIES):
        db.mark_failed(row_id)
    assert db.queue_count() == 0
    assert db.pending() == []


def test_queue_clear_removes_all():
    db.enqueue(plate="A", vehicle_type="car", started_at="a", ended_at="b")
    db.enqueue(plate="B", vehicle_type="car", started_at="a", ended_at="b")
    assert db.queue_clear() == 2
    assert db.queue_count() == 0


# ─── dead_letter ────────────────────────────────────────────────────────────
def test_dead_letter_add_and_count():
    payload = {"plate": "X", "vehicle_type": "car",
               "started_at": "a", "ended_at": "b", "confidence": 50.0,
               "camera_id": 1}
    db.dead_letter_add(payload=payload, error_code=422, error_body="bad")
    assert db.dead_letter_count() == 1
    rows = db.dead_letter_list()
    assert rows[0]["plate"] == "X"
    assert rows[0]["error_code"] == 422


# ─── cameras ────────────────────────────────────────────────────────────────
def test_cameras_replace_and_list():
    db.cameras_replace([
        {"id": 1, "name": "Entry", "stream_url": "rtsp://x/1"},
        {"id": 2, "name": "Tunnel", "stream_url": "rtsp://x/2"},
    ])
    cams = db.cameras_list()
    assert len(cams) == 2
    assert cams[0]["name"] == "Entry"


def test_cameras_replace_is_idempotent():
    db.cameras_replace([{"id": 1, "name": "A", "stream_url": "u1"}])
    db.cameras_replace([{"id": 1, "name": "A renamed", "stream_url": "u1"}])
    cams = db.cameras_list()
    assert len(cams) == 1
    assert cams[0]["name"] == "A renamed"


# ─── migration ──────────────────────────────────────────────────────────────
def test_legacy_session_migration_drops_password(tmp_path, monkeypatch):
    """Create a legacy DB with `session` table including password; migration
    must carry forward safe keys AND drop password + token rows."""
    p = tmp_path / "legacy.db"
    monkeypatch.setattr(db, "DB_PATH", p)
    monkeypatch.setattr(config, "DB_PATH", p)
    con = sqlite3.connect(p)
    con.executescript("""
        CREATE TABLE session (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO session VALUES ('api_url',  'http://localhost:3000');
        INSERT INTO session VALUES ('email',    'foo@example.com');
        INSERT INTO session VALUES ('password', 'topsecret');
        INSERT INTO session VALUES ('token',    'old-user-jwt');
        INSERT INTO session VALUES ('lavvaggio_id', '7');
        INSERT INTO session VALUES ('device_token', 'old-device-token');
    """)
    con.commit()
    con.close()

    db.init()

    # Safe keys carried over
    assert db.kv_get("api_url") == "http://localhost:3000"
    assert db.kv_get("lavvaggio_id") == "7"
    # Dangerous keys gone
    assert db.kv_get("password") is None
    assert db.kv_get("token") is None
    assert db.kv_get("device_token") is None
    # email is no longer relevant either
    assert db.kv_get("email") is None
    # session table dropped
    with sqlite3.connect(p) as c2:
        rows = c2.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='session'"
        ).fetchall()
        assert rows == []
