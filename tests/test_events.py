"""
Tests for events.EventDispatcher — the error-routing that decides whether a
washed-car event is marked sent, dead-lettered, or re-queued.

This is the most safety-critical glue in the detector: a misrouted failure
either loses a real event (false dead-letter) or loops forever. The ordering of
the except clauses matters too (LicenseRejected subclasses TransientFailure), so
that path is asserted explicitly.
"""
import pytest

import config
import db
import api
import events as events_module


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    p = tmp_path / "events-test.db"
    monkeypatch.setattr(db, "DB_PATH", p)
    monkeypatch.setattr(config, "DB_PATH", p)
    db.init()
    yield p


def _payload(**over):
    base = {
        "plate": "AB123CD", "vehicle_type": "car",
        "started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-01T00:00:05Z",
        "confidence": 95.0, "camera_id": 1, "client_event_id": "evt-1",
    }
    base.update(over)
    return base


def _dispatcher(monkeypatch, *, result=None, raises=None, on_rejected=None):
    def fake_post(payload):
        if raises is not None:
            raise raises
        return result
    monkeypatch.setattr(api, "post_event", fake_post)
    return events_module.EventDispatcher(on_license_rejected=on_rejected)


def test_success_marks_sent_and_persists_nothing(monkeypatch):
    d = _dispatcher(monkeypatch, result=True)
    d._handle(_payload())
    assert d.stats["sent"] == 1
    assert db.queue_count() == 0
    assert db.dead_letter_count() == 0


def test_permanent_failure_goes_to_dead_letter(monkeypatch):
    d = _dispatcher(monkeypatch, raises=api.PermanentFailure("bad", status=422, body="x"))
    d._handle(_payload())
    assert d.stats["dead_letter"] == 1
    assert db.dead_letter_count() == 1
    assert db.queue_count() == 0


def test_transient_failure_is_requeued(monkeypatch):
    d = _dispatcher(monkeypatch, raises=api.TransientFailure("backend down"))
    d._handle(_payload())
    assert d.stats["queued"] == 1
    assert db.queue_count() == 1
    assert db.dead_letter_count() == 0


def test_license_rejected_requeues_and_never_dead_letters(monkeypatch):
    fired = []
    d = _dispatcher(monkeypatch, raises=api.LicenseRejected("HTTP 401"),
                    on_rejected=lambda: fired.append(True))
    d._handle(_payload())
    # LicenseRejected subclasses TransientFailure but MUST be caught first:
    # re-queued, hook fired, and crucially NOT dead-lettered.
    assert d.stats["queued"] == 1
    assert db.queue_count() == 1
    assert db.dead_letter_count() == 0
    assert fired == [True]


def test_license_rejected_hook_error_is_swallowed(monkeypatch):
    def boom():
        raise RuntimeError("hook blew up")
    d = _dispatcher(monkeypatch, raises=api.LicenseRejected("401"), on_rejected=boom)
    d._handle(_payload())                 # must not raise
    assert db.queue_count() == 1


def test_submit_overflow_bypasses_to_offline_queue(monkeypatch):
    monkeypatch.setattr(api, "post_event", lambda p: True)
    d = events_module.EventDispatcher(queue_size=1)
    # Don't start the worker, so the in-process queue stays full.
    d._q.put_nowait(_payload())
    d.submit(_payload(client_event_id="evt-2"))
    assert d.stats["dropped_overflow"] == 1
    assert db.queue_count() == 1          # not dropped — persisted to sqlite
