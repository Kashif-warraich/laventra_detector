"""
Tests for api.post_event HTTP-status routing — which response maps to which
failure class. Getting this wrong means either losing real events (a transient
error treated as permanent → dead_letter) or looping forever (a permanent
rejection treated as transient).
"""
import requests

import pytest

import config
import db
import api


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    p = tmp_path / "api-test.db"
    monkeypatch.setattr(db, "DB_PATH", p)
    monkeypatch.setattr(config, "DB_PATH", p)
    db.init()
    db.kv_set("api_url", "http://localhost:3000")
    yield p


class FakeResp:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


def _payload():
    return {
        "plate": "AB123CD", "vehicle_type": "car",
        "started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-01T00:00:05Z",
        "confidence": 95.0, "camera_id": 1, "client_event_id": "evt-1",
    }


def _patch(monkeypatch, resp=None, raises=None):
    def fake_post(*args, **kwargs):
        if raises is not None:
            raise raises
        return resp
    monkeypatch.setattr(api.requests, "post", fake_post)


def test_200_is_success(monkeypatch):
    _patch(monkeypatch, FakeResp(200))
    assert api.post_event(_payload()) is True


def test_201_is_success(monkeypatch):
    _patch(monkeypatch, FakeResp(201))
    assert api.post_event(_payload()) is True


def test_401_raises_license_rejected(monkeypatch):
    _patch(monkeypatch, FakeResp(401))
    with pytest.raises(api.LicenseRejected):
        api.post_event(_payload())


def test_403_raises_license_rejected(monkeypatch):
    _patch(monkeypatch, FakeResp(403))
    with pytest.raises(api.LicenseRejected):
        api.post_event(_payload())


def test_422_raises_permanent_with_status(monkeypatch):
    _patch(monkeypatch, FakeResp(422, "validation failed"))
    with pytest.raises(api.PermanentFailure) as ei:
        api.post_event(_payload())
    assert ei.value.status == 422


def test_429_is_transient_not_permanent(monkeypatch):
    """Rate-limit must be retried, never dead-lettered."""
    _patch(monkeypatch, FakeResp(429, "slow down"))
    with pytest.raises(api.TransientFailure):
        api.post_event(_payload())
    # And explicitly NOT permanent (PermanentFailure is not a TransientFailure).
    _patch(monkeypatch, FakeResp(429))
    with pytest.raises(Exception) as ei:
        api.post_event(_payload())
    assert not isinstance(ei.value, api.PermanentFailure)


def test_408_is_transient(monkeypatch):
    _patch(monkeypatch, FakeResp(408))
    with pytest.raises(api.TransientFailure):
        api.post_event(_payload())


def test_500_is_transient(monkeypatch):
    _patch(monkeypatch, FakeResp(500, "boom"))
    with pytest.raises(api.TransientFailure):
        api.post_event(_payload())


def test_connection_error_is_transient(monkeypatch):
    _patch(monkeypatch, raises=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(api.TransientFailure):
        api.post_event(_payload())


def test_timeout_is_transient(monkeypatch):
    _patch(monkeypatch, raises=requests.exceptions.Timeout("slow"))
    with pytest.raises(api.TransientFailure):
        api.post_event(_payload())
