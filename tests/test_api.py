"""
Tests for the camera-URL-on-heartbeat sync.

Covers the three runtime pieces that let a portal-side stream_url change
propagate to a running detector without anyone touching the edge device:

  - api.send_heartbeat        — parses the camera config echoed in the response
  - api.HeartbeatThread._sync_camera — applies it only for the bound camera
  - camera.CameraStream.set_url       — swaps the live stream + persists the URL
"""
import requests

import pytest

import config
import db
import api
import camera as camera_module


# ─── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect db to a per-test temp file (set_url + _v1 both touch the DB)."""
    p = tmp_path / "laventra-test.db"
    monkeypatch.setattr(db, "DB_PATH", p)
    monkeypatch.setattr(config, "DB_PATH", p)
    db.init()
    db.kv_set("api_url", "http://localhost:3000")
    yield p


class FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code=200, payload=None, raise_json=False):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._raise_json = raise_json
        self.text = ""

    def json(self):
        if self._raise_json:
            raise ValueError("not JSON")
        return self._payload


def _patch_post(monkeypatch, resp=None, raises=None):
    def fake_post(*args, **kwargs):
        if raises is not None:
            raise raises
        return resp
    monkeypatch.setattr(api.requests, "post", fake_post)


# ─── send_heartbeat: response parsing ─────────────────────────────────────────
def test_send_heartbeat_returns_camera_from_data_envelope(monkeypatch):
    cam = {"id": 1, "stream_url": "rtsp://10.0.0.50/1", "kind": "rtsp"}
    _patch_post(monkeypatch, FakeResp(200, {
        "status": "true",
        "data": {"last_seen_at": "2026-06-16T00:00:00Z", "camera": cam},
    }))
    assert api.send_heartbeat(camera_id=1, camera_online=True) == cam


def test_send_heartbeat_returns_camera_when_no_envelope(monkeypatch):
    cam = {"id": 9, "stream_url": "http://10.0.0.9/video", "kind": "mjpeg"}
    _patch_post(monkeypatch, FakeResp(200, {"camera": cam}))
    assert api.send_heartbeat(camera_id=9) == cam


def test_send_heartbeat_returns_none_when_no_camera(monkeypatch):
    _patch_post(monkeypatch, FakeResp(200, {"data": {"last_seen_at": "x"}}))
    assert api.send_heartbeat(camera_id=1) is None


def test_send_heartbeat_returns_none_on_non_200(monkeypatch):
    _patch_post(monkeypatch, FakeResp(500, {"camera": {"id": 1, "stream_url": "u"}}))
    assert api.send_heartbeat(camera_id=1) is None


def test_send_heartbeat_returns_none_on_network_error(monkeypatch):
    _patch_post(monkeypatch, raises=requests.exceptions.ConnectionError("down"))
    assert api.send_heartbeat(camera_id=1) is None


def test_send_heartbeat_returns_none_on_bad_json(monkeypatch):
    _patch_post(monkeypatch, FakeResp(200, raise_json=True))
    assert api.send_heartbeat(camera_id=1) is None


def test_send_heartbeat_ignores_non_dict_camera(monkeypatch):
    _patch_post(monkeypatch, FakeResp(200, {"data": {"camera": "not-a-dict"}}))
    assert api.send_heartbeat(camera_id=1) is None


# ─── HeartbeatThread._sync_camera: which changes get applied ──────────────────
def _thread(camera_id, applied):
    return api.HeartbeatThread(
        camera_id=camera_id,
        on_camera_url=lambda url: applied.append(url),
    )


def test_sync_camera_applies_matching_camera():
    applied = []
    _thread(5, applied)._sync_camera({"id": 5, "stream_url": "rtsp://new/5"})
    assert applied == ["rtsp://new/5"]


def test_sync_camera_skips_id_mismatch():
    applied = []
    _thread(5, applied)._sync_camera({"id": 7, "stream_url": "rtsp://other/7"})
    assert applied == []


def test_sync_camera_applies_when_bound_id_unset():
    applied = []
    _thread(None, applied)._sync_camera({"id": 7, "stream_url": "rtsp://x/7"})
    assert applied == ["rtsp://x/7"]


def test_sync_camera_skips_when_stream_url_missing():
    applied = []
    _thread(5, applied)._sync_camera({"id": 5})
    assert applied == []


def test_sync_camera_skips_when_none_or_no_callback():
    applied = []
    _thread(5, applied)._sync_camera(None)
    api.HeartbeatThread(camera_id=5, on_camera_url=None)._sync_camera(
        {"id": 5, "stream_url": "rtsp://x/5"}
    )
    assert applied == []


# ─── CameraStream.set_url: live swap + persistence ────────────────────────────
def test_set_url_swaps_and_persists():
    s = camera_module.CameraStream("http://10.0.0.5/video")
    s.set_url("rtsp://10.0.0.99/stream")
    assert s.url == "rtsp://10.0.0.99/stream"
    assert db.kv_get("camera_url") == "rtsp://10.0.0.99/stream"


def test_set_url_updates_staleness_ceiling_for_scheme():
    s = camera_module.CameraStream("http://10.0.0.5/video")
    assert s._max_age_default == config.CAMERA_FRAME_MAX_AGE_MJPEG_S
    s.set_url("rtsp://10.0.0.99/stream")
    assert s._max_age_default == config.CAMERA_FRAME_MAX_AGE_RTSP_S


def test_set_url_noop_when_unchanged():
    s = camera_module.CameraStream("rtsp://same/1")
    s.set_url("rtsp://same/1")
    assert db.kv_get("camera_url") is None      # nothing persisted


def test_set_url_noop_when_blank():
    s = camera_module.CameraStream("rtsp://keep/1")
    s.set_url("   ")
    assert s.url == "rtsp://keep/1"
    assert db.kv_get("camera_url") is None
