"""
Tests for cameras.py — the camera registry: fetch + normalise from the backend,
local cache, and active-camera selection.
"""
import requests

import pytest

import config
import db
import cameras as cameras_module


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    p = tmp_path / "cameras-test.db"
    monkeypatch.setattr(db, "DB_PATH", p)
    monkeypatch.setattr(config, "DB_PATH", p)
    db.init()
    db.kv_set("api_url", "http://localhost:3000/api/v1")
    yield p


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self):
        return self._payload


def _patch_get(monkeypatch, resp=None, raises=None):
    def fake_get(*args, **kwargs):
        if raises is not None:
            raise raises
        return resp
    monkeypatch.setattr(cameras_module.requests, "get", fake_get)


# ─── fetch_remote ───────────────────────────────────────────────────────────
def test_fetch_remote_normalises_and_persists(monkeypatch):
    _patch_get(monkeypatch, FakeResp(200, {"cameras": [
        {"id": 1, "name": "Entry", "stream_url": "rtsp://x/1", "kind": "rtsp"},
        {"id": "2", "stream_url": "http://x/2"},     # id as string, name omitted
    ]}))
    cams = cameras_module.fetch_remote()
    by_id = {c["id"]: c for c in cams}
    assert by_id[1]["name"] == "Entry"
    assert by_id[2]["name"] == "Camera 2"            # default name when missing
    assert len(db.cameras_list()) == 2               # persisted to local cache


def test_fetch_remote_skips_cameras_missing_id_or_url(monkeypatch):
    _patch_get(monkeypatch, FakeResp(200, {"cameras": [
        {"id": 1, "stream_url": "rtsp://x/1"},
        {"id": 2},                                   # no stream_url → skip
        {"stream_url": "rtsp://x/9"},                # no id → skip
    ]}))
    cams = cameras_module.fetch_remote()
    assert [c["id"] for c in cams] == [1]


def test_fetch_remote_accepts_data_envelope(monkeypatch):
    _patch_get(monkeypatch, FakeResp(200, {"data": [{"id": 3, "stream_url": "u3"}]}))
    assert [c["id"] for c in cameras_module.fetch_remote()] == [3]


def test_fetch_remote_401_raises_fetch_error(monkeypatch):
    _patch_get(monkeypatch, FakeResp(401))
    with pytest.raises(cameras_module.CameraFetchError):
        cameras_module.fetch_remote()


def test_fetch_remote_5xx_raises_fetch_error(monkeypatch):
    _patch_get(monkeypatch, FakeResp(503))
    with pytest.raises(cameras_module.CameraFetchError):
        cameras_module.fetch_remote()


def test_fetch_remote_network_error_raises_fetch_error(monkeypatch):
    _patch_get(monkeypatch, raises=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(cameras_module.CameraFetchError):
        cameras_module.fetch_remote()


# ─── select / selected ────────────────────────────────────────────────────────
def test_select_sets_active_camera_and_url():
    db.cameras_replace([{"id": 5, "name": "Exit", "stream_url": "rtsp://x/5"}])
    cam = cameras_module.select(5)
    assert cam["id"] == 5
    assert db.kv_get("camera_id") == "5"
    assert db.kv_get("camera_url") == "rtsp://x/5"
    assert cameras_module.selected()["id"] == 5


def test_select_unknown_camera_raises():
    with pytest.raises(ValueError):
        cameras_module.select(999)


def test_selected_is_none_when_unset():
    assert cameras_module.selected() is None
