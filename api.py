"""
HTTP client for the Rails backend, authorised by the device license JWT.

All calls attach Authorization: Bearer <license_jwt>. There is no relogin,
no password, no recursion. If the license is revoked the backend returns 401
and we surface a `LicenseRevoked` upstream — the operator must re-activate.
"""
from __future__ import annotations

import logging
import threading
import time

import requests

import config
import db
import license as license_module

log = logging.getLogger("laventra")


# ─── Errors ─────────────────────────────────────────────────────────────────
class PermanentFailure(Exception):
    """Backend rejected the payload with 4xx — retrying will not help.

    Replaces the old `_PERMANENT_FAILURE = "PERMANENT"` string sentinel
    (audit finding H7). Carries the HTTP status + response body so callers
    can route to the dead_letter table.
    """
    def __init__(self, message: str, *, status: int, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class TransientFailure(Exception):
    """Backend unreachable / 5xx — caller should queue and retry later."""


# ─── Helpers ────────────────────────────────────────────────────────────────
def _v1() -> str:
    url = db.kv_get("api_url", "").rstrip("/")
    if url.endswith("/api/v1"):
        return url
    return f"{url}/api/v1"


def _headers() -> dict:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": config.USER_AGENT,
    }
    h.update(license_module.bearer_header())
    return h


# ─── Events ─────────────────────────────────────────────────────────────────
def post_event(payload: dict) -> bool:
    """
    POST one car_wash_event. Returns True on success.

    Raises PermanentFailure on 401 / 422 / other 4xx (caller writes dead_letter).
    Raises TransientFailure on network / 5xx (caller writes queue).
    """
    body = {"car_wash_event": _strip_none({
        "vehicle_plate": payload["plate"],
        "vehicle_type":  payload.get("vehicle_type"),
        "started_at":    payload["started_at"],
        "ended_at":      payload["ended_at"],
        "confidence":    _round_or_none(payload.get("confidence"), 2),
        "camera_id":     payload.get("camera_id"),
    })}
    url = f"{_v1()}/car_wash_events"
    try:
        r = requests.post(url, json=body, headers=_headers(),
                          timeout=config.API_TIMEOUT_S)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise TransientFailure(f"Backend unreachable: {e}") from e
    except requests.exceptions.RequestException as e:
        raise TransientFailure(f"Request error: {e}") from e

    if r.status_code in (200, 201):
        return True
    if r.status_code == 401:
        db.license_mark_revoked()
        raise PermanentFailure(
            "License rejected by backend",
            status=401, body=r.text[:400],
        )
    if 400 <= r.status_code < 500:
        raise PermanentFailure(
            f"Backend rejected event: HTTP {r.status_code}",
            status=r.status_code, body=r.text[:400],
        )
    raise TransientFailure(f"Backend HTTP {r.status_code}: {r.text[:200]}")


# ─── Heartbeat ──────────────────────────────────────────────────────────────
def send_heartbeat(*, camera_id=None, camera_online: bool = None,
                   queue_depth: int = None) -> bool:
    body = _strip_none({
        "version":       config.VERSION,
        "camera_id":     camera_id,
        "camera_online": camera_online,
        "queue_depth":   queue_depth,
    })
    try:
        r = requests.post(
            f"{_v1()}/devices/heartbeat",
            json=body if body else None,
            headers=_headers(),
            timeout=config.API_TIMEOUT_S,
        )
    except requests.exceptions.RequestException as e:
        log.debug(f"heartbeat: {e}")
        return False

    if r.status_code == 200:
        return True
    if r.status_code == 401:
        # License revoked — escalate (audit H5)
        db.license_mark_revoked()
        log.error("❌ Heartbeat: license rejected — backend revoked this device")
        return False
    log.warning(f"heartbeat failed: HTTP {r.status_code} — {r.text[:120]}")
    return False


class HeartbeatThread:
    """
    Periodic heartbeat. On 401 (license revoked) fires `on_revoked` once and exits.
    """

    def __init__(self, *, camera_status_fn=None, camera_id=None, on_revoked=None):
        self._cam_status = camera_status_fn or (lambda: True)
        self._camera_id = camera_id
        self._on_revoked = on_revoked
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat")
        self._thread.start()
        log.info("💓 Heartbeat started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        send_heartbeat(
            camera_id=self._camera_id,
            camera_online=bool(self._cam_status()),
            queue_depth=db.queue_count(),
        )
        while not self._stop.is_set():
            self._stop.wait(config.HEARTBEAT_INTERVAL_S)
            if self._stop.is_set():
                break
            try:
                ok = send_heartbeat(
                    camera_id=self._camera_id,
                    camera_online=bool(self._cam_status()),
                    queue_depth=db.queue_count(),
                )
                # Detect revocation: license was marked revoked by send_heartbeat()
                if not ok:
                    lic = db.license_get() or {}
                    if lic.get("revoked") and self._on_revoked:
                        try:
                            self._on_revoked()
                        except Exception as e:
                            log.error(f"on_revoked callback error: {e}")
                        return
            except Exception as e:
                log.warning(f"heartbeat loop error: {e}")


# ─── Offline queue flusher ──────────────────────────────────────────────────
class QueueFlusher:
    """
    Periodically retries events from the local sqlite queue.
    Routes PermanentFailure → dead_letter, TransientFailure → bumps retry_count.
    """

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        n = db.queue_count()
        if n:
            log.info(f"📥 {n} offline event(s) pending retry")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="queue-flusher")
        self._thread.start()
        log.info("🔄 Queue flusher started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(config.QUEUE_FLUSH_INTERVAL_S)
            if self._stop.is_set():
                break
            self._flush()

    def _flush(self) -> None:
        rows = db.pending()
        if not rows:
            return
        log.info(f"🔄 Retrying {len(rows)} queued event(s)…")
        sent = dropped = failed = 0
        for row in rows:
            payload = {
                "plate":        row["plate"],
                "vehicle_type": row["vehicle_type"],
                "started_at":   row["started_at"],
                "ended_at":     row["ended_at"],
                "confidence":   row["confidence"],
                "camera_id":    row["camera_id"],
            }
            try:
                post_event(payload)
                db.mark_sent(row["id"])
                sent += 1
            except PermanentFailure as pf:
                db.dead_letter_add(payload=payload,
                                   error_code=pf.status, error_body=pf.body)
                db.mark_sent(row["id"])
                dropped += 1
            except TransientFailure:
                db.mark_failed(row["id"])
                failed += 1
        if sent or dropped:
            msg = f"✅ Flushed {sent}/{len(rows)}"
            if dropped:
                msg += f"  dead-lettered {dropped}"
            log.info(msg)
        elif failed:
            log.warning(f"Flush: 0/{len(rows)} sent — backend still unreachable")


# ─── helpers ────────────────────────────────────────────────────────────────
def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _round_or_none(v, ndigits):
    return round(float(v), ndigits) if v is not None else None
