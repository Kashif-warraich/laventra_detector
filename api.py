"""
HTTP client for the Rails backend, authorised by the device license JWT.

All calls attach Authorization: Bearer <license_jwt>. There is no relogin,
no password, no recursion. If the license is revoked the backend returns 401
and we surface a `LicenseRevoked` upstream — the operator must re-activate.
"""
from __future__ import annotations

import logging
import random
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


class LicenseRejected(TransientFailure):
    """
    Backend rejected the credential (401/403) — the license is inactive,
    expired, or revoked.

    This is a *transient* failure, NOT permanent: the event is re-queued (never
    dead-lettered) so a temporary license lapse (billing hiccup, admin toggling
    a license, a brief expiry before renewal) loses zero washed-car events. A
    genuinely permanent revocation is handled separately — detection pauses (the
    LicenseChecker flips to invalid) and, if the queue keeps failing, events
    eventually age out to dead_letter via QUEUE_MAX_RETRIES, where they're
    visible instead of silently dropped.
    """


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
        "vehicle_plate":   payload["plate"],
        "vehicle_type":    payload.get("vehicle_type"),
        "started_at":      payload["started_at"],
        "ended_at":        payload["ended_at"],
        "confidence":      _round_or_none(payload.get("confidence"), 2),
        "camera_id":       payload.get("camera_id"),
        "client_event_id": payload.get("client_event_id"),
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
    # 401/403 = credential/license problem → TRANSIENT (re-queue + pause), never
    # dead_letter. The license may be temporarily inactive/expired; dropping the
    # event permanently here would lose a real washed car.
    if r.status_code in (401, 403):
        raise LicenseRejected(f"License rejected by backend (HTTP {r.status_code})")
    # 408 Request Timeout / 429 Too Many Requests are throttling/transient, not a
    # bad payload — retrying later WILL help, so queue (never dead_letter) so a
    # rate-limited burst doesn't permanently drop real washed-car events.
    if r.status_code in (408, 429):
        raise TransientFailure(f"Backend throttled (HTTP {r.status_code}) — will retry")
    # Other 4xx (notably 422 validation) = the payload itself is unacceptable;
    # retrying will not help → permanent → dead_letter for operator inspection.
    if 400 <= r.status_code < 500:
        raise PermanentFailure(
            f"Backend rejected event: HTTP {r.status_code}",
            status=r.status_code, body=r.text[:400],
        )
    raise TransientFailure(f"Backend HTTP {r.status_code}: {r.text[:200]}")


# ─── Heartbeat ──────────────────────────────────────────────────────────────
def send_heartbeat(*, camera_id=None, camera_online: bool = None,
                   queue_depth: int = None) -> dict | None:
    """
    POST a heartbeat. Returns the camera config the backend echoes back
    ({"id", "stream_url", "kind"}) so the caller can sync a portal-side
    stream_url change, or None on failure / when no camera config is returned.
    """
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
        return None

    if r.status_code != 200:
        log.warning(f"heartbeat: HTTP {r.status_code}")
        return None

    try:
        data = r.json() or {}
        payload = data.get("data", data) if isinstance(data.get("data"), dict) else data
        cam = payload.get("camera")
        return cam if isinstance(cam, dict) else None
    except Exception as e:
        log.debug(f"heartbeat: could not parse response ({e})")
        return None


class HeartbeatThread:
    """
    Periodic heartbeat. Sends device status to the backend on a regular cadence.
    """

    def __init__(self, *, camera_status_fn=None, camera_id=None,
                 on_camera_url=None):
        self._cam_status = camera_status_fn or (lambda: True)
        self._camera_id = camera_id
        # Called with the backend's current stream_url when it differs from what
        # we're streaming — lets a portal-side IP change propagate on the pulse.
        self._on_camera_url = on_camera_url
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

    def _sync_camera(self, camera: dict | None) -> None:
        """Apply a backend-echoed stream_url for the camera we're bound to."""
        if not camera or not self._on_camera_url:
            return
        try:
            url = camera.get("stream_url")
            if not url:
                return
            cid = camera.get("id")
            # Only act on the camera this detector is streaming from.
            if (self._camera_id is not None and cid is not None
                    and int(cid) != int(self._camera_id)):
                return
            self._on_camera_url(url)
        except Exception as e:
            log.debug(f"heartbeat camera sync skipped: {e}")

    def _loop(self) -> None:
        self._sync_camera(send_heartbeat(
            camera_id=self._camera_id,
            camera_online=bool(self._cam_status()),
            queue_depth=db.queue_count(),
        ))
        while not self._stop.is_set():
            # Jitter so a fleet of detectors doesn't heartbeat in lockstep.
            self._stop.wait(config.HEARTBEAT_INTERVAL_S
                            + random.uniform(0, config.PULSE_JITTER_S))
            if self._stop.is_set():
                break
            try:
                self._sync_camera(send_heartbeat(
                    camera_id=self._camera_id,
                    camera_online=bool(self._cam_status()),
                    queue_depth=db.queue_count(),
                ))
            except Exception as e:
                log.warning(f"heartbeat loop error: {e}")


# ─── Offline queue flusher ──────────────────────────────────────────────────
class QueueFlusher:
    """
    Periodically retries events from the local sqlite queue.

    Routing:
      success            → delete row (sent)
      PermanentFailure   → dead_letter (422 validation; un-retryable payload)
      LicenseRejected    → stop the batch (paused); row stays for next cycle
      TransientFailure   → bump retry_count; on the final allowed attempt the
                           row is moved to dead_letter instead of being stranded.

    `license_ok_fn` (optional) lets the flusher skip entirely while the license
    is known-invalid, so it doesn't burn retry_count against a backend that is
    deliberately rejecting us during a pause.
    """

    def __init__(self, *, license_ok_fn=None):
        self._stop = threading.Event()
        self._thread = None
        self._license_ok_fn = license_ok_fn

    def start(self) -> None:
        # Clean up any rows that already exhausted their retries in a prior run.
        db.sweep_exhausted()
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
            # Jitter so a fleet of detectors doesn't flush in lockstep.
            self._stop.wait(config.QUEUE_FLUSH_INTERVAL_S
                            + random.uniform(0, config.PULSE_JITTER_S))
            if self._stop.is_set():
                break
            self._flush()

    def _flush(self) -> None:
        # Skip while the license is known-invalid (avoid burning retries during
        # a pause). Checked via the live LicenseChecker flag when provided…
        if self._license_ok_fn is not None and not self._license_ok_fn():
            log.debug("Queue flush skipped — license paused")
            return
        # …and as a fallback, against the locally-stored JWT's own validity.
        lic = db.license_get() or {}
        token = lic.get("license_jwt")
        if not token:
            return
        try:
            license_module.verify_jwt(token)
        except license_module.LicenseInvalid:
            log.debug("Queue flush skipped — local JWT invalid")
            return

        rows = db.pending()
        if not rows:
            return
        log.info(f"🔄 Retrying {len(rows)} queued event(s)…")
        sent = dead = failed = 0
        for row in rows:
            payload = {
                "plate":           row["plate"],
                "vehicle_type":    row["vehicle_type"],
                "started_at":      row["started_at"],
                "ended_at":        row["ended_at"],
                "confidence":      row["confidence"],
                "camera_id":       row["camera_id"],
                "client_event_id": row["client_event_id"],
            }
            try:
                post_event(payload)
                db.mark_sent(row["id"])
                sent += 1
            except LicenseRejected:
                # License flipped to invalid mid-flush — stop hammering, keep the
                # row, let the next cycle (or a reactivation) handle it.
                log.debug("Queue flush paused — license rejected mid-batch")
                break
            except PermanentFailure as pf:
                db.dead_letter_add(payload=payload,
                                   error_code=pf.status, error_body=pf.body)
                db.mark_sent(row["id"])
                dead += 1
            except TransientFailure:
                # On the last allowed attempt, dead_letter rather than strand it.
                if (row["retry_count"] or 0) + 1 >= config.QUEUE_MAX_RETRIES:
                    db.dead_letter_add(payload=payload, error_code=0,
                                       error_body=f"retries exhausted (>= {config.QUEUE_MAX_RETRIES})")
                    db.mark_sent(row["id"])
                    dead += 1
                else:
                    db.mark_failed(row["id"])
                    failed += 1
        if sent or dead:
            msg = f"✅ Flushed {sent}/{len(rows)}"
            if dead:
                msg += f"  dead-lettered {dead}"
            log.info(msg)
        elif failed:
            log.warning(f"Flush: 0/{len(rows)} sent — backend still unreachable")


# ─── helpers ────────────────────────────────────────────────────────────────
def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _round_or_none(v, ndigits):
    return round(float(v), ndigits) if v is not None else None
