import logging
import threading
import time
import requests
import db
import auth

log = logging.getLogger("laventra")

MAX_RETRIES    = 10
RETRY_INTERVAL = 60
_url_fails     = 0
URL_FAIL_LIMIT = 3

# Exponential backoff settings for post_event retries
_BACKOFF_INITIAL = 5
_BACKOFF_MAX     = 60
_BACKOFF_RETRIES = 5


def _v1() -> str:
    url = db.session_get("api_url", "").rstrip("/")
    if url.endswith("/api/v1"):
        url = url[: -len("/api/v1")]
    return f"{url}/api/v1"


def post_event(
    lavvaggio_id: int,
    plate: str,
    vehicle_type: str,
    started_at: str,
    ended_at: str,
    device_id: int = None,
) -> bool:
    global _url_fails

    # Device-token auth: the backend infers lavvaggio_id/device_id from the token,
    # but we include them for symmetry with user-auth'd code paths.
    payload = {
        "car_wash_event": {
            "lavvaggio_id":  lavvaggio_id,
            "vehicle_plate": plate,
            "vehicle_type":  vehicle_type,
            "started_at":    started_at,
            "ended_at":      ended_at,
        }
    }
    if device_id:
        payload["car_wash_event"]["device_id"] = device_id

    # Always attach the persistent device token. We never re-login — if the token
    # is rejected, it means the device was revoked and only --setup can fix it.
    device_token = db.session_get("device_token", "")
    headers = {"Authorization": f"Bearer {device_token}"} if device_token else {}
    session = auth.get_session()

    for attempt in range(1):
        try:
            r = session.post(
                f"{_v1()}/car_wash_events",
                json=payload,
                headers=headers,
                timeout=10,
            )
            if r.status_code in (200, 201):
                _url_fails = 0
                log.info(f"✅ Event posted → {plate}")
                return True
            if r.status_code == 401:
                log.error("Device token rejected — run: python main.py --setup")
                return False
            if r.status_code == 422:
                log.error(f"Event rejected (422): {r.text[:200]}")
                return False
            log.error(f"POST failed: HTTP {r.status_code} — {r.text[:150]}")
            return False

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            exc_type = "timeout" if isinstance(exc, requests.exceptions.Timeout) else "unreachable"
            backoff_delay = _BACKOFF_INITIAL
            for retry in range(1, _BACKOFF_RETRIES + 1):
                log.warning(
                    f"API {exc_type} — {plate} — retrying in {backoff_delay}s "
                    f"({retry}/{_BACKOFF_RETRIES})"
                )
                time.sleep(backoff_delay)
                try:
                    r = session.post(
                        f"{_v1()}/car_wash_events",
                        json=payload,
                        headers=headers,
                        timeout=10,
                    )
                    if r.status_code in (200, 201):
                        _url_fails = 0
                        log.info(f"✅ Event posted → {plate} (after {retry} retries)")
                        return True
                    log.error(f"POST failed on retry: HTTP {r.status_code}")
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    backoff_delay = min(backoff_delay * 2, _BACKOFF_MAX)
                except Exception as e:
                    log.error(f"POST retry error ({plate}): {e}")
                    break
            _url_fails += 1
            log.warning(f"API unreachable after retries ({_url_fails}/{URL_FAIL_LIMIT}) — {plate}")
            if _url_fails >= URL_FAIL_LIMIT:
                _url_fails = 0
                auth.prompt_new_api_url()
            return False
        except Exception as e:
            log.error(f"POST error ({plate}): {e}")
            return False

    return False


class QueueFlusher:
    def __init__(self):
        self._stop   = threading.Event()
        self._thread = None

    def start(self) -> None:
        count = db.queue_count(MAX_RETRIES)
        if count:
            log.info(f"📥 {count} offline event(s) pending retry")
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="queue-flusher"
        )
        self._thread.start()
        log.info("🔄 Queue flusher started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(RETRY_INTERVAL)
            if self._stop.is_set():
                break
            self._flush()

    def _flush(self) -> None:
        rows = db.pending(MAX_RETRIES)
        if not rows:
            return
        log.info(f"🔄 Retrying {len(rows)} queued event(s)…")
        sent = 0
        for row in rows:
            ok = post_event(
                lavvaggio_id=row["lavvaggio_id"],
                plate=row["plate"],
                vehicle_type=row["vehicle_type"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                device_id=row["device_id"],
            )
            if ok:
                db.mark_sent(row["id"])
                sent += 1
            else:
                db.mark_failed(row["id"])
        if sent:
            log.info(f"✅ Flushed {sent}/{len(rows)} event(s)")
        else:
            log.warning(f"Flush: 0/{len(rows)} sent — API still unreachable")


# ── Heartbeat ────────────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL = 60  # seconds


def send_heartbeat(device_id: int, device_token: str) -> bool:
    """
    POST /api/v1/devices/{device_id}/heartbeat with the device token.
    Non-fatal: logs at debug level on success, warning on failure, never raises.
    """
    if not device_id or not device_token:
        log.debug("heartbeat skipped — missing device_id or device_token")
        return False

    url = f"{_v1()}/devices/{device_id}/heartbeat"
    headers = {"Authorization": f"Bearer {device_token}"}
    try:
        r = requests.post(url, headers=headers, timeout=8)
        if r.status_code == 200:
            log.debug(f"💓 heartbeat ok (device_id={device_id})")
            return True
        log.warning(f"heartbeat failed: HTTP {r.status_code} — {r.text[:120]}")
        return False
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        log.debug("heartbeat: API unreachable")
        return False
    except Exception as e:
        log.warning(f"heartbeat error: {e}")
        return False


class HeartbeatThread:
    """
    Background thread that POSTs a heartbeat to the backend every HEARTBEAT_INTERVAL
    seconds. Failures are non-fatal. Stopped cleanly on shutdown.
    """

    def __init__(self, device_id: int, device_token: str):
        self._device_id    = device_id
        self._device_token = device_token
        self._stop         = threading.Event()
        self._thread       = None

    def start(self) -> None:
        if not self._device_id or not self._device_token:
            log.warning("Heartbeat disabled — no device_id or device_token")
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="heartbeat"
        )
        self._thread.start()
        log.info("💓 Heartbeat started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        # Send one immediately so the backend knows we're online
        send_heartbeat(self._device_id, self._device_token)
        while not self._stop.is_set():
            self._stop.wait(HEARTBEAT_INTERVAL)
            if self._stop.is_set():
                break
            send_heartbeat(self._device_id, self._device_token)