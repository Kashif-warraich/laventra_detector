import logging
import threading
import requests
import db
import auth

log = logging.getLogger("laventra")

MAX_RETRIES    = 10
RETRY_INTERVAL = 60
_url_fails     = 0
URL_FAIL_LIMIT = 3


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

    session = auth.get_session()

    for attempt in range(2):
        try:
            r = session.post(
                f"{_v1()}/car_wash_events",
                json=payload,
                timeout=10,
            )
            if r.status_code in (200, 201):
                _url_fails = 0
                log.info(f"✅ Event posted → {plate}")
                return True
            if r.status_code == 401 and attempt == 0:
                log.warning("Token expired — re-logging in…")
                if auth.silent_relogin():
                    continue
                # Silent re-login failed — ask user interactively
                log.warning("Silent re-login failed — prompting for credentials")
                if auth.interactive_relogin():
                    continue
                return False
            if r.status_code == 422:
                log.error(f"Event rejected (422): {r.text[:200]}")
                return False
            log.error(f"POST failed: HTTP {r.status_code} — {r.text[:150]}")
            return False

        except requests.exceptions.ConnectionError:
            _url_fails += 1
            log.warning(f"API unreachable ({_url_fails}/{URL_FAIL_LIMIT}) — {plate}")
            if _url_fails >= URL_FAIL_LIMIT:
                _url_fails = 0
                auth.prompt_new_api_url()
            return False
        except requests.exceptions.Timeout:
            log.warning(f"API timeout — {plate}")
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