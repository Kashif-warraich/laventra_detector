"""
License lifecycle for the detector.

═══════════════════════════════════════════════════════════════════════════════
BACKEND CONTRACT  —  these endpoints must exist on the Rails API.
═══════════════════════════════════════════════════════════════════════════════

1) POST  /api/v1/license/activate
   Body  (JSON):
       {
         "license_code":        "LAVN-XXXXXX-XXXXXX-XXXXXX",
         "device_fingerprint":  "<sha256 hex of stable hw identifiers>",
         "hostname":            "<reported hostname>"
       }
   Auth: NONE (the license code IS the credential)
   Response 200:
       {
         "license_jwt":    "<RS256 JWT>",
         "public_key_id":  "<kid>",        # for future key rotation
         "server_time":    "2026-05-18T12:00:00.000Z"
       }
   Response 4xx:
       400 invalid format    409 already activated on another device
       410 code expired      404 unknown code

2) POST  /api/v1/license/refresh
   Body  (JSON, optional):  { "device_fingerprint": "..." }
   Auth: Authorization: Bearer <current license_jwt>
   Response 200:
       {
         "license_jwt":   "<new RS256 JWT, may be unchanged if no rotation>",
         "revoked":       false,
         "server_time":   "2026-05-18T12:00:00.000Z"
       }
   Response 401 → license revoked (admin deactivated the device)
   Response 403 → license expired and not renewable
   Response 5xx → transient failure; detector uses local JWT inside its
                  exp window + LICENSE_OFFLINE_GRACE_S

3) GET  /api/v1/cameras
   Auth: Authorization: Bearer <license_jwt>
   Response 200:
       {
         "cameras": [
           {
             "id":         42,
             "name":       "Tunnel Entry",
             "stream_url": "rtsp://10.0.0.50:554/stream1",
             "kind":       "rtsp"|"mjpeg"|"webcam",
             "metadata":   { ...arbitrary, may be null... }
           }
         ]
       }

4) POST /api/v1/car_wash_events      (auth: Bearer license_jwt)
5) POST /api/v1/devices/heartbeat    (auth: Bearer license_jwt)

═══════════════════════════════════════════════════════════════════════════════
JWT claims (RS256, signed by the backend's private key — detector verifies
against the public key bundled at config.PUBLIC_KEY_PATH):
    iss   = "laventra-backend"
    aud   = "laventra-detector"
    sub   = "<device_id as string>"
    iat   = unix seconds at issuance
    nbf   = iat
    exp   = iat + (e.g.) 30 days
    license_id           = "<uuid>"
    lavvaggio_id         = 123
    device_fingerprint   = "<sha256 hex>"   (binds license to hardware)
    features             = ["car_wash_events", "heartbeat"]
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import logging
import platform
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jwt as _jwt              # PyJWT
import requests

import config
import db

log = logging.getLogger("laventra")


# ─── Errors ─────────────────────────────────────────────────────────────────
class LicenseError(Exception):
    """Base class for all license-related failures."""


class LicenseInvalid(LicenseError):
    """JWT signature / claims invalid, or expired beyond grace."""


class LicenseRevoked(LicenseError):
    """Backend told us this license is revoked."""


class ActivationError(LicenseError):
    """Failed to activate the device with its license code."""


# ─── Device fingerprint ─────────────────────────────────────────────────────
def device_fingerprint() -> str:
    """
    Stable-ish identifier for this edge device. Hashes:
      - MAC of the primary network interface
      - machine name
      - platform (OS+arch)

    Deliberately fuzzy — a routine OS upgrade should not invalidate the license.
    If you need stronger binding (e.g. to a specific drive serial), extend here.
    """
    parts = [
        hex(uuid.getnode()),
        platform.node(),
        platform.machine(),
        platform.system(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


# ─── Public key loading ─────────────────────────────────────────────────────
_public_key_cache: bytes | None = None


def _load_public_key() -> bytes:
    """
    Read the RS256 public key. Cached after first read.

    The key file is shipped with the detector (config.PUBLIC_KEY_PATH).
    Rotation strategy: ship a new build of the detector with the new key,
    OR fetch /api/v1/license/public_keys at startup (future enhancement).
    """
    global _public_key_cache
    if _public_key_cache is not None:
        return _public_key_cache
    p = Path(config.PUBLIC_KEY_PATH)
    if not p.exists():
        raise LicenseInvalid(
            f"License public key missing at {p}. "
            "The detector cannot verify licenses without it."
        )
    _public_key_cache = p.read_bytes()
    return _public_key_cache


# ─── JWT verify ─────────────────────────────────────────────────────────────
def verify_jwt(token: str, *, check_fingerprint: bool = True) -> dict:
    """
    Validate the JWT. Raises LicenseInvalid on any failure. Returns claims dict.
    """
    try:
        key = _load_public_key()
        claims = _jwt.decode(
            token,
            key,
            algorithms=[config.JWT_ALGORITHM],
            audience=config.JWT_AUDIENCE,
            issuer=config.JWT_ISSUER,
            leeway=config.LICENSE_CLOCK_SKEW_S,
        )
    except _jwt.ExpiredSignatureError as e:
        raise LicenseInvalid(f"JWT expired: {e}") from e
    except _jwt.InvalidTokenError as e:
        raise LicenseInvalid(f"JWT invalid: {e}") from e
    except LicenseError:
        raise
    except Exception as e:
        raise LicenseInvalid(f"JWT verify error: {e}") from e

    if check_fingerprint:
        expected = device_fingerprint()
        claimed = claims.get("device_fingerprint")
        if claimed and claimed != expected:
            raise LicenseInvalid(
                "License device_fingerprint does not match this hardware. "
                "Has the license been moved between machines?"
            )
    return claims


# ─── Activation ─────────────────────────────────────────────────────────────
def activate(*, api_url: str, license_code: str) -> dict:
    """
    Exchange the license code for a signed license JWT.
    Stores license + api_url. Returns the verified claims.
    """
    url = _v1(api_url) + "/license/activate"
    fp = device_fingerprint()
    body = {
        "license_code": license_code.strip(),
        "device_fingerprint": fp,
        "hostname": hostname(),
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": config.USER_AGENT,
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=config.API_TIMEOUT_S)
    except requests.exceptions.RequestException as e:
        raise ActivationError(f"Cannot reach {api_url}: {e}") from e

    if r.status_code != 200:
        body_preview = (r.text or "")[:300]
        raise ActivationError(
            f"Activation failed (HTTP {r.status_code}): {body_preview}"
        )

    data = r.json() or {}
    # Rails wraps responses as { status, data: { ... } } — unwrap if needed
    payload = data.get("data", data) if isinstance(data.get("data"), dict) else data
    token = payload.get("license_jwt")
    if not token:
        raise ActivationError("Activation succeeded but response had no license_jwt")

    claims = verify_jwt(token)

    db.kv_set("api_url", api_url.rstrip("/"))
    # Persist the License code so the detector can silently re-activate when its
    # JWT has fully expired and a plain refresh is no longer possible (see
    # attempt_reactivate). Keeps "reconfigure only when the license expires" true
    # without an on-site visit after an admin renews. It is the same per-lavvaggio
    # license credential and lives alongside the JWT, which is already local.
    db.kv_set("license_code", license_code.strip())
    db.license_save(
        token,
        license_id=claims.get("license_id", ""),
        device_id=int(claims.get("sub") or claims.get("device_id") or 0),
        lavvaggio_id=int(claims.get("lavvaggio_id") or 0),
        issued_at=_ts_to_iso(claims.get("iat")),
        expires_at=_ts_to_iso(claims.get("exp")),
    )
    log.info(
        f"✅ License activated — device_id={claims.get('sub')} "
        f"lavvaggio_id={claims.get('lavvaggio_id')} "
        f"expires={_ts_to_iso(claims.get('exp'))}"
    )
    return claims


# ─── Self-heal via re-activation ────────────────────────────────────────────
def attempt_reactivate() -> bool:
    """
    Silently re-activate using the stored License code + api_url.

    Recovers the one case a plain refresh cannot: once the local JWT has fully
    expired, /license/refresh rejects us (our own bearer credential is expired),
    so a server-side renewal would otherwise need an on-site --activate. The
    backend accepts re-activation from the SAME device fingerprint, so this is
    safe to run unattended. Returns True on success.
    """
    code    = db.kv_get("license_code", "")
    api_url = db.kv_get("api_url", "")
    if not code or not api_url:
        return False
    try:
        activate(api_url=api_url, license_code=code)
        log.info("♻️  License self-healed via re-activation")
        return True
    except LicenseError as e:
        log.debug(f"Self-heal re-activation failed: {e}")
        return False


# ─── Boot-time revocation check ────────────────────────────────────────────
def check_revocation_online() -> None:
    """
    Boot-time revocation check. Asks the backend once if this license is
    still valid. Raises LicenseRevoked if the backend says so.
    Silently returns if the backend is offline — the JWT exp is the fallback.
    """
    lic = db.license_get()
    if not lic or not lic.get("license_jwt"):
        return
    api_url = db.kv_get("api_url", "")
    if not api_url:
        return
    headers = {
        "Authorization": f"Bearer {lic['license_jwt']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": config.USER_AGENT,
    }
    try:
        r = requests.post(
            _v1(api_url) + "/license/refresh",
            headers=headers,
            json={"device_fingerprint": device_fingerprint()},
            timeout=config.API_TIMEOUT_S,
        )
    except requests.exceptions.RequestException:
        log.debug("License revocation check: backend offline — trusting local JWT")
        return
    if r.status_code in (401, 403):
        db.license_mark_revoked()
        raise LicenseRevoked("License revoked by admin")
    if r.status_code == 200:
        data = r.json() or {}
        payload = data.get("data", data) if isinstance(data.get("data"), dict) else data
        if payload.get("revoked"):
            db.license_mark_revoked()
            raise LicenseRevoked("License revoked by admin")


# ─── Status check (boot-time) ───────────────────────────────────────────────
def ensure_valid() -> dict:
    """
    Boot-time: verify the stored JWT locally, then do a single online
    revocation check. Raises LicenseInvalid / LicenseRevoked on failure.
    Returns the verified claims.
    """
    lic = db.license_get()
    if not lic or not lic.get("license_jwt"):
        raise LicenseInvalid("No license — run: python main.py --activate CODE")
    if lic.get("revoked"):
        raise LicenseRevoked("License is revoked")
    claims = verify_jwt(lic["license_jwt"])
    check_revocation_online()
    return claims


def bearer_header() -> dict:
    """Return Authorization header for outgoing API calls. Empty if no license."""
    lic = db.license_get()
    if not lic or not lic.get("license_jwt") or lic.get("revoked"):
        return {}
    return {"Authorization": f"Bearer {lic['license_jwt']}"}


# ─── Runtime license checker ────────────────────────────────────────────────
def _try_fetch_jwt() -> str:
    """
    Attempt a license refresh against the backend.

    Returns:
        "valid"   — backend confirmed active; local JWT updated if renewed
        "invalid" — backend explicitly says inactive / revoked (401 / 403)
        "offline" — backend unreachable or 5xx; caller falls back to local JWT
    """
    lic = db.license_get()
    if not lic or not lic.get("license_jwt"):
        return "offline"
    api_url = db.kv_get("api_url", "")
    if not api_url:
        return "offline"

    headers = {
        "Authorization": f"Bearer {lic['license_jwt']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": config.USER_AGENT,
    }
    try:
        r = requests.post(
            _v1(api_url) + "/license/refresh",
            headers=headers,
            json={"device_fingerprint": device_fingerprint()},
            timeout=config.API_TIMEOUT_S,
        )
    except requests.exceptions.RequestException as e:
        log.debug(f"License refresh: backend unreachable ({e})")
        return "offline"

    if r.status_code in (401, 403):
        return "invalid"
    if r.status_code >= 500:
        return "offline"
    if r.status_code != 200:
        log.debug(f"License refresh: unexpected HTTP {r.status_code}")
        return "offline"

    # 200 — backend confirmed active; update JWT if a new one was issued
    data = r.json() or {}
    payload = data.get("data", data) if isinstance(data.get("data"), dict) else data
    if payload.get("revoked"):
        return "invalid"
    new_jwt = payload.get("license_jwt") or lic["license_jwt"]
    try:
        claims = verify_jwt(new_jwt)
        db.license_save(
            new_jwt,
            license_id=claims.get("license_id", ""),
            device_id=int(claims.get("sub") or 0),
            lavvaggio_id=int(claims.get("lavvaggio_id") or 0),
            issued_at=_ts_to_iso(claims.get("iat")),
            expires_at=_ts_to_iso(claims.get("exp")),
        )
    except LicenseInvalid as e:
        log.warning(f"License refresh: backend returned invalid JWT ({e})")
        return "offline"
    return "valid"


class LicenseChecker:
    """
    Background thread that validates the license on a schedule and exposes
    an `is_valid` flag for the main loop to gate detection.

    Normal cadence : LICENSE_CHECK_INTERVAL_S  (24 h)
    Retry cadence  : LICENSE_RETRY_INTERVAL_S  (5 min)

    State transitions
    ─────────────────
    valid → invalid  : is_valid set False  → main loop pauses detection
    invalid → valid  : is_valid set True   → main loop resumes detection

    The detector process is never killed by this thread. The main loop and
    queue flusher both read is_valid to decide whether to act.
    """

    def __init__(self):
        import threading
        self._stop  = threading.Event()
        self._wake  = threading.Event()   # cut the sleep short for an early re-check
        self._lock  = threading.Lock()
        self._valid = True   # boot check already passed — start as valid
        self._thread = None

    @property
    def is_valid(self) -> bool:
        with self._lock:
            return self._valid

    def _set_valid(self, value: bool) -> None:
        with self._lock:
            self._valid = value

    def start(self) -> None:
        import threading
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="license-checker"
        )
        self._thread.start()
        log.info("🔐 License checker started")

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()   # break the interruptible sleep so shutdown is prompt
        if self._thread:
            self._thread.join(timeout=3)

    def trigger_recheck(self) -> None:
        """
        Pause detection immediately and wake the checker to re-verify now.

        Called when a live API call returns 401/403 (LicenseRejected). Fail-safe:
        we flip to invalid first (pausing detection) and let the re-check confirm
        — so a real revocation is honoured within seconds instead of waiting up
        to LICENSE_RETRY_INTERVAL_S, and a fluke simply resumes on the next check.
        """
        self._set_valid(False)
        self._wake.set()

    def _loop(self) -> None:
        interval = config.LICENSE_CHECK_INTERVAL_S
        in_retry  = False

        # Check-then-sleep: verify immediately on start. Previously the loop slept
        # a full interval first, so up to LICENSE_CHECK_INTERVAL_S (24h) could pass
        # before the first runtime check and a mid-run revocation went unnoticed.
        while not self._stop.is_set():
            try:
                valid = self._check()
            except Exception as e:
                log.warning(f"License check error: {e}")
                valid = False

            self._set_valid(valid)

            if valid:
                if in_retry:
                    log.info("✅ License valid — detection will resume")
                    in_retry = False
                interval = config.LICENSE_CHECK_INTERVAL_S
            else:
                if not in_retry:
                    log.warning(
                        f"⚠️  License invalid — detection paused. "
                        f"Retrying every {config.LICENSE_RETRY_INTERVAL_S // 60} min. "
                        f"Detection resumes automatically once reactivated."
                    )
                    in_retry = True
                interval = config.LICENSE_RETRY_INTERVAL_S

            # Interruptible sleep: trigger_recheck() or stop() cuts it short.
            self._wake.clear()
            self._wake.wait(interval)
            if self._stop.is_set():
                break

    def _check(self) -> bool:
        """Return True if license is confirmed valid, False otherwise."""
        status = _try_fetch_jwt()

        if status == "valid":
            return True

        if status == "invalid":
            # Backend rejected our token. If our local JWT has also expired, a
            # refresh can't recover us — try a one-shot silent re-activation
            # (same fingerprint) in case the admin renewed the license. If it is
            # genuinely revoked/inactive this just fails and we stay invalid,
            # retrying on the normal 5-min cadence.
            log.warning("License check: backend says inactive or revoked — attempting self-heal")
            if attempt_reactivate():
                return True
            return False

        # Backend offline — fall back to local JWT expiry
        lic   = db.license_get() or {}
        token = lic.get("license_jwt")
        if not token:
            log.warning("License check: no local JWT and backend offline")
            return False
        try:
            verify_jwt(token)
            log.debug("License check: backend offline — local JWT still valid")
            return True
        except LicenseInvalid as e:
            log.warning(f"License check: backend offline and local JWT expired ({e})")
            return False


# ─── helpers ────────────────────────────────────────────────────────────────
def _v1(api_url: str) -> str:
    url = api_url.rstrip("/")
    if url.endswith("/api/v1"):
        return url
    return f"{url}/api/v1"


def _ts_to_iso(ts) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        ) + "Z"
    except Exception:
        return None
