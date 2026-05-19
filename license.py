"""
License lifecycle for the detector.

═══════════════════════════════════════════════════════════════════════════════
BACKEND CONTRACT  —  these endpoints must exist on the Rails API.
═══════════════════════════════════════════════════════════════════════════════

1) POST  /api/v1/license/activate
   Body  (JSON):
       {
         "activation_code":     "XXXX-YYYY-ZZZZ-AAAA",
         "device_fingerprint":  "<sha256 hex of stable hw identifiers>",
         "hostname":            "<reported hostname>"
       }
   Auth: NONE (the activation code IS the credential)
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
import time
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
    """Failed to exchange activation code for a license."""


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
def activate(*, api_url: str, activation_code: str) -> dict:
    """
    Exchange an activation code for a signed license JWT.
    Stores license + api_url. Returns the verified claims.
    """
    url = _v1(api_url) + "/license/activate"
    fp = device_fingerprint()
    body = {
        "activation_code": activation_code.strip(),
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


# ─── Refresh ────────────────────────────────────────────────────────────────
def refresh() -> dict | None:
    """
    Attempt to refresh the stored license. Returns claims on success, None if
    the backend is unreachable (caller falls back to LICENSE_OFFLINE_GRACE_S).
    Raises LicenseRevoked if the backend explicitly revoked us.
    """
    lic = db.license_get()
    if not lic or not lic.get("license_jwt"):
        raise LicenseInvalid("No license stored")
    api_url = db.kv_get("api_url", "")
    if not api_url:
        raise LicenseInvalid("No api_url stored")

    headers = {
        "Authorization": f"Bearer {lic['license_jwt']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": config.USER_AGENT,
    }
    body = {"device_fingerprint": device_fingerprint()}
    try:
        r = requests.post(
            _v1(api_url) + "/license/refresh",
            headers=headers,
            json=body,
            timeout=config.API_TIMEOUT_S,
        )
    except requests.exceptions.RequestException as e:
        log.debug(f"License refresh: backend unreachable ({e}) — staying on grace")
        return None

    if r.status_code == 401:
        db.license_mark_revoked()
        raise LicenseRevoked("Backend revoked this license")
    if r.status_code == 403:
        db.license_mark_revoked()
        raise LicenseRevoked("License non-renewable (expired admin-side)")
    if r.status_code >= 500:
        log.warning(f"License refresh: backend {r.status_code} — staying on grace")
        return None
    if r.status_code != 200:
        log.warning(f"License refresh: HTTP {r.status_code} {r.text[:200]} — staying on grace")
        return None

    data = r.json() or {}
    # Rails wraps responses as { status, data: { ... } } — unwrap if needed
    payload = data.get("data", data) if isinstance(data.get("data"), dict) else data
    if payload.get("revoked"):
        db.license_mark_revoked()
        raise LicenseRevoked("Refresh returned revoked=true")
    new_jwt = payload.get("license_jwt") or lic["license_jwt"]
    claims = verify_jwt(new_jwt)
    db.license_save(
        new_jwt,
        license_id=claims.get("license_id", ""),
        device_id=int(claims.get("sub") or 0),
        lavvaggio_id=int(claims.get("lavvaggio_id") or 0),
        issued_at=_ts_to_iso(claims.get("iat")),
        expires_at=_ts_to_iso(claims.get("exp")),
    )
    db.license_mark_refresh()
    log.debug(f"License refreshed — exp={_ts_to_iso(claims.get('exp'))}")
    return claims


# ─── Status check (boot-time) ───────────────────────────────────────────────
def ensure_valid() -> dict:
    """
    Boot-time: load + verify the stored JWT. Within grace window, do not require
    network. Outside grace, force refresh. Raises LicenseInvalid / LicenseRevoked.

    Returns the verified claims.
    """
    lic = db.license_get()
    if not lic or not lic.get("license_jwt"):
        raise LicenseInvalid("No license — run: python main.py --activate CODE")
    if lic.get("revoked"):
        raise LicenseRevoked("License is revoked")
    token = lic["license_jwt"]
    claims = verify_jwt(token)

    last_refresh = lic.get("last_refresh_at")
    needs_online = False
    if not last_refresh:
        needs_online = True
    else:
        try:
            ts = datetime.fromisoformat(last_refresh.replace("Z", "+00:00")).timestamp()
            if time.time() - ts > config.LICENSE_OFFLINE_GRACE_S:
                needs_online = True
        except Exception:
            needs_online = True

    if needs_online:
        log.info("License grace exceeded — refresh required")
        new_claims = refresh()
        if new_claims is not None:
            return new_claims
        raise LicenseInvalid(
            "Cannot reach backend to refresh license, and offline grace exceeded. "
            "Check network or re-activate."
        )
    return claims


def bearer_header() -> dict:
    """Return Authorization header for outgoing API calls. Empty if no license."""
    lic = db.license_get()
    if not lic or not lic.get("license_jwt") or lic.get("revoked"):
        return {}
    return {"Authorization": f"Bearer {lic['license_jwt']}"}


# ─── Background refresher ───────────────────────────────────────────────────
class LicenseRefresher:
    """
    Background thread: refreshes the license once every LICENSE_REFRESH_INTERVAL_S.
    On LicenseRevoked → invokes `on_revoked` callback (typically: shut down).
    """

    def __init__(self, on_revoked=None):
        import threading
        self._on_revoked = on_revoked
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        import threading
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="license-refresher"
        )
        self._thread.start()
        log.info("🔐 License refresher started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        # Refresh once shortly after startup, then on the regular cadence.
        self._stop.wait(60)
        while not self._stop.is_set():
            try:
                refresh()
            except LicenseRevoked as e:
                log.error(f"❌ License revoked: {e}")
                if self._on_revoked:
                    try:
                        self._on_revoked()
                    except Exception as cb_e:
                        log.error(f"on_revoked callback error: {cb_e}")
                return
            except LicenseError as e:
                log.warning(f"License refresh problem: {e}")
            except Exception as e:
                log.warning(f"License refresh unexpected error: {e}")
            self._stop.wait(config.LICENSE_REFRESH_INTERVAL_S)


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
