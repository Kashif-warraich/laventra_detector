"""
Camera registry — fetched from the backend, cached locally.

A lavvaggio may have multiple cameras (entry, tunnel, exit). The detector
streams from exactly one at a time; the operator picks via `--select-camera`.
"""
import json
import logging

import requests

import config
import db
import license as license_module

log = logging.getLogger("laventra")


class CameraFetchError(Exception):
    pass


def _v1() -> str:
    url = db.kv_get("api_url", "").rstrip("/")
    if url.endswith("/api/v1"):
        return url
    return f"{url}/api/v1"


def fetch_remote() -> list:
    """
    Pull the camera list from /api/v1/cameras. Persists to local DB on success.
    Raises CameraFetchError on failure (network or auth).
    """
    headers = {**license_module.bearer_header(),
               "Accept": "application/json",
               "User-Agent": config.USER_AGENT}
    try:
        r = requests.get(f"{_v1()}/cameras", headers=headers,
                         timeout=config.API_TIMEOUT_S)
    except requests.exceptions.RequestException as e:
        raise CameraFetchError(f"Cannot reach backend: {e}") from e

    if r.status_code == 401:
        raise CameraFetchError("License rejected — has it been revoked?")
    if r.status_code != 200:
        raise CameraFetchError(f"HTTP {r.status_code}: {r.text[:200]}")

    data = r.json() or {}
    cams = data.get("cameras") or data.get("data") or []
    normalised = []
    for c in cams:
        if not c.get("id") or not c.get("stream_url"):
            log.warning(f"Skipping camera with missing id/stream_url: {c}")
            continue
        normalised.append({
            "id": int(c["id"]),
            "name": c.get("name") or f"Camera {c['id']}",
            "stream_url": c["stream_url"],
            "kind": c.get("kind"),
            "metadata_json": json.dumps(c.get("metadata")) if c.get("metadata") else None,
        })
    db.cameras_replace(normalised)
    log.info(f"📷 Fetched {len(normalised)} camera(s) from backend")
    return db.cameras_list()


def local_list() -> list:
    """Return locally-cached cameras (no network)."""
    return db.cameras_list()


def selected() -> dict | None:
    """Return the currently-selected camera dict (None if none chosen)."""
    cid = db.kv_get("camera_id", "")
    if not cid or not cid.isdigit():
        return None
    return db.cameras_get(int(cid))


def select(camera_id: int) -> dict:
    """Mark a camera as the active one. Returns the camera dict."""
    cam = db.cameras_get(camera_id)
    if not cam:
        raise ValueError(f"No camera with id={camera_id} in local cache")
    db.kv_set("camera_id", camera_id)
    db.kv_set("camera_url", cam["stream_url"])
    log.info(f"📷 Selected camera #{cam['id']} — {cam['name']} → {cam['stream_url']}")
    return cam


def interactive_select() -> dict | None:
    """
    Refresh from backend (or fall back to cache), show list, prompt for choice.
    Returns chosen camera dict or None on cancel.
    """
    print()
    print("─" * 52)
    print("  Select camera")
    print("─" * 52)
    try:
        cams = fetch_remote()
    except CameraFetchError as e:
        log.warning(f"Could not refresh camera list: {e}")
        log.info("Falling back to cached list")
        cams = local_list()
    if not cams:
        log.error("No cameras available. Has the admin added any to this lavvaggio?")
        return None

    print()
    for i, c in enumerate(cams, 1):
        print(f"   {i:2}. #{c['id']}  {c['name']:<30}  {c['stream_url']}")
    print()
    try:
        raw = input(f"  Choose [1-{len(cams)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw.isdigit() or not (1 <= int(raw) <= len(cams)):
        log.warning("Invalid choice")
        return None
    return select(cams[int(raw) - 1]["id"])
