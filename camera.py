"""
Camera streaming — robust to dropouts, IP changes, and RTSP/MJPEG quirks.

Improvements over the v0.1 design (audit findings H2/H3/H4/M1/M12):
  - Scheme-aware capture:  rtsp://* uses cv2 with CAP_FFMPEG; http(s) MJPEG
                           uses a hand-rolled multipart reader.
  - MJPEG byte buffer is capped at config.CAMERA_MJPEG_BUF_MAX (prevents
    unbounded growth on a corrupt stream).
  - Subnet rediscovery runs in a separate thread with a 10-minute cooldown
    so the read loop never blocks on a 254-host scan.
  - Reconnect uses exponential backoff capped at CAMERA_RECONNECT_MAX_DELAY_S.
  - Every successful frame is tagged with a monotonic timestamp, so callers
    can drop stale frames (CAMERA_FRAME_MAX_AGE_S).
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import time
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np
import requests

import config
import db

log = logging.getLogger("laventra")


# ─── MJPEG-over-HTTP reader ────────────────────────────────────────────────
class _MJPEGCapture:
    """
    Minimal cv2.VideoCapture-compatible reader for multipart MJPEG over HTTP.
    Used because cv2/AVFoundation/FFmpeg are unreliable with phone-camera apps.
    """

    _CHUNK = 8192

    def __init__(self, url: str, timeout: float = 8.0):
        self._url = url
        self._timeout = timeout
        self._response = None
        self._iter = None
        self._buf = b""
        self._opened = False
        self._connect()

    def _connect(self) -> None:
        try:
            self._response = requests.get(
                self._url,
                stream=True,
                timeout=(self._timeout, self._timeout * 4),
                headers={"User-Agent": config.USER_AGENT},
            )
            self._response.raise_for_status()
            self._iter = self._response.iter_content(chunk_size=self._CHUNK)
            self._buf = b""
            self._opened = True
        except requests.exceptions.RequestException as e:
            log.warning(f"MJPEG connect failed ({self._url}): {e}")
            self._opened = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if not self._opened or self._iter is None:
            return False, None
        try:
            while True:
                # Cap buffer growth — protects against streams that never emit SOI.
                if len(self._buf) > config.CAMERA_MJPEG_BUF_MAX:
                    log.warning("MJPEG buffer overflowed without JPEG marker — resetting")
                    self._buf = b""

                a = self._buf.find(b"\xff\xd8")
                if a == -1:
                    chunk = next(self._iter, None)
                    if not chunk:
                        self._opened = False
                        return False, None
                    self._buf += chunk
                    continue
                b = self._buf.find(b"\xff\xd9", a + 2)
                if b == -1:
                    chunk = next(self._iter, None)
                    if not chunk:
                        self._opened = False
                        return False, None
                    self._buf += chunk
                    continue
                jpg = self._buf[a : b + 2]
                self._buf = self._buf[b + 2 :]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
                # corrupt frame — keep scanning
        except Exception as e:
            log.warning(f"MJPEG stream error: {e}")
            self._opened = False
            return False, None

    def release(self) -> None:
        self._opened = False
        self._iter = None
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None

    def set(self, *args, **kwargs): pass
    def get(self, *args, **kwargs): return 0.0


def _open_capture(src):
    """
    Open the right capture object for the source URL/index.
      - int                → local webcam (cv2.VideoCapture)
      - rtsp://...         → cv2.VideoCapture with FFmpeg backend
      - http(s)://...      → _MJPEGCapture (multipart JPEG)
    """
    if isinstance(src, int):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap
    if isinstance(src, str) and src.lower().startswith("rtsp://"):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap
    return _MJPEGCapture(src)


# ─── Subnet rediscovery (rate-limited, async) ──────────────────────────────
def _extract_port(url: str):
    try:
        p = urlparse(url)
        if p.port:
            return p.port
        if p.scheme == "http":
            return 80
        if p.scheme == "https":
            return 443
        if p.scheme == "rtsp":
            return 554
    except Exception:
        pass
    return None


def _local_subnet_hosts():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        return []
    try:
        net = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    except Exception:
        return []
    return [str(h) for h in net.hosts() if str(h) != local_ip]


def _scan_subnet_for_camera(reference_url: str) -> str | None:
    """Return a new URL (or None) where the camera now lives on this LAN."""
    port = _extract_port(reference_url)
    if not port:
        return None
    parsed = urlparse(reference_url)
    scheme = parsed.scheme or "http"
    path = parsed.path or "/video"
    hosts = _local_subnet_hosts()
    if not hosts:
        return None

    log.info(f"📡 Scanning local subnet for camera on port {port}…")
    candidates = []
    for ip in hosts:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(config.CAMERA_SCAN_PORT_TIMEOUT_S)
        try:
            if sock.connect_ex((ip, port)) == 0:
                candidates.append(ip)
        finally:
            try:
                sock.close()
            except Exception:
                pass
    if not candidates:
        return None

    for ip in candidates:
        url = urlunparse((scheme, f"{ip}:{port}", path, "", "", ""))
        try:
            cap = _open_capture(url)
            try:
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        return url
            finally:
                cap.release()
        except Exception:
            pass
    return None


# ─── CameraStream ──────────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, url: str):
        self._url = url
        self._cap = None
        self._frame = None
        self._frame_ts: float = 0.0       # monotonic timestamp of last frame
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._connected = False
        self._fail_count = 0
        self._last_rediscovery = 0.0
        self._rediscovery_lock = threading.Lock()

    @property
    def url(self) -> str:
        return self._url

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        log.info(f"📷 Opening camera: {self._url}")
        self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._release()

    def read(self, *, max_age_s: float = None):
        """
        Return the latest frame, or None if no fresh frame is available.
        A frame older than max_age_s (default config.CAMERA_FRAME_MAX_AGE_S) is
        treated as stale and None is returned — caller will loop again.
        """
        if max_age_s is None:
            max_age_s = config.CAMERA_FRAME_MAX_AGE_S
        with self._lock:
            if self._frame is None:
                return None
            age = time.monotonic() - self._frame_ts
            if age > max_age_s:
                return None
            return self._frame.copy()

    # ── internals ────────────────────────────────────────────────────────
    def _open(self) -> bool:
        self._release()
        src = int(self._url) if str(self._url).strip().isdigit() else self._url
        try:
            cap = _open_capture(src)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self._cap = cap
                    self._connected = True
                    self._fail_count = 0
                    log.info(f"✅ Camera connected: {self._url}")
                    return True
                cap.release()
                log.warning("Camera opened but no frames received")
            else:
                log.warning(f"Cannot open: {self._url}")
        except Exception as e:
            log.error(f"Camera open error: {e}")
        return False

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._connected = False

    def _maybe_start_rediscovery(self) -> None:
        """Kick off subnet scan in the background, rate-limited."""
        now = time.time()
        if now - self._last_rediscovery < config.CAMERA_REDISCOVERY_COOLDOWN_S:
            return
        if not self._rediscovery_lock.acquire(blocking=False):
            return
        self._last_rediscovery = now

        def _run():
            try:
                discovered = _scan_subnet_for_camera(self._url)
                if discovered and discovered != self._url:
                    log.info(f"📡 Camera found at new address: {discovered}")
                    self._url = discovered
                    db.kv_set("camera_url", discovered)
                    self._release()  # main loop will reopen
            except Exception as e:
                log.debug(f"Rediscovery error: {e}")
            finally:
                self._rediscovery_lock.release()

        threading.Thread(target=_run, daemon=True, name="camera-rediscovery").start()

    def _loop(self) -> None:
        delay = config.CAMERA_RECONNECT_DELAY_S
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._connected = False
                self._fail_count += 1
                log.info(
                    f"Camera not available — retrying in {delay}s "
                    f"(attempt {self._fail_count})"
                )
                # Kick off async rediscovery only for IP-based sources
                if not str(self._url).isdigit():
                    self._maybe_start_rediscovery()
                self._stop.wait(delay)
                self._open()
                if self._connected:
                    delay = config.CAMERA_RECONNECT_DELAY_S
                else:
                    delay = min(delay * 2, config.CAMERA_RECONNECT_MAX_DELAY_S)
                continue

            try:
                ret, frame = self._cap.read()
            except Exception as e:
                log.error(f"Camera read error: {e}")
                self._release()
                continue

            if ret and frame is not None:
                ts = time.monotonic()
                with self._lock:
                    self._frame = frame
                    self._frame_ts = ts
                self._connected = True
                self._fail_count = 0
            else:
                self._fail_count += 1
                if self._fail_count % 10 == 0:
                    log.warning(f"Camera read failing ({self._fail_count} times)…")
                if self._fail_count >= config.CAMERA_FAILURE_THRESHOLD:
                    log.warning("Reconnecting camera…")
                    self._release()
