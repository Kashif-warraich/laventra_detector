import cv2
import time
import socket
import ipaddress
import logging
import threading
from urllib.parse import urlparse, urlunparse
import db

log = logging.getLogger("laventra")

FAILURE_THRESHOLD = 30
RECONNECT_DELAY   = 10
# 5 minutes of consecutive failure at 10s intervals = 30 attempts
_WARN_AFTER_ATTEMPTS = 30

# ── Network-scan fallback ────────────────────────────────────────────────────
_SCAN_PORT_TIMEOUT = 0.3
_SCAN_VERIFY_TIMEOUT_MS = 1500


def _extract_port(url: str):
    """
    Extract the port from an HTTP(S) camera URL. Returns None for numeric
    sources (webcams) or malformed URLs.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https") and parsed.port:
            return parsed.port
        # Default HTTP camera ports if explicit port missing
        if parsed.scheme == "http":
            return 80
        if parsed.scheme == "https":
            return 443
    except Exception:
        pass
    return None


def _local_subnet_hosts():
    """
    Return an iterable of IPs in the /24 subnet the local machine is on,
    excluding our own IP. Skips loopback and link-local.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't actually send a packet — used to pick the outbound iface
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception as e:
        log.debug(f"Could not determine local IP: {e}")
        return []

    try:
        net = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    except Exception:
        return []

    return [str(h) for h in net.hosts() if str(h) != local_ip]


class CameraStream:
    def __init__(self, url: str):
        self._url        = url
        self._cap        = None
        self._frame      = None
        self._lock       = threading.Lock()
        self._stop       = threading.Event()
        self._thread     = None
        self._connected  = False
        self._fail_count = 0
        self._asking     = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        log.info(f"📷 Opening camera: {self._url}")
        self._open()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="camera"
        )
        self._thread.start()
        return True

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._release()

    def update_url(self, new_url: str) -> None:
        log.info(f"📷 Camera URL updated → {new_url}")
        self._url        = new_url
        self._fail_count = 0
        self._connected  = False
        self._release()
        db.session_set("camera_url", new_url)

    def _open(self) -> bool:
        self._release()
        src = int(self._url) if str(self._url).strip().isdigit() else self._url
        try:
            cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, _ = cap.read()
                if ret:
                    self._cap        = cap
                    self._connected  = True
                    self._fail_count = 0
                    log.info(f"✅ Camera connected: {self._url}")
                    return True
                cap.release()
                log.warning("Camera opened but no frames received")
            else:
                log.warning(f"Cannot open: {self._url}")
        except Exception as e:
            log.error(f"Camera open error: {e}")

        # ── Fallback: scan local subnet for the camera on the same port ──
        # Only for HTTP(S) URLs — skip for integer webcam sources.
        if not isinstance(src, int):
            discovered = self._discover_camera_on_network_sync()
            if discovered:
                log.info(f"📡 Camera found at new address: {discovered}")
                self._url = discovered
                db.session_set("camera_url", discovered)
                # Try to open the discovered URL once
                try:
                    cap = cv2.VideoCapture(discovered, cv2.CAP_AVFOUNDATION)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(discovered)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, _ = cap.read()
                        if ret:
                            self._cap        = cap
                            self._connected  = True
                            self._fail_count = 0
                            log.info(f"✅ Camera reconnected at: {discovered}")
                            return True
                        cap.release()
                except Exception as e:
                    log.error(f"Discovered camera open error: {e}")
        return False

    def _discover_camera_on_network_sync(self):
        """
        Scans the local /24 subnet for a host listening on the configured camera
        port, then verifies each candidate by attempting to open it with OpenCV
        using the same path as the original URL. Returns the first working URL,
        or None.
        """
        port = None
        raw_port = db.session_get("camera_port", "")
        if raw_port and str(raw_port).isdigit():
            port = int(raw_port)
        else:
            port = _extract_port(self._url)
        if not port:
            return None

        parsed = urlparse(self._url)
        path   = parsed.path or "/video"
        scheme = parsed.scheme or "http"

        hosts = _local_subnet_hosts()
        if not hosts:
            log.debug("Camera discovery: no subnet hosts to scan")
            return None

        log.info(f"📡 Scanning local subnet for camera on port {port}…")
        candidates = []
        for ip in hosts:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(_SCAN_PORT_TIMEOUT)
            try:
                if sock.connect_ex((ip, port)) == 0:
                    candidates.append(ip)
            except Exception:
                pass
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

        if not candidates:
            log.debug(f"Camera discovery: no hosts listening on :{port}")
            return None

        log.info(f"📡 {len(candidates)} candidate host(s) — verifying with OpenCV…")
        for ip in candidates:
            url = urlunparse((scheme, f"{ip}:{port}", path, "", "", ""))
            try:
                test = cv2.VideoCapture(url)
                if test.isOpened():
                    test.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, _ = test.read()
                    test.release()
                    if ret:
                        return url
                else:
                    test.release()
            except Exception:
                pass
        return None

    def _release(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap       = None
            self._connected = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._connected  = False
                self._fail_count += 1
                if self._fail_count >= _WARN_AFTER_ATTEMPTS:
                    log.warning(
                        f"Camera not available for {self._fail_count * RECONNECT_DELAY}s — "
                        f"still retrying ({self._url})"
                    )
                else:
                    log.info(
                        f"Camera not available, retrying in {RECONNECT_DELAY}s... "
                        f"(attempt {self._fail_count})"
                    )
                self._stop.wait(RECONNECT_DELAY)
                self._open()
                continue

            try:
                ret, frame = self._cap.read()
            except Exception as e:
                log.error(f"Camera read error: {e}")
                self._release()
                continue

            if ret:
                with self._lock:
                    self._frame = frame
                self._connected  = True
                self._fail_count = 0
            else:
                self._fail_count += 1
                if self._fail_count % 10 == 0:
                    log.warning(f"Camera read failing ({self._fail_count} times)…")
                if self._fail_count >= FAILURE_THRESHOLD:
                    log.warning("Reconnecting camera…")
                    self._release()

    def _prompt_new_url(self) -> str:
        print()
        print("─" * 52)
        print("  ⚠️  Camera is not responding")
        print("─" * 52)
        print(f"  Current: {self._url}")
        print()
        print("  1 — Enter a new camera URL")
        print("  2 — Keep retrying")
        print()
        try:
            choice = input("  Choice [2]: ").strip() or "2"
        except (EOFError, KeyboardInterrupt):
            return self._url
        if choice != "1":
            return self._url
        print()
        print("  Mac camera      : 0")
        print("  iPhone DroidCam : http://PHONE_IP:4747/video")
        print("  Android IPCam   : http://PHONE_IP:8080/video")
        print()
        while True:
            try:
                new_url = input(f"  New URL [{self._url}]: ").strip() or self._url
            except (EOFError, KeyboardInterrupt):
                return self._url
            src  = int(new_url) if new_url.strip().isdigit() else new_url
            test = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            if not test.isOpened():
                test = cv2.VideoCapture(src)
            ok = test.isOpened()
            test.release()
            if ok:
                return new_url
            log.warning(f"Cannot open: {new_url}")
            try:
                again = input("  Try another? (y/n) [y]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return self._url
            if again == "n":
                return self._url


def setup_camera_url() -> str:
    print()
    print("─" * 52)
    print("  Camera Setup")
    print("─" * 52)
    print()
    print("  Mac built-in camera : 0  ← easiest for testing")
    print("  iPhone DroidCam     : http://PHONE_IP:4747/video")
    print("  Android IP Webcam   : http://PHONE_IP:8080/video")
    print()

    saved = db.session_get("camera_url", "0")

    while True:
        try:
            url = input(f"  Camera URL [{saved}]: ").strip() or saved
        except (EOFError, KeyboardInterrupt):
            return saved

        log.info(f"Testing {url} …")
        src  = int(url) if url.strip().isdigit() else url
        test = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        if not test.isOpened():
            test = cv2.VideoCapture(src)
        ok = test.isOpened()
        test.release()

        if ok:
            db.session_set("camera_url", url)
            port = _extract_port(url)
            if port:
                db.session_set("camera_port", port)
            log.info("✅ Camera OK")
            return url

        log.warning(f"Cannot open: {url}")
        try:
            skip = input("  Save anyway? (y/n) [n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return saved
        if skip == "y":
            db.session_set("camera_url", url)
            port = _extract_port(url)
            if port:
                db.session_set("camera_port", port)
            return url