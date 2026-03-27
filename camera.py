import cv2
import time
import logging
import threading
import db

log = logging.getLogger("laventra")

FAILURE_THRESHOLD = 30
RECONNECT_DELAY   = 3


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
        return False

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
                if self._fail_count >= FAILURE_THRESHOLD and not self._asking:
                    self._asking = True
                    new_url = self._prompt_new_url()
                    if new_url and new_url != self._url:
                        self.update_url(new_url)
                    self._asking     = False
                    self._fail_count = 0
                log.warning(
                    f"Camera unavailable — retry in {RECONNECT_DELAY}s "
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
            log.info("✅ Camera OK")
            return url

        log.warning(f"Cannot open: {url}")
        try:
            skip = input("  Save anyway? (y/n) [n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return saved
        if skip == "y":
            db.session_set("camera_url", url)
            return url