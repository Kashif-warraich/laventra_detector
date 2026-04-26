"""
python main.py                         → run (setup on first launch)
python main.py --setup                 → re-run setup
python main.py --status                → show config and exit
python main.py --reset                 → clear session and start fresh
python main.py --debug                 → verbose logging
python main.py --test                  → test with configured camera (no API)
python main.py --test --source FILE    → test with a video file (no API)
python main.py --test --source 0       → test with laptop webcam (no API)
"""

import sys
import time
import signal
import argparse
from datetime import datetime, timezone, timedelta


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with Z suffix.

    Uses Z (not +00:00) so JavaScript's Date constructor parses it
    correctly in all browsers without extra frontend conversion.
    The backend (Rails / ActiveRecord) stores it as UTC regardless.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _utc_from_ts(ts: float) -> str:
    """Convert a Unix timestamp to a UTC ISO 8601 string with Z suffix."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

import cv2
import threading
import queue as _queue

import log_setup
import db
import auth
import lavvaggio as lav_module
import camera as cam_module
import detector as det_module
import api


# ── Plate-visit tracking ─────────────────────────────────────────────────────
# How often the detector is allowed to re-detect the same plate (reduces OCR load).
_PRESENCE_COOLDOWN = 3    # seconds

# How long a plate must be absent before we declare the visit over and post the event.
_PRESENCE_GRACE    = 8    # seconds

# Minimum event duration enforced in the payload (API rejects started_at == ended_at).
_MIN_EVENT_SECS    = 1    # seconds


class _PlateTracker:
    """
    Tracks vehicles by YOLO track_id — not by plate string.

    Visit lifecycle
    ───────────────
    ENTER  : YOLO assigns a new track_id → record started_at
    UPDATE : same track_id seen again → update last_seen, accumulate OCR reading
    EXIT   : track_id absent for _PRESENCE_GRACE seconds → pick best plate from
             all accumulated readings, post ONE event

    Using track_id (instead of plate string) means OCR noise across frames
    (e.g. "FV646PR" vs "FV66PR") no longer creates duplicate events — they
    are all collected as readings for the same physical vehicle visit.
    The best plate is chosen by frequency first, then confidence.
    """

    def __init__(self):
        # track_id (int) → visit dict
        self._active: dict = {}

    def update(
        self,
        track_id:     int,
        vehicle_type: str,
        plate:        str   = None,
        ocr_conf:     float = 0.0,
    ) -> bool:
        """
        Record a frame for this track. Returns True if it is a brand-new visit.
        plate may be None when YOLO sees the vehicle but OCR hasn't read yet.
        """
        now_ts  = time.time()
        now_iso = _utc_now()

        if track_id not in self._active:
            self._active[track_id] = {
                "started_ts":    now_ts,
                "started_iso":   now_iso,
                "last_seen_ts":  now_ts,
                "last_seen_iso": now_iso,
                "vehicle_type":  vehicle_type,
                "readings":      [],   # [(plate_str, ocr_conf), ...]
            }
            is_new = True
        else:
            v = self._active[track_id]
            v["last_seen_ts"]  = now_ts
            v["last_seen_iso"] = now_iso
            is_new = False

        if plate:
            self._active[track_id]["readings"].append((plate, ocr_conf))

        return is_new

    def collect_completed(self, grace: float = _PRESENCE_GRACE) -> list:
        """
        Return finalised events for tracks absent for at least `grace` seconds.
        Tracks with zero plate readings are silently discarded (car never identified).
        """
        now_ts    = time.time()
        completed = []
        for tid, v in list(self._active.items()):
            if now_ts - v["last_seen_ts"] >= grace:
                evt = _PlateTracker._make_event(tid, v)
                if evt:
                    completed.append(evt)
                else:
                    log.debug(f"Track #{tid}: no plate readings — skipping event")
                del self._active[tid]
        return completed

    def flush(self) -> list:
        """Force-complete all active visits (shutdown)."""
        completed = [e for e in
                     (_PlateTracker._make_event(t, v) for t, v in self._active.items())
                     if e]
        self._active.clear()
        return completed

    @staticmethod
    def _best_plate(readings: list) -> tuple:
        """
        Given [(plate, conf), …] return (best_plate_str, best_conf).

        Strategy: prefer the plate string that appears most often across frames
        (handles OCR noise). Break ties by highest confidence.
        """
        from collections import Counter
        if not readings:
            return None, 0.0

        freq      = Counter(p for p, _ in readings)
        max_freq  = freq.most_common(1)[0][1]
        # Candidates: all plates tied for highest frequency
        top       = {p for p, f in freq.items() if f == max_freq}
        # Among those, pick the one with the highest single-reading confidence
        best_plate, best_conf = max(
            ((p, c) for p, c in readings if p in top),
            key=lambda x: x[1],
        )
        return best_plate, best_conf

    @staticmethod
    def _make_event(track_id: int, v: dict):
        plate, ocr_conf = _PlateTracker._best_plate(v["readings"])
        if not plate:
            return None   # never got a valid plate read

        started_ts  = v["started_ts"]
        started_iso = v["started_iso"]
        last_ts     = v["last_seen_ts"]
        last_iso    = v["last_seen_iso"]

        # API requires ended_at strictly after started_at
        if last_ts < started_ts + _MIN_EVENT_SECS:
            ended_iso = _utc_from_ts(started_ts + _MIN_EVENT_SECS)
        else:
            ended_iso = last_iso

        return {
            "plate":        plate,
            "vehicle_type": v["vehicle_type"],
            "started_at":   started_iso,
            "ended_at":     ended_iso,
            "ocr_conf":     ocr_conf,
            "track_id":     track_id,
            "reading_count": len(v["readings"]),
            "all_readings": v["readings"],   # for debug logging
        }


def _post_event_and_queue(evt: dict, lav_id: int, dev_id, stats: dict) -> None:
    """Post a completed visit event; fall back to the offline queue on network failure."""
    plate      = evt["plate"]
    n_readings = evt.get("reading_count", "?")
    ocr_conf   = evt.get("ocr_conf", 0.0)
    log.info(
        f"📋  Posting event: {plate}  "
        f"{evt['started_at']}  →  {evt['ended_at']}  "
        f"(track #{evt.get('track_id','?')}  {n_readings} readings  ocr={ocr_conf:.0%})"
    )
    result = api.post_event(
        lavvaggio_id=lav_id,
        plate=plate,
        vehicle_type=evt["vehicle_type"],
        started_at=evt["started_at"],
        ended_at=evt["ended_at"],
        device_id=dev_id,
    )
    if result is True:
        stats["sent"] += 1
    elif result is api._PERMANENT_FAILURE:
        # Payload was permanently rejected (422/401) — no point queuing it
        log.warning(f"⚠️  Event for {plate} permanently rejected — not queued")
    else:
        # Network/temporary failure — queue for later retry
        stats["queued"] += 1
        db.enqueue(
            lavvaggio_id=lav_id,
            plate=plate,
            vehicle_type=evt["vehicle_type"],
            started_at=evt["started_at"],
            ended_at=evt["ended_at"],
            device_id=dev_id,
        )
        log.info(f"📥 Queued offline → {plate}")


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--setup",  action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--reset",       action="store_true")
    p.add_argument("--clear-queue", action="store_true",
                   help="Delete all queued events and exit")
    p.add_argument("--debug",  action="store_true")
    p.add_argument("--test",   action="store_true",
                   help="Test mode: live preview window, no API posting")
    p.add_argument("--post",   action="store_true",
                   help="Also post detected plates to the API (use with --test)")
    p.add_argument("--source", default=None,
                   help="Video file or camera index for --test (default: configured camera)")
    return p.parse_args()


_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    print()
    log.info("Stopping…")
    _shutdown = True


def _print_status() -> None:
    s = db.session_summary()
    q = db.queue_count()
    print()
    print("─" * 52)
    print("  LAVENTRA DETECTOR — STATUS")
    print("─" * 52)
    print(f"  API URL    : {s['api_url']}")
    print(f"  Email      : {s['email']}")
    print(f"  Lavvaggio  : {s['lavvaggio_name']} (id={s['lavvaggio_id']})")
    print(f"  Device     : {s['device_id'] or '—'}")
    print(f"  Camera     : {s['camera_url']}")
    print(f"  Token      : {'✅ set' if s['token_set'] == 'yes' else '❌ missing'}")
    print(f"  Queue      : {q} event(s) pending")
    print(f"  Ready      : {'✅ yes' if db.has_session() else '❌ run: python main.py'}")
    print("─" * 52)
    print()


def _run_setup() -> bool:
    print()
    print("═" * 52)
    print("  LAVENTRA DETECTOR — Setup")
    print("═" * 52)
    print("  Saved locally — only needed once.")
    print()

    if not auth.interactive_login():
        log.error("Setup cancelled — sign in failed")
        return False

    if not lav_module.interactive_select():
        log.error("Setup cancelled — no lavvaggio selected")
        return False

    # ── Provision a persistent device token ──
    # After this point the detector runs against the API with the device token,
    # not the user JWT — so the user's password can change without breaking it.
    lav_id, lav_name, _prev_dev_id = lav_module.load_from_db()
    log.info("Provisioning device API token…")
    result = auth.device_setup(
        lavvaggio_id=lav_id,
        serial_number=db.session_get("device_serial", None),
    )
    if not result.get("ok"):
        log.error(f"Device setup failed ({result.get('reason','unknown')}) — cannot continue")
        return False

    db.session_set("device_id",    result["device_id"])
    db.session_set("device_token", result["api_token"])
    if result.get("serial_number"):
        db.session_set("device_serial", result["serial_number"])

    # Clear the user password — the device token is what we use from now on
    db.session_set("password", "")
    # Also switch the shared HTTP session to the device token
    auth.apply_device_token()
    log.info(f"✅ Device token provisioned (device_id={result['device_id']})")

    cam_module.setup_camera_url()

    print()
    print("═" * 52)
    print("  ✅  Setup complete!")
    print("═" * 52)
    print(f"  Lavvaggio : {lav_name} (id={lav_id})")
    print(f"  Device    : {result['device_id']}")
    print(f"  Camera    : {db.session_get('camera_url', '—')}")
    print(f"  API       : {db.session_get('api_url')}")
    print()
    return True


def _interactive_reconnect() -> bool:
    """
    Called on startup when saved config exists but API is unreachable/unauthorized.
    Returns True if connected, False to proceed in offline mode.
    """
    print()
    print("─" * 52)
    print("  ⚠️  Cannot connect to Laventra API")
    print("─" * 52)
    s = db.session_summary()
    print(f"  URL   : {s['api_url']}")
    print(f"  Email : {s['email']}")
    print()
    print("  1 — API URL is wrong  (enter a new URL)")
    print("  2 — Credentials wrong (sign in again)")
    print("  3 — Continue offline  (queue events)")
    print()

    while True:
        try:
            choice = input("  Choice [3]: ").strip() or "3"
        except (EOFError, KeyboardInterrupt):
            return False

        if choice == "1":
            if auth.prompt_new_api_url():
                log.info("✅ Reconnected successfully")
                return True
            # Prompt again
            print()
            try:
                again = input("  Still offline. Try another option? (y/n) [y]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if again == "n":
                return False
            print()
            print("  1 — API URL is wrong  (enter a new URL)")
            print("  2 — Credentials wrong (sign in again)")
            print("  3 — Continue offline  (queue events)")
            print()
            continue

        if choice == "2":
            if auth.interactive_relogin():
                log.info("✅ Reconnected successfully")
                return True
            print()
            try:
                again = input("  Still offline. Try another option? (y/n) [y]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if again == "n":
                return False
            print()
            print("  1 — API URL is wrong  (enter a new URL)")
            print("  2 — Credentials wrong (sign in again)")
            print("  3 — Continue offline  (queue events)")
            print()
            continue

        # choice == "3" or anything else
        log.info("Continuing in offline mode — events will be queued")
        return False


def _run_test(source=None, post=False) -> None:
    import os

    # Resolve source: explicit arg > configured camera > webcam 0
    if source is None:
        source = db.session_get("camera_url", "0")
    src = int(source) if str(source).strip().isdigit() else source
    is_file = isinstance(src, str) and os.path.isfile(src)

    # When --post is requested, load lavvaggio/device and verify API connection
    lav_id = lav_name = dev_id = None
    if post:
        lav_id, lav_name, dev_id = lav_module.load_from_db()
        if not lav_id:
            log.error("No lavvaggio set — run: python main.py --setup")
            sys.exit(1)
        auth.load_saved_token()
        log.info("Verifying API session…")
        result = auth.verify_connection()
        if not result["ok"]:
            log.error("Cannot connect to the API — run without --post or fix setup")
            sys.exit(1)
        log.info("✅ API session verified")

    print()
    print("═" * 52)
    print("  LAVENTRA DETECTOR — Test Mode")
    print("═" * 52)
    print(f"  Source  : {source}")
    print(f"  Type    : {'video file' if is_file else 'camera feed'}")
    if post:
        print(f"  API     : enabled — posting events")
        print(f"  Lavvaggio : {lav_name} (id={lav_id})")
        print(f"  Device    : {dev_id or '—'}")
    else:
        print(f"  API     : disabled (test only)")
    print("═" * 52)
    print()
    print("  Press Q or Ctrl+C to quit")
    print()

    # yolov8s.pt is more accurate than yolov8n.pt — download it once with:
    #   python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
    log.info("Loading AI models…")
    detector = det_module.PlateDetector(
        yolo_model   = "yolov8s.pt" if __import__("os").path.exists("yolov8s.pt") else "yolov8n.pt",
        yolo_conf    = 0.25,          # low threshold — catch everything in test
        ocr_min_conf = 0.40,          # lower OCR bar so partial reads show
        cooldown_sec = 30 if (post and is_file) else (0 if is_file else 3),
        use_gpu      = False,
    )
    if not detector.ready:
        log.error("AI models failed to load — pip install ultralytics easyocr")
        sys.exit(1)

    # CAP_AVFOUNDATION only works for local device indices on macOS.
    # HTTP streams (DroidCam, IP Webcam) use _MJPEGCapture which is reliable.
    if is_file:
        cap = cv2.VideoCapture(src)
    else:
        cap = cam_module._open_cap(src)

    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        sys.exit(1)

    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    delay_ms     = max(1, int(1000 / fps_src)) if is_file else 1

    if is_file:
        duration = int(total_frames / fps_src) if fps_src else 0
        log.info(f"  Video : {total_frames} frames  {fps_src:.0f}fps  ~{duration}s")
        log.info(f"  Size  : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    log.info("✅ Source opened — press Q in the preview window to quit")
    log.info("─" * 52)

    # Target display width — keeps the window small and imshow fast
    DISPLAY_W = 480
    frame_interval = 1.0 / fps_src   # seconds per frame

    # ── Detection runs in a background thread so it never blocks the display ──
    # frame_q : main thread drops the latest frame here (size=1, always fresh)
    # result_q: detection thread posts results here (size=1, main reads latest)
    frame_q  = _queue.Queue(maxsize=1)
    result_q = _queue.Queue(maxsize=1)
    det_stop  = threading.Event()

    def _detect_worker():
        while not det_stop.is_set():
            try:
                frm = frame_q.get(timeout=0.3)
            except _queue.Empty:
                continue
            dets = detector.detect(frm)
            # replace any unconsumed result with the latest
            try:
                result_q.put_nowait(dets)
            except _queue.Full:
                try:
                    result_q.get_nowait()
                except _queue.Empty:
                    pass
                result_q.put_nowait(dets)

    det_thread = threading.Thread(target=_detect_worker, daemon=True, name="detector")
    det_thread.start()

    seen_plates = {}
    frame_n     = 0
    t_start     = time.time()
    last_dets   = []
    no_det_ctr  = 0
    stats       = {"detected": 0, "sent": 0, "queued": 0}

    while not _shutdown:
        t_frame = time.time()

        ret, frame = cap.read()
        if not ret:
            if is_file:
                log.info("End of video.")
            else:
                log.warning("Camera read failed — retrying…")
                time.sleep(0.05)
            break

        frame_n += 1

        # Feed latest frame to detector — drop if busy (never block display)
        try:
            frame_q.put_nowait(frame.copy())
        except _queue.Full:
            pass

        # Pick up latest detection result (non-blocking)
        try:
            dets = result_q.get_nowait()
            if dets:
                no_det_ctr = 0
                last_dets  = dets
                for det in dets:
                    plate    = det["plate"]     # may be None while reading
                    vtype    = det["type"]
                    track_id = det["track_id"]
                    # mark_seen is now handled inside detector; skip external call
                    if plate:
                        seen_plates[plate] = seen_plates.get(plate, 0) + 1
                        log.info(
                            f"🚗  #{track_id}  {plate:<12}  type={vtype:<10}  "
                            f"yolo={det['yolo_conf']:.0%}  ocr={det['ocr_conf']:.0%}  "
                            f"(seen {seen_plates[plate]}x)"
                        )
                        if post:
                            now_ts2     = time.time()
                            started_iso = _utc_now()
                            ended_iso   = _utc_from_ts(now_ts2 + 1)
                            stats["detected"] += 1
                            result = api.post_event(
                                lavvaggio_id=lav_id,
                                plate=plate,
                                vehicle_type=vtype,
                                started_at=started_iso,
                                ended_at=ended_iso,
                                device_id=dev_id,
                            )
                            if result is True:
                                stats["sent"] += 1
                                log.info(f"  ✅ Event posted: {plate}")
                            elif result is not api._PERMANENT_FAILURE:
                                stats["queued"] += 1
                                db.enqueue(
                                    lavvaggio_id=lav_id,
                                    plate=plate,
                                    vehicle_type=vtype,
                                    started_at=started_iso,
                                    ended_at=ended_iso,
                                    device_id=dev_id,
                                )
                                log.warning(f"  ⚠️  Event queued (API failed): {plate}")
            else:
                no_det_ctr += 1
                if no_det_ctr > 20:
                    last_dets = []
        except _queue.Empty:
            pass

        # Resize for display — imshow on large frames (888x1920) is very slow
        h_f, w_f = frame.shape[:2]
        dh = int(h_f * DISPLAY_W / w_f)
        vis = cv2.resize(detector.draw(frame, last_dets), (DISPLAY_W, dh))

        elapsed = int(time.time() - t_start)
        if is_file and total_frames:
            pct = int(frame_n / total_frames * 100)
            hud = f"  {pct}%  |  {len(seen_plates)} plate(s)  |  Q=quit  "
        else:
            hud = f"  {elapsed}s  |  {len(seen_plates)} plate(s)  |  Q=quit  "

        (hw, hh), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (0, 0), (hw + 8, hh + 8), (30, 30, 30), -1)
        cv2.putText(vis, hud, (4, hh + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Laventra — Test Mode", vis)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

        # Wall-clock pacing — sleep the remaining time in this frame slot
        if is_file:
            used = time.time() - t_frame
            remaining = frame_interval - used
            if remaining > 0:
                time.sleep(remaining)

    det_stop.set()
    det_thread.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()

    elapsed = int(time.time() - t_start)
    print()
    print("─" * 52)
    log.info(f"Test complete — {frame_n} frames  |  {elapsed}s  |  {len(seen_plates)} unique plate(s)")
    if seen_plates:
        log.info("Plates detected:")
        for plate, count in sorted(seen_plates.items(), key=lambda x: -x[1]):
            log.info(f"  {plate:<14}  ×{count}")
    else:
        log.info("No plates detected")
    if post:
        log.info(
            f"Events — detected={stats['detected']}  "
            f"sent={stats['sent']}  "
            f"queued={stats['queued']}"
        )
    print("─" * 52)


def _run_detector() -> None:
    lav_id, lav_name, _ = lav_module.load_from_db()
    cam_url             = db.session_get("camera_url", "0")
    device_id_raw       = db.session_get("device_id", "")
    dev_id              = int(device_id_raw) if device_id_raw and str(device_id_raw).isdigit() else None
    device_token        = db.session_get("device_token", "")

    if not lav_id:
        log.error("No lavvaggio set — run: python main.py --setup")
        sys.exit(1)

    if not device_token:
        log.error("No device token — your session predates this detector version.")
        log.error("  → Please re-run: python main.py --setup")
        sys.exit(1)

    print()
    print("═" * 52)
    print("  LAVENTRA DETECTOR — Running")
    print("═" * 52)
    print(f"  Lavvaggio : {lav_name} (id={lav_id})")
    print(f"  Device    : {dev_id or '—'}")
    print(f"  Camera    : {cam_url}")
    print(f"  API       : {db.session_get('api_url')}")
    print(f"  Queue     : {db.queue_count()} pending")
    print("═" * 52)
    print()

    # Attach the persistent device token to the shared HTTP session.
    # No user JWT, no re-login. Password changes never affect the detector.
    auth.apply_device_token()

    log.info("Verifying API session…")
    result = auth.verify_connection()
    if not result["ok"]:
        reason = result.get("reason", "unknown")
        if reason == "url":
            log.warning("API unreachable at startup — continuing; events will queue")
        elif reason == "auth":
            log.error("Device token rejected — re-run: python main.py --setup")
            sys.exit(1)
        else:
            log.warning(f"API check returned {reason} — continuing in degraded mode")
    else:
        log.info("✅ API session verified (device token)")

    flusher = api.QueueFlusher()
    flusher.start()

    log.info("Loading AI models…")
    detector = det_module.PlateDetector(
        yolo_model   = "yolov8n.pt",
        yolo_conf    = 0.40,
        ocr_min_conf = 0.55,
        cooldown_sec = _PRESENCE_COOLDOWN,
        use_gpu      = False,
    )
    if not detector.ready:
        log.error("AI models failed to load")
        log.error("  → pip install ultralytics easyocr")
        sys.exit(1)

    camera = cam_module.CameraStream(cam_url)
    camera.start()

    # Start heartbeat thread — tells the backend "I'm alive" every 60s.
    # Also reports camera connectivity so the backend can track it independently.
    heartbeat = api.HeartbeatThread(
        device_id=dev_id,
        device_token=device_token,
        camera_status_fn=lambda: camera.connected,
    )
    heartbeat.start()

    log.info("✅ Detector running — press Ctrl+C to stop")
    log.info(f"   Plate tracking: cooldown={_PRESENCE_COOLDOWN}s  grace={_PRESENCE_GRACE}s")
    log.info("─" * 52)

    tracker    = _PlateTracker()
    stats      = {"detected": 0, "sent": 0, "queued": 0}
    last_stats = time.time()
    frame_n    = 0

    while not _shutdown:
        try:
            frame = camera.read()
            if frame is None:
                # Camera not ready — still check for visits that have ended
                for evt in tracker.collect_completed():
                    _post_event_and_queue(evt, lav_id, dev_id, stats)
                time.sleep(0.05)
                continue

            frame_n += 1
            if frame_n % 3 != 0:
                continue

            detections = detector.detect(frame)

            for det in detections:
                track_id = det["track_id"]
                vtype    = det["type"]
                plate    = det["plate"]    # None when OCR not yet read
                ocr_conf = det["ocr_conf"]

                # mark_seen() is now called inside detector._detect_inner()
                # when OCR succeeds — no external call needed here.

                is_new = tracker.update(track_id, vtype, plate, ocr_conf)

                if plate:
                    stats["detected"] += 1
                    log.info(
                        f"🚗  #{track_id}  {plate:<12}  type={vtype:<12}  "
                        f"yolo={det['yolo_conf']:.0%}  ocr={ocr_conf:.0%}"
                        + ("  [arrived]" if is_new else "")
                    )
                elif is_new:
                    log.info(
                        f"🚗  #{track_id}  (reading plate…)  type={vtype:<12}  "
                        f"yolo={det['yolo_conf']:.0%}  [arrived]"
                    )

            # Post events for cars that have been absent for _PRESENCE_GRACE seconds
            for evt in tracker.collect_completed():
                _post_event_and_queue(evt, lav_id, dev_id, stats)

            if time.time() - last_stats >= 60:
                active  = len(tracker._active)
                cam_ok  = "✅" if camera.connected else "❌ reconnecting"
                log.info(
                    f"📊  detected={stats['detected']}  "
                    f"sent={stats['sent']}  "
                    f"queued={stats['queued']}  "
                    f"pending={db.queue_count()}  "
                    f"active_visits={active}  "
                    f"camera={cam_ok}"
                )
                last_stats = time.time()

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Main loop error (continuing): {e}")
            time.sleep(0.1)

    # Finalise any visits still active at shutdown
    log.info("Finalising active visits before shutdown…")
    for evt in tracker.flush():
        _post_event_and_queue(evt, lav_id, dev_id, stats)

    log.info("Shutting down…")
    try:
        camera.stop()
    except Exception as e:
        log.error(f"Camera stop error: {e}")
    try:
        flusher.stop()
    except Exception as e:
        log.error(f"Flusher stop error: {e}")
    try:
        heartbeat.stop()
    except Exception as e:
        log.error(f"Heartbeat stop error: {e}")

    print()
    print("─" * 52)
    log.info(
        f"Session — detected={stats['detected']}  "
        f"sent={stats['sent']}  "
        f"queued={stats['queued']}"
    )
    log.info("Goodbye.")
    print("─" * 52)


if __name__ == "__main__":
    args = _args()
    log  = log_setup.setup(debug=args.debug)
    db.init()

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if args.reset:
        try:
            confirm = input("⚠️  Clear all config? Type YES: ")
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if confirm.strip() == "YES":
            db.session_clear()
            log.info("✅ Session cleared")
        else:
            log.info("Cancelled")
        sys.exit(0)

    if args.clear_queue:
        n = db.queue_clear()
        print(f"✅ Cleared {n} queued event(s)")
        sys.exit(0)

    if args.status:
        _print_status()
        sys.exit(0)

    if args.test:
        _run_test(source=args.source, post=args.post)
        sys.exit(0)

    if args.setup or not db.has_session():
        ok = _run_setup()
        if not ok:
            sys.exit(1)
        if args.setup:
            sys.exit(0)

    _run_detector()