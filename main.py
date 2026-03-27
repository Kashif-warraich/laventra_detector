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
from datetime import datetime, timezone

import cv2

import log_setup
import db
import auth
import lavvaggio as lav_module
import camera as cam_module
import detector as det_module
import api


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--setup",  action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--reset",  action="store_true")
    p.add_argument("--debug",  action="store_true")
    p.add_argument("--test",   action="store_true",
                   help="Test mode: live preview window, no API posting")
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

    cam_module.setup_camera_url()

    lav_id, lav_name, dev_id = lav_module.load_from_db()
    print()
    print("═" * 52)
    print("  ✅  Setup complete!")
    print("═" * 52)
    print(f"  Lavvaggio : {lav_name} (id={lav_id})")
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


def _run_test(source=None) -> None:
    import os

    # Resolve source: explicit arg > configured camera > webcam 0
    if source is None:
        source = db.session_get("camera_url", "0")
    src = int(source) if str(source).strip().isdigit() else source
    is_file = isinstance(src, str) and os.path.isfile(src)

    print()
    print("═" * 52)
    print("  LAVENTRA DETECTOR — Test Mode")
    print("═" * 52)
    print(f"  Source  : {source}")
    print(f"  Type    : {'video file' if is_file else 'camera feed'}")
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
        cooldown_sec = 0 if is_file else 3,
        use_gpu      = False,
    )
    if not detector.ready:
        log.error("AI models failed to load — pip install ultralytics easyocr")
        sys.exit(1)

    # CAP_AVFOUNDATION is a macOS live-camera backend — never use it for files
    if is_file:
        cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        sys.exit(1)

    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    delay_ms     = max(1, int(1000 / fps_src)) if is_file else 1

    if is_file:
        duration = int(total_frames / fps_src) if fps_src else 0
        log.info(f"  Video : {total_frames} frames  {fps_src:.0f}fps  ~{duration}s")
    log.info(f"✅ Source opened — press Q in the preview window to quit")
    log.info("─" * 52)

    seen_plates    = {}   # plate → count
    frame_n        = 0
    t_start        = time.time()
    last_dets      = []   # keep last detections visible between runs

    while not _shutdown:
        ret, frame = cap.read()
        if not ret:
            if is_file:
                log.info("End of video.")
            else:
                log.warning("Camera read failed — retrying…")
                time.sleep(0.1)
            break

        frame_n += 1

        # Run detection every 3rd frame — same cadence as live camera
        if frame_n % 3 == 0:
            dets = detector.detect(frame)
            if dets:
                last_dets = dets   # update overlay only when something is found
            elif last_dets:
                # clear overlay after 15 frames of no detections
                last_dets = []

            for det in dets:
                plate = det["plate"]
                vtype = det["type"]
                detector.mark_seen(plate)
                seen_plates[plate] = seen_plates.get(plate, 0) + 1
                log.info(
                    f"🚗  {plate:<12}  type={vtype:<10}  "
                    f"yolo={det['yolo_conf']:.0%}  ocr={det['ocr_conf']:.0%}  "
                    f"(seen {seen_plates[plate]}x)"
                )

        # Draw the last known overlay on every frame for smooth display
        vis = detector.draw(frame, last_dets)

        # HUD: progress + unique plates count
        elapsed = int(time.time() - t_start)
        if is_file:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            pct  = int(frame_n / total_frames * 100)
            hud  = f"  {pct}%  |  {len(seen_plates)} plate(s) found  |  Q to quit  "
        else:
            hud  = f"  {elapsed}s  |  {len(seen_plates)} plate(s) found  |  Q to quit  "

        (hw, hh), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (0, 0), (hw + 8, hh + 10), (30, 30, 30), -1)
        cv2.putText(vis, hud, (4, hh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Laventra — Test Mode", vis)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

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
    print("─" * 52)


def _run_detector() -> None:
    lav_id, lav_name, dev_id = lav_module.load_from_db()
    cam_url                  = db.session_get("camera_url", "0")

    if not lav_id:
        log.error("No lavvaggio set — run: python main.py --setup")
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

    auth.load_saved_token()

    log.info("Verifying API session…")
    result = auth.verify_connection()
    if not result["ok"]:
        reason = result.get("reason", "unknown")
        if reason == "auth":
            log.warning("Token invalid — attempting silent re-login…")
            if not auth.silent_relogin():
                log.warning("Silent re-login failed")
                _interactive_reconnect()
        elif reason == "url":
            log.warning("API unreachable")
            _interactive_reconnect()
        else:
            log.warning("API connection failed")
            _interactive_reconnect()
    else:
        log.info("✅ API session verified")

    flusher = api.QueueFlusher()
    flusher.start()

    log.info("Loading AI models…")
    detector = det_module.PlateDetector(
        yolo_model   = "yolov8n.pt",
        yolo_conf    = 0.40,
        ocr_min_conf = 0.55,
        cooldown_sec = 30,
        use_gpu      = False,
    )
    if not detector.ready:
        log.error("AI models failed to load")
        log.error("  → pip install ultralytics easyocr")
        sys.exit(1)

    camera = cam_module.CameraStream(cam_url)
    camera.start()

    log.info("✅ Detector running — press Ctrl+C to stop")
    log.info("─" * 52)

    stats      = {"detected": 0, "sent": 0, "queued": 0}
    last_stats = time.time()
    frame_n    = 0

    while not _shutdown:
        try:
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_n += 1
            if frame_n % 3 != 0:
                continue

            detections = detector.detect(frame)

            for det in detections:
                plate   = det["plate"]
                vtype   = det["type"]
                now_iso = datetime.now(timezone.utc).isoformat()

                stats["detected"] += 1
                log.info(
                    f"🚗  {plate:<12}  "
                    f"type={vtype:<12}  "
                    f"yolo={det['yolo_conf']:.0%}  "
                    f"ocr={det['ocr_conf']:.0%}"
                )

                detector.mark_seen(plate)

                ok = api.post_event(
                    lavvaggio_id=lav_id,
                    plate=plate,
                    vehicle_type=vtype,
                    started_at=now_iso,
                    ended_at=now_iso,
                    device_id=dev_id,
                )

                if ok:
                    stats["sent"] += 1
                else:
                    stats["queued"] += 1
                    db.enqueue(
                        lavvaggio_id=lav_id,
                        plate=plate,
                        vehicle_type=vtype,
                        started_at=now_iso,
                        ended_at=now_iso,
                        device_id=dev_id,
                    )

            if time.time() - last_stats >= 60:
                cam_ok = "✅" if camera.connected else "❌ reconnecting"
                log.info(
                    f"📊  detected={stats['detected']}  "
                    f"sent={stats['sent']}  "
                    f"queued={stats['queued']}  "
                    f"pending={db.queue_count()}  "
                    f"camera={cam_ok}"
                )
                last_stats = time.time()

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Main loop error (continuing): {e}")
            time.sleep(0.1)

    log.info("Shutting down…")
    try:
        camera.stop()
    except Exception as e:
        log.error(f"Camera stop error: {e}")
    try:
        flusher.stop()
    except Exception as e:
        log.error(f"Flusher stop error: {e}")

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

    if args.status:
        _print_status()
        sys.exit(0)

    if args.test:
        _run_test(source=args.source)
        sys.exit(0)

    if args.setup or not db.has_session():
        ok = _run_setup()
        if not ok:
            sys.exit(1)
        if args.setup:
            sys.exit(0)

    _run_detector()