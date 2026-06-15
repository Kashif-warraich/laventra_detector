"""
Laventra detector — edge-device entry point.

Common operations
─────────────────
  python main.py --activate XXXX-YYYY-ZZZZ           Activate this device.
                                                       (One-time. Admin issues the
                                                        code from the web UI.)
  python main.py --select-camera                      Pick which camera to stream.
  python main.py --status                             Show config + license + queue.
  python main.py                                      Run the detector.
  python main.py --debug                              Run with verbose logging.
  python main.py --test --source FILE                Test mode on a video file.
  python main.py --deactivate                        Clear license + queue.
  python main.py --show-deadletter                   Print permanently-rejected events.
  python main.py --clear-queue                        Drop the offline retry queue.

There is no password, no login, no relogin. The license JWT (issued by the
admin's web console) is the only credential. If the backend revokes it the
detector exits 2 and the operator re-activates.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import cv2

import config
import db
import log_setup
import license as license_module
import cameras as cameras_module
import camera as camera_module
import detector as detector_module
import tracker as tracker_module
import api
import events as events_module

log = logging.getLogger("laventra")
_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    print()
    log.info("Stopping…")
    _shutdown = True


def _detect_gpu() -> bool:
    """Return True if a CUDA-capable GPU is available."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ─── CLI ───────────────────────────────────────────────────────────────────
def _args():
    p = argparse.ArgumentParser(prog="laventra-detector")
    p.add_argument("--activate", metavar="CODE",
                   help="Exchange an activation code for a license and exit")
    p.add_argument("--api-url", metavar="URL",
                   help="Backend API URL (used with --activate; saved for later)")
    p.add_argument("--select-camera", action="store_true",
                   help="Pick which camera to stream and exit")
    p.add_argument("--deactivate", action="store_true",
                   help="Clear local license + queue and exit")
    p.add_argument("--status", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--test", action="store_true",
                   help="Test mode: live preview window, no API posting")
    p.add_argument("--source", default=None,
                   help="Video file or camera index for --test "
                        "(default: selected camera)")
    p.add_argument("--clear-queue", action="store_true")
    p.add_argument("--show-deadletter", action="store_true")
    p.add_argument("--add-camera", metavar="URL",
                   help="Register a camera locally without backend round-trip "
                        "(useful for offline testing). Use with --camera-name and "
                        "optionally --camera-id.")
    p.add_argument("--camera-name", metavar="NAME", default=None)
    p.add_argument("--camera-id",   metavar="ID",   type=int, default=None)
    return p.parse_args()


# ─── Sub-commands ──────────────────────────────────────────────────────────
def _cmd_activate(args) -> int:
    if not args.api_url:
        api_url = db.kv_get("api_url", "")
        if not api_url:
            try:
                api_url = input("  Backend API URL: ").strip()
            except (EOFError, KeyboardInterrupt):
                return 1
    else:
        api_url = args.api_url
    if not api_url:
        log.error("No --api-url and none stored — cannot activate")
        return 1

    log.info(f"Activating against {api_url}…")
    try:
        claims = license_module.activate(api_url=api_url,
                                         activation_code=args.activate)
    except license_module.ActivationError as e:
        log.error(f"❌ Activation failed: {e}")
        return 1
    except license_module.LicenseInvalid as e:
        log.error(f"❌ License invalid: {e}")
        return 1

    print()
    print("─" * 52)
    print(f"  ✅ Activated as device #{claims.get('sub')}")
    print(f"     Lavvaggio #{claims.get('lavvaggio_id')}")
    print(f"     Expires   {license_module._ts_to_iso(claims.get('exp'))}")
    print("─" * 52)
    print()

    log.info("Fetching cameras…")
    try:
        cameras_module.fetch_remote()
    except cameras_module.CameraFetchError as e:
        log.warning(f"Could not fetch cameras now: {e}")
        log.info("Run `python main.py --select-camera` once the backend is reachable.")
        return 0

    cam = cameras_module.interactive_select()
    if not cam:
        log.info("No camera selected — run `python main.py --select-camera` later.")
    return 0


def _cmd_add_camera(args) -> int:
    """Register a camera locally — no backend call. For testing + offline provisioning."""
    if not args.add_camera:
        return 1
    # Assign next-available id if not provided
    cam_id = args.camera_id
    if cam_id is None:
        existing = [c["id"] for c in db.cameras_list()]
        cam_id = (max(existing) + 1) if existing else 1
    name = args.camera_name or f"Camera {cam_id}"

    # Merge with any existing cameras so this acts as an additive registration.
    existing = db.cameras_list()
    by_id = {c["id"]: c for c in existing}
    by_id[cam_id] = {
        "id":         cam_id,
        "name":       name,
        "stream_url": args.add_camera,
        "kind":       _guess_kind(args.add_camera),
    }
    db.cameras_replace(list(by_id.values()))
    db.kv_set("camera_id", cam_id)
    db.kv_set("camera_url", args.add_camera)
    print()
    print(f"  ✅ Added camera #{cam_id} — {name}")
    print(f"     URL: {args.add_camera}")
    print(f"     Selected as active camera.")
    print()
    return 0


def _guess_kind(url: str) -> str:
    u = url.lower()
    if u.startswith("rtsp://"):
        return "rtsp"
    if u.startswith("http"):
        return "mjpeg"
    return "webcam"


def _cmd_select_camera() -> int:
    if not db.has_license():
        log.error("No license — run: python main.py --activate CODE")
        return 1
    cam = cameras_module.interactive_select()
    return 0 if cam else 1


def _cmd_deactivate() -> int:
    try:
        confirm = input("⚠️  Clear license, cameras and offline queue? Type YES: ")
    except (EOFError, KeyboardInterrupt):
        return 0
    if confirm.strip() != "YES":
        log.info("Cancelled")
        return 0
    db.license_clear()
    db.cameras_replace([])
    n = db.queue_clear()
    db.kv_delete("camera_id")
    db.kv_delete("camera_url")
    log.info(f"✅ Cleared license, cameras, and {n} queued event(s)")
    return 0


def _cmd_status() -> int:
    s = db.status_summary()
    print()
    print("─" * 52)
    print(f"  LAVENTRA DETECTOR  v{config.VERSION}")
    print("─" * 52)
    print(f"  API URL         : {s['api_url']}")
    print(f"  License         : {'✅ ' if s['license_set']=='yes' else '❌ '}{s['license_set']}"
          f"  ({'REVOKED' if s['license_revoked']=='yes' else 'ok'})")
    print(f"  License expires : {s['license_expires']}")
    print(f"  License ID      : {s['license_id']}")
    print(f"  Device ID       : {s['device_id']}")
    print(f"  Lavvaggio       : {s['lavvaggio_id']}")
    cam = cameras_module.selected()
    if cam:
        print(f"  Camera          : #{cam['id']} {cam['name']} → {cam['stream_url']}")
    else:
        print(f"  Camera          : — (run --select-camera)")
    print(f"  Offline queue   : {s['queue']} pending")
    print(f"  Dead-letter     : {s['dead_letter']}")
    print(f"  Ready           : {'✅ yes' if db.has_license() and cam else '❌ no'}")
    print("─" * 52)
    print()
    return 0


def _cmd_show_deadletter() -> int:
    rows = db.dead_letter_list()
    if not rows:
        print("  Dead-letter is empty.")
        return 0
    print()
    print(f"  {len(rows)} dead-lettered event(s):")
    print("─" * 80)
    for r in rows:
        print(f"  [{r['created_at']}] HTTP {r['error_code']}  "
              f"plate={r['plate']} started={r['started_at']}")
        if r["error_body"]:
            print(f"      body: {r['error_body'][:160]}")
    print("─" * 80)
    return 0


# ─── Test mode (file or camera, no API) ────────────────────────────────────
def _cmd_test(source=None) -> int:
    import os
    if source is None:
        cam = cameras_module.selected()
        source = cam["stream_url"] if cam else "0"
    src = int(source) if str(source).strip().isdigit() else source
    is_file = isinstance(src, str) and os.path.isfile(src)

    print()
    print("═" * 52)
    print("  LAVENTRA — Test Mode")
    print("═" * 52)
    print(f"  Source : {source}")
    print(f"  Type   : {'video file' if is_file else 'camera feed'}")
    print("  Press Q to quit")
    print("═" * 52)
    print()

    log.info("Loading AI models…")
    use_gpu = _detect_gpu()
    if use_gpu:
        log.info("✅ CUDA detected — using GPU")
    detector = detector_module.PlateDetector(
        vehicle_conf=0.25,
        plate_conf=0.20,
        ocr_min_conf=0.40,
        cooldown_sec=0 if is_file else config.OCR_COOLDOWN_S,
        use_gpu=use_gpu,
    )
    if not detector.ready:
        log.error("AI models failed to load — pip install -r requirements.txt")
        return 1

    cap = cv2.VideoCapture(src) if is_file else camera_module._open_capture(src)
    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        return 1
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    seen_plates: dict[str, int] = {}
    t_start = time.time()
    frame_n = 0

    while not _shutdown:
        t_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            log.info("End of stream.")
            break
        frame_n += 1
        if frame_n % 3 != 0:
            continue
        dets = detector.detect(frame)
        for det in dets:
            if det.get("plate"):
                p = det["plate"]
                seen_plates[p] = seen_plates.get(p, 0) + 1
                log.info(f"🚗  #{det['track_id']}  {p:<12}  type={det['type']:<10} "
                         f"ocr={det['ocr_conf']:.0%}  (seen {seen_plates[p]}×)")

        h_f, w_f = frame.shape[:2]
        vis = cv2.resize(detector.draw(frame, dets),
                         (480, int(h_f * 480 / w_f)))
        cv2.imshow("Laventra — Test Mode", vis)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
        if is_file:
            used = time.time() - t_frame
            sleep = (1.0 / fps_src) - used
            if sleep > 0:
                time.sleep(sleep)

    cap.release()
    cv2.destroyAllWindows()
    elapsed = int(time.time() - t_start)
    print()
    print("─" * 52)
    log.info(f"Test done — {frame_n} frames  {elapsed}s  {len(seen_plates)} unique plate(s)")
    for plate, count in sorted(seen_plates.items(), key=lambda x: -x[1]):
        log.info(f"  {plate:<14}  ×{count}")
    print("─" * 52)
    return 0


# ─── Main runtime ──────────────────────────────────────────────────────────
def _run_detector() -> int:
    # 1) License check
    try:
        claims = license_module.ensure_valid()
    except license_module.LicenseRevoked as e:
        log.error(f"❌ {e}")
        log.error("   Run: python main.py --activate <new-code>")
        return 2
    except license_module.LicenseError as e:
        log.error(f"❌ {e}")
        return 1

    # 2) Camera selection
    cam = cameras_module.selected()
    if not cam:
        log.warning("No camera selected — refreshing list from backend…")
        try:
            cameras_module.fetch_remote()
        except cameras_module.CameraFetchError as e:
            log.error(f"Could not fetch cameras: {e}")
            log.error("  → Run: python main.py --select-camera")
            return 1
        cams = cameras_module.local_list()
        if len(cams) == 1:
            cam = cameras_module.select(cams[0]["id"])
        else:
            log.error("Run: python main.py --select-camera")
            return 1

    cam_url = cam["stream_url"]
    cam_id = cam["id"]

    print()
    print("═" * 52)
    print(f"  LAVENTRA DETECTOR  v{config.VERSION}  — Running")
    print("═" * 52)
    print(f"  Device     : #{claims.get('sub')}  ({license_module.hostname()})")
    print(f"  Lavvaggio  : #{claims.get('lavvaggio_id')}")
    print(f"  Camera     : #{cam_id} {cam['name']} → {cam_url}")
    print(f"  API        : {db.kv_get('api_url')}")
    print(f"  Queue      : {db.queue_count()} pending")
    print("═" * 52)
    print()

    # 3) Boot AI pipeline
    log.info("Loading AI models…")
    use_gpu = _detect_gpu()
    if use_gpu:
        log.info("✅ CUDA detected — using GPU")
    detector = detector_module.PlateDetector(use_gpu=use_gpu)
    if not detector.ready:
        log.error("AI models failed to load — pip install -r requirements.txt")
        return 1

    cam_stream = camera_module.CameraStream(cam_url)
    cam_stream.start()

    # Create the checker first so the dispatcher/flusher can reference it: a
    # license rejection on a live POST pauses detection immediately (via the
    # dispatcher hook), and the flusher skips retries while paused.
    checker = license_module.LicenseChecker()

    dispatcher = events_module.EventDispatcher(
        on_license_rejected=checker.trigger_recheck,
    )
    dispatcher.start()

    flusher = api.QueueFlusher(license_ok_fn=lambda: checker.is_valid)
    flusher.start()

    heartbeat = api.HeartbeatThread(
        camera_status_fn=lambda: cam_stream.connected,
        camera_id=cam_id,
    )
    heartbeat.start()

    checker.start()

    log.info("✅ Detector running — Ctrl+C to stop")
    log.info(f"   Visit grace = {config.PRESENCE_GRACE_S}s  "
             f"(one event per vehicle per wash visit)")
    log.info("─" * 52)

    visit_tracker    = tracker_module.VisitTracker()
    last_stats_ts    = time.time()
    last_visit_check_ts = 0.0
    license_was_valid   = True   # track transitions for log messages

    while not _shutdown:
        try:
            license_ok = checker.is_valid

            # ── License-invalid mode ─────────────────────────────────────────
            if not license_ok:
                if license_was_valid:
                    # Transition valid → invalid: flush any in-flight visits
                    for evt in visit_tracker.flush():
                        evt["camera_id"] = cam_id
                        dispatcher.submit(evt)
                    license_was_valid = False
                time.sleep(0.5)
                continue

            # ── Transition invalid → valid ───────────────────────────────────
            if not license_was_valid:
                log.info("▶️  License valid — detection resumed")
                license_was_valid = True

            # ── Normal detection ─────────────────────────────────────────────
            frame = cam_stream.read()
            if frame is None:
                if time.time() - last_visit_check_ts >= 1.0:
                    for evt in visit_tracker.collect_completed():
                        evt["camera_id"] = cam_id
                        dispatcher.submit(evt)
                    last_visit_check_ts = time.time()
                time.sleep(0.05)
                continue

            detections = detector.detect(frame)
            for det in detections:
                visit_tracker.update(
                    det["track_id"], det["type"],
                    det.get("plate"), det.get("ocr_conf", 0.0),
                )
                if det.get("plate"):
                    log.info(
                        f"🚗  #{det['track_id']}  {det['plate']:<12}  "
                        f"type={det['type']:<10}  ocr={det['ocr_conf']:.0%}"
                    )

            for evt in visit_tracker.collect_completed():
                evt["camera_id"] = cam_id
                dispatcher.submit(evt)
            last_visit_check_ts = time.time()

            if time.time() - last_stats_ts >= 60:
                s = dispatcher.stats
                log.info(
                    f"📊  sent={s['sent']}  queued={s['queued']}  "
                    f"dead={s['dead_letter']}  "
                    f"active_visits={visit_tracker.active_count}  "
                    f"pending={db.queue_count()}  "
                    f"camera={'✅' if cam_stream.connected else '❌'}"
                )
                last_stats_ts = time.time()

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Main loop error (continuing): {e}")
            time.sleep(0.1)

    # Finalise any in-flight visits before we stop accepting frames
    log.info("Finalising active visits…")
    for evt in visit_tracker.flush():
        evt["camera_id"] = cam_id
        dispatcher.submit(evt)

    log.info("Shutting down…")
    for stopper in (cam_stream.stop, dispatcher.stop, flusher.stop,
                    heartbeat.stop, checker.stop):
        try:
            stopper()
        except Exception as e:
            log.error(f"Stopper error: {e}")

    s = dispatcher.stats
    print()
    print("─" * 52)
    log.info(f"Session — sent={s['sent']}  queued={s['queued']}  "
             f"dead-lettered={s['dead_letter']}")
    log.info("Goodbye.")
    print("─" * 52)
    return 0


# ─── Entrypoint ────────────────────────────────────────────────────────────
def _force_utf8_stdio() -> None:
    """Switch stdout/stderr to UTF-8 on Windows where the default is cp1252."""
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> int:
    _force_utf8_stdio()
    args = _args()
    log_setup.setup(debug=args.debug)
    db.init()

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    if args.deactivate:
        return _cmd_deactivate()
    if args.clear_queue:
        n = db.queue_clear()
        print(f"✅ Cleared {n} queued event(s)")
        return 0
    if args.show_deadletter:
        return _cmd_show_deadletter()
    if args.add_camera:
        return _cmd_add_camera(args)
    if args.activate:
        if db.has_license():
            log.warning("⚠️  A license is already installed.")
            log.warning("   Run: python main.py --deactivate")
            log.warning("   Then: python main.py --activate <new-code>")
            return 1
        return _cmd_activate(args)
    if args.select_camera:
        return _cmd_select_camera()
    if args.status:
        return _cmd_status()
    if args.test:
        return _cmd_test(source=args.source)

    if not db.has_license():
        log.error("No license installed.")
        log.error("  → Run: python main.py --activate <code>")
        log.error("  (Dev mode without backend: use tools/generate_dev_license.py)")
        return 1

    # Friendly hint when running with a dev-minted license
    lic = db.license_get() or {}
    if str(lic.get("license_id", "")).startswith("dev-"):
        log.info("ℹ️  Running with a DEV license (locally minted, not from backend).")
        log.info("   To switch to a real backend-issued license:")
        log.info("     python main.py --deactivate")
        log.info("     python main.py --activate <code-from-admin>")

    return _run_detector()


if __name__ == "__main__":
    sys.exit(main())
