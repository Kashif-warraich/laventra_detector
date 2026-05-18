"""
Two-stage plate-recognition pipeline.

Stage 1 — Vehicle tracker (ultralytics YOLOv8)
    The vehicle's bounding box and a stable `track_id` survive across frames
    even when the plate is briefly occluded (soap, wiper, glare). One track_id
    = one car-wash visit = one event.

Stage 2 — Plate detector (plate-fine-tuned YOLOv5)
    Runs inside each vehicle bbox to localise the plate precisely. Replaces
    the v0.1 "crop the bottom 15% and hope" heuristic, which routinely missed
    motorbikes and angled plates.

Stage 3 — OCR (PaddleOCR, recognition-only)
    PaddleOCR's recognition model reads the plate crop directly. Falls back
    to EasyOCR if Paddle isn't available on the host.

Optional single-stage mode (config.USE_TWO_STAGE = False):
    The plate detector runs on the full frame. Faster (no vehicle pass) but
    can fragment a single visit into multiple events if the plate disappears
    from a few consecutive frames.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np

import config

log = logging.getLogger("laventra")

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ─── Model download (lazy, cached) ─────────────────────────────────────────
def _ensure_plate_model() -> str:
    """
    Return a local path to the plate YOLOv5 weights, downloading from
    Hugging Face if not present. Caches under config.PLATE_MODELS_DIR.
    """
    cache_dir = Path(config.PLATE_MODELS_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # huggingface_hub manages its own cache layout under cache_dir
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed. Run: pip install -r requirements.txt"
        ) from e

    try:
        path = hf_hub_download(
            repo_id=config.PLATE_MODEL_REPO,
            filename=config.PLATE_MODEL_FILE,
            cache_dir=str(cache_dir),
        )
        return path
    except Exception as e:
        raise RuntimeError(
            f"Could not download plate model {config.PLATE_MODEL_REPO}/"
            f"{config.PLATE_MODEL_FILE}: {e}\n"
            f"  Manual install: download best.pt from\n"
            f"  https://huggingface.co/{config.PLATE_MODEL_REPO}/tree/main\n"
            f"  and place it at {cache_dir}/{config.PLATE_MODEL_FILE}"
        ) from e


# ─── OCR backend (paddle preferred, easyocr fallback) ──────────────────────
class _OCRBackend:
    name = "none"

    def read(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        raise NotImplementedError


class _PaddleOCR(_OCRBackend):
    name = "paddleocr"

    def __init__(self, use_gpu: bool = False):
        from paddleocr import PaddleOCR
        # det=False: skip text detection — we already have the plate crop.
        # use_angle_cls=True: 90/180/270 rotated plates still read correctly.
        # lang='en': Latin-script plates; PaddleOCR has multilingual support
        # if you ever need it (lang='ch', 'it'...).
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
        )

    def read(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        # PaddleOCR expects RGB
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        # det=False → treats whole image as one text line; returns one result.
        result = self._ocr.ocr(rgb, det=False, cls=True)
        if not result or not result[0]:
            return None, 0.0
        text, conf = result[0][0]
        return text, float(conf)


class _EasyOCR(_OCRBackend):
    name = "easyocr"

    def __init__(self, use_gpu: bool = False):
        import easyocr
        self._reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    def read(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        best_text, best_conf = None, 0.0
        for (_, text, conf) in self._reader.readtext(rgb):
            if conf > best_conf:
                best_text, best_conf = text, float(conf)
        return best_text, best_conf


def _build_ocr(use_gpu: bool) -> _OCRBackend:
    try:
        ocr = _PaddleOCR(use_gpu=use_gpu)
        log.info("✅ OCR backend: PaddleOCR")
        return ocr
    except Exception as e:
        log.warning(f"PaddleOCR unavailable ({e}) — falling back to EasyOCR")
    try:
        ocr = _EasyOCR(use_gpu=use_gpu)
        log.info("✅ OCR backend: EasyOCR (fallback)")
        return ocr
    except Exception as e:
        log.error(f"Both OCR backends failed: {e}")
        raise


# ─── PlateDetector ─────────────────────────────────────────────────────────
class PlateDetector:
    def __init__(
        self,
        *,
        vehicle_model:   str   = None,
        vehicle_conf:    float = None,
        plate_conf:      float = None,
        ocr_min_conf:    float = None,
        cooldown_sec:    float = None,
        use_gpu:         bool  = None,
        use_two_stage:   bool  = None,
    ):
        self._vehicle_conf = vehicle_conf if vehicle_conf is not None else config.VEHICLE_CONF_THRESHOLD
        self._plate_conf = plate_conf if plate_conf is not None else config.PLATE_CONF_THRESHOLD
        self._ocr_min_conf = ocr_min_conf if ocr_min_conf is not None else config.OCR_MIN_CONF
        self._cooldown_sec = cooldown_sec if cooldown_sec is not None else config.OCR_COOLDOWN_S
        self._two_stage = use_two_stage if use_two_stage is not None else config.USE_TWO_STAGE
        self._use_gpu = use_gpu if use_gpu is not None else config.USE_GPU

        # track_id → last_ocr_ts; pruned periodically (audit H8)
        self._seen: dict[int, float] = {}
        self._last_prune_ts = time.time()

        self._vehicle_model = None
        self._plate_model = None
        self._ocr: _OCRBackend | None = None
        self._device = "cuda" if self._use_gpu else "cpu"

        self._load_vehicle_model(vehicle_model or config.VEHICLE_MODEL)
        self._load_plate_model()
        try:
            self._ocr = _build_ocr(use_gpu=self._use_gpu)
        except Exception:
            self._ocr = None

    def _load_vehicle_model(self, name: str) -> None:
        if not self._two_stage:
            log.info("Single-stage mode — skipping vehicle model")
            return
        try:
            from ultralytics import YOLO
            log.info(f"Loading vehicle model ({name}) on {self._device}…")
            self._vehicle_model = YOLO(name)
            self._vehicle_model.to(self._device)
            log.info("✅ Vehicle model ready")
        except Exception as e:
            log.error(f"Failed to load vehicle model: {e}")

    def _load_plate_model(self) -> None:
        try:
            from ultralytics import YOLO
            log.info(f"Resolving plate model ({config.PLATE_MODEL_REPO})…")
            weights = _ensure_plate_model()
            log.info(f"Loading plate model on {self._device}…")
            self._plate_model = YOLO(weights)
            self._plate_model.to(self._device)
            log.info("✅ Plate model ready")
        except Exception as e:
            log.error(f"Failed to load plate model: {e}")

    @property
    def ready(self) -> bool:
        if not self._plate_model or not self._ocr:
            return False
        if self._two_stage and not self._vehicle_model:
            return False
        return True

    def on_cooldown(self, track_id: int) -> bool:
        return time.time() - self._seen.get(track_id, 0) < self._cooldown_sec

    def mark_seen(self, track_id: int) -> None:
        self._seen[track_id] = time.time()

    # ── public ─────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list:
        if not self.ready:
            return []
        try:
            if self._two_stage:
                return self._detect_two_stage(frame)
            return self._detect_single_stage(frame)
        except Exception as e:
            log.error(f"Detection error (frame skipped): {e}")
            return []
        finally:
            self._maybe_prune_seen()

    # ── two-stage ──────────────────────────────────────────────────────
    def _detect_two_stage(self, frame: np.ndarray) -> list:
        h_orig, w_orig = frame.shape[:2]
        scale = min(1.0, 640 / w_orig)
        small = (
            cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))
            if scale < 1.0 else frame
        )

        results = self._vehicle_model.track(small, persist=True, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0])
            cls_id = int(box.cls[0])
            vconf = float(box.conf[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            if vconf < self._vehicle_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if scale < 1.0:
                inv = 1.0 / scale
                x1, y1, x2, y2 = int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_orig - 1, x2); y2 = min(h_orig - 1, y2)

            plate, ocr_conf, plate_box = None, 0.0, None
            if not self.on_cooldown(track_id):
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size > 0:
                    plate, ocr_conf, plate_box_local = self._detect_and_read_plate(vehicle_crop)
                    if plate_box_local is not None:
                        # translate to full-frame coords for drawing
                        px1, py1, px2, py2 = plate_box_local
                        plate_box = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
                if plate:
                    self.mark_seen(track_id)
                else:
                    # Short cooldown so we don't OCR-thrash a hopeless angle
                    self._seen[track_id] = time.time() - self._cooldown_sec * 0.7

            detections.append({
                "track_id":  track_id,
                "plate":     plate,
                "type":      VEHICLE_CLASSES[cls_id],
                "yolo_conf": vconf,
                "ocr_conf":  ocr_conf,
                "box":       (x1, y1, x2, y2),
                "plate_box": plate_box,
            })
        return detections

    # ── single-stage (plate detector only) ─────────────────────────────
    def _detect_single_stage(self, frame: np.ndarray) -> list:
        results = self._plate_model.track(frame, persist=True, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0])
            pconf = float(box.conf[0])
            if pconf < self._plate_conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate, ocr_conf = None, 0.0
            if not self.on_cooldown(track_id):
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    plate, ocr_conf = self._ocr_plate(crop)
                if plate:
                    self.mark_seen(track_id)
                else:
                    self._seen[track_id] = time.time() - self._cooldown_sec * 0.7

            detections.append({
                "track_id":  track_id,
                "plate":     plate,
                "type":      "vehicle",     # plate detector doesn't classify
                "yolo_conf": pconf,
                "ocr_conf":  ocr_conf,
                "box":       (x1, y1, x2, y2),
                "plate_box": (x1, y1, x2, y2),
            })
        return detections

    # ── helpers ────────────────────────────────────────────────────────
    def _detect_and_read_plate(
        self, vehicle_crop: np.ndarray,
    ) -> tuple[str | None, float, tuple | None]:
        """Run plate detector on a vehicle crop; OCR the best plate."""
        try:
            results = self._plate_model(vehicle_crop, verbose=False)[0]
        except Exception as e:
            log.debug(f"plate detect error: {e}")
            return None, 0.0, None

        best_plate, best_conf, best_box = None, 0.0, None
        for box in results.boxes:
            pconf = float(box.conf[0])
            if pconf < self._plate_conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Pad a few pixels for OCR
            ph, pw = vehicle_crop.shape[:2]
            pad_y = max(2, int((y2 - y1) * 0.05))
            pad_x = max(2, int((x2 - x1) * 0.05))
            px1 = max(0, x1 - pad_x); py1 = max(0, y1 - pad_y)
            px2 = min(pw - 1, x2 + pad_x); py2 = min(ph - 1, y2 + pad_y)
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            text, oconf = self._ocr_plate(plate_crop)
            # Combined score weights both the plate detector and the OCR
            score = oconf * pconf
            if text and score > best_conf:
                best_plate, best_conf, best_box = text, oconf, (px1, py1, px2, py2)
        return best_plate, best_conf, best_box

    def _ocr_plate(self, crop: np.ndarray) -> tuple[str | None, float]:
        """Upscale + OCR + clean. Returns (plate or None, conf 0..1)."""
        try:
            h, w = crop.shape[:2]
            if h < 48:
                scale = max(2, int(np.ceil(48 / h)))
                crop = cv2.resize(crop, (w * scale, h * scale),
                                  interpolation=cv2.INTER_CUBIC)
            text, conf = self._ocr.read(crop)
            if not text or conf < self._ocr_min_conf:
                return None, conf
            cleaned = _clean(text)
            if not cleaned or not _looks_like_plate(cleaned):
                return None, 0.0
            return cleaned, conf
        except Exception as e:
            log.debug(f"OCR error: {e}")
            return None, 0.0

    def _maybe_prune_seen(self) -> None:
        now = time.time()
        if now - self._last_prune_ts < 60:
            return
        cutoff = now - max(self._cooldown_sec * 10, 60)
        self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
        self._last_prune_ts = now

    # ── drawing ────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            plate = det.get("plate")
            track_id = det.get("track_id", "?")
            if plate:
                label = f"#{track_id} {plate}  {det['type']}  ocr={det['ocr_conf']:.0%}"
                color = (0, 200, 0)
            else:
                label = f"#{track_id} {det['type']}  yolo={det['yolo_conf']:.0%}"
                color = (0, 160, 200)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            pb = det.get("plate_box")
            if pb:
                cv2.rectangle(out, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 255), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly = x1, max(y1 - 10, th + 4)
            cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
            cv2.putText(out, label, (lx + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out


# ─── plate string helpers (also used by tests) ─────────────────────────────
def _clean(text: str) -> str | None:
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleaned if len(cleaned) >= config.PLATE_MIN_LEN else None


def _looks_like_plate(text: str) -> bool:
    if len(text) < config.PLATE_MIN_LEN or len(text) > config.PLATE_MAX_LEN:
        return False
    if text in config.OCR_BLACKLIST:
        return False
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    return has_letter and has_digit
