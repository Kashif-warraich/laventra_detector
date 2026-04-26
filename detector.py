import re
import time
import logging
import cv2
import numpy as np

log = logging.getLogger("laventra")

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Strings that OCR commonly mis-reads as plates but are actually watermarks,
# logos, or camera overlays. Add entries as you encounter them.
_PLATE_BLACKLIST = {
    "DR0IDCAM", "DROIDCAM",   # DroidCam free watermark
    "IPCAMERA", "IPWEBCAM",   # IP Webcam watermark
    "OPENCAM",  "CAMAPP",
}


class PlateDetector:
    def __init__(
        self,
        yolo_model:   str   = "yolov8n.pt",
        yolo_conf:    float = 0.40,
        ocr_min_conf: float = 0.55,
        cooldown_sec: int   = 3,
        use_gpu:      bool  = False,
    ):
        self._yolo_conf    = yolo_conf
        self._ocr_min_conf = ocr_min_conf
        self._cooldown_sec = cooldown_sec
        # Keys are track_ids (int) — time of last successful OCR for that track
        self._seen         = {}
        self._model        = None
        self._reader       = None
        self._load_yolo(yolo_model, use_gpu)
        self._load_ocr(use_gpu)

    def _load_yolo(self, model_name: str, use_gpu: bool) -> None:
        try:
            from ultralytics import YOLO
            log.info(f"Loading YOLO ({model_name})…")
            self._model = YOLO(model_name)
            self._model.to("cuda" if use_gpu else "cpu")
            log.info("✅ YOLO ready")
        except Exception as e:
            log.error(f"Failed to load YOLO: {e}")

    def _load_ocr(self, use_gpu: bool) -> None:
        try:
            import easyocr
            log.info("Loading EasyOCR…")
            self._reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            log.info("✅ EasyOCR ready")
        except Exception as e:
            log.error(f"Failed to load EasyOCR: {e}")

    @property
    def ready(self) -> bool:
        return self._model is not None and self._reader is not None

    def on_cooldown(self, track_id: int) -> bool:
        return time.time() - self._seen.get(track_id, 0) < self._cooldown_sec

    def mark_seen(self, track_id: int) -> None:
        self._seen[track_id] = time.time()

    def cooldown_remaining(self, track_id: int) -> int:
        return max(0, int(self._cooldown_sec - (time.time() - self._seen.get(track_id, 0))))

    def detect(self, frame: np.ndarray) -> list:
        """
        Returns a list of dicts, one per tracked vehicle:
          track_id  : stable int ID across frames (from YOLO tracker)
          plate     : best plate string found this frame, or None if OCR not run / found nothing
          type      : vehicle class string
          yolo_conf : detection confidence
          ocr_conf  : OCR confidence (0.0 if plate is None)
          box       : (x1, y1, x2, y2) in original frame coordinates
        """
        if not self.ready:
            return []
        try:
            return self._detect_inner(frame)
        except Exception as e:
            log.error(f"Detection error (frame skipped): {e}")
            return []

    def _detect_inner(self, frame: np.ndarray) -> list:
        # Resize to max 640px wide before YOLO — faster inference, same accuracy
        h_orig, w_orig = frame.shape[:2]
        scale_down = min(1.0, 640 / w_orig)
        small = (
            cv2.resize(frame, (int(w_orig * scale_down), int(h_orig * scale_down)))
            if scale_down < 1.0 else frame
        )

        # track() assigns stable IDs per vehicle across consecutive frames.
        # persist=True keeps the tracker state alive between calls.
        results    = self._model.track(small, persist=True, verbose=False)[0]
        detections = []

        for box in results.boxes:
            # track_id is None on the very first frame or if the tracker lost it
            if box.id is None:
                continue
            track_id  = int(box.id[0])
            cls_id    = int(box.cls[0])
            yolo_conf = float(box.conf[0])

            if cls_id not in VEHICLE_CLASSES:
                continue
            if yolo_conf < self._yolo_conf:
                continue

            # Scale box coords back to original frame size
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if scale_down < 1.0:
                inv = 1.0 / scale_down
                x1, y1, x2, y2 = (
                    int(x1 * inv), int(y1 * inv),
                    int(x2 * inv), int(y2 * inv),
                )

            plate, ocr_conf = None, 0.0

            # Only run OCR when this track's cooldown has expired.
            # Tracks without a plate reading still appear in results so the
            # PlateTracker knows the vehicle is still present.
            if not self.on_cooldown(track_id):
                h = y2 - y1
                candidates = [
                    frame[y2 - max(int(h * 0.15), 20) : y2, x1:x2],  # bottom 15%
                    frame[y2 - max(int(h * 0.25), 30) : y2, x1:x2],  # bottom 25%
                ]
                best_plate, best_conf = None, 0.0
                for crop in candidates:
                    if crop.size == 0:
                        continue
                    p, c = self._read_plate(crop)
                    if p and c > best_conf:
                        best_plate, best_conf = p, c

                if best_plate:
                    plate, ocr_conf = best_plate, best_conf
                    self.mark_seen(track_id)  # start cooldown for next OCR attempt
                else:
                    # OCR ran but found nothing valid — start a short cooldown
                    # so we don't hammer the same bad crop every frame
                    self._seen[track_id] = time.time() - self._cooldown_sec * 0.7

            detections.append({
                "track_id":  track_id,
                "plate":     plate,       # None = vehicle present, plate not read yet
                "type":      VEHICLE_CLASSES[cls_id],
                "yolo_conf": yolo_conf,
                "ocr_conf":  ocr_conf,
                "box":       (x1, y1, x2, y2),
            })

        return detections

    def _read_plate(self, crop: np.ndarray) -> tuple:
        try:
            h, w = crop.shape[:2]
            if h < 80:
                scale = max(2, int(np.ceil(80 / h)))
                crop  = cv2.resize(crop, (w * scale, h * scale),
                                   interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            eq    = clahe.apply(gray)
            _, bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            best_text, best_conf = None, 0.0
            for img in [cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB),
                        cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)]:
                for (_, text, conf) in self._reader.readtext(img):
                    if conf < self._ocr_min_conf:
                        continue
                    cleaned = self._clean(text)
                    if cleaned and self._looks_like_plate(cleaned) and conf > best_conf:
                        best_text, best_conf = cleaned, conf

            return best_text, best_conf
        except Exception as e:
            log.debug(f"OCR error: {e}")
            return None, 0.0

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and plate labels onto a copy of frame."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            track_id = det.get("track_id", "?")
            plate    = det.get("plate")

            if plate:
                label = f"#{track_id} {plate}  {det['type']}  {det['ocr_conf']:.0%}"
                color = (0, 200, 0)
            else:
                label = f"#{track_id} {det['type']}  {det['yolo_conf']:.0%}"
                color = (0, 160, 200)   # blue-ish while reading plate

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly = x1, max(y1 - 10, th + 4)
            cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
            cv2.putText(out, label, (lx + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out

    @staticmethod
    def _looks_like_plate(text: str) -> bool:
        if len(text) < 4 or len(text) > 10:
            return False
        if text in _PLATE_BLACKLIST:
            return False
        has_letter = any(c.isalpha() for c in text)
        has_digit  = any(c.isdigit() for c in text)
        return has_letter and has_digit

    @staticmethod
    def _clean(text: str) -> str | None:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        return cleaned if len(cleaned) >= 4 else None
