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


class PlateDetector:
    def __init__(
        self,
        yolo_model:   str   = "yolov8n.pt",
        yolo_conf:    float = 0.40,
        ocr_min_conf: float = 0.55,
        cooldown_sec: int   = 30,
        use_gpu:      bool  = False,
    ):
        self._yolo_conf    = yolo_conf
        self._ocr_min_conf = ocr_min_conf
        self._cooldown_sec = cooldown_sec
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

    def on_cooldown(self, plate: str) -> bool:
        return time.time() - self._seen.get(plate, 0) < self._cooldown_sec

    def mark_seen(self, plate: str) -> None:
        self._seen[plate] = time.time()

    def cooldown_remaining(self, plate: str) -> int:
        return max(0, int(self._cooldown_sec - (time.time() - self._seen.get(plate, 0))))

    def detect(self, frame: np.ndarray) -> list:
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
        if scale_down < 1.0:
            small = cv2.resize(frame, (int(w_orig * scale_down), int(h_orig * scale_down)))
        else:
            small = frame

        results    = self._model(small, verbose=False)[0]
        detections = []

        for box in results.boxes:
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

            h = y2 - y1

            # Plates sit at the very bottom of the vehicle bounding box.
            # Try bottom 15% and bottom 25% — tighter crops = less noise for OCR.
            candidates = [
                frame[y2 - max(int(h * 0.15), 20) : y2, x1:x2],  # bottom 15%
                frame[y2 - max(int(h * 0.25), 30) : y2, x1:x2],  # bottom 25%
            ]

            best_plate, best_conf = None, 0.0
            for crop in candidates:
                if crop.size == 0:
                    continue
                plate, conf = self._read_plate(crop)
                if plate and conf > best_conf:
                    best_plate, best_conf = plate, conf

            if not best_plate:
                continue

            if self.on_cooldown(best_plate):
                log.debug(f"⏳ {best_plate} cooldown ({self.cooldown_remaining(best_plate)}s)")
                continue

            detections.append({
                "plate":     best_plate,
                "type":      VEHICLE_CLASSES[cls_id],
                "yolo_conf": yolo_conf,
                "ocr_conf":  best_conf,
                "box":       (x1, y1, x2, y2),
            })

        return detections

    def _read_plate(self, crop: np.ndarray) -> tuple:
        try:
            # Scale up so characters are at least 80px tall
            h, w = crop.shape[:2]
            if h < 80:
                scale = max(2, int(np.ceil(80 / h)))
                crop  = cv2.resize(crop, (w * scale, h * scale),
                                   interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # CLAHE + Otsu: best for real-world plates with uneven lighting
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
            label = f"{det['plate']}  {det['type']}  {det['ocr_conf']:.0%}"

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly = x1, max(y1 - 10, th + 4)
            cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 6, ly + 2), (0, 200, 0), -1)
            cv2.putText(out, label, (lx + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out

    @staticmethod
    def _looks_like_plate(text: str) -> bool:
        """Filter out car logos, badges, stickers — plates have both letters and digits."""
        if len(text) < 4 or len(text) > 10:
            return False
        has_letter = any(c.isalpha() for c in text)
        has_digit  = any(c.isdigit() for c in text)
        return has_letter and has_digit

    @staticmethod
    def _clean(text: str) -> str | None:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        return cleaned if len(cleaned) >= 4 else None
