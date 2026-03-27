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
            log.info("  First run downloads ~6MB — please wait")
            self._model = YOLO(model_name)
            self._model.to("cuda" if use_gpu else "cpu")
            log.info("✅ YOLO ready")
        except Exception as e:
            log.error(f"Failed to load YOLO: {e}")

    def _load_ocr(self, use_gpu: bool) -> None:
        try:
            import easyocr
            log.info("Loading EasyOCR…")
            log.info("  First run downloads language models — please wait")
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
        results    = self._model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id    = int(box.cls[0])
            yolo_conf = float(box.conf[0])

            if cls_id not in VEHICLE_CLASSES:
                continue
            if yolo_conf < self._yolo_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop_y = y1 + (y2 - y1) // 2
            crop   = frame[crop_y:y2, x1:x2]
            if crop.size == 0:
                continue

            plate, ocr_conf = self._read_plate(crop)
            if not plate:
                continue

            if self.on_cooldown(plate):
                log.debug(f"⏳ {plate} cooldown ({self.cooldown_remaining(plate)}s)")
                continue

            detections.append({
                "plate":     plate,
                "type":      VEHICLE_CLASSES[cls_id],
                "yolo_conf": yolo_conf,
                "ocr_conf":  ocr_conf,
                "box":       (x1, y1, x2, y2),
            })

        return detections

    def _read_plate(self, crop: np.ndarray) -> tuple:
        try:
            gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray  = cv2.GaussianBlur(gray, (3, 3), 0)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            rgb   = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

            best_text = None
            best_conf = 0.0

            for (_, text, conf) in self._reader.readtext(rgb):
                if conf < self._ocr_min_conf:
                    continue
                cleaned = self._clean(text)
                if cleaned and conf > best_conf:
                    best_text = cleaned
                    best_conf = conf

            return best_text, best_conf
        except Exception as e:
            log.debug(f"OCR error: {e}")
            return None, 0.0

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and plate labels onto a copy of frame."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            plate    = det["plate"]
            vtype    = det["type"]
            ocr_conf = det["ocr_conf"]

            # Vehicle box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Label background + text
            label    = f"{plate}  {vtype}  {ocr_conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly   = x1, max(y1 - 10, th + 4)
            cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 6, ly + 2), (0, 200, 0), -1)
            cv2.putText(out, label, (lx + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out

    @staticmethod
    def _clean(text: str):
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        return cleaned if len(cleaned) >= 4 else None