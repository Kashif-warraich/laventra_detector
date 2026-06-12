"""
Central configuration for the Laventra detector.

Everything that's "a number that controls behavior" lives here so it can be
tuned without grepping for magic constants across modules.
"""
from pathlib import Path

VERSION = "0.2.0"
USER_AGENT = f"laventra-detector/{VERSION}"

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DB_PATH = ROOT_DIR / "laventra.db"
LOG_FILE = ROOT_DIR / "laventra.log"
PUBLIC_KEY_PATH = ROOT_DIR / "license_public_key.pem"

# ── API ──────────────────────────────────────────────────────────────────────
API_TIMEOUT_S = 10
HEARTBEAT_INTERVAL_S = 60
QUEUE_FLUSH_INTERVAL_S = 60
QUEUE_MAX_RETRIES = 10
QUEUE_BATCH_LIMIT = 20

# ── License ──────────────────────────────────────────────────────────────────
LICENSE_CHECK_INTERVAL_S = 24 * 60 * 60            # normal check cadence (24h)
LICENSE_RETRY_INTERVAL_S = 5 * 60                  # retry cadence when license is invalid (5min)
LICENSE_CLOCK_SKEW_S = 60                           # JWT exp/nbf leniency
JWT_ISSUER = "laventra-backend"
JWT_AUDIENCE = "laventra-detector"
JWT_ALGORITHM = "RS256"

# ── Camera ───────────────────────────────────────────────────────────────────
# When the camera drops, optionally scan the local /24 to find it at a new IP.
# Disable on customer networks where an active port scan would trip an IDS, or
# where the camera IP is pinned via a DHCP reservation.
CAMERA_AUTO_REDISCOVERY = True
CAMERA_RECONNECT_DELAY_S = 10
CAMERA_RECONNECT_MAX_DELAY_S = 60                   # cap exponential backoff
CAMERA_FAILURE_THRESHOLD = 30                       # consecutive read fails → drop+reopen
CAMERA_REDISCOVERY_COOLDOWN_S = 600                 # rate-limit subnet scans (10m)
CAMERA_MJPEG_BUF_MAX = 2_000_000                    # cap MJPEG byte buffer (2MB)
CAMERA_FRAME_MAX_AGE_S = 0.5                        # drop frames older than this in main loop
CAMERA_SCAN_PORT_TIMEOUT_S = 0.3

# ── Detection / OCR ──────────────────────────────────────────────────────────
# Stage 1 — vehicle tracker. Generic COCO model; we only use the bbox + track_id.
#   yolov8n.pt  = ~6MB, fast on CPU
#   yolov8s.pt  = ~22MB, more accurate, slower
VEHICLE_MODEL = "yolov8n.pt"
VEHICLE_CONF_THRESHOLD = 0.35

# Stage 2 — plate detector. Plate-fine-tuned YOLOv5 (keremberke/* on HF Hub).
#   yolov5n  = nano, ~4MB, fastest
#   yolov5m  = medium, ~40MB, recommended (especially with GPU)
PLATE_MODEL_REPO = "keremberke/yolov5n-license-plate"
PLATE_MODEL_FILE = "best.pt"
PLATE_CONF_THRESHOLD = 0.30
PLATE_MODELS_DIR = ROOT_DIR / "models"

# Optional: single-stage = run the plate detector directly on the full frame
# (skip the vehicle tracker). Less robust to occlusion but ~2x faster.
USE_TWO_STAGE = True

# OCR (PaddleOCR primary, EasyOCR fallback)
OCR_MIN_CONF = 0.55
OCR_COOLDOWN_S = 3                                   # min seconds between OCR attempts per track

# GPU autodetect; main.py overrides if torch.cuda.is_available().
USE_GPU = False

# ── Visit aggregation ────────────────────────────────────────────────────────
# Cars crawl through a wash tunnel; this is generous on purpose.
PRESENCE_GRACE_S = 20                                # finalise visit after this much absence
MIN_EVENT_DURATION_S = 1                             # API requires ended_at > started_at

# ── Plate format (Italy by default) ──────────────────────────────────────────
# Plate string normalisation rules + post-OCR validation regex.
# Italian standard plates: 2 letters + 3 digits + 2 letters, e.g. "AB123CD".
COUNTRY_CODE = "IT"
PLATE_FORMAT_REGEX = r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$"
PLATE_MIN_LEN = 5
PLATE_MAX_LEN = 10

# Position-aware confusable maps used by per-character voting.
# "AS_LETTER" turns a digit-looking glyph into the letter it most resembles.
# "AS_DIGIT"  turns a letter-looking glyph into the digit it most resembles.
OCR_CONFUSABLE_AS_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "4": "A"}
OCR_CONFUSABLE_AS_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2",
                           "Q": "0", "D": "0", "T": "7", "A": "4", "G": "6"}

# Plate-position kinds for the configured COUNTRY_CODE.
# "L" = letter required, "D" = digit required. Must equal PLATE_TARGET_LEN.
PLATE_POSITION_KINDS = ("L", "L", "D", "D", "D", "L", "L")    # Italian standard
PLATE_TARGET_LEN = len(PLATE_POSITION_KINDS)                   # 7

# Watermark / overlay strings OCR misreads as plates.
OCR_BLACKLIST = {
    "DR0IDCAM", "DROIDCAM",
    "IPCAMERA", "IPWEBCAM",
    "OPENCAM", "CAMAPP",
}
