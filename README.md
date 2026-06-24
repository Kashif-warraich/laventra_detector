# Laventra Detector — Setup Guide

Edge-device service that watches the car wash entrance, detects vehicles, and
reads license plates. It posts events to the Laventra API.

Requires **Python 3.11**. There's no login — the license JWT issued by the admin
is the only credential.

---

## General

### 1. License key

The detector verifies tokens signed by the backend using `license_public_key.pem`
(already in this repo). Only replace it if the backend rotated its keys — grab the
new one from the API's `config/license_keys/public_key.pem`.

### 2. Activation

Activate once per machine, using a code the admin issues from the web console:

```bash
python main.py --activate LAVN-XXXX-XXXX-XXXX --api-url http://localhost:3000
```

---

## Setup (macOS, Linux & Windows)

```bash
# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the vehicle detection model (the plate model downloads on first run)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Then activate the device (see General → Activation) and run it:

```bash
python main.py
```

---

## Common Commands

```bash
python main.py --status                    # license, camera, and queue status
python main.py --select-camera             # pick which camera to use
python main.py --test --source video.mp4   # test on a video file, no API posting
python main.py --debug                     # verbose logging
python main.py --deactivate                # wipe the license before moving machines
```

---

## Troubleshooting

* **`Signature verification failed`** — the public key doesn't match the backend.
  Copy the API's `config/license_keys/public_key.pem` here as `license_public_key.pem`.
* **`already bound to a different device`** — the license was activated on another
  machine. Ask an admin to clear the device fingerprint, then run `--activate` again.
</content>
