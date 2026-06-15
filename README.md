# Laventra Detector

Edge-device service that detects vehicles and reads license plates at car wash entrances.

---

## Setup

**Create a virtual environment and install dependencies**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Copy the license public key from the backend**

The detector uses this to verify tokens signed by the Rails backend. Grab it from `laventra_app/config/license_keys/public_key.pem` and place it here as `license_public_key.pem`. The file already exists in this repo — only replace it if the backend rotated its keys.

**Download the vehicle detection model**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

The plate model downloads automatically on first run.

**Activate against the backend** — run this once per machine
```bash
python main.py --activate LAVN-XXXXXX-XXXXXX-XXXXXX --api-url http://localhost:3000
```

---

## Run

```bash
python main.py
```

---

## Other Commands

```bash
python main.py --status                        # check license and camera status
python main.py --select-camera                 # pick which camera to use
python main.py --test --source video.mp4       # test with a video file, no API posting
python main.py --debug                         # verbose logging
python main.py --deactivate                    # wipe license (before moving to a new machine)
```

---

## Troubleshooting

**`Signature verification failed`** — the public key doesn't match the backend's private key. Copy `laventra_app/config/license_keys/public_key.pem` here and rename it to `license_public_key.pem`.

**`already bound to a different device`** — the license was activated on another machine. Ask an admin to clear the device fingerprint in the backend DB, then run `--activate` again.
