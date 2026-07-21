# Laventra Detector — Setup Guide

Edge service that detects vehicles, reads license plates, and sends events to the Laventra API.

**Requirements**

* Python 3.11

---

## General

### 1. Public key

The detector verifies licenses using `license_public_key.pem`, which is included in this repository.

If the backend rotates its keys, replace this file with the latest `config/license_keys/public_key.pem` from the API.

### 2. Activate the detector

Activate the detector once using the license code provided by an administrator.

```bash
python main.py --activate LAVN-XXXX-XXXX-XXXX --api-url http://localhost:3000
```

---

## Setup (macOS, Linux & Windows)

```bash
# Create a virtual environment
python3.11 -m venv venv

# Activate it
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the detector
python main.py
```


## Useful Commands

```bash
python main.py --status
python main.py --select-camera
python main.py --test --source video.mp4
python main.py --debug
python main.py --deactivate
```
