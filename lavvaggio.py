import logging
import requests
import db
import auth

log = logging.getLogger("laventra")


def _v1() -> str:
    url = db.session_get("api_url", "").rstrip("/")
    if url.endswith("/api/v1"):
        url = url[: -len("/api/v1")]
    return f"{url}/api/v1"


def fetch_devices(lavvaggio_id: int) -> list:
    try:
        r = auth.get_session().get(
            f"{_v1()}/devices",
            params={"lavvaggio_id": lavvaggio_id},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("data", [])
        if r.status_code == 401:
            if auth.silent_relogin():
                return fetch_devices(lavvaggio_id)
        log.error(f"Failed to fetch devices: HTTP {r.status_code}")
    except Exception as e:
        log.error(f"Error fetching devices: {e}")
    return []


def fetch_lavvaggios() -> list:
    try:
        r = auth.get_session().get(f"{_v1()}/lavvaggios", timeout=10)
        if r.status_code == 200:
            return r.json().get("data", [])
        if r.status_code == 401:
            log.warning("Token expired — re-logging in…")
            if auth.silent_relogin():
                return fetch_lavvaggios()
        log.error(f"Failed to fetch lavvaggios: HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach API")
    except Exception as e:
        log.error(f"Error fetching lavvaggios: {e}")
    return []


def interactive_select() -> bool:
    print()
    print("─" * 52)
    print("  Select Lavvaggio")
    print("─" * 52)
    print()
    log.info("Fetching lavvaggios…")

    lavvaggios = fetch_lavvaggios()

    if not lavvaggios:
        log.error("No lavvaggios found")
        try:
            manual = input("  Enter lavvaggio ID manually (or Enter to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if not manual.isdigit():
            return False
        lav_id   = int(manual)
        lav_name = f"Lavvaggio #{lav_id}"
        try:
            dev = input("  Device ID (or Enter to skip): ").strip()
            device_id = int(dev) if dev.isdigit() else None
        except (EOFError, KeyboardInterrupt):
            device_id = None
        _save(lav_id, lav_name, device_id)
        return True

    print(f"  Found {len(lavvaggios)} lavvaggio(s):\n")
    for i, lav in enumerate(lavvaggios, 1):
        status = lav.get("status", "?")
        marker = "🟢" if status == "active" else "🔴"
        partners = lav.get("partners", [])
        partner_count = f"{len(partners)} partner(s)"
        print(
            f"    {i:2}. {marker}  {lav['name']}"
            f"  |  {lav.get('city','?')}"
            f"  |  {partner_count}"
            f"  |  {status}"
        )
    print()

    while True:
        try:
            raw = input("  Enter number: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if raw.isdigit() and 1 <= int(raw) <= len(lavvaggios):
            chosen = lavvaggios[int(raw) - 1]
            break
        log.warning(f"Enter a number between 1 and {len(lavvaggios)}")

    # Fetch devices for the chosen lavvaggio
    device_id = _select_device(chosen["id"])

    _save(chosen["id"], chosen["name"], device_id)
    return True


def _select_device(lavvaggio_id: int):
    print()
    log.info("Fetching devices for this lavvaggio…")
    devices = fetch_devices(lavvaggio_id)

    if not devices:
        log.warning("No devices found for this lavvaggio")
        try:
            raw = input("  Enter device ID manually (or Enter to skip): ").strip()
            return int(raw) if raw.isdigit() else None
        except (EOFError, KeyboardInterrupt):
            return None

    if len(devices) == 1:
        dev = devices[0]
        log.info(f"✅ Device: {dev['serial_number']} (id={dev['id']})")
        return dev["id"]

    print(f"  Found {len(devices)} device(s):")
    print()
    for i, dev in enumerate(devices, 1):
        status = dev.get("status", "?")
        marker = "🟢" if status == "active" else "🔴"
        print(f"    {i:2}. {marker}  {dev['serial_number']}  |  fw={dev.get('firmware_version','?')}")
    print(f"    {len(devices)+1:2}.     Skip (no device)")
    print()

    while True:
        try:
            raw = input("  Enter number: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.isdigit():
            n = int(raw)
            if n == len(devices) + 1:
                return None
            if 1 <= n <= len(devices):
                return devices[n - 1]["id"]
        log.warning(f"Enter a number between 1 and {len(devices)+1}")


def _save(lav_id: int, lav_name: str, device_id) -> None:
    db.session_set("lavvaggio_id",   lav_id)
    db.session_set("lavvaggio_name", lav_name)
    db.session_set("device_id",      device_id or "")
    log.info(f"✅ Lavvaggio → {lav_name} (id={lav_id}, device={device_id})")


def load_from_db() -> tuple:
    lav_id   = int(db.session_get("lavvaggio_id",   "0") or 0)
    lav_name = db.session_get("lavvaggio_name",     "Unknown")
    dev_raw  = db.session_get("device_id",          "")
    dev_id   = int(dev_raw) if dev_raw and dev_raw.isdigit() else None
    return lav_id, lav_name, dev_id