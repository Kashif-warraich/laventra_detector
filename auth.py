import getpass
import logging
import requests
import db

log = logging.getLogger("laventra")

_session = requests.Session()
_session.headers.update({
    "Content-Type": "application/json",
    "Accept":       "application/json",
})

_login_failures = 0
LOGIN_FAILURE_LIMIT = 3


def _v1(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/api/v1"):
        url = url[: -len("/api/v1")]
    return f"{url}/api/v1"


def _apply_token(token: str) -> None:
    t = f"Bearer {token}" if not token.startswith("Bearer") else token
    _session.headers["Authorization"] = t


def _do_login(api_url: str, email: str, password: str):
    try:
        r = _session.post(
            f"{_v1(api_url)}/login",
            json={"user": {"email": email, "password": password}},
            timeout=10,
        )
        if r.status_code == 200:
            body  = r.json()
            token = (
                body.get("data", {}).get("token")
                or r.headers.get("Authorization")
                or body.get("token")
            )
            if token:
                return token
            log.error("Login OK but no token in response")
            return None
        if r.status_code == 401:
            log.error("Wrong email or password")
        else:
            log.error(f"Login failed — HTTP {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        log.error(f"Cannot reach {api_url}")
        log.error("  → Is your Rails server running?")
    except requests.exceptions.Timeout:
        log.error("Login timed out")
    except Exception as e:
        log.error(f"Login error: {e}")
    return None


def get_session() -> requests.Session:
    return _session


def interactive_login() -> bool:
    print()
    print("─" * 52)
    print("  LAVENTRA — Sign In")
    print("─" * 52)
    print()

    saved_url   = db.session_get("api_url",  "http://localhost:3000")
    saved_email = db.session_get("email",    "")

    while True:
        try:
            api_url  = input(f"  API URL   [{saved_url}]: ").strip()   or saved_url
            email    = input(f"  Email     [{saved_email}]: ").strip()  or saved_email
            password = getpass.getpass("  Password: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        if not password:
            log.warning("Password cannot be empty")
            continue

        print()
        log.info(f"Connecting to {api_url} …")
        token = _do_login(api_url, email, password)

        if token:
            db.session_set("api_url",  api_url)
            db.session_set("email",    email)
            db.session_set("password", password)
            db.session_set("token",    token)
            _apply_token(token)
            log.info(f"✅ Signed in as {email}")
            return True

        print()
        try:
            again = input("  Try again? (y/n) [y]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if again == "n":
            return False
        print()


def silent_relogin() -> bool:
    api_url  = db.session_get("api_url",  "")
    email    = db.session_get("email",    "")
    password = db.session_get("password", "")

    if not all([api_url, email, password]):
        log.warning("Cannot re-login — credentials missing")
        return False

    log.info("Re-logging in silently…")
    token = _do_login(api_url, email, password)
    if token:
        db.session_set("token", token)
        _apply_token(token)
        log.info("✅ Re-login successful")
        return True
    log.error("Silent re-login failed")
    return False


def load_saved_token() -> bool:
    token = db.session_get("token", "")
    if token:
        _apply_token(token)
        return True
    return False


def apply_device_token() -> bool:
    """
    Switches the shared requests session to use the persistent device API token.
    Called after setup and on every normal startup — the device token does not
    expire when the user changes their password.
    """
    dev_token = db.session_get("device_token", "")
    if not dev_token:
        return False
    _apply_token(dev_token)
    return True


def device_setup(lavvaggio_id: int, serial_number: str = None) -> dict:
    """
    Calls POST /api/v1/devices/setup with the current user JWT to mint a
    persistent device API token. Returns {"ok": True, "device_id": ..., "api_token": ...}
    on success or {"ok": False, "reason": "..."} on failure.

    After success the raw token is ONLY returned once by the backend — we must
    persist it immediately.
    """
    api_url = db.session_get("api_url", "")
    if not api_url:
        return {"ok": False, "reason": "no_api_url"}

    payload = {"lavvaggio_id": lavvaggio_id}
    if serial_number:
        payload["serial_number"] = serial_number

    try:
        r = _session.post(f"{_v1(api_url)}/devices/setup", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach API for device setup")
        return {"ok": False, "reason": "unreachable"}
    except requests.exceptions.Timeout:
        log.error("Device setup timed out")
        return {"ok": False, "reason": "timeout"}
    except Exception as e:
        log.error(f"Device setup error: {e}")
        return {"ok": False, "reason": "error"}

    if r.status_code == 200:
        body = r.json().get("data", {})
        device_id = body.get("device_id")
        api_token = body.get("api_token")
        if not (device_id and api_token):
            log.error("Device setup returned OK but missing device_id/api_token")
            return {"ok": False, "reason": "bad_response"}
        return {"ok": True, "device_id": device_id, "api_token": api_token,
                "serial_number": body.get("serial_number")}

    if r.status_code == 401:
        log.error("Device setup unauthorized — session expired")
        return {"ok": False, "reason": "auth"}
    if r.status_code == 403:
        log.error("Device setup forbidden — your account cannot register devices")
        return {"ok": False, "reason": "forbidden"}
    log.error(f"Device setup failed: HTTP {r.status_code} — {r.text[:200]}")
    return {"ok": False, "reason": f"http_{r.status_code}"}


def verify_connection() -> dict:
    """
    Returns {"ok": True} or {"ok": False, "reason": "auth"|"url"|"unknown"}
    """
    api_url = db.session_get("api_url", "")
    if not api_url:
        return {"ok": False, "reason": "url"}
    try:
        r = _session.get(f"{_v1(api_url)}/lavvaggios", params={"per_page": 1}, timeout=8)
        if r.status_code == 200:
            return {"ok": True}
        if r.status_code == 401:
            return {"ok": False, "reason": "auth"}
        return {"ok": False, "reason": "unknown"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "reason": "url"}
    except requests.exceptions.Timeout:
        return {"ok": False, "reason": "url"}
    except Exception as e:
        log.error(f"Connection check error: {e}")
        return {"ok": False, "reason": "unknown"}


def interactive_relogin() -> bool:
    """
    Prompts for credentials without asking for API URL again.
    Used when token expired and silent re-login fails.
    """
    print()
    print("─" * 52)
    print("  ⚠️  Session expired — please sign in again")
    print("─" * 52)
    print()

    api_url     = db.session_get("api_url",  "http://localhost:3000")
    saved_email = db.session_get("email",    "")

    while True:
        try:
            email    = input(f"  Email     [{saved_email}]: ").strip() or saved_email
            password = getpass.getpass("  Password: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        if not password:
            log.warning("Password cannot be empty")
            continue

        print()
        log.info(f"Signing in as {email}…")
        token = _do_login(api_url, email, password)

        if token:
            db.session_set("email",    email)
            db.session_set("password", password)
            db.session_set("token",    token)
            _apply_token(token)
            log.info(f"✅ Signed in as {email}")
            return True

        print()
        try:
            again = input("  Try again? (y/n) [y]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if again == "n":
            return False
        print()


def prompt_new_api_url() -> bool:
    print()
    print("─" * 52)
    print("  ⚠️  API is not responding")
    print("─" * 52)
    print(f"  Current URL: {db.session_get('api_url')}")
    print()
    print("  1 — Enter a new API URL")
    print("  2 — Continue in offline mode")
    print()

    try:
        choice = input("  Choice [2]: ").strip() or "2"
    except (EOFError, KeyboardInterrupt):
        return False

    if choice != "1":
        log.info("Continuing in offline mode")
        return False

    email    = db.session_get("email",    "")
    password = db.session_get("password", "")

    while True:
        try:
            new_url = input("  New API URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if not new_url:
            continue
        token = _do_login(new_url, email, password)
        if token:
            db.session_set("api_url", new_url)
            db.session_set("token",   token)
            _apply_token(token)
            log.info("✅ Connected to new URL")
            return True
        # URL may be correct but credentials wrong
        log.warning("Could not log in — credentials may have changed")
        try:
            fix_creds = input("  Update credentials for this URL? (y/n) [y]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if fix_creds != "n":
            try:
                email    = input(f"  Email     [{email}]: ").strip() or email
                password = getpass.getpass("  Password: ")
            except (EOFError, KeyboardInterrupt):
                return False
            token = _do_login(new_url, email, password)
            if token:
                db.session_set("api_url",  new_url)
                db.session_set("email",    email)
                db.session_set("password", password)
                db.session_set("token",    token)
                _apply_token(token)
                log.info("✅ Connected")
                return True
        try:
            again = input("  Try another URL? (y/n) [y]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if again == "n":
            return False