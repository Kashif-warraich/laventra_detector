"""
Dev-only helper to mint test licenses before the Rails backend implements
the /license/activate endpoint.

Usage
─────
  # 1) one-time: generate an RSA keypair (private stays on your dev machine,
  #              public goes into the detector beside config.PUBLIC_KEY_PATH)
  python tools/generate_dev_license.py keygen

  # 2) mint a test license for this machine (uses local fingerprint)
  python tools/generate_dev_license.py mint --device-id 1 --lavvaggio-id 1 \
         --days 30 --out dev_license.jwt

  # 3) install the license without hitting the backend
  python tools/generate_dev_license.py install --jwt-file dev_license.jwt \
         --api-url http://localhost:3000

After step 3, `python main.py` works against the local DB just like a real
activated device. When the backend is ready, replace step 3 with:
  python main.py --activate XXXX-YYYY-ZZZZ
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdio on Windows so the arrow characters in our prints render.
for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        try:
            _r(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Make sure the parent dir (laventra_detector/) is on sys.path so `config` etc. import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


PRIVATE_KEY_PATH = config.ROOT_DIR / "dev_license_private_key.pem"


def cmd_keygen(args):
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    if PRIVATE_KEY_PATH.exists() and not args.force:
        print(f"Refusing to overwrite {PRIVATE_KEY_PATH} — pass --force to rotate")
        sys.exit(1)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    PRIVATE_KEY_PATH.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    config.PUBLIC_KEY_PATH.write_bytes(key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    print(f"  private → {PRIVATE_KEY_PATH}")
    print(f"  public  → {config.PUBLIC_KEY_PATH}")
    print()
    print("  Share license_public_key.pem with the backend team — they store it as")
    print("  the public half. Backend keeps its OWN private key for prod licenses.")


def cmd_mint(args):
    import jwt as _jwt
    import license as lic_module

    if not PRIVATE_KEY_PATH.exists():
        print(f"Run keygen first — no private key at {PRIVATE_KEY_PATH}")
        sys.exit(1)

    fingerprint = args.fingerprint or lic_module.device_fingerprint()
    now = int(time.time())
    exp = now + args.days * 24 * 3600
    claims = {
        "iss":    config.JWT_ISSUER,
        "aud":    config.JWT_AUDIENCE,
        "sub":    str(args.device_id),
        "iat":    now,
        "nbf":    now,
        "exp":    exp,
        "license_id":         args.license_id or f"dev-{now}",
        "lavvaggio_id":       args.lavvaggio_id,
        "device_fingerprint": fingerprint,
        "features":           ["car_wash_events", "heartbeat"],
    }
    token = _jwt.encode(claims, PRIVATE_KEY_PATH.read_bytes(),
                        algorithm=config.JWT_ALGORITHM)

    Path(args.out).write_text(token, encoding="utf-8")
    exp_iso = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    print(f"  license → {args.out}")
    print(f"  device_id    = {args.device_id}")
    print(f"  lavvaggio_id = {args.lavvaggio_id}")
    print(f"  fingerprint  = {fingerprint[:24]}…")
    print(f"  expires      = {exp_iso}")


def cmd_install(args):
    import license as lic_module
    import db

    db.init()
    token = Path(args.jwt_file).read_text(encoding="utf-8").strip()
    claims = lic_module.verify_jwt(token)
    db.kv_set("api_url", args.api_url.rstrip("/"))
    db.license_save(
        token,
        license_id=claims.get("license_id", ""),
        device_id=int(claims.get("sub") or 0),
        lavvaggio_id=int(claims.get("lavvaggio_id") or 0),
        issued_at=lic_module._ts_to_iso(claims.get("iat")),
        expires_at=lic_module._ts_to_iso(claims.get("exp")),
    )
    db.license_mark_refresh()
    print(f"  Installed — device_id={claims.get('sub')} "
          f"lavvaggio_id={claims.get('lavvaggio_id')}")
    print(f"  api_url = {args.api_url}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("keygen", help="Generate RSA keypair for dev signing")
    pk.add_argument("--force", action="store_true")
    pk.set_defaults(func=cmd_keygen)

    pm = sub.add_parser("mint", help="Sign a dev license JWT")
    pm.add_argument("--device-id", type=int, required=True)
    pm.add_argument("--lavvaggio-id", type=int, required=True)
    pm.add_argument("--days", type=int, default=30)
    pm.add_argument("--license-id", default=None)
    pm.add_argument("--fingerprint", default=None,
                    help="Override device_fingerprint (default: this machine)")
    pm.add_argument("--out", default="dev_license.jwt")
    pm.set_defaults(func=cmd_mint)

    pi = sub.add_parser("install", help="Install a license without backend roundtrip")
    pi.add_argument("--jwt-file", required=True)
    pi.add_argument("--api-url", required=True)
    pi.set_defaults(func=cmd_install)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
