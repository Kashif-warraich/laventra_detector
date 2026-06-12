"""Tests for license.verify_jwt — signature, expiry, audience, issuer,
fingerprint mismatch, and tamper resistance.

We generate an ephemeral RSA keypair in the temp dir and point
config.PUBLIC_KEY_PATH at the new public half, so the real
license_public_key.pem on disk is not touched.
"""
import time
from pathlib import Path

import jwt as _jwt
import pytest

import config
import license as license_module


# ─── fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def temp_keypair(tmp_path, monkeypatch):
    """Generate a fresh RSA keypair and route license.py to the public half."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub_path = tmp_path / "test_pub.pem"
    pub_path.write_bytes(pub_pem)
    monkeypatch.setattr(config, "PUBLIC_KEY_PATH", pub_path)
    # Clear license.py's cached public key (it's loaded once per process).
    monkeypatch.setattr(license_module, "_public_key_cache", None)
    return priv_pem, pub_pem


def _mint(priv_pem: bytes, **overrides) -> str:
    fp = license_module.device_fingerprint()
    now = int(time.time())
    claims = {
        "iss": config.JWT_ISSUER,
        "aud": config.JWT_AUDIENCE,
        "sub": "1",
        "iat": now,
        "nbf": now,
        "exp": now + 3600,
        "license_id": "test-1",
        "lavvaggio_id": 1,
        "device_fingerprint": fp,
        "features": ["car_wash_events"],
    }
    claims.update(overrides)
    return _jwt.encode(claims, priv_pem, algorithm=config.JWT_ALGORITHM)


# ─── tests ─────────────────────────────────────────────────────────────────
def test_valid_jwt_verifies(temp_keypair):
    priv, _ = temp_keypair
    tok = _mint(priv)
    claims = license_module.verify_jwt(tok)
    assert claims["sub"] == "1"
    assert claims["lavvaggio_id"] == 1


def test_tampered_signature_rejected(temp_keypair):
    priv, _ = temp_keypair
    tok = _mint(priv)
    parts = tok.split(".")
    tampered = ".".join([parts[0], parts[1], parts[2][:-2] + "AA"])
    with pytest.raises(license_module.LicenseInvalid):
        license_module.verify_jwt(tampered)


def test_expired_jwt_rejected(temp_keypair):
    priv, _ = temp_keypair
    now = int(time.time())
    # exp 1h ago; account for clock-skew leeway (set well past it)
    tok = _mint(priv, iat=now - 7200, nbf=now - 7200, exp=now - 3600)
    with pytest.raises(license_module.LicenseInvalid):
        license_module.verify_jwt(tok)


def test_wrong_audience_rejected(temp_keypair):
    priv, _ = temp_keypair
    tok = _mint(priv, aud="some-other-app")
    with pytest.raises(license_module.LicenseInvalid):
        license_module.verify_jwt(tok)


def test_wrong_issuer_rejected(temp_keypair):
    priv, _ = temp_keypair
    tok = _mint(priv, iss="not-laventra")
    with pytest.raises(license_module.LicenseInvalid):
        license_module.verify_jwt(tok)


def test_wrong_fingerprint_rejected(temp_keypair):
    priv, _ = temp_keypair
    tok = _mint(priv, device_fingerprint="ff" * 32)
    with pytest.raises(license_module.LicenseInvalid):
        license_module.verify_jwt(tok)


def test_fingerprint_check_can_be_skipped(temp_keypair):
    priv, _ = temp_keypair
    tok = _mint(priv, device_fingerprint="ff" * 32)
    # check_fingerprint=False: useful for diagnostics
    claims = license_module.verify_jwt(tok, check_fingerprint=False)
    assert claims["device_fingerprint"] == "ff" * 32


def test_signed_by_wrong_key_rejected(temp_keypair, tmp_path, monkeypatch):
    """Mint with a different private key; should fail because public doesn't match."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_priv = other.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    tok = _mint(other_priv)
    with pytest.raises(license_module.LicenseInvalid):
        license_module.verify_jwt(tok)


def test_missing_public_key_fails_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_KEY_PATH", tmp_path / "does_not_exist.pem")
    monkeypatch.setattr(license_module, "_public_key_cache", None)
    with pytest.raises(license_module.LicenseInvalid):
        license_module.verify_jwt("anything.at.all")
