"""
Tests for detector.py plate-string helpers and OCR cooldown bookkeeping.

We exercise the pure logic only — no YOLO / OCR models are loaded. The cooldown
tests build a bare PlateDetector via object.__new__ and set just the attributes
the cooldown methods touch, so the heavy __init__ (model download) never runs.
"""
import time

import config
import detector as det


# ─── _clean ───────────────────────────────────────────────────────────────────
def test_clean_strips_non_alphanumeric_and_uppercases():
    assert det._clean("ab 123-cd") == "AB123CD"


def test_clean_drops_punctuation_and_spaces():
    assert det._clean("a!b@1#2$3%c^d") == "AB123CD"


def test_clean_returns_none_when_below_min_len():
    assert det._clean("AB1") is None          # < PLATE_MIN_LEN (5)


def test_clean_returns_none_when_all_symbols():
    assert det._clean("!!!---") is None


# ─── _looks_like_plate ─────────────────────────────────────────────────────────
def test_looks_like_plate_accepts_valid_italian():
    assert det._looks_like_plate("AB123CD") is True


def test_looks_like_plate_rejects_all_letters():
    assert det._looks_like_plate("ABCDEFG") is False     # no digit


def test_looks_like_plate_rejects_all_digits():
    assert det._looks_like_plate("1234567") is False     # no letter


def test_looks_like_plate_rejects_blacklisted_overlay_text():
    for bad in config.OCR_BLACKLIST:
        assert det._looks_like_plate(bad) is False


def test_looks_like_plate_enforces_length_bounds():
    assert det._looks_like_plate("A1") is False                       # < min
    assert det._looks_like_plate("A1B2C3D4E5F6") is False             # 12 > max (10)


# ─── OCR cooldown bookkeeping (no models loaded) ───────────────────────────────
def _bare_detector(cooldown=3.0):
    d = object.__new__(det.PlateDetector)
    d._seen = {}
    d._cooldown_sec = cooldown
    d._last_prune_ts = time.time()
    return d


def test_on_cooldown_false_for_unseen_track():
    assert _bare_detector().on_cooldown(7) is False


def test_mark_seen_puts_track_on_cooldown():
    d = _bare_detector(cooldown=100)
    d.mark_seen(7)
    assert d.on_cooldown(7) is True


def test_cooldown_expires_after_window():
    d = _bare_detector(cooldown=0.05)
    d.mark_seen(7)
    time.sleep(0.1)
    assert d.on_cooldown(7) is False


def test_prune_drops_stale_tracks_only():
    d = _bare_detector(cooldown=1.0)
    now = time.time()
    d._seen = {1: now - 1000, 2: now}     # track 1 is ancient, track 2 is fresh
    d._last_prune_ts = now - 120          # force the 60s-gated prune to run
    d._maybe_prune_seen()
    assert 1 not in d._seen
    assert 2 in d._seen


def test_prune_is_rate_limited():
    d = _bare_detector(cooldown=1.0)
    d._seen = {1: time.time() - 1000}
    d._last_prune_ts = time.time()        # just pruned → should be a no-op
    d._maybe_prune_seen()
    assert 1 in d._seen                    # not pruned because <60s elapsed
