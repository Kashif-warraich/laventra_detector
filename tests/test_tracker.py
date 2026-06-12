"""Tests for tracker.pick_best_plate, _weighted_vote, position correction,
and the VisitTracker visit lifecycle."""
import time

import tracker


# ─── pick_best_plate ───────────────────────────────────────────────────────
def test_empty_readings_returns_none():
    plate, conf = tracker.pick_best_plate([])
    assert plate is None
    assert conf == 0.0


def test_single_valid_reading_returns_unchanged():
    plate, conf = tracker.pick_best_plate([("AB123CD", 0.92)])
    assert plate == "AB123CD"
    assert conf == 0.92


def test_voting_picks_majority_per_position():
    # 3 readings agree on AB123CD, one outlier reads ABl23CD (lowercase L instead of 1).
    # The single bad read is overruled by the majority.
    readings = [
        ("AB123CD", 0.91),
        ("AB123CD", 0.88),
        ("AB123CD", 0.95),
        ("ABL23CD", 0.60),
    ]
    plate, _ = tracker.pick_best_plate(readings)
    assert plate == "AB123CD"


def test_voting_handles_partial_disagreement():
    # Position-level voting: pos 4 disagrees but most readings have '3'.
    readings = [
        ("AB123CD", 0.80),
        ("AB128CD", 0.50),     # 8 in pos 4 instead of 3 — outvoted
        ("AB123CD", 0.85),
        ("AB123CO", 0.70),     # outlier in pos 6
    ]
    plate, _ = tracker.pick_best_plate(readings)
    assert plate == "AB123CD"


def test_confidence_breaks_position_ties():
    # 2-vs-2 ties at pos 4 — '8' vs '3'. '3' has higher summed confidence so it wins.
    readings = [
        ("AB123CD", 0.95),   # vote: 3 → 0.95
        ("AB183CD", 0.50),   # vote: 8 → 0.50
        ("AB123CD", 0.70),   # vote: 3 → 0.70  → 3 total = 1.65
        ("AB183CD", 0.80),   # vote: 8 → 0.80  → 8 total = 1.30
    ]
    plate, _ = tracker.pick_best_plate(readings)
    assert plate == "AB123CD"


def test_position_correction_letter_position():
    # OCR consistently misread pos 0 as '8' (should be a letter). All readings agree.
    # _apply_position_kinds should map '8' → 'B' because pos 0 is a letter slot.
    readings = [("8B123CD", 0.85), ("8B123CD", 0.80), ("8B123CD", 0.88)]
    plate, _ = tracker.pick_best_plate(readings)
    assert plate == "BB123CD"
    assert tracker.looks_like_country_plate(plate)


def test_position_correction_digit_position():
    # OCR misread pos 3 as 'O' (should be digit). Map 'O' → '0'.
    readings = [("AB1O3CD", 0.85), ("AB1O3CD", 0.85), ("AB1O3CD", 0.85)]
    plate, _ = tracker.pick_best_plate(readings)
    assert plate == "AB103CD"
    assert tracker.looks_like_country_plate(plate)


def test_format_match_check_independent():
    assert tracker.looks_like_country_plate("AB123CD") is True
    assert tracker.looks_like_country_plate("123ABCD") is False
    assert tracker.looks_like_country_plate("AB12CD")  is False     # too short
    assert tracker.looks_like_country_plate("")        is False


def test_length_voting_picks_majority_length():
    # 3 readings of length 7, 1 reading of length 6. Length-7 wins.
    readings = [
        ("AB123CD", 0.80),
        ("AB123CD", 0.82),
        ("AB123CD", 0.85),
        ("AB12CD",  0.95),    # length-6 outlier, higher conf but only 1 vote
    ]
    plate, _ = tracker.pick_best_plate(readings)
    assert plate == "AB123CD"


# ─── VisitTracker ──────────────────────────────────────────────────────────
def test_tracker_creates_one_event_per_track():
    vt = tracker.VisitTracker(grace_s=0.05, min_event_s=0)
    vt.update(1, "car", "AB123CD", 0.9)
    vt.update(1, "car", "AB123CD", 0.85)
    vt.update(1, "car", "AB123CD", 0.91)
    time.sleep(0.1)
    events = vt.collect_completed()
    assert len(events) == 1
    assert events[0]["plate"] == "AB123CD"
    assert events[0]["track_id"] == 1
    assert events[0]["reading_count"] == 3


def test_tracker_separate_tracks_produce_separate_events():
    vt = tracker.VisitTracker(grace_s=0.05, min_event_s=0)
    vt.update(1, "car", "AB123CD", 0.9)
    vt.update(2, "car", "ZZ999AA", 0.9)
    time.sleep(0.1)
    events = vt.collect_completed()
    assert len(events) == 2
    plates = {e["plate"] for e in events}
    assert plates == {"AB123CD", "ZZ999AA"}


def test_tracker_skips_visit_with_no_plate_readings():
    vt = tracker.VisitTracker(grace_s=0.05, min_event_s=0)
    vt.update(1, "car", None, 0.0)   # vehicle seen but plate never read
    vt.update(1, "car", None, 0.0)
    time.sleep(0.1)
    events = vt.collect_completed()
    assert events == []


def test_tracker_flush_finalises_all():
    vt = tracker.VisitTracker(grace_s=999, min_event_s=0)
    vt.update(1, "car", "AB123CD", 0.9)
    vt.update(2, "car", "XX111YY", 0.85)
    assert vt.active_count == 2
    events = vt.flush()
    assert len(events) == 2
    assert vt.active_count == 0


def test_tracker_extends_short_visits_to_min_duration():
    vt = tracker.VisitTracker(grace_s=0.05, min_event_s=2)
    vt.update(1, "car", "AB123CD", 0.9)
    time.sleep(0.1)
    events = vt.collect_completed()
    # ended_at must be at least 2s after started_at
    assert len(events) == 1
    from datetime import datetime
    started = datetime.fromisoformat(events[0]["started_at"].replace("Z", "+00:00"))
    ended = datetime.fromisoformat(events[0]["ended_at"].replace("Z", "+00:00"))
    assert (ended - started).total_seconds() >= 1.9
