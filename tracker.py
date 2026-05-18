"""
Visit aggregator + best-plate picker.

A car-wash visit is one vehicle from arrival to departure. We accumulate every
OCR reading attached to its YOLO track_id and, at finalisation, choose the
single best plate string via per-character weighted voting with country-aware
position correction. The output is one event per visit (audit finding H6).

Picker logic
────────────
1. Bin readings by string length.
2. Take the length closest to PLATE_TARGET_LEN that has the most readings.
3. For each character position, pick the character whose summed `ocr_conf`
   across all readings of that length is highest (weighted vote).
4. If PLATE_POSITION_KINDS says a position must be a letter but voting picked
   a digit (or vice versa), substitute via OCR_CONFUSABLE_AS_LETTER /
   OCR_CONFUSABLE_AS_DIGIT.
5. If the corrected string matches PLATE_FORMAT_REGEX, return it. Otherwise
   fall back to the highest-confidence raw reading.
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import config

log = logging.getLogger("laventra")
_PLATE_RE = re.compile(config.PLATE_FORMAT_REGEX)


# ─── timestamp helpers ─────────────────────────────────────────────────────
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    ) + "Z"


# ─── public API ────────────────────────────────────────────────────────────
def looks_like_country_plate(plate: str) -> bool:
    """True if `plate` matches the configured country format regex."""
    return bool(_PLATE_RE.match(plate))


def pick_best_plate(readings: list[tuple[str, float]]) -> tuple[str | None, float]:
    """
    Choose the strongest plate string from a list of [(plate, conf), …].

    Returns (plate or None, conf in 0..1). See module docstring for the rules.
    """
    if not readings:
        return None, 0.0

    # 1) Bin by length, prefer the length nearest the target with most votes.
    by_len: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for plate, conf in readings:
        by_len[len(plate)].append((plate, conf))

    chosen_len = _choose_length(by_len)
    same_length = by_len[chosen_len]

    # 2) Per-character weighted vote.
    voted = _weighted_vote(same_length, chosen_len)
    avg_conf = sum(c for _, c in same_length) / len(same_length)

    # 3) Position-aware correction for the configured country format.
    corrected = _apply_position_kinds(voted)

    # 4) If the corrected string matches the format, that's our pick.
    if looks_like_country_plate(corrected):
        return corrected, avg_conf

    # 5) Otherwise prefer the raw voted string if non-empty, else the
    #    highest-confidence original reading.
    if voted:
        return voted, avg_conf
    best_plate, best_conf = max(readings, key=lambda x: x[1])
    return best_plate, best_conf


# ─── internals ─────────────────────────────────────────────────────────────
def _choose_length(by_len: dict[int, list]) -> int:
    """Length with the most readings; ties broken by proximity to target."""
    target = config.PLATE_TARGET_LEN
    return max(
        by_len.keys(),
        key=lambda L: (len(by_len[L]), -abs(L - target)),
    )


def _weighted_vote(readings: list[tuple[str, float]], length: int) -> str:
    """Per-position weighted vote across all readings of `length`."""
    if not readings:
        return ""
    out = []
    for pos in range(length):
        scores: dict[str, float] = defaultdict(float)
        for plate, conf in readings:
            scores[plate[pos]] += conf
        out.append(max(scores.items(), key=lambda kv: kv[1])[0])
    return "".join(out)


def _apply_position_kinds(plate: str) -> str:
    """
    Force each position to match its required kind (L=letter, D=digit) using
    the confusable maps. If the length doesn't match PLATE_POSITION_KINDS the
    string is returned unchanged.
    """
    kinds = config.PLATE_POSITION_KINDS
    if len(plate) != len(kinds):
        return plate

    corrected = []
    for ch, kind in zip(plate, kinds):
        if kind == "L":
            if ch.isalpha():
                corrected.append(ch)
            elif ch in config.OCR_CONFUSABLE_AS_LETTER:
                corrected.append(config.OCR_CONFUSABLE_AS_LETTER[ch])
            else:
                corrected.append(ch)
        elif kind == "D":
            if ch.isdigit():
                corrected.append(ch)
            elif ch in config.OCR_CONFUSABLE_AS_DIGIT:
                corrected.append(config.OCR_CONFUSABLE_AS_DIGIT[ch])
            else:
                corrected.append(ch)
        else:
            corrected.append(ch)
    return "".join(corrected)


# ─── VisitTracker ──────────────────────────────────────────────────────────
class VisitTracker:
    """
    One track_id = one visit. Finalise after `grace_s` seconds of absence.
    """

    def __init__(self, *, grace_s: float = None, min_event_s: float = None):
        self._active: dict[int, dict] = {}
        self._grace_s = grace_s if grace_s is not None else config.PRESENCE_GRACE_S
        self._min_event_s = min_event_s if min_event_s is not None else config.MIN_EVENT_DURATION_S

    def update(self, track_id: int, vehicle_type: str,
               plate: str | None = None, ocr_conf: float = 0.0) -> bool:
        now_ts = time.time()
        if track_id not in self._active:
            self._active[track_id] = {
                "started_ts":   now_ts,
                "started_iso":  _utc_now_iso(),
                "last_seen_ts": now_ts,
                "vehicle_type": vehicle_type,
                "readings":     [],
            }
            is_new = True
        else:
            v = self._active[track_id]
            v["last_seen_ts"] = now_ts
            is_new = False
        if plate:
            self._active[track_id]["readings"].append((plate, ocr_conf))
        return is_new

    def collect_completed(self, grace_s: float = None) -> list[dict]:
        grace = grace_s if grace_s is not None else self._grace_s
        now = time.time()
        out = []
        for tid in list(self._active.keys()):
            v = self._active[tid]
            if now - v["last_seen_ts"] >= grace:
                evt = self._make_event(tid, v)
                if evt:
                    out.append(evt)
                else:
                    log.debug(f"Track #{tid}: no valid plate readings — skipping event")
                del self._active[tid]
        return out

    def flush(self) -> list[dict]:
        out = []
        for tid, v in list(self._active.items()):
            evt = self._make_event(tid, v)
            if evt:
                out.append(evt)
        self._active.clear()
        return out

    @property
    def active_count(self) -> int:
        return len(self._active)

    def _make_event(self, track_id: int, v: dict) -> dict | None:
        plate, ocr_conf = pick_best_plate(v["readings"])
        if not plate:
            return None
        started_ts = v["started_ts"]
        last_ts = v["last_seen_ts"]
        if last_ts < started_ts + self._min_event_s:
            ended_iso = _ts_to_iso(started_ts + self._min_event_s)
        else:
            ended_iso = _ts_to_iso(last_ts)
        return {
            "plate":         plate,
            "vehicle_type":  v["vehicle_type"],
            "started_at":    v["started_iso"],
            "ended_at":      ended_iso,
            "confidence":    round(ocr_conf * 100, 2),
            "track_id":      track_id,
            "reading_count": len(v["readings"]),
        }
