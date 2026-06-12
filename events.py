"""
Event POST worker — runs in its own thread so the detector main loop
never blocks on the network (audit finding C3).

Flow per event
──────────────
  main loop  →  EventDispatcher.submit(payload)
                       ↓
                bounded in-process queue
                       ↓
                worker thread → api.post_event
                       ├─ success           → stats.sent++
                       ├─ PermanentFailure  → db.dead_letter_add
                       └─ TransientFailure  → db.enqueue (for QueueFlusher)
"""
from __future__ import annotations

import logging
import queue as _queue
import threading

import api
import db

log = logging.getLogger("laventra")


class EventDispatcher:
    def __init__(self, *, queue_size: int = 100):
        self._q: _queue.Queue = _queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = None
        self.stats = {"sent": 0, "queued": 0, "dead_letter": 0, "dropped_overflow": 0}

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="event-worker"
        )
        self._thread.start()
        log.info("📤 Event dispatcher started")

    def stop(self, *, drain_timeout: float = 10.0) -> None:
        """
        Stop accepting new events and try to drain pending ones up to
        `drain_timeout` seconds. Unsent events go to the sqlite queue.
        """
        # Mark stop, then wake the worker and let it drain.
        self._stop.set()
        # Put a sentinel so the worker doesn't sit on a queue.get
        try:
            self._q.put_nowait(_SENTINEL)
        except _queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=drain_timeout)
        # Anything left, write straight to the offline queue
        drained = 0
        while True:
            try:
                payload = self._q.get_nowait()
            except _queue.Empty:
                break
            if payload is _SENTINEL:
                continue
            db.enqueue(**_to_queue_kwargs(payload))
            drained += 1
        if drained:
            log.info(f"📥 {drained} unsent event(s) written to offline queue")

    def submit(self, payload: dict) -> None:
        """
        Hand a completed visit to the worker. Non-blocking: if the worker is
        backed up beyond queue_size we write straight to the sqlite queue
        rather than dropping the event.
        """
        try:
            self._q.put_nowait(payload)
        except _queue.Full:
            self.stats["dropped_overflow"] += 1
            log.warning("Event worker overloaded — bypassing to offline queue")
            db.enqueue(**_to_queue_kwargs(payload))

    # ── worker ───────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while True:
            try:
                payload = self._q.get(timeout=0.5)
            except _queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if payload is _SENTINEL:
                if self._stop.is_set():
                    return
                continue
            self._handle(payload)

    def _handle(self, payload: dict) -> None:
        plate = payload.get("plate", "?")
        try:
            api.post_event(payload)
            self.stats["sent"] += 1
            log.info(f"✅ Event posted → {plate}")
        except api.PermanentFailure as pf:
            self.stats["dead_letter"] += 1
            db.dead_letter_add(payload=payload, error_code=pf.status, error_body=pf.body)
            log.error(
                f"⚠️  Event permanently rejected → {plate} "
                f"(HTTP {pf.status}) → dead_letter"
            )
        except api.TransientFailure as tf:
            self.stats["queued"] += 1
            db.enqueue(**_to_queue_kwargs(payload))
            log.warning(f"📥 Event queued offline → {plate} ({tf})")


_SENTINEL = object()


def _to_queue_kwargs(payload: dict) -> dict:
    return {
        "plate":           payload["plate"],
        "vehicle_type":    payload.get("vehicle_type", "unknown"),
        "started_at":      payload["started_at"],
        "ended_at":        payload["ended_at"],
        "confidence":      payload.get("confidence"),
        "camera_id":       payload.get("camera_id"),
        "client_event_id": payload.get("client_event_id"),
    }
