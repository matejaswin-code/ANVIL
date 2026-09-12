"""Queryable pipeline status.

The event bus in `events` is a fan-out: subscribers hear what happens *while*
they are listening. That is the right shape for a live log, and the wrong shape
for the question "what is happening right now?" — a tab opened mid-run, or a
poll after a dropped connection, has missed every event that mattered.

This module keeps the answer to that question: one snapshot, updated in place by
the same emitters that publish to the bus, readable at any moment over HTTP.

Deliberately dependency-free. `events` imports this, and `resource_manager`
imports `events`, so importing either here would close a cycle. Backend
residency and VRAM figures are joined in at the HTTP layer instead, where both
are already reachable.
"""
from __future__ import annotations

import threading
import time
from typing import Any

_NOTE_CAP = 12


class PipelineStatus:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reset()

    def _reset(self) -> None:
        self._phase = "idle"
        self._session_id: str | None = None
        self._label: str = ""
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._steps: list[dict] = []
        self._error: str | None = None
        self._notes: list[dict] = []

    # -- writers, called from events.* -------------------------------------

    def begin(self, session_id: str, label: str = "") -> None:
        with self._lock:
            self._reset()
            self._phase = "running"
            self._session_id = session_id
            self._label = label
            self._started_at = time.time()

    def step_start(self, session_id: str, step: str, index: int, total: int, label: str) -> None:
        with self._lock:
            # A run can start at any step — resume picks up mid-pipeline — so the
            # first step seen is what opens the run, not a separate begin() call.
            if self._phase != "running" or self._session_id != session_id:
                self.begin(session_id)
            for s in self._steps:
                if s["key"] == step:
                    s.update(status="running", started_at=time.time(), ended_at=None)
                    break
            else:
                self._steps.append({
                    "key": step, "label": label or step, "index": index, "total": total,
                    "status": "running", "started_at": time.time(), "ended_at": None,
                })

    def step_done(self, session_id: str, step: str) -> None:
        with self._lock:
            for s in self._steps:
                if s["key"] == step:
                    s["status"] = "done"
                    s["ended_at"] = time.time()
                    break

    def note(self, message: str, level: str = "info") -> None:
        with self._lock:
            self._notes.append({"message": message, "level": level, "at": time.time()})
            if len(self._notes) > _NOTE_CAP:
                del self._notes[: len(self._notes) - _NOTE_CAP]

    def finish(self, session_id: str, error: str | None = None) -> None:
        with self._lock:
            self._phase = "failed" if error else "done"
            self._error = error
            self._ended_at = time.time()
            for s in self._steps:
                if s["status"] == "running":
                    # Whatever was mid-flight when the run ended is where it stopped.
                    s["status"] = "failed" if error else "done"
                    s["ended_at"] = self._ended_at

    # -- reader ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A complete picture of the run, safe to poll at any interval."""
        now = time.time()
        with self._lock:
            steps = []
            current = None
            for s in self._steps:
                end = s["ended_at"] or (now if s["status"] == "running" else None)
                entry = {
                    "key": s["key"],
                    "label": s["label"],
                    "index": s["index"],
                    "total": s["total"],
                    "status": s["status"],
                    "duration_s": round(end - s["started_at"], 1) if end else None,
                }
                steps.append(entry)
                if s["status"] == "running":
                    current = entry

            done = sum(1 for s in steps if s["status"] == "done")
            total = steps[-1]["total"] if steps else 0
            end = self._ended_at or (now if self._phase == "running" else None)

            return {
                "phase": self._phase,
                "session_id": self._session_id,
                "label": self._label,
                "started_at": self._started_at,
                "elapsed_s": round(end - self._started_at, 1) if (end and self._started_at) else None,
                "current": current,
                "steps": steps,
                "steps_done": done,
                "steps_total": total,
                "error": self._error,
                "notes": list(self._notes),
            }


PIPELINE_STATUS = PipelineStatus()
