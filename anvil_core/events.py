"""In-process event bus.

The pipeline runs on a worker thread; the HTTP layer streams these events to the
browser over SSE. Deliberately a plain queue fan-out rather than a message broker
— §1's "single linear script, not a service mesh" applies to progress reporting
just as much as to the pipeline itself.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

from .status import PIPELINE_STATUS

_subscribers: list[queue.Queue] = []
_lock = threading.Lock()
_history: list[dict] = []
_HISTORY_CAP = 200


def subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=512)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _lock:
        if q in _subscribers:
            _subscribers.remove(q)


def emit(event_type: str, **payload: Any) -> None:
    """Publish an event to every live subscriber. Never blocks the pipeline."""
    evt = {"type": event_type, "at": time.time(), **payload}
    with _lock:
        _history.append(evt)
        if len(_history) > _HISTORY_CAP:
            del _history[: len(_history) - _HISTORY_CAP]
        targets = list(_subscribers)
    for q in targets:
        try:
            q.put_nowait(evt)
        except queue.Full:
            # A stalled browser tab must not stall generation.
            pass


def recent(limit: int = 50) -> list[dict]:
    with _lock:
        return _history[-limit:]


def sse_format(evt: dict) -> str:
    return f"data: {json.dumps(evt)}\n\n"


# ---- convenience emitters used across the pipeline -----------------------
#
# Each one both publishes to the bus (for live subscribers) and updates the
# pollable snapshot in `status` (for anyone who wasn't listening at the time).
# Keeping both writes here means there is one place to change, not two.

def step_start(session_id: str, step: str, index: int, total: int, label: str = "") -> None:
    PIPELINE_STATUS.step_start(session_id, step, index, total, label or step)
    emit("step_start", session_id=session_id, step=step, index=index, total=total, label=label or step)


def step_done(session_id: str, step: str, index: int, total: int) -> None:
    PIPELINE_STATUS.step_done(session_id, step)
    emit("step_done", session_id=session_id, step=step, index=index, total=total)


def log(session_id: str, message: str, level: str = "info") -> None:
    PIPELINE_STATUS.note(message, level)
    emit("log", session_id=session_id, message=message, level=level)


def session_done(session_id: str, **payload: Any) -> None:
    PIPELINE_STATUS.finish(session_id)
    emit("session_done", session_id=session_id, **payload)


def session_failed(session_id: str, error: str) -> None:
    PIPELINE_STATUS.finish(session_id, error=error)
    emit("session_failed", session_id=session_id, error=error)
