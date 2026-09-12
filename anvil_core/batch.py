"""Batch / multi-generation queue (§2.12).

A queue *wrapper* around run_anvil_session, not a second state machine. Each
queued item becomes an ordinary session with its own session.json and goes
through all the same steps a single generation would.

Strictly sequential by design: the single-VRAM-resident policy (§2.5.5) means
there's no budget to run two generations concurrently, so a parallel runner would
just serialize inside resource_manager anyway — badly, with two sessions racing
for the same load/eject cycle.
"""
from __future__ import annotations

import json
import threading
import time
import uuid

from . import events, sessions, state_machine
from .config import SESSIONS_DIR
from .schemas import Session

BATCH_STATE_FILE = SESSIONS_DIR / "batch_state.json"

# status values a queue item moves through
PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


class BatchRunner:
    """Owns the queue and the worker thread. One batch at a time."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queue: list[dict] = []
        self._batch_id: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._running = False
        self._gate = None          # set by the server via attach_gate()
        self._gate_holder: str | None = None
        self._load_state()

    def attach_gate(self, gate) -> None:
        """Wire in the process-wide pipeline gate.

        The batch holds it for the whole run rather than per-item: releasing
        between items would let a single generation slip into the gap and start
        competing for the VRAM slot mid-batch.
        """
        self._gate = gate

    # -- persistence (§2.12: killed batch can offer "resume batch") ----------

    def _save_state(self) -> None:
        BATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BATCH_STATE_FILE.write_text(json.dumps({
            "batch_id": self._batch_id,
            "queue": self._queue,
            "saved_at": time.time(),
        }, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        if not BATCH_STATE_FILE.exists():
            return
        try:
            data = json.loads(BATCH_STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self._batch_id = data.get("batch_id")
        self._queue = data.get("queue", [])
        # Anything caught mid-flight by a crash goes back to pending — it never
        # finished, so it's owed a rerun, not a false 'done'.
        for item in self._queue:
            if item.get("status") == RUNNING:
                item["status"] = PENDING

    # -- queue management ---------------------------------------------------

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def snapshot(self) -> dict:
        with self._lock:
            counts = {s: sum(1 for i in self._queue if i["status"] == s)
                      for s in (PENDING, RUNNING, DONE, FAILED)}
            return {
                "batch_id": self._batch_id,
                "running": self._running,
                "stop_requested": self._stop_requested,
                "items": list(self._queue),
                "counts": counts,
                "resumable": (not self._running) and counts[PENDING] > 0 and (counts[DONE] + counts[FAILED]) > 0,
            }

    def add(self, *, mode: str, style_id: str, checklist: dict, flags: dict,
            procedural: bool = False, image_path: str | None = None, label: str = "") -> dict:
        """Snapshot the current form state as one queue item."""
        with self._lock:
            if self._running:
                raise RuntimeError("The queue is locked while a batch is running.")
            item = {
                "queue_id": uuid.uuid4().hex[:8],
                "mode": mode,
                "style_id": style_id,
                "checklist": dict(checklist),
                "flags": dict(flags),
                "procedural": procedural,
                "image_path": image_path,
                "label": label or checklist.get("object_type") or "(untitled)",
                "status": PENDING,
                "error": None,
                "session_id": None,
            }
            self._queue.append(item)
            self._save_state()
            events.emit("queue_changed", **self.snapshot())
            return item

    def remove(self, queue_id: str) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("The queue is locked while a batch is running.")
            self._queue = [i for i in self._queue if i["queue_id"] != queue_id]
            self._save_state()
            events.emit("queue_changed", **self.snapshot())

    def retry(self, queue_id: str) -> None:
        """Re-queue exactly one failed item, not the whole batch."""
        with self._lock:
            for item in self._queue:
                if item["queue_id"] == queue_id:
                    item["status"] = PENDING
                    item["error"] = None
            self._save_state()
            events.emit("queue_changed", **self.snapshot())

    def clear(self) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("The queue is locked while a batch is running.")
            self._queue = []
            self._batch_id = None
            self._save_state()
            events.emit("queue_changed", **self.snapshot())

    def request_stop(self) -> None:
        """Stop after the current item — not an immediate hard kill (§2.12).
        The in-flight item finishes; the rest stay pending."""
        with self._lock:
            self._stop_requested = True
        events.emit("batch_stopping")

    # -- execution ----------------------------------------------------------

    def start(self) -> str:
        with self._lock:
            if self._running:
                raise RuntimeError("A batch is already running.")
            pending = [i for i in self._queue if i["status"] == PENDING]
            if not pending:
                raise RuntimeError("Nothing pending in the queue.")
            # Settle the batch id BEFORE acquiring, so acquire and release use
            # the identical holder string. A mismatch here silently wedges the
            # gate — release() ignores a caller that isn't the current holder.
            self._batch_id = self._batch_id or uuid.uuid4().hex[:8]
            gate_holder = f"batch:{self._batch_id}"

        if self._gate is not None:
            self._gate.acquire(gate_holder, "batch")

        try:
            with self._lock:
                self._gate_holder = gate_holder
                self._running = True
                self._stop_requested = False
                self._save_state()

            events.emit("batch_started", batch_id=self._batch_id, total=len(pending))
            self._thread = threading.Thread(target=self._run, name="anvil-batch", daemon=True)
            self._thread.start()
            return self._batch_id
        except Exception:
            if self._gate is not None:
                self._gate.release(gate_holder)
            with self._lock:
                self._running = False
                self._gate_holder = None
            raise

    def _next_pending(self) -> dict | None:
        with self._lock:
            for item in self._queue:
                if item["status"] == PENDING:
                    return item
            return None

    def _run(self) -> None:
        try:
            self._run_loop()
        finally:
            # The gate must be released even if the loop itself blew up, or the
            # whole app is wedged until restart.
            with self._lock:
                self._running = False
                self._stop_requested = False
                snap = self.snapshot()
                self._save_state()
                holder = self._gate_holder
                self._gate_holder = None
            if self._gate is not None and holder:
                self._gate.release(holder)
            events.emit("batch_finished", **snap)

    def _run_loop(self) -> None:
        total = sum(1 for i in self._queue if i["status"] == PENDING)
        completed = 0

        while True:
            with self._lock:
                if self._stop_requested:
                    break
            item = self._next_pending()
            if item is None:
                break

            with self._lock:
                item["status"] = RUNNING
                self._save_state()
            events.emit("queue_changed", **self.snapshot())
            events.emit("batch_item_start", queue_id=item["queue_id"], label=item["label"],
                        index=completed + 1, total=total)

            try:
                session = Session(
                    mode=item["mode"],
                    style_id=item["style_id"],
                    procedural=item["procedural"],
                    checklist=dict(item["checklist"]),
                    flags=dict(item["flags"]),
                    uploaded_image_path=item.get("image_path"),
                    batch_id=self._batch_id,
                    label=item["label"],
                )
                sessions.save_session(session)
                item["session_id"] = session.session_id

                # Steps 1-9 only — no edit loop for queued items (§2.12).
                state_machine.run_generation_pipeline(session)

                with self._lock:
                    item["status"] = DONE
                    item["error"] = None
            except Exception as exc:
                # A failed item does NOT stop the batch (§2.12, §9).
                with self._lock:
                    item["status"] = FAILED
                    item["error"] = f"{type(exc).__name__}: {exc}"
                events.log("", f"Queue item '{item['label']}' failed: {exc}", "error")

            completed += 1
            with self._lock:
                self._save_state()
            events.emit("queue_changed", **self.snapshot())
            events.emit("batch_item_done", queue_id=item["queue_id"], status=item["status"],
                        error=item["error"], index=completed, total=total)


BATCH = BatchRunner()
