"""Strict single-VRAM-resident policy (§2.5.5).

At most ONE local VRAM-consuming model is resident at any moment — the reasoning
LLM, the image generator, the 3D backend, or UniRig. This is enforced here, in
one place, rather than left to per-step discretion. Every consumer calls
ensure_loaded() and gets the slot; whatever held it before is ejected first.

Cloud LLM providers occupy no slot (nothing local is loaded), which is why
switching LM Studio -> Gemini frees VRAM and needs no load on the other side.
"""
from __future__ import annotations

import gc
import threading
from typing import Any, Callable

from . import events

# Kinds that consume the single local VRAM slot
LOCAL_KINDS = ("llm", "image_gen", "model_gen", "rigger")


class ResourceManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resident_kind: str | None = None
        self._resident_key: str | None = None
        self._resident_obj: Any = None
        self._unload_fn: Callable[[Any], None] | None = None

    # ------------------------------------------------------------------
    @property
    def resident(self) -> dict:
        with self._lock:
            return {"kind": self._resident_kind, "key": self._resident_key}

    def ensure_loaded(
        self,
        kind: str,
        key: str,
        loader: Callable[[], Any],
        unloader: Callable[[Any], None] | None = None,
        occupies_vram: bool = True,
        session_id: str = "",
    ) -> Any:
        """Return a ready-to-use backend for `kind`/`key`, ejecting whatever else
        currently holds the VRAM slot.

        occupies_vram=False is the cloud-provider case: nothing is loaded locally,
        so it does NOT touch the slot at all. Evicting on a cloud call would be
        actively harmful — classify_edit() runs on every single edit, and evicting
        a resident 3D backend each time would mean paying a full cold reload per
        edit, which is exactly the cost using a cloud LLM is supposed to avoid.
        """
        with self._lock:
            if not occupies_vram:
                events.emit("backend_ready", kind=kind, key=key, local=False, session_id=session_id)
                return None

            if self._resident_kind == kind and self._resident_key == key:
                return self._resident_obj  # already warm, nothing to do

            if self._resident_obj is not None or self._resident_kind is not None:
                self._eject_locked(session_id=session_id)

            events.emit("backend_loading", kind=kind, key=key, session_id=session_id)
            obj = loader()
            self._resident_kind = kind
            self._resident_key = key
            self._resident_obj = obj
            self._unload_fn = unloader
            events.emit("backend_ready", kind=kind, key=key, local=True, session_id=session_id)
            return obj

    def eject(self, session_id: str = "") -> None:
        with self._lock:
            self._eject_locked(session_id=session_id)

    def _eject_locked(self, session_id: str = "") -> None:
        prev_kind, prev_key = self._resident_kind, self._resident_key
        if self._resident_obj is not None and self._unload_fn is not None:
            try:
                self._unload_fn(self._resident_obj)
            except Exception as exc:  # ejection must never abort the caller
                events.log(session_id, f"Backend unload raised (continuing): {exc}", "warn")
        self._resident_kind = None
        self._resident_key = None
        self._resident_obj = None
        self._unload_fn = None
        if prev_kind:
            events.emit("backend_ejected", kind=prev_kind, key=prev_key, session_id=session_id)
        _free_vram()


def _free_vram() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


RESOURCE_MANAGER = ResourceManager()
