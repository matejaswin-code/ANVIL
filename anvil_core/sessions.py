"""Session persistence & the resumable-session browser (§2.10, §4.4).

Every step writes session.json. That single fact is what makes both crash
recovery and the "which sessions can I resume" picker possible without a second
data model — the picker is a query over what checkpointing already produces.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from .config import SESSIONS_DIR
from .schemas import PIPELINE_STEPS, Session

# session_id reaches this module from HTTP path parameters, so it is untrusted.
# IDs we generate are 12 hex chars; anything that isn't plain alphanumeric plus
# dash/underscore is rejected outright rather than sanitised into something
# adjacent, because silently rewriting an ID would mean reads and writes could
# disagree about which directory they're touching.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidSessionId(ValueError):
    pass


def _validate_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SAFE_ID.match(session_id):
        raise InvalidSessionId(f"Invalid session id: {session_id!r}")
    return session_id


def session_dir(session_id: str) -> Path:
    d = SESSIONS_DIR / _validate_id(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_file(session_id: str) -> Path:
    return session_dir(session_id) / "session.json"


def save_session(session: Session) -> None:
    """Atomic-ish write: temp file then replace, so a crash mid-write can't leave
    a half-written session.json that resume would choke on."""
    session.updated_at = time.time()
    path = session_file(session.session_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session.as_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)


def load_session(session_id: str) -> Session | None:
    try:
        path = session_file(session_id)
    except InvalidSessionId:
        return None      # callers treat None as "no such session", which is correct here
    if not path.exists():
        return None
    try:
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def load_or_create_session(session_id: str | None = None, **kwargs) -> Session:
    if session_id:
        existing = load_session(session_id)
        if existing:
            return existing
    session = Session(**kwargs) if kwargs else Session()
    if session_id:
        session.session_id = session_id
    save_session(session)
    return session


def delete_session(session_id: str) -> bool:
    try:
        d = SESSIONS_DIR / _validate_id(session_id)
    except InvalidSessionId:
        return False
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


def list_resumable_sessions(limit: int = 20) -> list[dict]:
    """Scan sessions/ for session.json files where the pipeline hasn't finished
    (current_step hasn't reached step 10 / edit_loop), newest first.

    Deliberately no mesh/image thumbnail — generating a preview would mean loading
    the actual asset just to populate a picker list, which is more work than this
    needs to do (§2.10).
    """
    out: list[dict] = []
    if not SESSIONS_DIR.exists():
        return out

    for d in SESSIONS_DIR.iterdir():
        if not d.is_dir():
            continue
        s = load_session(d.name)
        if s is None:
            continue
        if s.current_step >= len(PIPELINE_STEPS):
            continue  # finished — not resumable
        if s.status == "running":
            # Was mid-flight when the process died; still resumable.
            pass
        out.append({
            "session_id": s.session_id,
            "mode": s.mode,
            "style_id": s.style_id,
            "label": s.display_label(),
            "current_step": s.current_step,
            "total_steps": len(PIPELINE_STEPS),
            "status": s.status,
            "procedural": s.procedural,
            "batch_id": s.batch_id,
            "last_modified": s.updated_at,
        })

    out.sort(key=lambda r: r["last_modified"], reverse=True)
    return out[:limit]


def list_all_sessions(limit: int = 50) -> list[dict]:
    out: list[dict] = []
    if not SESSIONS_DIR.exists():
        return out
    for d in SESSIONS_DIR.iterdir():
        if not d.is_dir():
            continue
        s = load_session(d.name)
        if s is None:
            continue
        out.append({
            "session_id": s.session_id,
            "mode": s.mode,
            "style_id": s.style_id,
            "label": s.display_label(),
            "current_step": s.current_step,
            "status": s.status,
            "final_mesh_path": s.final_mesh_path,
            "last_modified": s.updated_at,
        })
    out.sort(key=lambda r: r["last_modified"], reverse=True)
    return out[:limit]
