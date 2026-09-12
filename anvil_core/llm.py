"""Reasoning layer (§2.5.3, §6).

One underlying call — same endpoint, same error handling, same retry logic —
reused with four different prompts/schemas rather than four separate "AI
reasoning subsystems":

    fill_checklist()            — interview the user for mandatory fields
    caption_style_reference()   — turn a reference image/text into a style modifier
    classify_edit()             — property | shape | new_model | rig
    extract_property_params()   — turn a property instruction into structured params

Error handling is deliberately provider-aware (§9): a 401 is a different message
from a timeout, because they need different fixes from the user.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from . import events
from .config import SETTINGS
from .resource_manager import RESOURCE_MANAGER


class LLMError(RuntimeError):
    """Raised with a message that is safe and useful to show the user directly."""


class VisionUnsupportedError(LLMError):
    pass


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

def _endpoint_and_headers() -> tuple[str, dict, str]:
    provider = SETTINGS.llm_provider
    if provider == "lm_studio":
        base = SETTINGS.llm_endpoint.rstrip("/")
        return f"{base}/chat/completions", {"Content-Type": "application/json"}, SETTINGS.llm_model
    if provider == "openrouter":
        key = SETTINGS.openrouter_api_key
        if not key:
            raise LLMError("OpenRouter selected but OPENROUTER_API_KEY is empty. Add it in .env or the Settings panel.")
        return (
            "https://openrouter.ai/api/v1/chat/completions",
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            SETTINGS.llm_model or "anthropic/claude-3.5-sonnet",
        )
    if provider == "gemini":
        key = SETTINGS.gemini_api_key
        if not key:
            raise LLMError("Gemini selected but GEMINI_API_KEY is empty. Add it in .env or the Settings panel.")
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            SETTINGS.llm_model or "gemini-2.0-flash",
        )
    raise LLMError(f"Unknown LLM provider: {SETTINGS.llm_provider!r}")


def _lms_cli() -> str | None:
    """Path to LM Studio's CLI, or None if it isn't installed."""
    found = shutil.which("lms")
    if found:
        return found
    candidate = Path.home() / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms")
    return str(candidate) if candidate.exists() else None


def _unload_lm_studio(session_id: str = "") -> None:
    """Actually free the VRAM LM Studio is holding.

    LM Studio runs out-of-process, so the resource manager can mark the slot free
    without a single byte being released — which on a card that can't fit both
    the LLM and the 3D backend means the next step silently spills to system RAM
    and crawls, rather than failing outright.

    This is safe to do unconditionally only because LM Studio's just-in-time
    loading reloads the model on the next request (a few seconds). If JIT is
    turned off, the reload doesn't happen and the following LLM call fails with
    "No models loaded" — so the failure to unload here is logged, never raised:
    a slow pipeline beats a broken one.
    """
    cli = _lms_cli()
    if not cli:
        return
    try:
        subprocess.run(
            [cli, "unload", "--all"],
            capture_output=True, timeout=30, check=False,
        )
        events.log(session_id, "Released LM Studio's VRAM for the next backend", "info")
    except (OSError, subprocess.SubprocessError) as exc:
        events.log(session_id, f"Couldn't unload LM Studio ({exc}); it keeps its VRAM", "warn")


def _ensure_llm_slot(session_id: str = "") -> None:
    """Claim the VRAM slot for the LLM. Cloud providers occupy no slot but still
    eject whatever local model was resident (§2.5.5)."""
    is_local = SETTINGS.llm_provider == "lm_studio"
    RESOURCE_MANAGER.ensure_loaded(
        kind="llm",
        key=f"{SETTINGS.llm_provider}:{SETTINGS.llm_model}",
        loader=lambda: {"provider": SETTINGS.llm_provider},
        unloader=lambda _obj: _unload_lm_studio(session_id),
        occupies_vram=is_local,
        session_id=session_id,
    )


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    session_id: str = "",
    timeout: float = 120.0,
) -> str:
    """Single chat call with provider-aware error messages (§9)."""
    _ensure_llm_slot(session_id)
    url, headers, model = _endpoint_and_headers()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.post(url, headers=headers, json=payload)
    except httpx.ConnectError as exc:
        if SETTINGS.llm_provider == "lm_studio":
            raise LLMError(
                f"Can't reach LM Studio at {SETTINGS.llm_endpoint}. "
                "Is the local server running? Check the endpoint field in Settings."
            ) from exc
        raise LLMError(f"Network error reaching {SETTINGS.llm_provider}: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise LLMError(f"{SETTINGS.llm_provider} timed out after {timeout:.0f}s. Try again shortly.") from exc

    if res.status_code in (401, 403):
        raise LLMError(f"{SETTINGS.llm_provider} rejected the credentials ({res.status_code}). Check your API key in Settings.")
    if res.status_code == 429:
        raise LLMError(f"{SETTINGS.llm_provider} rate limit hit. Try again shortly.")
    if res.status_code >= 400:
        raise LLMError(f"{SETTINGS.llm_provider} returned HTTP {res.status_code}: {res.text[:200]}")

    try:
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Unexpected response shape from {SETTINGS.llm_provider}: {res.text[:200]}") from exc


def _parse_json(raw: str) -> Any:
    """Strip markdown fences and pull the first JSON object/array out."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[\{\[].*[\}\]]", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def chat_json(messages: list[dict], *, session_id: str = "", **kwargs) -> Any:
    """chat() + JSON parse, with the one retry from §9: on malformed JSON, retry
    once with the parse error appended. Twice-failed is the caller's problem —
    every caller here has a plain-UI fallback rather than looping forever."""
    raw = chat(messages, session_id=session_id, **kwargs)
    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        events.log(session_id, "LLM returned malformed JSON — retrying once", "warn")
        retry = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "That was not valid JSON. Reply with ONLY the raw JSON object, no prose, no markdown fences."},
        ]
        raw2 = chat(retry, session_id=session_id, **kwargs)
        return _parse_json(raw2)


# ---------------------------------------------------------------------------
# §6 — the four reasoning functions
# ---------------------------------------------------------------------------

def fill_checklist(
    checklist: dict,
    user_message: str,
    schema_fields: list[dict],
    *,
    history: list[dict] | None = None,
    mandatory_keys: list[str] | None = None,
    session_id: str = "",
) -> dict:
    """Extract whatever the user's message fills in, and ask about what's missing.

    Returns {"fields", "next_field", "next_question", "reply"}.

    `history` is the prior turns of the conversation as [{"role", "content"}].
    Without it every message is read in isolation, so "make it bigger" or "yes,
    that one" — the ordinary way people answer a follow-up — has nothing to refer
    back to. The values the model has already committed to are passed separately
    as `checklist`, so a long conversation cannot drift away from them.

    `reply` is the conversational sentence to show the user. It differs from
    `next_question`, which names the field the form should highlight: one run of
    the model produces both, so the chat and the form can never disagree about
    what is still outstanding.
    """
    field_list = "\n".join(
        f"- {f['key']}: {f['label']}"
        + (f" (e.g. {f['placeholder']})" if f.get("placeholder") else "")
        + (" [required]" if mandatory_keys and f["key"] in mandatory_keys else "")
        for f in schema_fields
    )
    system = (
        "You are helping someone specify a 3D asset to generate, by conversation. "
        "From their message, extract values for any of the listed fields they described. "
        "Never invent values they did not imply — an empty field is better than a guess. "
        "Ask about ONE missing required field at a time, in a natural, friendly sentence; "
        "if every required field is filled, say so and invite them to generate or add detail. "
        "Keep replies to one or two short sentences. "
        'Reply with ONLY a JSON object: {"fields": {"key": "value"}, "next_field": "key_or_null", '
        '"next_question": "short question naming the next empty required field, or null", '
        '"reply": "your conversational reply to the user"}'
    )
    convo = [{"role": "system", "content": system}]
    for turn in (history or [])[-10:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            convo.append({"role": role, "content": content})
    convo.append({
        "role": "user",
        "content": (
            f"Fields:\n{field_list}\n\n"
            f"Already filled: {json.dumps(checklist)}\n\n"
            f"My message: {user_message}"
        ),
    })

    result = chat_json(convo, session_id=session_id, temperature=0.4)
    if not isinstance(result, dict):
        raise LLMError("fill_checklist expected a JSON object")

    fields = result.get("fields")
    if not isinstance(fields, dict):
        fields = {}
    # Blank/None values would otherwise wipe fields the user already set.
    fields = {k: str(v).strip() for k, v in fields.items() if v not in (None, "") and str(v).strip()}

    return {
        "fields": fields,
        "next_field": result.get("next_field"),
        "next_question": result.get("next_question"),
        "reply": (result.get("reply") or result.get("next_question") or "").strip(),
    }


def caption_style_reference(
    *, image_path: str | None = None, text: str | None = None, session_id: str = ""
) -> str:
    """Turn a reference image or text description into a style-modifier string
    for the Custom preset.

    Fails clearly BEFORE any call when the active model can't do vision (§2.5.6)
    — sending the image anyway and getting a confusing response back is worse
    than a precise refusal.
    """
    if image_path:
        if not SETTINGS.llm_vision_capable:
            raise VisionUnsupportedError(
                "The active model isn't marked vision-capable, so it can't caption a reference image. "
                "Either switch to a vision-capable model in Settings (and tick 'Vision-capable'), "
                "or describe the style in text instead."
            )
        if SETTINGS.llm_provider == "gemini" and "flash" not in SETTINGS.llm_model and "pro" not in SETTINGS.llm_model:
            raise VisionUnsupportedError(
                f"Model {SETTINGS.llm_model!r} isn't a known vision-capable Gemini model. "
                "Pick e.g. gemini-2.0-flash, or describe the style in text."
            )
        path = Path(image_path)
        if not path.exists():
            raise LLMError(f"Reference image not found: {image_path}")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lower().lstrip(".") or "png"
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
        content = [
            {"type": "text", "text": (
                "Describe ONLY the visual style of this reference in one comma-separated line of "
                "generation-prompt modifiers (materials, shading, palette, level of detail, rendering "
                "approach). Do not describe the subject. No preamble."
            )},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
        messages = [{"role": "user", "content": content}]
    elif text:
        messages = [
            {"role": "system", "content": (
                "Convert the user's style description into one comma-separated line of generation-prompt "
                "style modifiers. Describe style only, never subject matter. No preamble."
            )},
            {"role": "user", "content": text},
        ]
    else:
        raise LLMError("caption_style_reference needs either an image path or text")

    return chat(messages, temperature=0.3, max_tokens=200, session_id=session_id).strip()


# Keyword pre-checks: resolve the two unambiguous categories without an LLM call
# at all (§6, §9 — wrong-category classification is more expensive than bad JSON).
_RIG_PATTERNS = (r"\brig\b", r"\brig this\b", r"\barmature\b", r"\bskeleton\b", r"\bbones?\b", r"\bskinning\b")
_NEW_MODEL_PATTERNS = (
    r"different design", r"completely different", r"start over", r"from scratch",
    r"scrap (this|it)", r"something else entirely", r"totally different",
)


def classify_edit(instruction: str, *, session_id: str = "") -> dict:
    """Classify an edit instruction into property | shape | new_model | rig.

    Returns {"category", "confidence", "needs_confirmation", "source"}.
    Ambiguous cases default to the cheaper path (property) so a misread costs a
    bpy.ops call rather than a full regeneration (§9).
    """
    lower = instruction.lower()
    for pat in _RIG_PATTERNS:
        if re.search(pat, lower):
            return {"category": "rig", "confidence": 1.0, "needs_confirmation": False, "source": "keyword"}
    for pat in _NEW_MODEL_PATTERNS:
        if re.search(pat, lower):
            return {"category": "new_model", "confidence": 1.0, "needs_confirmation": True, "source": "keyword"}

    system = (
        "Classify a 3D-asset edit instruction into exactly one category:\n"
        "- property: cheap in-scene change (scale, colour, position, rotation, rename)\n"
        "- shape: the geometry itself must change, requiring regeneration\n"
        "- new_model: the user wants a completely different asset\n"
        "- rig: the user wants a skeleton/armature added\n"
        'Reply with ONLY: {"category": "...", "confidence": 0.0-1.0}'
    )
    try:
        result = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": instruction}],
            temperature=0.0, max_tokens=100, session_id=session_id,
        )
        category = str(result.get("category", "property")).strip().lower()
        confidence = float(result.get("confidence", 0.5))
    except Exception as exc:
        events.log(session_id, f"classify_edit failed ({exc}) — defaulting to the cheaper 'property' path", "warn")
        return {"category": "property", "confidence": 0.0, "needs_confirmation": False, "source": "fallback"}

    if category not in ("property", "shape", "new_model", "rig"):
        category, confidence = "property", 0.0

    # Gate low-confidence expensive categories behind a confirmation rather than
    # silently burning a regeneration on a misread (§9).
    needs_confirmation = category in ("shape", "new_model") and confidence < 0.75
    return {"category": category, "confidence": confidence, "needs_confirmation": needs_confirmation, "source": "llm"}


def extract_property_params(instruction: str, *, session_id: str = "") -> dict:
    """Second, narrower call: turn a property instruction into structured params.

    Returns {"op": scale|recolor|reposition|rename, "params": {...}} with exactly
    one non-null field matching the op.
    """
    system = (
        "Turn a 3D object edit instruction into structured parameters.\n"
        "Ops and their params:\n"
        '  scale      -> {"factor": float}                 (relative, 1.0 = unchanged)\n'
        '  recolor    -> {"rgb": [r,g,b]}                  (0.0-1.0 each)\n'
        '  reposition -> {"delta": [x,y,z]}                (relative, Blender units)\n'
        '  rename     -> {"name": "string"}\n'
        'Reply with ONLY: {"op": "...", "params": {...}}'
    )
    result = chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": instruction}],
        temperature=0.0, max_tokens=200, session_id=session_id,
    )
    op = str(result.get("op", "")).strip().lower()
    params = result.get("params") or {}
    if op not in ("scale", "recolor", "reposition", "rename"):
        raise LLMError(f"extract_property_params returned an unknown op: {op!r}")
    return {"op": op, "params": params}


def health_check() -> dict:
    """Cheap reachability probe used by the Settings panel and verify script."""
    try:
        reply = chat(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0.0, max_tokens=10, timeout=20.0,
        )
        return {"ok": True, "provider": SETTINGS.llm_provider, "model": SETTINGS.llm_model, "reply": reply.strip()[:40]}
    except LLMError as exc:
        return {"ok": False, "provider": SETTINGS.llm_provider, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "provider": SETTINGS.llm_provider, "error": f"{type(exc).__name__}: {exc}"}
