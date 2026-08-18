"""Minimal NanoGPT subscription client with secret-safe defaults."""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import signal
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib import error, request

from . import __version__
from .palette import PALETTE_ORDER


DEFAULT_BASE_URL = "https://nano-gpt.com/api/subscription/v1"
CHAT_ENDPOINT = f"{DEFAULT_BASE_URL}/chat/completions"
MODELS_ENDPOINT = f"{DEFAULT_BASE_URL}/models?detailed=true"
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
LEGACY_PAINTING_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "painting_round",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["observation", "interpretation", "strokes", "done"],
            "properties": {
                "observation": {"type": "string"},
                "interpretation": {"type": "string"},
                "done": {"type": "boolean"},
                "strokes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "color", "width", "pressure", "opacity", "bezier", "taper"
                        ],
                        "properties": {
                            "color": {"type": "string", "enum": PALETTE_ORDER},
                            "width": {"type": "number", "minimum": 1, "maximum": 30},
                            "pressure": {"type": "number", "minimum": 0, "maximum": 1},
                            "opacity": {"type": "number", "minimum": 0.05, "maximum": 1},
                            "bezier": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["start", "control1", "control2", "end"],
                                "properties": {
                                    name: {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                                    }
                                    for name in ("start", "control1", "control2", "end")
                                },
                            },
                            "taper": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["start", "end"],
                                "properties": {
                                    "start": {"type": "number", "minimum": 0, "maximum": 1},
                                    "end": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

PAINTING_RESPONSE_FORMAT: dict[str, Any] = copy.deepcopy(LEGACY_PAINTING_RESPONSE_FORMAT)
_reflective_schema = PAINTING_RESPONSE_FORMAT["json_schema"]["schema"]
_reflective_schema["required"] = [
    "initial_idea",
    "observation",
    "interpretation",
    "strokes",
    "done",
    "title",
    "final_reflection",
]
_reflective_schema["properties"] = {
    "initial_idea": {
        "type": "string",
        "minLength": 1,
        "description": "State the drawing idea before making the first marks; repeat it unchanged on later rounds.",
    },
    **_reflective_schema["properties"],
    "title": {
        "type": "string",
        "description": "When done is true, give the finished work a concise title; otherwise return an empty string.",
    },
    "final_reflection": {
        "type": "string",
        "description": "When done is true, honestly assess the visible finished drawing; otherwise return an empty string.",
    },
}
_reflective_schema["properties"]["done"]["description"] = (
    "Set true only when stopping. A stopping response must contain zero strokes so the assessment describes the already-visible canvas."
)
_reflective_schema["properties"]["strokes"]["description"] = (
    "Return one or more strokes to continue, or an empty array when done is true."
)

PRE_TITLE_REFLECTIVE_RESPONSE_FORMAT: dict[str, Any] = copy.deepcopy(PAINTING_RESPONSE_FORMAT)
_pre_title_schema = PRE_TITLE_REFLECTIVE_RESPONSE_FORMAT["json_schema"]["schema"]
_pre_title_schema["required"].remove("title")
_pre_title_schema["properties"].pop("title")

_working_state = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "current_goal",
        "completed_parts",
        "last_change",
        "next_action",
        "alignment",
    ],
    "properties": {
        "current_goal": {"type": "string"},
        "completed_parts": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string"},
        },
        "last_change": {"type": "string"},
        "next_action": {"type": "string"},
        "alignment": {
            "type": "string",
            "enum": ["aligned", "uncertain", "drifting"],
        },
    },
}
_reflective_schema["required"].append("working_state")
_reflective_schema["properties"]["working_state"] = _working_state

class NanoGPTError(RuntimeError):
    """A sanitized NanoGPT client error."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status

    def __str__(self) -> str:
        if self.status is None:
            return self.message
        return f"[{self.status}] {self.message}"


def _base_url() -> str:
    return os.getenv("NANOGPT_SUBSCRIPTION_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _read_xdg_api_key() -> str | None:
    path = Path("~/.config/pi/nanogpt.env").expanduser()
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "NANOGPT_API_KEY":
            return value.strip().strip("\"'") or None
    return None


def get_api_key() -> str:
    value = os.getenv("NANOGPT_API_KEY") or _read_xdg_api_key()
    if value:
        return value
    raise NanoGPTError("Missing NANOGPT_API_KEY in the environment or ~/.config/pi/nanogpt.env")


def validate_model(model: str) -> str:
    """Accept plain subscription model IDs, but reject routing suffixes and controls."""
    if not isinstance(model, str) or not model.strip():
        raise NanoGPTError("Model must be a non-empty subscription model id")
    model = model.strip()
    if ":" in model or not _MODEL_ID.fullmatch(model):
        raise NanoGPTError(
            "Model must be a plain subscription model id with no routing/provider suffix"
        )
    return model


def image_to_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Accept": "application/json",
        "User-Agent": f"suffer-for-your-art/{__version__}",
    }


def _decode_json(response: Any) -> dict[str, Any]:
    body = response.read().decode("utf-8", errors="replace")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise NanoGPTError("NanoGPT returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise NanoGPTError("NanoGPT returned an unexpected JSON payload")
    return value


def serialize_chat_payload(payload: dict[str, Any]) -> bytes:
    """Serialize payload exactly as sent by _post_json."""

    return json.dumps(payload).encode("utf-8")


def build_chat_payload(
    messages: list[dict[str, Any]],
    model: str,
    *,
    min_strokes: int = 1,
    max_strokes: int = 5,
    reflective: bool = True,
    include_title: bool = True,
) -> dict[str, Any]:
    """Build a validated chat request payload for NanoGPT."""

    minimum_allowed = 0 if reflective else 1
    if not (minimum_allowed <= min_strokes <= max_strokes <= 20):
        raise NanoGPTError(
            f"Structured stroke count must satisfy {minimum_allowed} <= min <= max <= 20"
        )
    source_format = (
        PAINTING_RESPONSE_FORMAT
        if reflective and include_title
        else (PRE_TITLE_REFLECTIVE_RESPONSE_FORMAT if reflective else LEGACY_PAINTING_RESPONSE_FORMAT)
    )
    response_format = copy.deepcopy(source_format)
    strokes_schema = response_format["json_schema"]["schema"]["properties"]["strokes"]
    strokes_schema["minItems"] = min_strokes
    strokes_schema["maxItems"] = max_strokes
    return {
        "model": validate_model(model),
        "messages": messages,
        "response_format": response_format,
    }


@contextmanager
def _hard_deadline(seconds: int) -> Iterator[None]:
    """Enforce a wall-clock deadline; urllib's timeout only bounds individual socket operations."""
    supported = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    if not supported:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def expire(_signum: int, _frame: object) -> None:
        raise TimeoutError("NanoGPT wall-clock deadline exceeded")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _post_json(payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    headers = _headers()
    headers["Content-Type"] = "application/json"
    req = request.Request(
        f"{_base_url()}/chat/completions",
        method="POST",
        headers=headers,
        data=serialize_chat_payload(payload),
    )
    try:
        with _hard_deadline(timeout):
            with request.urlopen(req, timeout=timeout) as response:
                return _decode_json(response)
    except error.HTTPError as exc:
        # Never include response bodies: providers sometimes echo request metadata.
        raise NanoGPTError("NanoGPT subscription request failed", status=exc.code) from None
    except NanoGPTError:
        raise
    except Exception as exc:
        raise NanoGPTError(f"NanoGPT call failed ({type(exc).__name__})") from None


def list_models(timeout: int = 90) -> list[str]:
    req = request.Request(
        f"{_base_url()}/models?detailed=true",
        method="GET",
        headers=_headers(),
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = _decode_json(response)
    except error.HTTPError as exc:
        raise NanoGPTError("NanoGPT subscription models request failed", status=exc.code) from None
    except NanoGPTError:
        raise
    except Exception as exc:
        raise NanoGPTError(f"Unable to fetch subscription models ({type(exc).__name__})") from None

    items = raw.get("data", raw)
    models: list[str] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(item["id"])
    if not models:
        raise NanoGPTError("Unexpected models payload from the subscription endpoint")
    return models


def chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    *,
    min_strokes: int = 1,
    max_strokes: int = 5,
    timeout: int = 90,
    reflective: bool = True,
    include_title: bool = True,
) -> str:
    payload = build_chat_payload(
        messages=messages,
        model=model,
        min_strokes=min_strokes,
        max_strokes=max_strokes,
        reflective=reflective,
        include_title=include_title,
    )
    reply = _post_json(payload, timeout=timeout)
    try:
        content = reply["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError) as exc:
        raise NanoGPTError("NanoGPT response missing choices[0].message.content") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        text = "".join(part for part in text_parts if isinstance(part, str))
        if text:
            return text
    raise NanoGPTError("NanoGPT response content was not text")


def build_multimodal_user_message(prompt: str, image_path: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_path:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image_path)},
            }
        )
    return {"role": "user", "content": content}


__all__ = [
    "NanoGPTError",
    "list_models",
    "chat_completion",
    "build_chat_payload",
    "serialize_chat_payload",
    "image_to_data_url",
    "build_multimodal_user_message",
    "validate_model",
    "DEFAULT_BASE_URL",
    "CHAT_ENDPOINT",
    "MODELS_ENDPOINT",
    "LEGACY_PAINTING_RESPONSE_FORMAT",
    "PAINTING_RESPONSE_FORMAT",
    "PRE_TITLE_REFLECTIVE_RESPONSE_FORMAT",
]
