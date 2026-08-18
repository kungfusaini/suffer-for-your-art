"""Local gallery web app for browsing and replaying painting runs."""

from __future__ import annotations

import hashlib
import http.server
import json
import re
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from .feedback import (
    RECENT_FEEDBACK_PREFIX,
    REFERENCE_IMAGE_MESSAGE,
    STUDIO_GUIDE_MESSAGE,
    recent_feedback_text,
    recent_move_list,
    render_recent_stroke_map,
    render_studio_guide,
)
from .nanogpt import build_chat_payload, build_multimodal_user_message, serialize_chat_payload
from .planner import (
    CANVAS_MESSAGE,
    LEGACY_CANVAS_MESSAGE,
    LEGACY_PROMPT_TEMPLATE_VERSION,
    CAUSAL_FEEDBACK_PROMPT_TEMPLATE_VERSION,
    PROMPT_TEMPLATE_VERSION,
    REFLECTIVE_PROMPT_TEMPLATE_VERSION,
    STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
    TITLE_PROMPT_TEMPLATE_VERSION,
    WORKING_MEMORY_PROMPT_TEMPLATE_VERSION,
    SCHEMA_FIRST_PROMPT_TEMPLATE_VERSION,
    build_history_summary,
    prompt_for_model,
    prompt_for_model_v1,
    prompt_for_model_v2,
    prompt_for_model_v3,
    prompt_for_model_v4,
    prompt_for_model_v5,
    prompt_for_model_v6,
    prompt_for_model_v7,
)

from .brush import _blend_seed, render_strokes
from .validation import ValidationError, validate_strokes

# Minimal run id validator to prevent traversal and arbitrary path access.
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class GalleryError(Exception):
    """Domain error for gallery routing/validation failures."""

    status = HTTPStatus.BAD_REQUEST

    def __init__(self, message: str, status: HTTPStatus | int | None = None) -> None:
        super().__init__(message)
        if status is not None:
            self.status = HTTPStatus(status)


class GalleryNotFoundError(GalleryError):
    """Raised for missing or inaccessible gallery resources."""

    def __init__(self, message: str) -> None:
        super().__init__(message, HTTPStatus.NOT_FOUND)


class GalleryPayloadError(GalleryError):
    """Raised for malformed run JSON / process payloads."""

    def __init__(self, message: str) -> None:
        super().__init__(message, HTTPStatus.BAD_REQUEST)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    process_path: Path
    final_path: Path

    @property
    def canvas_size(self) -> int:
        canvas = self.manifest.get("canvas", {})
        return int(canvas.get("size", 0) or 0)

    @property
    def total_strokes(self) -> int:
        return int(self.manifest.get("total_strokes", 0) or 0)

    @property
    def cache_dir(self) -> Path:
        return self.run_dir / "_gallery_cache"

    @property
    def manifest_path(self) -> Path:
        return Path(self.run_dir / "manifest.json")


def validate_run_id(value: str) -> str:
    """Return a validated run_id path token.

    The run ID is intentionally constrained to a simple token shape so it can only
    address children of the artifacts directory.
    """

    if not isinstance(value, str):
        raise GalleryPayloadError("run_id must be a string")
    run_id = value.strip()
    if not RUN_ID_RE.fullmatch(run_id):
        raise GalleryPayloadError("invalid run_id")
    return run_id


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GalleryPayloadError(f"invalid JSON in {path.name}") from exc

    if not isinstance(raw, dict):
        raise GalleryPayloadError(f"manifest in {path.name} is not an object")
    return raw


def _safe_mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(ts).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ensure_within_root(root: Path, relative_path: str) -> Path:
    root_path = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise GalleryPayloadError(f"invalid frame path in manifest/event: {relative_path}") from exc
    return candidate


def _parse_round_query(value: str | None, *, event_count: int | None = None) -> int:
    if value is None:
        raise GalleryPayloadError("round query parameter is required")
    try:
        round_index = int(value)
    except ValueError as exc:
        raise GalleryPayloadError("round must be an integer") from exc
    if round_index < 1:
        raise GalleryPayloadError("round must be >= 1")
    if event_count is not None and round_index > event_count:
        raise GalleryPayloadError(f"round must be between 1 and {event_count}")
    return round_index


def _parse_raw_mode(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _redact_request_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    redacted = []
    for message in messages:
        redacted_entry: dict[str, Any] = {
            k: v
            for k, v in message.items()
            if k in {"role", "label"}
        }
        redacted_entry["content"] = message.get("content", "")
        if isinstance(redacted_entry["content"], list):
            redacted_content: list[dict[str, Any]] = []
            for part in redacted_entry["content"]:
                if not isinstance(part, dict):
                    redacted_content.append({"_type": type(part).__name__})
                    continue
                if part.get("type") == "image_url":
                    redacted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": "[redacted-data-url]"},
                        }
                    )
                else:
                    redacted_content.append({k: v for k, v in part.items() if k != "image_url"})
            redacted_entry["content"] = redacted_content
        redacted.append(redacted_entry)
    return redacted


def _event_strokes_used(event: dict[str, Any]) -> int:
    raw_count = event.get("strokes_count")
    if raw_count is not None:
        return int(raw_count)
    return len(event.get("strokes", []))


def reconstruct_round_request_payload(
    artifacts_dir: Path | str,
    run_id: str,
    round_index: int,
) -> dict[str, Any]:
    """Rebuild the outbound NanoGPT request for one historical paint round.

    The reconstruction mirrors PainterRunner's inputs exactly for the requested
    round: prompt history summary, remaining-budget batch calculation, and
    multimodal user message with the prior frame.
    """

    run_ctx = get_run_context(artifacts_dir, run_id)
    manifest = run_ctx.manifest

    model = manifest.get("model")
    if not isinstance(model, str) or not model.strip():
        raise GalleryPayloadError("run has no model; context reconstruction is only available for paint runs")

    try:
        max_strokes = int(manifest.get("max_strokes", 1))
        max_rounds = int(manifest.get("max_rounds", 0))
        events = load_run_events(run_ctx.run_dir, max_strokes=max_strokes)
    except (TypeError, ValueError) as exc:
        raise GalleryPayloadError("invalid manifest limits") from exc

    event_count = len(events)
    if not 1 <= round_index <= event_count:
        raise GalleryPayloadError(f"round must be between 1 and {event_count}")

    try:
        stroke_budget = (
            int(manifest.get("stroke_budget"))
            if manifest.get("stroke_budget") is not None
            else None
        )
    except (TypeError, ValueError) as exc:
        raise GalleryPayloadError("invalid stroke_budget in manifest") from exc

    history = events[: round_index - 1]
    strokes_used = sum(_event_strokes_used(event) for event in history)
    remaining = stroke_budget - strokes_used if stroke_budget is not None else max_strokes
    round_limit = min(max_strokes, remaining)
    prompt_template_version = str(
        manifest.get("prompt_template_version", LEGACY_PROMPT_TEMPLATE_VERSION)
    )
    if round_limit <= 0 and prompt_template_version not in {
        REFLECTIVE_PROMPT_TEMPLATE_VERSION,
        STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
        TITLE_PROMPT_TEMPLATE_VERSION,
        PROMPT_TEMPLATE_VERSION,
    }:
        raise GalleryPayloadError("no budget remains for the requested round")
    if prompt_template_version in {
        REFLECTIVE_PROMPT_TEMPLATE_VERSION,
        STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
        TITLE_PROMPT_TEMPLATE_VERSION,
        PROMPT_TEMPLATE_VERSION,
    } and round_index == max_rounds:
        round_limit = 0
    include_recent_feedback = False
    include_studio_guide = False
    if prompt_template_version == LEGACY_PROMPT_TEMPLATE_VERSION:
        prompt_builder = prompt_for_model_v1
        canvas_message = LEGACY_CANVAS_MESSAGE
    elif prompt_template_version == SCHEMA_FIRST_PROMPT_TEMPLATE_VERSION:
        prompt_builder = prompt_for_model_v2
        canvas_message = CANVAS_MESSAGE
    elif prompt_template_version == CAUSAL_FEEDBACK_PROMPT_TEMPLATE_VERSION:
        prompt_builder = prompt_for_model_v3
        canvas_message = CANVAS_MESSAGE
        include_recent_feedback = True
    elif prompt_template_version == REFLECTIVE_PROMPT_TEMPLATE_VERSION:
        prompt_builder = prompt_for_model_v4
        canvas_message = CANVAS_MESSAGE
        include_recent_feedback = True
    elif prompt_template_version == STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION:
        prompt_builder = prompt_for_model_v5
        canvas_message = CANVAS_MESSAGE
        include_recent_feedback = True
        include_studio_guide = True
    elif prompt_template_version == TITLE_PROMPT_TEMPLATE_VERSION:
        prompt_builder = prompt_for_model_v6
        canvas_message = CANVAS_MESSAGE
        include_recent_feedback = True
        include_studio_guide = True
    elif prompt_template_version == WORKING_MEMORY_PROMPT_TEMPLATE_VERSION:
        prompt_builder = prompt_for_model_v7
        canvas_message = CANVAS_MESSAGE
        include_recent_feedback = True
        include_studio_guide = True
    elif prompt_template_version == PROMPT_TEMPLATE_VERSION:
        prompt_builder = prompt_for_model
        canvas_message = CANVAS_MESSAGE
        include_recent_feedback = True
        include_studio_guide = True
    else:
        raise GalleryPayloadError(
            f"unsupported prompt_template_version: {prompt_template_version}"
        )

    summary = build_history_summary(history)
    messages = prompt_builder(
        prompt=str(manifest.get("prompt", "")),
        round_index=round_index,
        summary=summary,
        max_strokes=round_limit,
        max_rounds=max_rounds,
        strokes_used=strokes_used,
        stroke_budget=stroke_budget,
        exact_strokes=(
            round_limit
            if stroke_budget is not None
            and prompt_template_version not in {
                REFLECTIVE_PROMPT_TEMPLATE_VERSION,
                PROMPT_TEMPLATE_VERSION,
            }
            else None
        ),
    )

    previous_event: dict[str, Any] | None = None
    if round_index == 1:
        frame_path = run_ctx.run_dir / "frames" / "frame_000.png"
    else:
        previous_event = events[round_index - 2]
        previous_frame = previous_event.get("frame")
        if not isinstance(previous_frame, str) or not previous_frame:
            raise GalleryPayloadError("missing prior frame path for requested round")
        frame_path = _ensure_within_root(run_ctx.run_dir, previous_frame)

    try:
        frame_path = frame_path.resolve()
    except OSError as exc:
        raise GalleryPayloadError("invalid frame path in run data") from exc

    if not frame_path.exists():
        raise GalleryPayloadError("prior frame image not found")

    messages.append(build_multimodal_user_message(canvas_message, str(frame_path)))

    reference_image_path: Path | None = None
    if prompt_template_version == PROMPT_TEMPLATE_VERSION:
        reference_value = manifest.get("reference_image")
        if isinstance(reference_value, str) and reference_value:
            reference_image_path = _ensure_within_root(run_ctx.run_dir, reference_value)
            if not reference_image_path.exists():
                raise GalleryPayloadError("reference image not found")
            messages.append(
                build_multimodal_user_message(
                    REFERENCE_IMAGE_MESSAGE,
                    str(reference_image_path),
                )
            )

    studio_guide_path: Path | None = None
    studio_guide_sha256: str | None = None
    if include_studio_guide:
        studio_guide_path = (
            run_ctx.run_dir / "context" / f"studio_guide_round_{round_index:03d}.png"
        )
        if not studio_guide_path.exists():
            render_studio_guide(frame_path, studio_guide_path)
        studio_guide_sha256 = _sha256_bytes(studio_guide_path.read_bytes())
        messages.append(
            build_multimodal_user_message(STUDIO_GUIDE_MESSAGE, str(studio_guide_path))
        )

    recent_moves: list[dict[str, Any]] = []
    recent_map_path: Path | None = None
    recent_map_sha256: str | None = None
    if include_recent_feedback and previous_event is not None:
        recent_strokes = [dict(stroke) for stroke in previous_event["strokes"]]
        previous_round_seed_value = previous_event.get("round_seed")
        previous_round_seed = (
            int(previous_round_seed_value)
            if previous_round_seed_value is not None
            else _blend_seed(int(manifest["seed"]), round_index - 1)
        )
        for stroke_index, stroke in enumerate(recent_strokes):
            if stroke.get("seed") is None:
                stroke["seed"] = _blend_seed(previous_round_seed, stroke_index)
        recent_moves = recent_move_list(recent_strokes)
        recent_map_path = (
            run_ctx.run_dir
            / "context"
            / f"recent_strokes_round_{round_index:03d}.png"
        )
        if not recent_map_path.exists():
            render_recent_stroke_map(
                frame_path,
                recent_strokes,
                recent_map_path,
                round_seed=previous_round_seed,
                run_seed=int(manifest["seed"]),
            )
        recent_map_sha256 = _sha256_bytes(recent_map_path.read_bytes())
        messages.append(
            build_multimodal_user_message(
                recent_feedback_text(recent_moves),
                str(recent_map_path),
            )
        )

    readable_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        text_content = ""
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            text_content = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if message.get("role") == "system":
            label = "system"
        elif text_content.startswith(RECENT_FEEDBACK_PREFIX):
            label = "recent_stroke_map"
        elif text_content == REFERENCE_IMAGE_MESSAGE:
            label = "reference_image"
        elif text_content == STUDIO_GUIDE_MESSAGE:
            label = "studio_guide"
        elif text_content == canvas_message:
            label = "multimodal_canvas"
        else:
            label = "context_action_contract"
        readable_messages.append(
            {
                "label": label,
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }
        )

    reflective = prompt_template_version in {
        REFLECTIVE_PROMPT_TEMPLATE_VERSION,
        STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
        TITLE_PROMPT_TEMPLATE_VERSION,
        PROMPT_TEMPLATE_VERSION,
    }
    payload = build_chat_payload(
        messages=messages,
        model=model,
        min_strokes=(
            0 if reflective else (round_limit if stroke_budget is not None else 1)
        ),
        max_strokes=round_limit,
        reflective=reflective,
        include_title=prompt_template_version == PROMPT_TEMPLATE_VERSION,
    )

    return {
        "run_id": run_ctx.run_id,
        "round": round_index,
        "event_count": event_count,
        "strokes_before": strokes_used,
        "required_batch_size": round_limit,
        "canvas_path": str(frame_path.relative_to(run_ctx.run_dir)),
        "canvas_sha256": _sha256_bytes(frame_path.read_bytes()),
        "reference_image_path": (
            str(reference_image_path.relative_to(run_ctx.run_dir))
            if reference_image_path is not None
            else None
        ),
        "studio_guide_path": (
            str(studio_guide_path.relative_to(run_ctx.run_dir))
            if studio_guide_path is not None
            else None
        ),
        "studio_guide_sha256": studio_guide_sha256,
        "recent_moves": recent_moves,
        "recent_stroke_map_path": (
            str(recent_map_path.relative_to(run_ctx.run_dir))
            if recent_map_path is not None
            else None
        ),
        "recent_stroke_map_sha256": recent_map_sha256,
        "model": model,
        "prompt_template_version": prompt_template_version,
        "reconstruction": {
            "status": "deterministically_reconstructed",
            "caveat": (
                "The outbound body was not archived; this was rebuilt from the "
                "persisted manifest, event history, saved pre-round frame, "
                "and the same pure request builders."
            ),
        },
        "messages": readable_messages,
        "response_schema": payload["response_format"],
        "request_payload": payload,
    }


def discover_runs(artifacts_dir: Path | str) -> list[dict[str, Any]]:
    """Return valid gallery runs ordered newest-first."""

    root = Path(artifacts_dir)
    if not root.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        try:
            run_id = validate_run_id(run_id)
        except GalleryError:
            continue

        manifest_path = run_dir / "manifest.json"
        final_path = run_dir / "final.png"
        process_path = run_dir / "process.jsonl"

        if not manifest_path.exists() or not final_path.exists() or not process_path.exists():
            continue

        try:
            run_ctx = get_run_context(root, run_id)
            manifest = run_ctx.manifest
            total_strokes = run_ctx.total_strokes
            round_records = manifest.get("round_records", [])
            if not isinstance(round_records, list):
                round_records = []

            prompt = str(manifest.get("prompt", "")).strip()
            model = manifest.get("model")
            created_at = str(manifest.get("created_at", ""))
            run_date = created_at.split("T", 1)[0] if created_at else ""
            title = _short_title(prompt)
            runs.append(
                {
                    "run_id": run_id,
                    "title": title,
                    "prompt": prompt,
                    "model": model,
                    "status": manifest.get("status"),
                    "stroke_count": total_strokes,
                    "round_count": int(manifest.get("rounds_executed", 0) or 0),
                    "run_date": run_date,
                    "created_at": created_at,
                    "thumbnail": f"/api/runs/{run_id}/thumbnail",
                    "round_records": round_records,
                    "updated_at": _safe_mtime_iso(run_dir),
                }
            )
        except (GalleryError, KeyError, TypeError, ValueError, OSError):
            continue

    runs.sort(
        key=lambda run: (
            run.get("created_at", "") or "",
            run.get("updated_at", ""),
            run.get("run_id", ""),
        ),
        reverse=True,
    )
    return runs


def get_run_context(artifacts_dir: Path | str, run_id: str) -> RunContext:
    root = Path(artifacts_dir).resolve()
    validated_run_id = validate_run_id(run_id)
    run_dir = (root / validated_run_id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise GalleryNotFoundError("run directory is outside the artifacts root") from exc
    if not run_dir.is_dir():
        raise GalleryNotFoundError("run directory not found")

    manifest_path = run_dir / "manifest.json"
    process_path = run_dir / "process.jsonl"
    final_path = run_dir / "final.png"

    if not manifest_path.exists() or not process_path.exists() or not final_path.exists():
        raise GalleryNotFoundError("run is missing required files")

    manifest = dict(_read_json(manifest_path))
    try:
        max_strokes = int(manifest.get("max_strokes", 1) or 1)
        events = load_run_events(run_dir, max_strokes=max_strokes)
    except (TypeError, ValueError) as exc:
        raise GalleryPayloadError("invalid max_strokes in manifest") from exc
    actual_total = sum(len(event["strokes"]) for event in events)
    if actual_total < 1:
        raise GalleryNotFoundError("run has no recorded strokes")
    # Older manifests predate total_strokes; the process log is canonical.
    manifest["total_strokes"] = actual_total

    return RunContext(
        run_id=validated_run_id,
        run_dir=run_dir,
        manifest=manifest,
        process_path=process_path,
        final_path=final_path,
    )


def _short_title(prompt: str, max_words: int = 8) -> str:
    compact = " ".join(prompt.split())
    if not compact:
        return "Untitled run"
    parts = compact.split()
    if len(parts) <= max_words:
        return compact
    return " ".join(parts[:max_words]) + "..."


def load_run_events(run_dir: Path, max_strokes: int) -> list[dict[str, Any]]:
    process_path = run_dir / "process.jsonl"
    if not process_path.exists():
        raise GalleryNotFoundError("run process log missing")

    events: list[dict[str, Any]] = []
    for line_num, raw_line in enumerate(process_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise GalleryPayloadError(f"invalid event on line {line_num}: must be an object")
            strokes = validate_strokes(
                payload.get("strokes"),
                max_strokes=max_strokes,
                allow_empty=bool(payload.get("done")),
            )
            payload["strokes"] = strokes
            payload["round_seed"] = payload.get("round_seed")
            events.append(payload)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise GalleryPayloadError(f"invalid event on line {line_num}: {exc}") from exc
    if not events:
        raise GalleryNotFoundError("run has no recorded events")
    return events


def flatten_run_strokes(run_ctx: RunContext) -> list[dict[str, Any]]:
    manifest = run_ctx.manifest
    max_strokes = int(manifest.get("max_strokes", 1))
    run_seed = int(manifest.get("seed"))
    events = load_run_events(run_ctx.run_dir, max_strokes=max_strokes)

    strokes: list[dict[str, Any]] = []
    for round_index, event in enumerate(events):
        round_seed_value = event.get("round_seed")
        round_seed = int(round_seed_value) if round_seed_value is not None else None
        event_strokes = event["strokes"]
        for stroke_index, stroke in enumerate(event_strokes):
            stroke_copy = dict(stroke)
            if stroke_copy.get("seed") is None:
                effective_round_seed = (
                    round_seed
                    if round_seed is not None
                    else _blend_seed(run_seed, round_index + 1)
                )
                fallback_seed = _blend_seed(effective_round_seed, stroke_index)
                stroke_copy["seed"] = fallback_seed
            strokes.append(stroke_copy)
    return strokes


def _event_text(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    return value.strip() if isinstance(value, str) else ""


def run_details(artifacts_dir: Path | str, run_id: str) -> dict[str, Any]:
    run_ctx = get_run_context(artifacts_dir, run_id)
    manifest = run_ctx.manifest
    max_strokes = int(manifest.get("max_strokes", 1) or 1)
    events = load_run_events(run_ctx.run_dir, max_strokes=max_strokes)
    final_event = events[-1]
    rationale_event = events[1] if len(events) > 1 else events[0]
    return {
        "run_id": run_ctx.run_id,
        "title": (
            manifest.get("title")
            or _event_text(final_event, "title")
            or _short_title(str(manifest.get("prompt", "")))
        ),
        "prompt": manifest.get("prompt", ""),
        "model": manifest.get("model"),
        "status": manifest.get("status"),
        "rounds_executed": manifest.get("rounds_executed", 0),
        "round_limit": manifest.get("max_rounds"),
        "stroke_budget": manifest.get("stroke_budget"),
        "total_strokes": run_ctx.total_strokes,
        "canvas_size": run_ctx.canvas_size,
        "created_at": manifest.get("created_at"),
        "mode": manifest.get("mode"),
        "initial_vision": (
            manifest.get("initial_idea")
            or _event_text(events[0], "initial_idea")
            or _event_text(rationale_event, "interpretation")
        ),
        "working_state": manifest.get("working_state") or final_event.get("working_state", {}),
        "final_description": (
            manifest.get("final_reflection")
            or _event_text(final_event, "final_reflection")
            or _event_text(final_event, "observation")
        ),
        # Backwards-compatible aliases for older clients.
        "final_observation": (
            manifest.get("final_reflection")
            or _event_text(final_event, "final_reflection")
            or _event_text(final_event, "observation")
        ),
        "artistic_rationale": (
            manifest.get("initial_idea")
            or _event_text(events[0], "initial_idea")
            or _event_text(rationale_event, "interpretation")
        ),
        "updated_at": _safe_mtime_iso(run_ctx.run_dir),
    }


def run_events_summary(artifacts_dir: Path | str, run_id: str) -> dict[str, Any]:
    run_ctx = get_run_context(artifacts_dir, run_id)
    max_strokes = int(run_ctx.manifest.get("max_strokes", 1))
    events = load_run_events(run_ctx.run_dir, max_strokes=max_strokes)

    round_info = []
    for event in events:
        round_info.append(
            {
                "round": int(event.get("round", 0) or 0),
                "done": bool(event.get("done", False)),
                "strokes_count": int(len(event.get("strokes", []))),
                "total_strokes": int(event.get("total_strokes", 0) or 0),
            }
        )

    return {
        "run_id": run_ctx.run_id,
        "events": round_info,
        "event_count": len(round_info),
    }


def _frame_cache_path(cache_dir: Path, stroke_count: int) -> Path:
    return cache_dir / f"frame_{stroke_count:05d}.png"


def _parse_stroke_index(value: str | None, total_strokes: int) -> int:
    if value is None:
        raise GalleryPayloadError("strokes query parameter is required")
    try:
        stroke_count = int(value)
    except ValueError as exc:
        raise GalleryPayloadError("strokes must be an integer") from exc
    if stroke_count < 0:
        raise GalleryPayloadError("strokes must be >= 0")
    if stroke_count > total_strokes:
        raise GalleryPayloadError("strokes exceeds total recorded strokes")
    return stroke_count


def get_cumulative_frame(artifacts_dir: Path | str, run_id: str, stroke_count: int) -> Path:
    """Return a cached path to a cumulative PNG rendered after N strokes."""

    run_ctx = get_run_context(artifacts_dir, run_id)
    if int(stroke_count) < 0:
        raise GalleryPayloadError("strokes must be >= 0")

    total_strokes = run_ctx.total_strokes
    if stroke_count > total_strokes:
        raise GalleryPayloadError("strokes exceeds total recorded strokes")

    run_ctx.cache_dir.mkdir(parents=True, exist_ok=True)
    frame_path = _frame_cache_path(run_ctx.cache_dir, stroke_count)
    if frame_path.exists():
        return frame_path

    strokes = flatten_run_strokes(run_ctx)
    if stroke_count > len(strokes):
        raise GalleryPayloadError("requested strokes exceeds validated event stroke count")

    canvas_size = run_ctx.canvas_size
    if canvas_size <= 0:
        raise GalleryPayloadError("invalid canvas size")

    cached_indexes = []
    for existing in run_ctx.cache_dir.glob("frame_*.png"):
        try:
            cached_indexes.append(int(existing.stem.split("_", 1)[1]))
        except ValueError:
            continue
    prior_indexes = [index for index in cached_indexes if 0 <= index < stroke_count]
    start_index = max(prior_indexes, default=0)

    if start_index > 0:
        current = Image.open(_frame_cache_path(run_ctx.cache_dir, start_index)).convert("RGBA")
    else:
        current = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
        base_path = _frame_cache_path(run_ctx.cache_dir, 0)
        if not base_path.exists():
            current.save(base_path, format="PNG")

    if stroke_count == 0:
        if not frame_path.exists():
            current.save(frame_path, format="PNG")
        return frame_path

    # Fill cache from the nearest existing frame (or blank) to the requested count.
    run_seed = int(run_ctx.manifest.get("seed", 0))
    for index in range(start_index, stroke_count):
        render_strokes(current, strokes=[strokes[index]], seed=run_seed, run_seed=run_seed)
        cached = _frame_cache_path(run_ctx.cache_dir, index + 1)
        if not cached.exists():
            current.save(cached, format="PNG")

    return frame_path


def _mime_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "application/javascript; charset=utf-8"
    if suffix == ".png":
        return "image/png"
    if suffix == ".html":
        return "text/html; charset=utf-8"
    return "application/octet-stream"


class _GalleryRequestHandler(http.server.BaseHTTPRequestHandler):
    artifacts_dir: Path = Path("artifacts")
    static_dir: Path = Path(__file__).resolve().parent / "web"

    def _json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _binary(self, payload: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": str(message)}, status=status)

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route()
        except GalleryError as exc:
            self._error(exc.status, str(exc))
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "an unexpected error occurred")

    def _route(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path.startswith("/api/"):
            self._route_api(parsed)
            return

        if parsed.path == "/" or parsed.path == "":
            self._serve_static("index.html")
            return

        rel = parsed.path.lstrip("/")
        if rel in {"index.html", "styles.css", "app.js"}:
            self._serve_static(rel)
            return

        self._error(HTTPStatus.NOT_FOUND, "not found")

    def _route_api(self, parsed) -> None:
        parts = [segment for segment in parsed.path.split("/") if segment]
        if len(parts) < 2:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        if parts[0] != "api" or parts[1] != "runs":
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return

        query = parse_qs(parsed.query)

        if len(parts) == 2:
            self._json({"runs": discover_runs(self.artifacts_dir)})
            return

        run_id = validate_run_id(parts[2])
        if len(parts) == 3:
            if parts[2] == "":
                self._error(HTTPStatus.NOT_FOUND, "run_id required")
                return
            if run_id == "":
                self._error(HTTPStatus.NOT_FOUND, "run_id required")
                return
            self._json(run_details(self.artifacts_dir, run_id))
            return

        if len(parts) != 4:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return

        action = parts[3]
        if action == "thumbnail":
            run_ctx = get_run_context(self.artifacts_dir, run_id)
            self._binary(run_ctx.final_path.read_bytes(), _mime_for_path(run_ctx.final_path))
            return

        if action == "events":
            self._json(run_events_summary(self.artifacts_dir, run_id))
            return

        if action == "frame":
            strokes_value = query.get("strokes", [None])[0]
            run_ctx = get_run_context(self.artifacts_dir, run_id)
            stroke_count = _parse_stroke_index(strokes_value, run_ctx.total_strokes)
            path = get_cumulative_frame(self.artifacts_dir, run_id, stroke_count)
            self._binary(path.read_bytes(), "image/png")
            return

        if action == "context-map":
            round_value = query.get("round", [None])[0]
            run_ctx = get_run_context(self.artifacts_dir, run_id)
            events = load_run_events(
                run_ctx.run_dir,
                max_strokes=int(run_ctx.manifest.get("max_strokes", 1) or 1),
            )
            round_index = _parse_round_query(round_value, event_count=len(events))
            reconstructed = reconstruct_round_request_payload(
                self.artifacts_dir,
                run_id,
                round_index,
            )
            relative_path = reconstructed.get("recent_stroke_map_path")
            if not isinstance(relative_path, str) or not relative_path:
                raise GalleryNotFoundError("recent-stroke map is not available for this round")
            map_path = _ensure_within_root(run_ctx.run_dir, relative_path)
            self._binary(map_path.read_bytes(), "image/png")
            return

        if action == "context":
            round_value = query.get("round", [None])[0]
            run_ctx = get_run_context(self.artifacts_dir, run_id)
            events = load_run_events(run_ctx.run_dir, max_strokes=int(run_ctx.manifest.get("max_strokes", 1) or 1))
            round_index = _parse_round_query(round_value, event_count=len(events))
            reconstructed = reconstruct_round_request_payload(self.artifacts_dir, run_id, round_index)
            raw_mode = _parse_raw_mode(query.get("raw", [None])[0])
            payload = reconstructed["request_payload"]
            raw_body = serialize_chat_payload(payload)
            if raw_mode:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Request-Sha256", _sha256_bytes(raw_body))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(raw_body)))
                self.end_headers()
                self.wfile.write(raw_body)
                return

            safe_payload = dict(payload)
            safe_payload["messages"] = _redact_request_messages(payload["messages"])
            response = {
                "run_id": reconstructed["run_id"],
                "round": reconstructed["round"],
                "event_count": reconstructed["event_count"],
                "strokes_before": reconstructed["strokes_before"],
                "required_batch_size": reconstructed["required_batch_size"],
                "canvas_url": f"/api/runs/{run_id}/frame?strokes={reconstructed['strokes_before']}",
                "canvas_sha256": reconstructed["canvas_sha256"],
                "recent_moves": reconstructed["recent_moves"],
                "recent_stroke_map_url": (
                    f"/api/runs/{run_id}/context-map?round={reconstructed['round']}"
                    if reconstructed["recent_stroke_map_path"] is not None
                    else None
                ),
                "recent_stroke_map_sha256": reconstructed["recent_stroke_map_sha256"],
                "model": reconstructed["model"],
                "prompt_template_version": reconstructed["prompt_template_version"],
                "reconstruction": reconstructed["reconstruction"],
                "messages": _redact_request_messages(reconstructed["messages"]),
                "response_schema": reconstructed["response_schema"],
                "request": safe_payload,
                "raw_request_sha256": _sha256_bytes(raw_body),
            }
            self._json(response)
            return

        self._error(HTTPStatus.NOT_FOUND, "not found")

    def _serve_static(self, filename: str) -> None:
        if ".." in filename or filename.startswith("/"):
            self._error(HTTPStatus.BAD_REQUEST, "invalid file path")
            return
        path = self.static_dir / filename
        if not path.exists():
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return

        body = path.read_bytes()
        self._binary(body, _mime_for_path(path))

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ANN001
        # Keep server output clean for smoke tests and normal local use.
        return


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, artifacts_dir: str | Path = "artifacts") -> int:
    try:
        bind_host = host.strip()
        if not bind_host:
            raise GalleryError("host cannot be empty")
        if not (1 <= int(port) <= 65535):
            raise GalleryError("port must be in [1, 65535]", status=HTTPStatus.BAD_REQUEST)
    except ValueError as exc:
        raise GalleryError("port must be an integer") from exc

    artifacts_root = Path(artifacts_dir)
    static_dir = Path(__file__).resolve().parent / "web"

    if not static_dir.exists():
        raise GalleryError("web assets not found")

    handler_class = type(
        "GalleryRequestHandler",
        (_GalleryRequestHandler,),
        {"artifacts_dir": artifacts_root, "static_dir": static_dir},
    )

    server = http.server.ThreadingHTTPServer((bind_host, int(port)), handler_class)
    try:
        print(f"Serving gallery from {artifacts_root.resolve()} at http://{bind_host}:{int(port)}")
        print("Press Ctrl-C to stop.")
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "GalleryError",
    "GalleryNotFoundError",
    "GalleryPayloadError",
    "RunContext",
    "discover_runs",
    "get_run_context",
    "run_details",
    "run_events_summary",
    "flatten_run_strokes",
    "get_cumulative_frame",
    "validate_run_id",
    "run_server",
]
