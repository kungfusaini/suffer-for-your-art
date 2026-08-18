"""Validation for planner and stroke payloads."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .palette import PALETTE


class ValidationError(ValueError):
    """Raised for malformed planner payloads."""


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} must be a number") from exc


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValidationError(f"{field} must be true or false")


def _as_str(value: Any, field: str) -> str:
    if isinstance(value, str):
        return value
    raise ValidationError(f"{field} must be a string")


def validate_point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, Iterable):
        raise ValidationError(f"{label} must be a point pair")
    if not isinstance(value, (list, tuple)):
        # avoid accepting mapping-like iterables from malformed output.
        raise ValidationError(f"{label} must be an array")
    parts = list(value)
    if len(parts) != 2:
        raise ValidationError(f"{label} must have exactly 2 values")
    x = _as_float(parts[0], f"{label}[0]")
    y = _as_float(parts[1], f"{label}[1]")
    return x, y


def validate_normalized_point(value: Any, label: str) -> tuple[float, float]:
    x, y = validate_point(value, label)
    if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
        raise ValidationError(f"{label} must be normalized to [0, 1]")
    return x, y


def validate_bezier(value: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(value, Mapping):
        raise ValidationError("bezier must be an object")
    required = ("start", "control1", "control2", "end")
    missing = [k for k in required if k not in value]
    if missing:
        raise ValidationError(f"bezier missing fields: {', '.join(missing)}")
    return {
        "start": validate_normalized_point(value["start"], "bezier.start"),
        "control1": validate_normalized_point(value["control1"], "bezier.control1"),
        "control2": validate_normalized_point(value["control2"], "bezier.control2"),
        "end": validate_normalized_point(value["end"], "bezier.end"),
    }


def validate_taper(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValidationError("taper must be an object")
    start = _as_float(value.get("start", 0.5), "taper.start")
    end = _as_float(value.get("end", 0.5), "taper.end")
    if not (0.0 <= start <= 1.0) or not (0.0 <= end <= 1.0):
        raise ValidationError("taper.start/end must be in [0, 1]")
    return {"start": start, "end": end}


def validate_stroke(
    value: Mapping[str, Any],
    max_strokes: int = 5,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("stroke must be an object")

    color = _as_str(value.get("color"), "stroke.color").lower()
    if color not in PALETTE:
        raise ValidationError(f"unknown color '{color}'. Allowed: {', '.join(sorted(PALETTE))}")

    width = _as_float(value.get("width"), "stroke.width")
    if not (1.0 <= width <= 30.0):
        raise ValidationError("stroke.width must be in [1, 30]")

    pressure = _as_float(value.get("pressure"), "stroke.pressure")
    if not (0.0 <= pressure <= 1.0):
        raise ValidationError("stroke.pressure must be in [0, 1]")

    opacity = _as_float(value.get("opacity"), "stroke.opacity")
    if not (0.05 <= opacity <= 1.0):
        raise ValidationError("stroke.opacity must be in [0.05, 1]")

    bezier = validate_bezier(value.get("bezier"))

    taper = validate_taper(value.get("taper", {"start": 0.45, "end": 0.8}))
    if "seed" in value:
        try:
            seed = int(value["seed"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("stroke.seed must be an integer") from exc
    else:
        seed = None

    return {
        "color": color,
        "width": width,
        "pressure": pressure,
        "opacity": opacity,
        "bezier": bezier,
        "taper": taper,
        "seed": seed,
    }


def validate_strokes(
    value: Any,
    max_strokes: int,
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError("strokes must be an array")
    if not value and not allow_empty:
        raise ValidationError("strokes must contain at least one stroke")
    if len(value) > max_strokes:
        raise ValidationError(f"too many strokes ({len(value)}), max is {max_strokes}")

    return [validate_stroke(v, max_strokes=max_strokes) for v in value]


def validate_planner_payload(
    value: Any,
    max_strokes: int = 5,
    *,
    reflective: bool = False,
    require_title: bool = False,
    require_working_state: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("planner payload must be a JSON object")

    observation = _as_str(value.get("observation", ""), "observation")
    interpretation = _as_str(value.get("interpretation", ""), "interpretation")
    done = _as_bool(value.get("done"), "done")
    if done is None:
        raise ValidationError("done must be provided")

    strokes = validate_strokes(
        value.get("strokes", []),
        max_strokes=max_strokes,
        allow_empty=reflective and done,
    )
    result = {
        "observation": observation,
        "interpretation": interpretation,
        "done": done,
        "strokes": strokes,
    }
    if reflective:
        result["initial_idea"] = _as_str(value.get("initial_idea", ""), "initial_idea")
        result["title"] = _as_str(value.get("title", ""), "title")
        working_state = value.get("working_state", {})
        if require_working_state and not isinstance(working_state, Mapping):
            raise ValidationError("working_state must be an object")
        if require_working_state:
            required_state = {
                "current_goal", "completed_parts", "last_change", "next_action", "alignment"
            }
            if not required_state.issubset(working_state):
                raise ValidationError("working_state is missing required fields")
            if not isinstance(working_state["completed_parts"], list):
                raise ValidationError("working_state.completed_parts must be an array")
        result["working_state"] = dict(working_state) if isinstance(working_state, Mapping) else {}
        result["final_reflection"] = _as_str(
            value.get("final_reflection", ""), "final_reflection"
        )
        if not result["initial_idea"].strip():
            raise ValidationError("initial_idea must not be empty")
        if require_title and done and not result["title"].strip():
            raise ValidationError("title must not be empty when done is true")
        if done and not result["final_reflection"].strip():
            raise ValidationError("final_reflection must not be empty when done is true")
        if done and strokes:
            raise ValidationError("done=true must use zero strokes so reflection follows drawing")
    return result

__all__ = ["ValidationError", "validate_planner_payload", "validate_strokes", "validate_stroke"]
