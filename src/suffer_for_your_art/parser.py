"""Robust planner response parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from .validation import ValidationError, validate_planner_payload


FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    for match in FENCED_JSON.findall(text):
        candidates.append(match)

    if not candidates:
        # Fallback: first balanced object in text.
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidates.append(text[first : last + 1])

    # Deduplicate while preserving order.
    deduped: list[str] = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped


def parse_planner_output(
    text: str,
    max_strokes: int = 5,
    *,
    reflective: bool = False,
    require_title: bool = False,
    require_working_state: bool = False,
) -> dict[str, Any]:
    """Parse planner model output into a validated planner payload."""

    if not isinstance(text, str):
        raise ValidationError("planner output must be text")

    trimmed = text.strip()
    if not trimmed:
        raise ValidationError("planner output is empty")

    last_contract_error: ValidationError | None = None
    for candidate in _extract_json_candidates(trimmed):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            return validate_planner_payload(
                payload,
                max_strokes=max_strokes,
                reflective=reflective,
                require_title=require_title,
                require_working_state=require_working_state,
            )
        except ValidationError as exc:
            last_contract_error = exc

    if last_contract_error is not None:
        raise ValidationError(f"planner JSON violated the stroke contract: {last_contract_error}")
    raise ValidationError("planner output did not contain a JSON object")


__all__ = ["parse_planner_output"]
