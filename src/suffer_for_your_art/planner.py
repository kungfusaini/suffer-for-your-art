"""Planner prompt helpers and deterministic demo planner."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from .palette import PALETTE_ORDER


LEGACY_PROMPT_TEMPLATE_VERSION = "1.0"
SCHEMA_FIRST_PROMPT_TEMPLATE_VERSION = "2.0"
CAUSAL_FEEDBACK_PROMPT_TEMPLATE_VERSION = "3.0"
REFLECTIVE_PROMPT_TEMPLATE_VERSION = "4.0"
STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION = "5.0"
TITLE_PROMPT_TEMPLATE_VERSION = "6.0"
WORKING_MEMORY_PROMPT_TEMPLATE_VERSION = "7.0"
PROMPT_TEMPLATE_VERSION = "8.0"
LEGACY_CANVAS_MESSAGE = (
    "This is the current canvas. Base the next actions on what is actually visible."
)
CANVAS_MESSAGE = "CURRENT CANVAS"


@dataclass(frozen=True)
class PlannerPayload:
    observation: str
    interpretation: str
    done: bool
    strokes: list[dict[str, Any]]


def build_history_summary(
    history: list[dict[str, Any]],
    limit: int = 4,
    *,
    include_working_state: bool = False,
) -> str:
    if not history:
        return "No prior rounds yet."
    lines = []
    for entry in history[-limit:]:
        line = (
            f"Round {entry['round']}: done={entry['done']} | "
            f"observation='{entry['observation']}' | "
            f"interpretation='{entry['interpretation']}' | "
            f"strokes={entry['strokes_count']}"
        )
        if include_working_state and entry.get("working_state"):
            state = entry["working_state"]
            line += (
                f" | goal='{state.get('current_goal', '')}'"
                f" | completed={state.get('completed_parts', [])}"
                f" | changed='{state.get('last_change', '')}'"
                f" | next='{state.get('next_action', '')}'"
                f" | alignment='{state.get('alignment', '')}'"
            )
        lines.append(line)
    return "\n".join(lines)


def prompt_for_model_v1(
    prompt: str,
    round_index: int,
    summary: str,
    max_strokes: int,
    max_rounds: int,
    *,
    strokes_used: int = 0,
    stroke_budget: int | None = None,
    exact_strokes: int | None = None,
) -> list[dict[str, Any]]:
    instruction = (
        "You are an autonomous painter working on an evolving canvas. You cannot generate, "
        "replace, or edit pixels directly: your only actions are the validated brush strokes "
        "you return. Inspect the current canvas, let its existing marks influence your plan, "
        "and return only one strict JSON object with keys observation, interpretation, strokes, done."
    )
    stroke_requirement = (
        f"Return exactly {exact_strokes} strokes"
        if exact_strokes is not None
        else f"Return 1 to {max_strokes} strokes"
    )
    budget_context = (
        " Budget ledger: "
        f"original_total_strokes={stroke_budget}, "
        f"committed_strokes={strokes_used}, "
        f"strokes_remaining={stroke_budget - strokes_used}, "
        f"batch_size={exact_strokes if exact_strokes is not None else max_strokes}, "
        "required_to_use_full_budget=true."
        if stroke_budget is not None
        else ""
    )
    constraints = (
        f"{stroke_requirement}. Each stroke requires: color; width in 1..30; "
        "pressure in 0..1; opacity in 0.05..1; bezier start, control1, control2, and end "
        "as [x,y] coordinates in 0..1; and taper start/end in 0..1. "
        f"Allowed colors: {', '.join(PALETTE_ORDER)}. "
        f"This is round {round_index} of at most {max_rounds}. "
        f"{budget_context} "
        "Use the available canvas deliberately: develop composition, large relationships, and "
        "then refinement rather than clustering every mark in one tiny area. "
        "Set done=true only when the painting is complete or the total budget is exhausted."
    )
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": (
                f"INITIAL INTENTION: {prompt}\n"
                f"PROCESS HISTORY:\n{summary}\n\n"
                f"ACTION CONTRACT:\n{constraints}"
            ),
        },
    ]


def prompt_for_model_v2(
    prompt: str,
    round_index: int,
    summary: str,
    max_strokes: int,
    max_rounds: int,
    *,
    strokes_used: int = 0,
    stroke_budget: int | None = None,
    exact_strokes: int | None = None,
) -> list[dict[str, Any]]:
    """Build minimal user context; the response schema is the action interface."""

    required_batch = exact_strokes if exact_strokes is not None else max_strokes
    round_state = [
        f"round={round_index}",
        f"round_limit={max_rounds}",
        f"committed_strokes={strokes_used}",
    ]
    if stroke_budget is not None:
        round_state.extend(
            [
                f"original_total_strokes={stroke_budget}",
                f"strokes_remaining={stroke_budget - strokes_used}",
                f"required_batch_size={required_batch}",
            ]
        )
    else:
        round_state.append(f"maximum_batch_size={max_strokes}")
    round_state_text = "\n".join(round_state)

    return [
        {
            "role": "user",
            "content": (
                f"INITIAL INTENTION: {prompt}\n\n"
                f"ROUND STATE:\n{round_state_text}\n\n"
                f"PROCESS HISTORY:\n{summary}"
            ),
        }
    ]


def prompt_for_model_v3(
    prompt: str,
    round_index: int,
    summary: str,
    max_strokes: int,
    max_rounds: int,
    *,
    strokes_used: int = 0,
    stroke_budget: int | None = None,
    exact_strokes: int | None = None,
) -> list[dict[str, Any]]:
    """Build historical template 3 context."""

    return prompt_for_model_v2(
        prompt,
        round_index,
        summary,
        max_strokes,
        max_rounds,
        strokes_used=strokes_used,
        stroke_budget=stroke_budget,
        exact_strokes=exact_strokes,
    )


def prompt_for_model_v4(
    prompt: str,
    round_index: int,
    summary: str,
    max_strokes: int,
    max_rounds: int,
    *,
    strokes_used: int = 0,
    stroke_budget: int | None = None,
    exact_strokes: int | None = None,
) -> list[dict[str, Any]]:
    """Build template 4 context; its stroke budget is a ceiling, not a quota."""

    del exact_strokes
    round_state = [
        f"round={round_index}",
        f"round_limit={max_rounds}",
        f"committed_strokes={strokes_used}",
        f"maximum_batch_size={max_strokes}",
        "may_stop_early=true",
    ]
    if stroke_budget is not None:
        round_state.extend(
            [
                f"maximum_total_strokes={stroke_budget}",
                f"strokes_remaining={stroke_budget - strokes_used}",
                "stroke_budget_is_ceiling=true",
            ]
        )
    round_state_text = "\n".join(round_state)
    return [
        {
            "role": "user",
            "content": (
                f"INITIAL INTENTION: {prompt}\n\n"
                f"ROUND STATE:\n{round_state_text}\n\n"
                f"PROCESS HISTORY:\n{summary}"
            ),
        }
    ]


def prompt_for_model(
    prompt: str,
    round_index: int,
    summary: str,
    max_strokes: int,
    max_rounds: int,
    *,
    strokes_used: int = 0,
    stroke_budget: int | None = None,
    exact_strokes: int | None = None,
) -> list[dict[str, Any]]:
    """Build current template 7 context; working state is in process history."""

    return prompt_for_model_v6(
        prompt,
        round_index,
        summary,
        max_strokes,
        max_rounds,
        strokes_used=strokes_used,
        stroke_budget=stroke_budget,
        exact_strokes=exact_strokes,
    )


def prompt_for_model_v5(
    prompt: str,
    round_index: int,
    summary: str,
    max_strokes: int,
    max_rounds: int,
    *,
    strokes_used: int = 0,
    stroke_budget: int | None = None,
    exact_strokes: int | None = None,
) -> list[dict[str, Any]]:
    """Build historical template 5 context."""

    return prompt_for_model_v4(
        prompt,
        round_index,
        summary,
        max_strokes,
        max_rounds,
        strokes_used=strokes_used,
        stroke_budget=stroke_budget,
        exact_strokes=exact_strokes,
    )


def prompt_for_model_v6(
    prompt: str,
    round_index: int,
    summary: str,
    max_strokes: int,
    max_rounds: int,
    *,
    strokes_used: int = 0,
    stroke_budget: int | None = None,
    exact_strokes: int | None = None,
) -> list[dict[str, Any]]:
    """Build historical template 6 context."""

    return prompt_for_model_v5(
        prompt,
        round_index,
        summary,
        max_strokes,
        max_rounds,
        strokes_used=strokes_used,
        stroke_budget=stroke_budget,
        exact_strokes=exact_strokes,
    )


def prompt_for_model_v7(
    prompt: str,
    round_index: int,
    summary: str,
    max_strokes: int,
    max_rounds: int,
    *,
    strokes_used: int = 0,
    stroke_budget: int | None = None,
    exact_strokes: int | None = None,
) -> list[dict[str, Any]]:
    """Build historical template 7 context."""

    return prompt_for_model_v6(
        prompt,
        round_index,
        summary,
        max_strokes,
        max_rounds,
        strokes_used=strokes_used,
        stroke_budget=stroke_budget,
        exact_strokes=exact_strokes,
    )


def deterministic_demo_plan(
    prompt: str,
    round_index: int,
    max_strokes: int,
    max_rounds: int,
    seed: int,
    *,
    exact_strokes: int | None = None,
) -> dict[str, Any]:
    """Create a deterministic no-network planner response for demo mode."""
    digest = hashlib.sha256(f"{seed}:{prompt}:{round_index}".encode("utf-8")).hexdigest()
    base_value = int(digest[:8], 16)
    rng = random.Random(base_value)

    # Make visible progress and keep every round in-band with 1..max_strokes strokes.
    done = round_index >= max_rounds
    stroke_count = exact_strokes if exact_strokes is not None else (
        1 if done else 1 + (base_value % max_strokes)
    )

    strokes: list[dict[str, Any]] = []
    for i in range(stroke_count):
        start = (rng.random(), rng.random())
        ctrl1 = (
            max(0.0, min(1.0, start[0] + rng.uniform(-0.25, 0.25))),
            max(0.0, min(1.0, start[1] + rng.uniform(-0.25, 0.25))),
        )
        ctrl2 = (
            max(0.0, min(1.0, ctrl1[0] + rng.uniform(-0.25, 0.25))),
            max(0.0, min(1.0, ctrl1[1] + rng.uniform(-0.25, 0.25))),
        )
        end = (
            max(0.0, min(1.0, ctrl2[0] + rng.uniform(-0.25, 0.25))),
            max(0.0, min(1.0, ctrl2[1] + rng.uniform(-0.25, 0.25))),
        )

        strokes.append(
            {
                "color": PALETTE_ORDER[base_value % len(PALETTE_ORDER)],
                "width": 4 + (base_value % 12),
                "pressure": round(0.45 + (rng.random() * 0.45), 3),
                "opacity": round(0.35 + (rng.random() * 0.55), 3),
                "bezier": {
                    "start": [start[0], start[1]],
                    "control1": [ctrl1[0], ctrl1[1]],
                    "control2": [ctrl2[0], ctrl2[1]],
                    "end": [end[0], end[1]],
                },
                "taper": {
                    "start": round(0.45 + rng.uniform(0.0, 0.45), 2),
                    "end": round(0.30 + rng.uniform(0.0, 0.50), 2),
                },
            }
        )

    return {
        "observation": f"Round {round_index}: planned {stroke_count} deterministic brush strokes.",
        "interpretation": (
            f"Seeded brush plan for '{prompt[:42]}'; "
            f"{'halting after this round' if done else 'continuing next iteration'}"
        ),
        "strokes": strokes,
        "done": done,
    }


def planner_output_json(payload: dict[str, Any]) -> str:
    # Strict JSON output helper for tests.
    return json.dumps(payload, indent=2)


__all__ = [
    "CANVAS_MESSAGE",
    "LEGACY_CANVAS_MESSAGE",
    "LEGACY_PROMPT_TEMPLATE_VERSION",
    "SCHEMA_FIRST_PROMPT_TEMPLATE_VERSION",
    "CAUSAL_FEEDBACK_PROMPT_TEMPLATE_VERSION",
    "REFLECTIVE_PROMPT_TEMPLATE_VERSION",
    "STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION",
    "TITLE_PROMPT_TEMPLATE_VERSION",
    "WORKING_MEMORY_PROMPT_TEMPLATE_VERSION",
    "PROMPT_TEMPLATE_VERSION",
    "PlannerPayload",
    "build_history_summary",
    "prompt_for_model",
    "prompt_for_model_v1",
    "prompt_for_model_v2",
    "prompt_for_model_v3",
    "prompt_for_model_v4",
    "prompt_for_model_v5",
    "prompt_for_model_v6",
    "prompt_for_model_v7",
    "deterministic_demo_plan",
    "planner_output_json",
]
