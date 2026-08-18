"""Deterministic, context-only visual feedback for recent brush actions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .brush import render_strokes


RECENT_STROKE_LIMIT = 10
RECENT_FEEDBACK_PREFIX = "RECENT ACTION-OUTPUT MAP"
REFERENCE_IMAGE_MESSAGE = (
    "REFERENCE IMAGE\n"
    "The attached image is visual reference only. Study its subject, proportions, and "
    "relationships, but do not copy pixels or treat the reference as the current canvas. "
    "Your only artwork actions remain the brushstrokes in your response."
)
STUDIO_GUIDE_MESSAGE = (
    "STUDIO GUIDE\n"
    "The attached image is diagnostic context, not part of the painting. It is a washed "
    "copy of CURRENT CANVAS with a faint, unlabelled 3×3 compositional guide. The guide "
    "is only for rough placement and proportion; it is not a measurement grid or paint."
)


def render_studio_guide(
    current_canvas_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Render a quiet compositional guide without changing the canonical canvas."""

    source_path = Path(current_canvas_path)
    destination = Path(output_path)
    with Image.open(source_path) as source_image:
        source = source_image.convert("RGBA")
    grayscale = ImageOps.grayscale(source).convert("RGBA")
    wash = Image.new("RGBA", source.size, (255, 255, 255, 255))
    guide = Image.blend(grayscale, wash, 0.78)

    draw = ImageDraw.Draw(guide, "RGBA")
    width, height = guide.size
    line_color = (91, 104, 91, 22)
    center_color = (91, 104, 91, 34)
    for fraction in (1 / 3, 2 / 3):
        x = round((width - 1) * fraction)
        y = round((height - 1) * fraction)
        draw.line((x, 0, x, height - 1), fill=line_color, width=1)
        draw.line((0, y, width - 1, y), fill=line_color, width=1)
    center_x = round((width - 1) / 2)
    center_y = round((height - 1) / 2)
    radius = max(2, round(min(width, height) / 256))
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=center_color,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    guide.save(destination, format="PNG")
    return destination


def recent_move_list(
    strokes: list[dict[str, Any]],
    *,
    limit: int = RECENT_STROKE_LIMIT,
) -> list[dict[str, Any]]:
    """Return the last validated actions in map-label and execution order."""

    if limit < 1:
        raise ValueError("recent stroke limit must be positive")
    selected = strokes[-limit:]
    first_batch_index = len(strokes) - len(selected) + 1
    moves: list[dict[str, Any]] = []
    for offset, stroke in enumerate(selected):
        action = copy.deepcopy(stroke)
        action.pop("seed", None)
        moves.append(
            {
                "map_label": offset + 1,
                "previous_batch_index": first_batch_index + offset,
                "action": action,
            }
        )
    return moves


def recent_feedback_text(moves: list[dict[str, Any]]) -> str:
    """Describe the diagnostic map without implying that it is paint."""

    if not moves:
        return ""
    return (
        f"{RECENT_FEEDBACK_PREFIX}\n"
        "The attached image is diagnostic context, not part of the painting. "
        "It is an aligned, washed grayscale copy of CURRENT CANVAS with your "
        f"{len(moves)} most recently rendered strokes traced in red and numbered "
        "in execution order. Each numbered entry below is the exact validated "
        "action that produced the matching trace. Use this correspondence to "
        "learn how coordinates, width, pressure, opacity, and taper map to visible marks.\n\n"
        "RECENT MOVES:\n"
        + json.dumps(moves, ensure_ascii=False, separators=(",", ":"))
    )


def render_recent_stroke_map(
    current_canvas_path: str | Path,
    strokes: list[dict[str, Any]],
    output_path: str | Path,
    *,
    round_seed: int,
    run_seed: int,
    limit: int = RECENT_STROKE_LIMIT,
) -> Path:
    """Render a numbered diagnostic trace without mutating the canonical canvas."""

    source_path = Path(current_canvas_path)
    destination = Path(output_path)
    selected = strokes[-limit:]
    if not selected:
        raise ValueError("cannot render a recent-stroke map without strokes")

    with Image.open(source_path) as source_image:
        source = source_image.convert("RGBA")
    grayscale = ImageOps.grayscale(source).convert("RGBA")
    wash = Image.new("RGBA", source.size, (255, 255, 255, 255))
    diagnostic = Image.blend(grayscale, wash, 0.72)

    for stroke in selected:
        highlight = copy.deepcopy(stroke)
        highlight["color"] = "alizarin"
        highlight["opacity"] = 1.0
        render_strokes(
            diagnostic,
            strokes=[highlight],
            seed=round_seed,
            run_seed=run_seed,
        )

    draw = ImageDraw.Draw(diagnostic, "RGBA")
    width, height = diagnostic.size
    radius = max(7, round(min(width, height) / 64))
    font = ImageFont.load_default()
    for label, stroke in enumerate(selected, start=1):
        start = stroke["bezier"]["start"]
        cx = round(float(start[0]) * (width - 1))
        cy = round(float(start[1]) * (height - 1))
        cx = max(radius, min(width - radius - 1, cx))
        cy = max(radius, min(height - radius - 1, cy))
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(36, 33, 28, 245),
        )
        text = str(label)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (cx - text_w / 2, cy - text_h / 2 - bbox[1]),
            text,
            fill=(255, 250, 240, 255),
            font=font,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.save(destination, format="PNG")
    return destination


__all__ = [
    "RECENT_FEEDBACK_PREFIX",
    "RECENT_STROKE_LIMIT",
    "REFERENCE_IMAGE_MESSAGE",
    "STUDIO_GUIDE_MESSAGE",
    "render_studio_guide",
    "recent_feedback_text",
    "recent_move_list",
    "render_recent_stroke_map",
]
