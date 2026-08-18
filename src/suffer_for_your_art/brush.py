"""Deterministic Pillow brush renderer for normalized cubic Bezier strokes."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from .palette import PALETTE


def _blend_seed(base_seed: int, *parts: int) -> int:
    data = f"{base_seed}:" + ":".join(str(p) for p in parts)
    digest = hashlib.sha256(data.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _cubic_point(t: float, start: tuple[float, float], control1: tuple[float, float], control2: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    x0, y0 = start
    x1, y1 = control1
    x2, y2 = control2
    x3, y3 = end
    it = 1.0 - t
    b0 = it * it * it
    b1 = 3 * it * it * t
    b2 = 3 * it * t * t
    b3 = t * t * t
    return (
        b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3,
        b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3,
    )


def _sample_bezier(bezier: dict[str, tuple[float, float]], width_px: int, height_px: int, segments: int) -> list[tuple[float, float]]:
    start = bezier["start"]
    control1 = bezier["control1"]
    control2 = bezier["control2"]
    end = bezier["end"]
    points: list[tuple[float, float]] = []
    max_segments = max(1, segments)
    for i in range(max_segments + 1):
        t = i / max_segments
        x, y = _cubic_point(t, start, control1, control2, end)
        points.append((x * (width_px - 1), y * (height_px - 1)))
    return points


def _normal_at_segment(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return (0.0, 0.0)
    inv = 1.0 / length
    return (-dy * inv, dx * inv)


def _segment_width(base_width: float, t: float, taper: dict[str, float], pressure: float) -> float:
    taper_factor = (1.0 - t) * taper["start"] + t * taper["end"]
    return max(0.5, base_width * (0.45 + 0.55 * pressure) * taper_factor)


@dataclass(frozen=True)
class RenderStroke:
    width: int
    opacity: float
    points: list[tuple[float, float]]
    color: tuple[int, int, int]
    seed: int
    pressure: float
    taper: dict[str, float]


def _render_stroke(draw: ImageDraw.ImageDraw, canvas_w: int, canvas_h: int, stroke: RenderStroke) -> None:
    """Render one stylized stroke with seeded bristle variation."""

    segments = _sample_bezier(
        {
            "start": tuple(stroke.points[0]),
            "control1": tuple(stroke.points[1]),
            "control2": tuple(stroke.points[2]),
            "end": tuple(stroke.points[3]),
        },
        canvas_w,
        canvas_h,
        segments=32,
    )

    # Use seeded micro-variation for edge and bristle behaviour.
    rng = random.Random(stroke.seed)
    bristles = 4
    for bristle_idx in range(bristles):
        bristle_phase = bristle_idx / bristles
        width_scale = 0.72 + 0.28 * rng.random()
        alpha_scale = 0.75 + 0.25 * rng.random()

        for i in range(len(segments) - 1):
            p1 = segments[i]
            p2 = segments[i + 1]
            t = i / max(1, len(segments) - 1)
            t2 = bristle_phase + rng.uniform(-0.06, 0.06)
            width = max(
                1,
                int(round(_segment_width(stroke.width, t + 0.5 * (t2 - 0.5), stroke.taper, stroke.pressure) * width_scale)),
            )
            alpha = int(round(stroke.opacity * 255 * alpha_scale * (0.55 + 0.45 * stroke.pressure) * (0.4 + 0.6 * (1.0 - t))))
            if alpha <= 0:
                continue

            nx, ny = _normal_at_segment(p1, p2)
            jitter = 0.25 + 0.75 * stroke.pressure
            jitter_magnitude = (0.5 + rng.uniform(-0.35, 0.35)) * width * jitter
            ox = nx * jitter_magnitude
            oy = ny * jitter_magnitude
            jittered = (
                (p1[0] + ox, p1[1] + oy),
                (p2[0] + ox, p2[1] + oy),
            )
            draw.line(jittered, fill=(*stroke.color, alpha), width=width)


def render_strokes(
    canvas: Image.Image,
    strokes: list[dict[str, Any]],
    seed: int,
    run_seed: int,
) -> None:
    """Apply validated strokes in sequence to an RGBA canvas.

    All marks are deterministic from stroke payload and seeds.
    """
    draw = ImageDraw.Draw(canvas, "RGBA")

    width, height = canvas.size
    for idx, raw in enumerate(strokes):
        rgba = PALETTE[raw["color"]].rgb
        stroke_seed = raw["seed"] if raw["seed"] is not None else _blend_seed(run_seed, seed, idx)
        render_stroke = RenderStroke(
            width=int(max(1, round(float(raw["width"])))),
            opacity=float(raw["opacity"]),
            points=[raw["bezier"]["start"], raw["bezier"]["control1"], raw["bezier"]["control2"], raw["bezier"]["end"],],
            color=rgba,
            seed=stroke_seed,
            pressure=float(raw["pressure"]),
            taper=raw["taper"],
        )
        _render_stroke(draw, width, height, render_stroke)


def image_sha256(canvas: Image.Image) -> str:
    return hashlib.sha256(canvas.tobytes()).hexdigest()


__all__ = ["render_strokes", "image_sha256", "_blend_seed"]
