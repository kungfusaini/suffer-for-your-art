"""Painting execution loop and filesystem outputs."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypedDict

from PIL import Image

from .brush import _blend_seed, image_sha256, render_strokes
from .feedback import (
    REFERENCE_IMAGE_MESSAGE,
    STUDIO_GUIDE_MESSAGE,
    recent_feedback_text,
    recent_move_list,
    render_recent_stroke_map,
    render_studio_guide,
)
from .nanogpt import (
    NanoGPTError,
    build_multimodal_user_message,
    chat_completion,
)
from .parser import parse_planner_output
from .planner import (
    CANVAS_MESSAGE,
    CAUSAL_FEEDBACK_PROMPT_TEMPLATE_VERSION,
    PROMPT_TEMPLATE_VERSION,
    REFLECTIVE_PROMPT_TEMPLATE_VERSION,
    STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
    TITLE_PROMPT_TEMPLATE_VERSION,
    WORKING_MEMORY_PROMPT_TEMPLATE_VERSION,
    SCHEMA_FIRST_PROMPT_TEMPLATE_VERSION,
    build_history_summary,
    deterministic_demo_plan,
    prompt_for_model,
    prompt_for_model_v2,
    prompt_for_model_v3,
    prompt_for_model_v4,
    prompt_for_model_v5,
    prompt_for_model_v6,
    prompt_for_model_v7,
)
from .validation import ValidationError, validate_strokes


MAX_ROUNDS = 200
MAX_STROKES_PER_ROUND = 20
MAX_STROKE_BUDGET = 1000
MIN_CANVAS_SIZE = 128
MAX_CANVAS_SIZE = 2048


class PainterRuntimeError(RuntimeError):
    pass


class RoundRecord(TypedDict, total=False):
    round: int
    initial_idea: str
    observation: str
    interpretation: str
    done: bool
    title: str
    working_state: dict[str, Any]
    final_reflection: str
    strokes_count: int
    before_hash: str
    after_hash: str


def _write_jsonl(file_path: Path, entry: dict[str, Any]) -> None:
    with file_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True))
        fh.write("\n")


class PainterRunner:
    """Run the deterministic closed-loop painter."""

    def __init__(
        self,
        prompt: str,
        *,
        output_dir: str = "artifacts",
        model: str | None = None,
        reference_image: str | None = None,
        canvas_size: int = 512,
        max_rounds: int = 4,
        max_strokes: int = 5,
        stroke_budget: int | None = None,
        seed: int | None = None,
        yes: bool = False,
        dry_run: bool = True,
        prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise PainterRuntimeError("prompt must not be empty")
        if len(prompt) > 4_000:
            raise PainterRuntimeError("prompt must be at most 4000 characters")
        if not (1 <= max_rounds <= MAX_ROUNDS):
            raise PainterRuntimeError(f"max_rounds must be in [1, {MAX_ROUNDS}]")
        if not (1 <= max_strokes <= MAX_STROKES_PER_ROUND):
            raise PainterRuntimeError(
                f"max_strokes must be in [1, {MAX_STROKES_PER_ROUND}]"
            )
        if stroke_budget is not None:
            if not (1 <= stroke_budget <= MAX_STROKE_BUDGET):
                raise PainterRuntimeError(
                    f"stroke_budget must be in [1, {MAX_STROKE_BUDGET}]"
                )
            painting_rounds = (
                max_rounds - 1
                if prompt_template_version in {
                    REFLECTIVE_PROMPT_TEMPLATE_VERSION,
                    STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
                    TITLE_PROMPT_TEMPLATE_VERSION,
                    PROMPT_TEMPLATE_VERSION,
                }
                else max_rounds
            )
            if stroke_budget > painting_rounds * max_strokes:
                raise PainterRuntimeError(
                    "stroke_budget exceeds available painting-round capacity"
                )
        if not (MIN_CANVAS_SIZE <= canvas_size <= MAX_CANVAS_SIZE):
            raise PainterRuntimeError(
                f"canvas_size must be in [{MIN_CANVAS_SIZE}, {MAX_CANVAS_SIZE}]"
            )
        if prompt_template_version not in {
            SCHEMA_FIRST_PROMPT_TEMPLATE_VERSION,
            CAUSAL_FEEDBACK_PROMPT_TEMPLATE_VERSION,
            REFLECTIVE_PROMPT_TEMPLATE_VERSION,
            STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
            TITLE_PROMPT_TEMPLATE_VERSION,
            WORKING_MEMORY_PROMPT_TEMPLATE_VERSION,
            PROMPT_TEMPLATE_VERSION,
        }:
            raise PainterRuntimeError("unsupported live prompt template version")

        self.prompt = prompt.strip()
        self.output_dir = Path(output_dir)
        self.model = model
        self.reference_image = Path(reference_image).expanduser() if reference_image else None
        if self.reference_image is not None and not self.reference_image.is_file():
            raise PainterRuntimeError("reference_image must point to an existing file")
        self.canvas_size = canvas_size
        self.max_rounds = max_rounds
        self.max_strokes = max_strokes
        self.stroke_budget = stroke_budget
        self.yes = yes
        self.dry_run = dry_run
        self.prompt_template_version = prompt_template_version
        self.seed = seed if seed is not None else random.SystemRandom().randrange(0, 2**31)

    def _prepare_run(self) -> tuple[Path, Path, Path, Path, Path]:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"run-{ts}-{self.seed}"
        run_dir = self.output_dir / run_id
        frames_dir = run_dir / "frames"
        run_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        if self.reference_image is not None:
            context_dir = run_dir / "context"
            context_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.reference_image, context_dir / "reference.png")
        log_path = run_dir / "process.jsonl"
        final_png = run_dir / "final.png"
        return run_dir, frames_dir, manifest_path, log_path, final_png

    def _confirm_call(self, round_index: int) -> bool:
        if self.yes:
            return True
        answer = input(f"Round {round_index}: call NanoGPT? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def _recent_feedback_for_round(
        self,
        round_index: int,
        last_frame: Path | None,
    ) -> tuple[str, Path] | None:
        if (
            self.prompt_template_version not in {
                CAUSAL_FEEDBACK_PROMPT_TEMPLATE_VERSION,
                REFLECTIVE_PROMPT_TEMPLATE_VERSION,
                STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
                TITLE_PROMPT_TEMPLATE_VERSION,
                WORKING_MEMORY_PROMPT_TEMPLATE_VERSION,
                PROMPT_TEMPLATE_VERSION,
            }
            or round_index <= 1
            or last_frame is None
        ):
            return None

        run_dir = last_frame.parent.parent
        process_path = run_dir / "process.jsonl"
        try:
            lines = [
                line
                for line in process_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            event = json.loads(lines[-1])
            if int(event["round"]) != round_index - 1:
                raise ValueError("last event does not precede requested round")
            recent_strokes = validate_strokes(
                event["strokes"],
                max_strokes=self.max_strokes,
            )
            round_seed = int(event["round_seed"])
            expected_hash = str(event["canvas_after_hash"])
        except (OSError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise PainterRuntimeError(f"cannot build recent-stroke feedback: {exc}") from exc

        try:
            with Image.open(last_frame) as image:
                actual_hash = image_sha256(image.convert("RGBA"))
        except OSError as exc:
            raise PainterRuntimeError(f"cannot read current canvas for feedback: {exc}") from exc
        if actual_hash != expected_hash:
            raise PainterRuntimeError("recent-stroke feedback canvas hash mismatch")

        moves = recent_move_list(recent_strokes)
        map_path = run_dir / "context" / f"recent_strokes_round_{round_index:03d}.png"
        render_recent_stroke_map(
            last_frame,
            recent_strokes,
            map_path,
            round_seed=round_seed,
            run_seed=self.seed,
        )
        return recent_feedback_text(moves), map_path

    def _studio_guide_for_round(self, round_index: int, last_frame: Path | None) -> Path | None:
        if self.prompt_template_version not in {
            STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
            TITLE_PROMPT_TEMPLATE_VERSION,
            WORKING_MEMORY_PROMPT_TEMPLATE_VERSION,
            PROMPT_TEMPLATE_VERSION,
        } or last_frame is None:
            return None
        guide_path = last_frame.parent.parent / "context" / f"studio_guide_round_{round_index:03d}.png"
        return render_studio_guide(last_frame, guide_path)

    def _planner_from_model(
        self,
        round_index: int,
        last_frame: Path | None,
        history: list[RoundRecord],
        round_limit: int,
    ) -> dict[str, Any]:
        if self.model is None:
            raise PainterRuntimeError("model mode requested without model")
        if not self._confirm_call(round_index):
            raise PainterRuntimeError("user declined subscription call")

        summary = build_history_summary(
            history,
            include_working_state=self.prompt_template_version == PROMPT_TEMPLATE_VERSION,
        )
        strokes_used = sum(entry["strokes_count"] for entry in history)
        prompt_builder = {
            SCHEMA_FIRST_PROMPT_TEMPLATE_VERSION: prompt_for_model_v2,
            CAUSAL_FEEDBACK_PROMPT_TEMPLATE_VERSION: prompt_for_model_v3,
            REFLECTIVE_PROMPT_TEMPLATE_VERSION: prompt_for_model_v4,
            STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION: prompt_for_model_v5,
            TITLE_PROMPT_TEMPLATE_VERSION: prompt_for_model_v6,
            WORKING_MEMORY_PROMPT_TEMPLATE_VERSION: prompt_for_model_v7,
            PROMPT_TEMPLATE_VERSION: prompt_for_model,
        }[self.prompt_template_version]
        reflective = self.prompt_template_version in {
            REFLECTIVE_PROMPT_TEMPLATE_VERSION,
            STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
            TITLE_PROMPT_TEMPLATE_VERSION,
            WORKING_MEMORY_PROMPT_TEMPLATE_VERSION,
            PROMPT_TEMPLATE_VERSION,
        }
        messages = prompt_builder(
            prompt=self.prompt,
            round_index=round_index,
            summary=summary,
            max_strokes=round_limit,
            max_rounds=self.max_rounds,
            strokes_used=strokes_used,
            stroke_budget=self.stroke_budget,
            exact_strokes=(
                round_limit
                if self.stroke_budget is not None and not reflective
                else None
            ),
        )
        if last_frame:
            messages.append(
                build_multimodal_user_message(CANVAS_MESSAGE, str(last_frame))
            )
        if self.prompt_template_version == PROMPT_TEMPLATE_VERSION and last_frame:
            reference_path = last_frame.parent.parent / "context" / "reference.png"
            if reference_path.exists():
                messages.append(
                    build_multimodal_user_message(REFERENCE_IMAGE_MESSAGE, str(reference_path))
                )
        studio_guide = self._studio_guide_for_round(round_index, last_frame)
        if studio_guide is not None:
            messages.append(
                build_multimodal_user_message(STUDIO_GUIDE_MESSAGE, str(studio_guide))
            )
        recent_feedback = self._recent_feedback_for_round(round_index, last_frame)
        if recent_feedback is not None:
            feedback_text, feedback_map = recent_feedback
            messages.append(
                build_multimodal_user_message(feedback_text, str(feedback_map))
            )

        raw = chat_completion(
            messages,
            model=self.model,
            min_strokes=(
                0
                if reflective
                else (round_limit if self.stroke_budget is not None else 1)
            ),
            max_strokes=round_limit,
            timeout=180,
            reflective=reflective,
            include_title=self.prompt_template_version == PROMPT_TEMPLATE_VERSION,
        )
        try:
            return parse_planner_output(
                raw,
                max_strokes=round_limit,
                reflective=reflective,
                require_title=self.prompt_template_version == PROMPT_TEMPLATE_VERSION,
                require_working_state=self.prompt_template_version == PROMPT_TEMPLATE_VERSION,
            )
        except ValidationError as exc:
            raise PainterRuntimeError(f"planner output invalid: {exc}") from exc

    def _planner_from_demo(
        self,
        round_index: int,
        last_frame: Path | None,
        history: list[RoundRecord],
        round_limit: int,
    ) -> dict[str, Any]:
        del last_frame
        reflective = self.prompt_template_version in {
            REFLECTIVE_PROMPT_TEMPLATE_VERSION,
            STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
            TITLE_PROMPT_TEMPLATE_VERSION,
            WORKING_MEMORY_PROMPT_TEMPLATE_VERSION,
            PROMPT_TEMPLATE_VERSION,
        }
        terminal_round = reflective and (
            round_limit == 0 or round_index == self.max_rounds
        )
        raw = deterministic_demo_plan(
            self.prompt,
            round_index,
            round_limit,
            self.max_rounds,
            self.seed,
            exact_strokes=(
                0
                if terminal_round
                else (
                    round_limit
                    if self.stroke_budget is not None and not reflective
                    else None
                )
            ),
        )
        if reflective:
            raw.update(
                {
                    "initial_idea": f"Develop a clear visual response to: {self.prompt}",
                    "done": terminal_round,
                    "title": "Deterministic gestural study" if terminal_round else "",
                    "working_state": {
                        "current_goal": f"Develop a clear visual response to: {self.prompt}",
                        "completed_parts": [],
                        "last_change": "",
                        "next_action": "Continue the study." if not terminal_round else "",
                        "alignment": "aligned" if terminal_round else "uncertain",
                    },
                    "final_reflection": (
                        "The finished demo preserves its deterministic gestural character."
                        if terminal_round
                        else ""
                    ),
                }
            )
        elif self.stroke_budget is not None:
            used = sum(entry["strokes_count"] for entry in history)
            raw["done"] = used + round_limit >= self.stroke_budget
        return parse_planner_output(
            json.dumps(raw),
            max_strokes=round_limit,
            reflective=reflective,
            require_title=self.prompt_template_version == PROMPT_TEMPLATE_VERSION,
            require_working_state=self.prompt_template_version == PROMPT_TEMPLATE_VERSION,
        )

    def _run_loop(
        self,
        planner: Callable[[int, Path | None, list[RoundRecord], int], dict[str, Any]],
        mode: str,
        *,
        resume_run_dir: Path | None = None,
    ) -> dict[str, Any]:
        if resume_run_dir is None:
            run_dir, frames_dir, manifest_path, log_path, final_png = self._prepare_run()
            canvas = Image.new(
                "RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 255)
            )
            history: list[RoundRecord] = []
            initial_frame = frames_dir / "frame_000.png"
            canvas.save(initial_frame)
            frame_relpaths: list[str] = [str(initial_frame.relative_to(run_dir))]
            started_at = datetime.now(timezone.utc)
            last_frame: Path | None = initial_frame
            executed_rounds = 0
            total_strokes = 0
        else:
            run_dir = resume_run_dir.resolve()
            frames_dir = run_dir / "frames"
            manifest_path = run_dir / "manifest.json"
            log_path = run_dir / "process.jsonl"
            final_png = run_dir / "final.png"
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                events = [
                    json.loads(line)
                    for line in log_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                started_at = datetime.fromisoformat(str(existing["created_at"]))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise PainterRuntimeError(f"cannot resume invalid run: {exc}") from exc

            expected_settings = {
                "mode": mode,
                "prompt": self.prompt,
                "model": self.model,
                "reference_image": self.reference_image is not None,
                "canvas_size": self.canvas_size,
                "seed": self.seed,
                "max_rounds": self.max_rounds,
                "max_strokes": self.max_strokes,
                "stroke_budget": self.stroke_budget,
                "prompt_template_version": self.prompt_template_version,
            }
            actual_settings = {
                "mode": existing.get("mode"),
                "prompt": existing.get("prompt"),
                "model": existing.get("model"),
                "reference_image": bool(existing.get("reference_image")),
                "canvas_size": existing.get("canvas", {}).get("size"),
                "seed": existing.get("seed"),
                "max_rounds": existing.get("max_rounds"),
                "max_strokes": existing.get("max_strokes"),
                "stroke_budget": existing.get("stroke_budget"),
                "prompt_template_version": existing.get("prompt_template_version"),
            }
            if actual_settings != expected_settings:
                raise PainterRuntimeError("resume settings do not match the persisted run")
            if existing.get("status") == "done":
                raise PainterRuntimeError("cannot resume a completed run")
            if not events:
                raise PainterRuntimeError("cannot resume a run without completed events")
            rounds = [int(event.get("round", -1)) for event in events]
            if rounds != list(range(1, len(events) + 1)):
                raise PainterRuntimeError("cannot resume a non-contiguous process log")

            replay_path = replay_run(run_dir)
            canvas = Image.open(replay_path).convert("RGBA")
            history = [
                {
                    "round": int(event["round"]),
                    "initial_idea": str(event.get("initial_idea", "")),
                    "observation": str(event["observation"]),
                    "interpretation": str(event["interpretation"]),
                    "done": bool(event["done"]),
                    "title": str(event.get("title", "")),
                    "working_state": dict(event.get("working_state", {})),
                    "final_reflection": str(event.get("final_reflection", "")),
                    "strokes_count": int(event["strokes_count"]),
                    "before_hash": str(event["canvas_before_hash"]),
                    "after_hash": str(event["canvas_after_hash"]),
                }
                for event in events
            ]
            executed_rounds = rounds[-1]
            total_strokes = sum(record["strokes_count"] for record in history)
            frame_relpaths = [str(item) for item in existing.get("frames", [])]
            if len(frame_relpaths) != executed_rounds + 1:
                raise PainterRuntimeError("cannot resume a run with an invalid frame index")
            last_frame = run_dir / str(events[-1]["frame"])
            try:
                last_frame_hash = image_sha256(Image.open(last_frame).convert("RGBA"))
            except OSError as exc:
                raise PainterRuntimeError(f"cannot resume without the last frame: {exc}") from exc
            if last_frame_hash != history[-1]["after_hash"]:
                raise PainterRuntimeError("cannot resume: last frame hash mismatch")

        status = "interrupted"
        stop_reason = None
        reflective = self.prompt_template_version in {
            REFLECTIVE_PROMPT_TEMPLATE_VERSION,
            STUDIO_GUIDE_PROMPT_TEMPLATE_VERSION,
            TITLE_PROMPT_TEMPLATE_VERSION,
            WORKING_MEMORY_PROMPT_TEMPLATE_VERSION,
            PROMPT_TEMPLATE_VERSION,
        }

        for round_index in range(executed_rounds + 1, self.max_rounds + 1):
            round_seed = _blend_seed(self.seed, round_index)
            remaining = (
                self.stroke_budget - total_strokes
                if self.stroke_budget is not None
                else self.max_strokes
            )
            terminal_round = reflective and (
                remaining <= 0 or round_index == self.max_rounds
            )
            round_limit = 0 if terminal_round else min(self.max_strokes, remaining)
            if not reflective and round_limit <= 0:
                status = "done"
                stop_reason = "stroke_budget_exhausted"
                break

            try:
                parsed = planner(round_index, last_frame, history, round_limit)
            except PainterRuntimeError as exc:
                status = "stopped"
                stop_reason = str(exc)
                break
            except NanoGPTError as exc:
                status = "failed"
                stop_reason = str(exc)
                break

            if not parsed["done"] and not parsed["strokes"]:
                status = "failed"
                stop_reason = "planner returned zero strokes before done"
                break
            if reflective and parsed["done"] and parsed["strokes"]:
                status = "failed"
                stop_reason = "planner must stop with zero strokes so reflection follows drawing"
                break
            if terminal_round and not parsed["done"]:
                status = "failed"
                stop_reason = "planner did not stop on the reserved reflection round"
                break
            if not reflective and not parsed["strokes"]:
                status = "failed"
                stop_reason = "planner returned zero strokes"
                break
            if (
                not reflective
                and self.stroke_budget is not None
                and len(parsed["strokes"]) != round_limit
            ):
                status = "failed"
                stop_reason = f"planner returned {len(parsed['strokes'])} strokes; expected {round_limit}"
                break

            for idx, stroke in enumerate(parsed["strokes"]):
                if stroke["seed"] is None:
                    stroke["seed"] = _blend_seed(round_seed, idx)

            before_hash = image_sha256(canvas)
            if parsed["strokes"]:
                render_strokes(
                    canvas,
                    strokes=parsed["strokes"],
                    seed=round_seed,
                    run_seed=self.seed,
                )
            after_hash = image_sha256(canvas)

            frame_path = frames_dir / f"frame_{round_index:03d}.png"
            canvas.save(frame_path)
            last_frame = frame_path
            frame_relpaths.append(str(frame_path.relative_to(run_dir)))

            total_strokes += len(parsed["strokes"])
            budget_exhausted = (
                self.stroke_budget is not None and total_strokes >= self.stroke_budget
            )
            effective_done = (
                bool(parsed["done"])
                if reflective
                else budget_exhausted or (
                    self.stroke_budget is None and bool(parsed["done"])
                )
            )
            history_entry: RoundRecord = {
                "round": round_index,
                "observation": parsed["observation"],
                "interpretation": parsed["interpretation"],
                "done": effective_done,
                "strokes_count": len(parsed["strokes"]),
                "before_hash": before_hash,
                "after_hash": after_hash,
            }
            if reflective:
                history_entry["initial_idea"] = parsed["initial_idea"]
                history_entry["title"] = parsed["title"]
                history_entry["working_state"] = parsed.get("working_state", {})
                history_entry["final_reflection"] = parsed["final_reflection"]
            history.append(history_entry)
            executed_rounds = round_index
            event = {
                "round": round_index,
                "run_seed": self.seed,
                "round_seed": round_seed,
                "prompt": self.prompt,
                "done": effective_done,
                "model_done": parsed["done"],
                "total_strokes": total_strokes,
                "stroke_budget": self.stroke_budget,
                "observation": parsed["observation"],
                "interpretation": parsed["interpretation"],
                "strokes": parsed["strokes"],
                "strokes_count": len(parsed["strokes"]),
                "canvas_before_hash": before_hash,
                "canvas_after_hash": after_hash,
                "frame": str(frame_path.relative_to(run_dir)),
            }
            if reflective:
                event["initial_idea"] = parsed["initial_idea"]
                event["title"] = parsed["title"]
                event["working_state"] = parsed.get("working_state", {})
                event["final_reflection"] = parsed["final_reflection"]
            _write_jsonl(log_path, event)

            if reflective and parsed["done"]:
                status = "done"
                stop_reason = "model_declared_done"
                break
            if not reflective and budget_exhausted:
                status = "done"
                stop_reason = "stroke_budget_exhausted"
                break
            if not reflective and self.stroke_budget is None and parsed["done"]:
                status = "done"
                stop_reason = "model_declared_done"
                break

        if status == "interrupted":
            status = "stopped"
            stop_reason = "max_rounds_reached"

        if last_frame is not None:
            final_png.write_bytes(last_frame.read_bytes())
        else:
            canvas.save(final_png)

        manifest = {
            "schema_version": "1.0",
            "prompt_template_version": self.prompt_template_version,
            "run_id": run_dir.name,
            "created_at": started_at.isoformat(),
            "mode": mode,
            "prompt": self.prompt,
            "model": self.model,
            "canvas": {
                "size": self.canvas_size,
                "color_mode": "RGBA",
            },
            "seed": self.seed,
            "max_rounds": self.max_rounds,
            "max_strokes": self.max_strokes,
            "stroke_budget": self.stroke_budget,
            "total_strokes": total_strokes,
            "status": status,
            "stop_reason": stop_reason,
            "rounds_executed": executed_rounds,
            "manifest_path": str(manifest_path),
            "process_log": str(log_path),
            "final_image": str(final_png),
            "final_canvas_hash": image_sha256(canvas),
            "round_records": history,
            "initial_idea": next(
                (entry.get("initial_idea", "") for entry in history if entry.get("initial_idea")),
                "",
            ),
            "title": next(
                (
                    entry.get("title", "")
                    for entry in reversed(history)
                    if entry.get("title")
                ),
                "",
            ),
            "working_state": next(
                (
                    entry.get("working_state", {})
                    for entry in reversed(history)
                    if entry.get("working_state")
                ),
                {},
            ),
            "final_reflection": next(
                (
                    entry.get("final_reflection", "")
                    for entry in reversed(history)
                    if entry.get("final_reflection")
                ),
                "",
            ),
            "frames": frame_relpaths,
            "dry_run_mode": self.dry_run,
            "reference_image": (
                "context/reference.png"
                if (run_dir / "context" / "reference.png").exists()
                else None
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def run_demo(self) -> dict[str, Any]:
        return self._run_loop(self._planner_from_demo, mode="demo")

    def run_paint(self) -> dict[str, Any]:
        if not self.model:
            raise PainterRuntimeError("model required for paint mode")
        return self._run_loop(self._planner_from_model, mode="paint")

    def resume_paint(self, run_dir: str | Path) -> dict[str, Any]:
        if not self.model:
            raise PainterRuntimeError("model required for paint mode")
        return self._run_loop(
            self._planner_from_model,
            mode="paint",
            resume_run_dir=Path(run_dir),
        )


def resume_paint_run(run_dir: str | Path, *, yes: bool = False) -> dict[str, Any]:
    """Resume a failed or stopped paint run from its last verified event."""
    root = Path(run_dir).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        model = str(manifest["model"])
        if not model or model == "None":
            raise ValueError("missing model")
        runner = PainterRunner(
            prompt=str(manifest["prompt"]),
            output_dir=str(root.parent),
            model=model,
            reference_image=(
                str(root / str(manifest["reference_image"]))
                if manifest.get("reference_image")
                else None
            ),
            canvas_size=int(manifest["canvas"]["size"]),
            max_rounds=int(manifest["max_rounds"]),
            max_strokes=int(manifest["max_strokes"]),
            stroke_budget=(
                int(manifest["stroke_budget"])
                if manifest.get("stroke_budget") is not None
                else None
            ),
            seed=int(manifest["seed"]),
            yes=yes,
            dry_run=False,
            prompt_template_version=str(
                manifest.get(
                    "prompt_template_version",
                    SCHEMA_FIRST_PROMPT_TEMPLATE_VERSION,
                )
            ),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PainterRuntimeError(f"cannot resume invalid run: {exc}") from exc
    return runner.resume_paint(root)


def replay_run(run_dir: str | Path) -> Path:
    """Rebuild a final canvas using only the run's validated stroke events."""
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    log_path = root / "process.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        size = int(manifest["canvas"]["size"])
        run_seed = int(manifest["seed"])
        max_strokes = int(manifest["max_strokes"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PainterRuntimeError(f"invalid run manifest: {exc}") from exc

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PainterRuntimeError(f"cannot read process log: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            before_hash = str(event["canvas_before_hash"])
            after_hash = str(event["canvas_after_hash"])
            round_seed = int(event["round_seed"])
            strokes = validate_strokes(
                event["strokes"],
                max_strokes=max_strokes,
                allow_empty=bool(event.get("done")),
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, ValidationError) as exc:
            raise PainterRuntimeError(f"invalid process event on line {line_number}: {exc}") from exc

        if image_sha256(canvas) != before_hash:
            raise PainterRuntimeError(f"provenance hash mismatch before event {line_number}")
        render_strokes(canvas, strokes=strokes, seed=round_seed, run_seed=run_seed)
        if image_sha256(canvas) != after_hash:
            raise PainterRuntimeError(f"provenance hash mismatch after event {line_number}")

    final_hash = image_sha256(canvas)
    if final_hash != manifest.get("final_canvas_hash"):
        raise PainterRuntimeError("replayed canvas does not match the manifest final hash")
    output_path = root / "replay.png"
    canvas.save(output_path)
    return output_path


__all__ = [
    "PainterRunner",
    "PainterRuntimeError",
    "RoundRecord",
    "resume_paint_run",
    "MAX_ROUNDS",
    "MAX_STROKES_PER_ROUND",
    "MAX_STROKE_BUDGET",
    "MIN_CANVAS_SIZE",
    "MAX_CANVAS_SIZE",
    "replay_run",
]
