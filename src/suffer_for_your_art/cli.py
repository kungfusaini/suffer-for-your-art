"""CLI entry points for the painting MVP."""

from __future__ import annotations

import argparse
import json
import math
import sys

from .core import (
    MAX_ROUNDS,
    MAX_STROKE_BUDGET,
    MAX_STROKES_PER_ROUND,
    PainterRuntimeError,
    PainterRunner,
    replay_run,
    resume_paint_run,
)
from .gallery import GalleryError, run_server
from .nanogpt import NanoGPTError, list_models


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="suffer-for-your-art",
        description="Pure-digital closed-loop painter MVP",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--prompt", required=True, help="Text prompt describing what to paint")
    common.add_argument(
        "--output-dir", default="artifacts", help="Output directory (default: artifacts)"
    )
    common.add_argument(
        "--canvas-size", type=int, default=512, help="Canvas size in pixels (square)"
    )
    common.add_argument(
        "--max-rounds",
        type=int,
        help=(
            f"Maximum feedback rounds (1-{MAX_ROUNDS}; default: 4, or enough "
            "rounds to spend --stroke-budget)"
        ),
    )
    common.add_argument(
        "--max-strokes",
        type=int,
        default=5,
        help=(
            f"Maximum strokes per round (1-{MAX_STROKES_PER_ROUND}; "
            f"{MAX_STROKES_PER_ROUND} is the structured-output hard cap)"
        ),
    )
    common.add_argument(
        "--stroke-budget",
        type=int,
        help=(
            f"Allow up to this many total strokes across feedback rounds "
            f"(1-{MAX_STROKE_BUDGET})"
        ),
    )
    common.add_argument("--seed", type=int, help="Optional deterministic seed")

    demo_parser = subparsers.add_parser(
        "demo", help="Run offline deterministic planner only", parents=[common]
    )
    demo_parser.set_defaults(func=_run_demo)

    paint_parser = subparsers.add_parser(
        "paint", help="Run with NanoGPT subscription planning", parents=[common]
    )
    paint_parser.add_argument(
        "--model", required=True, help="Plain model id returned by the models command"
    )
    paint_parser.add_argument(
        "--reference-image",
        help="Optional reference image shown to the artist as non-painting context",
    )
    paint_parser.add_argument(
        "--yes", action="store_true", help="Skip interactive confirmation for API calls"
    )
    paint_parser.set_defaults(func=_run_paint)

    resume_parser = subparsers.add_parser(
        "resume", help="Resume a failed paint run from its last verified event"
    )
    resume_parser.add_argument(
        "run_dir", help="Failed run directory containing manifest.json and process.jsonl"
    )
    resume_parser.add_argument(
        "--yes", action="store_true", help="Skip interactive confirmation for remaining API calls"
    )
    resume_parser.set_defaults(func=_run_resume)

    models_parser = subparsers.add_parser(
        "models", help="List available NanoGPT subscription models"
    )
    models_parser.set_defaults(func=_run_models)

    replay_parser = subparsers.add_parser(
        "replay", help="Rebuild a run from its stroke event log"
    )
    replay_parser.add_argument(
        "run_dir", help="Run directory containing manifest.json and process.jsonl"
    )
    replay_parser.set_defaults(func=_run_replay)

    gallery_parser = subparsers.add_parser(
        "gallery", help="Run local gallery web player"
    )
    gallery_parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    gallery_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000)",
    )
    gallery_parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Artifacts directory to browse (default: artifacts)",
    )
    gallery_parser.set_defaults(func=_run_gallery)

    return parser


def _resolved_rounds(args: argparse.Namespace) -> int:
    if args.max_rounds is not None:
        return args.max_rounds
    if args.stroke_budget is not None and args.max_strokes > 0:
        return math.ceil(args.stroke_budget / args.max_strokes) + 1
    return 4


def _run_models(_args: argparse.Namespace) -> int:
    try:
        models = list_models()
    except NanoGPTError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(models))
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    try:
        output = replay_run(args.run_dir)
    except PainterRuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


def _run_resume(args: argparse.Namespace) -> int:
    print(
        "Warning: resuming may call NanoGPT once per remaining round. "
        "Persisted events will be verified before any new call.",
        file=sys.stderr,
    )
    try:
        manifest = resume_paint_run(args.run_dir, yes=args.yes)
    except (NanoGPTError, PainterRuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_manifest_summary(manifest), indent=2))
    return 1 if manifest["status"] == "failed" else 0


def _run_gallery(args: argparse.Namespace) -> int:
    try:
        return run_server(host=args.host, port=args.port, artifacts_dir=args.artifacts_dir)
    except GalleryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _runner_from_args(args: argparse.Namespace, *, model: str | None, yes: bool) -> PainterRunner:
    return PainterRunner(
        prompt=args.prompt,
        output_dir=args.output_dir,
        model=model,
        reference_image=getattr(args, "reference_image", None),
        canvas_size=args.canvas_size,
        max_rounds=_resolved_rounds(args),
        max_strokes=args.max_strokes,
        stroke_budget=args.stroke_budget,
        seed=args.seed,
        yes=yes,
        dry_run=model is None,
    )


def _run_demo(args: argparse.Namespace) -> int:
    try:
        manifest = _runner_from_args(args, model=None, yes=True).run_demo()
    except PainterRuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(_manifest_summary(manifest), indent=2))
    return 0


def _run_paint(args: argparse.Namespace) -> int:
    max_calls = _resolved_rounds(args)
    if max_calls > 1:
        print(
            f"Warning: this paint run may call NanoGPT up to {max_calls} times "
            "(1 per round). Use --yes to auto-confirm. "
            "The stroke budget is a ceiling and the painter may stop early.",
            file=sys.stderr,
        )

    try:
        manifest = _runner_from_args(args, model=args.model, yes=args.yes).run_paint()
    except (NanoGPTError, PainterRuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(_manifest_summary(manifest), indent=2))
    return 1 if manifest["status"] == "failed" else 0


def _manifest_summary(manifest: dict) -> dict:
    return {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "mode": manifest["mode"],
        "rounds_executed": manifest["rounds_executed"],
        "total_strokes": manifest["total_strokes"],
        "stroke_budget": manifest["stroke_budget"],
        "stop_reason": manifest["stop_reason"],
        "manifest": manifest["manifest_path"],
        "process_log": manifest["process_log"],
        "final_image": manifest["final_image"],
        "final_frame_count": len(manifest["round_records"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def run_as_script() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run_as_script()


__all__ = ["main"]
