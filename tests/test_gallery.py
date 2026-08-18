from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suffer_for_your_art.brush import image_sha256, render_strokes
from suffer_for_your_art.gallery import (
    GalleryNotFoundError,
    GalleryPayloadError,
    _redact_request_messages,
    discover_runs,
    get_cumulative_frame,
    get_run_context,
    reconstruct_round_request_payload,
    run_details,
    validate_run_id,
)
from suffer_for_your_art.nanogpt import serialize_chat_payload


def _build_event(strokes: list[dict[str, Any]], round_seed: int, total_strokes: int, *, done: bool) -> dict[str, Any]:
    return {
        "round": 1,
        "done": done,
        "strokes_count": len(strokes),
        "total_strokes": total_strokes,
        "run_seed": 111,
        "round_seed": round_seed,
        "observation": f"Observation after {total_strokes} strokes",
        "interpretation": f"Interpretation after {total_strokes} strokes",
        "strokes": strokes,
    }


def _write_demo_run(root: Path, *, run_id: str, created_at: str, total_strokes: int = 2) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)

    run_seed = 111
    size = 64

    stroke_a = {
        "color": "carbon",
        "width": 5,
        "pressure": 0.78,
        "opacity": 0.88,
        "bezier": {
            "start": [0.1, 0.1],
            "control1": [0.2, 0.2],
            "control2": [0.3, 0.3],
            "end": [0.4, 0.4],
        },
        "taper": {"start": 0.45, "end": 0.85},
        "seed": 1001,
    }

    stroke_b = {
        "color": "azure",
        "width": 4,
        "pressure": 0.55,
        "opacity": 0.9,
        "bezier": {
            "start": [0.7, 0.7],
            "control1": [0.76, 0.76],
            "control2": [0.84, 0.84],
            "end": [0.95, 0.95],
        },
        "taper": {"start": 0.42, "end": 0.84},
        "seed": 1002,
    }

    events = [
        _build_event([stroke_a], round_seed=1111, total_strokes=1, done=False),
        _build_event([stroke_b], round_seed=1112, total_strokes=2, done=True),
    ]

    for i, event in enumerate(events, start=1):
        event["round"] = i

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    for event in events:
        render_strokes(canvas, strokes=event["strokes"], seed=run_seed, run_seed=run_seed)

    final_hash = image_sha256(canvas)
    final_png = run_dir / "final.png"
    canvas.save(final_png)

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created_at,
        "mode": "demo",
        "prompt": "gallery smoke test",
        "model": None,
        "canvas": {"size": size, "color_mode": "RGBA"},
        "seed": run_seed,
        "max_rounds": 2,
        "max_strokes": 1,
        "stroke_budget": None,
        "total_strokes": total_strokes,
        "status": "done",
        "stop_reason": None,
        "rounds_executed": 2,
        "manifest_path": "manifest.json",
        "process_log": "process.jsonl",
        "final_image": "final.png",
        "final_canvas_hash": final_hash,
        "round_records": [],
        "frames": ["frames/frame_000.png"],
        "dry_run_mode": True,
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    lines = []
    for event in events:
        event["strokes_count"] = len(event["strokes"])
        lines.append(json.dumps(event, sort_keys=True))

    (run_dir / "process.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_run_for_context(root: Path, *, run_id: str, created_at: str) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()

    run_seed = 222
    size = 64

    stroke_a = {
        "color": "carbon",
        "width": 5,
        "pressure": 0.78,
        "opacity": 0.88,
        "bezier": {
            "start": [0.1, 0.1],
            "control1": [0.2, 0.2],
            "control2": [0.3, 0.3],
            "end": [0.4, 0.4],
        },
        "taper": {"start": 0.45, "end": 0.85},
        "seed": 1001,
    }

    events = [
        _build_event([stroke_a], round_seed=1111, total_strokes=1, done=False),
    ]

    event = events[0]
    event["round"] = 1
    event["frame"] = "frames/frame_000.png"

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    render_strokes(canvas, strokes=event["strokes"], seed=run_seed, run_seed=run_seed)

    final_hash = image_sha256(canvas)
    final_png = run_dir / "final.png"
    canvas.save(final_png)

    blank = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    blank.save(frames_dir / "frame_000.png", format="PNG")

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created_at,
        "mode": "paint",
        "prompt": "single round model reconstruction test",
        "model": "vendor/model-name",
        "canvas": {"size": size, "color_mode": "RGBA"},
        "seed": run_seed,
        "max_rounds": 2,
        "max_strokes": 3,
        "stroke_budget": None,
        "total_strokes": 1,
        "status": "done",
        "stop_reason": None,
        "rounds_executed": 1,
        "manifest_path": "manifest.json",
        "process_log": "process.jsonl",
        "final_image": "final.png",
        "final_canvas_hash": final_hash,
        "round_records": [],
        "frames": ["frames/frame_000.png"],
        "dry_run_mode": True,
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    lines = [
        json.dumps(event, sort_keys=True),
    ]

    (run_dir / "process.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_run_2_round_for_context(root: Path, *, run_id: str, created_at: str) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()

    run_seed = 333
    size = 64

    stroke_a = {
        "color": "carbon",
        "width": 5,
        "pressure": 0.78,
        "opacity": 0.88,
        "bezier": {
            "start": [0.1, 0.1],
            "control1": [0.2, 0.2],
            "control2": [0.3, 0.3],
            "end": [0.4, 0.4],
        },
        "taper": {"start": 0.45, "end": 0.85},
        "seed": 1001,
    }
    stroke_a2 = {
        "color": "ink",
        "width": 6,
        "pressure": 0.66,
        "opacity": 0.8,
        "bezier": {
            "start": [0.15, 0.22],
            "control1": [0.2, 0.24],
            "control2": [0.35, 0.4],
            "end": [0.5, 0.5],
        },
        "taper": {"start": 0.48, "end": 0.88},
        "seed": 1005,
    }
    stroke_b = {
        "color": "azure",
        "width": 4,
        "pressure": 0.55,
        "opacity": 0.9,
        "bezier": {
            "start": [0.7, 0.7],
            "control1": [0.76, 0.76],
            "control2": [0.84, 0.84],
            "end": [0.95, 0.95],
        },
        "taper": {"start": 0.42, "end": 0.84},
        "seed": 1002,
    }

    round1 = _build_event([stroke_a, stroke_a2], round_seed=1111, total_strokes=2, done=False)
    round1["round"] = 1
    round1["frame"] = "frames/frame_001.png"
    round1["model_done"] = False

    round2 = _build_event([stroke_b], round_seed=1112, total_strokes=3, done=True)
    round2["round"] = 2
    round2["frame"] = "frames/frame_002.png"
    round2["model_done"] = True

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    blank = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    render_strokes(canvas, strokes=round1["strokes"], seed=run_seed, run_seed=run_seed)
    blank.save(frames_dir / "frame_000.png", format="PNG")
    frame_001 = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    frame_001.paste(canvas)
    frame_001.save(frames_dir / "frame_001.png", format="PNG")
    render_strokes(canvas, strokes=round2["strokes"], seed=run_seed, run_seed=run_seed)
    canvas.save(frames_dir / "frame_002.png", format="PNG")
    final_png = run_dir / "final.png"
    canvas.save(final_png)

    final_hash = image_sha256(canvas)

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created_at,
        "mode": "paint",
        "prompt": "two-round model reconstruction test",
        "model": "vendor/model-name",
        "canvas": {"size": size, "color_mode": "RGBA"},
        "seed": run_seed,
        "max_rounds": 2,
        "max_strokes": 3,
        "stroke_budget": 5,
        "total_strokes": 3,
        "status": "done",
        "stop_reason": None,
        "rounds_executed": 2,
        "manifest_path": "manifest.json",
        "process_log": "process.jsonl",
        "final_image": "final.png",
        "final_canvas_hash": final_hash,
        "round_records": [],
        "frames": ["frames/frame_000.png", "frames/frame_001.png", "frames/frame_002.png"],
        "dry_run_mode": False,
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    lines = [
        json.dumps(round1, sort_keys=True),
        json.dumps(round2, sort_keys=True),
    ]

    (run_dir / "process.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_run_without_model(root: Path, *, run_id: str, created_at: str) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()

    run_seed = 444
    size = 64
    stroke_a = {
        "color": "carbon",
        "width": 5,
        "pressure": 0.78,
        "opacity": 0.88,
        "bezier": {
            "start": [0.1, 0.1],
            "control1": [0.2, 0.2],
            "control2": [0.3, 0.3],
            "end": [0.4, 0.4],
        },
        "taper": {"start": 0.45, "end": 0.85},
        "seed": 1001,
    }

    event = _build_event([stroke_a], round_seed=1111, total_strokes=1, done=True)
    event["round"] = 1
    event["frame"] = "frames/frame_001.png"

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    render_strokes(canvas, strokes=event["strokes"], seed=run_seed, run_seed=run_seed)
    frame_001 = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    frame_001.paste(canvas)
    frame_001.save(frames_dir / "frame_001.png", format="PNG")

    final_png = run_dir / "final.png"
    canvas.save(final_png)
    final_hash = image_sha256(canvas)

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created_at,
        "mode": "paint",
        "prompt": "paint run missing model",
        "model": None,
        "canvas": {"size": size, "color_mode": "RGBA"},
        "seed": run_seed,
        "max_rounds": 2,
        "max_strokes": 3,
        "stroke_budget": None,
        "total_strokes": 1,
        "status": "done",
        "stop_reason": None,
        "rounds_executed": 1,
        "manifest_path": "manifest.json",
        "process_log": "process.jsonl",
        "final_image": "final.png",
        "final_canvas_hash": final_hash,
        "round_records": [],
        "frames": ["frames/frame_001.png"],
        "dry_run_mode": False,
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    lines = [
        json.dumps(event, sort_keys=True),
    ]

    (run_dir / "process.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class GalleryValidationTests(unittest.TestCase):
    def test_validate_run_id(self):
        self.assertEqual(validate_run_id("run-2026-test"), "run-2026-test")
        with self.assertRaises(GalleryPayloadError):
            validate_run_id("../bad")
        with self.assertRaises(GalleryPayloadError):
            validate_run_id("")
        with self.assertRaises(GalleryPayloadError):
            validate_run_id("bad$name")

    def test_context_route_normal_redacts_image_data_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_model_run_for_context(root, run_id="run-model", created_at="2026-08-16T00:00:00")

            reconstructed = reconstruct_round_request_payload(root, "run-model", 1)
            payload = reconstructed["request_payload"]
            raw_body = serialize_chat_payload(payload).decode("utf-8")

            safe_request = dict(payload)
            safe_request["messages"] = _redact_request_messages(payload["messages"])
            normal_response = {
                "run_id": reconstructed["run_id"],
                "round": reconstructed["round"],
                "event_count": reconstructed["event_count"],
                "strokes_before": reconstructed["strokes_before"],
                "required_batch_size": reconstructed["required_batch_size"],
                "canvas_url": f"/api/runs/run-model/frame?strokes={reconstructed['strokes_before']}",
                "canvas_sha256": reconstructed["canvas_sha256"],
                "model": reconstructed["model"],
                "reconstruction": reconstructed["reconstruction"],
                "messages": _redact_request_messages(reconstructed["messages"]),
                "response_schema": reconstructed["response_schema"],
                "request": safe_request,
                "raw_request_sha256": hashlib.sha256(serialize_chat_payload(payload)).hexdigest(),
            }

            normal_body = json.dumps(normal_response, ensure_ascii=False)
            self.assertNotIn("data:image", normal_body)
            self.assertIn("data:image/png;base64", raw_body)

    def test_reconstruct_round2_reuses_round1_history_and_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "run-model-2-round"
            _write_model_run_2_round_for_context(root, run_id=run_id, created_at="2026-08-16T00:00:00")

            reconstructed = reconstruct_round_request_payload(root, run_id, 2)

            self.assertEqual(reconstructed["round"], 2)
            self.assertEqual(reconstructed["event_count"], 2)
            self.assertEqual(reconstructed["prompt_template_version"], "1.0")
            self.assertEqual(reconstructed["strokes_before"], 2)
            self.assertEqual(reconstructed["required_batch_size"], 3)
            self.assertEqual(reconstructed["canvas_path"], "frames/frame_001.png")

            frame_path = root / run_id / "frames" / "frame_001.png"
            with frame_path.open("rb") as handle:
                expected_hash = hashlib.sha256(handle.read()).hexdigest()
            self.assertEqual(reconstructed["canvas_sha256"], expected_hash)

            messages = reconstructed["messages"]
            system_message = next(
                (item for item in messages if item.get("label") == "system"),
                {},
            )
            self.assertIn("You are an autonomous painter", system_message.get("content", ""))
            context_message = next(
                (item for item in messages if item.get("label") == "context_action_contract"),
                {},
            )
            self.assertIn("Round 1:", context_message.get("content", ""))
            self.assertIn("observation='Observation after 2 strokes'", context_message.get("content", ""))
            self.assertIn("interpretation='Interpretation after 2 strokes'", context_message.get("content", ""))
            self.assertIn("original_total_strokes=5", context_message.get("content", ""))
            self.assertIn("committed_strokes=2", context_message.get("content", ""))
            self.assertIn("strokes_remaining=3", context_message.get("content", ""))
            self.assertIn("batch_size=3", context_message.get("content", ""))
            self.assertIn("Return exactly 3 strokes", context_message.get("content", ""))

            schema = reconstructed["response_schema"]["json_schema"]["schema"]["properties"]["strokes"]
            self.assertEqual(schema["minItems"], 3)
            self.assertEqual(schema["maxItems"], 3)

    def test_reconstruct_schema_first_prompt_has_no_system_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "run-model-schema-first"
            _write_model_run_2_round_for_context(root, run_id=run_id, created_at="2026-08-16T00:00:00")
            manifest_path = root / run_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["prompt_template_version"] = "2.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            reconstructed = reconstruct_round_request_payload(root, run_id, 2)

            self.assertEqual(reconstructed["prompt_template_version"], "2.0")
            self.assertNotIn("system", [item["label"] for item in reconstructed["messages"]])
            payload_messages = reconstructed["request_payload"]["messages"]
            self.assertEqual([item["role"] for item in payload_messages], ["user", "user"])
            self.assertEqual(payload_messages[1]["content"][0]["text"], "CURRENT CANVAS")
            context = reconstructed["messages"][0]["content"]
            self.assertIn("INITIAL INTENTION:", context)
            self.assertIn("ROUND STATE:", context)
            self.assertIn("required_batch_size=3", context)
            self.assertIn("PROCESS HISTORY:", context)
            self.assertNotIn("You are an autonomous painter", context)
            self.assertNotIn("ACTION CONTRACT", context)

    def test_reconstruct_template3_includes_numbered_recent_move_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "run-model-causal-feedback"
            _write_model_run_2_round_for_context(root, run_id=run_id, created_at="2026-08-16T00:00:00")
            manifest_path = root / run_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["prompt_template_version"] = "3.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            reconstructed = reconstruct_round_request_payload(root, run_id, 2)
            repeated = reconstruct_round_request_payload(root, run_id, 2)

            self.assertEqual(reconstructed["prompt_template_version"], "3.0")
            self.assertEqual(
                [item["label"] for item in reconstructed["messages"]],
                ["context_action_contract", "multimodal_canvas", "recent_stroke_map"],
            )
            self.assertEqual(len(reconstructed["recent_moves"]), 2)
            self.assertEqual(reconstructed["recent_moves"][0]["map_label"], 1)
            self.assertEqual(reconstructed["recent_moves"][1]["previous_batch_index"], 2)
            self.assertNotIn("seed", reconstructed["recent_moves"][0]["action"])
            map_path = root / run_id / reconstructed["recent_stroke_map_path"]
            self.assertTrue(map_path.exists())
            self.assertEqual(
                reconstructed["recent_stroke_map_sha256"],
                hashlib.sha256(map_path.read_bytes()).hexdigest(),
            )
            with Image.open(map_path) as diagnostic, Image.open(
                root / run_id / "frames" / "frame_001.png"
            ) as current_canvas:
                self.assertEqual(diagnostic.size, current_canvas.size)
            payload_messages = reconstructed["request_payload"]["messages"]
            self.assertEqual([item["role"] for item in payload_messages], ["user", "user", "user"])
            raw_body = serialize_chat_payload(reconstructed["request_payload"])
            self.assertEqual(raw_body.count(b"data:image/png;base64,"), 2)
            self.assertIn(b"RECENT ACTION-OUTPUT MAP", raw_body)
            self.assertEqual(
                raw_body,
                serialize_chat_payload(repeated["request_payload"]),
            )

    def test_reconstruct_round2_rejects_out_of_range_round(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "run-model-range"
            _write_model_run_2_round_for_context(root, run_id=run_id, created_at="2026-08-16T00:00:00")

            with self.assertRaises(GalleryPayloadError):
                reconstruct_round_request_payload(root, run_id, 0)
            with self.assertRaises(GalleryPayloadError):
                reconstruct_round_request_payload(root, run_id, 3)

    def test_reconstruct_round_rejects_paint_run_without_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "run-no-model"
            _write_model_run_without_model(root, run_id=run_id, created_at="2026-08-16T00:00:00")

            with self.assertRaises(GalleryPayloadError):
                reconstruct_round_request_payload(root, run_id, 1)

    def test_discover_runs_filters_and_orders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_run(root, run_id="run-new", created_at="2026-08-16T00:00:00")
            _write_demo_run(root, run_id="run-old", created_at="2026-08-15T00:00:00")

            invalid_run = root / "not-a-run"
            invalid_run.mkdir()
            (invalid_run / "manifest.json").write_text("{}", encoding="utf-8")

            # missing process
            no_process = root / "bad-run"
            no_process.mkdir()
            manifest = {"schema_version":"1.0"}
            (no_process / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (no_process / "final.png").write_bytes(b"notreal")

            # Legacy manifests omitted total_strokes; canonical event data still makes them valid.
            old_manifest_path = root / "run-old" / "manifest.json"
            old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
            old_manifest.pop("total_strokes")
            old_manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")

            runs = discover_runs(root)
            ids = [item["run_id"] for item in runs]
            self.assertEqual(ids, ["run-new", "run-old"])
            self.assertEqual(runs[1]["stroke_count"], 2)

    def test_discover_runs_empty_when_no_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(discover_runs(Path(tmpdir)), [])

    def test_run_details_exposes_final_reflection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_run(root, run_id="run-reflection", created_at="2026-08-16T00:00:00")

            detail = run_details(root, "run-reflection")
            self.assertEqual(detail["final_observation"], "Observation after 2 strokes")
            self.assertEqual(detail["artistic_rationale"], "Interpretation after 2 strokes")


class GalleryFrameTests(unittest.TestCase):
    def test_get_cumulative_frame_builds_each_stroke_from_scratch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_run(root, run_id="run-frame", created_at="2026-08-16T10:00:00")

            f0 = get_cumulative_frame(root, "run-frame", 0)
            f1 = get_cumulative_frame(root, "run-frame", 1)
            f2 = get_cumulative_frame(root, "run-frame", 2)

            i0 = Image.open(f0).convert("RGBA")
            i1 = Image.open(f1).convert("RGBA")
            i2 = Image.open(f2).convert("RGBA")

            self.assertNotEqual(image_sha256(i0), image_sha256(i1))
            self.assertNotEqual(image_sha256(i1), image_sha256(i2))

            expected1 = Image.new("RGBA", i0.size, (255, 255, 255, 255))
            run_seed = 111
            stroke1 = {
                "color": "carbon",
                "width": 5,
                "pressure": 0.78,
                "opacity": 0.88,
                "bezier": {
                    "start": [0.1, 0.1],
                    "control1": [0.2, 0.2],
                    "control2": [0.3, 0.3],
                    "end": [0.4, 0.4],
                },
                "taper": {"start": 0.45, "end": 0.85},
                "seed": 1001,
            }
            render_strokes(expected1, strokes=[stroke1], seed=run_seed, run_seed=run_seed)
            self.assertEqual(image_sha256(i1), image_sha256(expected1))
            final = Image.open(root / "run-frame" / "final.png").convert("RGBA")
            self.assertEqual(image_sha256(i2), image_sha256(final))

    def test_get_cumulative_frame_uses_cached_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_run(root, run_id="run-cache", created_at="2026-08-16T10:00:00")

            first = get_cumulative_frame(root, "run-cache", 2)
            mtime_before = os.stat(first).st_mtime
            second = get_cumulative_frame(root, "run-cache", 2)
            mtime_after = os.stat(second).st_mtime

            self.assertEqual(first, second)
            self.assertEqual(mtime_before, mtime_after)

            # A request below the only remaining cached frame must rebuild safely.
            cache_dir = root / "run-cache" / "_gallery_cache"
            for cached in cache_dir.glob("frame_*.png"):
                if cached.name != "frame_00002.png":
                    cached.unlink()
            lower = get_cumulative_frame(root, "run-cache", 1)
            self.assertTrue(lower.exists())

    def test_get_run_context_missing_run_is_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(GalleryNotFoundError):
                get_run_context(Path(tmpdir), "missing")


if __name__ == "__main__":
    unittest.main()
