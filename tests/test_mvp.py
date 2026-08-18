from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suffer_for_your_art.brush import image_sha256, render_strokes
from suffer_for_your_art.core import PainterRunner, PainterRuntimeError, replay_run
from suffer_for_your_art.feedback import (
    RECENT_FEEDBACK_PREFIX,
    STUDIO_GUIDE_MESSAGE,
    recent_move_list,
    render_recent_stroke_map,
    render_studio_guide,
)
from suffer_for_your_art.nanogpt import NanoGPTError, validate_model
from suffer_for_your_art.parser import parse_planner_output
from suffer_for_your_art.planner import (
    build_history_summary,
    deterministic_demo_plan,
    prompt_for_model,
)
from suffer_for_your_art.validation import ValidationError, validate_planner_payload


class ValidationTests(unittest.TestCase):
    def test_out_of_bounds_stroke_rejected(self):
        payload = {
            "observation": "bad",
            "interpretation": "bad",
            "done": False,
            "strokes": [
                {
                    "color": "carbon",
                    "width": 5,
                    "pressure": 0.5,
                    "opacity": 0.9,
                    "bezier": {
                        "start": [1.2, 0.5],
                        "control1": [0.2, 0.2],
                        "control2": [0.4, 0.4],
                        "end": [0.6, 0.6],
                    },
                }
            ],
        }
        with self.assertRaises(ValidationError):
            validate_planner_payload(payload, max_strokes=5)

    def test_parser_accepts_fenced_json(self):
        text = """\nSome chatter\n```json\n{\"observation\": \"a\", \"interpretation\": \"b\", \"done\": false, \"strokes\": []}\n```\n"""
        # should fail because empty strokes is contract violation (must be 1..N)
        with self.assertRaises(ValidationError):
            parse_planner_output(text)

        text = """```json\n{\"observation\": \"a\", \"interpretation\": \"b\", \"done\": false, \"strokes\": [\n  {\n    \"color\": \"carbon\",\n    \"width\": 4,\n    \"pressure\": 0.5,\n    \"opacity\": 0.8,\n    \"bezier\": {\n      \"start\": [0.1, 0.1],\n      \"control1\": [0.2, 0.2],\n      \"control2\": [0.3, 0.3],\n      \"end\": [0.4, 0.4]\n    }\n  }\n]}\n```"""
        parsed = parse_planner_output(text)
        self.assertEqual(parsed["observation"], "a")
        self.assertFalse(parsed["done"])
        self.assertEqual(len(parsed["strokes"]), 1)

    def test_subscription_model_id_rejects_routing_suffix(self):
        self.assertEqual(validate_model("vendor/model-name"), "vendor/model-name")
        with self.assertRaises(NanoGPTError):
            validate_model("model-name:fast")

    def test_runtime_limits_rounds_canvas_and_budget_capacity(self):
        with self.assertRaises(PainterRuntimeError):
            PainterRunner(prompt="x", max_rounds=201)
        with self.assertRaises(PainterRuntimeError):
            PainterRunner(prompt="x", max_strokes=21)
        with self.assertRaises(PainterRuntimeError):
            PainterRunner(prompt="x", canvas_size=64)
        with self.assertRaises(PainterRuntimeError):
            PainterRunner(prompt="x", stroke_budget=1001)
        with self.assertRaises(PainterRuntimeError):
            PainterRunner(prompt="x", max_rounds=20, max_strokes=5, stroke_budget=101)

    def test_working_state_is_carried_in_current_history_summary(self):
        summary = build_history_summary(
            [
                {
                    "round": 1,
                    "done": False,
                    "observation": "A mark is visible.",
                    "interpretation": "The first gesture landed.",
                    "strokes_count": 1,
                    "working_state": {
                        "current_goal": "Establish the main form",
                        "completed_parts": ["first gesture"],
                        "last_change": "A dark curve appeared.",
                        "next_action": "Continue the contour",
                        "alignment": "aligned",
                    },
                }
            ],
            include_working_state=True,
        )
        self.assertIn("goal='Establish the main form'", summary)
        self.assertIn("completed=['first gesture']", summary)
        self.assertIn("next='Continue the contour'", summary)
        self.assertIn("alignment='aligned'", summary)

    def test_planner_prompt_is_schema_first_with_budget_state(self):
        messages = prompt_for_model(
            prompt="Test composition",
            round_index=7,
            summary="Round 6: done=False | observation='x' | interpretation='y' | strokes=5",
            max_strokes=12,
            max_rounds=200,
            strokes_used=13,
            stroke_budget=1000,
            exact_strokes=12,
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        user_content = messages[0]["content"]
        self.assertIn("INITIAL INTENTION: Test composition", user_content)
        self.assertIn("ROUND STATE:", user_content)
        self.assertIn("maximum_total_strokes=1000", user_content)
        self.assertIn("committed_strokes=13", user_content)
        self.assertIn("strokes_remaining=987", user_content)
        self.assertIn("maximum_batch_size=12", user_content)
        self.assertIn("may_stop_early=true", user_content)
        self.assertIn("stroke_budget_is_ceiling=true", user_content)
        self.assertIn("PROCESS HISTORY:", user_content)
        self.assertNotIn("You are an autonomous painter", user_content)
        self.assertNotIn("ACTION CONTRACT", user_content)


class ReplayAndDemoTests(unittest.TestCase):
    def _run_demo(self, output_dir: Path, seed: int, max_rounds: int = 2) -> dict:
        runner = PainterRunner(
            prompt="A compact red circle over a blue field",
            output_dir=str(output_dir),
            max_rounds=max_rounds,
            max_strokes=3,
            seed=seed,
            yes=True,
            dry_run=True,
        )
        return runner.run_demo()

    def test_recent_stroke_map_is_deterministic_and_does_not_mutate_canvas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "current.png"
            map_a = root / "map-a.png"
            map_b = root / "map-b.png"
            payload = deterministic_demo_plan(
                "feedback map",
                round_index=1,
                max_strokes=12,
                max_rounds=1,
                seed=31,
                exact_strokes=12,
            )
            strokes = parse_planner_output(json.dumps(payload), max_strokes=12)["strokes"]
            for index, stroke in enumerate(strokes):
                stroke["seed"] = 1000 + index
            canvas = Image.new("RGBA", (128, 128), (255, 255, 255, 255))
            render_strokes(canvas, strokes, seed=77, run_seed=31)
            canvas.save(source_path)
            source_before = source_path.read_bytes()

            render_recent_stroke_map(
                source_path,
                strokes,
                map_a,
                round_seed=77,
                run_seed=31,
            )
            render_recent_stroke_map(
                source_path,
                strokes,
                map_b,
                round_seed=77,
                run_seed=31,
            )
            moves = recent_move_list(strokes)

            self.assertEqual(source_path.read_bytes(), source_before)
            self.assertEqual(map_a.read_bytes(), map_b.read_bytes())
            self.assertEqual(len(moves), 10)
            self.assertEqual(moves[0]["previous_batch_index"], 3)
            self.assertEqual(moves[-1]["map_label"], 10)
            self.assertNotIn("seed", moves[0]["action"])
            with Image.open(map_a) as diagnostic:
                self.assertEqual(diagnostic.size, canvas.size)
                self.assertIsNotNone(ImageChops.difference(canvas, diagnostic.convert("RGBA")).getbbox())

    def test_studio_guide_is_deterministic_and_does_not_mutate_canvas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "current.png"
            guide_a = root / "guide-a.png"
            guide_b = root / "guide-b.png"
            canvas = Image.new("RGBA", (128, 128), (255, 255, 255, 255))
            canvas.save(source_path)
            source_before = source_path.read_bytes()

            render_studio_guide(source_path, guide_a)
            render_studio_guide(source_path, guide_b)

            self.assertEqual(source_path.read_bytes(), source_before)
            self.assertEqual(guide_a.read_bytes(), guide_b.read_bytes())
            with Image.open(guide_a) as guide:
                self.assertEqual(guide.size, canvas.size)
                self.assertIsNotNone(ImageChops.difference(canvas, guide.convert("RGBA")).getbbox())

    def test_template5_sends_quiet_studio_guide_and_recent_action_output_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            captured_messages = []
            call_count = 0

            def fake_chat(messages, model, **_kwargs):
                del model
                nonlocal call_count
                call_count += 1
                captured_messages.append(messages)
                if call_count == 3:
                    return json.dumps(
                        {
                            "initial_idea": "A sparse centered gesture.",
                            "observation": "The canvas has the intended marks.",
                            "interpretation": "The study is complete.",
                            "strokes": [],
                            "done": True,
                            "title": "Sparse gesture",
                            "final_reflection": "The composition is complete.",
                        }
                    )
                payload = deterministic_demo_plan(
                    "Map my last moves",
                    round_index=call_count,
                    max_strokes=1,
                    max_rounds=3,
                    seed=43,
                    exact_strokes=1,
                )
                payload.update(
                    {
                        "initial_idea": "A sparse centered gesture.",
                        "done": False,
                        "final_reflection": "",
                    }
                )
                return json.dumps(payload)

            runner = PainterRunner(
                prompt="Map my last moves",
                output_dir=tmpdir,
                model="test/model",
                max_rounds=3,
                max_strokes=1,
                stroke_budget=2,
                seed=43,
                yes=True,
                dry_run=False,
                prompt_template_version="5.0",
            )
            with patch("suffer_for_your_art.core.chat_completion", side_effect=fake_chat):
                manifest = runner.run_paint()

            self.assertEqual(manifest["prompt_template_version"], "5.0")
            self.assertEqual(len(captured_messages[0]), 3)
            self.assertEqual(len(captured_messages[1]), 4)
            studio_message = captured_messages[1][2]
            self.assertEqual(studio_message["content"][0]["text"], STUDIO_GUIDE_MESSAGE)
            self.assertTrue(studio_message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
            run_dir = Path(manifest["manifest_path"]).parent
            self.assertTrue((run_dir / "context" / "studio_guide_round_002.png").exists())

    def test_template8_sends_reference_image_as_context_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "hand-reference.png"
            Image.new("RGBA", (32, 32), (240, 240, 240, 255)).save(reference)
            captured_messages = []
            call_count = 0

            def fake_chat(messages, model, **_kwargs):
                del model
                nonlocal call_count
                call_count += 1
                captured_messages.append(messages)
                return json.dumps(
                    {
                        "initial_idea": "Study the reference subject with deliberate brush gestures.",
                        "observation": "The study is complete.",
                        "interpretation": "The visible marks are sufficient for this study.",
                        "strokes": [] if call_count == 2 else [deterministic_demo_plan("ref", 1, 1, 2, 7, exact_strokes=1)["strokes"][0]],
                        "done": call_count == 2,
                        "title": "Reference Study" if call_count == 2 else "",
                        "working_state": {
                            "current_goal": "Study the reference subject",
                            "completed_parts": [],
                            "last_change": "",
                            "next_action": "Stop" if call_count == 2 else "Continue",
                            "alignment": "aligned" if call_count == 2 else "uncertain",
                        },
                        "final_reflection": "The study is complete." if call_count == 2 else "",
                    }
                )

            runner = PainterRunner(
                prompt="A reference study",
                output_dir=tmpdir,
                model="test/model",
                reference_image=str(reference),
                max_rounds=2,
                max_strokes=1,
                stroke_budget=1,
                seed=7,
                yes=True,
                dry_run=False,
            )
            with patch("suffer_for_your_art.core.chat_completion", side_effect=fake_chat):
                manifest = runner.run_paint()

            self.assertEqual(manifest["prompt_template_version"], "8.0")
            self.assertEqual(len(captured_messages[0]), 4)
            ref_message = next(
                message for message in captured_messages[0]
                if isinstance(message["content"], list)
                and message["content"][0]["text"].startswith("REFERENCE IMAGE")
            )
            self.assertTrue(ref_message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertTrue((Path(manifest["manifest_path"]).parent / "context" / "reference.png").exists())

    def test_template3_sends_recent_action_output_map_after_first_round(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            captured_messages = []
            call_count = 0

            def fake_chat(messages, model, **_kwargs):
                del model
                nonlocal call_count
                call_count += 1
                captured_messages.append(messages)
                payload = deterministic_demo_plan(
                    "Map my last moves",
                    round_index=call_count,
                    max_strokes=2,
                    max_rounds=2,
                    seed=41,
                    exact_strokes=2,
                )
                payload["done"] = False
                return json.dumps(payload)

            runner = PainterRunner(
                prompt="Map my last moves",
                output_dir=tmpdir,
                model="test/model",
                max_rounds=2,
                max_strokes=2,
                stroke_budget=4,
                seed=41,
                yes=True,
                dry_run=False,
                prompt_template_version="3.0",
            )
            with patch("suffer_for_your_art.core.chat_completion", side_effect=fake_chat):
                manifest = runner.run_paint()

            self.assertEqual(manifest["prompt_template_version"], "3.0")
            self.assertEqual(len(captured_messages), 2)
            self.assertEqual(len(captured_messages[0]), 2)
            self.assertEqual(len(captured_messages[1]), 3)
            recent_message = captured_messages[1][-1]
            self.assertEqual(recent_message["role"], "user")
            self.assertTrue(recent_message["content"][0]["text"].startswith(RECENT_FEEDBACK_PREFIX))
            self.assertTrue(
                recent_message["content"][1]["image_url"]["url"].startswith(
                    "data:image/png;base64,"
                )
            )
            run_dir = Path(manifest["manifest_path"]).parent
            self.assertTrue((run_dir / "context" / "recent_strokes_round_002.png").exists())
            replay_path = replay_run(run_dir)
            self.assertEqual(
                image_sha256(Image.open(replay_path).convert("RGBA")),
                manifest["final_canvas_hash"],
            )

    def test_deterministic_replay_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp1 = Path(tmpdir) / "run1"
            tmp2 = Path(tmpdir) / "run2"
            tmp1.mkdir()
            tmp2.mkdir()
            manifest_a = self._run_demo(tmp1, seed=1337, max_rounds=3)
            manifest_b = self._run_demo(tmp2, seed=1337, max_rounds=3)

            hash_a = image_sha256(Image.open(manifest_a["final_image"]).convert("RGBA"))
            hash_b = image_sha256(Image.open(manifest_b["final_image"]).convert("RGBA"))
            replay_path = replay_run(Path(manifest_a["manifest_path"]).parent)
            replay_hash = image_sha256(Image.open(replay_path).convert("RGBA"))
            self.assertEqual(hash_a, hash_b)
            self.assertEqual(hash_a, replay_hash)
            self.assertEqual(manifest_a["round_records"], manifest_b["round_records"])

    def test_demo_run_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = self._run_demo(Path(tmpdir), seed=2026, max_rounds=2)

            final_png = Path(manifest["final_image"])
            log_path = Path(manifest["process_log"])
            self.assertTrue(final_png.exists())
            self.assertTrue(log_path.exists())
            self.assertEqual(manifest["prompt_template_version"], "8.0")
            self.assertTrue(manifest["title"])
            self.assertTrue(manifest["working_state"])
            self.assertTrue(manifest["initial_idea"])
            self.assertTrue(manifest["final_reflection"])

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreater(len(lines), 0)
            self.assertLessEqual(len(lines), 2)
            self.assertTrue(lines[0]["strokes_count"] >= 1)

    def test_exact_budget_ignores_premature_model_done(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = PainterRunner(
                prompt="Premature done regression",
                output_dir=tmpdir,
                max_rounds=2,
                max_strokes=2,
                stroke_budget=4,
                seed=7,
                yes=True,
                dry_run=True,
                prompt_template_version="3.0",
            )

            def premature_done_planner(round_index, _last_frame, _history, round_limit):
                payload = deterministic_demo_plan(
                    runner.prompt,
                    round_index,
                    round_limit,
                    runner.max_rounds,
                    runner.seed,
                    exact_strokes=round_limit,
                )
                payload["done"] = True
                return parse_planner_output(json.dumps(payload), max_strokes=round_limit)

            manifest = runner._run_loop(premature_done_planner, mode="test")
            self.assertEqual(manifest["status"], "done")
            self.assertEqual(manifest["stop_reason"], "stroke_budget_exhausted")
            self.assertEqual(manifest["total_strokes"], 4)
            self.assertEqual(manifest["rounds_executed"], 2)
            self.assertFalse(manifest["round_records"][0]["done"])
            self.assertTrue(manifest["round_records"][1]["done"])

    def test_reflective_painter_can_stop_early_after_inspecting_canvas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = PainterRunner(
                prompt="A simple smiley face",
                output_dir=tmpdir,
                max_rounds=4,
                max_strokes=2,
                stroke_budget=4,
                seed=19,
                yes=True,
                dry_run=True,
            )

            def early_stopping_planner(round_index, _last_frame, _history, round_limit):
                if round_index == 1:
                    payload = deterministic_demo_plan(
                        runner.prompt,
                        round_index,
                        round_limit,
                        runner.max_rounds,
                        runner.seed,
                        exact_strokes=2,
                    )
                    payload.update(
                        {
                            "initial_idea": "A round face with two eyes and a curved smile.",
                            "done": False,
                            "final_reflection": "",
                        }
                    )
                else:
                    payload = {
                        "initial_idea": "A round face with two eyes and a curved smile.",
                        "observation": "The face reads clearly on the current canvas.",
                        "interpretation": "No more marks are needed.",
                        "strokes": [],
                        "done": True,
                        "title": "A cheerful face",
                        "final_reflection": "It is simple, cheerful, and immediately recognizable.",
                    }
                return parse_planner_output(
                    json.dumps(payload),
                    max_strokes=round_limit,
                    reflective=True,
                )

            manifest = runner._run_loop(early_stopping_planner, mode="test")
            events = [
                json.loads(line)
                for line in Path(manifest["process_log"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest["status"], "done")
            self.assertEqual(manifest["stop_reason"], "model_declared_done")
            self.assertEqual(manifest["total_strokes"], 2)
            self.assertEqual(manifest["rounds_executed"], 2)
            self.assertEqual(manifest["initial_idea"], events[0]["initial_idea"])
            self.assertEqual(manifest["final_reflection"], events[-1]["final_reflection"])
            self.assertEqual(events[-1]["strokes"], [])
            replay_path = replay_run(Path(manifest["manifest_path"]).parent)
            self.assertEqual(
                image_sha256(Image.open(replay_path).convert("RGBA")),
                manifest["final_canvas_hash"],
            )

    def test_resume_continues_verified_partial_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = {
                "prompt": "Resume regression",
                "output_dir": tmpdir,
                "max_rounds": 2,
                "max_strokes": 2,
                "stroke_budget": 4,
                "seed": 11,
                "yes": True,
                "dry_run": True,
                "prompt_template_version": "2.0",
            }
            runner = PainterRunner(**settings)

            def stopping_planner(round_index, _last_frame, _history, round_limit):
                if round_index == 2:
                    raise PainterRuntimeError("simulated interruption")
                payload = deterministic_demo_plan(
                    runner.prompt,
                    round_index,
                    round_limit,
                    runner.max_rounds,
                    runner.seed,
                    exact_strokes=round_limit,
                )
                return parse_planner_output(json.dumps(payload), max_strokes=round_limit)

            partial = runner._run_loop(stopping_planner, mode="demo")
            run_dir = Path(partial["manifest_path"]).parent
            original_line = (run_dir / "process.jsonl").read_text(encoding="utf-8")
            self.assertEqual(partial["status"], "stopped")
            self.assertEqual(partial["total_strokes"], 2)

            resumed_rounds = []
            resumer = PainterRunner(**settings)

            def resumed_planner(round_index, _last_frame, _history, round_limit):
                resumed_rounds.append(round_index)
                payload = deterministic_demo_plan(
                    resumer.prompt,
                    round_index,
                    round_limit,
                    resumer.max_rounds,
                    resumer.seed,
                    exact_strokes=round_limit,
                )
                return parse_planner_output(json.dumps(payload), max_strokes=round_limit)

            completed = resumer._run_loop(
                resumed_planner,
                mode="demo",
                resume_run_dir=run_dir,
            )
            final_log = (run_dir / "process.jsonl").read_text(encoding="utf-8")
            self.assertEqual(resumed_rounds, [2])
            self.assertTrue(final_log.startswith(original_line))
            self.assertEqual(len(final_log.splitlines()), 2)
            self.assertEqual(completed["status"], "done")
            self.assertEqual(completed["stop_reason"], "stroke_budget_exhausted")
            self.assertEqual(completed["total_strokes"], 4)
            replay_path = replay_run(run_dir)
            self.assertEqual(
                image_sha256(Image.open(replay_path).convert("RGBA")),
                completed["final_canvas_hash"],
            )

    def test_exact_100_stroke_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = PainterRunner(
                prompt="One hundred marks",
                output_dir=tmpdir,
                canvas_size=128,
                max_rounds=10,
                max_strokes=10,
                stroke_budget=100,
                seed=100,
                yes=True,
                dry_run=True,
                prompt_template_version="3.0",
            )
            manifest = runner.run_demo()
            events = [
                json.loads(line)
                for line in Path(manifest["process_log"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest["status"], "done")
            self.assertEqual(manifest["stop_reason"], "stroke_budget_exhausted")
            self.assertEqual(manifest["total_strokes"], 100)
            self.assertEqual([event["strokes_count"] for event in events], [10] * 10)
            replay_path = replay_run(Path(manifest["manifest_path"]).parent)
            self.assertEqual(
                image_sha256(Image.open(manifest["final_image"]).convert("RGBA")),
                image_sha256(Image.open(replay_path).convert("RGBA")),
            )

    def test_exact_1000_stroke_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = PainterRunner(
                prompt="A thousand-stroke budget",
                output_dir=tmpdir,
                canvas_size=128,
                max_rounds=200,
                max_strokes=5,
                stroke_budget=1000,
                seed=1000,
                yes=True,
                dry_run=True,
                prompt_template_version="3.0",
            )
            manifest = runner.run_demo()
            events = [
                json.loads(line)
                for line in Path(manifest["process_log"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest["status"], "done")
            self.assertEqual(manifest["stop_reason"], "stroke_budget_exhausted")
            self.assertEqual(manifest["total_strokes"], 1000)
            self.assertEqual(manifest["rounds_executed"], 200)
            self.assertEqual(len(events), 200)
            self.assertTrue(all(event["strokes_count"] == 5 for event in events))
            replay_path = replay_run(Path(manifest["manifest_path"]).parent)
            self.assertEqual(
                image_sha256(Image.open(manifest["final_image"]).convert("RGBA")),
                image_sha256(Image.open(replay_path).convert("RGBA")),
            )


if __name__ == "__main__":
    unittest.main()
