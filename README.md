# Suffer for Your Art

A tiny, pure-digital experiment in closed-loop AI painting.

The painter begins with a blank canvas and can change it only by issuing validated brush-stroke actions. After each batch, a vision model sees the accumulated canvas and chooses what to do next. There is no finished target image and no direct pixel-generation path.

## What the MVP proves

- Every visible mark has **stroke provenance**.
- Strokes are normalized cubic Bézier paths with pressure, opacity, taper, pigment, and deterministic bristle variation.
- The canvas is built sequentially from `frame_000.png`, which is genuinely blank.
- Each round records the model's observation, interpretation, strokes, seeds, and before/after canvas hashes.
- A run can be replayed from its event log and checked against the original final hash.

## Setup

Python 3.11+ and Pillow are required.

```bash
python3 -m pip install -e .
```

For a no-install source checkout, prefix commands with `PYTHONPATH=src`.

## Offline demo

The demo exercises the exact validation, rendering, logging, and replay pipeline without credentials. Its planner is deterministic plumbing—not an AI artist.

```bash
suffer-for-your-art demo \
  --prompt "A red form leaning into an empty blue field" \
  --seed 42 \
  --max-rounds 4 \
  --max-strokes 3
```

Each run creates:

```text
artifacts/run-.../
├── manifest.json
├── process.jsonl
├── final.png
└── frames/
    ├── frame_000.png
    ├── frame_001.png
    └── ...
```

Replay it using only the recorded stroke events:

```bash
suffer-for-your-art replay artifacts/run-...
```

This writes `replay.png` and rejects altered or non-reproducible event histories.

## NanoGPT subscription mode

The live painter uses NanoGPT's subscription endpoints—not its pay-as-you-go endpoint.

Credentials are read only from:

1. `NANOGPT_API_KEY` in the process environment; or
2. `~/.config/pi/nanogpt.env`.

```bash
mkdir -p ~/.config/pi
printf 'NANOGPT_API_KEY=%s\n' 'your-key' > ~/.config/pi/nanogpt.env
chmod 600 ~/.config/pi/nanogpt.env
```

Never commit that file. List the model IDs included in the subscription:

```bash
suffer-for-your-art models
```

Choose a listed **multimodal/vision-capable** model that accepts OpenAI-compatible `image_url` input. `gemma-4-12b-it` has completed verified 100-stroke and 1,000-stroke runs in this project.

For a painting with an upper limit of 1,000 strokes, use `--max-strokes 20` (the painter may finish earlier):

```bash
suffer-for-your-art paint \
  --model 'gemma-4-12b-it' \
  --prompt "A figure gradually dissolving into its surroundings" \
  --stroke-budget 1000 \
  --max-strokes 20
```

The strict response schema requires an initial idea before the first marks. After every batch, the model sees the changed canvas before choosing whether to add strokes or stop. A terminal zero-stroke response records a self-assessment of the visible finished work.

If a transport failure interrupts a run after validated rounds have been persisted, resume it without repeating completed calls:

```bash
suffer-for-your-art resume artifacts/run-... --yes
```

Resume first verifies the manifest settings, contiguous event log, deterministic replay, final persisted frame hash, and prompt-template version. It then appends only the remaining rounds to the same provenance chain. Completed runs cannot be resumed.

Prompt templates 2.0–4.0 are schema-first: they send no role-playing system message. The model receives the initial intention, round/budget state, recent process history, and untouched current canvas; the strict JSON response schema is the painting action interface. Templates 3.0 and 4.0 also supply, after the first round, a separate context-only image that traces and numbers the 10 most recently rendered strokes over a washed grayscale canvas plus the exact matching ordered action list. Template 4.0 adds schema-required `initial_idea` and `final_reflection`; its stroke budget is a ceiling, so it can stop early with a final zero-stroke assessment. Historical templates retain exact reconstruction behavior, and runs without a recorded prompt-template version retain the legacy 1.0 reconstruction.

Without `--yes`, the CLI confirms every subscription call. `--yes` explicitly permits the full bounded loop. A 1,000-stroke ceiling with 20-stroke batches uses up to **51** calls (including a final assessment call). `--max-rounds` is hard-limited to 200 (for capacity planning), while `--stroke-budget` is limited to 1,000.

The default base is `https://nano-gpt.com/api/subscription/v1`. `NANOGPT_SUBSCRIPTION_BASE_URL` may override it. Routing suffixes such as `:fast` are rejected so a run does not accidentally bypass subscription coverage.

## Gallery player

A local web UI is included for browsing runs and stepping through one-stroke cumulative frames.

Run the server from the repository (or installed package):

```bash
PYTHONPATH=src python -m suffer_for_your_art gallery
```

Optional host/port/artifacts overrides:

```bash
PYTHONPATH=src python -m suffer_for_your_art gallery --host 127.0.0.1 --port 8000 --artifacts-dir artifacts
```

The UI renders these JSON endpoints:

- `GET /api/runs`
- `GET /api/runs/<run_id>`
- `GET /api/runs/<run_id>/events`
- `GET /api/runs/<run_id>/frame?strokes=<n>`
- `GET /api/runs/<run_id>/context?round=<n>`
- `GET /api/runs/<run_id>/context?round=<n>&raw=1`
- `GET /api/runs/<run_id>/context-map?round=<n>` (template 3.0 rounds after the first)

Controls:

- two-pane layout (run list + canvas)
- play/pause, prev/next, restart with seamless frame swaps
- live canonical-frame scrubbing while dragging the timeline
- one-tap speed presets (0.5×, 1×, 2×, 4×, 8×, 16×)
- left/right arrows for step, space to toggle playback
- per-round model-context inspector with the seen canvas, recent numbered stroke map/action list when available, messages, budget state, response schema, hashes, and raw request JSON
- no canvas approximation in JS: every frame is rendered from canonical stroke data on the Python server

Historical request bodies were not archived during painting. The inspector labels them as deterministic reconstructions built from the persisted manifest, event history, exact saved pre-round frame, and the same pure request builder now used for live calls. Raw mode includes the image data URL but never includes authorization headers or API credentials.

If user motion preference is reduced, playback uses longer per-step delays and the UI relies on static CSS transitions only.

## Stroke contract

The model must return one JSON object:

```json
{
  "observation": "What is actually visible now",
  "interpretation": "How the current canvas changes the intention",
  "strokes": [
    {
      "color": "carbon",
      "width": 8,
      "pressure": 0.6,
      "opacity": 0.8,
      "bezier": {
        "start": [0.1, 0.7],
        "control1": [0.25, 0.45],
        "control2": [0.55, 0.4],
        "end": [0.8, 0.2]
      },
      "taper": {"start": 0.7, "end": 0.3}
    }
  ],
  "done": false
}
```

Constraints:

- 1–20 strokes per round
- optional total ceiling of 1–1,000 strokes (`--stroke-budget`; the painter may stop early)
- coordinates normalized to `[0, 1]`
- width `1–30`, pressure `0–1`, opacity `0.05–1`
- named palette only: `carbon`, `umber`, `ink`, `alizarin`, `emerald`, `azure`, `ochre`
- malformed or out-of-contract output is rejected before rendering

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests cover action validation, routing-suffix rejection, safety limits, fenced JSON parsing, deterministic replay/hash behavior, exact 100-stroke and 1,000-stroke budgets, validated interruption resume, and gallery discovery/frame cache discovery and rendering validation.

## MVP limitations

- Pigments currently alpha-blend; there is no wetness, drying, viscosity, contamination, or chemical color mixing.
- Pressure is a rendering parameter, not a simulated brush-body model.
- The offline demo planner does not inspect or understand the canvas.
- Live NanoGPT behavior depends on the selected model's vision and structured-output reliability.
- There is no dedicated time-lapse export yet; saved frames are still the process record.
