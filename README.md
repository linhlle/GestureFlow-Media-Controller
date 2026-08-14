# GestureFlow

Control macOS with hand gestures. One hand, a webcam, and your cursor,
clicks, scrolling, volume, and keyboard shortcuts.

MediaPipe supplies 21 hand landmarks per frame; a Random Forest classifies
poses; a majority-vote debouncer and three finite state machines decide whether
a movement was actually meant as a command. A companion website lets you try the
recognizer with no install and build your own command bindings.

[![tests](https://github.com/linhlle/GestureFlow-Media-Controller/actions/workflows/tests.yml/badge.svg)](https://github.com/linhlle/GestureFlow-Media-Controller/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What works today

| Capability | Status |
| :--- | :--- |
| Pause / resume kill switch (rock horns, held) | Working |
| Cursor control from index-finger position | Working |
| Left click (thumb + index pinch, fires on release) | Working |
| Right click (middle + index pinch) | Working |
| Drag and drop (pinch and hold) | Working |
| Scroll (closed fist, vertical wrist velocity) | Working |
| Swipe left/right (closed fist, horizontal flick) | Working |
| Zoom (thumb-index spread) | Working |
| Dwell click for accessibility | Working, off by default |
| System volume (thumb up, vertical wrist velocity) | Working |
| Three trained poses bound to configurable actions | Working |
| User-defined bindings via a validated config file | Working |
| Per-user calibration of pinch thresholds | Working |
| App-context profiles (bindings per frontmost app) | Working |
| Hot-reload of the config while running | Working |
| Headless benchmarking with per-stage timings | Working |
| Record and replay landmark takes | Working |
| Browser demo running the same trained model | Working |
| Local bridge serving the UI with live state | Working |
| Windows and Linux support | **Not implemented** — actions go through `osascript` |
| Two-handed gestures | **Not implemented** — MediaPipe is capped at one hand |
| Dynamic gestures (swipes, waves) | **Not implemented** — the model sees single frames only |

## Measured vs. not measured

This project claims no performance number it cannot produce on demand. Nothing
in this README, the code, or the website quotes a frame rate, latency figure,
accuracy percentage, or false-trigger rate.

| Quantity | State | How to produce it |
| :--- | :--- | :--- |
| End-to-end latency (p50/p95/p99) | **Measurable, not yet measured** | `python -m gestureflow bench --seconds 30` |
| Per-stage timings and FPS | **Measurable, not yet measured** | same |
| Queue depth and dropped frames | **Measurable, not yet measured** | same |
| False triggers per minute | **Measurable, not yet measured** | `record --label no-intent`, then `false-triggers` |
| Model accuracy / precision / recall | **Measurable, not yet measured** | `python scripts/train_model.py` writes `models/metrics.json` |
| Browser demo frame rate | Displayed live in the demo | shown in the tab, for that machine only |

Two of these deserve a caveat, and both are recorded in the artifacts
themselves rather than only here:

- **Accuracy is measured against a leaky split.** Training rows are logged one
  per video frame while a pose is held, so consecutive rows are near-duplicates.
  A random train/test split puts frame *k* in training and frame *k+1* in test.
  Whatever `metrics.json` reports is an optimistic upper bound, not a
  generalization estimate. A split grouped by recording session would be needed
  for that, and the dataset does not currently carry a session column.
- **Bench and false-trigger numbers describe one machine.** They depend on the
  camera, CPU, and lighting in front of them. The JSON reports say so.

Verifiable facts, as opposed to measurements:

| Fact | Value |
| :--- | :--- |
| Feature vector | 63 floats (21 landmarks × xyz, wrist-relative, max-abs scaled) |
| Model | `RandomForestClassifier`, 100 trees, classes `[0 1 2 3]`, 3,264 nodes |
| Training data | 941 labelled frames (335 / 202 / 202 / 202) |
| Tests | 615, all passing |
| Pipeline threads | 6, plus one for the frontmost-app poll when profiles are configured |

---

## Install

```bash
git clone https://github.com/linhlle/GestureFlow-Media-Controller.git
cd GestureFlow-Media-Controller
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Then grant macOS Accessibility permission — see [SETUP.md](SETUP.md), which
lists every step that needs a human.

```bash
gestureflow run
```

Press <kbd>q</kbd> in the window to quit.

## Command line

```
gestureflow run                       live control with the HUD
gestureflow bench --seconds 30        headless timing report (dry run)
gestureflow record out.jsonl --label no-intent
gestureflow replay take.jsonl         replay a take, print the actions
gestureflow false-triggers *.jsonl    count actions over no-intent footage
gestureflow validate                  check a command config
gestureflow selftest                  check the detectors, no camera needed
gestureflow calibrate                 fit pinch thresholds to your hand
gestureflow calibration               show the calibration in effect
gestureflow bridge                    serve the web UI locally with live state
```

## Gestures

Poses go through the classifier; everything else is geometry read straight off
the landmarks. Geometric modes are active only while the classifier reads
Neutral, and at most one runs at a time.

| Gesture | Mode | Default action |
| :--- | :--- | :--- |
| Index + pinky up, middle + ring down, hold ~1.5 s | paused | Toggle all control off/on |
| Index finger up | cursor | Move the pointer |
| Thumb + index pinch, hold, release | left click | Click |
| Thumb + index pinch, keep holding, move | drag | Press, drag, drop |
| Middle + index pinch, hold, release | right click | Right click |
| Closed fist, move vertically | scroll | Scroll |
| Closed fist, flick sideways | swipe | Switch desktop |
| Index up + thumb out, spread or close | zoom | Zoom in / out |
| Thumb up, move vertically | volume | System volume |
| Hold the pointer still (off by default) | dwell | Click |
| L-Shape (label 1) | command | Spotlight |
| High-Five (label 2) | command | Mission Control |
| 2-Finger (label 3) | command | App Switcher |

Modes are arbitrated by a single precedence ladder, so two can never be active
at once — exclusivity is structural rather than a property each mode has to
maintain against every other. Scroll and swipe share the fist and are separated
by which axis the motion favours; genuinely diagonal motion fires neither.

Clicks fire on the **release** edge of a held pinch, not the press. A hand
passing through a pinch shape does not click, and holding does not auto-repeat.
The gap between the close and open thresholds is hysteresis — without it a hand
hovering at the boundary chatters clicks.

Every geometric threshold is a fraction of your hand's own size (wrist to
middle knuckle), not an absolute distance, so gestures behave the same near and
far from the camera. A closed fist resolves to scroll and never to a click,
decided once per frame rather than left to threshold tuning.

## Custom commands

Bindings live in a config file, not in the source:

```yaml
version: 1
neutral_label: 0

gestures:
  - label: 1
    name: Spotlight
    action:
      type: hotkey
      keys: [command, space]
```

Nine action types: `hotkey`, `keypress`, `media`, `launch`, `url`, `text`,
`chord`, `applescript`, and `shell`. Bindings are keyed either by `label` (a
pose the model classifies) or by `gesture` (a geometric gesture needing no
training data). A `profiles:` block can change bindings based on the frontmost
app. Build one at the website's command builder, or copy
`configs/commands.default.yaml` to `~/.gestureflow/commands.yaml` and edit it.
`gestureflow validate` checks it; the app re-reads it while running.

A config file is treated as untrusted input, because it can arrive by download.
There is no `eval`. Key names are checked against a whitelist. App names are
matched against an anchored character class. `shell` takes an argv list and runs
without a shell, so pipes, redirects, and `&&` are literal text rather than
syntax.

## Architecture

```
┌───────────────┐  Queue[2]  ┌────────────────┐  Queue[1]  ┌──────────────┐
│ CaptureThread │ ─────────► │ InferenceThread│ ─────────► │ Main (render)│
│ camera + MP   │            │ forest + FSMs  │            │ HUD only     │
└───────────────┘            └────────────────┘            └──────┬───────┘
                                                                  │ actions
                                                           ┌──────▼───────┐
                                                           │ ActionThread │
                                                           │ pyautogui,   │
                                                           │ osascript    │
                                                           └──────────────┘
```

Both queue hand-offs drop the **oldest** entry when full. A landmark frame from
200 ms ago is worse than useless for cursor control, because acting on it moves
the pointer to where the hand *was*.

Action dispatch has drop semantics that differ by type. Cursor moves coalesce to
the newest position — delivering a stale one just walks the pointer somewhere
the user no longer wants it. Clicks, scrolls, and hotkeys are never dropped,
because silently swallowing a click a user consciously performed is a
correctness bug, not a performance trade-off.

The main thread does the least work of any thread in the system: it decides what
should happen and hands the doing to the action thread, so a slow keyboard
shortcut cannot stall rendering or stop the inference queue draining.

## The website

Static files in `web/`, deployable to Vercel with no build step.

The live demo runs the **actual** trained forest, exported to JSON and evaluated
in JavaScript — not a reimplementation. `scripts/export_model_json.py` verifies
the export reproduces scikit-learn's `predict_proba` before writing, and the
parity suites re-verify it in CI. Reimplementing the classifier would have
drifted the first time anyone retrained.

The demo cannot control your computer, and the copy says so plainly rather than
implying otherwise. A sandboxed page cannot move the cursor or press keys. Real
system control stays in the Python app.

Two test suites keep the JS and Python recognizers from diverging: a textual one
comparing every shared constant, and a numeric one running the JS under Node
over generated fixtures, diffing feature vectors, FSM state sequences, click
edges, scroll deltas, debouncer votes, and forest probabilities.

## Development

```bash
pip install -e ".[dev,train]"
pytest                      # 615 tests
pytest -m "not slow"        # skip property-based and Node-subprocess tests
ruff check gestureflow scripts tests
```

Retraining:

```bash
python scripts/data_logger.py      # hold 0-3 to record samples
python scripts/train_model.py      # writes the model and metrics.json
python scripts/export_model_json.py # refresh the browser demo's copy
```

## Repository layout

| Path | Purpose |
| :--- | :--- |
| `gestureflow/` | The application package |
| `gestureflow/app.py` | Pipeline wiring, HUD, gesture routing |
| `gestureflow/commands.py` | Config schema, validation, hot-reload |
| `gestureflow/metrics.py` | Instrumentation |
| `gestureflow/replay.py` | Record and replay |
| `gestureflow/smoothing.py` | One Euro cursor filter |
| `gestureflow/modes.py` | The mode enum and precedence ladder |
| `gestureflow/calibration.py` | Per-user threshold fitting |
| `gestureflow/frontmost.py` | Frontmost-app provider for profiles |
| `gestureflow/bridge.py` | Optional localhost server |
| `web/` | The static website |
| `scripts/` | Data collection, training, model export |
| `configs/` | The default command bindings |
| `tests/` | 615 tests |
| `PLAN.md` | Build plan and the web/desktop architecture decision |
| `DIAGNOSIS.md` | Root-cause analysis of the cursor and scroll regressions |
| `SETUP.md` | Every step that needs a human |

## License

MIT.
