# GestureFlow — Build Plan

This document is the working plan for taking GestureFlow from a single-machine
prototype to a correct, instrumented, user-configurable desktop app with a
deployable companion website.

It was written after a full read of the codebase at commit `21db12b`. Every
bug listed in Phase 1 was reproduced locally before being written down.

---

## Current state (baseline)

| Fact | Value | Verified by |
| :--- | :--- | :--- |
| Pipeline threads | 5 (capture, inference, main/render, vol-worker, vol-sync) | source read |
| Feature vector | 63 floats (21 landmarks × xyz, wrist-relative, max-abs scaled) | `utils.normalize_landmarks` |
| Model | `RandomForestClassifier`, 100 trees, `classes_ = [0 1 2 3]`, 3,264 nodes | unpickled |
| Training data | 941 rows, classes 335/202/202/202 | `data/gesture_data.csv` |
| Tests | 68 collected — 66 pass, **2 fail** | `pytest tests/` |
| Accuracy | **not measured** (printed at train time, never persisted) | no metrics file exists |
| FPS / latency | **not measured** (no instrumentation of any kind) | `CaptureResult.timestamp` is written and never read |
| False-trigger rate | **not measured** | no replay harness exists |

The two failing tests are correct; the code under them is wrong. Both are fixed
in Phase 1.

---

## Architecture decision: web ↔ desktop handoff

### The constraint

A page served from `https://gestureflow.vercel.app` runs in a browser sandbox.
It cannot move the host cursor, press host hotkeys, or call `osascript`. No
amount of engineering changes this, and pretending otherwise in the UI would be
dishonest. So the split is fixed:

- **Browser**: camera access, hand landmarks, gesture *recognition*, config authoring.
- **Desktop (Python)**: everything that touches the operating system.

The open question is only how a config authored in the browser reaches the
desktop app, and whether the browser can observe live desktop state.

### Option A — Config export / import (download a file, app loads it)

The web command builder produces a `commands.yaml`/`.json`. The user saves it to
`~/.gestureflow/commands.yaml` (or passes `--commands <path>`). The desktop app
validates and loads it, with optional hot-reload so edits apply without restart.

**Pros**
- Zero network surface. Nothing listens on a port, nothing is exposed.
- The site stays a pure static build — deploys to Vercel with no serverless
  functions, no secrets, no runtime, no cold starts.
- Works with the site open in any browser, offline, or not open at all.
- The config file is a real artifact the user owns, can diff, and can commit.

**Cons**
- Manual file move (download → `~/.gestureflow/`). One extra step.
- The browser cannot show live desktop gesture state.

### Option B — localhost WebSocket bridge

The Python app runs a WebSocket server on `127.0.0.1:8765`. The web UI connects,
receives live gesture state, and pushes configs directly.

**Pros**
- Live gesture state visible in the web UI; config push with no file handling.

**Cons — the decisive ones**
- **Mixed content.** An `https://` page opening `ws://localhost` is inconsistently
  permitted across browsers. Chrome and Firefox have carve-outs treating loopback
  as potentially trustworthy; **Safari — the default browser on the macOS this app
  targets — is the strictest**, and Chrome's Private Network Access work is
  tightening loopback access further. Building the primary handoff on a behaviour
  that varies by browser and is actively being restricted is a bad bet.
- Any process on the machine can talk to an unauthenticated loopback port. Doing
  this properly means an origin allowlist and a pairing token — real work, for a
  feature that is a convenience, not the product.
- The user must have the desktop app running *before* the site is useful, which
  inverts the onboarding order: the site's job is to explain the app to someone
  who has not installed it yet.

### Decision

**Option A is the chosen handoff and is implemented end to end.** It is the path
every visitor gets, it is the one the deployed site depends on, and it has no
browser-compatibility caveat.

**Option B ships as an optional local mode, and sidesteps its own main flaw.**
`gestureflow bridge` serves the *same* web UI from `http://127.0.0.1:8765`. When
the UI is served from loopback, the page origin is already `http://localhost`, so
`ws://localhost` is same-origin and no mixed-content rule applies in any browser.
The bridge is opt-in, binds to loopback only, and checks the `Origin` header.

So: the deployed site is self-sufficient via file export; users who want live
state run one command and get it, without relying on a browser carve-out.

---

## Phase 1 — Correctness

Every fix gets a test that fails before and passes after.

1. **`time.monotonic()` epoch bug.** `ClickFSM._last_click_time`,
   `GestureDebouncer._last_cmd_time`, `ScrollFSM._last_scroll_time` all init to
   `0.0` and are compared against `time.monotonic()`, whose reference point Python
   leaves undefined. Measured on this machine: `monotonic()` reads ~0.35 at process
   start, so every gated action is suppressed for the first cooldown-seconds after
   launch (1.3 s for hotkeys, 0.4/0.6 s for clicks). Init to `-math.inf`.
2. **Scroll float overshoot.** `0.500 - 0.480 = 0.020000000000000018`;
   `/0.010 = 2.0000000000000018`; `ceil → 3`. A 50 % overshoot at `exp=1.0`.
   Round to a sane precision before `ceil`.
3. **Capture drops the newest frame.** `put_nowait` + `except Full: pass` keeps
   stale frames and discards the fresh one, contradicting its own comment. Evict
   oldest instead, and count the drop.
4. **`data_logger.py` broken import.** `from src.utils import ...`; the package is
   `gestureflow`. Also switch it to `data_path()`.
5. **Frame-rate-dependent cursor smoothing.** `move_mouse_smooth` applies a fixed
   `1/5` step per *frame*. Convert to a time-constant exponential filter so the
   feel is identical at 15 and 60 FPS.
6. **Cursor origin sweep.** `_ploc` starts at `(0, 0)`; the first move sweeps from
   the screen corner. Seed from the live cursor position.

Cleanup:
- Delete `gestureflow/infref.py` (stale duplicate), `models/gesture_recognizer.task`
  (8.4 MB, referenced nowhere), `ClickState.RELEASING` (set then immediately reset
  in the same call — never externally observable).
- Restore `_draw_scroll_indicator` (it is a 10-line function and works).
- Route `scripts/*.py` through `data_path()` / `models_path()`.
- Startup validation: hard-fail if any predictable model class has no binding
  (`set(model.classes_) - {neutral} ⊆ set(bindings)`); warn if a binding targets a
  class the model cannot predict (which is the actual current defect — labels 4
  and 5 are bound to Screenshot and Do Not Disturb but `classes_` is `[0 1 2 3]`).

## Phase 2 — Systems

1. **Clock injection** into `ClickFSM`, `ScrollFSM`, `GestureDebouncer`. Prereq for
   deterministic tests and replay.
2. **Instrumentation** (`gestureflow/metrics.py`). Propagate the existing unused
   `CaptureResult.timestamp`; stamp every stage boundary; record per-stage
   durations, end-to-end capture→action latency as **p50/p95/p99**, per-stage FPS,
   queue-depth histogram, and drop counters. `gestureflow bench --seconds N` runs
   headless and writes a JSON report.
3. **Action thread** (`gestureflow/actions.py`). `pyautogui.moveTo/click/hotkey`
   currently block the render loop. Move them to a dedicated thread behind a
   bounded queue with **per-action-type drop semantics**: cursor moves coalesce to
   the latest (only the newest position matters); clicks, hotkeys, and scrolls are
   never dropped. Measure p95 before/after with (2).
4. **Replay harness** (`gestureflow/replay.py`). Record landmark streams to JSONL;
   replay through the real pipeline with an injected clock. Enables reproducible
   regression tests, A/B threshold comparison, and a **false-trigger rate**
   measurement over footage labelled "no intent".
5. **Graceful degradation.** Camera reconnect with exponential backoff + a visible
   HUD banner; a real stop `Event` for both volume workers; skip landmark drawing
   when the frame budget is exceeded.
6. **Tests.** `ClickFSM`, `capture`, `inference`, `controller` are all currently
   untested. Add unit tests plus **hypothesis** property tests for:
   - **mode exclusivity** — for any landmark configuration, at most one of
     {cursor, scroll, volume} is active;
   - click fires only on a `HELD → open` edge;
   - no double-fire within cooldown.
   Plus one full-pipeline integration test (fake landmarks + stub model + injected
   clock → asserted action sequence).

## Phase 3 — Declarative command system

`gestureflow/commands.py`. Replaces the hardcoded `GESTURE_MAP` dict.

- YAML (JSON also accepted) with a versioned schema.
- Closed action set, each with a typed schema and **no `eval` anywhere**:
  `hotkey`, `keypress`, `media` (play/pause, next, prev, volume up/down/mute),
  `launch` (open an app by name), `applescript`, `shell` (argv list, never a
  shell string).
- Ships `configs/commands.default.yaml` reproducing today's bindings exactly.
- Validates on load with precise, line-attributable errors.
- Optional hot-reload via mtime polling.
- This exact format is what the web command builder exports.

## Phase 4 — Web product

Static site in `web/`, deployable to Vercel with no build step.

- **Landing** — what it is; an explicit, unmissable statement of what runs in the
  browser vs. what needs the desktop app.
- **Guide** — gesture vocabulary, action mapping, camera + macOS permission setup,
  install and run steps.
- **Live demo** — MediaPipe Tasks Vision (JS/WASM) for landmarks, plus a faithful
  JS port of `normalize_landmarks`, the geometric FSMs, and the Random Forest
  (exported to JSON — 3,264 nodes total, small enough to ship in full fidelity).
  Labelled throughout as a **recognition preview that does not control the computer**.
- **Command builder** — pick a gesture, assign an action from the Phase-3 schema,
  export a valid `commands.yaml`. Round-trips: an existing config can be imported
  and edited.
- **Handoff** — the Option A download flow, end to end, plus the optional bridge.

## Phase 5 — Packaging, CI, docs

- `pyproject.toml`, `conftest.py`, pinned Python, `pytest.ini` settings.
- GitHub Actions: install, run the suite with coverage, on push and PR.
- README rewritten to match reality, with an explicit **measured vs. not measured**
  table.
- `SETUP.md` — every manual step, in order.

---

## MANUAL STEPS (things only the repo owner can do)

Collected here as they are identified; the authoritative ordered list lives in
`SETUP.md`.

1. **macOS Accessibility permission** — required for PyAutoGUI to move the cursor,
   click, and send hotkeys. System Settings → Privacy & Security → Accessibility →
   enable the terminal/IDE running the app. Cannot be scripted; macOS requires a
   human at the GUI.
2. **macOS Camera permission** — granted on first `cv2.VideoCapture` call via a
   system prompt. Must be accepted manually.
3. **macOS Screen Recording permission** — only needed if a binding takes
   screenshots. Not required by the default config.
4. **Automation / System Events permission** — first time an `applescript` action
   runs, macOS prompts to allow controlling System Events. Must be accepted once.
5. **Vercel deploy** — requires the owner's Vercel account and CLI login. Cannot be
   done from here.
6. **Retrain before adding gestures 4/5** — the shipped model predicts 4 classes.
   Screenshot and Do Not Disturb bindings need new labelled data (`gestureflow
   record-data`) and a retrain. Requires a human performing gestures at a camera.
7. **Generating performance numbers** — `bench` and `replay` must run on real
   hardware with a real camera. No number is claimed anywhere until the owner runs
   them.

---

## Honesty rules for this repo

No performance figure — accuracy, FPS, latency, "real-time", false-trigger rate —
appears in code, README, or site copy unless instrumentation in this repository
actually produced it on this machine. Where a number does not exist yet, the text
says it does not exist yet and names the command that would produce it.
