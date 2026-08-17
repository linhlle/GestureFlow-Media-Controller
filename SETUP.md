# SETUP — steps that need a human

Everything in this list requires a person at the keyboard. macOS permission
grants cannot be scripted by design, deploys need account credentials, and
retraining needs someone to perform gestures at a camera.

Work through it in order. Steps 1–5 get the desktop app running; the rest are
optional.

---

## 1. Install the app

```bash
git clone https://github.com/liz112200/GestureFlow-Media-Controller.git
cd GestureFlow-Media-Controller
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Verify:

```bash
gestureflow validate
gestureflow selftest
```

Expect `OK:` followed by three bindings, then `[selftest] OK`. Neither needs a
camera. If either fails, nothing below will work.

---

## 2. Grant macOS Accessibility permission — REQUIRED

**Without this the app runs, recognizes gestures, displays them in the HUD, and
does nothing at all.** It is the single most common reason GestureFlow appears
broken.

1. Open **System Settings**
2. **Privacy & Security → Accessibility**
3. Click **+**
4. Add the application you will launch GestureFlow from — the *terminal*, not
   Python itself:
   - Terminal.app → `/System/Applications/Utilities/Terminal.app`
   - iTerm2 → `/Applications/iTerm.app`
   - VS Code → `/Applications/Visual Studio Code.app`
5. Make sure its toggle is **on**
6. **Quit and reopen that application.** The permission is only picked up on a
   fresh launch.

If you later run GestureFlow from a different terminal, repeat this for that
one. The permission is per-application, not per-user.

---

## 3. Allow camera access — REQUIRED

macOS prompts the first time the app opens the camera. Click **Allow**.

If you dismissed the prompt or clicked Don't Allow:

1. **System Settings → Privacy & Security → Camera**
2. Enable the same application you added in step 2
3. Quit and reopen it

---

## 4. Run it

```bash
gestureflow run
```

A window opens showing your camera with an overlay. Press <kbd>q</kbd> **in that
window** (not Ctrl-C in the terminal) to quit cleanly.

Check each mode works:

- Point your index finger up (thumb tucked) → the cursor should track smoothly
- Pinch thumb to index, hold about a fifth of a second, release → a click
- Close your fist, hold it still for about a quarter second until the HUD reads
  SCROLL, then move your whole hand up and down → scrolling. The HUD must never
  read RIGHT CLICK while your fist is closed.
- Thumb up, index down, move vertically → volume
- Pinch and **keep holding** for about half a second, move, then release →
  press, drag and drop
- Closed fist, **flick sideways** → switches desktop
- Index up with the thumb out sideways, then spread or close them → zoom
- **Index and pinky up, middle and ring down, hold ~1.5 s** → the window says
  PAUSED and nothing reaches your Mac. Hold it again to resume.

If a gesture misfires, the pause switch is the fastest way to stop everything.

---

## 4b. Calibrate — OPTIONAL but recommended

```bash
gestureflow calibrate
```

Four short recordings: pinch closed, hand open, middle-finger pinch, hand open
again. It fits the pinch thresholds to your hand and writes
`~/.gestureflow/calibration.json`.

Thresholds are already relative to your hand's size, which handles sitting
closer to or further from the camera. What that cannot handle is that hands
differ from each other — so **if clicking feels unreliable while poses work
fine, run this first.**

Check what is in effect, or go back to the shipped defaults:

```bash
gestureflow calibration          # show current values
rm ~/.gestureflow/calibration.json   # revert
```

Nothing here can break the app: a missing, corrupt or nonsensical calibration
file falls back to the defaults with a note on stdout.

---

## 5. Install your own command bindings — OPTIONAL

Either build one on the website's command builder and download it, or copy the
default:

```bash
mkdir -p ~/.gestureflow
cp configs/commands.default.yaml ~/.gestureflow/commands.yaml
```

Edit it, then check it:

```bash
gestureflow validate
```

The app re-reads the file while running, so edits apply without a restart. A
config that fails to parse is ignored and the previous one keeps running.

The default config binds the three trained poses plus the four geometric
gestures (swipe left/right, zoom in/out). Turning one off is as simple as
deleting its entry.

**Extra permissions, only if you use the features that need them:**

- **AppleScript actions** — the first time one runs, macOS asks to allow
  controlling System Events. Accept it. Later, under **Privacy & Security →
  Automation**.
- **App-context profiles** (step 5b) — reading the frontmost app uses the same
  **Automation** permission. Declining it is not fatal: the default profile
  applies and everything else carries on.
- **Screenshot shortcuts** — need **Privacy & Security → Screen Recording**,
  and that one requires a full restart of the application.

None is needed by the default config.

---

## 5b. App-context profiles — OPTIONAL

Make a gesture mean different things in different apps. Add a `profiles:` block
to `~/.gestureflow/commands.yaml`:

```yaml
profiles:
  - name: presentation
    match:
      apps: [Keynote, Microsoft PowerPoint]
    gestures:
      - gesture: swipe_right
        name: Next slide
        action: { type: keypress, key: right }
      - gesture: swipe_left
        name: Previous slide
        action: { type: keypress, key: left }
```

A profile lists only what it changes; the rest keeps working as it does by
default. The first matching profile wins, so precedence reads top to bottom.

**This is the one new feature that needs a permission.** The first time the app
asks which application is frontmost, macOS prompts for Automation access. Accept
it. If you decline, the default profile applies and the app prints one line
saying so — nothing else changes.

Verify it is working by switching apps with `gestureflow run` in the foreground;
the terminal prints `[controller] Profile: presentation (Keynote)` on each
change.

---

## 5c. Dwell click — OPTIONAL, accessibility

Off by default, because when it is on, resting the pointer clicks. For users who
cannot pinch:

```bash
DWELL_ENABLED=1 gestureflow run
```

Tune with `DWELL_SECONDS` (default 1.0) and `DWELL_RADIUS_PX` (default 40).

---

## 6. Deploy the website — OPTIONAL, needs your Vercel account

The site is static files in `web/`. It cannot be deployed from a sandbox; it
needs your credentials.

```bash
npm i -g vercel
vercel login
vercel --prod
```

`vercel.json` is already configured: output directory `web`, clean URLs, and a
`Permissions-Policy` header allowing camera access on the site's own origin.

Before deploying, make sure the exported model is current:

```bash
python scripts/export_model_json.py
```

This writes `web/models/forest.json` and verifies it reproduces scikit-learn's
predictions exactly. **If you have retrained the model, the site ships the old
one until you rerun this.**

No environment variables or secrets are required. There is no backend.

---

## 7. Run the local bridge — OPTIONAL

Serves the same web UI from your own machine, with live gesture state, and lets
the command builder write your config directly.

```bash
gestureflow bridge
```

Opens `http://127.0.0.1:8765`. It binds to loopback only and refuses any other
host. Config pushes are written to `~/.gestureflow/commands.yaml`.

Keep it running while you use the builder. Stop it with Ctrl-C.

If port 8765 is taken:

```bash
gestureflow bridge --port 9000
```

---

## 8. Collect data and retrain — OPTIONAL, needs you at a camera

Necessary if the shipped model reads your hands poorly, or to add gestures. It
was trained on one person's hands in one room, so it may well read yours less
reliably.

```bash
python scripts/data_logger.py
```

Hold <kbd>0</kbd>–<kbd>3</kbd> while holding the matching pose. Aim for a few
hundred frames per class.

**Vary your position while recording.** Change your distance from the camera,
your hand angle, and the lighting. Samples collected in one fixed position
produce a model that only works in that position.

Then:

```bash
python scripts/train_model.py        # writes the model and models/metrics.json
python scripts/export_model_json.py  # refresh the browser demo
```

To add a gesture beyond the four shipped, record it as label 4 or higher,
retrain, and bind it in your config. **The app refuses to start if a pose it can
recognize has no action bound** — that is deliberate, so an unbound gesture is a
startup error rather than a gesture that mysteriously does nothing.

---

## 9. Generate the performance numbers — OPTIONAL

None of these have been run. They need real hardware, a real camera, and in the
false-trigger case, several minutes of you working normally.

**Latency and frame rate:**

```bash
gestureflow bench --seconds 30
```

Runs headless without touching your cursor, prints per-stage timings, and writes
a JSON report to `bench/`. Latency comes out as p50/p95/p99.

**False-trigger rate:**

```bash
mkdir -p takes
gestureflow record takes/no-intent-1.jsonl --label no-intent --seconds 300
```

While recording, **work normally and do not try to control anything** — type,
talk, gesture, reach for a drink. Then:

```bash
gestureflow false-triggers takes/*.jsonl --out reports/false-triggers.json
```

Every action over that footage is a false trigger by definition. Record several
takes in different settings before quoting the number anywhere.

**Model accuracy:**

```bash
python scripts/train_model.py
```

Writes `models/metrics.json`. Read the caveat it records: the split is random
over per-frame samples, so the accuracy is an optimistic upper bound rather than
a generalization estimate.

---

## Troubleshooting

| Symptom | Cause |
| :--- | :--- |
| Gestures show in the HUD, nothing happens | Step 2 not done, or the terminal was not restarted after granting |
| App exits complaining about an unbound gesture | Your config has no action for a pose the model predicts |
| "Camera lost — reconnecting" | Another app has the camera. The website's demo counts. |
| Black window, no video | Step 3 not done |
| Cursor moves when you meant to scroll | Fist not fully closed; all four fingertips must sit below their knuckles |
| Cursor stalls or hops instead of tracking | Run `gestureflow selftest`. If it fails, the geometric detectors have regressed — see DIAGNOSIS.md |
| Fist triggers a right click | Run `gestureflow selftest`; this was a real bug, fixed by resolving a fist to scroll before the click FSMs see it |
| Volume changes when you meant to point | Tuck your thumb; a raised thumb means volume mode |
| Poses recognized unreliably | Retrain on your own hands (step 8) |
| `gestureflow: command not found` | The venv is not active, or `pip install -e .` was not run |
| Clicking is unreliable but poses are fine | Run `gestureflow calibrate` (step 4b) |
| Nothing responds, window says PAUSED | Kill switch is on. Hold index + pinky up, middle + ring down, ~1.5 s |
| A long pinch drags when you wanted a click | Release sooner, or raise `DRAG_HOLD_SECONDS` |
| Swiping also scrolls | Flick more horizontally; near-diagonal motion deliberately fires neither |
| Profiles never switch | Automation permission declined — see step 5b |
| Zoom fires Spotlight instead | A wide L-shape overlaps the trained L-Shape pose; rebind one of them |
