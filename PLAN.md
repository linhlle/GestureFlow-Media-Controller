# GestureFlow — feature plan

Written after re-reading the codebase at `9409ffe` (388 tests green). Supersedes
the original build plan, which is now history — see `git log`.

The last two rounds both broke working features by changing several interacting
things at once. So the organising principle here is: **one detector per phase,
its own module, its own tests, suite green before the next phase starts.** No
phase touches another phase's code.

---

## The invariant everything must respect

Cursor, scroll, and volume are three readings of the same hand, kept apart today
by a set of independent predicates in `GestureRouter`:

```python
def cursor_enabled(...):  # not suppressed, not pinching, not scrolling, index up
def scroll_enabled(...):  # not suppressed, scroll_active
def volume_enabled(...):  # not suppressed, not scrolling, index down, thumb up
```

Exclusivity is *emergent* — it holds because each predicate happens to exclude
the others, and a property test checks that it does. Adding five more modes to
that structure means every new predicate must exclude every existing one, which
is `n²` chances to get it wrong and exactly how the thumb-gate regression
happened.

**Phase 0 replaces this with a single precedence ladder.** One function returns
one mode; the `*_enabled` predicates become `mode is Mode.X` wrappers. Then
exclusivity is *structural* — two modes cannot both be active because the
function returns one value — and the property test degenerates to "the ladder
returns something", which cannot regress.

Precedence, highest first. The order encodes intent: safety, then explicit
commands, then whole-hand poses, then fine-grained ones.

| # | Mode | Claimed when |
| --: | :--- | :--- |
| 1 | `NONE` | no hand in frame |
| 2 | `PAUSED` | the kill switch is engaged |
| 3 | `COMMAND` | classifier settled on a pose, or an action fired this frame |
| 4 | `SCROLL` | fist held, motion mostly vertical |
| 5 | `SWIPE` | fist held, motion mostly horizontal |
| 6 | `ZOOM` | thumb+index spread pose, armed |
| 7 | `DRAG` | left pinch held past the drag threshold |
| 8 | `CLICK` | either pinch FSM pressing or held |
| 9 | `VOLUME` | index down, thumb up and clear |
| 10 | `CURSOR` | index extended |
| 11 | `TRACKING` | hand visible, nothing claimed it |

Phase 0 ships with the ladder returning exactly today's answers — the five new
modes are added later, each in its own phase. That is deliberate: it lets the
existing 388 tests prove the refactor changed nothing before any new behaviour
lands.

---

## Ships now — geometric and temporal, no retraining

Every detector below reads landmark geometry or its motion over time. None needs
a single new training sample.

### Phase 0 — Mode ladder (refactor, no behaviour change)

- `Mode` enum, `GestureRouter.active_mode()`.
- `cursor_enabled` / `scroll_enabled` / `volume_enabled` become wrappers.
- **Tests:** the entire existing suite must pass untouched. Plus a new
  property test asserting `active_mode()` is total (never raises, always
  returns a `Mode`) over arbitrary landmark configurations, and that each
  wrapper agrees with the ladder.

### Phase 1 — Pause / resume kill switch

A safety feature, so the priority is *never fires by accident*, ahead of *easy
to fire*.

**Pose: index and pinky extended, middle and ring curled** — "rock horns".

Not the open palm the brief suggested, for a concrete reason: open palm is the
model's `High-Five` class, bound to Mission Control. Holding it for 1.5 s would
fire Mission Control once or twice on the way to pausing, because the
debouncer's command cooldown is 1.3 s. Rock horns is not one of the four trained
classes, needs four simultaneous conditions, and is close to impossible to strike
accidentally while typing or gesturing.

It also avoids the thumb entirely. Thumb geometry has already produced one
silent, total failure in this codebase (`_thumb_raised` firing on 79% of
frames); a safety switch should not depend on the least reliable landmark.

- `gestureflow/pause_fsm.py`: `PauseFSM`, hold timer, latch, progress for the HUD.
- Toggling **consumes the frame** — no other action dispatches on it.
- While paused: capture, inference, and the HUD keep running; the dispatcher
  receives nothing. A large `PAUSED` banner in the HUD.
- Runs *above* the command-suppression gate, so it works even if the classifier
  misreads the pose.
- **Tests:** holds for the full duration before toggling; a single dropped frame
  resets the timer; the pose must be released before it can toggle again;
  no action of any kind is emitted while paused; the toggle frame emits nothing
  else. Property test: for any landmark sequence, `paused` only changes on a
  frame where the pose has been continuously held for the configured time.

### Phase 2 — Drag and drop

Extends `ClickFSM` with a `DRAGGING` state:

```
IDLE -> PRESSING -> HELD --(release)--------------> click        (unchanged)
                     |
                     +--(held past drag threshold)-> DRAGGING
                                                        |
                                                        +--(release)--> drop
```

A short pinch-and-release is still a click, bit for bit. Only holding *past* the
threshold enters the drag, and the threshold is measured from when `HELD` was
reached, so it composes with the existing hold-frames requirement rather than
replacing it.

- New actions `MouseDown` / `MouseUp`; `SystemController` gains
  `mouse_down()` / `mouse_up()`.
- **Tests:** the existing click tests must pass unchanged. New: a long hold
  emits exactly one press and, on release, exactly one release; a short hold
  still emits a click and never a press; releasing mid-drag always emits the
  release (a drag must never be left stuck down); hand leaving frame during a
  drag releases the button — otherwise walking away leaves the mouse held.

### Phase 3 — Swipe left / right

Same fist as scroll, arbitrated by axis. Scroll is vertical, swipe is
horizontal, and a shared dominance rule decides which one a given motion is.

Doing it any other way would need a second whole-hand pose, and the ones left
over are either trained classes or too close to the fist to separate reliably.
Arbitration also means the two can never both fire on one frame, which is the
property that matters.

- `gestureflow/swipe_fsm.py`: arms on the same fist gate, requires
  `|dx| > ratio * |dy|` and `|dx| > sensitivity`, fires once, then requires the
  hand to slow below a release threshold before it can fire again.
- `ScrollFSM` gains the complementary guard (`|dy| >= ratio * |dx|`).
- Bindable as named gestures `swipe_left` / `swipe_right`. Default binding:
  `ctrl+left` / `ctrl+right` (macOS desktop switching).
- **Tests:** pure vertical motion produces byte-identical scroll output to
  today (guards the `ScrollFSM` change); pure horizontal produces swipe and zero
  scroll; diagonal resolves to exactly one; one continuous sweep fires exactly
  once regardless of how many frames it spans; direction is correct in both
  senses.

### Phase 4 — Dwell click (accessibility)

For users who cannot pinch. **Off by default** — it turns resting the pointer
into a click, which is surprising unless asked for.

- `gestureflow/dwell_fsm.py`: tracks the cursor target; if it stays within
  `radius_px` for `seconds`, emit one click, then require the pointer to leave
  the radius before re-arming.
- Only runs while the ladder says `CURSOR`, so it cannot collide with pinching.
- **Tests:** fires after the configured time; does not fire if the pointer
  leaves the radius; fires exactly once per dwell; re-arms only after leaving;
  disabled by default and emits nothing when disabled.

### Phase 5 — Zoom

Thumb-index spread, as asked, but *armed* like scroll rather than instantaneous,
because the same two landmarks already mean "left click" when they are close
together.

The two ranges are kept disjoint by construction: the click FSM fires below
`close_threshold`, zoom only arms above `open_threshold` and requires
middle/ring/pinky curled. Once armed, `ZOOM` outranks `CLICK` and `CURSOR` in
the ladder, so nothing else can fire while zooming.

- `gestureflow/zoom_fsm.py`: pose held N frames → `ZOOMING`; changes in
  normalized thumb-index distance emit `zoom_in` / `zoom_out` steps with a
  cooldown.
- Bindable as `zoom_in` / `zoom_out`. Defaults `command+=` / `command+-`.
- **Tests:** spreading zooms in, closing zooms out; does not arm while the pinch
  is closed; a click gesture never emits a zoom and vice versa; arming
  suppresses cursor.

---

## New action types — no new gestures

Existing gestures gain more to do. All schema-validated against closed sets;
no `eval`, no arbitrary strings reaching a shell.

`media` already exists (`playpause`, `next`, `previous`, `volumeup`,
`volumedown`, `mute`) and is unchanged. Adding:

| Type | Shape | Validation |
| :--- | :--- | :--- |
| `url` | `{type: url, url: "https://..."}` | scheme must be `http`/`https`; length capped. `javascript:`, `file:`, `data:` rejected |
| `text` | `{type: text, text: "..."}` | printable characters only, length capped, typed rather than pasted so the clipboard is untouched |
| `chord` | `{type: chord, steps: [{keys: [...], delay: 0.1}, ...]}` | each step validated as a hotkey; step count and delay both capped |

- Ship an updated `configs/commands.default.yaml` that reproduces today's three
  bindings **exactly**, plus the new named-gesture defaults.
- Schema version goes to 2; version 1 configs still load unchanged.
- **Tests:** every new type round-trips through YAML and JSON; injection-shaped
  inputs rejected (`javascript:` URLs, control characters in text, oversized
  chords); the shipped default still produces the identical three bindings a
  version-1 config did.

---

## Bigger features

### A. Per-user calibration — **building this**

Thresholds are already fractions of hand scale, which fixed *distance from
camera*. What it did not fix is that people's hands differ: one person's closed
pinch is 0.20 of their hand scale, another's is 0.35. A single default cannot
suit both.

- `gestureflow calibrate`: guided CLI. Records an open hand, then a closed
  pinch, then a comfortable rest pose. Computes the user's median hand scale and
  the distribution of pinch distances in each state, and picks thresholds in the
  gap between them (with hysteresis preserved).
- Writes `~/.gestureflow/calibration.json`.
- `AppConfig` loads it when present, falls back to today's defaults when absent
  or malformed. Calibration never makes the app fail to start.
- **Tests:** threshold derivation from synthetic distance distributions;
  refuses to write nonsense (open pinch below closed pinch, too few samples);
  missing/corrupt file falls back silently to defaults; a calibrated config
  round-trips.

### B. App-context profiles — **planned, built if Phase A lands clean**

- Schema gains `profiles:` with `match:` rules on the frontmost app's bundle id
  or name, each carrying its own `gestures:` list. A `default` profile is
  required.
- `gestureflow/frontmost.py`: pluggable provider. The macOS implementation
  shells out to `osascript` on a background thread with a cache, so the render
  path never blocks on it. Tests use a fake provider, so the whole resolution
  layer is testable with no OS involvement.
- Hot-reload continues to apply; switching apps swaps the active `CommandSet`.
- **Tests:** match rules resolve to the right profile; unknown app falls back to
  `default`; a profile inherits the default's bindings unless it overrides them;
  the provider failing (no permission, osascript missing) degrades to `default`
  rather than breaking.

**Why A first:** calibration fixes something already known to be wrong for
anyone whose hands differ from the author's. Profiles add reach but nothing is
broken without them.

---

## Requires you at a camera — not built

Listed so the split is explicit. None of the phases above depends on any of it.

| Idea | Why it needs you |
| :--- | :--- |
| New ML poses (OK sign, pointing left/right, thumbs-down) | Needs a few hundred labelled frames each and a retrain |
| Two-handed gestures | MediaPipe is capped at one hand here; needs a config change *and* new data |
| Dynamic gestures (wave, circle) | The model classifies single frames; needs a temporal model and sequence data |
| Verifying every detector above on real hands | I cannot open the camera |

---

## MANUAL STEPS — only you can do these

1. **Run the calibration wizard.** `gestureflow calibrate` — needs your hands
   at the camera. Everything works without it; it just fits the thresholds to
   you.
2. **Verify each new gesture on real hardware.** Exact commands are in the
   summary at the end and in `SETUP.md`. I can test the logic, not the camera.
3. **Accessibility permission for the new mouse-down/up path.** Already covered
   by the existing Accessibility grant — no new prompt expected, but drag is the
   first feature to hold a button down, so confirm it works.
4. **Automation permission** if you bind an `applescript` action, and on first
   use of `url` actions macOS may ask which browser to use.
5. **Re-deploy the website** after the builder and guide updates, from your
   Vercel account.
6. **Optional retraining** if you want any of the "requires you at a camera"
   ideas.

---

## Guardrails, run after every phase

- Full suite green.
- `gestureflow selftest` passes (detector sanity against recorded data).
- The five earlier fixes re-verified by execution: monotonic epoch seeded
  `-inf`, scroll rounding, freshest-frame capture, `_stop` not shadowing
  `Thread._stop`, no cursor fling on the gesture-fire frame.
- Python ↔ JS parity suites green; every gesture the demo visualises uses the
  same thresholds in both.
- No performance number anywhere the instrumentation did not produce.
