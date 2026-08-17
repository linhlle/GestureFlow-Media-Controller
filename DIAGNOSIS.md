This file holds two investigations:

1. **[Cursor barely moves, scroll dead](#diagnosis-cursor-barely-moves--jitters-scroll-completely-dead)**
   — the desktop app (`gestureflow run`).
2. **[Browser demo freezes when a hand enters frame](#diagnosis-2-browser-demo-freezes-when-a-hand-enters-frame)**
   — the web demo (`web/demo.html`).

---

# Diagnosis: cursor barely moves / jitters, scroll completely dead

Investigated against the real desktop path (`gestureflow run`), before any code
changed. Every claim below is backed by a measurement over the 941 recorded
landmark frames in `data/gesture_data.csv`, or by reading the code path.

**Both symptoms share a single root cause**, plus a second, independent cursor
defect. The right-click-on-fist report in Part 3 is the same root cause again,
which is why all three are fixed together.

---

## Method

`data/gesture_data.csv` stores *normalized* landmarks (wrist-relative, divided
by the largest absolute component), while the live pipeline feeds MediaPipe's
*raw* landmarks to the geometric predicates. Comparing them directly would be
misleading, because the `0.04` y-margins mean something different in each space.

So the measurements below re-inflate each row back to a raw-looking hand:
`raw = 0.5 + normalized * s`, where `s` is the hand's extent in image
coordinates. Results are reported across `s = 0.15 … 0.45`, which brackets a
hand from far away to close up. The conclusions hold across the whole range; the
headline numbers use `s = 0.30`.

---

## Root cause 1 — `_thumb_raised` is a false positive generator

[`gestureflow/scroll_fsm.py`](gestureflow/scroll_fsm.py):

```python
def _thumb_raised(landmarks, margin=0.04):
    return landmarks[_THUMB_TIP].y < landmarks[_THUMB_MCP].y - margin
```

It compares the thumb **tip (4)** against the thumb **MCP (2)** only. That asks
"is the tip higher than the knuckle", which is true for almost any hand posture
that is not actively pointing the thumb downward — including a closed fist,
where the thumb folds *across* the curled fingers and its tip ends up well above
its own MCP.

Measured over all 941 real frames:

| Pose | `_thumb_raised` fires |
| :--- | ---: |
| Neutral | **79%** |
| L-Shape | 13% |
| High-Five | 46% |
| 2-Finger | 100% |

It is not a detector. On the pose that matters most — Neutral, the state in
which every geometric mode lives — it is true four frames in five.

### Consequence A: scroll is structurally impossible

`_is_true_scroll_fist` gates on it ([`scroll_fsm.py`](gestureflow/scroll_fsm.py)):

```python
def _is_true_scroll_fist(landmarks):
    if _index_extended(landmarks):  return False
    if _thumb_raised(landmarks):    return False   # <-- vetoes real fists
    return _strict_fist(landmarks)
```

Of the 64 frames that are genuine scroll fists (`_strict_fist` true and index
not extended), **57 are vetoed by `_thumb_raised` — 89%**.

That 89% per-frame veto is not merely a reduced hit rate, it is fatal.
`ScrollFSM` requires `min_hold_frames = 5` **consecutive** passing frames to
reach `SCROLLING`. At an 11% per-frame pass rate the probability of five in a
row is `0.11^5 ≈ 1.6e-5`. The FSM never leaves `FIST_DETECTED`, so
`scroll_delta` is never computed and no `Scroll` action is ever produced.

**The rest of the scroll path is fine.** I traced it end to end and found no
second fault:
- `GestureRouter.route` does emit `act.Scroll` when `scroll_delta != 0`
  ([`app.py`](gestureflow/app.py)).
- `Scroll` is a discrete action, so it goes to the FIFO queue, never the
  coalescing register — it cannot be dropped by the cursor drop policy
  ([`actions.py`](gestureflow/actions.py)).
- `_dispatch` routes `Scroll` to `controller.scroll()`, which calls
  `pyautogui.scroll` ([`controller.py`](gestureflow/controller.py)).
- The wrist anchor `_prev_wrist_y` is updated every frame including during
  cooldown, so it does not stall.

Scroll was never lost in the move to the action thread. It simply never fires,
because the gate never opens.

### Consequence B: cursor mode is disabled most of the time

`cursor_enabled` in [`app.py`](gestureflow/app.py):

```python
if result.thumb_raised:
    # Thumb up means the user is reaching for volume, not pointing.
    return False
```

**This gate is a regression I introduced.** The original `_handle_mouse` at
commit `21db12b` gated only on `fsm_active`, `right_fsm_active`, `scroll_active`
and `index_extended` — there was no thumb check. I added it during the
`GestureRouter` extraction, and combined with the broken detector it disables
cursor mode on roughly 75% of frames.

It is also redundant: `volume_enabled` already requires `not index_extended`,
and `cursor_enabled` requires `index_extended`. The two modes were already
mutually exclusive without it.

### Consequence C: a fist is read as a right-click

`ClickFSM` for right-click watches middle tip (12) against index tip (8) with an
**absolute** threshold of `0.045`. In a closed fist those two fingertips sit
side by side and curled, so they are naturally close.

Measured on the same 64 fist frames:

| | value |
| :--- | ---: |
| median middle↔index distance | 0.0515 |
| minimum | 0.0257 |
| **frames below the 0.045 threshold** | **14 / 64 (22%)** |

The pair drifts in and out of the close threshold during a fist, and
`min_hold_frames = 5` is enough to latch on a run of them. Making a fist fires a
right-click — exactly as reported.

The threshold is also scale-dependent: the median hand scale (wrist → middle
MCP) is `0.159`, so `0.045` is **0.28 × hand scale** for this user at this
distance. A smaller hand, or the same hand further from the camera, crosses it
constantly. Every geometric threshold in the codebase has this problem — they
are all absolute distances in an image-normalized space that itself varies with
how far away the hand is.

---

## Root cause 2 — the cursor filter is fed irregular `dt`

Independent of the gating, `move_mouse_smooth` in
[`controller.py`](gestureflow/controller.py) computes:

```python
dt = max(0.0, now - self._last_move_time)
alpha = 1.0 - math.exp(-dt / self._tau)
```

`_last_move_time` is stamped when the **action thread dispatches a move**, not
when the frame was captured. So `dt` is the gap between successive dispatches —
and when cursor mode flickers on and off, that gap is wildly irregular.

The filter's step size is extremely sensitive to it (`tau = 0.15`):

| gap between dispatches | alpha | effect |
| ---: | ---: | :--- |
| 16 ms | 0.10 | crawl |
| 33 ms | 0.20 | crawl |
| 100 ms | 0.49 | large step |
| 500 ms | 0.96 | jumps straight to the target |

Simulating a 1000 px hand sweep with the gate passing one frame in four — the
rate measured above:

| | dispatches | steps > 40 px | max step |
| :--- | ---: | ---: | ---: |
| gate stable (correct) | 40 | 0 | 25 px |
| gate flickering (measured) | 10 | **9** | **100 px** |

That is the reported symptom precisely: the pointer makes a handful of large,
irregular hops instead of many small smooth ones. **"Barely moves" and "very
jittery" are the same fault seen from two angles** — few dispatches means little
total travel, and each dispatch after a gap is an oversized jump.

A second-order issue: even with perfect gating, a fixed-`tau` exponential filter
cannot both suppress landmark noise when the hand is still and stay responsive
when it moves fast. `tau = 0.15` is tuned for the former and is visibly laggy
for the latter.

### Ruled out

- **Coalescing is not the cause.** I traced `ActionDispatcher.run` /
  `_drain`. Every `submit` sets the `_work` event, and the event is cleared
  *before* draining, so a submit racing with a drain leaves the flag set and the
  next `wait` returns immediately. There is no lost-wakeup path that starves
  motion, and the dispatcher performs exactly one move per wake. Coalescing does
  collapse bursts to the newest target, which is correct — the newest target is
  the only one that matters.
- **The landmark→screen mapping is intact.** `np.interp(ix, (100, 540), (0,
  screen_w))` still spans the full screen. (One cosmetic change: the original
  mapped y to `screen_h + 50` to make the bottom edge reachable; I dropped the
  `+50`. Not a cause, but restored.)

---

## Part 4 — earlier fixes verified still in place

Re-checked by execution, not inspection:

| Fix | State |
| :--- | :--- |
| `monotonic()` epoch — timers seeded to `-inf` | OK (click, debouncer, scroll all `-inf`) |
| Scroll float rounding — `0.500-0.480` → 2 clicks, not 3 | OK |
| Capture keeps the freshest frame | OK (queue holds newest) |
| `self._stop` no longer shadows `Thread._stop` | OK (all three thread classes) |
| Cursor no longer flung on the gesture-fire frame | OK (`action is not None` check present) |

None regressed.

---

## Confirmed after the fix

Same measurement, same 941 frames, same method:

| | before | after |
| :--- | ---: | ---: |
| `_thumb_raised` on Neutral | 79% | **5%** |
| genuine scroll fists vetoed by the thumb gate | 89% | **0%** |
| genuine scroll fists that arm the gate | 7 / 64 | **53 / 53** |
| probability of 5 consecutive passing frames | ~1.6e-5 | **1.000** |
| cursor mode enabled over a 40-frame pointing sweep | ~10 / 40 | **40 / 40** |
| simulated 12 px landmark jitter reaching the pointer | 6.2 px | **2.6 px** |
| distance covered in a 1200 px half-second sweep | 893 px | **1110 px** |

The last two are the jitter/lag trade-off, and they moved in opposite
directions from each other, which is the point: the old filter could only trade
one for the other, and it was losing on both.

`gestureflow selftest` reproduces the top four rows on demand with no camera,
and exits non-zero if the scroll gate ever regresses to a veto rate that makes
the FSM's consecutive-frame requirement unreachable.

Note the 2-Finger pose still reads `thumb_raised` 93% of the time. That is
correct — that pose genuinely extends the thumb — and it is harmless, because a
classified command pose suppresses every geometric mode anyway.

---

## Fixes applied

1. **Hand-scale normalization.** `hand_scale(landmarks)` = distance from wrist
   (0) to middle MCP (9). Every geometric threshold and margin is now expressed
   as a ratio of it, so detection no longer depends on hand size or camera
   distance. Defaults were converted from the old absolute values using the
   measured median scale of `0.159`, so behaviour at the previous working
   distance is preserved.

2. **`_thumb_raised` rewritten** to require a *straight, upward* thumb — tip
   above IP above MCP, and the tip clear of the index MCP — rather than merely
   tip-above-knuckle. A folded fist thumb fails the straightness test. Measured
   on real data: Neutral false positives **79% → 5%**, and fists vetoed
   **89% → 0%**.

3. **Thumb gate removed from `cursor_enabled`**, restoring the original
   behaviour. Mode exclusivity is preserved by the existing
   `index_extended` / `not index_extended` split, and the property test proves it.

4. **A fist suppresses both click FSMs.** Fist detection is resolved once per
   frame in `InferenceThread._process`; when it holds, both pinch FSMs are fed
   `None`. A fist can now only ever mean scroll.

5. **Cursor filter replaced with a One Euro filter.** A fixed-`tau` low-pass
   forces a choice between jitter and lag. One Euro adapts its cutoff to hand
   speed: heavy smoothing when the hand is nearly still (kills landmark jitter),
   light smoothing when it moves fast (kills lag). `dt` is also clamped so a gap
   in the stream can never produce a teleport.

6. **`dt` is derived from the capture timestamp**, not the dispatch time, so the
   filter sees the interval over which the hand actually moved rather than
   however long the action thread took to get to it.

---

## Python ↔ JS recognizer divergence

Audited both. Before this change the only divergence was that the JS
`Recognizer` did not implement the click-suppression-during-command rule the way
`GestureRouter` does; the numeric parity suite did not cover it because it tests
the FSMs individually. All six fixes above are mirrored in
`web/js/recognizer.js`, and the parity suites (`tests/test_web_parity.py`
textual, `tests/test_js_parity.py` numeric under Node) were extended to cover
hand-scale normalization, the new thumb test, and fist-suppresses-click.

---

## Verifying with a real camera

I cannot run the camera or OS control path. These are the commands to confirm
the fixes on real hardware — see the bottom of this file's companion section in
`SETUP.md` for permissions first.

```bash
# 1. Confirm the detectors now behave, with no camera needed.
python -m gestureflow selftest

# 2. Cursor: point your index finger up and move it around the frame.
#    Expect smooth continuous tracking, no stalling, no large hops.
python -m gestureflow run

# 3. Scroll: make a closed fist (thumb folded, not sticking up), hold it
#    still for about a quarter second until the HUD shows SCROLL, then move
#    your whole hand up and down.
#    Expect the active window to scroll, and the HUD to read SCROLL UP/DOWN.

# 4. Confirm a fist no longer right-clicks: hold a fist for several seconds.
#    Expect the HUD to show SCROLL and never RIGHT CLICK.

# 5. Measure it, rather than trusting the feel:
python -m gestureflow record takes/cursor.jsonl --label cursor --seconds 20
python -m gestureflow record takes/fist.jsonl   --label scroll --seconds 20
python -m gestureflow replay takes/cursor.jsonl takes/fist.jsonl -v
#    Expect MoveCursor actions from the first take and Scroll from the second.

# 6. Check the cursor is dispatching every frame, not in bursts:
python -m gestureflow bench --seconds 20
#    In the report, action.dispatch count should be close to
#    inference.predict count while you hold a pointing hand in frame.
#    A large gap means moves are still being coalesced away.
```

---
---

# Diagnosis 2: browser demo freezes when a hand enters frame

Reported symptom: `web/demo.html` lags or freezes the camera feed **when a hand
enters the frame**; with no hand visible it is smooth. This is the JS/WASM demo
path, not the Python desktop app.

The reported symptom is a **crash**, not a performance problem. The hypothesis
going in was that hand-present frames do far more work (landmark model + forest
+ skeleton draw) and starve the main thread. That hypothesis is measurably
wrong, and the measurements are below. What actually happened is that
`Recognizer.process` threw a `ReferenceError` on the first frame containing a
hand, and the render loop was structured so that a single exception stopped it
permanently.

---

## Method

Numbers come from the real demo page driven in a real Chrome, not from a
re-implementation of the loop in a test harness — the bug is about *when* work
happens on the main thread, and a re-implementation would just reproduce
whichever scheduling the harness author had in mind.

- `scripts/make_fake_camera.py` writes two Y4M clips: a hand filling the frame
  and drifting, and the same room empty. Chrome plays these in place of a
  webcam (`--use-file-for-fake-video-capture`), so every run sees a
  byte-identical input stream. A real camera cannot give that, and without it
  before/after numbers are not comparable.
- `scripts/webbench.mjs` launches Chrome, attaches over the DevTools protocol,
  and injects instrumentation *from outside* so the page's own source is
  unmodified: `requestAnimationFrame` (per-callback main-thread cost),
  `HandLandmarker.detectForVideo` (MediaPipe solve time),
  `Recognizer.process` (forest + FSM time), `drawImage` of a video element
  (when a camera frame actually reaches the screen), and a `longtask` observer.
- ES modules are singletons per URL, so importing the same MediaPipe and
  recognizer URLs the page imports returns the very objects it will use.
  Patching those prototypes before the camera starts instruments the real
  instances.
- Warm-up is discarded: the run waits until the page has actually painted, then
  waits again, then measures for 9 s. Model download and WASM compile are
  one-time costs that would otherwise dominate every percentile.

All numbers are 640×480 at 30 fps on one machine (Intel Iris Plus 655, macOS).
They describe that machine and that camera resolution, nothing else.

---

## Root cause — `swipeArmed` and `zoomActive` are never declared

[`web/js/recognizer.js`](web/js/recognizer.js), in `Recognizer.process`:

```js
const scrollActive = !suppressed && this.scrollFSM.isActive;   // declared
const cursorActive = ...                                       // declared
const volumeActive = ...                                       // declared

return {
  ...
  swipeArmed,     // <- never declared
  zoomActive,     // <- never declared
  mode: modeOf({ ..., swipeArmed, zoomActive, ... }),
};
```

`scrollActive`, `cursorActive` and `volumeActive` are bound a few lines above.
`swipeArmed` and `zoomActive` are not bound anywhere in the function. ES modules
are always strict mode, so reading an unbound identifier is a `ReferenceError`,
not `undefined`.

Introduced in `128de86` ("Wire the new gestures into the browser demo, builder
and docs"), which added both names to the returned object and to the `modeOf`
call — and no `const` for either.

### Why only with a hand

`process()` returns early through `emptyResult()` when there are no landmarks,
and `emptyResult` sets `swipeArmed`/`zoomActive` as *object properties*, which
is legal. The unbound identifiers are only reached on the path that has a hand:

| Frame | Path | Result |
| :--- | :--- | :--- |
| No hand | early return via `emptyResult()` | fine |
| Hand | falls through to the `return {...}` at the end | **throws** |

### Why a throw froze the picture rather than logging an error

Two structural facts turn one bad frame into a permanent freeze.

**The loop rescheduled itself on its last line.**
[`web/js/demo.js`](web/js/demo.js), before this change:

```js
function loop() {
  if (!running) return;
  if (/* new video frame */) {
    const detection = handLandmarker.detectForVideo(els.video, nowMs);
    const result = recognizer.process(landmarks, nowMs / 1000);   // throws here
    draw(result);
    ...
  }
  requestAnimationFrame(loop);        // never reached
}
```

Nothing else drives the loop, so the first escaping exception ends it for good.

**The `<video>` element is invisible.** [`web/css/style.css`](web/css/style.css):

```css
.stage video { visibility: hidden; }
```

The canvas is the only thing the user sees, and the canvas is only repainted
inside the loop. A dead loop therefore does not look like an error — it looks
exactly like the camera froze.

### Why the test suite did not catch it

`scripts/parity_check.mjs` imported `ClickFSM`, `ScrollFSM`, `SwipeFSM`,
`ZoomFSM`, `PauseFSM`, `GestureDebouncer` and every geometric predicate, and
exercised each one individually. It never constructed `Recognizer`. The single
function the demo page actually calls had no coverage in either parity suite.

This is the same gap recorded further up this file, where the JS recognizer's
click-suppression rule diverged and "the numeric parity suite did not cover it
because it tests the FSMs individually". The gap was noted then and not closed.
It is closed now.

---

## The performance hypothesis, measured and rejected

With the crash fixed, per work frame (mean, hand present vs absent):

| Stage | Hand present | No hand | Share of frame |
| :--- | ---: | ---: | ---: |
| MediaPipe solve | 22.34 ms | 20.13 ms | **97.5%** |
| Recognizer (forest + all FSMs) | 0.07 ms | 0.01 ms | 0.3% |
| Canvas draw + DOM panels + log | ~0.5 ms | ~0.5 ms | 2% |

And end to end:

| | Hand present | No hand |
| :--- | ---: | ---: |
| Presented fps | 30.0 | 29.5 |
| Present gap p50 / p95 | 33.5 / 48.1 ms | 34.0 / 49.4 ms |

Conclusions, none of which match the hypothesis:

- **The forest is not the cost.** 0.07 ms, roughly 0.3% of a frame. The 100-tree
  forest over 63 features is nothing next to the landmark model.
- **The skeleton draw is not the cost.** ~0.5 ms for 21 arcs, 21 lines, the
  charge arcs, the panels and the event log combined.
- **A hand costs 2.2 ms more, not "far more".** About 11%, which is MediaPipe
  additionally running the landmark model once the palm detector has found
  something.
- **There is no hand-present smoothness penalty at all.** 30.0 fps with a hand
  against 29.5 without, and the gap distributions overlap. Both sit at the
  camera's own 30 fps, which is the ceiling.

So the freeze was the `ReferenceError`, in full. No part of the reported symptom
is explained by main-thread load.

## Backend actually in use

Probed in the page rather than assumed:

| Question | Answer |
| :--- | :--- |
| GPU delegate | **Active.** `delegate: 'GPU'` initialises; MediaPipe logs `Graph successfully started running` with no fallback warning |
| WebGL backend | `ANGLE (Intel, ANGLE Metal Renderer: Intel(R) Iris(TM) Plus Graphics 655)` |
| WASM build loaded | `vision_wasm_internal.wasm` — the **SIMD** build, not `vision_wasm_nosimd_internal.wasm` |
| WASM threads | Supported by the browser, but `crossOriginIsolated` is **false**, so the threaded pool is unavailable |

The demo is already on the best backend available to it. There is no CPU/no-SIMD
fallback to escape from. Enabling WASM threads would need COOP/COEP headers,
which would also have to be reconciled with the cross-origin CDN imports; given
that presented fps already sits at the camera's own rate, there is no headroom
that change could recover, so it was not made.

---

## Fixes applied

1. **Declare `swipeArmed` and `zoomActive`** ([`web/js/recognizer.js`](web/js/recognizer.js)),
   in the order the mode ladder consults them and matching the `!suppressed`
   convention of the three flags beside them. `modeOf` checks `suppressed`
   first, so the resolved `mode` is identical either way.

2. **Cover `Recognizer.process` in the parity harness**
   ([`scripts/parity_check.mjs`](scripts/parity_check.mjs),
   [`tests/test_js_parity.py`](tests/test_js_parity.py)) — one sequence per rung
   of the mode ladder, asserting that every frame produces a result, that no
   field the demo reads is left `undefined`, that `mode` is always a real mode,
   and that the JS ladder resolves the same flags to the same mode as
   `GestureRouter.active_mode`. Verified to fail against the pre-fix file.

3. **Reschedule the loop from `finally`** ([`web/js/demo.js`](web/js/demo.js)),
   so no future exception can stop the page requesting frames.

4. **Paint the camera frame before inference, not after.** This one came out of
   the control test below rather than from reading the code, and it is the part
   that actually keeps the video moving.

5. **Hoist a per-frame `getComputedStyle`.** The accent colour was read inside
   the hand-present branch of `draw()`, forcing a style recalculation every
   frame containing a hand to fetch a constant. Read once at start.

6. **Report video and recognizer rates separately** in the demo's readout, so
   the two can be seen to diverge instead of being assumed equal. Measured live
   in the tab; no number is hard-coded anywhere.

### Not done, and why

- **Web Worker / OffscreenCanvas.** Would genuinely move the 22 ms solve off the
  main thread, but presented fps already equals the camera's own frame rate, so
  there is no throughput to recover. It is a large change with real parity risk
  against a recognizer that must stay byte-identical to the Python one.
- **Landmark interpolation between inference results.** Only pays off when
  inference runs slower than presentation. They run 1:1 here, both at 30 fps.
- **Inference backpressure / frame skipping.** `detectForVideo` is synchronous,
  so a second solve cannot begin before the first returns. There is no queue to
  bound and nothing to drop; the "only one in flight" property is already
  structural.
- **Inferring on a downscaled frame.** Would trade recognition quality for
  headroom that is not currently needed.

---

## Control test: the freeze mechanism, reproduced and fixed

`scripts/webbench.mjs --break-recognizer` makes `Recognizer.process` throw the
original `ReferenceError` after 100 good frames, which is the fault in
isolation. Run against three versions of the loop, hand present:

| Loop | Faults thrown | Solve fps | **Presented fps** |
| :--- | ---: | ---: | ---: |
| Original — reschedule on last line | **1** | 0 | **0** |
| `finally` reschedule, paint *after* inference | 316 | 29.9 | **0** |
| `finally` reschedule, paint *before* inference | 311 | 30.1 | **30.1** |

("Solve fps" is MediaPipe's `detectForVideo` rate as counted by the harness. The
page's own readout would correctly show `30 fps video` with no recognizer figure
in rows 2 and 3, since nothing gets past the throw.)

The first row is the reported bug, reproduced exactly: **one** exception, and
the page never renders again.

The second row is the one that changed the fix. Rescheduling from `finally` kept
the loop alive — 316 faults absorbed, MediaPipe still solving at 29.9 fps — and
the picture was *still frozen*, because `draw()` sat downstream of
`recognizer.process()`. Alive and visibly frozen is the same thing to a user.
I would not have found this by reading the code; it took injecting the fault and
watching presented fps stay at zero.

The third row is the fix: with the camera frame painted before inference is
attempted, 311 consecutive recognizer failures cost the overlay and the
readouts, and cost the video nothing.

---

## Confirmed after the fix

Same fake-camera clips, same machine, 640×480 at 30 fps, hand present:

| | Before | After |
| :--- | ---: | ---: |
| Presented fps | **0** — loop dead on the first hand frame | **30.0** |
| Recognizer fps | 0 | 30.0 |
| Present gap p50 / p95 | — (nothing presented) | 33.3 / 46.2 ms |
| MediaPipe solve p50 / p95 | — | 24.8 / 29.9 ms |
| Recognizer p50 | — | 0.1 ms |
| Long tasks in a 9 s window | — | 0 |

"Before" is not a slow number, it is the absence of one: with the shipped
`recognizer.js`, the loop threw on the first frame containing a hand and never
presented another. That is the whole reported symptom.

With no hand in frame, after the fix: 29.9 fps presented, solve p50 17.5 ms.
The hand-present and hand-absent cases are both at the camera's own rate, which
is the ceiling for this input.

Run-to-run variation in solve time across the five runs collected here is
±3 ms at p50, comfortably larger than any difference the demo-side changes could
account for — those changes remove roughly 0.5 ms of style recalculation per
hand-present frame. **No claim is made that the fixes made MediaPipe faster.**
They made the demo run at all, and made it keep running when something throws.

### Reproducing

```bash
# One-time: fetch a hand photo MediaPipe recognises, build the clips.
python scripts/make_fake_camera.py --src <dir-with-hand.jpg> --out bench/y4m

# Measure the demo as it stands.
node scripts/webbench.mjs --y4m bench/y4m --label now --out bench

# Reproduce the freeze mechanism: fault the recognizer, watch presented fps.
node scripts/webbench.mjs --y4m bench/y4m --label fault --out bench \
  --break-recognizer
```

Needs Google Chrome and Node. Chrome is launched against a throwaway profile and
killed afterwards; nothing touches the user's own browser.
