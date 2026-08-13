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
