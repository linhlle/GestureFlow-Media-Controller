"""Numeric parity between the JS recognizer and the Python original.

test_web_parity.py compares constants textually. That catches a threshold
drifting but not the harder failure: constants that agree while the logic does
not. This runs the actual JavaScript over fixtures generated from the Python
implementations and diffs the results.

Skipped when Node is unavailable, so the suite still runs on a machine without
it -- the textual checks remain the floor, and CI installs Node so these run
there.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gestureflow.click_fsm import ClickFSM
from gestureflow.config import DEFAULT_CONFIG
from gestureflow.debouncer import GestureDebouncer
from gestureflow.inference import InferenceResult
from gestureflow.modes import Mode
from gestureflow.scroll_fsm import (
    ScrollFSM,
    _index_extended,
    _is_true_scroll_fist,
    _strict_fist,
    _thumb_raised,
    _velocity_to_clicks,
)
from gestureflow.utils import PROJECT_ROOT, normalize_landmarks

NODE = shutil.which("node")
HARNESS = PROJECT_ROOT / "scripts" / "parity_check.mjs"
FOREST_JSON = PROJECT_ROOT / "web" / "models" / "forest.json"

pytestmark = pytest.mark.skipif(
    NODE is None or not HARNESS.exists(),
    reason="node or the parity harness is unavailable",
)


def lm(rows):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in rows]


def pinch_rows(distance: float):
    """A hand of realistic scale (0.20) with the pinch pair `distance` apart.

    Scale matters now that thresholds are ratios of it: a fixture with an
    absurd hand scale would make every pinch read as closed in both languages,
    so the parity test would pass without exercising the thresholds at all.
    """
    rows = [[float(i) * 0.01, float(i) * 0.01, 0.0] for i in range(21)]
    rows[0] = [0.5, 0.75, 0.0]     # wrist
    rows[9] = [0.5, 0.55, 0.0]     # middle MCP -> hand_scale 0.20
    rows[4] = [0.5, 0.5, 0.0]
    rows[8] = [0.5 + distance, 0.5, 0.0]
    return rows


def fist_rows(wrist_y: float):
    rows = [[0.5, 0.5, 0.0] for _ in range(21)]
    rows[0] = [0.5, wrist_y, 0.0]
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        rows[mcp] = [0.5, 0.35, 0.0]
        rows[tip] = [0.5, 0.45, 0.0]
    rows[6] = [0.5, 0.40, 0.0]
    rows[2] = [0.5, 0.40, 0.0]
    rows[4] = [0.5, 0.45, 0.0]
    return rows


def open_hand_rows():
    rows = [[0.5, 0.5, 0.0] for _ in range(21)]
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        rows[mcp] = [0.5, 0.50, 0.0]
        rows[tip] = [0.5, 0.30, 0.0]
    return rows


def pointing_rows():
    rows = [[0.5, 0.6, 0.0] for _ in range(21)]
    rows[0] = [0.5, 0.8, 0.0]
    rows[8] = [0.45, 0.30, 0.0]
    rows[6] = [0.45, 0.55, 0.0]
    rows[4] = [0.75, 0.70, 0.0]
    rows[2] = [0.75, 0.66, 0.0]
    return rows


def thumb_up_rows():
    rows = fist_rows(0.5)
    rows[2] = [0.5, 0.50, 0.0]
    rows[4] = [0.5, 0.30, 0.0]
    return rows


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    """Build fixtures, run the JS harness once, return its output."""
    rng = np.random.default_rng(20240812)

    normalize_cases = [
        [[float(v) for v in row] for row in rng.uniform(-1, 2, size=(21, 3))]
        for _ in range(12)
    ]
    normalize_cases.append([[0.5, 0.5, 0.0] for _ in range(21)])   # degenerate
    normalize_cases.append(fist_rows(0.5))
    normalize_cases.append(pointing_rows())

    scroll_cfg = DEFAULT_CONFIG.scroll
    velocity_cases = [
        [v, {"sensitivity": scroll_cfg.sensitivity,
             "velocity_exponent": scroll_cfg.velocity_exponent,
             "step": scroll_cfg.step}]
        for v in [0.0, 0.005, 0.009, 0.01, 0.016, -0.016, 0.02,
                  0.5 - 0.48, 0.5 - 0.46, 0.05, -0.05, 0.123]
    ]

    predicate_cases = [
        fist_rows(0.5), open_hand_rows(), pointing_rows(), thumb_up_rows(),
        *[[[float(v) for v in row] for row in rng.uniform(0, 1, size=(21, 3))]
          for _ in range(8)],
    ]

    # At hand_scale 0.20 the close threshold is 0.056 and open is 0.082.
    # Full landmark rows are sent, not bare distances, so both sides filter the
    # exact same hand -- see the note in parity_check.mjs.
    _click_distance_runs = [
        [0.2, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2],           # a clean click
        [0.01] * 20,                                        # held, never clicks
        [0.01, 0.01, 0.07, 0.07, 0.01, 0.2],                # inside the dead band
        [0.2, 0.02, 0.2, 0.02, 0.2, 0.02, 0.2],             # chatter
        list(rng.uniform(0.0, 0.12, size=40).round(6)),      # adversarial sweep
    ]
    click_sequences = [
        [[pinch_rows(float(d)), i / 30.0] for i, d in enumerate(run)]
        for run in _click_distance_runs
    ]

    scroll_sequences = [
        [[fist_rows(0.5 - i * 0.01), i / 30.0] for i in range(25)],
        [[open_hand_rows(), i / 30.0] for i in range(10)],
        [[fist_rows(0.5), i / 30.0] for i in range(8)]
        + [[fist_rows(0.4 - i * 0.02), (8 + i) / 30.0] for i in range(12)],
    ]

    debounce_sequences = [
        [[1, 0.99, i / 30.0] for i in range(20)],
        [[1, 0.30, i / 30.0] for i in range(20)],
        [[int(v), float(c), i / 30.0] for i, (v, c) in enumerate(
            zip(rng.integers(0, 4, size=30), rng.uniform(0.5, 1.0, size=30)))],
    ]

    forest_samples = [
        [float(v) for v in row]
        for row in rng.uniform(-1, 1, size=(25, 63))
    ]

    fixtures = {
        "normalize": normalize_cases,
        "velocity": velocity_cases,
        "predicates": predicate_cases,
        "clickSequences": click_sequences,
        "scrollSequences": scroll_sequences,
        "debounce": debounce_sequences,
        "forestPath": str(FOREST_JSON) if FOREST_JSON.exists() else None,
        "forestSamples": forest_samples if FOREST_JSON.exists() else None,
    }

    path = tmp_path_factory.mktemp("parity") / "fixtures.json"
    path.write_text(json.dumps(fixtures))

    proc = subprocess.run(
        [NODE, str(HARNESS), str(path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(f"parity harness failed:\n{proc.stderr}")

    return fixtures, json.loads(proc.stdout)


# ---------------------------------------------------------------------------

class TestFeatureVectorParity:
    def test_normalization_matches(self, parity):
        fixtures, js = parity
        for i, rows in enumerate(fixtures["normalize"]):
            expected = normalize_landmarks(lm(rows))
            got = js["normalize"][i]
            assert len(got) == 63
            assert got == pytest.approx(expected, abs=1e-9), (
                f"case {i}: JS feature vector differs from Python"
            )


class TestScrollMathParity:
    def test_velocity_to_clicks_matches(self, parity):
        fixtures, js = parity
        for i, (velocity, _cfg) in enumerate(fixtures["velocity"]):
            expected = _velocity_to_clicks(velocity, DEFAULT_CONFIG.scroll)
            assert js["velocityToClicks"][i] == expected, (
                f"velocity {velocity}: JS gave {js['velocityToClicks'][i]}, "
                f"Python gave {expected}"
            )


class TestPredicateParity:
    def test_geometric_predicates_match(self, parity):
        fixtures, js = parity
        for i, rows in enumerate(fixtures["predicates"]):
            landmarks = lm(rows)
            got = js["predicates"][i]
            assert got["indexExtended"] == _index_extended(landmarks), i
            assert got["thumbRaised"] == _thumb_raised(landmarks), i
            assert got["strictFist"] == _strict_fist(landmarks), i
            assert got["isTrueScrollFist"] == _is_true_scroll_fist(landmarks), i


class TestClickFSMParity:
    def test_state_and_click_edges_match(self, parity):
        fixtures, js = parity
        for s, sequence in enumerate(fixtures["clickSequences"]):
            clock = _SeqClock()
            fsm = ClickFSM(DEFAULT_CONFIG.click, clock=clock)
            for i, (rows, t) in enumerate(sequence):
                clock.now = t
                fsm.update(lm(rows))
                got = js["clickSequences"][s][i]
                assert got["state"] == fsm.state.name, (
                    f"sequence {s} frame {i}: state {got['state']} vs "
                    f"{fsm.state.name}"
                )
                assert got["fired"] == fsm.click_fired, (
                    f"sequence {s} frame {i}: click edge differs"
                )
                assert got["progress"] == pytest.approx(fsm.hold_progress,
                                                        abs=1e-6)


class TestScrollFSMParity:
    def test_states_and_deltas_match(self, parity):
        fixtures, js = parity
        for s, sequence in enumerate(fixtures["scrollSequences"]):
            clock = _SeqClock()
            fsm = ScrollFSM(DEFAULT_CONFIG.scroll, clock=clock)
            for i, (rows, t) in enumerate(sequence):
                clock.now = t
                fsm.update(lm(rows))
                got = js["scrollSequences"][s][i]
                assert got["state"] == fsm.state.name, (
                    f"sequence {s} frame {i}: {got['state']} vs {fsm.state.name}"
                )
                assert got["delta"] == fsm.scroll_delta, (
                    f"sequence {s} frame {i}: delta {got['delta']} vs "
                    f"{fsm.scroll_delta}"
                )


class TestDebouncerParity:
    def test_votes_and_actions_match(self, parity):
        fixtures, js = parity
        for s, sequence in enumerate(fixtures["debounce"]):
            clock = _SeqClock()
            db = GestureDebouncer(
                DEFAULT_CONFIG.debounce,
                confidence_threshold=DEFAULT_CONFIG.inference.confidence_threshold,
                clock=clock,
            )
            for i, (label, conf, t) in enumerate(sequence):
                clock.now = t
                action = db.update(label, conf)
                got = js["debounce"][s][i]
                assert got["action"] == action, (
                    f"sequence {s} frame {i}: action {got['action']} vs {action}"
                )
                assert got["stable"] == db.stable_gesture, s
                assert got["score"] == db.vote_score, s


@pytest.mark.skipif(not FOREST_JSON.exists(), reason="forest.json not exported")
class TestForestParity:
    def test_browser_forest_matches_sklearn(self, parity):
        from gestureflow.app import load_model

        fixtures, js = parity
        model = load_model()
        expected = model.predict_proba(np.array(fixtures["forestSamples"]))
        for i, probs in enumerate(js["forest"]):
            assert probs == pytest.approx(expected[i], abs=1e-6), (
                f"sample {i}: the browser classifier disagrees with sklearn"
            )


class _SeqClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# Detectors added in the feature round
# ---------------------------------------------------------------------------

def horns_rows(wrist_y: float = 0.75):
    """Index and pinky up, middle and ring down."""
    scale = 0.20
    knuckle = wrist_y - scale
    rows = [[0.5, wrist_y - 0.10, 0.0] for _ in range(21)]
    rows[0] = [0.5, wrist_y, 0.0]
    rows[9] = [0.52, knuckle, 0.0]
    rows[6] = [0.45, knuckle, 0.0]
    rows[8] = [0.45, knuckle - 0.12, 0.0]
    rows[18] = [0.62, knuckle, 0.0]
    rows[20] = [0.62, knuckle - 0.12, 0.0]
    rows[12] = [0.52, knuckle + 0.10, 0.0]
    rows[13] = [0.57, knuckle, 0.0]
    rows[16] = [0.57, knuckle + 0.10, 0.0]
    return rows


def zoom_rows(spread: float = 0.20):
    scale = 0.20
    knuckle = 0.75 - scale
    rows = [[0.5, 0.65, 0.0] for _ in range(21)]
    rows[0] = [0.5, 0.75, 0.0]
    rows[9] = [0.56, knuckle, 0.0]
    rows[5] = [0.50, knuckle, 0.0]
    rows[6] = [0.50, knuckle - 0.06, 0.0]
    rows[8] = [0.50, knuckle - 0.16, 0.0]
    rows[2] = [0.52, knuckle + 0.06, 0.0]
    rows[3] = [0.52 + spread * 0.4, knuckle + 0.05, 0.0]
    rows[4] = [0.52 + spread, knuckle + 0.04, 0.0]
    for tip, mcp in ((12, 9), (16, 13), (20, 17)):
        rows[mcp] = [0.56, knuckle, 0.0]
        rows[tip] = [0.56, knuckle + 0.08, 0.0]
    return rows


def sliding_fist_rows(wrist_x: float, wrist_y: float = 0.75):
    scale = 0.20
    dx = wrist_x - 0.5
    knuckle = wrist_y - scale
    rows = [[0.5 + dx, wrist_y - 0.10, 0.0] for _ in range(21)]
    rows[0] = [wrist_x, wrist_y, 0.0]
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        rows[mcp] = [0.5 + dx, knuckle, 0.0]
        rows[pip] = [0.5 + dx, knuckle + 0.03, 0.0]
        rows[tip] = [0.5 + dx, knuckle + 0.08, 0.0]
    rows[9] = [0.5 + dx, knuckle, 0.0]
    rows[2] = [0.5 + dx, knuckle + 0.10, 0.0]
    rows[3] = [0.5 + dx, knuckle + 0.07, 0.0]
    rows[4] = [0.5 + dx, knuckle + 0.05, 0.0]
    return rows


@pytest.fixture(scope="module")
def new_parity(tmp_path_factory):
    fixtures = {
        "normalize": [], "velocity": [], "predicates": [],
        "clickSequences": [], "scrollSequences": [], "debounce": [],
        # The recognizer needs the real exported forest; without it the JS
        # classifier degrades to a constant Neutral and the ladder below is
        # never exercised past its first rung.
        "forestPath": str(FOREST_JSON), "forestSamples": None,
        # One sequence per rung of the mode ladder. The sliding fist is the
        # one that mattered historically: it is the path that referenced
        # swipeArmed before it was declared.
        "recognizerSequences": [
            [[None, i / 30.0] for i in range(5)]
            + [[pointing_rows(), (5 + i) / 30.0] for i in range(15)],
            [[sliding_fist_rows(0.5 + i * 0.04), i / 30.0] for i in range(30)],
            [[sliding_fist_rows(0.5, 0.75 - i * 0.015), i / 30.0]
             for i in range(30)],
            [[zoom_rows(0.25 + i * 0.03), i / 30.0] for i in range(20)],
            [[pinch_rows(0.02), i / 30.0] for i in range(25)],
            [[horns_rows(), i / 30.0] for i in range(60)],
            [[thumb_up_rows(), i / 30.0] for i in range(15)],
        ],
        "swipeSequences": [
            [[sliding_fist_rows(0.5 + i * 0.04), i / 30.0] for i in range(30)],
            [[sliding_fist_rows(0.5 - i * 0.04), i / 30.0] for i in range(30)],
            [[sliding_fist_rows(0.5, 0.75 - i * 0.015), i / 30.0]
             for i in range(30)],
        ],
        "zoomSequences": [
            [[zoom_rows(0.25 + i * 0.03), i / 30.0] for i in range(20)],
            [[zoom_rows(0.85 - i * 0.03), i / 30.0] for i in range(20)],
        ],
        "pauseSequences": [
            [[horns_rows(), i / 30.0] for i in range(60)],
            [[horns_rows() if i % 20 != 19 else zoom_rows(), i / 30.0]
             for i in range(60)],
        ],
        "newPredicates": [
            horns_rows(), zoom_rows(0.25), zoom_rows(0.05),
            sliding_fist_rows(0.5),
        ],
    }
    path = tmp_path_factory.mktemp("parity2") / "fixtures.json"
    path.write_text(json.dumps(fixtures))
    proc = subprocess.run([NODE, str(HARNESS), str(path)],
                          capture_output=True, text=True,
                          cwd=str(PROJECT_ROOT), timeout=120)
    if proc.returncode != 0:
        pytest.fail(f"parity harness failed:\n{proc.stderr}")
    return fixtures, json.loads(proc.stdout)


class TestNewPredicateParity:
    def test_rock_horns_and_zoom_pose_match(self, new_parity):
        from gestureflow.config import DEFAULT_CONFIG
        from gestureflow.pause_fsm import rock_horns
        from gestureflow.zoom_fsm import thumb_index_angle, zoom_pose

        fixtures, js = new_parity
        for i, rows in enumerate(fixtures["newPredicates"]):
            landmarks = lm(rows)
            got = js["newPredicates"][i]
            assert got["rockHorns"] == rock_horns(landmarks), i
            assert got["zoomPose"] == zoom_pose(landmarks,
                                                DEFAULT_CONFIG.zoom), i
            assert got["thumbIndexAngle"] == pytest.approx(
                thumb_index_angle(landmarks), abs=1e-5), i


class TestSwipeParity:
    def test_states_and_directions_match(self, new_parity):
        from gestureflow.config import DEFAULT_CONFIG
        from gestureflow.swipe_fsm import SwipeFSM

        fixtures, js = new_parity
        for s, sequence in enumerate(fixtures["swipeSequences"]):
            clock = _SeqClock()
            fsm = SwipeFSM(DEFAULT_CONFIG.swipe, clock=clock)
            for i, (rows, t) in enumerate(sequence):
                clock.now = t
                fsm.update(lm(rows))
                got = js["swipeSequences"][s][i]
                assert got["state"] == fsm.state.name, (
                    f"sequence {s} frame {i}: {got['state']} vs {fsm.state.name}"
                )
                assert got["direction"] == fsm.direction, (
                    f"sequence {s} frame {i}: direction differs"
                )


class TestZoomParity:
    def test_states_and_directions_match(self, new_parity):
        from gestureflow.config import DEFAULT_CONFIG
        from gestureflow.zoom_fsm import ZoomFSM

        fixtures, js = new_parity
        for s, sequence in enumerate(fixtures["zoomSequences"]):
            clock = _SeqClock()
            fsm = ZoomFSM(DEFAULT_CONFIG.zoom, clock=clock)
            for i, (rows, t) in enumerate(sequence):
                clock.now = t
                fsm.update(lm(rows))
                got = js["zoomSequences"][s][i]
                assert got["state"] == fsm.state.name, s
                assert got["direction"] == fsm.direction, s


class TestPauseParity:
    def test_toggles_land_on_the_same_frames(self, new_parity):
        from gestureflow.config import DEFAULT_CONFIG
        from gestureflow.pause_fsm import PauseFSM

        fixtures, js = new_parity
        for s, sequence in enumerate(fixtures["pauseSequences"]):
            clock = _SeqClock()
            fsm = PauseFSM(DEFAULT_CONFIG.pause, clock=clock)
            for i, (rows, t) in enumerate(sequence):
                clock.now = t
                fsm.update(lm(rows))
                got = js["pauseSequences"][s][i]
                assert got["paused"] == fsm.paused, (
                    f"sequence {s} frame {i}: paused differs"
                )
                assert got["toggled"] == fsm.toggled, (
                    f"sequence {s} frame {i}: toggle edge differs"
                )
                assert got["progress"] == pytest.approx(fsm.progress, abs=1e-5)


# The names the JS ladder uses, mapped onto the Python Mode enum. Python folds
# both pinch FSMs into a single CLICK rung; the demo distinguishes them so the
# on-screen pill can say which button it would have pressed.
_JS_MODE_TO_PY = {
    "none": "none", "paused": "paused", "command": "command",
    "scroll": "scroll", "swipe": "swipe", "zoom": "zoom", "drag": "drag",
    "left-click": "click", "right-click": "click",
    "volume": "volume", "cursor": "cursor", "tracking": "tracking",
}


class TestRecognizerParity:
    """Covers Recognizer.process, the function the demo page actually calls.

    Everything else in this file exercises a component. Nothing exercised the
    thing that composes them, and a ReferenceError in it shipped: process()
    threw on the first frame containing a hand, demo.js never reached its
    trailing requestAnimationFrame, and the canvas -- which is the only thing
    the page shows, the <video> being hidden -- froze on its last painted
    frame. Hence the reported "freezes when a hand enters".
    """

    def test_process_survives_every_sequence(self, new_parity):
        fixtures, js = new_parity
        assert len(js["recognizer"]) == len(fixtures["recognizerSequences"])
        for s, sequence in enumerate(fixtures["recognizerSequences"]):
            # A short result array means process() threw partway through.
            assert len(js["recognizer"][s]) == len(sequence), (
                f"sequence {s}: process() produced "
                f"{len(js['recognizer'][s])} of {len(sequence)} frames"
            )

    def test_every_field_the_demo_reads_is_populated(self, new_parity):
        _, js = new_parity
        for s, frames in enumerate(js["recognizer"]):
            for i, frame in enumerate(frames):
                assert frame["undefinedKeys"] == [], (
                    f"sequence {s} frame {i}: process() left "
                    f"{frame['undefinedKeys']} undefined"
                )

    def test_mode_is_always_a_real_mode(self, new_parity):
        _, js = new_parity
        valid = {m.value for m in Mode} | {"left-click", "right-click"}
        for s, frames in enumerate(js["recognizer"]):
            for i, frame in enumerate(frames):
                assert frame["mode"] in valid, (
                    f"sequence {s} frame {i}: unknown mode {frame['mode']!r}"
                )

    def test_ladder_agrees_with_the_python_router(self, new_parity):
        """The two precedence ladders must resolve identical flags identically.

        This compares the arbitration itself rather than the detectors -- the
        detectors are compared frame by frame in the classes above. Feeding
        GestureRouter the flags JS reported isolates the one thing left that
        could differ: the order the rungs are checked in.
        """
        from gestureflow.app import GestureRouter
        from gestureflow.capture import CaptureResult
        from gestureflow.click_fsm import ClickState
        from gestureflow.config import DEFAULT_CONFIG
        from gestureflow.scroll_fsm import ScrollState

        _, js = new_parity
        router = GestureRouter(DEFAULT_CONFIG)

        for s, frames in enumerate(js["recognizer"]):
            for i, frame in enumerate(frames):
                if frame["paused"]:
                    # Python resolves PAUSED inside the router from a flag the
                    # JS side resolves by returning early; both are checked by
                    # TestPauseParity and there is no ladder left to compare.
                    continue
                landmarks = (lm([[0.5, 0.5, 0.0]] * 21)
                             if frame["hasLandmarks"] else None)
                if landmarks is not None:
                    # _volume_pose reads these two directly.
                    landmarks[4].y = 0.4 if frame["volumeActive"] else 0.6
                    landmarks[5].y = 0.5
                result = InferenceResult(
                    capture=CaptureResult(frame=None, landmarks=landmarks,
                                          hand_lm_obj=None, timestamp=0.0),
                    stable_gesture=frame["stableGesture"],
                    vote_score=0, confidence=1.0,
                    raw_prediction=frame["rawPrediction"],
                    action=frame["action"],
                    click_fired=False, fsm_active=frame["fsmActive"],
                    fsm_state=ClickState.IDLE, hold_progress=0.0,
                    right_click_fired=False,
                    right_fsm_active=frame["rightFsmActive"],
                    right_fsm_state=ClickState.IDLE, right_hold_progress=0.0,
                    scroll_delta=frame["scrollDelta"],
                    scroll_active=frame["scrollActive"],
                    scroll_state=ScrollState.IDLE,
                    index_extended=frame["indexExtended"],
                    thumb_raised=frame["thumbRaised"],
                    zoom_active=frame["zoomActive"],
                    swipe_armed=frame["swipeArmed"],
                    dragging=frame["dragging"],
                )
                assert _JS_MODE_TO_PY[frame["mode"]] == \
                    router.active_mode(result).value, (
                        f"sequence {s} frame {i}: JS said {frame['mode']!r}, "
                        f"Python said {router.active_mode(result).value!r}"
                    )
