"""Keep the JavaScript port honest against the Python original.

web/js/recognizer.js and web/js/schema.js reimplement logic that already exists
in Python. Duplication like that rots silently: someone tunes a threshold in
config.py, the website keeps classifying with the old one, and the demo quietly
stops matching the app it is advertising.

These tests read the JS source and assert the constants and structures match.
They are deliberately textual rather than executing JS, so they need no Node
in CI -- what they are guarding against is a value drifting, not a logic bug,
and a value drifting is exactly what text can catch.

The forest export is checked properly: the JSON the browser loads is evaluated
with a Python mirror of the JS traversal and compared against sklearn.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from gestureflow.commands import (
    ACTION_TYPES,
    MEDIA_ACTIONS,
    SCHEMA_VERSION,
    VALID_KEYS,
)
from gestureflow.config import DEFAULT_CONFIG
from gestureflow.utils import PROJECT_ROOT

WEB = PROJECT_ROOT / "web"
RECOGNIZER_JS = WEB / "js" / "recognizer.js"
SCHEMA_JS = WEB / "js" / "schema.js"
FOREST_JSON = WEB / "models" / "forest.json"

pytestmark = pytest.mark.skipif(
    not RECOGNIZER_JS.exists(), reason="web/ not present"
)


def js_source(path: Path) -> str:
    return path.read_text()


def js_number(source: str, key: str) -> float:
    """Pull `key: <number>` out of a JS object literal."""
    match = re.search(rf"\b{re.escape(key)}:\s*([0-9.]+)", source)
    assert match, f"could not find {key} in the JS source"
    return float(match.group(1))


# ---------------------------------------------------------------------------
# Recognizer constants
# ---------------------------------------------------------------------------

class TestRecognizerParity:
    @pytest.fixture(scope="class")
    def src(self):
        return js_source(RECOGNIZER_JS)

    def test_confidence_threshold_matches(self, src):
        assert js_number(src, "confidenceThreshold") == \
            DEFAULT_CONFIG.inference.confidence_threshold

    def test_vote_window_matches(self, src):
        assert js_number(src, "voteWindowSize") == \
            DEFAULT_CONFIG.debounce.vote_window_size

    def test_vote_threshold_matches(self, src):
        assert js_number(src, "voteThreshold") == \
            DEFAULT_CONFIG.debounce.vote_threshold

    def test_command_cooldown_matches(self, src):
        assert js_number(src, "cmdCooldown") == \
            DEFAULT_CONFIG.debounce.cmd_cooldown

    def test_click_thresholds_match(self, src):
        block = re.search(r"click:\s*\{([^}]*)\}", src).group(1)
        assert js_number(block, "close") == DEFAULT_CONFIG.click.close_threshold
        assert js_number(block, "open") == DEFAULT_CONFIG.click.open_threshold
        assert js_number(block, "minHoldFrames") == \
            DEFAULT_CONFIG.click.min_hold_frames
        assert js_number(block, "cooldown") == DEFAULT_CONFIG.click.cooldown

    def test_right_click_thresholds_match(self, src):
        block = re.search(r"rightClick:\s*\{([^}]*)\}", src).group(1)
        assert js_number(block, "close") == \
            DEFAULT_CONFIG.right_click.close_threshold
        assert js_number(block, "open") == \
            DEFAULT_CONFIG.right_click.open_threshold
        assert js_number(block, "minHoldFrames") == \
            DEFAULT_CONFIG.right_click.min_hold_frames
        assert js_number(block, "cooldown") == DEFAULT_CONFIG.right_click.cooldown

    def test_scroll_parameters_match(self, src):
        block = re.search(r"scroll:\s*\{([^}]*)\}", src, re.S).group(1)
        assert js_number(block, "sensitivity") == \
            DEFAULT_CONFIG.scroll.sensitivity
        assert js_number(block, "minHoldFrames") == \
            DEFAULT_CONFIG.scroll.min_hold_frames
        assert js_number(block, "cooldown") == DEFAULT_CONFIG.scroll.cooldown
        assert js_number(block, "step") == DEFAULT_CONFIG.scroll.step
        assert js_number(block, "velocityExponent") == \
            DEFAULT_CONFIG.scroll.velocity_exponent

    def test_landmark_indices_match_the_python_constants(self, src):
        from gestureflow import scroll_fsm as sf
        expected = {
            "WRIST": 0,
            "THUMB_MCP": sf._THUMB_MCP,
            "THUMB_TIP": sf._THUMB_TIP,
            "INDEX_PIP": sf._INDEX_PIP,
            "INDEX_TIP": sf._INDEX_TIP,
            "MIDDLE_TIP": sf._MIDDLE_TIP,
            "RING_TIP": sf._RING_TIP,
            "PINKY_TIP": sf._PINKY_TIP,
        }
        block = re.search(r"export const LM = \{(.*?)\};", src, re.S).group(1)
        for name, value in expected.items():
            match = re.search(rf"\b{name}:\s*(\d+)", block)
            assert match, f"LM.{name} missing from recognizer.js"
            assert int(match.group(1)) == value, f"LM.{name} drifted"

    def test_feature_vector_length_matches(self, src):
        assert "new Array(63)" in src, "the JS must produce 63 features"

    def test_cooldown_seeds_are_negative_infinity(self, src):
        # The Python fix was seeding with -inf; the JS must not reintroduce the
        # bug by seeding with 0.
        assert src.count("-Infinity") >= 3
        assert not re.search(r"last\w*Time\s*=\s*0\b", src)

    def test_click_precision_matches(self, src):
        from gestureflow.scroll_fsm import _CLICK_PRECISION
        match = re.search(r"CLICK_PRECISION\s*=\s*(\d+)", src)
        assert match and int(match.group(1)) == _CLICK_PRECISION

    def test_geometric_margins_match(self, src):
        # _index_extended / _thumb_raised use a 0.04 margin; _strict_fist 0.03.
        assert "margin = 0.04" in src
        assert "threshold = 0.03" in src


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

class TestSchemaParity:
    @pytest.fixture(scope="class")
    def src(self):
        return js_source(SCHEMA_JS)

    def test_schema_version_matches(self, src):
        match = re.search(r"SCHEMA_VERSION\s*=\s*(\d+)", src)
        assert match and int(match.group(1)) == SCHEMA_VERSION

    def test_action_types_match(self, src):
        block = re.search(r"ACTION_TYPES = \[(.*?)\];", src, re.S).group(1)
        found = set(re.findall(r"id:\s*'([a-z]+)'", block))
        assert found == ACTION_TYPES

    def test_media_actions_match(self, src):
        block = re.search(r"MEDIA_ACTIONS = \[(.*?)\];", src, re.S).group(1)
        found = set(re.findall(r"'([a-z]+)'", block))
        assert found == MEDIA_ACTIONS

    def test_key_whitelist_matches(self, src):
        block = re.search(r"VALID_KEYS = new Set\(\[(.*?)\]\);", src, re.S).group(1)

        # Both quote styles appear: the apostrophe key is written "'".
        # Matching only one style silently misaligns and swallows the
        # entries after it.
        literals = re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"",
                              block)
        found = {(a or b).replace("\\\\", "\\") for a, b in literals}
        found.discard("")   # regex artifact from scanning across separators
        # The JS builds function keys and alphanumerics programmatically.
        found |= {f"f{i}" for i in range(1, 21)}
        found |= set("abcdefghijklmnopqrstuvwxyz")
        found |= set("0123456789")
        found.discard("abcdefghijklmnopqrstuvwxyz")
        found.discard("0123456789")

        missing = VALID_KEYS - found
        extra = found - VALID_KEYS
        assert not missing, f"keys the app accepts but the builder rejects: {missing}"
        assert not extra, f"keys the builder offers but the app rejects: {extra}"

    def test_app_name_pattern_matches(self, src):
        from gestureflow.commands import _APP_NAME_RE
        match = re.search(r"APP_NAME_RE = /(.+?)/;", src)
        assert match, "APP_NAME_RE missing from schema.js"
        assert match.group(1) == _APP_NAME_RE.pattern

    def test_model_classes_match_the_shipped_model(self, src):
        from gestureflow.app import load_model
        model = load_model()
        block = re.search(r"MODEL_CLASSES = \[(.*?)\];", src).group(1)
        found = [int(v) for v in re.findall(r"\d+", block)]
        assert found == [int(c) for c in model.classes_]

    def test_applescript_limit_matches(self, src):
        assert "4000" in src, "the 4000-character applescript cap must match"


# ---------------------------------------------------------------------------
# Exported forest
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FOREST_JSON.exists(),
                    reason="forest.json not exported")
class TestForestExport:
    @pytest.fixture(scope="class")
    def payload(self):
        return json.loads(FOREST_JSON.read_text())

    def test_schema_and_shape(self, payload):
        assert payload["schema"] == "gestureflow.forest/1"
        assert payload["n_features"] == 63
        assert payload["n_trees"] == len(payload["trees"])

    def test_classes_match_the_pickled_model(self, payload):
        from gestureflow.app import load_model
        model = load_model()
        assert payload["classes"] == [int(c) for c in model.classes_]

    def test_leaf_marker_is_sklearns_tree_leaf(self, payload):
        # -1 is TREE_LEAF; -2 is TREE_UNDEFINED and appears in `feature`.
        # Confusing the two makes the traversal loop forever.
        from sklearn.tree import _tree
        assert payload["leaf_marker"] == _tree.TREE_LEAF

    def test_every_traversal_terminates(self, payload):
        """No node may reach a leaf-less cycle."""
        leaf = payload["leaf_marker"]
        for t, tree in enumerate(payload["trees"]):
            for node in range(len(tree["left"])):
                seen = set()
                cur = node
                while tree["left"][cur] != leaf:
                    assert cur not in seen, f"cycle in tree {t}"
                    seen.add(cur)
                    cur = tree["left"][cur]

    def test_reproduces_sklearn_predictions(self, payload):
        """The browser must classify identically to the desktop app."""
        from gestureflow.app import load_model
        from scripts.export_model_json import _predict_proba_py

        model = load_model()
        rng = np.random.default_rng(1234)
        samples = rng.uniform(-1.0, 1.0, size=(60, payload["n_features"]))
        expected = model.predict_proba(samples)

        for i, sample in enumerate(samples):
            got = np.array(_predict_proba_py(payload, sample))
            assert np.allclose(got, expected[i], atol=1e-4), (
                f"sample {i}: exported forest disagrees with sklearn"
            )

    def test_reproduces_predictions_on_real_training_rows(self, payload):
        """Random vectors are not real hands; check actual samples too."""
        import pandas as pd

        from gestureflow.app import load_model
        from gestureflow.utils import data_path
        from scripts.export_model_json import _predict_proba_py

        csv = data_path("gesture_data.csv")
        if not csv.exists():
            pytest.skip("no dataset present")

        df = pd.read_csv(csv).sample(n=40, random_state=7)
        features = df.iloc[:, :-1].to_numpy()
        model = load_model()
        expected = model.predict_proba(features)

        for i, row in enumerate(features):
            got = np.array(_predict_proba_py(payload, row))
            assert np.allclose(got, expected[i], atol=1e-4)


# ---------------------------------------------------------------------------
# Site sanity
# ---------------------------------------------------------------------------

class TestSiteContent:
    PAGES = ["index.html", "demo.html", "builder.html", "guide.html"]

    @pytest.mark.parametrize("page", PAGES)
    def test_page_exists(self, page):
        assert (WEB / page).is_file()

    @pytest.mark.parametrize("page", PAGES)
    def test_referenced_local_assets_exist(self, page):
        html = (WEB / page).read_text()
        for ref in re.findall(r'(?:href|src)="(/[^"]+)"', html):
            path = ref.split("#")[0].split("?")[0]
            if not path or path == "/":
                continue
            target = WEB / path.lstrip("/")
            assert target.exists(), f"{page} references missing {path}"

    def test_no_unmeasured_performance_claims(self):
        """The site must not quote numbers the repo cannot produce.

        Guards the honesty rule directly: if someone later adds "30 FPS" or
        "98% accurate" to the copy, this fails.
        """
        banned = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:fps|ms|milliseconds)\b"
            r"|\b\d+(?:\.\d+)?%\s*(?:accurate|accuracy|precision|recall)\b"
            r"|\baccuracy of \d+"
            r"|\b\d+(?:\.\d+)?\s*ms\s*latency\b",
            re.I,
        )
        for page in self.PAGES:
            text = (WEB / page).read_text()
            hits = banned.findall(text)
            assert not hits, f"{page} contains an unmeasured claim: {hits}"

    def test_demo_page_states_it_cannot_control_the_computer(self):
        html = (WEB / "demo.html").read_text().lower()
        assert "does not control your computer" in html

    def test_landing_page_explains_the_browser_desktop_split(self):
        html = (WEB / "index.html").read_text().lower()
        assert "cannot move your cursor" in html
