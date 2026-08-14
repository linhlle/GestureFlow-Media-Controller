"""Per-user calibration.

The wizard itself needs a person at a camera, so what is tested here is
everything downstream of that: the threshold derivation, the file round-trip,
and — most importantly — that a bad or missing calibration can never stop the
app from starting. Calibration is a refinement, and a refinement that can brick
startup is worse than no refinement.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gestureflow import calibration as calib
from gestureflow.calibration import (
    CALIBRATION_SCHEMA,
    MIN_SAMPLES,
    Calibration,
    CalibrationError,
    derive,
)
from gestureflow.config import AppConfig, calibrated_config


def samples(centre: float, n: int = 40, spread: float = 0.02):
    """A tight cluster around `centre`, like a held pose."""
    return [centre + (i % 5 - 2) * spread / 4 for i in range(n)]


class TestDerivation:
    def test_thresholds_land_between_the_two_measured_states(self):
        result = derive(closed=samples(0.20), opened=samples(0.90))
        assert 0.20 < result.click_close < result.click_open < 0.90

    def test_hysteresis_is_preserved(self):
        """The dead band is what stops the click chattering; it must survive."""
        result = derive(closed=samples(0.20), opened=samples(0.90))
        assert result.click_open > result.click_close

    def test_a_tight_pincher_gets_tighter_thresholds(self):
        tight = derive(closed=samples(0.10), opened=samples(0.60))
        loose = derive(closed=samples(0.35), opened=samples(1.10))
        assert tight.click_close < loose.click_close
        assert tight.click_open < loose.click_open

    def test_right_click_is_derived_when_not_measured(self):
        result = derive(closed=samples(0.20), opened=samples(0.90))
        assert result.right_click_close < result.click_close
        assert result.right_click_open < result.click_open

    def test_measured_right_click_is_used_when_given(self):
        result = derive(closed=samples(0.20), opened=samples(0.90),
                        right_closed=samples(0.12), right_opened=samples(0.70))
        assert 0.12 < result.right_click_close < result.right_click_open < 0.70

    def test_sample_counts_are_recorded(self):
        result = derive(closed=samples(0.20, n=31), opened=samples(0.90, n=42))
        assert result.samples["closed"] == 31
        assert result.samples["opened"] == 42

    def test_hand_scale_is_carried_through(self):
        result = derive(closed=samples(0.20), opened=samples(0.90),
                        hand_scale=0.187)
        assert result.hand_scale == 0.187


class TestDerivationRefusesNonsense:
    """Encoding a bad recording would be worse than having no calibration."""

    def test_too_few_closed_samples(self):
        with pytest.raises(CalibrationError, match="sample"):
            derive(closed=samples(0.20, n=MIN_SAMPLES - 1),
                   opened=samples(0.90))

    def test_too_few_open_samples(self):
        with pytest.raises(CalibrationError, match="sample"):
            derive(closed=samples(0.20),
                   opened=samples(0.90, n=MIN_SAMPLES - 1))

    def test_an_open_hand_no_wider_than_the_pinch(self):
        """Almost always means both steps recorded the same pose."""
        with pytest.raises(CalibrationError, match="no wider"):
            derive(closed=samples(0.50), opened=samples(0.30))

    def test_identical_distributions(self):
        with pytest.raises(CalibrationError, match="no wider"):
            derive(closed=samples(0.40), opened=samples(0.40))

    def test_overlapping_right_click_samples(self):
        with pytest.raises(CalibrationError, match="overlap"):
            derive(closed=samples(0.20), opened=samples(0.90),
                   right_closed=samples(0.60), right_opened=samples(0.30))

    def test_the_error_says_what_to_do_about_it(self):
        with pytest.raises(CalibrationError) as exc:
            derive(closed=samples(0.50), opened=samples(0.30))
        assert "again" in str(exc.value).lower()


class TestDerivationProperties:
    @given(
        closed_centre=st.floats(min_value=0.05, max_value=0.5),
        gap=st.floats(min_value=0.15, max_value=1.2),
    )
    @settings(max_examples=250, deadline=None)
    def test_thresholds_are_always_ordered_and_inside_the_gap(
        self, closed_centre, gap,
    ):
        result = derive(closed=samples(closed_centre, spread=0.001),
                        opened=samples(closed_centre + gap, spread=0.001))
        assert result.click_close < result.click_open
        assert result.right_click_close < result.right_click_open
        assert result.click_close > 0

    @given(
        closed_centre=st.floats(min_value=0.05, max_value=0.5),
        gap=st.floats(min_value=0.15, max_value=1.2),
    )
    @settings(max_examples=250, deadline=None)
    def test_anything_derive_accepts_survives_a_round_trip(
        self, closed_centre, gap, tmp_path_factory,
    ):
        result = derive(closed=samples(closed_centre, spread=0.001),
                        opened=samples(closed_centre + gap, spread=0.001))
        path = tmp_path_factory.mktemp("cal") / "calibration.json"
        result.write(path)
        loaded = calib.load(path)
        assert loaded is not None
        assert loaded.click_close == pytest.approx(result.click_close, abs=1e-3)


class TestFileRoundTrip:
    def test_write_then_load(self, tmp_path):
        original = derive(closed=samples(0.20), opened=samples(0.90),
                          hand_scale=0.19)
        path = original.write(tmp_path / "calibration.json")
        loaded = calib.load(path)

        assert loaded is not None
        assert loaded.click_close == pytest.approx(original.click_close, abs=1e-3)
        assert loaded.click_open == pytest.approx(original.click_open, abs=1e-3)
        assert loaded.hand_scale == pytest.approx(0.19, abs=1e-3)

    def test_it_creates_the_directory(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "calibration.json"
        derive(closed=samples(0.20), opened=samples(0.90)).write(target)
        assert target.exists()

    def test_the_file_explains_itself(self, tmp_path):
        path = derive(closed=samples(0.20),
                      opened=samples(0.90)).write(tmp_path / "c.json")
        raw = json.loads(path.read_text())
        assert raw["schema"] == CALIBRATION_SCHEMA
        assert "Delete this file" in raw["note"]


class TestLoadingNeverBreaksStartup:
    """Every one of these must return None, not raise."""

    def test_missing_file(self, tmp_path):
        assert calib.load(tmp_path / "nope.json") is None

    def test_unparseable_file(self, tmp_path, capsys):
        path = tmp_path / "c.json"
        path.write_text("{not json at all")
        assert calib.load(path) is None
        assert "unreadable" in capsys.readouterr().out

    def test_wrong_schema(self, tmp_path, capsys):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"schema": "something/else"}))
        assert calib.load(path) is None
        assert "unrecognized" in capsys.readouterr().out

    def test_missing_values(self, tmp_path, capsys):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"schema": CALIBRATION_SCHEMA,
                                    "click_close": 0.2}))
        assert calib.load(path) is None
        assert "malformed" in capsys.readouterr().out

    def test_a_json_list_instead_of_an_object(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("[1, 2, 3]")
        assert calib.load(path) is None

    @pytest.mark.parametrize("values,why", [
        ({"click_close": 0.0, "click_open": 0.4,
          "right_click_close": 0.2, "right_click_open": 0.3},
         "a zero close threshold can never be crossed"),
        ({"click_close": 0.5, "click_open": 0.2,
          "right_click_close": 0.2, "right_click_open": 0.3},
         "inverted hysteresis would make the FSM chatter"),
        ({"click_close": -0.1, "click_open": 0.4,
          "right_click_close": 0.2, "right_click_open": 0.3},
         "negative distances are impossible"),
        ({"click_close": 0.2, "click_open": 99.0,
          "right_click_close": 0.2, "right_click_open": 0.3},
         "an unreachable open threshold means the click never releases"),
    ])
    def test_values_that_would_break_a_gesture_are_rejected(
        self, tmp_path, capsys, values, why,
    ):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"schema": CALIBRATION_SCHEMA, **values}))
        assert calib.load(path) is None, f"accepted a file where {why}"
        assert "out of range" in capsys.readouterr().out


class TestConfigOverlay:
    def test_no_calibration_leaves_the_config_untouched(self):
        base = AppConfig()
        assert calibrated_config(base, None) == base

    def test_calibration_replaces_only_the_pinch_thresholds(self):
        base = AppConfig()
        cal = Calibration(click_close=0.11, click_open=0.33,
                          right_click_close=0.09, right_click_open=0.29)
        result = calibrated_config(base, cal)

        assert result.click.close_threshold == 0.11
        assert result.click.open_threshold == 0.33
        assert result.right_click.close_threshold == 0.09
        assert result.right_click.open_threshold == 0.29

        # Everything else is not personal and must be left alone.
        assert result.scroll == base.scroll
        assert result.swipe == base.swipe
        assert result.zoom == base.zoom
        assert result.mouse == base.mouse
        assert result.click.min_hold_frames == base.click.min_hold_frames
        assert result.click.cooldown == base.click.cooldown

    def test_the_shipped_default_config_is_never_calibrated(self):
        """DEFAULT_CONFIG is the baseline the parity suites compare against."""
        from gestureflow.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.click.close_threshold == 0.28
        assert DEFAULT_CONFIG.click.open_threshold == 0.41


class TestSummarize:
    def test_reports_absence_plainly(self):
        assert "No calibration" in calib.summarize(None)[0]

    def test_reports_the_values(self):
        result = derive(closed=samples(0.20), opened=samples(0.90))
        lines = calib.summarize(result)
        assert any("click" in line for line in lines)
        assert any("hand scale" in line for line in lines)
