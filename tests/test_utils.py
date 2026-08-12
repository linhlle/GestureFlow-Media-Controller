"""
tests/test_utils.py
-------------------
Unit tests for gestureflow.utils.normalize_landmarks.

These tests use a plain SimpleNamespace instead of MediaPipe objects so the
test suite runs without a camera, MediaPipe, or macOS.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gestureflow.utils import normalize_landmarks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmarks(coords: list[tuple[float, float, float]]):
    """Create a list of fake landmark objects from (x, y, z) tuples."""
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in coords]


def _flat_hand(n: int = 21) -> list:
    """21 landmarks all at the same absolute position (degenerate case)."""
    return _make_landmarks([(0.5, 0.5, 0.0)] * n)


def _identity_hand() -> list:
    """Wrist at origin, landmark 1 at (1, 0, 0) — minimal non-degenerate hand."""
    # Landmark 1 sits at (1, 0, 0) relative to the wrist, which is the origin.
    return _make_landmarks(
        [(0.0, 0.0, 0.0)] + [(1.0, 0.0, 0.0)] + [(0.0, 0.0, 0.0)] * 19
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_length_is_63(self):
        lms = _make_landmarks([(float(i), 0.0, 0.0) for i in range(21)])
        result = normalize_landmarks(lms)
        assert len(result) == 63

    def test_returns_list_of_floats(self):
        lms = _make_landmarks([(float(i) * 0.1, 0.0, 0.0) for i in range(21)])
        result = normalize_landmarks(lms)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)


class TestWristRelative:
    def test_first_triplet_is_always_zero(self):
        """Landmark 0 minus itself = (0, 0, 0)."""
        lms = _make_landmarks([(0.3, 0.7, 0.1)] + [(float(i), 0.0, 0.0) for i in range(1, 21)])
        result = normalize_landmarks(lms)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_translation_invariant(self):
        """Shifting every landmark by the same offset should not change output."""
        base_coords = [(float(i) * 0.05, float(i) * 0.03, 0.0) for i in range(21)]
        offset = (0.2, 0.4, 0.1)
        shifted_coords = [(x + offset[0], y + offset[1], z + offset[2]) for x, y, z in base_coords]

        result_base = normalize_landmarks(_make_landmarks(base_coords))
        result_shifted = normalize_landmarks(_make_landmarks(shifted_coords))

        for a, b in zip(result_base, result_shifted):
            assert a == pytest.approx(b, abs=1e-9)


class TestNormalization:
    def test_max_absolute_value_is_one(self):
        """After normalisation the maximum absolute value must be exactly 1.0."""
        lms = _make_landmarks([(float(i) * 0.07, float(i) * 0.03, 0.0) for i in range(21)])
        result = normalize_landmarks(lms)
        assert max(abs(v) for v in result) == pytest.approx(1.0)

    def test_all_values_in_minus_one_to_one(self):
        lms = _make_landmarks([(float(i) * 0.07, float(i) * 0.03, 0.01 * i) for i in range(21)])
        result = normalize_landmarks(lms)
        assert all(-1.0 <= v <= 1.0 for v in result)

    def test_scale_invariant(self):
        """Scaling every landmark by a constant should not change the output."""
        base_coords = [(float(i) * 0.05, float(i) * 0.04, 0.0) for i in range(21)]
        scaled_coords = [(x * 2.0, y * 2.0, z * 2.0) for x, y, z in base_coords]

        result_base = normalize_landmarks(_make_landmarks(base_coords))
        result_scaled = normalize_landmarks(_make_landmarks(scaled_coords))

        for a, b in zip(result_base, result_scaled):
            assert a == pytest.approx(b, abs=1e-9)


class TestDegenerateCases:
    def test_all_landmarks_at_same_position_returns_zeros(self):
        """When max_val == 0 the function must return 63 zeros, not raise."""
        result = normalize_landmarks(_flat_hand())
        assert result == [0.0] * 63

    def test_empty_list_returns_zeros(self):
        result = normalize_landmarks([])
        assert result == [0.0] * 63

    def test_single_landmark_returns_zeros(self):
        lms = _make_landmarks([(0.5, 0.3, 0.1)])
        result = normalize_landmarks(lms)
        assert result == [0.0] * 63


class TestKnownValues:
    def test_one_landmark_offset_on_x_axis(self):
        """Landmark 1 is at (1, 0, 0) relative to wrist; all others at wrist.
        After normalisation, index 3 (lm1_x) should be 1.0."""
        lms = _make_landmarks(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] + [(0.0, 0.0, 0.0)] * 19
        )
        result = normalize_landmarks(lms)
        assert result[3] == pytest.approx(1.0)  # lm1_x after normalisation
        assert result[4] == pytest.approx(0.0)  # lm1_y
        assert result[5] == pytest.approx(0.0)  # lm1_z
