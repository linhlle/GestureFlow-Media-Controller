"""Tests for the instrumentation and replay harnesses.

These two exist so performance and false-trigger claims can be backed by
measurement. If they are wrong, every number they produce is wrong, so the
percentile maths in particular is pinned down against hand-computed values.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gestureflow.capture import CaptureResult
from gestureflow.config import DEFAULT_CONFIG
from gestureflow.metrics import (
    MetricsRecorder,
    NullMetrics,
    Series,
    percentile,
)
from gestureflow.replay import (
    RecordingHeader,
    RecordingWriter,
    VirtualClock,
    false_trigger_report,
    read_recording,
    replay,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_median_of_a_known_series(self):
        assert percentile(list(range(1, 101)), 50) == 50

    def test_p95_and_p99_of_a_known_series(self):
        values = list(range(1, 101))
        assert percentile(values, 95) == 95
        assert percentile(values, 99) == 99

    def test_returns_an_observed_value_never_an_interpolated_one(self):
        # Nearest-rank: with sub-millisecond timings, interpolation invents
        # measurements that never happened.
        values = [1.0, 2.0, 100.0]
        assert percentile(values, 95) in values

    def test_empty_series_is_nan_not_zero(self):
        # Zero would read as "instant"; NaN reads as "no data", which is true.
        assert math.isnan(percentile([], 50))

    def test_single_sample(self):
        assert percentile([42.0], 99) == 42.0

    def test_p99_never_exceeds_the_maximum(self):
        values = sorted(float(i) for i in range(1, 1001))
        assert percentile(values, 99) <= values[-1]


class TestSeries:
    def test_summary_reports_count_and_percentiles(self):
        s = Series("x")
        for v in range(1, 101):
            s.add(v / 1000.0)                       # 1ms .. 100ms
        summary = s.summary()
        assert summary["count"] == 100
        assert summary["p50_ms"] == pytest.approx(50.0, abs=0.01)
        assert summary["p95_ms"] == pytest.approx(95.0, abs=0.01)

    def test_empty_series_reports_zero_count(self):
        assert Series("x").summary() == {"count": 0}

    def test_total_observed_survives_the_retention_cap(self):
        s = Series("x")
        s.samples = type(s.samples)(maxlen=10)
        for i in range(100):
            s.add(float(i))
        assert s.total_observed == 100
        assert s.summary()["retained"] == 10


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class TestMetricsRecorder:
    def test_timer_records_elapsed_time(self):
        clock = FakeClock()
        m = MetricsRecorder(clock=clock)
        with m.timer("stage"):
            clock.now += 0.025
        assert m.snapshot()["stages"]["stage"]["p50_ms"] == pytest.approx(25.0)

    def test_counters_accumulate(self):
        m = MetricsRecorder()
        m.count("drops", 3)
        m.count("drops")
        assert m.snapshot()["counters"]["drops"] == 4

    def test_queue_depth_histogram(self):
        m = MetricsRecorder()
        for depth in (0, 0, 1, 2, 2, 2):
            m.observe_queue("capture->inference", depth)
        hist = m.snapshot()["queue_depth_histogram"]["capture->inference"]
        assert hist == {"0": 2, "1": 1, "2": 3}

    def test_end_to_end_latency_uses_the_capture_timestamp(self):
        clock = FakeClock()
        m = MetricsRecorder(clock=clock)
        clock.now = 10.0
        m.record_end_to_end(captured_at=9.94)
        assert m.snapshot()["stages"]["end_to_end"]["p50_ms"] == pytest.approx(60.0)

    def test_fps_is_derived_from_the_active_window(self):
        clock = FakeClock()
        m = MetricsRecorder(clock=clock)
        for _ in range(31):                          # 31 samples over 1s
            m.record("capture.grab", 0.001)
            clock.now += 1 / 30.0
        assert m.fps("capture.grab") == pytest.approx(30.0, rel=0.05)

    def test_report_includes_a_provenance_note(self):
        report = MetricsRecorder().report()
        assert "do not generalize" in report["note"]

    def test_write_json_is_valid_json(self, tmp_path):
        m = MetricsRecorder()
        m.record("stage", 0.01)
        path = m.write_json(tmp_path / "sub" / "report.json")
        data = json.loads(path.read_text())
        assert data["schema"] == "gestureflow.metrics/1"

    def test_format_text_does_not_crash_on_an_empty_recorder(self):
        assert "Elapsed" in MetricsRecorder().format_text()


class TestNullMetrics:
    def test_records_nothing(self):
        m = NullMetrics()
        m.record("stage", 1.0)
        m.count("x")
        m.observe_queue("h", 3)
        snap = m.snapshot()
        assert snap["stages"] == {}
        assert snap["counters"] == {}

    def test_timer_is_still_usable(self):
        with NullMetrics().timer("stage"):
            pass


# ---------------------------------------------------------------------------
# Recording / replay
# ---------------------------------------------------------------------------

def _lms(wrist_y=0.5):
    from types import SimpleNamespace
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lm[0] = SimpleNamespace(x=0.5, y=wrist_y, z=0.0)
    return lm


def _capture(landmarks, t):
    return CaptureResult(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                         landmarks=landmarks, hand_lm_obj=None, timestamp=t)


class _StubModel:
    classes_ = np.array([0, 1, 2, 3])

    def predict_proba(self, features):
        return np.array([[1.0, 0.0, 0.0, 0.0]])


class TestRecording:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "take.jsonl"
        with RecordingWriter(path, RecordingHeader(label="no-intent")) as w:
            for i in range(5):
                w.write(_capture(_lms(0.5 - i * 0.01), t=100.0 + i / 30.0))

        recording = read_recording(path)
        assert recording.label == "no-intent"
        assert len(recording) == 5
        assert recording.duration == pytest.approx(4 / 30.0, abs=1e-4)

    def test_timestamps_are_stored_relative_to_the_first_frame(self, tmp_path):
        path = tmp_path / "take.jsonl"
        with RecordingWriter(path) as w:
            w.write(_capture(_lms(), t=9999.0))
            w.write(_capture(_lms(), t=9999.5))
        recording = read_recording(path)
        assert recording.frames[0]["t"] == 0.0
        assert recording.frames[1]["t"] == pytest.approx(0.5)

    def test_frames_without_a_hand_round_trip_as_null(self, tmp_path):
        path = tmp_path / "take.jsonl"
        with RecordingWriter(path) as w:
            w.write(_capture(None, t=0.0))
        assert read_recording(path).frames[0]["lm"] is None

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_recording(path)

    def test_unknown_schema_is_rejected(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"schema": "something/9"}) + "\n")
        with pytest.raises(ValueError, match="unsupported recording schema"):
            read_recording(path)


class TestVirtualClock:
    def test_advances_monotonically(self):
        clock = VirtualClock()
        clock.advance_to(5.0)
        clock.advance_to(3.0)               # a backwards step must not rewind
        assert clock() == 5.0


class TestReplayDeterminism:
    def _recording(self, tmp_path, frames=30):
        path = tmp_path / "take.jsonl"
        with RecordingWriter(path, RecordingHeader(label="no-intent")) as w:
            for i in range(frames):
                w.write(_capture(_lms(0.5), t=i / 30.0))
        return read_recording(path)

    def test_same_input_produces_the_same_actions_every_time(self, tmp_path):
        recording = self._recording(tmp_path)
        model = _StubModel()
        first = replay(recording, model, DEFAULT_CONFIG)
        second = replay(recording, model, DEFAULT_CONFIG)
        assert first.counts() == second.counts()
        assert len(first.actions) == len(second.actions)

    def test_a_still_hand_produces_no_discrete_actions(self, tmp_path):
        recording = self._recording(tmp_path)
        result = replay(recording, _StubModel(), DEFAULT_CONFIG)
        assert result.discrete_actions() == []

    def test_replay_reports_frames_and_duration(self, tmp_path):
        recording = self._recording(tmp_path, frames=30)
        result = replay(recording, _StubModel(), DEFAULT_CONFIG)
        assert result.frames == 30
        assert result.duration == pytest.approx(29 / 30.0, abs=1e-3)


class TestFalseTriggerReport:
    def _result(self, tmp_path, name, frames=60):
        path = tmp_path / name
        with RecordingWriter(path, RecordingHeader(label="no-intent")) as w:
            for i in range(frames):
                w.write(_capture(_lms(0.5), t=i / 30.0))
        return replay(read_recording(path), _StubModel(), DEFAULT_CONFIG)

    def test_clean_footage_reports_zero(self, tmp_path):
        report = false_trigger_report([self._result(tmp_path, "a.jsonl")])
        assert report["totals"]["false_triggers"] == 0
        assert report["totals"]["false_triggers_per_minute"] == 0.0

    def test_totals_aggregate_across_takes(self, tmp_path):
        results = [self._result(tmp_path, "a.jsonl"),
                   self._result(tmp_path, "b.jsonl")]
        report = false_trigger_report(results)
        assert report["totals"]["takes"] == 2
        assert report["totals"]["frames"] == 120

    def test_report_carries_its_own_caveat(self, tmp_path):
        report = false_trigger_report([self._result(tmp_path, "a.jsonl")])
        assert "not a general property" in report["note"]

    def test_cursor_movement_is_not_a_false_trigger(self, tmp_path):
        # Continuous cursor motion is expected; only discrete events count.
        result = self._result(tmp_path, "a.jsonl")
        report = false_trigger_report([result])
        assert "MoveCursor" not in report["takes"][0]["by_type"]
