"""Tests for the command line.

Argparse wiring breaks silently: a renamed argument or a subcommand pointing at
the wrong function only shows up when a human runs it. These exercise the
parser and the subcommands that need no camera.
"""

from __future__ import annotations

import json

import pytest

from gestureflow.cli import build_parser, main
from gestureflow.utils import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "commands.default.yaml"
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_classifier.pkl"


class TestParser:
    def test_subcommand_is_required(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    @pytest.mark.parametrize("command", [
        "run", "bench", "record", "replay", "false-triggers", "validate",
        "bridge",
    ])
    def test_every_subcommand_parses(self, command):
        argv = {
            "run": ["run"],
            "bench": ["bench"],
            "record": ["record", "out.jsonl"],
            "replay": ["replay", "take.jsonl"],
            "false-triggers": ["false-triggers", "take.jsonl"],
            "validate": ["validate"],
            "bridge": ["bridge"],
        }[command]
        args = build_parser().parse_args(argv)
        assert callable(args.func)

    def test_bench_defaults_to_a_dry_run(self):
        # Benchmarking must not fling the real cursor around by default.
        args = build_parser().parse_args(["bench"])
        assert args.live is False

    def test_bench_seconds_is_a_float(self):
        assert build_parser().parse_args(["bench", "--seconds", "2.5"]).seconds == 2.5

    def test_record_accepts_a_label(self):
        args = build_parser().parse_args(
            ["record", "t.jsonl", "--label", "no-intent"])
        assert args.label == "no-intent"

    def test_replay_takes_several_recordings(self):
        args = build_parser().parse_args(["replay", "a.jsonl", "b.jsonl"])
        assert len(args.recordings) == 2

    def test_bridge_defaults_to_loopback(self):
        args = build_parser().parse_args(["bridge"])
        assert args.host == "127.0.0.1"
        assert args.port == 8765


class TestValidate:
    def test_shipped_config_validates(self, capsys):
        assert main(["validate", "--commands", str(DEFAULT_CONFIG_PATH)]) == 0
        out = capsys.readouterr().out
        assert "OK:" in out
        assert "Spotlight" in out

    def test_broken_config_exits_nonzero(self, tmp_path, capsys):
        path = tmp_path / "commands.yaml"
        path.write_text("version: 1\nneutral_label: 0\ngestures:\n"
                        "  - label: 1\n    name: X\n"
                        "    action:\n      type: eval\n")
        assert main(["validate", "--commands", str(path)]) == 1
        assert "INVALID" in capsys.readouterr().out

    def test_missing_config_exits_nonzero(self, tmp_path, capsys):
        assert main(["validate", "--commands", str(tmp_path / "nope.yaml")]) == 1

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="no model")
    def test_model_check_passes_for_the_shipped_pair(self, capsys):
        code = main(["validate", "--commands", str(DEFAULT_CONFIG_PATH),
                     "--model", str(MODEL_PATH)])
        assert code == 0
        assert "MISMATCH" not in capsys.readouterr().out

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="no model")
    def test_unbound_predictable_class_is_reported(self, tmp_path, capsys):
        # The model predicts 1, 2, 3; bind only 1.
        path = tmp_path / "commands.yaml"
        path.write_text(
            "version: 1\nneutral_label: 0\ngestures:\n"
            "  - label: 1\n    name: Spotlight\n"
            "    action:\n      type: hotkey\n      keys: [command, space]\n"
        )
        code = main(["validate", "--commands", str(path),
                     "--model", str(MODEL_PATH)])
        assert code == 1
        assert "MISMATCH" in capsys.readouterr().out


class TestReplayCommands:
    """Replay and false-triggers run entirely from a file."""

    def _recording(self, path, label="no-intent", frames=30):
        import numpy as np

        from gestureflow.capture import CaptureResult
        from gestureflow.replay import RecordingHeader, RecordingWriter

        def still_hand(t):
            from types import SimpleNamespace
            lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
            return CaptureResult(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                landmarks=lm, hand_lm_obj=None, timestamp=t)

        with RecordingWriter(path, RecordingHeader(label=label)) as w:
            for i in range(frames):
                w.write(still_hand(i / 30.0))
        return path

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="no model")
    def test_replay_prints_a_summary(self, tmp_path, capsys):
        take = self._recording(tmp_path / "take.jsonl")
        assert main(["replay", str(take), "--model", str(MODEL_PATH)]) == 0
        out = capsys.readouterr().out
        assert "no-intent" in out
        assert "30 frames" in out

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="no model")
    def test_false_triggers_writes_a_report(self, tmp_path, capsys):
        take = self._recording(tmp_path / "take.jsonl")
        out_path = tmp_path / "report.json"
        code = main(["false-triggers", str(take), "--model", str(MODEL_PATH),
                     "--out", str(out_path)])
        assert code == 0

        report = json.loads(out_path.read_text())
        assert report["schema"] == "gestureflow.false_trigger/1"
        assert report["totals"]["takes"] == 1
        assert "not a general property" in report["note"]

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="no model")
    def test_false_triggers_skips_takes_not_labelled_no_intent(
        self, tmp_path, capsys,
    ):
        # Counting a deliberate gesture as a false trigger would make the
        # number meaningless.
        take = self._recording(tmp_path / "take.jsonl", label="clicking")
        code = main(["false-triggers", str(take), "--model", str(MODEL_PATH)])
        assert code == 1
        assert "Skipping" in capsys.readouterr().out

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="no model")
    def test_force_includes_other_labels(self, tmp_path, capsys):
        take = self._recording(tmp_path / "take.jsonl", label="clicking")
        code = main(["false-triggers", str(take), "--model", str(MODEL_PATH),
                     "--force"])
        assert code == 0


class TestErrorHandling:
    def test_command_config_errors_are_reported_not_raised(self, tmp_path,
                                                           capsys):
        path = tmp_path / "commands.yaml"
        path.write_text("gestures: [")
        assert main(["validate", "--commands", str(path)]) == 1
        assert "INVALID" in capsys.readouterr().out
