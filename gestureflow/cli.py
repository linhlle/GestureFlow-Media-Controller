"""GestureFlow command line.

    python -m gestureflow run                 live control with the HUD
    python -m gestureflow bench --seconds 30  headless timing report
    python -m gestureflow record out.jsonl    capture a landmark take
    python -m gestureflow replay take.jsonl   replay a take, print actions
    python -m gestureflow false-triggers ...  count actions over no-intent takes
    python -m gestureflow validate [config]   check a command config
    python -m gestureflow selftest            check detectors, no camera needed
    python -m gestureflow bridge              serve the local web UI
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from gestureflow.commands import (
    CommandConfigError,
    load_commands,
    resolve_commands_path,
)
from gestureflow.config import DEFAULT_CONFIG


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--commands", type=Path, default=None,
        help="path to a command config (default: ~/.gestureflow/commands.yaml, "
             "else configs/commands.default.yaml)",
    )


def cmd_run(args) -> int:
    from gestureflow.app import run
    return run(DEFAULT_CONFIG, commands_path=args.commands,
               metrics_out=args.metrics_out)


def cmd_bench(args) -> int:
    from gestureflow.app import run_headless
    return run_headless(
        seconds=args.seconds,
        cfg=DEFAULT_CONFIG,
        commands_path=args.commands,
        metrics_out=args.out,
        dry_run=not args.live,
    )


def cmd_record(args) -> int:
    """Record a landmark take from the camera."""
    import queue
    import threading

    import cv2

    from gestureflow.capture import CaptureThread
    from gestureflow.replay import RecordingHeader, RecordingWriter

    stop = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=4)
    cap = CaptureThread(q, DEFAULT_CONFIG, stop)

    header = RecordingHeader(
        label=args.label,
        note=args.note or "",
        frame_width=DEFAULT_CONFIG.camera.width,
        frame_height=DEFAULT_CONFIG.camera.height,
    )

    print(f"[record] Label: {args.label}")
    if args.label == "no-intent":
        print("[record] Act naturally and do NOT try to control anything: "
              "type, talk, gesture. Every action this take produces on replay "
              "is a false trigger.")
    print(f"[record] Recording to {args.out} — press 'q' to stop.")

    cap.start()
    deadline = time.monotonic() + args.seconds if args.seconds else None

    try:
        with RecordingWriter(args.out, header) as writer:
            while not stop.is_set():
                if deadline and time.monotonic() > deadline:
                    break
                try:
                    capture = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                writer.write(capture)

                if not args.headless:
                    frame = capture.frame.copy()
                    cv2.putText(frame,
                                f"REC [{args.label}]  {writer.frames} frames",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                    cv2.imshow("GestureFlow recorder", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frames = writer.frames
    finally:
        stop.set()
        cap.join(timeout=3.0)
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"[record] Wrote {frames} frames to {args.out}")
    return 0


def cmd_replay(args) -> int:
    from gestureflow.app import load_model
    from gestureflow.replay import read_recording, replay

    model = load_model(args.model)
    total = 0
    for path in args.recordings:
        recording = read_recording(path)
        result = replay(recording, model, DEFAULT_CONFIG)
        total += len(result.discrete_actions())

        print(f"\n{path}  [{recording.label}]  "
              f"{result.frames} frames, {result.duration:.1f}s")
        counts = result.counts()
        if not counts:
            print("  no actions")
        for name in sorted(counts):
            print(f"  {name:<12} {counts[name]}")

        if args.verbose:
            for action in result.discrete_actions():
                print(f"    {action}")

    print(f"\nTotal discrete actions across {len(args.recordings)} take(s): {total}")
    return 0


def cmd_false_triggers(args) -> int:
    from gestureflow.app import load_model
    from gestureflow.replay import false_trigger_report, read_recording, replay

    model = load_model(args.model)
    results = []
    for path in args.recordings:
        recording = read_recording(path)
        if recording.label != "no-intent" and not args.force:
            print(f"[false-triggers] Skipping {path}: label is "
                  f"{recording.label!r}, expected 'no-intent'. "
                  f"Pass --force to include it anyway.")
            continue
        results.append(replay(recording, model, DEFAULT_CONFIG))

    if not results:
        print("[false-triggers] No no-intent recordings to evaluate.")
        print("[false-triggers] Record some with: "
              "python -m gestureflow record take.jsonl --label no-intent")
        return 1

    report = false_trigger_report(results)
    totals = report["totals"]
    print(json.dumps(report, indent=2))
    print()
    print(f"{totals['false_triggers']} action(s) fired over "
          f"{totals['duration_s']:.1f}s of no-intent footage "
          f"({totals['false_triggers_per_minute']} per minute).")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"Report written to {args.out}")
    return 0


def cmd_validate(args) -> int:
    try:
        path = resolve_commands_path(args.commands)
        commands = load_commands(args.commands)
    except CommandConfigError as exc:
        print(f"INVALID: {exc}")
        return 1

    total = len(commands.bindings) + len(commands.named)
    print(f"OK: {path}")
    print(f"  version {commands.version}, neutral label "
          f"{commands.neutral_label}, {total} binding(s)")
    for label in commands.labels():
        binding = commands.bindings[label]
        print(f"  pose {label}: {binding.name:<20} {binding.action.describe()}")
    for gesture in commands.gesture_names():
        binding = commands.named[gesture]
        print(f"  {gesture:<12} {binding.name:<20} "
              f"{binding.action.describe()}")

    if args.model is not False:
        from gestureflow.app import load_model
        from gestureflow.utils import BindingError, validate_bindings
        try:
            model = load_model(args.model or None)
        except FileNotFoundError as exc:
            print(f"\n  (skipping model check: {exc})")
            return 0
        try:
            warnings_ = validate_bindings(model.classes_, commands.labels(),
                                          commands.neutral_label)
        except BindingError as exc:
            print(f"\nMISMATCH: {exc}")
            return 1
        for w in warnings_:
            print(f"\nWARNING: {w}")
    return 0


def cmd_selftest(args) -> int:
    """Check the geometric detectors against the recorded dataset.

    No camera needed. This is the check that would have caught the thumb-raised
    regression: it reports how often each predicate fires on real hands, and
    fails if the scroll gate is being vetoed at a rate that makes the FSM's
    consecutive-frame requirement impossible to satisfy.
    """
    from types import SimpleNamespace

    from gestureflow.scroll_fsm import (
        _index_extended,
        _is_true_scroll_fist,
        _strict_fist,
        _thumb_raised,
    )
    from gestureflow.utils import data_path

    try:
        import pandas as pd
    except ImportError:
        print("[selftest] pandas is required: pip install -e '.[train]'")
        return 1

    csv = data_path("gesture_data.csv")
    if not csv.exists():
        print(f"[selftest] No dataset at {csv}")
        return 1

    df = pd.read_csv(csv)
    scale = args.scale

    def to_raw(row):
        # The CSV holds normalized landmarks; re-inflate to raw image space so
        # the predicates see the kind of values the live pipeline gives them.
        v = row[:-1].to_numpy(dtype=float)
        return [SimpleNamespace(x=0.5 + v[i * 3] * scale,
                                y=0.6 + v[i * 3 + 1] * scale,
                                z=v[i * 3 + 2] * scale) for i in range(21)]

    names = ["Neutral", "L-Shape", "High-Five", "2-Finger"]
    print(f"Geometric detectors over {len(df)} recorded frames "
          f"(hand scale {scale}):\n")
    print(f"{'pose':<11}{'n':>6}{'thumbRaised':>13}{'indexExt':>10}"
          f"{'strictFist':>12}{'scrollFist':>12}")
    print("-" * 64)

    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        counts = [0, 0, 0, 0]
        for _, row in sub.iterrows():
            lm = to_raw(row)
            counts[0] += _thumb_raised(lm)
            counts[1] += _index_extended(lm)
            counts[2] += _strict_fist(lm)
            counts[3] += _is_true_scroll_fist(lm)
        n = len(sub)
        name = names[int(label)] if int(label) < len(names) else str(label)
        print(f"{name:<11}{n:>6}{counts[0] / n:>12.0%}{counts[1] / n:>10.0%}"
              f"{counts[2] / n:>11.0%}{counts[3] / n:>11.0%}")

    fists = [to_raw(r) for _, r in df.iterrows()
             if _strict_fist(to_raw(r)) and not _index_extended(to_raw(r))]
    print()
    if not fists:
        print("[selftest] No fist-like frames in the dataset; "
              "cannot check the scroll gate.")
        return 0

    blocked = sum(_thumb_raised(lm) for lm in fists)
    armed = sum(_is_true_scroll_fist(lm) for lm in fists)
    rate = blocked / len(fists)
    print(f"Scroll gate: {armed}/{len(fists)} genuine fists arm it; "
          f"{blocked} vetoed by the thumb test ({rate:.0%})")

    # The FSM needs min_hold_frames consecutive passes. A high per-frame veto
    # rate compounds into never firing at all.
    holds = DEFAULT_CONFIG.scroll.min_hold_frames
    chance = (1.0 - rate) ** holds
    print(f"Probability of {holds} consecutive passing frames: {chance:.4f}")

    if chance < 0.5:
        print("\n[selftest] FAIL: the scroll gate is vetoed too often for the "
              "FSM to reach SCROLLING. This is the regression described in "
              "DIAGNOSIS.md.")
        return 1

    print("\n[selftest] OK")
    return 0


def cmd_bridge(args) -> int:
    from gestureflow.bridge import serve
    return serve(host=args.host, port=args.port,
                 commands_path=args.commands, open_browser=not args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gestureflow",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="live gesture control with the HUD")
    _add_common(p_run)
    p_run.add_argument("--metrics-out", type=Path, default=None,
                       help="write a timing report here on exit")
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser(
        "bench", help="run headless for N seconds and report stage timings")
    _add_common(p_bench)
    p_bench.add_argument("--seconds", type=float, default=30.0)
    p_bench.add_argument("--out", type=Path, default=None,
                         help="JSON report path (default: bench/bench-<ts>.json)")
    p_bench.add_argument("--live", action="store_true",
                         help="actually perform actions (default is a dry run "
                              "so benchmarking does not move your cursor)")
    p_bench.set_defaults(func=cmd_bench)

    p_rec = sub.add_parser("record", help="record a landmark take to JSONL")
    p_rec.add_argument("out", type=Path)
    p_rec.add_argument("--label", default="unlabelled",
                       help="'no-intent' marks a take for false-trigger counting")
    p_rec.add_argument("--note", default="")
    p_rec.add_argument("--seconds", type=float, default=0.0,
                       help="stop automatically after N seconds")
    p_rec.add_argument("--headless", action="store_true")
    p_rec.set_defaults(func=cmd_record)

    p_rep = sub.add_parser("replay", help="replay takes through the pipeline")
    p_rep.add_argument("recordings", nargs="+", type=Path)
    p_rep.add_argument("--model", type=Path, default=None)
    p_rep.add_argument("--verbose", "-v", action="store_true")
    p_rep.set_defaults(func=cmd_replay)

    p_ft = sub.add_parser(
        "false-triggers",
        help="count actions fired over no-intent footage")
    p_ft.add_argument("recordings", nargs="+", type=Path)
    p_ft.add_argument("--model", type=Path, default=None)
    p_ft.add_argument("--out", type=Path, default=None)
    p_ft.add_argument("--force", action="store_true",
                      help="include takes not labelled 'no-intent'")
    p_ft.set_defaults(func=cmd_false_triggers)

    p_val = sub.add_parser("validate", help="check a command config")
    _add_common(p_val)
    p_val.add_argument("--model", type=Path, default=None,
                       help="also check bindings against this model's classes")
    p_val.set_defaults(func=cmd_validate)

    p_self = sub.add_parser(
        "selftest",
        help="check the geometric detectors against the recorded dataset")
    p_self.add_argument("--scale", type=float, default=0.30,
                        help="hand extent in image coords to simulate")
    p_self.set_defaults(func=cmd_selftest)

    p_bridge = sub.add_parser(
        "bridge", help="serve the web UI locally with live gesture state")
    _add_common(p_bridge)
    p_bridge.add_argument("--host", default="127.0.0.1")
    p_bridge.add_argument("--port", type=int, default=8765)
    p_bridge.add_argument("--no-browser", action="store_true")
    p_bridge.set_defaults(func=cmd_bridge)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return args.func(args)
    except CommandConfigError as exc:
        print(f"[gestureflow] Command config error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n[gestureflow] Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
