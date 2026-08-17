"""Build Y4M clips for Chrome's fake camera device.

The browser demo can only be measured honestly if something is actually waving
a hand at it. Chrome will play a Y4M file in place of a real webcam
(``--use-file-for-fake-video-capture``), which gives a byte-identical input
stream on every run -- something a real camera cannot do, and the reason the
before/after numbers in DIAGNOSIS.md are comparable at all.

Two clips get written:

* ``hand.y4m``   -- a hand filling the frame, gently drifting so the tracker and
                    the velocity-based detectors see real motion rather than a
                    frozen image.
* ``nohand.y4m`` -- the same room with no hand in it.

Y4M is written by hand here because it is a header, the literal word FRAME, and
raw I420 planes. That is less machinery than depending on ffmpeg.

Usage:
    python scripts/make_fake_camera.py --src <dir-of-jpgs> --out <dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

WIDTH = 640
HEIGHT = 480
FPS = 30


def _backdrop(width: int, height: int) -> np.ndarray:
    """A plausible dim room: vertical gradient plus fixed-pattern sensor noise.

    Content barely matters for timing -- the palm detector is a fixed-cost CNN
    that runs the same convolutions over whatever it is handed. What matters is
    that it is not a flat colour, so the encoder and the detector both do
    realistic work.
    """
    ramp = np.linspace(70, 30, height, dtype=np.float32)[:, None]
    frame = np.repeat(ramp, width, axis=1)
    frame = np.dstack([frame * 0.95, frame, frame * 1.05])
    rng = np.random.default_rng(7)
    frame += rng.normal(0.0, 3.0, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _compose(backdrop: np.ndarray, hand: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Paste the hand photo into the backdrop at an offset."""
    out = backdrop.copy()
    hh, hw = hand.shape[:2]
    x0 = (WIDTH - hw) // 2 + dx
    y0 = (HEIGHT - hh) // 2 + dy
    x0 = max(0, min(WIDTH - hw, x0))
    y0 = max(0, min(HEIGHT - hh, y0))
    out[y0:y0 + hh, x0:x0 + hw] = hand
    return out


def _write_y4m(path: Path, frames: list[np.ndarray]) -> None:
    header = f"YUV4MPEG2 W{WIDTH} H{HEIGHT} F{FPS}:1 Ip A1:1 C420mpeg2\n"
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        for bgr in frames:
            # cv2 gives I420 as a single (H*3/2, W) plane stack: Y, then U, V.
            i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
            fh.write(b"FRAME\n")
            fh.write(i420.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Directory holding the hand photo(s).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hand", default="pointing_up.jpg",
                    help="Which photo to composite in.")
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    n = int(args.seconds * FPS)
    backdrop = _backdrop(WIDTH, HEIGHT)

    src = cv2.imread(str(args.src / args.hand))
    if src is None:
        raise SystemExit(f"could not read {args.src / args.hand}")
    # Scale so the hand occupies most of the frame height, as it would if you
    # were actually gesturing at a laptop webcam.
    scale = (HEIGHT * 0.88) / src.shape[0]
    hand = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hand_frames = []
    still_frames = []
    for i in range(n):
        t = i / FPS
        # A slow lissajous drift: enough motion for the velocity detectors and
        # the frame-to-frame tracker, not so much that the hand leaves frame.
        dx = int(38 * math.sin(t * 1.7))
        dy = int(26 * math.sin(t * 1.1 + 0.6))
        hand_frames.append(_compose(backdrop, hand, dx, dy))
        # The empty room still needs per-frame noise, or the camera stack can
        # coalesce identical frames and flatter the no-hand measurement.
        rng = np.random.default_rng(1000 + i)
        noise = rng.normal(0.0, 2.0, backdrop.shape)
        still_frames.append(np.clip(backdrop + noise, 0, 255).astype(np.uint8))

    _write_y4m(args.out / "hand.y4m", hand_frames)
    _write_y4m(args.out / "nohand.y4m", still_frames)
    for name in ("hand.y4m", "nohand.y4m"):
        size = (args.out / name).stat().st_size
        print(f"{name}: {n} frames, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
