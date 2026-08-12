"""Collect labelled hand-landmark samples for training.

Hold a number key while holding a pose to append frames for that label.
Rows are the same 63-float normalized vector the live pipeline uses, so
training data and inference data cannot drift apart.

Run from anywhere:  python scripts/data_logger.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gestureflow.config import DEFAULT_CONFIG  # noqa: E402
from gestureflow.utils import data_path, normalize_landmarks  # noqa: E402

# Index in this list == the integer label written to the CSV.
GESTURE_NAMES = ["Neutral", "L-Shape", "High-Five", "2-Finger"]
TARGET_PER_CLASS = 250


def _existing_counts(csv_file: Path) -> dict:
    counts = {str(i): 0 for i in range(len(GESTURE_NAMES))}
    if not csv_file.is_file():
        return counts
    with csv_file.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return counts
        for row in reader:
            if row and row[-1] in counts:
                counts[row[-1]] += 1
    return counts


def main() -> None:
    csv_file = data_path("gesture_data.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_draws = mp.solutions.drawing_utils
    mp_cfg = DEFAULT_CONFIG.mediapipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=mp_cfg.max_num_hands,
        min_tracking_confidence=mp_cfg.min_tracking_confidence,
        min_detection_confidence=mp_cfg.min_detection_confidence,
    )

    cap = cv2.VideoCapture(DEFAULT_CONFIG.camera.device_index)
    deadline = time.monotonic() + 5.0
    while not cap.isOpened():
        if time.monotonic() > deadline:
            print("[data-logger] ERROR: could not open camera.")
            return
        time.sleep(0.1)

    counts = _existing_counts(csv_file)
    print(f"[data-logger] Writing to {csv_file}")
    print(f"[data-logger] Hold 0-{len(GESTURE_NAMES) - 1} to record, 'q' to quit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            cv2.rectangle(frame, (0, 0), (240, 30 * len(GESTURE_NAMES) + 10),
                          (0, 0, 0), -1)
            y = 30
            for label, count in sorted(counts.items()):
                color = (0, 255, 0) if count >= TARGET_PER_CLASS else (255, 255, 255)
                name = GESTURE_NAMES[int(label)]
                cv2.putText(frame, f"{label} {name}: {count}", (10, y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
                y += 30

            key = cv2.waitKey(1) & 0xFF

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draws.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS
                    )

                    if ord("0") <= key <= ord(str(len(GESTURE_NAMES) - 1)):
                        label = chr(key)
                        row = normalize_landmarks(hand_lms.landmark)
                        row.append(label)

                        write_header = not csv_file.is_file()
                        with csv_file.open("a", newline="") as f:
                            writer = csv.writer(f)
                            if write_header:
                                header = [
                                    f"lm{i}_{c}"
                                    for i in range(21)
                                    for c in ("x", "y", "z")
                                ] + ["label"]
                                writer.writerow(header)
                            writer.writerow(row)

                        counts[label] += 1

            cv2.imshow("GestureFlow data logger", frame)
            if key == ord("q"):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    print("[data-logger] Final counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label} {GESTURE_NAMES[int(label)]}: {count}")


if __name__ == "__main__":
    main()
