from __future__ import annotations

import queue
from pathlib import Path
from typing import Any, Sequence


def drop_oldest_put(q: queue.Queue, item: Any) -> int:
    """Put ``item`` on ``q``, evicting the oldest entry if the queue is full.

    Every stage of this pipeline prefers fresh data over complete data: a
    landmark frame from 200 ms ago is worse than useless for cursor control,
    because acting on it moves the cursor to where the hand *was*.  So when a
    consumer falls behind, the backlog is discarded rather than the new frame.

    Returns the number of items dropped (0 or 1) so callers can count them.
    """
    try:
        q.put_nowait(item)
        return 0
    except queue.Full:
        pass

    dropped = 0
    try:
        q.get_nowait()
        dropped = 1
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        # A competing producer refilled the slot. Dropping the item we were
        # handed is correct here -- the queue already holds something newer
        # than whatever we just evicted.
        dropped += 1

    return dropped


def normalize_landmarks(landmark_list: Sequence[Any]) -> list[float]:
    if not landmark_list:
        return [0.0] * 63

    base_x, base_y, base_z = landmark_list[0].x, landmark_list[0].y, landmark_list[0].z

    relative: list[float] = []

    for lm in landmark_list:
        relative.extend([
            lm.x - base_x,
            lm.y - base_y,
            lm.z - base_z,
        ])


    max_val = max(abs(v) for v in relative)
    if max_val == 0.0:
        return [0.0] * 63
    
    return [v / max_val for v in relative]


class BindingError(RuntimeError):
    """Raised when the command map and the model disagree about labels."""


def validate_bindings(
    model_classes: Sequence[int],
    bound_labels: Sequence[int],
    neutral_label: int = 0,
) -> list[str]:
    """Check that every gesture the model can predict has an action bound.

    Returns a list of *warnings* (bindings the model can never trigger) and
    raises :class:`BindingError` for the fatal case (a predictable gesture with
    no binding, which would silently do nothing at runtime).

    The reverse direction is the defect this repo actually shipped with: the
    command map bound labels 4 and 5 to Screenshot and Do Not Disturb, but the
    model's classes_ is [0 1 2 3], so those two gestures were dead config that
    could never fire.  That is a warning rather than an error because a user
    may legitimately be mid-way through collecting data for a new gesture.
    """
    predictable = {int(c) for c in model_classes} - {neutral_label}
    bound = {int(b) for b in bound_labels}

    missing = sorted(predictable - bound)
    if missing:
        raise BindingError(
            f"The model can predict gesture label(s) {missing} but no command "
            f"is bound to them, so recognizing that gesture would do nothing. "
            f"Add a binding for each, or retrain without those classes.\n"
            f"  model classes : {sorted(int(c) for c in model_classes)}\n"
            f"  bound labels  : {sorted(bound)}"
        )

    unreachable = sorted(bound - predictable - {neutral_label})
    return [
        f"Gesture label {label} has a command bound to it, but the loaded model "
        f"cannot predict that class (classes: "
        f"{sorted(int(c) for c in model_classes)}). The binding will never fire "
        f"until the model is retrained with that gesture."
        for label in unreachable
    ]


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

def data_path(filename: str) -> Path:
    return PROJECT_ROOT / "data" / filename
 
 
def models_path(filename: str) -> Path:
    return PROJECT_ROOT / "models" / filename




