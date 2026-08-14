"""What a frame means.

One enum, one answer per frame. See GestureRouter.active_mode for the
precedence ladder that assigns it, and PLAN.md for why the modes arbitrate
through a single function rather than a set of independent predicates.
"""

from __future__ import annotations

from enum import Enum


class Mode(Enum):
    """Mutually exclusive interpretations of the hand, highest priority first."""

    NONE = "none"            # no hand in frame
    PAUSED = "paused"        # kill switch engaged; nothing dispatches
    COMMAND = "command"      # a classified pose owns this frame
    SCROLL = "scroll"        # fist, vertical motion
    SWIPE = "swipe"          # fist, horizontal motion
    ZOOM = "zoom"            # thumb-index spread, armed
    DRAG = "drag"            # left pinch held past the drag threshold
    CLICK = "click"          # a pinch FSM is pressing or held
    VOLUME = "volume"        # index down, thumb up
    CURSOR = "cursor"        # index extended
    TRACKING = "tracking"    # hand visible, nothing claimed it

    def __str__(self) -> str:
        return self.value


# Modes in which the geometric detectors that read a "resting" hand should not
# run. Kept next to the enum so a new mode has to make an explicit choice.
SUPPRESSES_GEOMETRY = frozenset({Mode.NONE, Mode.PAUSED, Mode.COMMAND})
