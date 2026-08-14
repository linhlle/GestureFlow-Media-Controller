from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraConfig:
    device_index: int = field(default_factory=lambda: _env_int("CAMERA_INDEX", 0))
    width: int = 640
    height: int = 480
    # Seconds to wait for a device to report itself open before giving up on
    # this attempt and scheduling a retry.
    open_timeout: float = field(
        default_factory=lambda: _env_float("CAMERA_OPEN_TIMEOUT", 5.0)
    )
    # Consecutive failed reads before we treat the device as gone rather than
    # briefly stalled. At ~30 FPS this is roughly a third of a second.
    read_failure_limit: int = field(
        default_factory=lambda: _env_int("CAMERA_READ_FAILURE_LIMIT", 10)
    )
    reconnect_backoff_min: float = field(
        default_factory=lambda: _env_float("CAMERA_BACKOFF_MIN", 0.5)
    )
    reconnect_backoff_max: float = field(
        default_factory=lambda: _env_float("CAMERA_BACKOFF_MAX", 10.0)
    )


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MediaPipeConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.8
    min_tracking_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Inference / classifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceConfig:
    # Minimum predict_proba score to count a prediction as confident
    confidence_threshold: float = field(
        default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.80)
    )


# ---------------------------------------------------------------------------
# Gesture debounce / stability engine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DebounceConfig:
    vote_window_size: int = field(
        default_factory=lambda: _env_int("VOTE_WINDOW", 10)
    )
    vote_threshold: int = field(
        default_factory=lambda: _env_int("VOTE_THRESHOLD", 7)
    )
    cmd_cooldown: float = field(
        default_factory=lambda: _env_float("CMD_COOLDOWN", 1.3)
    )


# ---------------------------------------------------------------------------
# Mouse tracking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MouseConfig:
    # One Euro filter parameters. A fixed-time-constant low-pass forces a
    # choice between jitter and lag: enough smoothing to settle a still hand
    # is too much to follow a fast one. One Euro adapts its cutoff to hand
    # speed instead, so a nearly-still hand is filtered hard and a fast one
    # is barely filtered at all.
    #
    # min_cutoff (Hz): the cutoff when the hand is stationary. Lower = steadier
    # pointer, but more lag when motion starts.
    min_cutoff: float = field(
        default_factory=lambda: _env_float("CURSOR_MIN_CUTOFF", 0.4)
    )
    # beta: how aggressively the cutoff opens up with speed. Higher = more
    # responsive to fast movement, at the cost of letting more jitter through
    # during it (which the eye does not notice while the pointer is moving).
    beta: float = field(
        default_factory=lambda: _env_float("CURSOR_BETA", 0.002)
    )
    # Cutoff for the internal speed estimate. Rarely worth changing.
    derivative_cutoff: float = field(
        default_factory=lambda: _env_float("CURSOR_DCUTOFF", 0.6)
    )
    # Longest gap the filter will treat as a real interval. A larger gap means
    # the stream stalled (hand left frame, thread descheduled); without this
    # the filter would compute a near-1.0 blend factor and teleport the cursor.
    max_dt: float = field(
        default_factory=lambda: _env_float("CURSOR_MAX_DT", 0.1)
    )
    frame_margin: int = field(
        default_factory=lambda: _env_int("FRAME_MARGIN", 100)
    )


# ---------------------------------------------------------------------------
# Volume control
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolumeConfig:
    sensitivity: float = field(
        default_factory=lambda: _env_float("VOL_SENSITIVITY", 0.02)
    )
    cooldown: float = field(
        default_factory=lambda: _env_float("VOL_COOLDOWN", 0.15)
    )
    step: int = field(
        default_factory=lambda: _env_int("VOL_STEP", 5)
    )
    # Seconds between background polls that re-sync the cached volume value
    # with the real system volume (catches external changes)
    sync_interval: float = field(
        default_factory=lambda: _env_float("VOL_SYNC_INTERVAL", 2.0)
    )

# ---------------------------------------------------------------------------
# Left-click detection (thumb + index pinch FSM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClickConfig:
    # Thresholds are fractions of the hand's own size (wrist to middle MCP),
    # not absolute image distances.  Converted from the previous absolute
    # values using the median hand scale measured over the recorded dataset
    # (0.159), so the feel at the original working distance is unchanged --
    # but the gesture now behaves the same near and far from the camera.
    close_threshold: float = field(
        default_factory=lambda: _env_float("CLICK_CLOSE", 0.28)
    )
    open_threshold: float = field(
        default_factory=lambda: _env_float("CLICK_OPEN", 0.41)
    )
    min_hold_frames: int = field(
        default_factory=lambda: _env_int("CLICK_HOLD_FRAMES", 4)
    )
    cooldown: float = field(
        default_factory=lambda: _env_float("CLICK_COOLDOWN", 0.4)
    )

# ---------------------------------------------------------------------------
# Drag and drop (left pinch held past a threshold)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DragConfig:
    enabled: bool = True
    # Measured from the moment the pinch reaches HELD, not from first contact,
    # so it composes with min_hold_frames instead of racing it. A deliberate
    # click is well under this; a deliberate drag is comfortably over.
    hold_seconds: float = field(
        default_factory=lambda: _env_float("DRAG_HOLD_SECONDS", 0.55)
    )


# ---------------------------------------------------------------------------
# Right-click detection (middle finger + index pinch FSM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RightClickConfig:
    # Tighter than the left-click pinch. Middle and index fingertips sit close
    # together in any curled hand, so this pair needs a deliberate touch to
    # count. A fist additionally suppresses both click FSMs outright -- see
    # InferenceThread._process.
    close_threshold: float = field(
        default_factory=lambda: _env_float("RCLICK_CLOSE", 0.22)
    )
    open_threshold: float = field(
        default_factory=lambda: _env_float("RCLICK_OPEN", 0.38)
    )
    min_hold_frames: int = field(
        default_factory=lambda: _env_int("RCLICK_HOLD_FRAMES", 5)
    )
    cooldown: float = field(
        default_factory=lambda: _env_float("RCLICK_COOLDOWN", 0.6)
    )


# ---------------------------------------------------------------------------
# Scroll detection (fist-drag FSM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScrollConfig:
    # Wrist velocity per frame, as a fraction of hand scale. Also converted
    # from the previous absolute 0.008 at the measured median scale.
    sensitivity: float = field(
        default_factory=lambda: _env_float("SCROLL_SENSITIVITY", 0.05)
    )
    min_hold_frames: int = field(
        default_factory=lambda: _env_int("SCROLL_HOLD_FRAMES", 5)
    )
    cooldown: float = field(
        default_factory=lambda: _env_float("SCROLL_COOLDOWN", 0.05)
    )
    step: int = field(
        default_factory=lambda: _env_int("SCROLL_STEP", 2)
    )
    velocity_exponent: float = field(
        default_factory=lambda: _env_float("SCROLL_EXPONENT", 1.6)
    )
    # Vertical travel must at least tie horizontal for a motion to count as a
    # scroll. Swipe reads the other side of the same rule, so one hand movement
    # can never be both.
    axis_ratio: float = field(
        default_factory=lambda: _env_float("SCROLL_AXIS_RATIO", 1.0)
    )




# ---------------------------------------------------------------------------
# Queue sizes — bounded queues prevent runaway memory if a thread falls behind
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueueConfig:
    inference_queue_size: int = 2
    action_queue_size: int = 1


# ---------------------------------------------------------------------------
# Zoom (thumb-index spread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZoomConfig:
    enabled: bool = True
    # Thumb-index gap, in hand-widths, below which zoom will not arm. Sits
    # above the click's open_threshold on purpose: a pinch closed enough to
    # click can never be read as a zoom.
    min_separation: float = field(
        default_factory=lambda: _env_float("ZOOM_MIN_SEPARATION", 0.55)
    )
    # Change in that gap, per frame, before a zoom step fires.
    sensitivity: float = field(
        default_factory=lambda: _env_float("ZOOM_SENSITIVITY", 0.06)
    )
    min_hold_frames: int = field(
        default_factory=lambda: _env_int("ZOOM_HOLD_FRAMES", 4)
    )
    cooldown: float = field(
        default_factory=lambda: _env_float("ZOOM_COOLDOWN", 0.12)
    )
    # How far the other three fingertips must sit below their knuckles.
    curl_margin: float = field(
        default_factory=lambda: _env_float("ZOOM_CURL_MARGIN", 0.12)
    )
    # Angle between thumb and index. This is what separates a zoom pose from
    # ordinary pointing -- a pointing hand also holds the thumb away from the
    # index, so distance alone would read a cursor gesture as a zoom.
    # Measured over the recorded dataset: a relaxed Neutral hand sits around
    # 36 degrees and 65 gives it zero false positives, which is the number that
    # matters since Neutral is the state every geometric mode lives in.
    min_angle_degrees: float = field(
        default_factory=lambda: _env_float("ZOOM_MIN_ANGLE", 65.0)
    )


# ---------------------------------------------------------------------------
# Dwell click (accessibility)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DwellConfig:
    # Off by default: when this is on, resting the pointer clicks. That is the
    # point of the feature and also deeply surprising if you did not ask for it.
    enabled: bool = field(
        default_factory=lambda: _env_bool("DWELL_ENABLED", False)
    )
    seconds: float = field(
        default_factory=lambda: _env_float("DWELL_SECONDS", 1.0)
    )
    # In screen pixels, because the thing being held still is the pointer the
    # user can see, not the hand.
    radius_px: float = field(
        default_factory=lambda: _env_float("DWELL_RADIUS_PX", 40.0)
    )


# ---------------------------------------------------------------------------
# Swipe (horizontal flick on a held fist)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwipeConfig:
    enabled: bool = True
    # Hand-widths of horizontal travel per frame before a flick counts.
    # Comfortably above the drift of a hand trying to hold still.
    sensitivity: float = field(
        default_factory=lambda: _env_float("SWIPE_SENSITIVITY", 0.16)
    )
    min_hold_frames: int = field(
        default_factory=lambda: _env_int("SWIPE_HOLD_FRAMES", 3)
    )
    cooldown: float = field(
        default_factory=lambda: _env_float("SWIPE_COOLDOWN", 0.6)
    )
    # Horizontal must beat vertical by this factor. Above 1.0 on purpose: an
    # ambiguous diagonal should fire nothing rather than guess.
    axis_ratio: float = field(
        default_factory=lambda: _env_float("SWIPE_AXIS_RATIO", 1.5)
    )
    # Speed must fall to this fraction of the threshold before the next swipe
    # can start, so one flick is one fire rather than one per frame.
    release_ratio: float = field(
        default_factory=lambda: _env_float("SWIPE_RELEASE_RATIO", 0.5)
    )


# ---------------------------------------------------------------------------
# Pause / resume kill switch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PauseConfig:
    enabled: bool = True
    # Held continuously; any frame that breaks the pose resets the timer.
    # Long enough to be deliberate, short enough not to be a chore.
    hold_seconds: float = field(
        default_factory=lambda: _env_float("PAUSE_HOLD_SECONDS", 1.5)
    )
    # Fraction of hand scale a finger must clear its joint by to count as
    # extended or curled.
    margin: float = field(
        default_factory=lambda: _env_float("PAUSE_MARGIN", 0.25)
    )


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HudConfig:
    # If drawing one frame costs more than 1/min_fps seconds, the HUD sheds
    # its most expensive element (the landmark skeleton) until it fits again.
    # Recognition is never affected -- only the overlay gets simpler.
    min_fps: float = field(default_factory=lambda: _env_float("HUD_MIN_FPS", 20.0))


# ---------------------------------------------------------------------------
# Top-level convenience object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    debounce: DebounceConfig = field(default_factory=DebounceConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    queues: QueueConfig = field(default_factory=QueueConfig)
    click: ClickConfig = field(default_factory=ClickConfig)
    drag: DragConfig = field(default_factory=DragConfig)
    right_click: RightClickConfig = field(default_factory=RightClickConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    swipe: SwipeConfig = field(default_factory=SwipeConfig)
    dwell: DwellConfig = field(default_factory=DwellConfig)
    zoom: ZoomConfig = field(default_factory=ZoomConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    pause: PauseConfig = field(default_factory=PauseConfig)

# Module-level default instance — import this everywhere
DEFAULT_CONFIG = AppConfig()


