from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

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
    # Time constant of the cursor smoothing filter, in seconds.  Replaces the
    # old per-frame SMOOTH_FACTOR, which made the cursor feel different at
    # every frame rate.  Larger = smoother and laggier; the filter closes
    # ~63% of the remaining distance to the target every tau seconds.
    smoothing_tau: float = field(
        default_factory=lambda: _env_float("SMOOTH_TAU", 0.15)
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
    close_threshold: float = field(
        default_factory=lambda: _env_float("CLICK_CLOSE", 0.045)
    )
    open_threshold: float = field(
        default_factory=lambda: _env_float("CLICK_OPEN", 0.065)
    )
    min_hold_frames: int = field(
        default_factory=lambda: _env_int("CLICK_HOLD_FRAMES", 4)
    )
    cooldown: float = field(
        default_factory=lambda: _env_float("CLICK_COOLDOWN", 0.4)
    )

# ---------------------------------------------------------------------------
# Right-click detection (middle finger + index pinch FSM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RightClickConfig:
    close_threshold: float = field(
        default_factory=lambda: _env_float("RCLICK_CLOSE", 0.045)
    )
    open_threshold: float = field(
        default_factory=lambda: _env_float("RCLICK_OPEN", 0.065)
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
    sensitivity: float = field(
        default_factory=lambda: _env_float("SCROLL_SENSITIVITY", 0.008)
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




# ---------------------------------------------------------------------------
# Queue sizes — bounded queues prevent runaway memory if a thread falls behind
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueueConfig:
    inference_queue_size: int = 2
    action_queue_size: int = 1


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
    right_click: RightClickConfig = field(default_factory=RightClickConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    hud: HudConfig = field(default_factory=HudConfig)

# Module-level default instance — import this everywhere
DEFAULT_CONFIG = AppConfig()


