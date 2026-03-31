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
    device_index: int = 0
    width: int = 640
    height: int = 480


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MediaPipeConfig:
    max_num_hands: int = 1
    min_detection_tracking: float = 0.8
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
    smooth_factor: float = field(
        default_factory=lambda: _env_float("SMOOTH_FACTOR", 5.0)
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
# Gesture → hotkey map
# ---------------------------------------------------------------------------

GESTURE_MAP: dict[int, dict] = {
    1: {"name": "Spotlight",        "keys": ["command", "space"]},
    2: {"name": "Mission Control",  "keys": ["ctrl", "up"]},
    3: {"name": "App Switcher",     "keys": ["command", "tab"]}
}


# ---------------------------------------------------------------------------
# Queue sizes — bounded queues prevent runaway memory if a thread falls behind
# ---------------------------------------------------------------------------
 
@dataclass(frozen=True)
class QueueConfig:
    inference_queue_size: int = 2
    action_queue_size: int = 1


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
 
 
# Module-level default instance — import this everywhere
DEFAULT_CONFIG = AppConfig()
 

