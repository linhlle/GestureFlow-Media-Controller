from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any


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


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

def data_path(filename: str) -> Path:
    return PROJECT_ROOT / "data" / filename
 
 
def models_path(filename: str) -> Path:
    return PROJECT_ROOT / "models" / filename




