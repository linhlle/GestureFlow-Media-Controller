"""Print a summary of the collected gesture dataset.

Run from anywhere:  python scripts/data_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gestureflow.utils import data_path  # noqa: E402


def main() -> None:
    csv_file = data_path("gesture_data.csv")
    if not csv_file.exists():
        print(f"No dataset at {csv_file}")
        return

    df = pd.read_csv(csv_file)
    print("--- DATASET SUMMARY ---")
    print(f"Path:           {csv_file}")
    print(f"Total frames:   {len(df)}")
    print(f"Feature columns:{df.shape[1] - 1}")
    print("\nSamples per gesture:")
    print(df["label"].value_counts().sort_index().to_string())
    print(f"\nMissing values: {df.isnull().sum().sum()}")

    dupes = df.duplicated().sum()
    if dupes:
        print(f"Exact duplicate rows: {dupes}")


if __name__ == "__main__":
    main()
