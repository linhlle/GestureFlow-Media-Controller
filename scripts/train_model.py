"""Train the gesture classifier and persist both the model and its metrics.

Run from anywhere:  python scripts/train_model.py [--no-plot]

Writes:
  models/gesture_classifier.pkl
  models/metrics.json          <- so evaluation numbers survive the run
  models/confusion_matrix.png  <- unless --no-plot
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gestureflow.utils import data_path, models_path  # noqa: E402

TARGET_NAMES = ["Neutral", "L-Shape", "High-Five", "2-Finger"]
N_ESTIMATORS = 100
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Rows are logged one per video frame while a pose is held, so consecutive
# rows are near-duplicates of each other.  A random split therefore puts frame
# k in train and frame k+1 in test, and the reported accuracy is optimistic.
# It is recorded anyway -- but recorded *with this caveat attached*, so nobody
# quotes it as a generalization estimate.
SPLIT_CAVEAT = (
    "Random split over per-frame samples. Consecutive rows are adjacent video "
    "frames of the same held pose, so train and test are correlated and this "
    "accuracy is an optimistic upper bound, not a generalization estimate. A "
    "grouped split by recording session is needed for that."
)


def train_model(make_plot: bool = True) -> dict:
    csv_file = data_path("gesture_data.csv")
    if not csv_file.exists():
        print(f"[train] ERROR: no dataset at {csv_file}")
        print("[train] Run scripts/data_logger.py to collect samples first.")
        return {}

    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"[train] Training on {len(X_train)} frames")
    print(f"[train] Testing on  {len(X_test)} frames")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 62)
    print(f"Hold-out accuracy: {accuracy * 100:.2f}%")
    print("=" * 62)
    print(f"\nCAVEAT: {SPLIT_CAVEAT}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    model_file = models_path("gesture_classifier.pkl")
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as f:
        pickle.dump(model, f)
    print(f"[train] Model written to {model_file}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(csv_file.relative_to(csv_file.parent.parent)),
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        },
        "split": {
            "kind": "random_stratified",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "caveat": SPLIT_CAVEAT,
        },
        "model": {
            "kind": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "classes": [int(c) for c in model.classes_],
        },
        "metrics": {
            "accuracy": accuracy,
            "per_class": {
                TARGET_NAMES[int(label)] if int(label) < len(TARGET_NAMES)
                else str(label): stats
                for label, stats in report.items()
                if label.isdigit()
            },
            "macro_avg": report.get("macro avg"),
            "weighted_avg": report.get("weighted avg"),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": [
                TARGET_NAMES[i] if i < len(TARGET_NAMES) else str(i)
                for i in range(cm.shape[0])
            ],
        },
    }

    metrics_file = models_path("metrics.json")
    with metrics_file.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics written to {metrics_file}")

    if make_plot:
        _plot_confusion(cm, models_path("confusion_matrix.png"))

    return metrics


def _plot_confusion(cm, out_file: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[train] matplotlib/seaborn not installed; skipping plot.")
        return

    labels = TARGET_NAMES[: cm.shape[0]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Gesture recognition confusion matrix")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[train] Confusion matrix written to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true",
                        help="skip writing confusion_matrix.png")
    args = parser.parse_args()
    train_model(make_plot=not args.no_plot)
