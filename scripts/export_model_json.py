"""Export the Random Forest to JSON so the browser can run the same classifier.

The web demo must agree with the desktop app about what a gesture is. Two ways
to achieve that: reimplement the model in JS (guaranteed to drift), or export
the actual fitted trees and evaluate them with a few lines of JS (cannot
drift). This does the second.

The forest is small -- 100 trees, ~3,300 nodes total -- so the whole thing
ships in full fidelity rather than being pruned or approximated.

Layout: parallel arrays per tree, which is how sklearn stores them internally
and which minifies far better than nested node objects.

Run:  python scripts/export_model_json.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gestureflow.utils import models_path  # noqa: E402

# sklearn's sentinels: children_left/right hold TREE_LEAF (-1) at a leaf, while
# `feature` holds TREE_UNDEFINED (-2) there. Confusing the two produces a
# traversal that never reaches a leaf and loops forever.
_LEAF = -1
# Thresholds and probabilities are rounded before export. Landmark coordinates
# carry nowhere near this much precision, and it roughly halves the file.
_THRESHOLD_DP = 6
_PROBA_DP = 5


def export(model_path: Path, out_path: Path, indent: int | None = None) -> dict:
    with model_path.open("rb") as f:
        model = pickle.load(f)

    trees = []
    for estimator in model.estimators_:
        t = estimator.tree_
        # value[i] holds class counts at node i; normalize to probabilities so
        # the browser does not have to.
        values = []
        for row in t.value:
            counts = row[0]
            total = float(counts.sum())
            if total <= 0:
                values.append([0.0] * len(counts))
            else:
                values.append([round(float(c) / total, _PROBA_DP) for c in counts])

        trees.append({
            "feature": [int(v) for v in t.feature],
            "threshold": [round(float(v), _THRESHOLD_DP) for v in t.threshold],
            "left": [int(v) for v in t.children_left],
            "right": [int(v) for v in t.children_right],
            "value": values,
        })

    payload = {
        "schema": "gestureflow.forest/1",
        "kind": "RandomForestClassifier",
        "n_features": int(model.n_features_in_),
        "classes": [int(c) for c in model.classes_],
        "n_trees": len(trees),
        "leaf_marker": _LEAF,
        "trees": trees,
        "note": (
            "Exported from models/gesture_classifier.pkl. Evaluating these "
            "trees in JS reproduces sklearn's predict_proba: average the "
            "per-tree class probabilities."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=indent, separators=(",", ":")
                  if indent is None else None)

    total_nodes = sum(len(t["feature"]) for t in trees)
    size_kb = out_path.stat().st_size / 1024.0
    print(f"[export] {len(trees)} trees, {total_nodes} nodes, "
          f"{model.n_features_in_} features, classes {list(model.classes_)}")
    print(f"[export] Wrote {out_path} ({size_kb:.0f} KB)")
    return payload


def verify(model_path: Path, payload: dict, n_samples: int = 200) -> bool:
    """Check the exported trees reproduce sklearn's predict_proba.

    Without this, a subtle export bug would give the website a classifier that
    quietly disagrees with the desktop app -- the exact drift the export was
    meant to prevent.
    """
    import numpy as np

    with model_path.open("rb") as f:
        model = pickle.load(f)

    rng = np.random.default_rng(0)
    samples = rng.uniform(-1.0, 1.0, size=(n_samples, payload["n_features"]))
    expected = model.predict_proba(samples)

    worst = 0.0
    for i, sample in enumerate(samples):
        got = _predict_proba_py(payload, sample)
        worst = max(worst, float(np.max(np.abs(np.array(got) - expected[i]))))

    ok = worst < 1e-4
    status = "OK" if ok else "MISMATCH"
    print(f"[export] Verification {status}: max probability delta "
          f"{worst:.2e} over {n_samples} random samples")
    return ok


def _predict_proba_py(payload: dict, features) -> list:
    """Reference implementation, mirroring the JS in web/js/forest.js."""
    n_classes = len(payload["classes"])
    totals = [0.0] * n_classes
    leaf = payload["leaf_marker"]

    for tree in payload["trees"]:
        node = 0
        while tree["left"][node] != leaf:
            if features[tree["feature"][node]] <= tree["threshold"][node]:
                node = tree["left"][node]
            else:
                node = tree["right"][node]
        for c, p in enumerate(tree["value"][node]):
            totals[c] += p

    n = len(payload["trees"])
    return [t / n for t in totals]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=models_path("gesture_classifier.pkl"))
    parser.add_argument("--out", type=Path,
                        default=Path("web/models/forest.json"))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    payload = export(args.model, args.out)
    if not args.skip_verify:
        if not verify(args.model, payload):
            sys.exit(1)
