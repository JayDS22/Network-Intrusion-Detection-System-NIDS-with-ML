"""Evaluate a saved NIDS ensemble against a freshly generated test set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_collection.traffic_simulator import generate_labeled_dataset
from feature_engineering.extractor    import FeatureExtractor
from models.ensemble_detector         import EnsembleDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "saved"))
    ap.add_argument("--packets", type=int, default=15_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"Loading ensemble from {args.model}")
    ens = EnsembleDetector.load(args.model)

    print(f"Generating {args.packets:,} test packets...")
    pkts = generate_labeled_dataset(args.packets, seed=args.seed)
    X = FeatureExtractor(window_seconds=30).transform_batch(pkts)
    y = np.array([p["label"] for p in pkts])
    y_bin = (y != "benign").astype(np.int32)

    preds = ens.predict(X)
    yp_bin = np.array([1 if p.is_attack else 0 for p in preds])
    yp_lbl = np.array([p.attack_type for p in preds])
    score  = np.array([p.score for p in preds])

    acc   = accuracy_score(y_bin, yp_bin)
    multi = accuracy_score(y, yp_lbl)
    auc   = roc_auc_score(y_bin, score) if len(set(y_bin)) > 1 else float("nan")
    fpr   = float(np.mean(yp_bin[y_bin == 0])) if (y_bin == 0).any() else 0.0
    tpr   = float(np.mean(yp_bin[y_bin == 1])) if (y_bin == 1).any() else 0.0

    print("\n=== Performance ===")
    print(f" binary accuracy : {acc*100:.2f}%")
    print(f" multiclass acc  : {multi*100:.2f}%")
    print(f" AUC             : {auc:.3f}")
    print(f" true positive   : {tpr*100:.2f}%")
    print(f" false positive  : {fpr*100:.2f}%\n")
    print(classification_report(y, yp_lbl, zero_division=0))

    print("Confusion matrix (binary):")
    print(confusion_matrix(y_bin, yp_bin))


if __name__ == "__main__":
    main()
