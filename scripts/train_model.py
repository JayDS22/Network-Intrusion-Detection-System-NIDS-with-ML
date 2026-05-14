"""
Train the NIDS ensemble (Isolation Forest + Random Forest + LSTM) on
synthetic traffic and persist artefacts to models/saved.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --packets 60000 --epochs 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_collection.traffic_simulator import generate_labeled_dataset
from feature_engineering.extractor    import FeatureExtractor
from feature_engineering.preprocessor import Preprocessor
from models.isolation_forest          import IsolationForestDetector
from models.random_forest             import RandomForestClassifier_
from models.lstm_model                import LSTMSequenceModel
from models.ensemble_detector         import EnsembleDetector


def _section(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Train NIDS ensemble")
    ap.add_argument("--packets", type=int, default=40_000,
                    help="Number of synthetic packets to generate")
    ap.add_argument("--epochs", type=int, default=4,
                    help="LSTM training epochs")
    ap.add_argument("--no-tf", action="store_true",
                    help="Skip TensorFlow LSTM and use sklearn fallback")
    ap.add_argument("--out", default=str(ROOT / "models" / "saved"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _section(f"[1/5] Generating {args.packets:,} synthetic packets")
    t0 = time.time()
    pkts = generate_labeled_dataset(args.packets, seed=args.seed)
    print(f"  done in {time.time()-t0:.1f}s, "
          f"{sum(1 for p in pkts if p['label']!='benign'):,} malicious")

    _section("[2/5] Extracting features (40 per packet)")
    t0 = time.time()
    extractor = FeatureExtractor(window_seconds=30)
    X = extractor.transform_batch(pkts)
    y = np.array([p["label"] for p in pkts])
    y_bin = (y != "benign").astype(np.int32)
    print(f"  feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")

    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                      random_state=args.seed, stratify=y_bin)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    yb_tr, yb_te = y_bin[idx_tr], y_bin[idx_te]

    _section("[3/5] Fitting preprocessor (StandardScaler + PCA 95%)")
    pre = Preprocessor(scaler="standard", pca_variance=0.95)
    Xp_tr = pre.fit_transform(X_tr)
    Xp_te = pre.transform(X_te)
    print(f"  reduced dim: {X_tr.shape[1]} -> {Xp_tr.shape[1]}")

    _section("[4/5] Training individual detectors")
    print("  Isolation Forest..."); t0 = time.time()
    iforest = IsolationForestDetector(n_estimators=100, contamination=0.15,
                                      random_state=args.seed).fit(Xp_tr)
    print(f"    fit in {time.time()-t0:.1f}s")

    print("  Random Forest...");    t0 = time.time()
    rforest = RandomForestClassifier_(n_estimators=200, max_depth=20,
                                      random_state=args.seed).fit(Xp_tr, y_tr)
    print(f"    fit in {time.time()-t0:.1f}s  "
          f"({len(rforest.classes_)} classes)")

    print("  LSTM sequence model..."); t0 = time.time()
    lstm = LSTMSequenceModel(sequence_length=8,
                             hidden_units=48,
                             epochs=args.epochs,
                             batch_size=128,
                             use_tf=not args.no_tf).fit(Xp_tr, yb_tr)
    print(f"    fit in {time.time()-t0:.1f}s "
          f"(backend: {'TensorFlow' if lstm.use_tf else 'sklearn'})")

    _section("[5/5] Building ensemble and evaluating")
    ensemble = EnsembleDetector(
        preproc=pre, if_model=iforest, rf_model=rforest, lstm_model=lstm,
        weights={"isolation_forest": 0.3,
                 "random_forest":    0.4,
                 "lstm":             0.3},
        threshold=0.55,
    )
    preds = ensemble.predict(X_te)
    yp_bin = np.array([1 if p.is_attack else 0 for p in preds])
    yp_lbl = np.array([p.attack_type for p in preds])
    score  = np.array([p.score for p in preds])

    acc  = accuracy_score(yb_te, yp_bin)
    auc  = roc_auc_score(yb_te, score) if len(set(yb_te)) > 1 else float("nan")
    print(f"  Binary accuracy  : {acc*100:.2f}%")
    print(f"  AUC              : {auc:.3f}")
    fpr = np.mean(yp_bin[yb_te == 0]) * 100 if (yb_te == 0).any() else 0
    print(f"  False positive   : {fpr:.2f}%")
    print(f"  Multiclass acc.  : {accuracy_score(y_te, yp_lbl)*100:.2f}%")
    print()
    print(classification_report(y_te, yp_lbl, zero_division=0))

    out = args.out
    os.makedirs(out, exist_ok=True)
    ensemble.save(out)

    import json
    with open(os.path.join(out, "training_metrics.json"), "w") as fh:
        json.dump({"accuracy": float(acc), "auc": float(auc),
                   "fpr": float(fpr / 100),
                   "n_classes": len(rforest.classes_),
                   "n_features_pca": int(Xp_tr.shape[1]),
                   "lstm_backend": "tf" if lstm.use_tf else "sklearn"}, fh,
                  indent=2)

    _section(f"Ensemble saved to {out}")


if __name__ == "__main__":
    main()
