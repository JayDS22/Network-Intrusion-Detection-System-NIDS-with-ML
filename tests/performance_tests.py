"""Throughput / latency benchmark."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_collection.traffic_simulator import generate_labeled_dataset
from feature_engineering.extractor    import FeatureExtractor
from feature_engineering.preprocessor import Preprocessor
from models.ensemble_detector         import EnsembleDetector
from models.isolation_forest          import IsolationForestDetector
from models.lstm_model                import LSTMSequenceModel
from models.random_forest             import RandomForestClassifier_


def build():
    pkts = generate_labeled_dataset(3000, seed=0)
    X = FeatureExtractor().transform_batch(pkts)
    y = np.array([p["label"] for p in pkts])
    yb = (y != "benign").astype(int)
    pre = Preprocessor().fit(X)
    Xp = pre.transform(X)
    return EnsembleDetector(
        preproc=pre,
        if_model=IsolationForestDetector(n_estimators=40).fit(Xp),
        rf_model=RandomForestClassifier_(n_estimators=40).fit(Xp, y),
        lstm_model=LSTMSequenceModel(sequence_length=4, epochs=1,
                                     use_tf=False).fit(Xp, yb),
        threshold=0.55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=int, default=50_000)
    args = ap.parse_args()

    print(f"Building ensemble...")
    ens = build()
    print(f"Generating {args.packets:,} packets...")
    pkts = generate_labeled_dataset(args.packets, seed=99)
    ex = FeatureExtractor()
    X = ex.transform_batch(pkts)

    t0 = time.perf_counter()
    preds = ens.predict(X)
    elapsed = time.perf_counter() - t0
    pps = len(pkts) / elapsed
    print(f"\n=== Performance ===")
    print(f" packets    : {len(pkts):,}")
    print(f" elapsed    : {elapsed:.2f}s")
    print(f" throughput : {pps:,.0f} packets/sec")
    print(f" latency    : {elapsed/len(pkts)*1000:.3f} ms/packet")
    print(f" alerts     : {sum(p.is_attack for p in preds):,}")


if __name__ == "__main__":
    main()
