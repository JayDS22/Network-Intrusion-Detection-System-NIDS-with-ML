"""
Write a labelled synthetic dataset to data/synthetic for use by
external trainers (notebooks, Spark jobs, etc.).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_collection.traffic_simulator import generate_labeled_dataset
from feature_engineering.extractor    import FEATURE_NAMES, FeatureExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=int, default=50_000)
    ap.add_argument("--out",
                    default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pkts = generate_labeled_dataset(args.packets, seed=args.seed)

    pd.DataFrame(pkts).to_csv(os.path.join(args.out, "packets.csv"), index=False)

    extractor = FeatureExtractor(window_seconds=30)
    X = extractor.transform_batch(pkts)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = [p["label"] for p in pkts]
    df.to_csv(os.path.join(args.out, "features.csv"), index=False)

    print(f"Wrote {args.packets:,} rows to {args.out}/{{packets.csv,features.csv}}")


if __name__ == "__main__":
    main()
