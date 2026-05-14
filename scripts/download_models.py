"""
Fetch (or train) pre-trained ensemble models.

In a real release this would pull tarballs from a release server. For the
demo we just delegate to train_model.py so the dashboard always has
something to load.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    saved = ROOT / "models" / "saved" / "ensemble_meta.joblib"
    if saved.exists():
        print(f"Models already present at {saved.parent}")
        return
    print("No pretrained models bundled with this repo. Training a quick "
          "ensemble (~30s) so the demo works out of the box.")
    subprocess.check_call([
        sys.executable,
        str(ROOT / "scripts" / "train_model.py"),
        "--packets", "20000",
        "--epochs",  "3",
        "--no-tf",
    ])


if __name__ == "__main__":
    main()
