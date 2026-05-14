"""Isolation Forest wrapper for unsupervised anomaly scoring."""
from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Returns a [0, 1] anomaly score where 1 is highly anomalous."""

    def __init__(self,
                 n_estimators: int = 100,
                 contamination: float = 0.1,
                 random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._score_min: Optional[float] = None
        self._score_max: Optional[float] = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        # Store training-time score range so predict_proba can min-max scale.
        raw = -self.model.score_samples(X)
        self._score_min = float(np.min(raw))
        self._score_max = float(np.max(raw))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = -self.model.score_samples(X)
        if self._score_max is None or self._score_max == self._score_min:
            return np.clip(raw, 0, 1)
        return np.clip(
            (raw - self._score_min) / (self._score_max - self._score_min),
            0.0, 1.0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return -1 for anomaly, +1 for normal."""
        return self.model.predict(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model,
                     "score_min": self._score_min,
                     "score_max": self._score_max}, path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = blob["model"]
        obj._score_min = blob["score_min"]
        obj._score_max = blob["score_max"]
        return obj
