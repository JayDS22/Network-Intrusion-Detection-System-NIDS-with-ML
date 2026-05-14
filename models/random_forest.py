"""Random Forest classifier over attack category labels."""
from __future__ import annotations

import os
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifier_:
    """Wraps sklearn RandomForestClassifier and persists its class list."""

    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 20,
                 class_weight: str = "balanced",
                 random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        self.classes_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier_":
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def malicious_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability mass assigned to any non-benign class."""
        proba = self.model.predict_proba(X)
        if "benign" in self.classes_:
            i = self.classes_.index("benign")
            return 1.0 - proba[:, i]
        return proba.max(axis=1)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestClassifier_":
        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = blob["model"]
        obj.classes_ = blob["classes"]
        return obj
