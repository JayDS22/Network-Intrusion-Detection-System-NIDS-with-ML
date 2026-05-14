"""
Feature scaling and PCA for the ensemble.

Fits a scaler (default StandardScaler) and a PCA preserving 95% variance.
Persists both to a single joblib file so inference uses an identical
transform.
"""
from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


_SCALERS = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
}


class Preprocessor:

    def __init__(self, scaler: str = "standard", pca_variance: float = 0.95):
        if scaler not in _SCALERS:
            raise ValueError(f"Unknown scaler '{scaler}'")
        self.scaler = _SCALERS[scaler]()
        self.pca: Optional[PCA] = (
            PCA(n_components=pca_variance, svd_solver="full")
            if 0 < pca_variance < 1
            else None
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> "Preprocessor":
        Xs = self.scaler.fit_transform(X)
        if self.pca is not None:
            self.pca.fit(Xs)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform.")
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs) if self.pca is not None else Xs

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self.scaler,
                     "pca":    self.pca,
                     "fitted": self._fitted}, path)

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.scaler  = blob["scaler"]
        obj.pca     = blob["pca"]
        obj._fitted = blob["fitted"]
        return obj
