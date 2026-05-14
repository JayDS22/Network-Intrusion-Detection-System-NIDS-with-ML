"""
Sequence-based detector.

When TensorFlow is available, trains a small LSTM that consumes a rolling
window of feature vectors and returns a malicious probability.

Falls back to a sklearn GradientBoostingClassifier on the flattened
window when TF is missing or hangs on import.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import importlib
import importlib.util

import joblib
import numpy as np

# Detect TF without importing it. The actual import can deadlock on some
# macOS Pythons, so defer it until the TF branch is taken.
_HAVE_TF = importlib.util.find_spec("tensorflow") is not None


def _import_tf():
    tf = importlib.import_module("tensorflow")
    layers    = importlib.import_module("tensorflow.keras.layers")
    models    = importlib.import_module("tensorflow.keras.models")
    callbacks = importlib.import_module("tensorflow.keras.callbacks")
    return tf, layers, models, callbacks


class LSTMSequenceModel:
    """Predicts the probability that the most recent packet is malicious,
    using a sliding window of the preceding feature vectors as context."""

    def __init__(self,
                 sequence_length: int = 10,
                 hidden_units: int = 64,
                 epochs: int = 5,
                 batch_size: int = 64,
                 use_tf: Optional[bool] = None):
        self.sequence_length = sequence_length
        self.hidden_units    = hidden_units
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.use_tf          = _HAVE_TF if use_tf is None else use_tf and _HAVE_TF
        self._model = None
        self._n_features: Optional[int] = None

    def _make_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        seqs = []
        labels: List = []
        sl = self.sequence_length
        for i in range(len(X)):
            if i + 1 < sl:
                pad = np.zeros((sl - i - 1, X.shape[1]), dtype=X.dtype)
                window = np.vstack([pad, X[: i + 1]])
            else:
                window = X[i + 1 - sl : i + 1]
            seqs.append(window)
            if y is not None:
                labels.append(y[i])
        seqs_arr = np.asarray(seqs, dtype=np.float32)
        if y is None:
            return seqs_arr, None
        return seqs_arr, np.asarray(labels)

    def fit(self, X: np.ndarray, y_binary: np.ndarray) -> "LSTMSequenceModel":
        self._n_features = X.shape[1]
        Xs, ys = self._make_sequences(X, y_binary.astype(np.float32))
        if self.use_tf:
            self._fit_tf(Xs, ys)
        else:
            self._fit_fallback(Xs, ys)
        return self

    def _fit_tf(self, Xs: np.ndarray, ys: np.ndarray) -> None:
        tf, layers, models, callbacks = _import_tf()
        tf.random.set_seed(42)
        model = models.Sequential([
            layers.Input(shape=(self.sequence_length, self._n_features)),
            layers.Masking(mask_value=0.0),
            layers.LSTM(self.hidden_units, return_sequences=False),
            layers.Dropout(0.25),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        early = callbacks.EarlyStopping(patience=2, restore_best_weights=True,
                                        monitor="val_loss")
        model.fit(Xs, ys,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_split=0.15,
                  verbose=0,
                  callbacks=[early])
        self._model = model

    def _fit_fallback(self, Xs: np.ndarray, ys: np.ndarray) -> None:
        from sklearn.ensemble import GradientBoostingClassifier
        flat = Xs.reshape(Xs.shape[0], -1)
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                         random_state=42)
        clf.fit(flat, ys)
        self._model = clf

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs, _ = self._make_sequences(X)
        if self.use_tf:
            return self._model.predict(Xs, verbose=0).ravel()
        return self._model.predict_proba(Xs.reshape(Xs.shape[0], -1))[:, 1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = {"sequence_length": self.sequence_length,
                "hidden_units":    self.hidden_units,
                "epochs":          self.epochs,
                "batch_size":      self.batch_size,
                "n_features":      self._n_features,
                "use_tf":          self.use_tf}
        joblib.dump(meta, path + ".meta")
        if self.use_tf:
            self._model.save(path + ".keras")
        else:
            joblib.dump(self._model, path + ".pkl")

    @classmethod
    def load(cls, path: str) -> "LSTMSequenceModel":
        meta = joblib.load(path + ".meta")
        obj = cls(sequence_length=meta["sequence_length"],
                  hidden_units=meta["hidden_units"],
                  epochs=meta["epochs"],
                  batch_size=meta["batch_size"],
                  use_tf=meta["use_tf"])
        obj._n_features = meta["n_features"]
        if meta["use_tf"]:
            tf, *_ = _import_tf()
            obj._model = tf.keras.models.load_model(path + ".keras")
        else:
            obj._model = joblib.load(path + ".pkl")
        return obj
