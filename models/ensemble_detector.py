"""
Ensemble detector that combines Isolation Forest, Random Forest and
LSTM into a single decision via weighted voting.

    score = w_if * IF_anomaly + w_rf * RF_malicious + w_lstm * LSTM_malicious

The attack category label comes from the Random Forest, which is the only
supervised classifier in the stack.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import joblib
import numpy as np

from feature_engineering.preprocessor import Preprocessor
from .isolation_forest import IsolationForestDetector
from .random_forest    import RandomForestClassifier_
from .lstm_model       import LSTMSequenceModel


@dataclass
class EnsemblePrediction:
    is_attack: bool
    score: float                       # ensemble score in [0, 1]
    attack_type: str                   # RF predicted class
    confidence: float                  # RF max probability
    components: Dict[str, float]       # per-model scores

    def to_dict(self):
        return asdict(self)


class EnsembleDetector:

    def __init__(self,
                 preproc: Optional[Preprocessor] = None,
                 if_model: Optional[IsolationForestDetector] = None,
                 rf_model: Optional[RandomForestClassifier_] = None,
                 lstm_model: Optional[LSTMSequenceModel] = None,
                 weights: Optional[Dict[str, float]] = None,
                 threshold: float = 0.65):
        self.preproc    = preproc
        self.if_model   = if_model
        self.rf_model   = rf_model
        self.lstm_model = lstm_model
        self.weights = weights or {
            "isolation_forest": 0.3,
            "random_forest":    0.4,
            "lstm":             0.3,
        }
        self.threshold = threshold

    def predict(self, X_raw: np.ndarray) -> List[EnsemblePrediction]:
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)
        Xp = self.preproc.transform(X_raw) if self.preproc else X_raw

        if_p   = self.if_model.predict_proba(Xp)   if self.if_model   else np.zeros(len(Xp))
        rf_p   = self.rf_model.malicious_proba(Xp) if self.rf_model   else np.zeros(len(Xp))
        lstm_p = self.lstm_model.predict_proba(Xp) if self.lstm_model else np.zeros(len(Xp))

        w = self.weights
        score = (w["isolation_forest"] * if_p
                 + w["random_forest"]  * rf_p
                 + w["lstm"]           * lstm_p)

        if self.rf_model is not None:
            classes = self.rf_model.classes_
            proba   = self.rf_model.predict_proba(Xp)
            cls_idx = proba.argmax(axis=1)
            attack_type = [classes[i] for i in cls_idx]
            confidence  = proba.max(axis=1)
        else:
            attack_type = ["unknown"] * len(Xp)
            confidence  = score

        out: List[EnsemblePrediction] = []
        for i in range(len(Xp)):
            is_attack = bool(score[i] >= self.threshold and attack_type[i] != "benign")
            out.append(EnsemblePrediction(
                is_attack=is_attack,
                score=float(score[i]),
                attack_type=attack_type[i] if is_attack else "benign",
                confidence=float(confidence[i]),
                components={
                    "isolation_forest": float(if_p[i]),
                    "random_forest":    float(rf_p[i]),
                    "lstm":             float(lstm_p[i]),
                },
            ))
        return out

    def predict_one(self, x_raw: np.ndarray) -> EnsemblePrediction:
        return self.predict(x_raw)[0]

    def save(self, base_dir: str) -> None:
        os.makedirs(base_dir, exist_ok=True)
        if self.preproc:    self.preproc.save(os.path.join(base_dir, "preproc.joblib"))
        if self.if_model:   self.if_model.save(os.path.join(base_dir, "iforest.joblib"))
        if self.rf_model:   self.rf_model.save(os.path.join(base_dir, "rforest.joblib"))
        if self.lstm_model: self.lstm_model.save(os.path.join(base_dir, "lstm"))
        joblib.dump({"weights": self.weights,
                     "threshold": self.threshold},
                    os.path.join(base_dir, "ensemble_meta.joblib"))

    @classmethod
    def load(cls, base_dir: str) -> "EnsembleDetector":
        meta = joblib.load(os.path.join(base_dir, "ensemble_meta.joblib"))
        preproc = (Preprocessor.load(os.path.join(base_dir, "preproc.joblib"))
                   if os.path.exists(os.path.join(base_dir, "preproc.joblib")) else None)
        if_m = (IsolationForestDetector.load(os.path.join(base_dir, "iforest.joblib"))
                if os.path.exists(os.path.join(base_dir, "iforest.joblib")) else None)
        rf_m = (RandomForestClassifier_.load(os.path.join(base_dir, "rforest.joblib"))
                if os.path.exists(os.path.join(base_dir, "rforest.joblib")) else None)
        lstm_m = None
        if os.path.exists(os.path.join(base_dir, "lstm.meta")):
            lstm_m = LSTMSequenceModel.load(os.path.join(base_dir, "lstm"))
        return cls(preproc=preproc,
                   if_model=if_m, rf_model=rf_m, lstm_model=lstm_m,
                   weights=meta["weights"], threshold=meta["threshold"])
