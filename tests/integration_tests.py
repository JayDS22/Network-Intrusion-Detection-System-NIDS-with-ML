"""End-to-end tests: simulator, extractor, ensemble, stream processor."""
from __future__ import annotations

import time

import numpy as np

from data_collection.traffic_simulator import TrafficSimulator
from feature_engineering.extractor    import FeatureExtractor
from feature_engineering.preprocessor import Preprocessor
from models.ensemble_detector         import EnsembleDetector
from models.isolation_forest          import IsolationForestDetector
from models.lstm_model                import LSTMSequenceModel
from models.random_forest             import RandomForestClassifier_
from real_time.rule_engine            import RuleEngine
from real_time.stream_processor       import StreamProcessor


def _quick_ensemble():
    pkts = TrafficSimulator(seed=11, attack_probability=0.3).sample_batch(1500)
    X = FeatureExtractor().transform_batch(pkts)
    y = np.array([p["label"] for p in pkts])
    yb = (y != "benign").astype(int)
    pre = Preprocessor().fit(X)
    Xp  = pre.transform(X)
    return EnsembleDetector(
        preproc=pre,
        if_model=IsolationForestDetector(n_estimators=25).fit(Xp),
        rf_model=RandomForestClassifier_(n_estimators=30).fit(Xp, y),
        lstm_model=LSTMSequenceModel(sequence_length=4, epochs=1,
                                     use_tf=False).fit(Xp, yb),
        threshold=0.5,
    )


def test_stream_processor_produces_events():
    ens = _quick_ensemble()
    sim = TrafficSimulator(seed=22, attack_probability=0.5)
    src = iter(sim.sample_batch(800))     # fast finite source
    proc = StreamProcessor(ensemble=ens, rule_engine=RuleEngine(),
                           source=src, extractor=FeatureExtractor())
    proc.start()
    time.sleep(2.0)
    proc.stop()
    events  = proc.snapshot_events()
    metrics = proc.snapshot_metrics()
    assert metrics["packets"] > 0
    assert metrics["alerts"]  > 0
    assert len(events) > 0
    assert any(e["detector"] in ("ml", "rule", "ml+rule") for e in events)
