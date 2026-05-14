"""Unit tests for feature extraction, simulator, models, rule engine."""
from __future__ import annotations

import numpy as np

from data_collection.traffic_simulator import (TrafficSimulator, AttackType,
                                               generate_labeled_dataset)
from feature_engineering.extractor    import FEATURE_NAMES, FeatureExtractor
from feature_engineering.preprocessor import Preprocessor
from feature_engineering.sliding_window import SlidingWindow
from models.isolation_forest          import IsolationForestDetector
from models.random_forest             import RandomForestClassifier_
from models.lstm_model                import LSTMSequenceModel
from models.ensemble_detector         import EnsembleDetector
from real_time.rule_engine            import RuleEngine


def test_simulator_emits_benign_and_attacks():
    sim = TrafficSimulator(seed=1, attack_probability=0.5)
    pkts = sim.sample_batch(500)
    labels = {p["label"] for p in pkts}
    assert AttackType.BENIGN.value in labels
    assert len(labels) > 1, "simulator should produce multiple categories"


def test_simulator_packet_shape():
    sim = TrafficSimulator(seed=2)
    p = sim.sample_batch(1)[0]
    for key in ("ts", "src_ip", "dst_ip", "src_port", "dst_port",
                "protocol", "length", "label"):
        assert key in p


def test_feature_extractor_shape():
    ex = FeatureExtractor(window_seconds=30)
    pkts = TrafficSimulator(seed=3).sample_batch(200)
    X = ex.transform_batch(pkts)
    # sample_batch may emit > 200 (attack bursts inflate the count)
    assert X.shape == (len(pkts), len(FEATURE_NAMES))
    assert np.all(np.isfinite(X))


def test_feature_extractor_streaming_consistency():
    pkts = TrafficSimulator(seed=4).sample_batch(50)
    ex1, ex2 = FeatureExtractor(), FeatureExtractor()
    batch = ex1.transform_batch(pkts)
    rows  = np.stack([ex2.transform_packet(p) for p in pkts])
    assert np.allclose(batch, rows)


def test_preprocessor_fit_transform():
    X = np.random.RandomState(0).randn(200, 35).astype(np.float32)
    pp = Preprocessor(scaler="standard", pca_variance=0.95)
    Xp = pp.fit_transform(X)
    assert Xp.shape[0] == 200
    assert Xp.shape[1] <= 35


def test_sliding_window_eviction():
    w = SlidingWindow(span_seconds=1.0)
    for t in np.linspace(0, 0.9, 10):
        w.add(t, "x")
    assert len(w) == 10
    w.add(5.0, "y")
    assert len(w) == 1


def test_port_scan_rule():
    re_eng = RuleEngine({"port_scan": {"distinct_ports_threshold": 5,
                                       "window_seconds": 10}})
    hits = []
    for port in range(1, 8):
        hits = re_eng.evaluate({
            "ts": 0.1 * port, "src_ip": "1.2.3.4", "dst_ip": "10.0.0.5",
            "src_port": 30000, "dst_port": port, "protocol": "TCP",
            "length": 60, "flags": "SYN", "payload_len": 0,
            "payload_signature": "", "label": "port_scan",
        })
    assert any(h.name == "port_scan" for h in hits)


def test_sql_injection_rule():
    re_eng = RuleEngine()
    hits = re_eng.evaluate({
        "ts": 1.0, "src_ip": "8.8.8.8", "dst_ip": "10.0.0.5",
        "src_port": 30000, "dst_port": 80, "protocol": "TCP",
        "length": 200, "flags": "PSH+ACK", "payload_len": 80,
        "payload_signature": "' OR 1=1 --",
        "label": "sql_injection",
    })
    assert any(h.name == "sql_injection" for h in hits)


def test_isolation_forest_predict_proba():
    X = np.random.RandomState(0).randn(200, 10).astype(np.float32)
    m = IsolationForestDetector(n_estimators=20).fit(X)
    p = m.predict_proba(X)
    assert p.shape == (200,)
    assert (p >= 0).all() and (p <= 1).all()


def test_random_forest_classifier_classes():
    X = np.random.RandomState(0).randn(400, 8).astype(np.float32)
    y = np.where(X[:, 0] > 0, "benign", "ddos_syn_flood")
    m = RandomForestClassifier_(n_estimators=20).fit(X, y)
    assert set(m.classes_) == {"benign", "ddos_syn_flood"}
    proba = m.predict_proba(X)
    assert proba.shape == (400, 2)


def test_lstm_fallback_predicts():
    X = np.random.RandomState(0).randn(120, 6).astype(np.float32)
    y = np.random.RandomState(0).randint(0, 2, 120)
    m = LSTMSequenceModel(sequence_length=4, epochs=1,
                          use_tf=False).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (120,)


def test_ensemble_predict_returns_correct_size():
    # Train a tiny ensemble end-to-end and verify shape.
    pkts = generate_labeled_dataset(800, seed=5)
    ex   = FeatureExtractor()
    X    = ex.transform_batch(pkts)
    y    = np.array([p["label"] for p in pkts])
    yb   = (y != "benign").astype(int)
    pre  = Preprocessor().fit(X)
    Xp   = pre.transform(X)
    iforest = IsolationForestDetector(n_estimators=20).fit(Xp)
    rforest = RandomForestClassifier_(n_estimators=20).fit(Xp, y)
    lstm    = LSTMSequenceModel(sequence_length=4, epochs=1,
                                use_tf=False).fit(Xp, yb)
    ens = EnsembleDetector(pre, iforest, rforest, lstm, threshold=0.5)
    preds = ens.predict(X[:32])
    assert len(preds) == 32
    assert all(0 <= p.score <= 1 for p in preds)
