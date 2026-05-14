"""
Real-time stream processor.

Glues together the packet source, feature extractor, ensemble and rule
engine. Publishes DetectionEvents onto a thread-safe in-memory queue
that the dashboard reads.
"""
from __future__ import annotations

import os
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np

from data_collection.traffic_simulator import TrafficSimulator, SEVERITY, AttackType
from feature_engineering.extractor    import FeatureExtractor
from models.ensemble_detector         import EnsembleDetector, EnsemblePrediction
from real_time.rule_engine            import RuleEngine, RuleHit


@dataclass
class DetectionEvent:
    id: str
    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    src_country: str
    label_truth: str
    label_pred: str
    severity: str
    score: float
    confidence: float
    detector: str           # "ml" | "rule" | "ml+rule"
    components: Dict[str, float] = field(default_factory=dict)
    rule_hits: List[str]    = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class StreamProcessor:
    """Background worker that pulls packets from a simulator or live source."""

    def __init__(self,
                 ensemble: Optional[EnsembleDetector],
                 rule_engine: Optional[RuleEngine] = None,
                 source: Optional[Iterable[Dict]] = None,
                 extractor: Optional[FeatureExtractor] = None,
                 max_events: int = 2000):
        self.ensemble    = ensemble
        self.rule_engine = rule_engine or RuleEngine()
        self.extractor   = extractor or FeatureExtractor(window_seconds=30)
        self.source      = source or TrafficSimulator().stream()

        self._events: Deque[DetectionEvent] = deque(maxlen=max_events)
        self._counters = {
            "packets":         0,
            "alerts":          0,
            "alerts_critical": 0,
            "alerts_high":     0,
            "alerts_medium":   0,
            "alerts_low":      0,
            "ml_only":         0,
            "rule_only":       0,
            "ml_plus_rule":    0,
            "false_positives": 0,
            "true_positives":  0,
            "true_negatives":  0,
            "false_negatives": 0,
        }
        self._latency_ms: Deque[float] = deque(maxlen=1000)
        self._pps_buf:    Deque[float] = deque(maxlen=300)   # 5 minutes at 1Hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="nids-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def snapshot_events(self, limit: int = 200) -> List[Dict]:
        with self._lock:
            return [e.to_dict() for e in list(self._events)[-limit:]]

    def snapshot_metrics(self) -> Dict:
        with self._lock:
            return {
                **self._counters,
                "avg_latency_ms":
                    float(np.mean(self._latency_ms)) if self._latency_ms else 0.0,
                "p95_latency_ms":
                    float(np.percentile(self._latency_ms, 95)) if self._latency_ms else 0.0,
                "pps_recent":
                    float(np.mean(self._pps_buf)) if self._pps_buf else 0.0,
            }

    def _run(self) -> None:
        last_pps_tick = time.time()
        pps_count = 0
        for pkt in self.source:
            if self._stop.is_set():
                break

            t0 = time.perf_counter()
            self._handle_packet(pkt)
            self._latency_ms.append((time.perf_counter() - t0) * 1000)

            pps_count += 1
            now = time.time()
            if now - last_pps_tick >= 1.0:
                with self._lock:
                    self._pps_buf.append(pps_count)
                pps_count = 0
                last_pps_tick = now

    def _handle_packet(self, pkt: Dict) -> None:
        x = self.extractor.transform_packet(pkt)

        ml: Optional[EnsemblePrediction] = None
        if self.ensemble is not None:
            try:
                ml = self.ensemble.predict_one(x)
            except Exception:
                ml = None

        hits = self.rule_engine.evaluate(pkt)

        ml_attack   = bool(ml and ml.is_attack)
        rule_attack = len(hits) > 0
        truth       = pkt.get("label", AttackType.BENIGN.value)
        truth_attack = truth != AttackType.BENIGN.value

        with self._lock:
            self._counters["packets"] += 1

            if not (ml_attack or rule_attack):
                if truth_attack:
                    self._counters["false_negatives"] += 1
                else:
                    self._counters["true_negatives"]  += 1
                return

            if ml_attack and rule_attack:
                detector = "ml+rule"
                self._counters["ml_plus_rule"] += 1
            elif ml_attack:
                detector = "ml"
                self._counters["ml_only"] += 1
            else:
                detector = "rule"
                self._counters["rule_only"] += 1

            severity = self._merge_severity(ml, hits)
            label    = (hits[0].name if hits
                        else (ml.attack_type if ml else "unknown"))
            score    = ml.score if ml else 0.85
            conf     = ml.confidence if ml else 0.9

            ev = DetectionEvent(
                id=str(uuid.uuid4())[:8],
                ts=pkt["ts"],
                src_ip=pkt.get("src_ip", ""),
                dst_ip=pkt.get("dst_ip", ""),
                src_port=int(pkt.get("src_port", 0)),
                dst_port=int(pkt.get("dst_port", 0)),
                protocol=pkt.get("protocol", ""),
                length=int(pkt.get("length", 0)),
                src_country=pkt.get("src_country", ""),
                label_truth=truth,
                label_pred=label,
                severity=severity,
                score=float(score),
                confidence=float(conf),
                detector=detector,
                components=(ml.components if ml else {}),
                rule_hits=[h.name for h in hits],
            )
            self._events.append(ev)
            self._counters["alerts"] += 1
            self._counters[f"alerts_{severity}"] = self._counters.get(
                f"alerts_{severity}", 0) + 1
            if truth_attack:
                self._counters["true_positives"] += 1
            else:
                self._counters["false_positives"] += 1

    @staticmethod
    def _merge_severity(ml: Optional[EnsemblePrediction],
                        hits: List[RuleHit]) -> str:
        order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        best = "low"
        for h in hits:
            if order.get(h.severity, 0) > order.get(best, 0):
                best = h.severity
        if ml and ml.is_attack:
            ml_sev = SEVERITY.get(AttackType(ml.attack_type), "medium") \
                     if ml.attack_type in [a.value for a in AttackType] else "medium"
            if order.get(ml_sev, 0) > order.get(best, 0):
                best = ml_sev
        return best
