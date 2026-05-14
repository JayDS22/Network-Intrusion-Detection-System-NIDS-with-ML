"""Top-row KPI cards."""
from __future__ import annotations

from typing import Dict

import streamlit as st


def kpi_card(label: str, value: str, delta: str = "", color: str = "#7af0ff") -> str:
    return f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value' style='color:{color};'>{value}</div>
      <div class='kpi-delta'>{delta}</div>
    </div>
    """


def render(metrics: Dict) -> None:
    cols = st.columns(6)
    items = [
        ("PACKETS PROCESSED", f"{metrics.get('packets', 0):,}",
         f"~{metrics.get('pps_recent', 0):.0f} pps", "#7af0ff"),
        ("ALERTS",            f"{metrics.get('alerts', 0):,}",
         "since start", "#ff9f3b"),
        ("CRITICAL",          f"{metrics.get('alerts_critical', 0):,}",
         f"H: {metrics.get('alerts_high', 0)}", "#ff3b3b"),
        ("DETECTION RATE",    f"{_det_rate(metrics):.1f}%",
         "TP / (TP+FN)", "#7cffa9"),
        ("FALSE POSITIVE",    f"{_fp_rate(metrics):.2f}%",
         "FP / (FP+TN)", "#ffd23b"),
        ("AVG LATENCY",       f"{metrics.get('avg_latency_ms', 0):.1f} ms",
         f"p95: {metrics.get('p95_latency_ms', 0):.1f}", "#b894ff"),
    ]
    for col, (label, value, delta, color) in zip(cols, items):
        with col:
            st.markdown(kpi_card(label, value, delta, color),
                        unsafe_allow_html=True)


def _det_rate(m: Dict) -> float:
    tp, fn = m.get("true_positives", 0), m.get("false_negatives", 0)
    return 100.0 * tp / max(1, tp + fn)


def _fp_rate(m: Dict) -> float:
    fp, tn = m.get("false_positives", 0), m.get("true_negatives", 0)
    return 100.0 * fp / max(1, fp + tn)
