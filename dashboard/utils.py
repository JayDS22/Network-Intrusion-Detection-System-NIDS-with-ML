"""Dashboard helpers: caching, theming, formatting."""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, Optional

import streamlit as st
import yaml

# Resolve project root relative to this file so the dashboard
# can be launched from anywhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SEVERITY_COLORS = {
    "critical": "#ff3b3b",
    "high":     "#ff9f3b",
    "medium":   "#ffd23b",
    "low":      "#3bd1ff",
    "info":     "#6c757d",
}

ATTACK_EMOJI = {
    "ddos_syn_flood":    "🌊",
    "ddos_udp_flood":    "🌊",
    "port_scan":         "🔭",
    "brute_force_ssh":   "🔨",
    "brute_force":       "🔨",
    "sql_injection":     "💉",
    "xss":               "🪤",
    "xss_attempt":       "🪤",
    "dns_tunneling":     "🕳️",
    "data_exfiltration": "📤",
    "lateral_movement":  "↔️",
    "malware_c2":        "👾",
    "reconnaissance":    "🛰️",
    "mitm_arp_spoof":    "🎭",
    "benign":            "✅",
    "unknown":           "❓",
}


@st.cache_resource
def load_config(path: Optional[str] = None) -> Dict:
    path = path or os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def fmt_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_full_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def severity_pill(sev: str) -> str:
    color = SEVERITY_COLORS.get(sev, "#666")
    return (f"<span style='background:{color};color:#0d0f14;"
            "padding:2px 10px;border-radius:10px;font-weight:600;"
            f"font-size:0.78rem;'>{sev.upper()}</span>")
