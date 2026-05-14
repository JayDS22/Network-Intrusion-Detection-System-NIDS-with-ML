"""
Packet to feature vector conversion for the ML detectors.

40 numeric features per packet: packet primitives, payload signatures,
per-source rolling counts, and per-flow timing stats. Aggregates are kept
in incremental counters so each packet costs O(1) on average instead of
rescanning the window.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List

import numpy as np


FEATURE_NAMES: List[str] = [
    # primitives (12)
    "pkt_len", "src_port", "dst_port", "payload_len",
    "is_tcp", "is_udp", "is_icmp",
    "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh",
    # signature heuristics (8)
    "has_sql_keyword", "has_script_tag", "has_long_subdomain",
    "has_encrypted_blob", "has_auth_fail", "has_c2_beacon",
    "has_psexec_rdp", "has_icmp_sweep",
    # payload statistics (3)
    "payload_entropy", "payload_ascii_ratio", "is_outbound",
    # rolling per-source aggregates over the current window (10)
    "src_pkt_count", "src_byte_count", "src_unique_dst_ips",
    "src_unique_dst_ports", "src_syn_count", "src_rst_count",
    "src_avg_pkt_len", "src_pps", "src_protocol_diversity",
    "src_country_risk",
    # rolling per-flow aggregates (7)
    "flow_pkt_count", "flow_byte_count", "flow_avg_iat",
    "flow_duration", "flow_pkts_per_sec", "flow_bytes_per_sec",
    "flow_payload_ratio",
]
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

_SQL_PAT     = re.compile(r"(union\s+select|or\s+1=1|drop\s+table|--|/\*|'\s*or)", re.I)
_SCRIPT_PAT  = re.compile(r"(<script|onerror=|javascript:|<img\s+src=)", re.I)
_HIGH_RISK_COUNTRIES = {"CN", "RU", "IR", "KP", "UA"}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


class _SrcAggregate:
    """Per source-IP rolling counters."""
    __slots__ = ("pkt_count", "byte_count", "first_ts", "last_ts",
                 "syn_count", "rst_count",
                 "dst_ips", "dst_ports", "protocols")

    def __init__(self) -> None:
        self.pkt_count = 0
        self.byte_count = 0
        self.first_ts = self.last_ts = 0.0
        self.syn_count = self.rst_count = 0
        self.dst_ips:   Counter = Counter()
        self.dst_ports: Counter = Counter()
        self.protocols: Counter = Counter()

    def add(self, pkt: Dict) -> None:
        if self.pkt_count == 0:
            self.first_ts = pkt["ts"]
        self.last_ts    = pkt["ts"]
        self.pkt_count += 1
        self.byte_count += pkt.get("length", 0)
        flags = pkt.get("flags") or ""
        if "SYN" in flags: self.syn_count += 1
        if "RST" in flags: self.rst_count += 1
        self.dst_ips[pkt.get("dst_ip")]   += 1
        self.dst_ports[pkt.get("dst_port")] += 1
        self.protocols[pkt.get("protocol")] += 1

    def remove(self, pkt: Dict) -> None:
        self.pkt_count  -= 1
        self.byte_count -= pkt.get("length", 0)
        flags = pkt.get("flags") or ""
        if "SYN" in flags: self.syn_count -= 1
        if "RST" in flags: self.rst_count -= 1
        for ctr, key in ((self.dst_ips, pkt.get("dst_ip")),
                         (self.dst_ports, pkt.get("dst_port")),
                         (self.protocols, pkt.get("protocol"))):
            ctr[key] -= 1
            if ctr[key] <= 0:
                del ctr[key]


class _FlowAggregate:
    """Per (src, dst, dport) flow counters."""
    __slots__ = ("pkt_count", "byte_count", "payload_count",
                 "first_ts", "last_ts", "prev_ts", "iat_sum")

    def __init__(self) -> None:
        self.pkt_count = 0
        self.byte_count = 0
        self.payload_count = 0
        self.first_ts = self.last_ts = self.prev_ts = 0.0
        self.iat_sum = 0.0

    def add(self, pkt: Dict) -> None:
        ts = pkt["ts"]
        if self.pkt_count == 0:
            self.first_ts = ts
        else:
            self.iat_sum += max(0.0, ts - self.prev_ts)
        self.prev_ts = self.last_ts = ts
        self.pkt_count += 1
        self.byte_count    += pkt.get("length", 0)
        self.payload_count += pkt.get("payload_len", 0)

    def remove(self, pkt: Dict) -> None:
        self.pkt_count -= 1
        self.byte_count    -= pkt.get("length", 0)
        self.payload_count -= pkt.get("payload_len", 0)


class FeatureExtractor:
    """Streaming feature extractor with rolling window aggregates."""

    def __init__(self, window_seconds: int = 30, max_window_pkts: int = 50_000):
        self.window_seconds = window_seconds
        self.max_window_pkts = max_window_pkts
        self._window: Deque[Dict] = deque(maxlen=max_window_pkts)
        self._src:  Dict[str, _SrcAggregate]  = defaultdict(_SrcAggregate)
        self._flow: Dict[tuple, _FlowAggregate] = defaultdict(_FlowAggregate)

    def transform_packet(self, pkt: Dict) -> np.ndarray:
        self._add(pkt)
        self._evict(pkt["ts"])
        return self._row(pkt)

    def transform_batch(self, packets: List[Dict]) -> np.ndarray:
        out = np.zeros((len(packets), len(FEATURE_NAMES)), dtype=np.float32)
        for i, pkt in enumerate(packets):
            self._add(pkt)
            self._evict(pkt["ts"])
            out[i] = self._row(pkt)
        return out

    def _add(self, pkt: Dict) -> None:
        self._window.append(pkt)
        self._src[pkt.get("src_ip", "")].add(pkt)
        flow_key = (pkt.get("src_ip"), pkt.get("dst_ip"), pkt.get("dst_port"))
        self._flow[flow_key].add(pkt)
        pkt["_flow_key"] = flow_key

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._window and self._window[0]["ts"] < cutoff:
            old = self._window.popleft()
            src = old.get("src_ip", "")
            agg = self._src.get(src)
            if agg is not None:
                agg.remove(old)
                if agg.pkt_count <= 0:
                    self._src.pop(src, None)
            fk = old.get("_flow_key")
            if fk is not None:
                fagg = self._flow.get(fk)
                if fagg is not None:
                    fagg.remove(old)
                    if fagg.pkt_count <= 0:
                        self._flow.pop(fk, None)

    def _row(self, pkt: Dict) -> np.ndarray:
        f = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

        f[_IDX["pkt_len"]]     = pkt.get("length", 0)
        f[_IDX["src_port"]]    = pkt.get("src_port", 0)
        f[_IDX["dst_port"]]    = pkt.get("dst_port", 0)
        f[_IDX["payload_len"]] = pkt.get("payload_len", 0)
        proto = pkt.get("protocol", "")
        f[_IDX["is_tcp"]]  = float(proto == "TCP")
        f[_IDX["is_udp"]]  = float(proto == "UDP")
        f[_IDX["is_icmp"]] = float(proto == "ICMP")
        flags = pkt.get("flags") or ""
        f[_IDX["flag_syn"]] = float("SYN" in flags)
        f[_IDX["flag_ack"]] = float("ACK" in flags)
        f[_IDX["flag_fin"]] = float("FIN" in flags)
        f[_IDX["flag_rst"]] = float("RST" in flags)
        f[_IDX["flag_psh"]] = float("PSH" in flags)

        sig = pkt.get("payload_signature") or ""
        f[_IDX["has_sql_keyword"]]    = float(bool(_SQL_PAT.search(sig)))
        f[_IDX["has_script_tag"]]     = float(bool(_SCRIPT_PAT.search(sig)))
        f[_IDX["has_long_subdomain"]] = float("long-subdomain" in sig)
        f[_IDX["has_encrypted_blob"]] = float("encrypted_blob" in sig)
        f[_IDX["has_auth_fail"]]      = float("auth_fail" in sig)
        f[_IDX["has_c2_beacon"]]      = float("c2_beacon" in sig)
        f[_IDX["has_psexec_rdp"]]     = float("psexec" in sig.lower())
        f[_IDX["has_icmp_sweep"]]     = float("icmp_sweep" in sig)
        f[_IDX["payload_entropy"]]    = _entropy(sig)
        if sig:
            ascii_chars = sum(1 for c in sig if 32 <= ord(c) < 127)
            f[_IDX["payload_ascii_ratio"]] = ascii_chars / len(sig)
        f[_IDX["is_outbound"]] = float(pkt.get("src_ip", "").startswith("10."))

        sagg = self._src.get(pkt.get("src_ip", ""))
        if sagg and sagg.pkt_count:
            dur = max(1e-3, sagg.last_ts - sagg.first_ts)
            f[_IDX["src_pkt_count"]]         = sagg.pkt_count
            f[_IDX["src_byte_count"]]        = sagg.byte_count
            f[_IDX["src_unique_dst_ips"]]    = len(sagg.dst_ips)
            f[_IDX["src_unique_dst_ports"]]  = len(sagg.dst_ports)
            f[_IDX["src_syn_count"]]         = sagg.syn_count
            f[_IDX["src_rst_count"]]         = sagg.rst_count
            f[_IDX["src_avg_pkt_len"]]       = sagg.byte_count / sagg.pkt_count
            f[_IDX["src_pps"]]               = sagg.pkt_count / dur
            f[_IDX["src_protocol_diversity"]] = len(sagg.protocols)
            f[_IDX["src_country_risk"]]      = float(
                pkt.get("src_country", "") in _HIGH_RISK_COUNTRIES)

        fk = (pkt.get("src_ip"), pkt.get("dst_ip"), pkt.get("dst_port"))
        fagg = self._flow.get(fk)
        if fagg and fagg.pkt_count:
            dur = max(1e-3, fagg.last_ts - fagg.first_ts)
            avg_iat = fagg.iat_sum / max(1, fagg.pkt_count - 1)
            f[_IDX["flow_pkt_count"]]      = fagg.pkt_count
            f[_IDX["flow_byte_count"]]     = fagg.byte_count
            f[_IDX["flow_avg_iat"]]        = avg_iat
            f[_IDX["flow_duration"]]       = dur
            f[_IDX["flow_pkts_per_sec"]]   = fagg.pkt_count / dur
            f[_IDX["flow_bytes_per_sec"]]  = fagg.byte_count / dur
            f[_IDX["flow_payload_ratio"]]  = (
                fagg.payload_count / max(1, fagg.byte_count))

        return f
