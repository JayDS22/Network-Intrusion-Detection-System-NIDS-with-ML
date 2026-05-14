"""
Signature and heuristic rules.

Runs alongside the ML ensemble. Catches deterministic patterns (SYN
floods, port sweeps, SQLi signatures, etc.) that the model treats
probabilistically.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


_SQL_RE    = re.compile(r"(union\s+select|or\s+1=1|drop\s+table|--|/\*|'\s*or)", re.I)
_XSS_RE    = re.compile(r"(<script|onerror=|javascript:|<img\s+src=)", re.I)


@dataclass
class RuleHit:
    name: str
    severity: str
    description: str
    src_ip: str
    dst_ip: str
    details: Dict


class RuleEngine:
    """Stateful rule evaluator over a packet stream."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.cfg = {
            "port_scan_threshold": cfg.get("port_scan", {}).get("distinct_ports_threshold", 15),
            "port_scan_window":    cfg.get("port_scan", {}).get("window_seconds", 10),
            "ddos_syn_rate":       cfg.get("ddos", {}).get("syn_rate_per_second", 500),
            "ddos_window":         cfg.get("ddos", {}).get("window_seconds", 5),
            "bf_threshold":        cfg.get("brute_force", {}).get("failed_attempts_threshold", 10),
            "bf_window":           cfg.get("brute_force", {}).get("window_seconds", 60),
            "bf_ports":            set(cfg.get("brute_force", {}).get("target_ports", [22, 21, 3389])),
            "dns_len_threshold":   cfg.get("dns_tunneling", {}).get("avg_query_length_threshold", 60),
        }
        # Per-source rolling buffers: src_ip -> deque[(ts, port)]
        self._scan_buf: Dict[str, Deque[Tuple[float, int]]] = defaultdict(deque)
        # Per-target SYN counter: dst_ip -> deque[ts]
        self._syn_buf:  Dict[str, Deque[float]] = defaultdict(deque)
        # Brute-force tracker:  (src,dst,port) -> deque[ts]
        self._bf_buf:   Dict[Tuple[str, str, int], Deque[float]] = defaultdict(deque)
    # Public: evaluate a single packet

    def evaluate(self, pkt: Dict) -> List[RuleHit]:
        hits: List[RuleHit] = []

        ts = pkt["ts"]
        src, dst = pkt.get("src_ip", ""), pkt.get("dst_ip", "")
        port = int(pkt.get("dst_port", 0))
        flags = pkt.get("flags", "") or ""
        proto = pkt.get("protocol", "")
        sig = pkt.get("payload_signature", "") or ""
        buf = self._scan_buf[src]
        buf.append((ts, port))
        cutoff = ts - self.cfg["port_scan_window"]
        while buf and buf[0][0] < cutoff:
            buf.popleft()
        distinct_ports = {p for _, p in buf}
        if len(distinct_ports) >= self.cfg["port_scan_threshold"]:
            hits.append(RuleHit(
                name="port_scan",
                severity="high",
                description=f"{src} probed {len(distinct_ports)} ports on {dst} "
                            f"in {self.cfg['port_scan_window']}s",
                src_ip=src, dst_ip=dst,
                details={"distinct_ports": len(distinct_ports)},
            ))
        if "SYN" in flags and "ACK" not in flags:
            sbuf = self._syn_buf[dst]
            sbuf.append(ts)
            cutoff = ts - self.cfg["ddos_window"]
            while sbuf and sbuf[0] < cutoff:
                sbuf.popleft()
            rate = len(sbuf) / max(0.5, self.cfg["ddos_window"])
            if rate >= self.cfg["ddos_syn_rate"]:
                hits.append(RuleHit(
                    name="ddos_syn_flood",
                    severity="critical",
                    description=f"{dst} receiving {rate:.0f} SYN/s "
                                f"(threshold {self.cfg['ddos_syn_rate']}/s)",
                    src_ip=src, dst_ip=dst,
                    details={"syn_rate": rate},
                ))
        if port in self.cfg["bf_ports"]:
            key = (src, dst, port)
            bbuf = self._bf_buf[key]
            bbuf.append(ts)
            cutoff = ts - self.cfg["bf_window"]
            while bbuf and bbuf[0] < cutoff:
                bbuf.popleft()
            if len(bbuf) >= self.cfg["bf_threshold"]:
                hits.append(RuleHit(
                    name="brute_force",
                    severity="high",
                    description=f"{src} attempted {len(bbuf)} auths on "
                                f"{dst}:{port} in {self.cfg['bf_window']}s",
                    src_ip=src, dst_ip=dst,
                    details={"attempts": len(bbuf), "port": port},
                ))
        if _SQL_RE.search(sig):
            hits.append(RuleHit(
                name="sql_injection",
                severity="high",
                description=f"SQL injection signature detected from {src}",
                src_ip=src, dst_ip=dst,
                details={"signature": sig[:80]},
            ))
        if _XSS_RE.search(sig):
            hits.append(RuleHit(
                name="xss_attempt",
                severity="medium",
                description=f"XSS payload from {src}",
                src_ip=src, dst_ip=dst,
                details={"signature": sig[:80]},
            ))
        if proto == "UDP" and port == 53 and pkt.get("payload_len", 0) >= self.cfg["dns_len_threshold"]:
            hits.append(RuleHit(
                name="dns_tunneling",
                severity="high",
                description=f"DNS query from {src} with payload "
                            f"{pkt.get('payload_len')} bytes",
                src_ip=src, dst_ip=dst,
                details={"payload_len": pkt.get("payload_len")},
            ))
        if (pkt.get("src_ip", "").startswith("10.")
                and pkt.get("payload_len", 0) > 1100):
            hits.append(RuleHit(
                name="data_exfiltration",
                severity="critical",
                description=f"Large outbound payload {src} -> {dst}",
                src_ip=src, dst_ip=dst,
                details={"payload_len": pkt.get("payload_len")},
            ))

        return hits
