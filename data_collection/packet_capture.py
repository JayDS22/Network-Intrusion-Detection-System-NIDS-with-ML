"""
Scapy-based live packet capture.

Requires root or cap_net_raw. Exposes the same stream() interface as the
simulator, so PacketCapture(...).stream() can be passed straight to
StreamProcessor.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from scapy.all import sniff, Packet, TCP, UDP, IP, ICMP   # type: ignore
    _HAVE_SCAPY = True
except Exception:                                              # pragma: no cover
    _HAVE_SCAPY = False


def _packet_to_dict(pkt: "Packet") -> Optional[Dict]:
    if not _HAVE_SCAPY:
        return None
    if IP not in pkt:
        return None
    ip = pkt[IP]
    proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else \
            "ICMP" if ICMP in pkt else "OTHER"
    sport = int(pkt[TCP].sport) if TCP in pkt else (
            int(pkt[UDP].sport) if UDP in pkt else 0)
    dport = int(pkt[TCP].dport) if TCP in pkt else (
            int(pkt[UDP].dport) if UDP in pkt else 0)

    flags = ""
    if TCP in pkt:
        names = []
        f = pkt[TCP].flags
        for letter, name in [("S", "SYN"), ("A", "ACK"), ("F", "FIN"),
                             ("R", "RST"), ("P", "PSH"), ("U", "URG")]:
            if letter in str(f):
                names.append(name)
        flags = "+".join(names)

    raw = bytes(pkt.payload) if pkt.payload else b""
    payload_signature = ""
    if raw:
        try:
            payload_signature = raw.decode("utf-8", errors="ignore")[:160]
        except Exception:
            payload_signature = ""

    return {
        "ts":       float(pkt.time),
        "src_ip":   ip.src,
        "dst_ip":   ip.dst,
        "src_port": sport,
        "dst_port": dport,
        "protocol": proto,
        "length":   int(len(pkt)),
        "flags":    flags,
        "payload_len": len(raw),
        "payload_signature": payload_signature,
        "label":    "unknown",
        "src_country": "?",
    }


class PacketCapture:
    """Wraps ``scapy.sniff`` with the same ``stream()`` interface as the
    synthetic simulator."""

    def __init__(self, interface: str = "en0",
                 bpf_filter: str = "tcp or udp",
                 snaplen: int = 65535,
                 promiscuous: bool = True):
        self.interface   = interface
        self.bpf_filter  = bpf_filter
        self.snaplen     = snaplen
        self.promiscuous = promiscuous

    def stream(self) -> Iterator[Dict]:
        if not _HAVE_SCAPY:
            raise RuntimeError(
                "scapy is not available; live capture is disabled. "
                "Install scapy or use the simulator.")
        if os.geteuid() != 0:                                  # pragma: no cover
            print("[warn] Live packet capture usually requires root.")

        def _gen():
            queue = []
            def _cb(p):
                d = _packet_to_dict(p)
                if d:
                    queue.append(d)
            t0 = time.time()
            sniff(iface=self.interface, filter=self.bpf_filter,
                  prn=_cb, store=False, timeout=None)
            while queue:
                yield queue.pop(0)
        return _gen()


def main():                                                    # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--filter", default="tcp or udp")
    args = ap.parse_args()
    cap = PacketCapture(args.iface, args.filter)
    for pkt in cap.stream():
        print(pkt)


if __name__ == "__main__":
    main()
