"""Offline PCAP file parser (uses scapy if installed)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from scapy.all import rdpcap                           # type: ignore
    _HAVE_SCAPY = True
except Exception:                                          # pragma: no cover
    _HAVE_SCAPY = False

from data_collection.packet_capture import _packet_to_dict


def parse_pcap(path: str) -> Iterator[Dict]:
    if not _HAVE_SCAPY:
        raise RuntimeError("scapy not available; cannot parse PCAP.")
    for pkt in rdpcap(path):
        d = _packet_to_dict(pkt)
        if d:
            yield d


def main():                                                # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    args = ap.parse_args()
    for i, p in enumerate(parse_pcap(args.pcap)):
        print(p)
        if i > 50:
            break


if __name__ == "__main__":
    main()
