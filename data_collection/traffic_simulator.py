"""
Synthetic network traffic generator.

Produces packet dicts in the same shape as the live capture path, with
configurable attack injection. Used for training the ensemble and for
driving the demo dashboard when no live interface is available.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional

import numpy as np


class AttackType(str, Enum):
    BENIGN              = "benign"
    DDOS_SYN_FLOOD      = "ddos_syn_flood"
    DDOS_UDP_FLOOD      = "ddos_udp_flood"
    PORT_SCAN           = "port_scan"
    BRUTE_FORCE_SSH     = "brute_force_ssh"
    SQL_INJECTION       = "sql_injection"
    XSS                 = "xss"
    DNS_TUNNELING       = "dns_tunneling"
    DATA_EXFILTRATION   = "data_exfiltration"
    LATERAL_MOVEMENT    = "lateral_movement"
    MITM                = "mitm_arp_spoof"
    MALWARE_C2          = "malware_c2"
    RECONNAISSANCE      = "reconnaissance"


SEVERITY = {
    AttackType.BENIGN:            "info",
    AttackType.DDOS_SYN_FLOOD:    "critical",
    AttackType.DDOS_UDP_FLOOD:    "critical",
    AttackType.PORT_SCAN:         "high",
    AttackType.BRUTE_FORCE_SSH:   "high",
    AttackType.SQL_INJECTION:     "high",
    AttackType.XSS:               "medium",
    AttackType.DNS_TUNNELING:     "high",
    AttackType.DATA_EXFILTRATION: "critical",
    AttackType.LATERAL_MOVEMENT:  "high",
    AttackType.MITM:              "critical",
    AttackType.MALWARE_C2:        "critical",
    AttackType.RECONNAISSANCE:    "medium",
}


COMMON_PORTS  = [80, 443, 22, 21, 25, 53, 110, 143, 3306, 3389, 8080, 8443]
COMMON_PROTOS = ["TCP", "UDP", "ICMP"]


@dataclass
class GeoIP:
    """Offline geo lookup. Maps an IP to a country via its first two octets."""
    countries: List[str] = field(default_factory=lambda: [
        "US", "CN", "RU", "BR", "DE", "IN", "GB", "FR", "JP", "NL",
        "CA", "AU", "KR", "SG", "IR", "UA", "MX", "ES", "IT", "ZA",
    ])

    def lookup(self, ip: str) -> str:
        a, b, *_ = ip.split(".")
        idx = (int(a) * 7 + int(b) * 13) % len(self.countries)
        return self.countries[idx]


GEO = GeoIP()


def _rand_public_ip() -> str:
    while True:
        a = random.randint(1, 223)
        if a in (10, 127, 169, 172, 192):
            continue
        return f"{a}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _rand_private_ip() -> str:
    return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


class TrafficSimulator:
    """Generates synthetic packet dicts shaped like real capture events.

    Each packet has: ts, src_ip, dst_ip, src_port, dst_port, protocol,
    length, flags, payload_len, payload_signature, label, src_country.
    """

    def __init__(
        self,
        packets_per_second: int = 250,
        attack_probability: float = 0.08,
        seed: Optional[int] = None,
    ) -> None:
        self.pps = packets_per_second
        self.attack_p = attack_probability
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._next_attack_at = 0.0
        self._home_subnet = "10.0.0."  # protected hosts live here


    def stream(self, duration_seconds: Optional[float] = None) -> Iterator[Dict]:
        """Yield packets at roughly the configured rate."""
        start = time.time()
        while True:
            now = time.time()
            if duration_seconds is not None and now - start >= duration_seconds:
                return

            burst_size = max(1, int(self.pps / 10))   # 100ms ticks
            for pkt in self._burst(burst_size, ts=now):
                yield pkt
            time.sleep(0.1)

    def sample_batch(self, n: int) -> List[Dict]:
        """Non-blocking batch generation for training data."""
        out: List[Dict] = []
        ts0 = time.time()
        for i in range(n):
            ts = ts0 + i * (1.0 / max(1, self.pps))
            out.extend(self._burst(1, ts=ts))
        return out

    def _burst(self, n: int, ts: float) -> List[Dict]:
        if random.random() < self.attack_p:
            attack = random.choice([
                AttackType.DDOS_SYN_FLOOD,
                AttackType.PORT_SCAN,
                AttackType.BRUTE_FORCE_SSH,
                AttackType.SQL_INJECTION,
                AttackType.XSS,
                AttackType.DNS_TUNNELING,
                AttackType.DATA_EXFILTRATION,
                AttackType.LATERAL_MOVEMENT,
                AttackType.MALWARE_C2,
                AttackType.RECONNAISSANCE,
            ])
            return self._attack_burst(attack, n, ts)
        return [self._benign_packet(ts + i * 1e-4) for i in range(n)]

    def _benign_packet(self, ts: float) -> Dict:
        proto = random.choices(COMMON_PROTOS, weights=[0.75, 0.20, 0.05])[0]
        src   = _rand_public_ip()
        dst   = self._home_subnet + str(random.randint(2, 50))
        dport = random.choice(COMMON_PORTS)
        sport = random.randint(1024, 65535)
        length = int(np.clip(np.random.lognormal(6.5, 0.6), 40, 1500))
        flags  = random.choice(["ACK", "PSH+ACK", "ACK", "ACK", "FIN+ACK"])
        return {
            "ts": ts,
            "src_ip": src, "dst_ip": dst,
            "src_port": sport, "dst_port": dport,
            "protocol": proto,
            "length": length,
            "flags": flags if proto == "TCP" else "",
            "payload_len": max(0, length - 40),
            "payload_signature": "",
            "label": AttackType.BENIGN.value,
            "src_country": GEO.lookup(src),
        }

    def _attack_burst(self, attack: AttackType, n: int, ts: float) -> List[Dict]:
        if attack == AttackType.DDOS_SYN_FLOOD:    return self._ddos_syn(n, ts)
        if attack == AttackType.PORT_SCAN:         return self._port_scan(n, ts)
        if attack == AttackType.BRUTE_FORCE_SSH:   return self._brute_force(n, ts)
        if attack == AttackType.SQL_INJECTION:     return self._sql_injection(n, ts)
        if attack == AttackType.XSS:               return self._xss(n, ts)
        if attack == AttackType.DNS_TUNNELING:     return self._dns_tunnel(n, ts)
        if attack == AttackType.DATA_EXFILTRATION: return self._exfil(n, ts)
        if attack == AttackType.LATERAL_MOVEMENT:  return self._lateral(n, ts)
        if attack == AttackType.MALWARE_C2:        return self._c2(n, ts)
        if attack == AttackType.RECONNAISSANCE:    return self._recon(n, ts)
        return [self._benign_packet(ts)]

    def _ddos_syn(self, n: int, ts: float) -> List[Dict]:
        target = self._home_subnet + str(random.randint(2, 10))
        out = []
        for i in range(max(n, 30)):                       # high volume
            src = _rand_public_ip()
            out.append({
                "ts": ts + i * 1e-5,
                "src_ip": src, "dst_ip": target,
                "src_port": random.randint(1024, 65535),
                "dst_port": random.choice([80, 443]),
                "protocol": "TCP",
                "length": 60,
                "flags": "SYN",
                "payload_len": 0,
                "payload_signature": "",
                "label": AttackType.DDOS_SYN_FLOOD.value,
                "src_country": GEO.lookup(src),
            })
        return out

    def _port_scan(self, n: int, ts: float) -> List[Dict]:
        attacker = _rand_public_ip()
        target   = self._home_subnet + str(random.randint(2, 10))
        out = []
        for i in range(max(n, 25)):
            out.append({
                "ts": ts + i * 1e-4,
                "src_ip": attacker, "dst_ip": target,
                "src_port": random.randint(40000, 65000),
                "dst_port": random.randint(1, 1024),       # sweep low ports
                "protocol": "TCP",
                "length": 60,
                "flags": random.choice(["SYN", "FIN", "NULL", "XMAS"]),
                "payload_len": 0,
                "payload_signature": "",
                "label": AttackType.PORT_SCAN.value,
                "src_country": GEO.lookup(attacker),
            })
        return out

    def _brute_force(self, n: int, ts: float) -> List[Dict]:
        attacker = _rand_public_ip()
        target   = self._home_subnet + str(random.randint(2, 10))
        port     = random.choice([22, 21, 3389, 23])
        out = []
        for i in range(max(n, 15)):
            out.append({
                "ts": ts + i * 5e-4,
                "src_ip": attacker, "dst_ip": target,
                "src_port": random.randint(40000, 65000),
                "dst_port": port,
                "protocol": "TCP",
                "length": random.randint(80, 200),
                "flags": "PSH+ACK",
                "payload_len": random.randint(40, 160),
                "payload_signature": "auth_fail",
                "label": AttackType.BRUTE_FORCE_SSH.value,
                "src_country": GEO.lookup(attacker),
            })
        return out

    def _sql_injection(self, n: int, ts: float) -> List[Dict]:
        attacker = _rand_public_ip()
        target   = self._home_subnet + str(random.randint(2, 10))
        sig = random.choice([
            "' OR 1=1 --",
            "UNION SELECT password FROM users",
            "'; DROP TABLE users; --",
            "admin'/*",
        ])
        out = []
        for i in range(max(n, 5)):
            out.append({
                "ts": ts + i * 1e-3,
                "src_ip": attacker, "dst_ip": target,
                "src_port": random.randint(40000, 65000),
                "dst_port": random.choice([80, 443, 8080]),
                "protocol": "TCP",
                "length": random.randint(400, 1300),
                "flags": "PSH+ACK",
                "payload_len": random.randint(360, 1200),
                "payload_signature": sig,
                "label": AttackType.SQL_INJECTION.value,
                "src_country": GEO.lookup(attacker),
            })
        return out

    def _xss(self, n: int, ts: float) -> List[Dict]:
        attacker = _rand_public_ip()
        target   = self._home_subnet + str(random.randint(2, 10))
        sig = random.choice([
            "<script>alert(1)</script>",
            "<img src=x onerror=fetch('/x')>",
            "javascript:eval(atob(",
        ])
        return [{
            "ts": ts + i * 1e-3,
            "src_ip": attacker, "dst_ip": target,
            "src_port": random.randint(40000, 65000),
            "dst_port": random.choice([80, 443]),
            "protocol": "TCP",
            "length": random.randint(300, 900),
            "flags": "PSH+ACK",
            "payload_len": random.randint(260, 860),
            "payload_signature": sig,
            "label": AttackType.XSS.value,
            "src_country": GEO.lookup(attacker),
        } for i in range(max(n, 4))]

    def _dns_tunnel(self, n: int, ts: float) -> List[Dict]:
        attacker = _rand_public_ip()
        target   = self._home_subnet + str(random.randint(2, 10))
        return [{
            "ts": ts + i * 1e-3,
            "src_ip": target, "dst_ip": attacker,           # outbound DNS
            "src_port": random.randint(40000, 65000),
            "dst_port": 53,
            "protocol": "UDP",
            "length": random.randint(200, 600),
            "flags": "",
            "payload_len": random.randint(160, 560),
            "payload_signature": "long-subdomain-base64",
            "label": AttackType.DNS_TUNNELING.value,
            "src_country": GEO.lookup(attacker),
        } for i in range(max(n, 6))]

    def _exfil(self, n: int, ts: float) -> List[Dict]:
        attacker = _rand_public_ip()
        target   = self._home_subnet + str(random.randint(2, 10))
        return [{
            "ts": ts + i * 1e-3,
            "src_ip": target, "dst_ip": attacker,           # large outbound
            "src_port": random.randint(40000, 65000),
            "dst_port": random.choice([443, 22]),
            "protocol": "TCP",
            "length": random.randint(1200, 1500),
            "flags": "PSH+ACK",
            "payload_len": random.randint(1100, 1460),
            "payload_signature": "encrypted_blob",
            "label": AttackType.DATA_EXFILTRATION.value,
            "src_country": GEO.lookup(attacker),
        } for i in range(max(n, 10))]

    def _lateral(self, n: int, ts: float) -> List[Dict]:
        src = self._home_subnet + str(random.randint(2, 50))
        dst = self._home_subnet + str(random.randint(2, 50))
        port = random.choice([445, 3389, 22, 5985])
        return [{
            "ts": ts + i * 1e-3,
            "src_ip": src, "dst_ip": dst,
            "src_port": random.randint(40000, 65000),
            "dst_port": port,
            "protocol": "TCP",
            "length": random.randint(100, 400),
            "flags": "PSH+ACK",
            "payload_len": random.randint(60, 360),
            "payload_signature": "psexec/rdp",
            "label": AttackType.LATERAL_MOVEMENT.value,
            "src_country": GEO.lookup(src),
        } for i in range(max(n, 8))]

    def _c2(self, n: int, ts: float) -> List[Dict]:
        attacker = _rand_public_ip()
        target   = self._home_subnet + str(random.randint(2, 10))
        return [{
            "ts": ts + i * 1.0,                              # beaconing cadence
            "src_ip": target, "dst_ip": attacker,
            "src_port": random.randint(40000, 65000),
            "dst_port": random.choice([443, 8443, 53, 4444]),
            "protocol": "TCP",
            "length": random.randint(100, 300),
            "flags": "PSH+ACK",
            "payload_len": random.randint(60, 260),
            "payload_signature": "c2_beacon",
            "label": AttackType.MALWARE_C2.value,
            "src_country": GEO.lookup(attacker),
        } for i in range(max(n, 5))]

    def _recon(self, n: int, ts: float) -> List[Dict]:
        attacker = _rand_public_ip()
        return [{
            "ts": ts + i * 1e-3,
            "src_ip": attacker,
            "dst_ip": self._home_subnet + str(random.randint(2, 254)),
            "src_port": random.randint(40000, 65000),
            "dst_port": random.choice([80, 443, 22]),
            "protocol": "ICMP" if random.random() < 0.6 else "TCP",
            "length": random.randint(40, 100),
            "flags": "" if random.random() < 0.6 else "SYN",
            "payload_len": 0,
            "payload_signature": "icmp_sweep",
            "label": AttackType.RECONNAISSANCE.value,
            "src_country": GEO.lookup(attacker),
        } for i in range(max(n, 12))]


def generate_labeled_dataset(n_packets: int = 50_000, seed: int = 42):
    """Build a labelled packet list for training scripts."""
    sim = TrafficSimulator(packets_per_second=1000,
                           attack_probability=0.25,
                           seed=seed)
    return sim.sample_batch(n_packets)


if __name__ == "__main__":
    sim = TrafficSimulator(seed=0)
    for i, pkt in enumerate(sim.sample_batch(10)):
        print(pkt)
        if i >= 9:
            break
