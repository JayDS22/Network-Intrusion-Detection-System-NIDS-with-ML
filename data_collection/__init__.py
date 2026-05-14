"""Data collection layer: live capture, PCAP parsing, and synthetic traffic."""
from .traffic_simulator import TrafficSimulator, AttackType

__all__ = ["TrafficSimulator", "AttackType"]
