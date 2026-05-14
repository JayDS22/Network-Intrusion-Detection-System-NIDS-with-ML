"""
Alert hub. Builds Alert objects from detection events and dispatches them
to enabled channels based on severity.

In demo mode every channel except console is off, so alerts are printed
and appended to logs/alerts.log.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

from .email_notifier import EmailNotifier
from .slack_notifier import SlackNotifier


@dataclass
class Alert:
    ts: float
    severity: str            # critical | high | medium | low
    title: str
    description: str
    src_ip: str
    dst_ip: str
    attack_type: str
    confidence: float
    detector: str            # ml | rule | ml+rule

    def to_dict(self):       return asdict(self)


class NotificationSystem:

    def __init__(self, cfg: Dict, log_dir: str = "logs"):
        self.cfg = cfg or {}
        self.enabled  = bool(self.cfg.get("enabled", True))
        self.channels = self.cfg.get("channels", {"console": True})
        self.severity_routing = self.cfg.get("severity_routing", {})

        self.email  = EmailNotifier(self.cfg.get("email", {}))
        self.slack  = SlackNotifier(self.cfg.get("slack", {}))
        self.webhook_url = self.cfg.get("webhook", {}).get("url", "")

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "alerts.log")
        self.logger = logging.getLogger("nids.alerts")
        if not self.logger.handlers:
            handler = logging.FileHandler(self._log_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def emit(self, event_dict: Dict) -> Optional[Alert]:
        if not self.enabled:
            return None
        sev = event_dict.get("severity", "low")
        alert = Alert(
            ts=event_dict["ts"],
            severity=sev,
            title=f"{event_dict.get('label_pred', 'unknown')} from "
                  f"{event_dict.get('src_ip')}",
            description=(f"{event_dict.get('detector','?').upper()} flagged "
                         f"{event_dict.get('label_pred','unknown')} "
                         f"({event_dict.get('src_ip')} -> "
                         f"{event_dict.get('dst_ip')}:"
                         f"{event_dict.get('dst_port')})"),
            src_ip=event_dict.get("src_ip", ""),
            dst_ip=event_dict.get("dst_ip", ""),
            attack_type=event_dict.get("label_pred", "unknown"),
            confidence=float(event_dict.get("confidence", 0.0)),
            detector=event_dict.get("detector", "ml"),
        )

        # Append to file log no matter what
        self.logger.info(json.dumps({
            "ts_human": datetime.utcfromtimestamp(alert.ts).isoformat(),
            **alert.to_dict(),
        }))

        # Route to channels
        for ch in self.severity_routing.get(sev, ["console"]):
            try:
                self._route(ch, alert)
            except Exception:
                pass
        return alert

    def _route(self, channel: str, alert: Alert) -> None:
        if not self.channels.get(channel, False) and channel != "console":
            return
        if channel == "console":
            print(f"[ALERT][{alert.severity.upper()}] {alert.title}: "
                  f"{alert.description}")
        elif channel == "email":
            self.email.send(
                subject=f"[NIDS][{alert.severity.upper()}] {alert.attack_type}",
                body=f"{alert.description}\nconfidence={alert.confidence:.2f}",
            )
        elif channel == "slack":
            self.slack.send(
                f":rotating_light: *{alert.severity.upper()}* "
                f"{alert.attack_type}\n{alert.description}"
            )
        elif channel == "webhook":
            try:
                import urllib.request
                req = urllib.request.Request(
                    self.webhook_url,
                    data=json.dumps(alert.to_dict()).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=3)
            except Exception:
                pass

    def recent(self, n: int = 100) -> List[Dict]:
        if not os.path.exists(self._log_path):
            return []
        with open(self._log_path) as fh:
            lines = fh.readlines()[-n:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
