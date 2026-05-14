"""Slack incoming-webhook notifier. Disabled by default."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict


class SlackNotifier:
    def __init__(self, config: Dict):
        self.cfg = config or {}

    def send(self, text: str) -> bool:
        url = os.path.expandvars(self.cfg.get("webhook_url", ""))
        if not url or url.startswith("${"):
            return False
        payload = json.dumps({
            "channel": self.cfg.get("channel", "#nids-alerts"),
            "text":    text,
            "username": "nids-bot",
        }).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=4) as r:
                return 200 <= r.status < 300
        except Exception:
            return False
