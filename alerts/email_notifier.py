"""SMTP email notifier. Disabled by default; enable via config.yaml."""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict


class EmailNotifier:
    def __init__(self, config: Dict):
        self.cfg = config or {}

    def send(self, subject: str, body: str) -> bool:
        cfg = self.cfg
        host  = cfg.get("smtp_host")
        port  = int(cfg.get("smtp_port", 587))
        user  = os.path.expandvars(cfg.get("username", "")) or cfg.get("username", "")
        pwd   = os.path.expandvars(cfg.get("password", "")) or cfg.get("password", "")
        sender = cfg.get("from_addr")
        recipients = cfg.get("to_addrs", [])
        if not (host and sender and recipients):
            return False

        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(recipients), subject
        msg.attach(MIMEText(body, "plain"))
        try:
            with smtplib.SMTP(host, port) as s:
                s.starttls()
                if user and pwd:
                    s.login(user, pwd)
                s.sendmail(sender, recipients, msg.as_string())
            return True
        except Exception:
            return False
