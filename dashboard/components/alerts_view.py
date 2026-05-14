"""Recent-alert table."""
from __future__ import annotations

from typing import Dict, List

import streamlit as st

from dashboard.utils import (ATTACK_EMOJI, SEVERITY_COLORS, fmt_ts,
                             severity_pill)


def render(events: List[Dict], limit: int = 20) -> None:
    if not events:
        st.info("No alerts yet. System is healthy.")
        return

    rows = list(reversed(events[-limit:]))
    html = ["<div class='alerts-table'>"]
    html.append("<div class='alerts-header'>"
                "<span>Time</span><span>Severity</span><span>Type</span>"
                "<span>Source</span><span>Destination</span>"
                "<span>Detector</span><span>Score</span></div>")
    for ev in rows:
        sev   = ev.get("severity", "low")
        emoji = ATTACK_EMOJI.get(ev.get("label_pred", ""), "*")
        color = SEVERITY_COLORS.get(sev, "#666")
        html.append(
            "<div class='alerts-row' style='border-left:4px solid "
            f"{color};'>"
            f"<span class='muted'>{fmt_ts(ev['ts'])}</span>"
            f"<span>{severity_pill(sev)}</span>"
            f"<span>{emoji} <code>{ev.get('label_pred')}</code></span>"
            f"<span><code>{ev.get('src_ip')}</code>"
            f" <span class='muted'>({ev.get('src_country','?')})</span></span>"
            f"<span><code>{ev.get('dst_ip')}:{ev.get('dst_port')}</code></span>"
            f"<span><span class='detector-{ev.get('detector','ml')}'>"
            f"{ev.get('detector','ml')}</span></span>"
            f"<span>{ev.get('score', 0):.2f}</span>"
            "</div>"
        )
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)
