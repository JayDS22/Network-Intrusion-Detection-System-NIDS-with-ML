"""Plotly chart helpers for the dashboard."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY = "#7af0ff"
DANGER  = "#ff3b3b"
WARN    = "#ff9f3b"
SUCCESS = "#7cffa9"

ATTACK_COLOR = {
    "benign":            "#3b8d3b",
    "ddos_syn_flood":    "#ff3b3b",
    "ddos_udp_flood":    "#ff5d5d",
    "port_scan":         "#ff9f3b",
    "brute_force_ssh":   "#ffbf5d",
    "brute_force":       "#ffbf5d",
    "sql_injection":     "#ff7af0",
    "xss":               "#b894ff",
    "xss_attempt":       "#b894ff",
    "dns_tunneling":     "#7af0ff",
    "data_exfiltration": "#ff3bd1",
    "lateral_movement":  "#3bff9f",
    "malware_c2":        "#d1ff3b",
    "reconnaissance":    "#5dffd1",
    "mitm_arp_spoof":    "#ff5dbf",
}


def _empty_fig(msg: str = "Waiting for data...") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper",
                          yref="paper", showarrow=False,
                          font=dict(size=14, color="#888"))],
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def traffic_timeseries(events: List[Dict], pps_buf: List[float]) -> go.Figure:
    """Stacked area chart of packets/second; events overlay as red bars."""
    if not pps_buf:
        return _empty_fig()

    df = pd.DataFrame({"second": list(range(-len(pps_buf), 0)),
                       "pps":    pps_buf})
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["second"], y=df["pps"], mode="lines",
        name="packets/sec", line=dict(color=PRIMARY, width=2),
        fill="tozeroy", fillcolor="rgba(122,240,255,0.15)",
    ))

    if events:
        df_e = pd.DataFrame(events)
        df_e = df_e.tail(200)
        df_e["sec_offset"] = (df_e["ts"] - df_e["ts"].max()).clip(lower=-len(pps_buf))
        fig.add_trace(go.Scatter(
            x=df_e["sec_offset"], y=[max(pps_buf) * 0.9] * len(df_e),
            mode="markers", name="alerts",
            marker=dict(size=8, color=DANGER, symbol="x"),
            hovertext=df_e["label_pred"] + " from " + df_e["src_ip"],
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Live Traffic & Alerts (last 5 min)",
        xaxis_title="seconds ago",
        yaxis_title="packets/sec",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320, hovermode="x unified",
    )
    return fig


def protocol_donut(events: List[Dict]) -> go.Figure:
    counts = Counter(e["protocol"] for e in events) if events else Counter()
    if not counts:
        return _empty_fig("No protocols yet")
    fig = go.Figure(data=[go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.55,
        marker=dict(colors=[PRIMARY, WARN, SUCCESS, DANGER]),
        textinfo="label+percent",
    )])
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      title="Protocol Distribution (alerted flows)",
                      height=320,
                      margin=dict(l=10, r=10, t=40, b=10),
                      showlegend=False)
    return fig


def severity_bar(metrics: Dict) -> go.Figure:
    sev = ["critical", "high", "medium", "low"]
    vals = [metrics.get(f"alerts_{s}", 0) for s in sev]
    fig = go.Figure(data=[go.Bar(
        x=sev, y=vals, marker_color=[DANGER, WARN, "#ffd23b", PRIMARY],
        text=vals, textposition="outside",
    )])
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      title="Alerts by Severity",
                      yaxis=dict(title="count"),
                      height=320,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def attack_type_bar(events: List[Dict]) -> go.Figure:
    if not events:
        return _empty_fig("No alerts yet")
    counts = Counter(e["label_pred"] for e in events)
    labels = [l for l, _ in counts.most_common(10)]
    values = [counts[l] for l in labels]
    colors = [ATTACK_COLOR.get(l, PRIMARY) for l in labels]
    fig = go.Figure(data=[go.Bar(
        y=labels, x=values, orientation="h",
        marker_color=colors, text=values, textposition="outside",
    )])
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      title="Top Attack Categories",
                      xaxis_title="alerts",
                      height=380,
                      margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


def top_talkers(events: List[Dict]) -> go.Figure:
    if not events:
        return _empty_fig("No alerts yet")
    counts = Counter(e["src_ip"] for e in events)
    top    = counts.most_common(10)
    fig = go.Figure(data=[go.Bar(
        y=[ip for ip, _ in top],
        x=[c for _, c in top],
        orientation="h",
        marker_color=WARN,
        text=[c for _, c in top],
        textposition="outside",
    )])
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      title="Top Talkers (Source IPs)",
                      xaxis_title="alerts",
                      height=380,
                      margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


def geo_map(events: List[Dict]) -> go.Figure:
    if not events:
        return _empty_fig("No geo data yet")
    counts = Counter(e.get("src_country", "") for e in events if e.get("src_country"))
    if not counts:
        return _empty_fig("No geo data yet")
    df = pd.DataFrame({"country": list(counts.keys()),
                       "alerts":  list(counts.values())})
    fig = px.choropleth(df, locations="country", locationmode="ISO-3",
                        color="alerts", color_continuous_scale="Reds",
                        title="Source-IP Geography")
    # locationmode "ISO-3" needs 3-letter codes; we ship 2-letter -> use 'country names' fallback
    fig = px.choropleth(df, locations="country", locationmode="country names",
                        color="alerts", color_continuous_scale="Reds",
                        title="Source-IP Geography (alerts)")
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      height=420,
                      margin=dict(l=10, r=10, t=40, b=10),
                      geo=dict(bgcolor="rgba(0,0,0,0)",
                               landcolor="#1f2530",
                               coastlinecolor="#3a4150",
                               showframe=False))
    return fig


def ensemble_breakdown(events: List[Dict]) -> go.Figure:
    """Radar of mean component scores across recent alerts."""
    if not events:
        return _empty_fig("No alerts yet")
    rf = []; iforest = []; lstm = []
    for e in events:
        c = e.get("components", {}) or {}
        if c:
            iforest.append(c.get("isolation_forest", 0))
            rf.append(c.get("random_forest", 0))
            lstm.append(c.get("lstm", 0))
    if not rf:
        return _empty_fig("No ML components")
    avg = [sum(iforest)/len(iforest), sum(rf)/len(rf), sum(lstm)/len(lstm)]
    fig = go.Figure(data=go.Scatterpolar(
        r=avg + [avg[0]],
        theta=["Isolation Forest", "Random Forest", "LSTM", "Isolation Forest"],
        fill="toself",
        line=dict(color=PRIMARY),
        fillcolor="rgba(122,240,255,0.25)",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      title="Ensemble Component Mean Score",
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      height=320,
                      margin=dict(l=10, r=10, t=40, b=10),
                      showlegend=False)
    return fig
