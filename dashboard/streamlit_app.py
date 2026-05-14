"""
Streamlit dashboard.

Usage:
    streamlit run dashboard/streamlit_app.py

Starts a StreamProcessor over the synthetic TrafficSimulator. Live
capture is not used here so no root privileges are required.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from alerts.notification_system        import NotificationSystem
from dashboard.components              import alerts_view, charts, metrics
from dashboard.utils                   import (ATTACK_EMOJI, fmt_full_ts,
                                               load_config)
from data_collection.traffic_simulator import TrafficSimulator
from feature_engineering.extractor     import FeatureExtractor
from models.ensemble_detector          import EnsembleDetector
from real_time.rule_engine             import RuleEngine
from real_time.stream_processor        import StreamProcessor


st.set_page_config(
    page_title="NIDS-ML | Real-time Network Intrusion Detection",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root{
  --bg:#0d0f14;
  --panel:#161a23;
  --panel-2:#1f2530;
  --text:#e4e7ee;
  --muted:#7a8497;
  --accent:#7af0ff;
  --danger:#ff3b3b;
}
.stApp{
  background: radial-gradient(circle at top left, #131826 0%, #0a0c12 60%) !important;
  color: var(--text);
}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem; padding-bottom:1rem;}

/* Brand bar */
.brand-bar{
  display:flex; align-items:center; justify-content:space-between;
  padding: 0.85rem 1.1rem;
  background: linear-gradient(90deg, rgba(122,240,255,0.10), rgba(255,59,59,0.10));
  border: 1px solid rgba(122,240,255,0.25);
  border-radius: 14px; margin-bottom: 1rem;
}
.brand-title{font-size:1.45rem; font-weight:700; letter-spacing:0.4px;}
.brand-sub  {color: var(--muted); font-size: 0.85rem;}
.brand-status{display:flex; align-items:center; gap:0.5rem;}
.status-dot{
  width:10px;height:10px;border-radius:50%;
  background:#7cffa9; box-shadow:0 0 12px #7cffa9;
  animation: pulse 1.6s infinite;
}


@keyframes pulse{
  0%{transform:scale(1);opacity:1;}
  50%{transform:scale(1.25);opacity:0.6;}
  100%{transform:scale(1);opacity:1;}
}

/* KPI cards */
.kpi-card{
  background: var(--panel);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 0.95rem 1rem;
  height: 100%;
  box-shadow: 0 4px 14px rgba(0,0,0,0.30);
  transition: transform 0.18s ease;
}
.kpi-card:hover{transform: translateY(-2px); border-color: rgba(122,240,255,0.4);}
.kpi-label{color: var(--muted); font-size: 0.72rem; letter-spacing: 1px;}
.kpi-value{font-size: 1.95rem; font-weight: 700; margin-top: 0.2rem;}
.kpi-delta{color: var(--muted); font-size: 0.72rem; margin-top: 0.3rem;}

/* Section headers */
.section-h{
  font-weight:700; font-size:1.05rem; letter-spacing:0.4px;
  color: var(--accent); margin: 0.4rem 0 0.4rem 0.2rem;
  text-transform: uppercase;
}

/* Alerts table */
.alerts-table{display:flex; flex-direction:column; gap:6px;}
.alerts-header, .alerts-row{
  display:grid;
  grid-template-columns: 90px 90px 1.6fr 1.8fr 1.8fr 100px 70px;
  align-items:center; gap:10px; padding:8px 12px;
  background: var(--panel);
  border-radius: 10px;
  font-size:0.86rem;
}
.alerts-header{
  color: var(--muted); font-size:0.7rem; letter-spacing:1px;
  background: transparent; padding-bottom:0;
}
.alerts-row code{
  background: rgba(122,240,255,0.07);
  padding: 2px 7px; border-radius: 6px; color:#cfd6e3;
}
.muted{color: var(--muted);}
.detector-ml      {color:#7af0ff;}
.detector-rule    {color:#ffd23b;}
.detector-ml\\+rule{color:#7cffa9;}
.detector-ml.rule {color:#7cffa9;}

/* Streamlit overrides */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #11141c, #0c0e15);
  border-right: 1px solid rgba(255,255,255,0.06);
}
.stRadio>div, .stSelectbox>div, .stCheckbox>div{color: var(--text)!important;}

/* DataFrame look */
.stDataFrame, .stTable{background: var(--panel); border-radius: 10px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ML ensemble...")
def boot(cfg_path: str = "config.yaml"):
    cfg = load_config(cfg_path if os.path.isabs(cfg_path)
                      else str(ROOT / cfg_path))
    model_dir = str(ROOT / cfg["storage"]["model_dir"])
    ensemble = None
    if os.path.exists(os.path.join(model_dir, "ensemble_meta.joblib")):
        try:
            ensemble = EnsembleDetector.load(model_dir)
        except Exception as exc:                                # pragma: no cover
            st.warning(f"Could not load trained models: {exc}. "
                       "Falling back to rule-only detection.")
            ensemble = None
    else:
        st.warning("No trained models found in `models/saved/`. "
                   "Run **`python scripts/train_model.py`** to enable the "
                   "ML ensemble (rules still active).")

    rule_engine = RuleEngine(cfg.get("rules", {}))
    sim = TrafficSimulator(
        packets_per_second=cfg["simulator"]["packets_per_second"],
        attack_probability=cfg["simulator"]["attack_probability"],
    )
    extractor = FeatureExtractor(window_seconds=30)
    proc = StreamProcessor(ensemble=ensemble,
                           rule_engine=rule_engine,
                           source=sim.stream(),
                           extractor=extractor,
                           max_events=2000)
    proc.start()
    notif = NotificationSystem(cfg.get("alerts", {}),
                               log_dir=str(ROOT / "logs"))
    return cfg, proc, notif


cfg, processor, notifier = boot()


with st.sidebar:
    st.markdown("### NIDS-ML Control Center")
    st.caption("v1.0.0  |  Demo mode (synthetic traffic)")

    refresh = st.slider("Refresh interval (seconds)",
                        1, 10, int(cfg["dashboard"]["refresh_seconds"]))
    show_n  = st.slider("Recent alerts shown", 10, 100, 25)

    st.markdown("---")
    st.markdown("**Simulator**")
    st.write(f"- {cfg['simulator']['packets_per_second']} pps target")
    st.write(f"- {cfg['simulator']['attack_probability']*100:.0f}% attack bursts")

    st.markdown("---")
    st.markdown("**Models loaded**")
    if processor.ensemble:
        st.success("[ok] Isolation Forest")
        st.success("[ok] Random Forest")
        if processor.ensemble.lstm_model:
            backend = "TF" if getattr(processor.ensemble.lstm_model, "use_tf", False) else "GBM"
            st.success(f"[ok] LSTM ({backend})")
        else:
            st.info("LSTM not loaded")
    else:
        st.warning("ML ensemble offline\nRun: `python scripts/train_model.py`")

    st.markdown("---")
    st.markdown("**Detection channels**")
    for ch, enabled in cfg.get("alerts", {}).get("channels", {}).items():
        prefix = "[x]" if enabled else "[ ]"
        st.write(f"{prefix} {ch}")

    st.markdown("---")
    st.button("Reset counters",
              on_click=lambda: processor._counters.update(
                  {k: 0 for k in processor._counters}))


st.markdown(f"""
<div class='brand-bar'>
  <div>
    <div class='brand-title'>NIDS-ML Real-time Intrusion Detection</div>
    <div class='brand-sub'>
      Ensemble (Isolation Forest + Random Forest + LSTM) ·
      Signature engine · Multi-channel alerting
    </div>
  </div>
  <div class='brand-status'>
    <div class='status-dot'></div>
    <div>
      <div style='font-weight:600;'>LIVE</div>
      <div class='brand-sub'>{fmt_full_ts(time.time())}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


m      = processor.snapshot_metrics()
events = processor.snapshot_events(limit=cfg["dashboard"]["max_alerts_displayed"])

if "seen_alert_ids" not in st.session_state:
    st.session_state.seen_alert_ids = set()
for ev in events:
    if ev["id"] not in st.session_state.seen_alert_ids:
        notifier.emit(ev)
        st.session_state.seen_alert_ids.add(ev["id"])


metrics.render(m)


st.markdown("<div class='section-h'>Live Traffic Telemetry</div>",
            unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])
with c1:
    st.plotly_chart(charts.traffic_timeseries(events,
                    list(processor._pps_buf)),
                    config={"displayModeBar": False},
                    use_container_width=True)
with c2:
    st.plotly_chart(charts.ensemble_breakdown(events),
                    config={"displayModeBar": False},
                    use_container_width=True)


st.markdown("<div class='section-h'>Threat Analytics</div>",
            unsafe_allow_html=True)
c3, c4, c5 = st.columns(3)
with c3:
    st.plotly_chart(charts.attack_type_bar(events),
                    config={"displayModeBar": False},
                    use_container_width=True)
with c4:
    st.plotly_chart(charts.severity_bar(m),
                    config={"displayModeBar": False},
                    use_container_width=True)
with c5:
    st.plotly_chart(charts.protocol_donut(events),
                    config={"displayModeBar": False},
                    use_container_width=True)


st.markdown("<div class='section-h'>Network Geography &amp; Top Talkers</div>",
            unsafe_allow_html=True)
c6, c7 = st.columns([1, 2])
with c6:
    st.plotly_chart(charts.top_talkers(events),
                    config={"displayModeBar": False},
                    use_container_width=True)
with c7:
    st.plotly_chart(charts.geo_map(events),
                    config={"displayModeBar": False},
                    use_container_width=True)


st.markdown("<div class='section-h'>Recent Alerts</div>",
            unsafe_allow_html=True)
alerts_view.render(events, limit=show_n)


with st.expander("System internals"):
    cols = st.columns(4)
    cols[0].metric("Throughput",       f"{m.get('pps_recent', 0):.0f} pps")
    cols[1].metric("ML+Rule consensus", m.get("ml_plus_rule", 0))
    cols[2].metric("ML only",           m.get("ml_only", 0))
    cols[3].metric("Rule only",         m.get("rule_only", 0))
    st.json({k: v for k, v in m.items() if not k.startswith("_")},
            expanded=False)


time.sleep(refresh)
st.rerun()
