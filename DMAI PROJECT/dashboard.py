"""Streamlit dashboard.  Run:  streamlit run dashboard.py
Shows the live annotated feed, crowd graph and recent alerts with snapshots."""
from pathlib import Path

import pandas as pd
import streamlit as st

import config as C
from modules import db

st.set_page_config(page_title="Smart Street Surveillance", page_icon="📹", layout="wide")
db.init_db()

st.title("Smart Street Surveillance System")
st.caption("Crowd estimation · Weapon detection · Fight recognition · Night mode")

cameras = db.list_cameras()
if not cameras:
    st.info("No camera yet. Start the pipeline first:  python main.py --source <video> --camera cam1")
    st.stop()

with st.sidebar:
    st.header("Filters")
    by_id = {c["id"]: c for c in cameras}
    cam_id = st.selectbox("Camera", list(by_id),
                          format_func=lambda i: f"{by_id[i]['name']} ({by_id[i]['status']})")
    cam = by_id[cam_id]
    types = st.multiselect("Alert types", ["crowd", "weapon", "fight"], default=["crowd", "weapon", "fight"])
    include_false = st.checkbox("Show false alarms", value=False)
    minutes = st.slider("Crowd graph window (minutes)", 1, 60, 10)


@st.fragment(run_every="1s")
def live_panel():
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Live feed")
        img = C.LIVE_DIR / f"{cam['name']}.jpg"
        try:
            st.image(img.read_bytes())
        except OSError:
            st.info("Waiting for frames from main.py ...")
    with right:
        status = next((c["status"] for c in db.list_cameras() if c["id"] == cam["id"]), "offline")
        counts = db.counts_since(cam["id"], minutes)
        current = counts[-1]["people_count"] if counts else 0
        a, b, c = st.columns(3)
        a.metric("People now", current)
        b.metric("Crowd limit", cam["crowd_limit"])
        c.metric("Alerts today", db.alerts_today(cam["id"]))
        st.write(f"Camera status: **{status}**")
        st.subheader("Crowd count")
        if counts:
            df = pd.DataFrame(counts)
            df["recorded_at"] = pd.to_datetime(df["recorded_at"])
            st.line_chart(df.set_index("recorded_at")["people_count"])
        else:
            st.info("No counts yet.")


@st.fragment(run_every="3s")
def alerts_panel():
    st.subheader("Recent alerts")
    rows = db.recent_alerts(cam["id"], types, include_false, limit=20)
    if not rows:
        st.info("No alerts yet.")
        return
    icons = {"crowd": "👥", "weapon": "🔪", "fight": "🥊"}
    for r in rows:
        with st.container(border=True):
            img_col, text_col, btn_col = st.columns([2, 4, 2])
            snap = Path(r["snapshot_path"] or "")
            if snap.is_file():
                img_col.image(snap.read_bytes())
            text_col.markdown(f"**{icons.get(r['alert_type'], '⚠️')} {r['alert_type'].upper()}**  ·  {r['created_at']}")
            text_col.write(r["message"])
            text_col.caption(f"confidence {r['confidence']}")
            if r["reviewed"]:
                btn_col.write("❌ False alarm" if r["is_false_alarm"] else "✅ Genuine")
            else:
                if btn_col.button("✅ Genuine", key=f"g{r['id']}"):
                    db.review_alert(r["id"], False)
                    st.rerun()
                if btn_col.button("❌ False alarm", key=f"f{r['id']}"):
                    db.review_alert(r["id"], True)
                    st.rerun()


live_panel()
alerts_panel()
