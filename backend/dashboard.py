"""
Remote-viewable Streamlit dashboard for the DC power monitoring project.

Run locally:
    streamlit run dashboard.py

Deploy for true remote access (public URL) via Streamlit Community Cloud,
Render, or Railway -- point them at this file as the entry point.
"""

import time
import streamlit as st

from thingspeak_fetch import fetch_recent
from isolation_forest_analysis import detect_anomalies

st.set_page_config(page_title="DC Power Monitor", layout="wide")
st.title("IoT-Based Remote DC Power Signal Monitoring and Anomaly Detection")
st.caption("ESP32 -> ThingSpeak -> Isolation Forest -> Live Dashboard")

REFRESH_SECONDS = 10

placeholder = st.empty()

while True:
    df = fetch_recent(results=200)

    with placeholder.container():
        if df.empty:
            st.info("Waiting for data from ESP32 / ThingSpeak...")
        else:
            df = detect_anomalies(df)

            col1, col2, col3 = st.columns(3)
            latest = df.iloc[-1]
            col1.metric("Latest voltage", f"{latest['voltage']:.3f} V")
            col2.metric("On-device flag", "ANOMALY" if latest["onboard_anomaly"] else "normal")
            col3.metric("Isolation Forest flag", "ANOMALY" if latest["iforest_anomaly"] else "normal")

            st.subheader("Voltage over time")
            chart_df = df.set_index("created_at")[["voltage"]]
            st.line_chart(chart_df)

            st.subheader("Recent readings")
            st.dataframe(
                df[["created_at", "voltage", "onboard_anomaly", "iforest_anomaly"]]
                .sort_values("created_at", ascending=False)
                .head(20),
                use_container_width=True,
            )

    time.sleep(REFRESH_SECONDS)
