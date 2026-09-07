"""
Streamlit dashboard: live vibration features, battery discharge curve, and a
3-way anomaly comparison (Isolation Forest vs Autoencoder, plus raw feature trend).

Run:
    streamlit run dashboard.py
Assumes the FastAPI service runs at API_BASE.
"""

import time
import requests
import pandas as pd
import streamlit as st

API_BASE = "http://localhost:8000"
REFRESH_SECONDS = 5

st.set_page_config(page_title="Vibration Signal Monitor", layout="wide")
st.title("Computer-Aided Vibration Signal Analysis & Anomaly Detection")
st.caption("ESP32 + MPU6050 (MQTT) -> SQLite -> Isolation Forest + Autoencoder -> Dashboard")

placeholder = st.empty()

while True:
    try:
        history = requests.get(f"{API_BASE}/history", params={"n": 300}, timeout=5).json()
        comparison = requests.get(f"{API_BASE}/model-comparison", timeout=5).json()
        battery = requests.get(f"{API_BASE}/battery", timeout=5).json()
    except requests.RequestException:
        history, comparison, battery = [], {}, []

    with placeholder.container():
        if not history:
            st.info("Waiting for data... ensure mqtt_subscriber.py and the API are running.")
        else:
            df = pd.DataFrame(history)
            latest = df.iloc[-1]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMS", f"{latest['rms']:.4f} g")
            c2.metric("Battery", f"{latest['vbat']:.3f} V" if latest["vbat"] else "n/a")
            c3.metric("Isolation Forest", "ANOMALY" if latest.get("iforest_flag") else "normal")
            c4.metric("Autoencoder", "ANOMALY" if latest.get("autoencoder_flag") else "normal")

            st.subheader("Vibration features over time")
            st.line_chart(df.set_index("ts")[["rms", "peak", "std"]])

            if battery:
                st.subheader("Battery discharge curve")
                bdf = pd.DataFrame(battery)
                st.line_chart(bdf.set_index("ts")[["vbat"]])

            st.subheader("Model comparison (last 500 windows)")
            st.json(comparison)

            st.subheader("Recent windows")
            st.dataframe(df.sort_values("id", ascending=False).head(20),
                         use_container_width=True)

    time.sleep(REFRESH_SECONDS)
