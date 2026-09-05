"""
Runs sklearn's IsolationForest on fetched voltage readings for a deeper,
retrospective anomaly check -- complements the ESP32's lightweight on-device
z-score flag with a proper unsupervised ML model, per the anchor paper's approach.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    df must have a 'voltage' column.
    Adds a column 'iforest_anomaly' (1 = anomaly, 0 = normal).
    """
    if len(df) < 10:
        df = df.copy()
        df["iforest_anomaly"] = 0
        return df

    X = df[["voltage"]].values

    # simple engineered features help Isolation Forest catch trend-based anomalies,
    # not just single-point outliers
    df = df.copy()
    df["rolling_mean"] = df["voltage"].rolling(window=5, min_periods=1).mean()
    df["rolling_std"] = df["voltage"].rolling(window=5, min_periods=1).std().fillna(0)
    X = df[["voltage", "rolling_mean", "rolling_std"]].values

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)  # -1 = anomaly, 1 = normal

    df["iforest_anomaly"] = (preds == -1).astype(int)
    return df


if __name__ == "__main__":
    from thingspeak_fetch import fetch_recent

    df = fetch_recent()
    result = detect_anomalies(df)
    print(result.tail(20))
