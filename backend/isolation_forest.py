"""
Isolation Forest anomaly detection on vibration feature windows (rms, peak, std),
following the approach in the anchor paper (PMC12609400). Reads from SQLite, writes
its anomaly flags back for the dashboard comparison.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from db import fetch_recent, update_flags


def score_and_store(contamination=0.05, n=300):
    rows = fetch_recent(n=n)
    if len(rows) < 10:
        print("Not enough data yet for Isolation Forest.")
        return

    X = np.array([[r["rms"], r["peak"], r["std"]] for r in rows], dtype=float)
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)  # -1 anomaly, 1 normal

    for row, pred in zip(rows, preds):
        update_flags(row["id"], iforest_flag=int(pred == -1))

    print(f"Isolation Forest scored {len(rows)} windows, "
          f"{(preds == -1).sum()} anomalies.")


if __name__ == "__main__":
    score_and_store()
