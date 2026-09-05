"""
Fetches recent readings from a ThingSpeak channel into a pandas DataFrame.

Fill in CHANNEL_ID (and READ_API_KEY if your channel is private) below.
"""

import requests
import pandas as pd

CHANNEL_ID = "YOUR_CHANNEL_ID"
READ_API_KEY = None  # set to your Read API Key string if the channel is private


def fetch_recent(results: int = 200) -> pd.DataFrame:
    """Fetch the most recent `results` entries from the ThingSpeak channel."""
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    params = {"results": results}
    if READ_API_KEY:
        params["api_key"] = READ_API_KEY

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    feeds = payload.get("feeds", [])
    if not feeds:
        return pd.DataFrame(columns=["created_at", "voltage", "onboard_anomaly"])

    df = pd.DataFrame(feeds)
    df = df.rename(columns={"field1": "voltage", "field2": "onboard_anomaly"})
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["voltage"] = pd.to_numeric(df["voltage"], errors="coerce")
    df["onboard_anomaly"] = pd.to_numeric(df["onboard_anomaly"], errors="coerce").fillna(0).astype(int)
    return df[["created_at", "voltage", "onboard_anomaly"]].dropna(subset=["voltage"])


if __name__ == "__main__":
    df = fetch_recent()
    print(df.tail(20))
