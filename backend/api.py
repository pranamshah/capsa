"""
FastAPI service exposing vibration windows and model outputs.

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET /latest
    GET /history?n=300
    GET /model-comparison
    GET /battery      -> battery voltage series for the discharge curve
"""

from fastapi import FastAPI
from db import fetch_recent, init_db

app = FastAPI(title="Vibration Monitor API")
init_db()


@app.get("/latest")
def latest():
    rows = fetch_recent(n=1)
    return rows[0] if rows else {}


@app.get("/history")
def history(n: int = 300):
    return fetch_recent(n=n)


@app.get("/model-comparison")
def model_comparison(n: int = 500):
    rows = fetch_recent(n=n)
    return {
        "total_windows": len(rows),
        "iforest_flagged": sum(1 for r in rows if r["iforest_flag"]),
        "autoencoder_flagged": sum(1 for r in rows if r["autoencoder_flag"]),
    }


@app.get("/battery")
def battery(n: int = 500):
    rows = fetch_recent(n=n)
    return [{"ts": r["ts"], "vbat": r["vbat"]} for r in rows if r["vbat"] is not None]
