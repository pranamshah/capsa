"""SQLite helper for storing incoming vibration feature windows + battery voltage."""

import sqlite3
from contextlib import contextmanager

DB_PATH = "vibration.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT (datetime('now')),
                rms REAL NOT NULL,
                peak REAL NOT NULL,
                std REAL NOT NULL,
                vbat REAL,
                iforest_flag INTEGER,
                autoencoder_flag INTEGER
            )
        """)
        conn.commit()


def insert_window(rms, peak, std, vbat):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO windows (rms, peak, std, vbat) VALUES (?, ?, ?, ?)",
            (rms, peak, std, vbat),
        )
        conn.commit()


def fetch_recent(n=500):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, ts, rms, peak, std, vbat, iforest_flag, autoencoder_flag "
            "FROM windows ORDER BY id DESC LIMIT ?", (n,),
        )
        rows = cur.fetchall()
    cols = ["id", "ts", "rms", "peak", "std", "vbat", "iforest_flag", "autoencoder_flag"]
    return [dict(zip(cols, r)) for r in reversed(rows)]


def update_flags(row_id, iforest_flag=None, autoencoder_flag=None):
    with get_conn() as conn:
        if iforest_flag is not None:
            conn.execute("UPDATE windows SET iforest_flag=? WHERE id=?", (iforest_flag, row_id))
        if autoencoder_flag is not None:
            conn.execute("UPDATE windows SET autoencoder_flag=? WHERE id=?", (autoencoder_flag, row_id))
        conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"Initialized {DB_PATH}")
