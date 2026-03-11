import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "hotel_predictions.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at     TEXT    NOT NULL,
            hotel          TEXT,
            lead_time      INTEGER,
            deposit_type   TEXT,
            market_segment TEXT,
            adr            REAL,
            features_json  TEXT    NOT NULL,
            prediction     INTEGER NOT NULL,
            cancel_prob    REAL    NOT NULL
        )
        """)
        conn.commit()


def insert_prediction(created_at, hotel, lead_time, deposit_type,
                      market_segment, adr, features_json, prediction, cancel_prob):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO predictions
               (created_at, hotel, lead_time, deposit_type, market_segment,
                adr, features_json, prediction, cancel_prob)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (created_at, hotel, lead_time, deposit_type, market_segment,
             adr, features_json, prediction, cancel_prob),
        )
        conn.commit()


def fetch_latest(limit=30):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """SELECT id, created_at, hotel, lead_time, deposit_type,
                      market_segment, adr, prediction, cancel_prob
               FROM predictions
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        return cur.fetchall()


def fetch_stats():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """SELECT
                COUNT(*)                                          AS total,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END)  AS total_canceled,
                ROUND(AVG(cancel_prob) * 100, 1)                  AS avg_prob,
                ROUND(AVG(adr), 2)                                AS avg_adr
               FROM predictions"""
        )
        return cur.fetchone()