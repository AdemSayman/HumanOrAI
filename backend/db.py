# db.py
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "history.sqlite3"

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        text_preview TEXT NOT NULL,
        text_len INTEGER NOT NULL,
        final_label TEXT NOT NULL,
        logreg_ai REAL,
        svm_ai REAL,
        nb_ai REAL
    )
    """)
    conn.commit()
    conn.close()

def insert_history(item: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO history (created_at, text_preview, text_len, final_label, logreg_ai, svm_ai, nb_ai)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        item["created_at"],
        item["text_preview"],
        item["text_len"],
        item["final_label"],
        item.get("logreg_ai"),
        item.get("svm_ai"),
        item.get("nb_ai"),
    ))
    conn.commit()
    conn.close()

def list_history(limit: int = 50):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT * FROM history
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def clear_history():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()
    conn.close()
