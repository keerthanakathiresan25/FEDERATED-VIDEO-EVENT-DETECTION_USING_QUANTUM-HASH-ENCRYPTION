# web_ui/database.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "results.db"

def init_db():
    """Create database and detections table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            event TEXT,
            confidence REAL,
            duration TEXT,
            key TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_result(filename, result):
    """Insert a detection record."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO detections (filename, event, confidence, duration, key)
        VALUES (?, ?, ?, ?, ?)
  """, (filename, result["event"], result["confidence"], result["duration"], result["key"]))
    conn.commit()
    conn.close()

def get_all_results():
    """Fetch all saved detections."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows
