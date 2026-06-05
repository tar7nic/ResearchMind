import sqlite3
import json
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "researchmind.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates the queries table if it doesn't exist.
    Call this once at app startup.
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query       TEXT NOT NULL,
                report      TEXT NOT NULL,
                scorecard   TEXT NOT NULL,
                timestamp   TEXT NOT NULL
            )
        """)
        conn.commit()


def save_research_record(query: str, report: str, scorecard: dict):
    """
    Persists a completed research record to SQLite.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO queries (query, report, scorecard, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (query, report, json.dumps(scorecard), datetime.utcnow().isoformat()),
        )
        conn.commit()


def get_all_records(limit: int = 50) -> list[dict]:
    """
    Retrieves past research records, most recent first.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_record_by_id(record_id: int) -> dict | None:
    """
    Retrieves a single research record by ID.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM queries WHERE id = ?",
            (record_id,),
        ).fetchone()

    return _row_to_dict(row) if row else None


def delete_record(record_id: int):
    """
    Deletes a research record by ID.
    """
    with get_connection() as conn:
        conn.execute("DELETE FROM queries WHERE id = ?", (record_id,))
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["scorecard"] = json.loads(d["scorecard"])
    return d