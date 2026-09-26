from __future__ import annotations

import sqlite3
from pathlib import Path


def list_tables(sqlite_db: str | Path) -> list[str]:
    with sqlite3.connect(sqlite_db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def describe_table(sqlite_db: str | Path, table_name: str) -> list[dict[str, str | int | None]]:
    with sqlite3.connect(sqlite_db) as conn:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": row[3],
            "default_value": row[4],
            "pk": row[5],
        }
        for row in rows
    ]

