"""SQLite helpers for the Football Intelligence Platform.

The module keeps a single place for connection handling, schema bootstrap and
small query utilities so that the rest of the code never touches raw sqlite3.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_DB_PATH = BASE_DIR / "football.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the database file and apply the schema (idempotent)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = get_connection(path)
    try:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()


def query_all(
    connection: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """Run a query and return every row as a plain dict."""
    rows = connection.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def query_one(
    connection: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
) -> dict[str, Any] | None:
    """Run a query and return the first row as a plain dict (or None)."""
    row = connection.execute(sql, params).fetchone()
    return dict(row) if row is not None else None


def scalar(
    connection: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
    default: Any = 0,
) -> Any:
    """Run a query and return the first column of the first row."""
    row = connection.execute(sql, params).fetchone()
    if row is None or row[0] is None:
        return default
    return row[0]


def table_count(connection: sqlite3.Connection, table: str) -> int:
    """Return the number of rows in a table (0 when the table is empty/missing)."""
    return int(scalar(connection, f"SELECT COUNT(*) FROM {table}", default=0))


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield chunks of `size` items - handy for batched SQL inserts."""
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
