"""SQLite helpers for the platform.

Table schemas and data access logic will be added in later stages.
"""

import sqlite3
from pathlib import Path


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str | Path) -> None:
    """Ensure the database directory and file exist. No schema is created yet."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = get_connection(db_path)
    try:
        # No tables are defined at this stage; the connection validates the file.
        pass
    finally:
        connection.close()
