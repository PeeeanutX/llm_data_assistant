"""Load an Excel workbook into an on-disk SQLite database.

Each worksheet becomes a table. The database is then re-opened in read-only
mode and wrapped in a LangChain ``SQLDatabase`` so the agent can introspect
it without being able to mutate it.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


def load_excel_to_sqlite(excel_path: Path, sqlite_path: Path) -> SQLDatabase:
    """Materialise ``excel_path`` into a SQLite file and return a LangChain DB.

    Each sheet is loaded into a table named after the sheet, replacing the
    table if it already exists. The returned ``SQLDatabase`` uses a read-only
    connection so the agent cannot mutate the schema or data.

    Raises:
        FileNotFoundError: If ``excel_path`` does not exist.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel workbook not found: {excel_path}")

    logger.info("Loading %s into SQLite at %s", excel_path, sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(sqlite_path) as conn, pd.ExcelFile(excel_path) as workbook:
        for sheet_name in workbook.sheet_names:
            frame = workbook.parse(sheet_name)
            frame.to_sql(sheet_name, conn, index=False, if_exists="replace")
            logger.debug("Loaded sheet '%s' (%d rows) into SQLite", sheet_name, len(frame))

    return _build_read_only_database(sqlite_path)


def _build_read_only_database(sqlite_path: Path) -> SQLDatabase:
    """Wrap ``sqlite_path`` in a read-only LangChain ``SQLDatabase``."""

    def _connect() -> sqlite3.Connection:
        # ``check_same_thread=False`` lets SQLAlchemy share the connection
        # across Streamlit's worker threads.
        return sqlite3.connect(
            f"file:{sqlite_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )

    engine = create_engine(f"sqlite:///{sqlite_path}", creator=_connect)
    return SQLDatabase(engine)
