"""Idempotent schema reconciliation for the interactions log.

Suited to a PoC: the interactions table is frozen by the experiment design,
so a single ``initialize_schema`` call on startup is simpler than a full
Alembic migration tree. DDL uses ``psycopg2.sql`` so identifiers are safely
quoted even though the table name is a trusted constant.
"""
from __future__ import annotations

import logging

from psycopg2 import sql

from .pool import PostgresPool

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {tbl} (
    session_id TEXT,
    id SERIAL,
    participant_id TEXT,
    treatment TEXT,
    user_query TEXT,
    assistant_response TEXT,
    intermediate_steps TEXT,
    simplified_intermediate_steps TEXT,
    user_query_sent_time TIMESTAMP,
    response_displayed_time TIMESTAMP,
    explanation_button_displayed_time TIMESTAMP,
    explanation_clicked_time TIMESTAMP,
    explanation_clicked BOOLEAN DEFAULT FALSE,
    explanation_displayed_time TIMESTAMP,
    confidence_percent REAL,
    o_score REAL,
    s_score REAL,
    confidence_view TEXT,
    PRIMARY KEY (session_id, id)
)
"""

# Columns added after the table's original creation. ADD COLUMN IF NOT EXISTS
# keeps this safe for databases that predate the confidence feature.
_CONFIDENCE_COLUMNS = (
    ("confidence_percent", "REAL"),
    ("o_score", "REAL"),
    ("s_score", "REAL"),
    ("confidence_view", "TEXT"),
)


def initialize_schema(pool: PostgresPool, table_name: str) -> None:
    """Create the interactions table if missing, then add any new columns."""
    logger.info("Initialising schema for table %s", table_name)
    with pool.cursor() as cur:
        cur.execute(sql.SQL(_CREATE_TABLE_SQL).format(tbl=sql.Identifier(table_name)))
        for column, col_type in _CONFIDENCE_COLUMNS:
            cur.execute(
                sql.SQL("ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} ").format(
                    tbl=sql.Identifier(table_name),
                    col=sql.Identifier(column),
                )
                + sql.SQL(col_type)
            )
    logger.info("Schema for table %s is ready", table_name)
