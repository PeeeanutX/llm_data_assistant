"""Persistence layer for per-turn interaction records.

The legacy code passed fourteen positional arguments around. Here the shape
of a record is captured by a dataclass, and the repository exposes two
narrow methods (``save`` and ``mark_explanation_clicked``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from psycopg2 import sql

from .pool import PostgresPool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InteractionRecord:
    """A single user/agent exchange to be persisted."""

    session_id: str
    question_id: int
    participant_id: str | None
    treatment: str | None
    user_query: str
    assistant_response: str
    intermediate_steps: str
    simplified_intermediate_steps: str
    user_query_sent_time: datetime | None
    response_displayed_time: datetime | None
    explanation_button_displayed_time: datetime | None
    explanation_clicked_time: datetime | None
    explanation_clicked: bool | None
    explanation_displayed_time: datetime | None
    # Confidence (BSDetector) -- optional; None when not a scored data answer.
    confidence_percent: float | None = None
    o_score: float | None = None
    s_score: float | None = None
    confidence_view: str | None = None


# Column order used by ``INSERT``. Kept beside the SQL to keep the two in sync.
_INSERT_COLUMNS: tuple[str, ...] = (
    "session_id",
    "id",
    "participant_id",
    "treatment",
    "user_query",
    "assistant_response",
    "intermediate_steps",
    "simplified_intermediate_steps",
    "user_query_sent_time",
    "response_displayed_time",
    "explanation_button_displayed_time",
    "explanation_clicked_time",
    "explanation_clicked",
    "explanation_displayed_time",
    "confidence_percent",
    "o_score",
    "s_score",
    "confidence_view",
)


class InteractionRepository:
    """Repository for the interactions log table."""

    def __init__(self, pool: PostgresPool, table_name: str) -> None:
        self._pool = pool
        self._table = table_name
        self._insert_stmt = self._build_insert_stmt(table_name)
        self._click_stmt = self._build_click_stmt(table_name)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def save(self, record: InteractionRecord) -> None:
        """Insert ``record`` as a new row."""
        with self._pool.cursor() as cur:
            cur.execute(self._insert_stmt, self._record_to_row(record))
        logger.debug(
            "Saved interaction session_id=%s id=%d",
            record.session_id,
            record.question_id,
        )

    def mark_explanation_clicked(self, session_id: str, question_id: int) -> None:
        """Flip ``explanation_clicked`` to TRUE and record the click time."""
        with self._pool.cursor() as cur:
            cur.execute(self._click_stmt, (session_id, question_id))
        logger.debug(
            "Marked explanation_clicked for session_id=%s id=%d",
            session_id,
            question_id,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _record_to_row(record: InteractionRecord) -> tuple:
        return (
            record.session_id,
            record.question_id,
            record.participant_id,
            record.treatment,
            record.user_query,
            record.assistant_response,
            record.intermediate_steps,
            record.simplified_intermediate_steps,
            record.user_query_sent_time,
            record.response_displayed_time,
            record.explanation_button_displayed_time,
            record.explanation_clicked_time,
            record.explanation_clicked,
            record.explanation_displayed_time,
            record.confidence_percent,
            record.o_score,
            record.s_score,
            record.confidence_view,
        )

    @staticmethod
    def _build_insert_stmt(table_name: str) -> sql.Composed:
        columns = sql.SQL(", ").join(sql.Identifier(c) for c in _INSERT_COLUMNS)
        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in _INSERT_COLUMNS)
        return sql.SQL("INSERT INTO {tbl} ({cols}) VALUES ({vals})").format(
            tbl=sql.Identifier(table_name),
            cols=columns,
            vals=placeholders,
        )

    @staticmethod
    def _build_click_stmt(table_name: str) -> sql.Composed:
        return sql.SQL(
            "UPDATE {tbl} "
            "SET explanation_clicked = TRUE, "
            "    explanation_clicked_time = CURRENT_TIMESTAMP, "
            "    explanation_displayed_time = CURRENT_TIMESTAMP "
            "WHERE session_id = %s AND id = %s"
        ).format(tbl=sql.Identifier(table_name))
