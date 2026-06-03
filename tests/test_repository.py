"""Integration tests for :class:`app.db.repository.InteractionRepository`.

Marked ``db``: these require an ephemeral PostgreSQL instance provided by
``pytest-postgresql`` and are skipped automatically when the server binaries
are unavailable (see ``conftest.py``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from app.db.repository import InteractionRecord

pytestmark = pytest.mark.db


def _record(**overrides) -> InteractionRecord:
    base: dict[str, Any] = {
        "session_id": "sess-1",
        "question_id": 1,
        "participant_id": "p042",
        "treatment": "2",
        "user_query": "What is the highest sale?",
        "assistant_response": "22,638.48 dollars",
        "intermediate_steps": "**Step 1** ...",
        "simplified_intermediate_steps": "I looked at the tables ...",
        "user_query_sent_time": datetime(2026, 5, 28, 3, 1, 21),
        "response_displayed_time": datetime(2026, 5, 28, 3, 1, 43),
        "explanation_button_displayed_time": None,
        "explanation_clicked_time": None,
        "explanation_clicked": None,
        "explanation_displayed_time": None,
    }
    base.update(overrides)
    return InteractionRecord(**base)


def _fetch_one(pg_dsn: str, session_id: str, question_id: int) -> dict:
    import psycopg2

    conn = psycopg2.connect(pg_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT participant_id, treatment, user_query, assistant_response,
                       explanation_clicked, explanation_clicked_time
                FROM interactions_experiment
                WHERE session_id = %s AND id = %s
                """,
                (session_id, question_id),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None, "expected the row to exist"
    return {
        "participant_id": row[0],
        "treatment": row[1],
        "user_query": row[2],
        "assistant_response": row[3],
        "explanation_clicked": row[4],
        "explanation_clicked_time": row[5],
    }


def test_save_persists_all_core_fields(repository, pg_dsn) -> None:
    repository.save(_record())
    row = _fetch_one(pg_dsn, "sess-1", 1)
    assert row["participant_id"] == "p042"
    assert row["treatment"] == "2"
    assert row["user_query"] == "What is the highest sale?"
    assert row["assistant_response"] == "22,638.48 dollars"


def test_mark_explanation_clicked_sets_flag_and_timestamp(repository, pg_dsn) -> None:
    repository.save(_record())
    repository.mark_explanation_clicked("sess-1", 1)
    row = _fetch_one(pg_dsn, "sess-1", 1)
    assert row["explanation_clicked"] is True
    assert row["explanation_clicked_time"] is not None


def test_save_supports_multiple_questions_per_session(repository, pg_dsn) -> None:
    repository.save(_record(question_id=1))
    repository.save(_record(question_id=2, user_query="second question"))
    assert _fetch_one(pg_dsn, "sess-1", 2)["user_query"] == "second question"
