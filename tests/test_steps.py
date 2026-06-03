"""Unit tests for :mod:`app.agent.steps`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.agent.steps import _table_from_query, prettify_intermediate_steps


@dataclass
class _Action:
    """Stand-in for an agent action with ``.tool`` / ``.tool_input``."""

    tool: str
    tool_input: Any


def test_empty_steps_returns_empty_string() -> None:
    assert prettify_intermediate_steps([]) == ""


def test_list_tables_step_uses_its_description() -> None:
    steps = [(_Action("sql_db_list_tables", {}), "Orders, People")]
    out = prettify_intermediate_steps(steps)
    assert "list of available tables" in out
    assert "Orders, People" in out


def test_schema_step_includes_table_name() -> None:
    steps = [(_Action("sql_db_schema", {"table_names": "Orders"}), "CREATE TABLE ...")]
    out = prettify_intermediate_steps(steps)
    assert "Orders" in out
    assert "schema" in out.lower()


def test_query_step_includes_sql_and_table() -> None:
    query = 'SELECT "Sales" FROM "Orders" ORDER BY "Sales" DESC LIMIT 1;'
    steps = [(_Action("sql_db_query", {"query": query}), "[(22638.48,)]")]
    out = prettify_intermediate_steps(steps)
    assert query in out
    assert "Orders" in out
    assert "22638.48" in out


def test_unknown_tool_falls_back_to_generic_label() -> None:
    steps = [
        (_Action("sql_db_list_tables", {}), "Orders"),
        (_Action("sql_db_query_checker", {"query": "SELECT 1"}), "ok"),
    ]
    out = prettify_intermediate_steps(steps)
    assert "Performed an action" in out


def test_multi_step_output_is_numbered() -> None:
    steps = [
        (_Action("sql_db_list_tables", {}), "Orders"),
        (_Action("sql_db_schema", {"table_names": "Orders"}), "schema"),
    ]
    out = prettify_intermediate_steps(steps)
    assert "Step 1" in out
    assert "Step 2" in out


@pytest.mark.parametrize(
    "query, expected",
    [
        ('SELECT * FROM "Orders" LIMIT 1', '"Orders"'),
        ("select a from orders where x = 1", "orders"),  # lowercase keyword
        ("SELECT 1", ""),  # no FROM clause
        ("", ""),  # empty
        ('SELECT * FROM "Orders" o JOIN "People" p', '"Orders"'),
    ],
)
def test_table_from_query(query: str, expected: str) -> None:
    assert _table_from_query(query) == expected


def test_extract_sql_evidence_collects_queries_and_results() -> None:
    from app.agent.steps import extract_sql_evidence

    steps = [
        (_Action("sql_db_list_tables", {}), "Orders, People"),
        (_Action("sql_db_schema", {"table_names": "Orders"}), "CREATE TABLE ..."),
        (_Action("sql_db_query", {"query": 'SELECT MAX("Sales") FROM "Orders"'}), "[(22638.48,)]"),
    ]
    evidence = extract_sql_evidence(steps)
    assert 'SELECT MAX("Sales") FROM "Orders"' in evidence
    assert "22638.48" in evidence
    # Schema-inspection steps are excluded.
    assert "CREATE TABLE" not in evidence
    assert "sql_db_list_tables" not in evidence


def test_extract_sql_evidence_empty_when_no_query() -> None:
    from app.agent.steps import extract_sql_evidence

    steps = [(_Action("sql_db_list_tables", {}), "Orders, People")]
    assert extract_sql_evidence(steps) == ""
