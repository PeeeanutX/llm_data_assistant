"""Render the SQL agent's intermediate steps as human-readable Markdown.

The agent emits a list of ``(AgentAction, observation)`` tuples; this module
turns them into the prose paragraphs that the explainer LLM then rewrites.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from app.prompts import ACTION_DESCRIPTIONS

logger = logging.getLogger(__name__)


def prettify_intermediate_steps(steps: Iterable[tuple[Any, Any]]) -> str:
    """Convert agent intermediate steps to a Markdown-formatted string."""
    step_list = list(steps)
    total_steps = len(step_list)
    paragraphs: list[str] = []

    for index, (action, observation) in enumerate(step_list, start=1):
        paragraphs.append(
            _format_step(
                index=index,
                total=total_steps,
                tool=getattr(action, "tool", ""),
                tool_input=getattr(action, "tool_input", None),
                observation=observation,
            )
        )

    return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _format_step(
    index: int,
    total: int,
    tool: str,
    tool_input: Any,
    observation: Any,
) -> str:
    """Format a single step paragraph."""
    template = ACTION_DESCRIPTIONS.get(tool)
    if template is None:
        description = _wrap_step_header(index, total, "Performed an action.")
        return f"{description}\n\n**The result of this action returns:** {observation}"

    table_name = _extract_table_name(tool, tool_input)
    description_body = template.format(table_name) if "{}" in template else template
    description = _wrap_step_header(index, total, description_body)

    query_line = ""
    if tool == "sql_db_query":
        query_text = _extract_query_text(tool_input)
        if query_text:
            query_line = f"**Now the following SQL query has been run:** `{query_text}`"

    return f"{description}\n\n{query_line}\n\n**The result of this action returns:** {observation}"


def _wrap_step_header(index: int, total: int, body: str) -> str:
    if total > 1:
        return f"**Step {index}:** \n\n **{body}**"
    return f"**{body}**"


def _extract_table_name(tool: str, tool_input: Any) -> str:
    """Best-effort extraction of the target table name for display purposes."""
    if not isinstance(tool_input, dict):
        return ""

    if tool == "sql_db_schema":
        return str(tool_input.get("table_names", ""))

    if tool == "sql_db_query":
        query = tool_input.get("query", "")
        return _table_from_query(query)

    return ""


def _table_from_query(query: str) -> str:
    """Pull the first token after ``FROM`` from a SQL query.

    Note: this is intentionally permissive; failures fall back to an empty
    string so the UI still renders.
    """
    if not query:
        return ""
    upper = query.upper()
    marker = " FROM "
    idx = upper.find(marker)
    if idx == -1:
        return ""
    tail = query[idx + len(marker) :].strip()
    if not tail:
        return ""
    return tail.split()[0]


def _extract_query_text(tool_input: Any) -> str:
    if isinstance(tool_input, dict):
        return str(tool_input.get("query", ""))
    return ""


def extract_sql_evidence(steps: Iterable[tuple[Any, Any]]) -> str:
    """Pull executed SQL queries and their results, for self-reflection context.

    Returns a compact, human-readable block listing each ``sql_db_query`` and
    the rows it returned. Schema-inspection steps (list_tables, schema) are
    skipped. Empty string if no query was executed.
    """
    parts: list[str] = []
    n = 0
    for action, observation in steps:
        if getattr(action, "tool", "") != "sql_db_query":
            continue
        n += 1
        query = _extract_query_text(getattr(action, "tool_input", None))
        parts.append(f"SQL query {n}:\n{query}\nResult {n}:\n{observation}")
    return "\n\n".join(parts)
