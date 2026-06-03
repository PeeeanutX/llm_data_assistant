"""Self-Reflection Certainty (the S term of BSDetector), grounded in evidence.

In this app the answer comes from a SQL agent, so asking the model whether the
answer is "verifiable by reliable sources" fails — a fact about a private
database cannot be corroborated from world knowledge, and S collapses to a
floor. Instead we give the model the SQL it ran and the rows it returned, and
ask whether the answer *faithfully reflects that result*, treating the database
as the source of truth. That is a question the model can actually answer.

Round 0 is a neutral check; later rounds use a more critical reviewer framing.
Scores are averaged across rounds and scaled to [0, 1].

The LLM call is injected as ``complete_fn`` so this module has no dependency on
any particular client and is unit-testable in isolation.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, List

logger = logging.getLogger(__name__)

# A function that takes a prompt and returns the model's text response.
CompleteFn = Callable[[str], str]

_ANCHORS = (
    "Anchors:\n"
    "- 0-10: the SQL does not answer the question, or the answer clearly misreads the result\n"
    "- 11-30: likely wrong query for the question, or a misread result\n"
    "- 31-45: leaning wrong / substantial doubt about the interpretation\n"
    "- 46-55: unsure / could go either way\n"
    "- 56-70: probably faithful, but some interpretation risk\n"
    "- 71-85: faithful interpretation, only minor wording/rounding risk\n"
    "- 86-94: the SQL fits the question and the answer reports the rows well\n"
    "- 95-100: the SQL clearly answers the question AND the answer faithfully reports the returned rows\n\n"
)


def build_self_reflection_prompt(
    question: str,
    proposed_answer: str,
    round_idx: int,
    evidence: str = "",
) -> str:
    """Build the (evidence-grounded) self-reflection prompt for a given round."""
    if round_idx == 0:
        lead = (
            "You are checking whether a proposed answer correctly and faithfully "
            "reflects the result of a database query.\n"
        )
    else:
        lead = (
            "A critical reviewer scrutinises whether the SQL truly answers the "
            "question and whether the proposed answer faithfully reports the "
            "returned rows. If the query is wrong for the question, or the answer "
            "misreads, mis-rounds, or mislabels the result, reduce the "
            "probability accordingly.\n"
        )

    evidence_block = evidence.strip() or "(no SQL query was recorded for this answer)"

    return (
        f"{lead}"
        "Treat the database as the source of truth: do NOT try to verify the data "
        "against outside knowledge. Judge ONLY whether the answer correctly "
        "interprets the query and its results.\n"
        "Estimate the probability (0-100) that the proposed answer correctly "
        "reflects the query result, considering whether the SQL answers the "
        "question (right columns, aggregation, filters) and whether the answer "
        "reports the returned rows correctly (value, units, rounding).\n"
        f"{_ANCHORS}"
        f"User question:\n{question}\n\n"
        f"SQL query and results:\n{evidence_block}\n\n"
        f"Proposed answer:\n{proposed_answer}\n\n"
        "Output ONLY a single integer 0-100."
    )


def parse_self_reflection_choice(raw_output: str) -> int:
    """Parse the model's reply into an integer percent in [0, 100]."""
    if not raw_output:
        return 0
    text = raw_output.strip()
    if not text:
        return 0

    exact = re.fullmatch(r"\s*(\d{1,3})\s*%?\s*", text)
    if exact:
        return max(0, min(100, int(exact.group(1))))

    anywhere = re.search(r"\b(\d{1,3})\b", text)
    if not anywhere:
        return 0
    return max(0, min(100, int(anywhere.group(1))))


def self_reflection_certainty(
    question: str,
    proposed_answer: str,
    complete_fn: CompleteFn,
    *,
    rounds: int = 3,
    evidence: str = "",
) -> float:
    """Return the Self-Reflection Certainty S in [0, 1].

    ``evidence`` is the SQL-query-and-results context; when provided, the model
    judges faithfulness to that result rather than external verifiability.
    """
    rounds = max(1, rounds)
    percents: List[float] = []
    for i in range(rounds):
        prompt = build_self_reflection_prompt(question, proposed_answer, i, evidence)
        try:
            raw = complete_fn(prompt)
        except Exception:
            logger.exception("Self-reflection round %d failed; scoring 0", i)
            raw = ""
        percents.append(float(parse_self_reflection_choice(raw)))

    return (sum(percents) / len(percents)) / 100.0
