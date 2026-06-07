"""Streamlit chat rendering and per-turn orchestration."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from app.agent.explainer import StepExplainer
from app.agent.tokens import TokenBudget
from app.agent.steps import extract_sql_evidence, prettify_intermediate_steps
from app.db.repository import InteractionRecord, InteractionRepository
from app.uncertainty.scorer import ConfidenceResult, ConfidenceScorer
from app.prompts import NO_EXPLANATION_MESSAGE
from app.ui.session import (
    get_message_history,
    get_session_id,
    increment_question_counter,
)

logger = logging.getLogger(__name__)

# The treatment value in which the explanation is hidden behind a button.
TREATMENT_GATED_EXPLANATION: int = 2

# DEV OVERRIDE: when True, ignore the URL ``treatment`` parameter entirely and
# always gate the explanation behind the button. The button is only shown when
# certainty is below 80%; at 80%+ no button is shown at all. It never
# auto-opens -- the user must click to reveal the explanation. Set back to
# False to restore the A/B treatment behaviour.
_DEV_FORCE_GATED: bool = True

# Certainty below this percentage offers a (collapsed) explanation button;
# at or above it, no button is shown. The certainty score itself is never
# surfaced in the UI -- it is only computed, logged, and persisted (backend).
CERTAINTY_BUTTON_THRESHOLD: float = 85.0


def _is_gated(treatment: int | None) -> bool:
    """Whether the explanation should be gated behind the button."""
    if _DEV_FORCE_GATED:
        return True
    return treatment == TREATMENT_GATED_EXPLANATION


def _should_offer_button(confidence_percent: float | None) -> bool:
    """Whether to show the explanation button for a gated answer.

    Only low-certainty answers (below ``CERTAINTY_BUTTON_THRESHOLD``) get a
    button; high-certainty answers -- and answers without a score, e.g.
    greetings -- show nothing.
    """
    return (
        confidence_percent is not None
        and confidence_percent < CERTAINTY_BUTTON_THRESHOLD
    )


# --------------------------------------------------------------------------- #
# History rendering
# --------------------------------------------------------------------------- #
def render_chat_history(
    treatment: int | None,
    repository: InteractionRepository,
) -> None:
    """Replay all previously-stored messages."""
    gated = _is_gated(treatment)
    for message in st.session_state.get("messages", []):
        if "expander" not in message:
            st.chat_message(message["role"]).write(message["content"])
            continue
        _render_explanation_message(message, gated=gated, repository=repository)


def _render_explanation_message(
    message: dict,
    gated: bool,
    repository: InteractionRepository,
) -> None:
    interaction_id: int = message["interaction_id"]
    label: str = message.get("expander", "See explanation")
    content: str = message["content"]
    revealed = st.session_state.get(_expander_state_key(interaction_id), False)
    confidence_percent = message.get("confidence_percent")

    # Non-gated control, or the user already revealed it -> show the explanation.
    if not gated or revealed:
        with st.expander(label, expanded=True):
            st.write(content)
        return

    # Gated: only low-certainty answers offer the (collapsed) button. It stays
    # user-invoked; high-certainty answers show nothing.
    if _should_offer_button(confidence_percent):
        st.button(
            "See explanation",
            key=f"button_{interaction_id}",
            on_click=_handle_explanation_click,
            args=(repository, interaction_id),
        )


# --------------------------------------------------------------------------- #
# New-turn orchestration
# --------------------------------------------------------------------------- #
def render_new_turn(
    user_query: str,
    treatment: int | None,
    participant_id: str | None,
    agent: Any,
    explainer: StepExplainer,
    repository: InteractionRepository,
    token_budget: TokenBudget,
    scorer: ConfidenceScorer,
    confidence_view: str,
) -> None:
    """Drive a single user/agent exchange end-to-end."""
    user_query_sent_time = pd.Timestamp.now()
    question_id = increment_question_counter()

    _append_and_render_user_message(user_query)
    msgs = get_message_history()
    msgs.add_user_message(user_query)
    truncated_history = token_budget.fit(msgs.messages, pending_text=user_query)

    response = _invoke_agent(agent, user_query, truncated_history)
    response_content: str = response["output"]
    response_displayed_time = pd.Timestamp.now()

    _append_and_render_assistant_message(response_content)
    msgs.add_ai_message(response_content)

    inter_steps = response.get("intermediate_steps") or []

    # Confidence is only meaningful for answers grounded in a database query;
    # skip greetings / "not related to the database" responses.
    confidence: ConfidenceResult | None = None
    if inter_steps:
        evidence = extract_sql_evidence(inter_steps)
        confidence = _compute_confidence(
            scorer, user_query, response_content, evidence
        )
        # The certainty score is intentionally never rendered in the UI; it is
        # only logged and persisted (backend). It just gates the button below.
    explanation_content, prettified_steps = _build_explanation(
        inter_steps=inter_steps, explainer=explainer
    )

    explanation_button_displayed_time: pd.Timestamp | None = None
    explanation_displayed_time: pd.Timestamp | None = None

    st.session_state.messages.append(
        {
            "role": "assistant",
            "expander": "See explanation",
            "content": explanation_content,
            "interaction_id": question_id,
            "confidence_percent": confidence.percent if confidence else None,
        }
    )

    # Gated: only low-certainty answers (<80%) get a "See explanation" button,
    # and it stays collapsed until the user clicks it (user-invoked). High-
    # certainty answers (>=80%) show no button at all. The non-gated control
    # always shows the explanation.
    if _is_gated(treatment):
        if _should_offer_button(confidence.percent if confidence else None):
            explanation_button_displayed_time = pd.Timestamp.now()
            _render_gated_explanation(
                question_id=question_id,
                content=explanation_content,
                repository=repository,
            )
    else:
        explanation_displayed_time = pd.Timestamp.now()
        with st.expander("See explanation", expanded=True):
            st.write(explanation_content)

    _persist_interaction(
        repository=repository,
        question_id=question_id,
        participant_id=participant_id,
        treatment=treatment,
        user_query=user_query,
        assistant_response=response_content,
        intermediate_steps=prettified_steps,
        simplified_intermediate_steps=explanation_content,
        user_query_sent_time=user_query_sent_time,
        response_displayed_time=response_displayed_time,
        explanation_button_displayed_time=explanation_button_displayed_time,
        explanation_displayed_time=explanation_displayed_time,
        confidence=confidence,
        confidence_view=confidence_view,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _append_and_render_user_message(user_query: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)


def _append_and_render_assistant_message(content: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": content})
    st.chat_message("assistant").write(content)


def _compute_confidence(
    scorer: ConfidenceScorer, question: str, answer: str, evidence: str
) -> ConfidenceResult | None:
    with st.spinner("Estimating confidence..."):
        try:
            return scorer.score(question, answer, evidence)
        except Exception:
            logger.exception("Confidence scoring failed")
            return None



def _invoke_agent(agent: Any, user_query: str, history: list) -> dict:
    with st.spinner("Analyzing the database..."):
        try:
            return agent.invoke({"input": user_query, "history": history})
        except Exception as exc:
            logger.exception("Agent invocation failed")
            st.error(f"Error: {exc}")
            return {"output": str(exc), "intermediate_steps": []}


def _build_explanation(
    inter_steps: list,
    explainer: StepExplainer,
) -> tuple[str, str]:
    """Return ``(explanation_content, prettified_steps)``."""
    prettified = prettify_intermediate_steps(inter_steps)
    if not inter_steps:
        return NO_EXPLANATION_MESSAGE, prettified

    with st.spinner("Generating explanation..."):
        try:
            return explainer.explain(prettified), prettified
        except Exception:
            logger.exception("Explanation generation failed")
            return NO_EXPLANATION_MESSAGE, prettified


def _render_gated_explanation(
    question_id: int,
    content: str,
    repository: InteractionRepository,
) -> None:
    clicked = st.button(
        "See explanation",
        key=f"button_{question_id}",
        on_click=_handle_explanation_click,
        args=(repository, question_id),
    )
    if clicked or st.session_state.get(_expander_state_key(question_id), False):
        with st.expander("See explanation", expanded=True):
            st.write(content)


def _handle_explanation_click(
    repository: InteractionRepository,
    interaction_id: int,
) -> None:
    """Streamlit ``on_click`` callback for the gated explanation button."""
    try:
        repository.mark_explanation_clicked(get_session_id(), interaction_id)
    except Exception:
        logger.exception("Failed to record explanation click")
    st.session_state[_expander_state_key(interaction_id)] = True


def _expander_state_key(interaction_id: int) -> str:
    return f"expander_{interaction_id}"


def _to_pydatetime(ts: pd.Timestamp | None) -> datetime | None:
    return ts.to_pydatetime() if ts is not None else None


def _persist_interaction(
    repository: InteractionRepository,
    question_id: int,
    participant_id: str | None,
    treatment: int | None,
    user_query: str,
    assistant_response: str,
    intermediate_steps: str,
    simplified_intermediate_steps: str,
    user_query_sent_time: pd.Timestamp,
    response_displayed_time: pd.Timestamp,
    explanation_button_displayed_time: pd.Timestamp | None,
    explanation_displayed_time: pd.Timestamp | None,
    confidence: ConfidenceResult | None = None,
    confidence_view: str | None = None,
) -> None:
    record = InteractionRecord(
        session_id=get_session_id(),
        question_id=question_id,
        participant_id=participant_id,
        treatment=str(treatment) if treatment is not None else None,
        user_query=user_query,
        assistant_response=assistant_response,
        intermediate_steps=intermediate_steps,
        simplified_intermediate_steps=simplified_intermediate_steps,
        user_query_sent_time=_to_pydatetime(user_query_sent_time),
        response_displayed_time=_to_pydatetime(response_displayed_time),
        explanation_button_displayed_time=_to_pydatetime(explanation_button_displayed_time),
        explanation_clicked_time=None,
        explanation_clicked=None,
        explanation_displayed_time=_to_pydatetime(explanation_displayed_time),
        confidence_percent=confidence.percent if confidence else None,
        o_score=confidence.observed if confidence else None,
        s_score=confidence.self_reflection if confidence else None,
        confidence_view=confidence_view if confidence else None,
    )
    try:
        repository.save(record)
    except Exception:
        logger.exception("Failed to persist interaction record")
        st.warning("This interaction could not be saved to the database.")
