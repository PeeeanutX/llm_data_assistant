"""Streamlit session-state helpers.

The interaction-log primary key is ``(session_id, question_counter)``; both
counters live here. Query-parameter parsing is also centralised so that bad
inputs (e.g. ``treatment=foo``) cannot crash the app.
"""

from __future__ import annotations

import logging
import uuid

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

logger = logging.getLogger(__name__)

LANGCHAIN_HISTORY_KEY: str = "langchain_messages"
"""Streamlit session-state key used by :class:`StreamlitChatMessageHistory`."""

# UI session-state keys
_SESSION_ID_KEY = "session_id"
_QUESTION_COUNTER_KEY = "question_counter"
_MESSAGES_KEY = "messages"

# URL query parameter names
_PARTICIPANT_PARAM = "participant"
_TREATMENT_PARAM = "treatment"


def ensure_session_state(welcome_message: str) -> None:
    """Initialise per-session state on first run; no-op on subsequent reruns."""
    if _SESSION_ID_KEY not in st.session_state:
        st.session_state[_SESSION_ID_KEY] = str(uuid.uuid4())
        logger.info("Started new session %s", st.session_state[_SESSION_ID_KEY])

    if _QUESTION_COUNTER_KEY not in st.session_state:
        st.session_state[_QUESTION_COUNTER_KEY] = 0

    if _MESSAGES_KEY not in st.session_state:
        st.session_state[_MESSAGES_KEY] = [{"role": "assistant", "content": welcome_message}]

    msgs = get_message_history()
    if len(msgs.messages) == 0:
        msgs.add_ai_message(welcome_message)


def reset_session_state(welcome_message: str) -> None:
    """Clear the chat history without invalidating the session ID."""
    get_message_history().clear()
    st.session_state[_QUESTION_COUNTER_KEY] = 0
    st.session_state[_MESSAGES_KEY] = [{"role": "assistant", "content": welcome_message}]
    logger.info("Reset chat history for session %s", st.session_state.get(_SESSION_ID_KEY))


def get_message_history() -> StreamlitChatMessageHistory:
    """Return the LangChain-compatible message history backed by session state."""
    return StreamlitChatMessageHistory(key=LANGCHAIN_HISTORY_KEY)


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #
def get_session_id() -> str:
    return st.session_state[_SESSION_ID_KEY]


def get_question_counter() -> int:
    return st.session_state[_QUESTION_COUNTER_KEY]


def increment_question_counter() -> int:
    """Increment and return the new value."""
    st.session_state[_QUESTION_COUNTER_KEY] += 1
    return st.session_state[_QUESTION_COUNTER_KEY]


def get_participant_id() -> str | None:
    """Return the participant ID from the URL, or ``None`` if absent."""
    return st.query_params.get(_PARTICIPANT_PARAM)


def get_treatment() -> int | None:
    """Return the treatment integer from the URL, or ``None`` if absent/invalid.

    Note: the legacy script compared the raw string value of this query
    parameter to the integer ``2``, which never matched. Parsing here makes
    the comparison correct everywhere else.
    """
    raw = st.query_params.get(_TREATMENT_PARAM)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Could not parse treatment query param: %r", raw)
        return None
