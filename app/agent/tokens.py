"""Token counting and history-truncation for the chat memory."""
from __future__ import annotations

import logging
from typing import Iterable, List, Protocol

import tiktoken

logger = logging.getLogger(__name__)


class _HasContent(Protocol):
    """Structural type for any object exposing a ``content`` string."""

    content: str


def _get_encoding(model_name: str) -> tiktoken.Encoding:
    """Resolve a tiktoken encoding with a fallback for unknown models."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.debug(
            "No tiktoken encoding for %s; falling back to cl100k_base", model_name
        )
        return tiktoken.get_encoding("cl100k_base")


class TokenBudget:
    """Count tokens and trim chat history to fit a context window.

    Args:
        model_name: Name of the model whose tokenizer should be used.
        max_total_tokens: Hard upper bound on tokens the model can accept.
        reserved_tokens: Tokens reserved for the system prompt + response.
    """

    def __init__(
        self,
        model_name: str,
        max_total_tokens: int,
        reserved_tokens: int,
    ) -> None:
        self._encoding = _get_encoding(model_name)
        self._max_total_tokens = max_total_tokens
        self._reserved_tokens = reserved_tokens

    def count(self, messages: Iterable[_HasContent]) -> int:
        """Return the total token count across ``messages``."""
        total = 0
        for message in messages:
            content = message.content
            text = content if isinstance(content, str) else str(content)
            total += len(self._encoding.encode(text))
        return total

    def fit(
        self,
        messages: List[_HasContent],
        pending_text: str = "",
    ) -> List[_HasContent]:
        """Return the tail of ``messages`` that fits in the history budget."""
        pending_tokens = (
            len(self._encoding.encode(pending_text)) if pending_text else 0
        )
        history_budget = (
            self._max_total_tokens - self._reserved_tokens - pending_tokens
        )
        if history_budget <= 0:
            logger.warning("Pending text exceeds budget; sending empty history")
            return []

        kept: List[_HasContent] = []
        running_total = 0
        for message in reversed(messages):
            content = message.content
            text = content if isinstance(content, str) else str(content)
            message_tokens = len(self._encoding.encode(text))
            if running_total + message_tokens > history_budget:
                break
            kept.append(message)
            running_total += message_tokens

        kept.reverse()
        return kept
