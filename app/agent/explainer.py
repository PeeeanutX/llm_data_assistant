"""Second LLM pass that rewrites intermediate steps for non-technical users."""

from __future__ import annotations

import logging
from typing import Any

from openai import BadRequestError, OpenAI

from app.prompts import EXPLAINER_SYSTEM_PROMPT, build_explainer_user_prompt

logger = logging.getLogger(__name__)

# OpenAI deprecated ``max_tokens`` in favour of ``max_completion_tokens`` for
# newer model families (o-series, GPT-5 and later). Older models still expect
# ``max_tokens``. We default to the newer name and fall back automatically.
_NEW_TOKEN_PARAM = "max_completion_tokens"
_LEGACY_TOKEN_PARAM = "max_tokens"


class StepExplainer:
    """Wraps an OpenAI chat-completion call dedicated to explaining steps.

    The class probes for the correct token-budget parameter on the first call
    and remembers the result for the lifetime of the instance.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 1_000,
        temperature: float = 0.1,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Optimistic default; swapped on the first BadRequestError if needed.
        self._token_param: str = _NEW_TOKEN_PARAM

    def explain(self, prettified_intermediate_steps: str) -> str:
        """Return a natural-language explanation of the intermediate steps."""
        messages = [
            {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_explainer_user_prompt(prettified_intermediate_steps),
            },
        ]
        response = self._create_with_fallback(messages)
        content = response.choices[0].message.content or ""
        return content.strip()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _create_with_fallback(self, messages: list[dict]) -> Any:
        """Call ``chat.completions.create``, swapping the token parameter if
        the model rejects the one we attempted first."""
        try:
            return self._call(messages, token_param=self._token_param)
        except BadRequestError as exc:
            other = self._alternate_token_param()
            if other is None or not _looks_like_token_param_error(exc):
                raise
            logger.info(
                "Model %s rejected %r; retrying with %r",
                self._model,
                self._token_param,
                other,
            )
            response = self._call(messages, token_param=other)
            # Remember the working parameter so we don't retry every turn.
            self._token_param = other
            return response

    def _call(self, messages: list[dict], token_param: str) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            token_param: self._max_tokens,
        }
        return self._client.chat.completions.create(**kwargs)

    def _alternate_token_param(self) -> str | None:
        if self._token_param == _NEW_TOKEN_PARAM:
            return _LEGACY_TOKEN_PARAM
        if self._token_param == _LEGACY_TOKEN_PARAM:
            return _NEW_TOKEN_PARAM
        return None


def _looks_like_token_param_error(exc: BadRequestError) -> bool:
    """Best-effort detection of "wrong token parameter" errors from OpenAI."""
    message = str(exc).lower()
    return (
        "max_tokens" in message
        or "max_completion_tokens" in message
        or "unsupported_parameter" in message
    )
