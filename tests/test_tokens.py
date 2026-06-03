"""Unit tests for :mod:`app.agent.tokens`."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.tokens import TokenBudget


@dataclass
class _Msg:
    """Minimal stand-in for a chat message exposing ``.content``."""

    content: str


@pytest.fixture
def budget() -> TokenBudget:
    return TokenBudget(model_name="gpt-4o-mini", max_total_tokens=100, reserved_tokens=10)


def test_count_is_additive(budget: TokenBudget) -> None:
    one = budget.count([_Msg("hello world")])
    two = budget.count([_Msg("hello world"), _Msg("hello world")])
    assert one > 0
    assert two == 2 * one


def test_count_empty_is_zero(budget: TokenBudget) -> None:
    assert budget.count([]) == 0


def test_unknown_model_falls_back_without_error() -> None:
    tb = TokenBudget(model_name="some-future-model-x", max_total_tokens=100, reserved_tokens=10)
    assert tb.count([_Msg("hello")]) > 0


def test_fit_keeps_everything_under_budget(budget: TokenBudget) -> None:
    messages = [_Msg("a"), _Msg("b"), _Msg("c")]
    assert budget.fit(messages, pending_text="") == messages


def test_fit_drops_oldest_when_over_budget() -> None:
    tb = TokenBudget(model_name="gpt-4o-mini", max_total_tokens=12, reserved_tokens=0)
    messages = [_Msg("word " * 5), _Msg("word " * 5), _Msg("recent")]
    kept = tb.fit(messages, pending_text="")
    assert kept
    assert kept[-1].content == "recent"
    assert len(kept) < len(messages)


def test_fit_preserves_chronological_order() -> None:
    tb = TokenBudget(model_name="gpt-4o-mini", max_total_tokens=1000, reserved_tokens=0)
    messages = [_Msg("first"), _Msg("second"), _Msg("third")]
    kept = tb.fit(messages, pending_text="")
    assert [m.content for m in kept] == ["first", "second", "third"]


def test_fit_returns_empty_when_pending_exhausts_budget() -> None:
    tb = TokenBudget(model_name="gpt-4o-mini", max_total_tokens=5, reserved_tokens=5)
    messages = [_Msg("anything")]
    assert tb.fit(messages, pending_text="a very long pending message indeed") == []
