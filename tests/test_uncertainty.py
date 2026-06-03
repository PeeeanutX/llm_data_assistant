"""Unit tests for the pure parts of the confidence (BSDetector) feature.

The NLI model and the LLM calls are not exercised here -- ``observed`` takes an
injected NLI scorer and ``self_reflection_certainty`` takes an injected
completion function, so both are tested with fakes.
"""
from __future__ import annotations

from typing import List

import pytest

from app.uncertainty.observed import observed_consistency
from app.uncertainty.reflection import (
    parse_self_reflection_choice,
    self_reflection_certainty,
)
from app.uncertainty.score import combine_confidence


# --------------------------------------------------------------------------- #
# score.combine_confidence
# --------------------------------------------------------------------------- #
def test_combine_is_convex_blend() -> None:
    assert combine_confidence(1.0, 0.0, beta=0.7) == pytest.approx(0.7)
    assert combine_confidence(0.0, 1.0, beta=0.7) == pytest.approx(0.3)
    assert combine_confidence(0.8, 0.4, beta=0.5) == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# reflection.parse_self_reflection_choice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("87", 87),
        ("  92 %", 92),
        ("I'd estimate around 75 out of 100", 75),
        ("150", 100),  # clamped
        ("", 0),
        ("no number here", 0),
    ],
)
def test_parse_self_reflection_choice(raw: str, expected: int) -> None:
    assert parse_self_reflection_choice(raw) == expected


def test_self_reflection_certainty_averages_rounds() -> None:
    replies = iter(["80", "60", "40"])
    s = self_reflection_certainty(
        "q", "a", lambda _prompt: next(replies), rounds=3
    )
    # mean(80, 60, 40) / 100 = 0.6
    assert s == pytest.approx(0.6)


def test_self_reflection_handles_complete_failure() -> None:
    def boom(_prompt: str) -> str:
        raise RuntimeError("api down")

    # A failing round scores 0 rather than crashing.
    s = self_reflection_certainty("q", "a", boom, rounds=2)
    assert s == 0.0


# --------------------------------------------------------------------------- #
# observed.observed_consistency  (with a fake NLI scorer)
# --------------------------------------------------------------------------- #
class _FakeNli:
    """Returns preset contradiction probabilities, ignoring inputs."""

    def __init__(self, probs: List[float]) -> None:
        self._probs = probs

    def contradiction_probs(self, premises: List[str], hypotheses: List[str]) -> List[float]:
        return list(self._probs[: len(premises)])


def test_observed_empty_samples_is_zero() -> None:
    assert observed_consistency("ref", [], _FakeNli([])) == 0.0


def test_observed_no_contradiction_scores_high() -> None:
    # All pairs have p(contradiction)=0 -> s_i=1; with alpha=0.8 and no exact
    # match (r_i=0): o_i = 0.8*1 + 0.2*0 = 0.8.
    nli = _FakeNli([0.0, 0.0, 0.0])
    o = observed_consistency("ref", ["a", "b", "c"], nli, alpha=0.8)
    assert o == pytest.approx(0.8)


def test_observed_exact_match_pushes_to_one() -> None:
    # Samples equal to the reference get r_i=1; with p(contradiction)=0 -> o_i=1.
    nli = _FakeNli([0.0, 0.0])
    o = observed_consistency("ref", ["ref", "ref"], nli, alpha=0.8)
    assert o == pytest.approx(1.0)


class _DirectionalNli:
    """Returns different contradiction probs for the two NLI call directions."""

    def __init__(self, first: List[float], second: List[float]) -> None:
        self._calls = [first, second]
        self._i = 0

    def contradiction_probs(self, premises: List[str], hypotheses: List[str]) -> List[float]:
        out = self._calls[min(self._i, 1)]
        self._i += 1
        return list(out[: len(premises)])


def test_observed_averages_both_nli_directions() -> None:
    # Direction 1 (y_i -> y): p=0.2 ; Direction 2 (y -> y_i): p=0.6
    # averaged p = 0.4 -> s = 0.6 -> o = 0.8*0.6 + 0.2*0 = 0.48
    # (a max-based combine would give p=0.6 -> s=0.4 -> o=0.32, so this
    # distinguishes the paper's averaging from the old conservative max.)
    nli = _DirectionalNli([0.2], [0.6])
    o = observed_consistency("ref", ["different"], nli, alpha=0.8)
    assert o == pytest.approx(0.48)


def test_self_reflection_prompt_includes_evidence() -> None:
    from app.uncertainty.reflection import build_self_reflection_prompt

    evidence = 'SQL query 1:\nSELECT MAX("Sales") FROM "Orders"\nResult 1:\n[(22638.48,)]'
    prompt = build_self_reflection_prompt("highest sales?", "22,638.48 dollars", 0, evidence)
    assert "22638.48" in prompt
    assert "source of truth" in prompt
    # The reframed prompt must NOT ask for external verification.
    assert "reliable sources" not in prompt


def test_self_reflection_certainty_threads_evidence() -> None:
    from app.uncertainty.reflection import self_reflection_certainty

    seen = {}

    def capture(prompt: str) -> str:
        seen["prompt"] = prompt
        return "90"

    s = self_reflection_certainty(
        "q", "a", capture, rounds=1, evidence="SQL query 1:\nSELECT 1\nResult 1:\n[(1,)]"
    )
    assert s == pytest.approx(0.9)
    assert "SELECT 1" in seen["prompt"]


class _RecordingAgent:
    """Captures the 'input' it is invoked with; returns a fixed answer."""

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def invoke(self, payload: dict) -> dict:
        self.inputs.append(payload["input"])
        return {"output": "answer"}


def _make_scorer(agent: _RecordingAgent, use_diversity: bool):
    from app.uncertainty.scorer import ConfidenceScorer

    return ConfidenceScorer(
        sampling_agent=agent,
        nli=_FakeNli([0.0, 0.0, 0.0]),
        complete_fn=lambda _p: "90",
        k=3,
        use_diversity_prompt=use_diversity,
    )


def test_diversity_prompt_applied_to_samples_only_when_enabled() -> None:
    agent = _RecordingAgent()
    scorer = _make_scorer(agent, use_diversity=True)
    scorer._generate_samples("What's the highest sales?")
    assert len(agent.inputs) == 3
    # Question text is preserved, but wrapped with the CoT diversity prefix.
    assert all("What's the highest sales?" in inp for inp in agent.inputs)
    assert all(inp != "What's the highest sales?" for inp in agent.inputs)
    assert all("step by step" in inp.lower() for inp in agent.inputs)


def test_diversity_prompt_can_be_disabled() -> None:
    agent = _RecordingAgent()
    scorer = _make_scorer(agent, use_diversity=False)
    scorer._generate_samples("What's the highest sales?")
    # Bare question, no wrapping.
    assert all(inp == "What's the highest sales?" for inp in agent.inputs)
