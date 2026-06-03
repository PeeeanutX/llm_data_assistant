"""Observed Consistency (the O term of BSDetector).

Given a reference answer y and k independently sampled answers {y_i}, score how
consistent they are. For each sample:

    s_i = 1 - 0.5 * (p_contradiction(y_i -> y) + p_contradiction(y -> y_i))
    r_i = 1[y_i == y]                                                     # exact match
    o_i = alpha * s_i + (1 - alpha) * r_i
    O   = mean(o_i)

The NLI scorer is injected (duck-typed: it must expose
``contradiction_probs(premises, hypotheses) -> list[float]``) so this module
carries no torch/transformers dependency and is unit-testable with a fake.

s_i averages the contradiction probability over both orderings to mitigate the
NLI model's positional bias, matching the paper's Appendix A.1:
    s_i = 0.5 * ((1 - p_contradiction) + (1 - p'_contradiction)).
"""
from __future__ import annotations

import logging
from typing import List, Protocol

logger = logging.getLogger(__name__)

ALPHA_DEFAULT: float = 0.8


class NliScorer(Protocol):
    """Anything that can score contradiction probability for text pairs."""

    def contradiction_probs(
        self, premises: List[str], hypotheses: List[str]
    ) -> List[float]: ...


def observed_consistency(
    reference_answer: str,
    samples: List[str],
    nli: NliScorer,
    *,
    alpha: float = ALPHA_DEFAULT,
) -> float:
    """Return the Observed Consistency score O in [0, 1]."""
    if not samples:
        return 0.0

    ref = reference_answer.strip()
    samples_norm = [s.strip() for s in samples]
    k = len(samples_norm)

    # Direction 1: y_i -> y ; Direction 2: y -> y_i
    p_yi_to_y = nli.contradiction_probs(samples_norm, [ref] * k)
    p_y_to_yi = nli.contradiction_probs([ref] * k, samples_norm)

    total = 0.0
    for i in range(k):
        # Average both NLI directions to reduce positional bias (paper A.1).
        p_contra = 0.5 * (p_yi_to_y[i] + p_y_to_yi[i])
        s_i = 1.0 - p_contra
        r_i = 1.0 if samples_norm[i] == ref else 0.0
        total += alpha * s_i + (1.0 - alpha) * r_i

    return total / k
