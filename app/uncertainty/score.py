"""Overall confidence: C = beta * O + (1 - beta) * S (BSDetector eq. 1)."""
from __future__ import annotations

BETA_DEFAULT: float = 0.7


def combine_confidence(observed: float, self_reflection: float, beta: float = BETA_DEFAULT) -> float:
    """Combine Observed Consistency O and Self-Reflection Certainty S into C."""
    return beta * observed + (1.0 - beta) * self_reflection
