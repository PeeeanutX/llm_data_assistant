"""Confidence orchestration for the data assistant.

Ties together the three BSDetector pieces for one (question, answer) pair:

1. Observed Consistency O -- resample the SQL agent k times at high temperature
   and NLI-compare each sample against the reference answer.
2. Self-Reflection Certainty S -- ask the model to rate its own answer.
3. Combine: C = beta * O + (1 - beta) * S.

Resampling uses the *agent* (not a plain LLM call) because in this app the
answer the user sees is the agent's SQL-grounded output, so observed
consistency must measure the agent's agreement with itself.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List

from app.uncertainty.observed import ALPHA_DEFAULT, NliScorer, observed_consistency
from app.uncertainty.reflection import CompleteFn, self_reflection_certainty
from app.uncertainty.score import BETA_DEFAULT, combine_confidence

logger = logging.getLogger(__name__)

# Applied to the SAMPLING runs only (never the reference answer), mirroring the
# paper's diversity mechanism: it modifies the prompt used to sample varied
# responses with a Chain-of-Thought nudge (Section 3.1 / Figure 6a), which the
# paper found essential for sample diversity (Table 3b) beyond temperature
# alone. Kept soft so it does not fight the SQL agent's own output format.
_DIVERSITY_PREFIX = (
    "Reason step by step about how to interpret this question and compute the "
    "answer from the database, then give your final answer.\n\nQuestion: "
)


@dataclass(frozen=True)
class ConfidenceResult:
    """Outcome of scoring one answer."""

    percent: float  # C * 100, in [0, 100]
    observed: float  # O in [0, 1]
    self_reflection: float  # S in [0, 1]


class ConfidenceScorer:
    """Computes BSDetector confidence for the data assistant's answers."""

    def __init__(
        self,
        sampling_agent: Any,
        nli: NliScorer,
        complete_fn: CompleteFn,
        *,
        k: int = 5,
        rounds: int = 3,
        alpha: float = ALPHA_DEFAULT,
        beta: float = BETA_DEFAULT,
        use_diversity_prompt: bool = True,
    ) -> None:
        self._sampling_agent = sampling_agent
        self._nli = nli
        self._complete_fn = complete_fn
        self._k = max(1, k)
        self._rounds = max(1, rounds)
        self._alpha = alpha
        self._beta = beta
        self._use_diversity_prompt = use_diversity_prompt

    def score(
        self, question: str, reference_answer: str, evidence: str = ""
    ) -> ConfidenceResult:
        """Return the confidence for ``reference_answer`` to ``question``.

        ``evidence`` is the SQL-query-and-results context for the answer; it is
        passed to self-reflection so S judges faithfulness to the query result
        rather than external verifiability.
        """
        total_start = time.perf_counter()

        # Observed Consistency: generating the k resamples is the dominant cost
        # (k agent invocations); the NLI comparison is timed separately because
        # on the 'inference' backend it is an HTTP round-trip.
        sampling_start = time.perf_counter()
        samples = self._generate_samples(question)
        sampling_seconds = time.perf_counter() - sampling_start

        nli_start = time.perf_counter()
        observed = observed_consistency(
            reference_answer, samples, self._nli, alpha=self._alpha
        )
        nli_seconds = time.perf_counter() - nli_start

        # Self-Reflection Certainty: `rounds` LLM rating calls.
        reflection_start = time.perf_counter()
        reflection = self_reflection_certainty(
            question,
            reference_answer,
            self._complete_fn,
            rounds=self._rounds,
            evidence=evidence,
        )
        reflection_seconds = time.perf_counter() - reflection_start

        confidence = combine_confidence(observed, reflection, beta=self._beta)
        total_seconds = time.perf_counter() - total_start
        logger.info(
            "Confidence: O=%.3f S=%.3f C=%.3f | timings(s): "
            "observed_sampling=%.3f observed_nli=%.3f reflection=%.3f total=%.3f",
            observed,
            reflection,
            confidence,
            sampling_seconds,
            nli_seconds,
            reflection_seconds,
            total_seconds,
        )
        return ConfidenceResult(
            percent=confidence * 100.0,
            observed=observed,
            self_reflection=reflection,
        )

    def _generate_samples(self, question: str) -> List[str]:
        """Resample the agent k times; failures contribute an empty answer.

        When enabled, a Chain-of-Thought diversity prefix is applied to the
        sampled question (the reference answer is left untouched), per the
        paper's diversity sampling.
        """
        sampled_input = (
            _DIVERSITY_PREFIX + question
            if self._use_diversity_prompt
            else question
        )
        samples: List[str] = []
        for i in range(self._k):
            try:
                result = self._sampling_agent.invoke(
                    {"input": sampled_input, "history": []}
                )
                samples.append(str(result.get("output", "")))
            except Exception:
                logger.exception("Sample %d/%d failed", i + 1, self._k)
                samples.append("")
        return samples
