"""NLI model wrapper for contradiction scoring (the heavy dependency).

Loads a HuggingFace MNLI model and exposes ``contradiction_probs`` for batches
of (premise, hypothesis) pairs. Kept free of Streamlit so it can be loaded once
and cached by the caller (see streamlit_app._load_nli).

Default model is the CPU-friendly DeBERTa-v3-base; swap via the ``nli_model``
setting (xsmall = faster, large = the thesis original, best on GPU).
"""
from __future__ import annotations

import logging
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class NliModel:
    """Wraps an MNLI sequence-classification model for contradiction scoring."""

    def __init__(self, model_name: str) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading NLI model %s on %s", model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        self._contra_idx = self._find_contradiction_index()

    def _find_contradiction_index(self) -> int:
        id2label: Dict[int, str] = {
            i: lab.lower() for i, lab in self._model.config.id2label.items()
        }
        for idx, lab in id2label.items():
            if "contradiction" in lab:
                return idx
        raise RuntimeError(f"No 'contradiction' label found in {id2label}")

    @torch.no_grad()
    def contradiction_probs(
        self, premises: List[str], hypotheses: List[str]
    ) -> List[float]:
        """Return p(contradiction) for each (premise, hypothesis) pair."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must be the same length")
        if not premises:
            return []

        enc = self._tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, self._contra_idx].tolist()


def load_nli_model(model_name: str) -> NliModel:
    """Construct an :class:`NliModel`. Heavy; the caller should cache it."""
    return NliModel(model_name)
