"""NLI model wrappers for contradiction scoring (the heavy dependency).

Two interchangeable backends expose the same ``contradiction_probs(premises,
hypotheses) -> list[float]`` surface (the :class:`~app.uncertainty.observed.NliScorer`
protocol):

* :class:`NliModel` -- runs a HuggingFace MNLI model locally via
  torch/transformers. Best quality and no per-call latency, but pulls in the
  heavy torch+transformers stack (~1.4 GB on first download).
* :class:`InferenceNliModel` -- calls the hosted HuggingFace Inference
  Providers API over HTTP. Needs only ``requests`` and an API token, so it runs
  anywhere torch cannot (limited disk/RAM, no GPU, locked-down environment).
  The model must be served as ``text-classification`` (e.g.
  ``FacebookAI/roberta-large-mnli``); ``zero-shot-classification`` checkpoints
  are not accepted by this sentence-pair call.

``load_nli_model`` picks the backend. Both are kept free of Streamlit so they
can be loaded once and cached by the caller (see streamlit_app._load_nli).

Default model is ``FacebookAI/roberta-large-mnli`` (text-classification, so it
works with either backend); swap via the ``nli_model`` setting -- e.g. the
lighter DeBERTa-v3-base for CPU ``local`` runs (it is zero-shot on the router
and so unusable with the ``inference`` backend).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NliModel:
    """Wraps an MNLI sequence-classification model for contradiction scoring.

    torch/transformers are imported lazily here so that merely importing this
    module (e.g. to use :class:`InferenceNliModel`) does not require them.
    """

    def __init__(self, model_name: str) -> None:
        import torch  # local import: only the local backend needs torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
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

    def contradiction_probs(
        self, premises: List[str], hypotheses: List[str]
    ) -> List[float]:
        """Return p(contradiction) for each (premise, hypothesis) pair."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must be the same length")
        if not premises:
            return []

        with self._torch.no_grad():
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
            probs = self._torch.softmax(logits, dim=-1)
            return probs[:, self._contra_idx].tolist()


class InferenceNliModel:
    """Contradiction scorer backed by the HuggingFace Inference Providers API.

    Sends each (premise, hypothesis) pair as a sentence-pair text-classification
    request and reads back the ``contradiction`` label's probability. Requires
    only ``requests`` and an API token -- no torch/transformers.

    The target model must be served as ``text-classification`` so the provider
    returns a 3-way (contradiction/neutral/entailment) softmax;
    ``FacebookAI/roberta-large-mnli`` works. Checkpoints tagged
    ``zero-shot-classification`` (e.g. ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``)
    are rejected -- run those with the ``local`` backend instead.
    """

    # HuggingFace retired api-inference.huggingface.co in favour of the routed
    # Inference Providers endpoint; ``hf-inference`` is the first-party provider.
    _DEFAULT_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model}"
    _MAX_RETRIES = 4  # retry transient 503s while the model cold-starts
    _DEFAULT_COLD_START_WAIT = 5.0  # seconds, when the API gives no estimate

    def __init__(
        self,
        model_name: str,
        api_token: Optional[str] = None,
        *,
        endpoint: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        import requests  # local import: only this backend needs requests

        url = endpoint or self._DEFAULT_ENDPOINT.format(model=model_name)
        logger.info("Using NLI Inference API at %s", url)
        self._url = url
        self._timeout = timeout
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        self._session.headers.update(headers)

    @staticmethod
    def _contradiction_from_scores(scores: List[Dict[str, object]]) -> float:
        """Pull the contradiction probability out of one label-score list."""
        for entry in scores:
            label = str(entry.get("label", "")).lower()
            if "contradiction" in label:
                return float(entry["score"])
        raise RuntimeError(f"No 'contradiction' label in Inference API reply: {scores}")

    def _cold_start_wait(self, resp) -> float:
        """Seconds to wait before retrying a 503, honouring the API estimate."""
        try:
            estimate = float(resp.json().get("estimated_time"))
        except (ValueError, TypeError, AttributeError):
            estimate = self._DEFAULT_COLD_START_WAIT
        # Never block longer than a single request is allowed to take.
        return max(0.0, min(estimate, self._timeout))

    def _post(self, payload: dict):
        """POST the batch, retrying through provider cold starts.

        A cold-starting provider may return HTTP 503 with an ``estimated_time``
        *or* simply hold the connection open until it times out, so transient
        network failures (read timeouts / dropped connections) are retried the
        same way as an explicit 503.
        """
        import time

        import requests

        transient_errors = (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        )
        for attempt in range(self._MAX_RETRIES):
            last = attempt == self._MAX_RETRIES - 1
            try:
                resp = self._session.post(
                    self._url, json=payload, timeout=self._timeout
                )
            except transient_errors as exc:
                # No response arrived (e.g. ReadTimeout): treat it as a cold
                # start and retry, since the model may finish loading meanwhile.
                if last:
                    raise RuntimeError(
                        f"NLI Inference API unreachable after "
                        f"{self._MAX_RETRIES} attempts: {exc}"
                    ) from exc
                wait = min(self._DEFAULT_COLD_START_WAIT, self._timeout)
                logger.info(
                    "NLI request failed (%s); retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    self._MAX_RETRIES - 1,
                    wait,
                )
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 503:
                # Provider is still loading the model; back off and retry.
                if not last:
                    wait = self._cold_start_wait(resp)
                    logger.info(
                        "NLI model loading on provider; retry %d/%d in %.1fs",
                        attempt + 1,
                        self._MAX_RETRIES - 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"NLI Inference API still loading (503) after "
                    f"{self._MAX_RETRIES} attempts: {resp.text}"
                )
            # A 400 mentioning zero-shot means the model is the wrong task for
            # this sentence-pair call; surface a pointed hint (other 400s, e.g.
            # a malformed batch, are left to speak for themselves).
            hint = ""
            if resp.status_code == 400 and "zero-shot" in resp.text.lower():
                hint = (
                    " -- the model is served as 'zero-shot-classification'; "
                    "use a 'text-classification' NLI model (e.g. "
                    "FacebookAI/roberta-large-mnli) or the 'local' backend"
                )
            raise RuntimeError(
                f"NLI Inference API request failed ({resp.status_code}): "
                f"{resp.text}{hint}"
            )

    def contradiction_probs(
        self, premises: List[str], hypotheses: List[str]
    ) -> List[float]:
        """Return p(contradiction) for each (premise, hypothesis) pair."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must be the same length")
        if not premises:
            return []

        # One request for the whole batch: a list of sentence-pair inputs. Ask
        # for every label's score (top_k=None) rather than just the argmax, and
        # truncate over-long pairs server-side -- the model caps at 512 tokens
        # and would otherwise 400 (mirrors the local backend's truncation).
        payload = {
            "inputs": [
                {"text": premise, "text_pair": hypothesis}
                for premise, hypothesis in zip(premises, hypotheses)
            ],
            "parameters": {"top_k": None, "truncation": True},
        }
        data = self._post(payload)

        # The API returns a list aligned with the inputs; each element is itself
        # the list of {label, score} dicts. (When a single input is sent some
        # deployments unwrap one level, so normalise both shapes.)
        if data and isinstance(data[0], dict) and "label" in data[0]:
            data = [data]
        if len(data) != len(premises):
            raise RuntimeError(
                f"Expected {len(premises)} NLI results, got {len(data)}: {data}"
            )
        return [self._contradiction_from_scores(scores) for scores in data]


def load_nli_model(
    model_name: str,
    *,
    backend: str = "local",
    api_token: Optional[str] = None,
    endpoint: Optional[str] = None,
    timeout: float = 60.0,
):
    """Construct an NLI scorer for the chosen backend.

    ``backend`` is ``"local"`` (torch/transformers) or ``"inference"`` (hosted
    HuggingFace Inference API). Heavy to build; the caller should cache it.
    """
    backend = backend.lower()
    if backend == "local":
        return NliModel(model_name)
    if backend == "inference":
        return InferenceNliModel(
            model_name, api_token, endpoint=endpoint, timeout=timeout
        )
    raise ValueError(f"Unknown NLI backend {backend!r}; expected 'local' or 'inference'")
