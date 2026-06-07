"""Diagnostic: does evidence-grounded self-reflection (S) actually discriminate?

Scores a faithful answer and several deliberately unfaithful answers against
the SAME SQL evidence, and prints S for each. If S is well-calibrated the
faithful case should score high and the unfaithful cases noticeably lower; if
every case scores ~0.95, S has saturated into a ceiling and is not carrying
information beyond Observed Consistency (O).

Run from the project root (uses your .env / environment):

    python scripts/check_reflection_calibration.py

Or score one custom case:

    python scripts/check_reflection_calibration.py \
        --question "What's the highest sales?" \
        --evidence 'SQL query 1:\nSELECT MAX("Sales") FROM "Orders"\nResult 1:\n[(22638.48,)]' \
        --answer "The highest sales is 22,638.48 dollars."

This script does NOT require PostgreSQL -- only an OpenAI key.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the project importable when run as `python scripts/...`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.uncertainty.reflection import self_reflection_certainty  # noqa: E402

# Default calibration scenario: the canonical "highest sales" answer plus
# three ways an answer can be unfaithful to the same query result.
_DEFAULT_QUESTION = "What's the highest sales?"
_DEFAULT_EVIDENCE = (
    'SQL query 1:\nSELECT MAX("Sales") FROM "Orders"\nResult 1:\n[(22638.48,)]'
)
_DEFAULT_CASES = {
    "faithful": "The highest sales is 22,638.48 dollars.",
    "wrong value": "The highest sales is 5.00 dollars.",
    "wrong question": "The highest profit is 22,638.48 dollars.",
    "fabricated extra": (
        "The highest sales is 22,638.48 dollars, sold in Berlin to ACME Corp."
    ),
}


def _make_complete_fn(api_key: str, model: str):
    """Plain completion with the same max_completion_tokens fallback as the app."""
    from openai import BadRequestError, OpenAI

    client = OpenAI(api_key=api_key)
    state = {"param": "max_completion_tokens"}

    def complete(prompt: str) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            state["param"]: 256,
        }
        try:
            resp = client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            msg = str(exc).lower()
            if "max_tokens" not in msg and "max_completion_tokens" not in msg:
                raise
            other = (
                "max_tokens"
                if state["param"] == "max_completion_tokens"
                else "max_completion_tokens"
            )
            kwargs.pop(state["param"])
            kwargs[other] = 256
            state["param"] = other
            resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=None)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--answer", default=None, help="Score a single custom answer.")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_PROJECT_ROOT / ".env")
    except Exception:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set (check your .env).", file=sys.stderr)
        return 1
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    complete = _make_complete_fn(api_key, model)

    question = args.question or _DEFAULT_QUESTION
    evidence = args.evidence or _DEFAULT_EVIDENCE

    print(f"model={model}  rounds={args.rounds}")
    print(f"question: {question}\n")

    if args.answer is not None:
        s = self_reflection_certainty(
            question, args.answer, complete, rounds=args.rounds, evidence=evidence
        )
        print(f"S = {s:.3f}")
        return 0

    scores: dict[str, float] = {}
    for label, answer in _DEFAULT_CASES.items():
        s = self_reflection_certainty(
            question, answer, complete, rounds=args.rounds, evidence=evidence
        )
        scores[label] = s
        print(f"{label:16s} S = {s:.3f}")

    faithful = scores.get("faithful", 0.0)
    worst_unfaithful = max(
        (v for k, v in scores.items() if k != "faithful"), default=0.0
    )
    spread = faithful - worst_unfaithful
    print(f"\nspread (faithful - worst unfaithful) = {spread:.3f}")
    if spread >= 0.3:
        print("=> S discriminates: faithful clearly outscores unfaithful answers.")
    elif spread >= 0.1:
        print("=> Weak discrimination: consider strengthening the adversarial round.")
    else:
        print("=> S appears SATURATED (near-constant high); it is not adding signal "
              "beyond O. Worth tuning the prompt or reconsidering S.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
