# Data Assistant — research PoC

A Streamlit chatbot for a research experiment on whether showing the
SQL agent's reasoning steps affects how users interact with it. The agent
answers natural-language questions about an Excel workbook by translating
them into SQL through a LangChain SQL agent; every interaction is logged
to PostgreSQL with timing metadata for the A/B comparison.

This is the PoC-appropriate version: deliberately conservative on
tooling so that the artifact is reproducible by reviewers and survives
on pinned dependencies for the lifetime of the study (and beyond).

## Project layout

```
.
├── streamlit_app.py            Entry point
├── Dockerfile / compose.yaml   Optional dev stack (app + Postgres)
├── pytest.ini                  Test runner config
├── requirements.txt            Runtime dependencies
├── requirements-dev.txt        Test dependencies
└── app/
    ├── config.py               Pydantic settings (env-driven)
    ├── logging_setup.py        Root logger configuration
    ├── prompts.py              System prompts and static strings
    ├── data_loader.py          Excel -> read-only SQLite
    ├── db/
    │   ├── pool.py             psycopg2 connection pool
    │   ├── schema.py           Idempotent table creation
    │   └── repository.py       InteractionRecord + InteractionRepository
    ├── agent/
    │   ├── factory.py          LangChain SQL agent (create_sql_agent)
    │   ├── tokens.py           Token counting + history truncation
    │   ├── steps.py            Prettifies intermediate agent steps
    │   └── explainer.py        Second LLM pass that simplifies steps
    ├── uncertainty/            Confidence scoring (BSDetector)
    │   ├── nli.py              DeBERTa NLI wrapper (contradiction probs)
    │   ├── observed.py         Observed Consistency O
    │   ├── reflection.py       Self-Reflection Certainty S
    │   ├── score.py            C = beta*O + (1-beta)*S
    │   └── scorer.py           Orchestrates O + S for one answer
    └── ui/
        ├── session.py          Session state and query-param parsing
        └── chat.py             Chat rendering and per-turn orchestration
```

## Quick start (local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `DATABASE_URL`.
3. Run:
   ```bash
   streamlit run streamlit_app.py
   ```

The interactions table is created automatically on startup if it does not
already exist.

## Quick start (Docker)

```bash
export OPENAI_API_KEY=sk-...
docker compose up --build
```

Boots Postgres + the app; visit http://localhost:8501. Place the source
workbook under `./streamlit_agent/` first — it is mounted into the container.

## A/B experiment URL parameters

| Parameter      | Type    | Effect                                                                                                       |
|----------------|---------|--------------------------------------------------------------------------------------------------------------|
| `participant`  | string  | Recorded against every interaction row.                                                                      |
| `treatment`    | `1`/`2` | `2` hides the step explanation behind a "See explanation" button; any other value shows it immediately.      |

Example: `https://.../?participant=p042&treatment=2`

## Confidence scoring (BSDetector)

Each answer that is **grounded in a database query** is given a numeric
confidence score, following Chen & Mueller (2024), "Quantifying Uncertainty in
Answers from any Language Model" (ACL 2024):

```
C = beta * O + (1 - beta) * S
```

- **O — Observed Consistency.** The SQL agent is re-run `CONFIDENCE_K` times at
  `CONFIDENCE_SAMPLE_TEMPERATURE` (with a Chain-of-Thought diversity prefix on
  the sampling runs, per the paper; toggle via `CONFIDENCE_USE_DIVERSITY_PROMPT`),
  and each resampled answer is compared to the reference answer with a DeBERTa
  NLI model (1 - p(contradiction), averaged across both directions per paper
  A.1, blended with an exact-match indicator via `CONFIDENCE_ALPHA`).
- **S — Self-Reflection Certainty.** The model is shown the SQL query that was
  run and the rows it returned, and asked over `CONFIDENCE_ROUNDS` rounds to
  estimate the probability the answer *faithfully reflects that result*,
  treating the database as the source of truth (rather than asking whether the
  fact is verifiable against outside knowledge -- which fails for private data).
- The score is shown as a **numeric certainty meter** under the answer
  (`CONFIDENCE_VIEW=numerical`) and logged to PostgreSQL.

Greetings and off-topic replies (no SQL query) are **not** scored.

**Cost & latency.** Scoring multiplies the per-question OpenAI calls: roughly
`CONFIDENCE_K` full agent runs plus `CONFIDENCE_ROUNDS` reflection calls. With
the defaults that is ~20+ extra calls per data question. The five resamples are
currently sequential.

**NLI model / hardware.** `NLI_MODEL` defaults to `DeBERTa-v3-base`, sized for
CPU (the model downloads ~750 MB on first run and is cached for the process).
On CPU the NLI inference is a few seconds per question; the LLM calls dominate
latency. Swap `NLI_MODEL` to a smaller MNLI model for more speed, or to the
`...large...` variant on GPU for best NLI quality.

**Logged columns** (added to `interactions_experiment`, non-destructively):
`confidence_percent`, `o_score`, `s_score`, `confidence_view`.

**Adaptations from the paper / original feature.** (1) The answer is produced
by a multi-step SQL agent grounded in real data, so confidence is computed over
the agent's natural-language summary and observed consistency resamples the
whole agent. (2) **Self-reflection is grounded in the query evidence**: it is
given the executed SQL and the returned rows and asked whether the answer
faithfully reflects them, because the paper's external-verifiability framing
collapses to a floor on private-database facts (the model cannot corroborate a
dataset value from world knowledge). This is the key adaptation for a data
assistant. (3) S uses a numeric 0-100 estimate with anchors rather than the
paper's multiple-choice scale (the calibration diagnostic confirms it does not
suffer the saturation the paper warns of for numeric scales). Observed
Consistency otherwise follows the paper: it averages both NLI directions (A.1)
and applies the Chain-of-Thought diversity prefix to sampling (Section 3.1).

**Validating S.** Self-reflection should reward faithful answers *and*
penalise unfaithful ones. To check it has not simply saturated at a high
ceiling, run the calibration diagnostic (needs only an OpenAI key, no DB):

```bash
python scripts/check_reflection_calibration.py
```

It scores a faithful answer and several deliberately unfaithful ones against
the same SQL evidence and reports the spread.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                 # full suite
pytest -m "not db"     # pure unit tests only (no Postgres needed)
pytest --cov=app       # with coverage
```

Pure tests cover `tokens.py`, `steps.py`, and the confidence math in `app/uncertainty/` (the NLI model and LLM calls are injected as fakes). The DB-backed test
(`tests/test_repository.py`, marked `db`) spins up an ephemeral PostgreSQL
via `pytest-postgresql`, runs `initialize_schema`, and exercises the
repository end-to-end. It skips automatically if the PostgreSQL server
binaries (`pg_ctl`, `initdb`) are not on `PATH`.

## Notes on what changed from the original single-file script

- The 700-line monolith is split into focused modules, but only as much as
  needed for a researcher or collaborator to navigate the code.
- Bug fixes that were preventing intended behaviour:
  - `treatment` is now parsed as `int` (the original compared a `str` to `2`,
    which never matched — treatment 2 was effectively dead).
  - The PostgreSQL context manager rolls back on exception instead of
    committing.
  - The step explainer sends `max_completion_tokens` by default with a
    one-shot fallback to `max_tokens`, so it works with both modern and
    older OpenAI models.
- Dead code (`process_user_query` with SQL injection, `execute_sql_query`,
  the unused `chain_with_history`) was removed.
- Configuration moved to a pydantic-settings class so missing env vars
  fail fast and explicitly.
