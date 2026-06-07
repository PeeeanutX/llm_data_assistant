"""Application configuration loaded from environment variables.

All runtime knobs live here so that the rest of the codebase never reads
``os.environ`` directly. This keeps configuration explicit, testable, and
documented in a single place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read from environment variables (or a ``.env`` file in the
    project root). Field defaults reflect the behaviour of the legacy script.
    """

    # ---- OpenAI ----------------------------------------------------------
    openai_api_key: SecretStr = Field(..., description="OpenAI API key.")
    openai_model: str = Field(
        default="gpt-5.4-mini",
        description="Model used for both the SQL agent and the step explainer.",
    )

    # ---- PostgreSQL ------------------------------------------------------
    database_url: str = Field(..., description="PostgreSQL connection URL.")
    interactions_table: str = Field(
        default="interactions_experiment",
        description="Name of the table that stores the per-turn interaction log.",
    )
    pg_pool_min: int = Field(default=1, ge=1)
    pg_pool_max: int = Field(default=5, ge=1)

    # ---- Data loading ----------------------------------------------------
    excel_file_path: Path = Field(
        default=Path("streamlit_agent/(US)Sample-Superstore.xlsx"),
        description="Source Excel workbook that backs the SQL agent.",
    )
    sqlite_db_path: Path = Field(
        default=Path("database.db"),
        description="On-disk SQLite database produced from the Excel workbook.",
    )

    # ---- Token budget for the agent's chat history -----------------------
    max_total_tokens: int = Field(default=14_600, ge=1)
    reserved_tokens: int = Field(default=1_000, ge=0)

    # ---- Confidence / uncertainty (BSDetector) ---------------------------
    confidence_view: str = Field(
        default="numerical",
        description="How confidence is shown. Currently only 'numerical'.",
    )
    confidence_k: int = Field(
        default=5, ge=1, description="Samples for Observed Consistency (O)."
    )
    confidence_rounds: int = Field(
        default=3, ge=1, description="Rounds for Self-Reflection Certainty (S)."
    )
    confidence_alpha: float = Field(
        default=0.8, ge=0.0, le=1.0, description="NLI vs exact-match trade-off in O."
    )
    confidence_beta: float = Field(
        default=0.7, ge=0.0, le=1.0, description="O vs S trade-off in C."
    )
    confidence_sample_temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Temperature for O resampling."
    )
    confidence_use_diversity_prompt: bool = Field(
        default=True,
        description=(
            "Apply a Chain-of-Thought diversity prefix to the O sampling runs "
            "(paper Section 3.1); reference answer is unaffected."
        ),
    )
    nli_model: str = Field(
        default="FacebookAI/roberta-large-mnli",
        description=(
            "HuggingFace NLI model for Observed Consistency. The default is "
            "served as text-classification, so it works with both backends. "
            "For the 'local' backend on CPU, the lighter "
            "'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli' is a good swap "
            "(but it is zero-shot-classification on the router, so it is NOT "
            "usable with the 'inference' backend)."
        ),
    )
    nli_backend: str = Field(
        default="inference",
        description=(
            "Where the NLI model runs: 'local' (torch/transformers on this "
            "machine) or 'inference' (hosted HuggingFace Inference Providers "
            "router over HTTP, no torch required). The 'inference' backend "
            "needs a text-classification NLI model (e.g. roberta-large-mnli); "
            "zero-shot-classification checkpoints only work with 'local'."
        ),
    )
    hf_api_token: SecretStr | None = Field(
        default=None,
        description="HuggingFace API token, used when nli_backend='inference'.",
    )
    nli_inference_endpoint: str | None = Field(
        default=None,
        description=(
            "Override URL for the NLI Inference API. Defaults to the "
            "hf-inference router endpoint for `nli_model` when unset."
        ),
    )
    nli_inference_timeout: float = Field(
        default=60.0,
        gt=0.0,
        description=(
            "Per-request timeout (seconds) for the NLI Inference API. Generous "
            "enough to absorb a provider cold start; transient timeouts and "
            "dropped connections are retried (see InferenceNliModel._post)."
        ),
    )

    # ---- Explainer LLM call ---------------------------------------------
    explanation_max_tokens: int = Field(default=1_000, ge=1)
    explanation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide singleton ``Settings`` instance."""
    return Settings()
