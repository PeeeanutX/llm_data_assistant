"""Factory for the SQL agent (LangChain AgentExecutor)."""
from __future__ import annotations

import logging
from typing import Any

from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from app.prompts import SQL_AGENT_PREFIX, SQL_AGENT_SUFFIX

logger = logging.getLogger(__name__)


def build_sql_agent(
    api_key: str, model: str, db: SQLDatabase, *, temperature: float = 0.0
) -> Any:
    """Construct the SQL agent executor.

    Returns LangChain's ``AgentExecutor`` (the legacy but still-supported
    ``create_sql_agent`` API). Its response dict is ``{"output": str,
    "intermediate_steps": list[(AgentAction, observation)]}`` -- the shape
    the UI layer consumes directly.

    ``temperature`` defaults to 0 for the primary agent; the confidence
    sampler builds a second agent at a higher temperature to obtain diverse
    answers for observed-consistency scoring.
    """
    logger.info("Building SQL agent (model=%s, temperature=%.1f)", model, temperature)
    llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature, streaming=True)
    prompt = _build_prompt()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        prompt=prompt,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SQL_AGENT_PREFIX),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template(SQL_AGENT_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
