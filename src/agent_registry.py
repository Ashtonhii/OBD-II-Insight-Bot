from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ollama_pal import PALResult, ask_question_on_csv
from ollama_rag import RAGResult, ask_vehicle_diagnostics


AgentType = Literal["pal", "rag"]


@dataclass
class AgentResponse:
    agent: AgentType
    payload: PALResult | RAGResult


def run_pal_agent(
    *,
    csv_path: str | Path,
    question: str,
    model: str = "granite-code:8b",
    conversation_context: str = "",
) -> AgentResponse:
    result = ask_question_on_csv(
        csv_path=csv_path,
        question=question,
        model=model,
        conversation_context=conversation_context,
    )
    return AgentResponse(agent="pal", payload=result)


def run_rag_agent(
    *,
    question: str,
    docs_dir: str | Path = "knowledge/diagnostics",
    model: str = "granite3.3",
    top_k: int = 4,
    conversation_context: str = "",
) -> AgentResponse:
    result = ask_vehicle_diagnostics(
        question=question,
        docs_dir=docs_dir,
        model=model,
        top_k=top_k,
        conversation_context=conversation_context,
    )
    return AgentResponse(agent="rag", payload=result)


def run_agent(agent: AgentType, **kwargs: Any) -> AgentResponse:
    if agent == "pal":
        return run_pal_agent(**kwargs)
    if agent == "rag":
        return run_rag_agent(**kwargs)
    raise ValueError(f"Unsupported agent type: {agent}")
