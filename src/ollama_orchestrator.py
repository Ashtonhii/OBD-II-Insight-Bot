from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib import error, request

from agent_registry import AgentResponse, run_pal_agent, run_rag_agent


Route = Literal["pal", "rag"]


@dataclass
class RouteDecision:
    route: Route
    rationale: str


@dataclass
class OrchestratorResult:
    decision: RouteDecision
    response: AgentResponse


class OllamaOrchestrator:
    def __init__(
        self,
        model: str = "granite3.3",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        req = request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise ConnectionError(
                "Could not reach Ollama. Ensure Ollama is running and granite3.3 is installed."
            ) from exc

        data = json.loads(raw)
        return data["message"]["content"]

    @staticmethod
    def _extract_json(text: str) -> dict[str, str]:
        raw = text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            for part in parts:
                chunk = part.strip()
                if chunk.startswith("json"):
                    raw = chunk.removeprefix("json").strip()
                    break
                if chunk.startswith("{"):
                    raw = chunk
                    break

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Router returned invalid JSON: {text}") from exc

        route = str(parsed.get("route", "")).strip().lower()
        rationale = str(parsed.get("rationale", "")).strip()

        if route not in {"pal", "rag"}:
            raise ValueError(f"Router returned unsupported route '{route}'. Raw: {text}")

        if not rationale:
            rationale = "Route selected by orchestrator."

        return {"route": route, "rationale": rationale}

    def decide_route(self, question: str, csv_path: str | Path | None = None) -> RouteDecision:
        csv_available = bool(csv_path)

        system_prompt = (
            "You are a strict routing agent for a multi-agent vehicle assistant. "
            "Choose exactly one route:\n"
            "- 'pal' for questions that require analysis of CSV telemetry/data logs.\n"
            "- 'rag' for general vehicle diagnostics knowledge, troubleshooting, DTC meaning, repair workflow.\n"
            "Return ONLY valid JSON with keys: route, rationale."
        )

        user_prompt = (
            f"Question: {question}\n"
            f"CSV available: {csv_available}\n\n"
            "Rules:\n"
            "1) If the question asks to compute, filter, aggregate, trend, or compare values from a dataset/log, choose 'pal'.\n"
            "2) If the question asks for general diagnostic guidance/knowledge independent of provided telemetry, choose 'rag'.\n"
            "3) If unclear, prefer 'rag' unless explicit dataset computation is requested."
        )

        content = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        parsed = self._extract_json(content)
        route: Route = "pal" if parsed["route"] == "pal" else "rag"

        if route == "pal" and not csv_available:
            route = "rag"
            parsed["rationale"] = (
                f"{parsed['rationale']} PAL selected, but no CSV was provided; falling back to RAG."
            )

        return RouteDecision(route=route, rationale=parsed["rationale"])

    def route_and_run(
        self,
        *,
        question: str,
        csv_path: str | Path | None = None,
        pal_model: str = "granite-code:8b",
        rag_model: str = "granite3.3",
        docs_dir: str | Path = "knowledge/diagnostics",
        top_k: int = 4,
    ) -> OrchestratorResult:
        decision = self.decide_route(question=question, csv_path=csv_path)

        if decision.route == "pal":
            if not csv_path:
                raise ValueError("PAL route selected but csv_path was not provided.")
            response = run_pal_agent(csv_path=csv_path, question=question, model=pal_model)
        else:
            response = run_rag_agent(question=question, docs_dir=docs_dir, model=rag_model, top_k=top_k)

        return OrchestratorResult(decision=decision, response=response)


def orchestrate_question(
    *,
    question: str,
    csv_path: str | Path | None = None,
    router_model: str = "granite3.3",
    pal_model: str = "granite-code:8b",
    rag_model: str = "granite3.3",
    docs_dir: str | Path = "knowledge/diagnostics",
    top_k: int = 4,
) -> OrchestratorResult:
    orchestrator = OllamaOrchestrator(model=router_model)
    return orchestrator.route_and_run(
        question=question,
        csv_path=csv_path,
        pal_model=pal_model,
        rag_model=rag_model,
        docs_dir=docs_dir,
        top_k=max(1, top_k),
    )
