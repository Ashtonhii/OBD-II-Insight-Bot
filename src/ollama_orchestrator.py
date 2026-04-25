from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib import error, request

from agent_registry import AgentResponse, run_pal_agent, run_rag_agent
from conversation_memory import ConversationMemory


Route = Literal["pal", "rag"]


@dataclass
class RouteDecision:
    route: Route
    rationale: str
    rewritten_question: str = ""


@dataclass
class OrchestratorResult:
    decision: RouteDecision
    response: AgentResponse


class OllamaOrchestrator:
    def __init__(
        self,
        model: str = "granite3.3",
        base_url: str = "http://localhost:11434",
        memory: ConversationMemory | None = None,
        memory_turns_for_routing: int = 5,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.memory = memory or ConversationMemory()
        self.memory_turns_for_routing = max(1, memory_turns_for_routing)

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

    def decide_route(
        self,
        question: str,
        csv_path: str | Path | None = None,
        session_id: str | None = None,
    ) -> RouteDecision:
        csv_available = bool(csv_path)
        memory_context = "No prior conversation memory."
        if session_id:
            memory_context = self.memory.format_recent_context(
                session_id=session_id,
                max_turns=self.memory_turns_for_routing,
            )

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
            f"Recent conversation memory:\n{memory_context}\n\n"
            "Rules:\n"
            "1) If the question asks to compute, filter, aggregate, trend, or compare values from a dataset/log, choose 'pal'.\n"
            "2) If the question asks for general diagnostic guidance/knowledge independent of provided telemetry, choose 'rag'.\n"
            "3) If truly unclear and no explicit data-analysis intent is present, choose 'rag'."
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

    @staticmethod
    def _extract_answer_text(response: AgentResponse) -> str:
        payload = response.payload

        if response.agent == "pal":
            return str(payload.answer)

        if response.agent == "rag":
            return str(payload.answer)

        return ""

    def _rewrite_if_referential(self, question: str, session_id: str) -> str:
        """Rewrite a follow-up question into a self-contained one using recent history.

        Only fires when the question contains a pronoun/reference word and the
        session has at least one prior turn. Returns the original question unchanged
        if no rewrite is needed or if the LLM fails to produce one.
        """
        reference_triggers = {
            "that", "it", "this", "those", "them", "its", "that value",
            "the same", "above", "previously", "based on", "now filter",
            "now what", "how does that", "convert that", "what about",
        }
        q_lower = question.lower()
        if not any(trigger in q_lower for trigger in reference_triggers):
            return question

        memory_context = self.memory.format_recent_context(
            session_id=session_id,
            max_turns=2,
            max_chars_per_answer=300,
        )
        if not memory_context or memory_context == "No prior conversation memory.":
            return question

        system_prompt = (
            "You are a question rewriter for a vehicle diagnostics assistant. "
            "Your only job is to rewrite a follow-up question into a fully self-contained question "
            "by replacing pronouns and vague references with the specific values or terms from the conversation history. "
            "Return ONLY the rewritten question as a single sentence. No explanation, no prefix, no punctuation changes beyond what is needed."
        )
        user_prompt = (
            f"Conversation history:\n{memory_context}\n\n"
            f"Follow-up question: {question}\n\n"
            "Rewritten question:"
        )

        try:
            rewritten = self._chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            ).strip()
            # Reject the rewrite if the model returned something clearly wrong
            if rewritten and len(rewritten) < 300 and "\n" not in rewritten:
                return rewritten
        except Exception:
            pass

        return question

    def route_and_run(
        self,
        *,
        question: str,
        csv_path: str | Path | None = None,
        session_id: str | None = None,
        pal_model: str = "granite-code:8b",
        rag_model: str = "granite3.3",
        docs_dir: str | Path = "knowledge/diagnostics/fault_codes_database.md",
        top_k: int = 4,
    ) -> OrchestratorResult:
        original_question = question
        if session_id:
            question = self._rewrite_if_referential(question, session_id)

        decision = self.decide_route(question=question, csv_path=csv_path, session_id=session_id)
        decision.rewritten_question = question if question != original_question else original_question

        # Extract conversation context for agents if session exists
        conversation_context = ""
        if session_id:
            conversation_context = self.memory.format_recent_context(
                session_id=session_id,
                max_turns=3,  # Keep agent context tight (3 prior turns)
                max_chars_per_answer=200,  # Shorter answers in agent context
            )

        if decision.route == "pal":
            if not csv_path:
                raise ValueError("PAL route selected but csv_path was not provided.")
            response = run_pal_agent(
                csv_path=csv_path,
                question=question,
                model=pal_model,
                conversation_context=conversation_context,
            )
        else:
            response = run_rag_agent(
                question=question,
                docs_dir=docs_dir,
                model=rag_model,
                top_k=top_k,
                conversation_context=conversation_context,
            )

        if session_id:
            self.memory.append_turn(
                session_id=session_id,
                question=question,
                selected_route=decision.route,
                answer=self._extract_answer_text(response),
                csv_path=str(csv_path) if csv_path else None,
                rationale=decision.rationale,
            )

        return OrchestratorResult(decision=decision, response=response)


def orchestrate_question(
    *,
    question: str,
    csv_path: str | Path | None = None,
    session_id: str | None = "default",
    router_model: str = "granite3.3",
    pal_model: str = "granite-code:8b",
    rag_model: str = "granite3.3",
    docs_dir: str | Path = "knowledge/diagnostics/fault_codes_database.md",
    top_k: int = 4,
) -> OrchestratorResult:
    orchestrator = OllamaOrchestrator(model=router_model)
    return orchestrator.route_and_run(
        question=question,
        csv_path=csv_path,
        session_id=session_id,
        pal_model=pal_model,
        rag_model=rag_model,
        docs_dir=docs_dir,
        top_k=max(1, top_k),
    )
