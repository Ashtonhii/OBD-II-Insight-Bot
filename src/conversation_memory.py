from __future__ import annotations

import os
import json
import re
from datetime import datetime, timezone
from typing import Any

import redis


_SAFE_SESSION_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


class ConversationMemory:
    def __init__(
        self,
        redis_url: str | None = None,
        key_prefix: str = "obd:orchestrator:memory:",
    ) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = str(key_prefix).strip() or "obd:orchestrator:memory:"
        self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)

        try:
            self.client.ping()
        except redis.RedisError as exc:
            raise ConnectionError(
                f"Could not connect to Redis at '{self.redis_url}'. "
                "Set REDIS_URL or start Redis before running the orchestrator."
            ) from exc

    @staticmethod
    def _sanitize_session_id(session_id: str) -> str:
        cleaned = _SAFE_SESSION_PATTERN.sub("_", str(session_id).strip())
        return cleaned or "default"

    def _session_key(self, session_id: str) -> str:
        safe_id = self._sanitize_session_id(session_id)
        return f"{self.key_prefix}{safe_id}"

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        key = self._session_key(session_id)
        raw = self.client.get(key)
        if not raw:
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        turns = payload.get("turns", [])
        if not isinstance(turns, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in turns:
            if not isinstance(item, dict):
                continue
            normalized.append(item)
        return normalized

    def append_turn(
        self,
        *,
        session_id: str,
        question: str,
        selected_route: str,
        answer: str,
        csv_path: str | None,
        rationale: str,
        max_turns_to_keep: int = 50,
    ) -> None:
        turns = self.load_session(session_id)

        turns.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "question": str(question).strip(),
                "selected_route": str(selected_route).strip().lower(),
                "csv_path": (str(csv_path).strip() if csv_path else ""),
                "router_rationale": str(rationale).strip(),
                "answer": str(answer).strip(),
            }
        )

        if max_turns_to_keep > 0 and len(turns) > max_turns_to_keep:
            turns = turns[-max_turns_to_keep:]

        payload = {"session_id": self._sanitize_session_id(session_id), "turns": turns}
        self.client.set(self._session_key(session_id), json.dumps(payload, ensure_ascii=False))

    def format_recent_context(self, session_id: str, max_turns: int = 5, max_chars_per_answer: int = 280) -> str:
        turns = self.load_session(session_id)
        if not turns:
            return "No prior conversation memory."

        selected = turns[-max_turns:] if max_turns > 0 else turns
        lines: list[str] = []
        for idx, turn in enumerate(selected, start=1):
            question = str(turn.get("question", "")).strip()
            route = str(turn.get("selected_route", "")).strip().lower() or "unknown"
            answer = str(turn.get("answer", "")).strip()
            csv_path = str(turn.get("csv_path", "")).strip()

            if max_chars_per_answer > 0 and len(answer) > max_chars_per_answer:
                answer = answer[:max_chars_per_answer].rstrip() + "..."

            lines.append(f"Turn {idx} question: {question}")
            lines.append(f"Turn {idx} route: {route}")
            if csv_path:
                lines.append(f"Turn {idx} csv_path: {csv_path}")
            lines.append(f"Turn {idx} answer: {answer}")

        return "\n".join(lines)
