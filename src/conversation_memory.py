from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_SESSION_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


class ConversationMemory:
    def __init__(self, memory_dir: str | Path = "data/memory/orchestrator") -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_session_id(session_id: str) -> str:
        cleaned = _SAFE_SESSION_PATTERN.sub("_", str(session_id).strip())
        return cleaned or "default"

    def _session_file(self, session_id: str) -> Path:
        safe_id = self._sanitize_session_id(session_id)
        return self.memory_dir / f"{safe_id}.json"

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        path = self._session_file(session_id)
        if not path.exists():
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
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
        self._session_file(session_id).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
