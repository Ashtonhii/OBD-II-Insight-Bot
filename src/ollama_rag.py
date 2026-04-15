from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, request


WORD_PATTERN = re.compile(r"[a-z0-9_]+")
SUPPORTED_DOC_SUFFIXES = {".txt", ".md"}


@dataclass
class RetrievalChunk:
    source: Path
    chunk_id: int
    text: str


@dataclass
class RAGResult:
    answer: str
    contexts: list[RetrievalChunk]


class OllamaDiagnosticsRAG:
    def __init__(
        self,
        model: str = "granite3.3",
        base_url: str = "http://localhost:11434",
        docs_dir: str | Path = "knowledge/diagnostics",
        chunk_size: int = 900,
        chunk_overlap: int = 180,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.docs_dir = Path(docs_dir)
        self.chunk_size = max(chunk_size, 250)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def _chat(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
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
    def _tokenize(text: str) -> list[str]:
        return WORD_PATTERN.findall(text.lower())

    def _discover_docs(self) -> list[Path]:
        if not self.docs_dir.exists():
            raise FileNotFoundError(
                f"Diagnostics knowledge directory not found: '{self.docs_dir}'. "
                "Create it and add .md/.txt files."
            )

        files = [
            path
            for path in self.docs_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOC_SUFFIXES
        ]
        if not files:
            raise FileNotFoundError(
                f"No .md/.txt documents found under '{self.docs_dir}'."
            )
        return sorted(files)

    def _chunk_text(self, text: str) -> list[str]:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return []

        chunks: list[str] = []
        start = 0
        n = len(clean)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunk = clean[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _build_chunks(self, files: Iterable[Path]) -> list[RetrievalChunk]:
        all_chunks: list[RetrievalChunk] = []
        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = self._chunk_text(text)
            for index, chunk in enumerate(chunks, start=1):
                all_chunks.append(RetrievalChunk(source=file_path, chunk_id=index, text=chunk))
        if not all_chunks:
            raise ValueError("Documents were found, but no non-empty text chunks were produced.")
        return all_chunks

    def _tfidf_vectors(self, chunks: list[RetrievalChunk]) -> tuple[list[Counter[str]], dict[str, float]]:
        tokenized_docs: list[list[str]] = [self._tokenize(chunk.text) for chunk in chunks]
        term_freqs: list[Counter[str]] = [Counter(tokens) for tokens in tokenized_docs]

        doc_freq: Counter[str] = Counter()
        for token_set in (set(tokens) for tokens in tokenized_docs):
            doc_freq.update(token_set)

        total_docs = len(chunks)
        idf = {
            term: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for term, freq in doc_freq.items()
        }
        return term_freqs, idf

    @staticmethod
    def _norm(weighted: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in weighted.values()))

    def _retrieve(self, question: str, chunks: list[RetrievalChunk], top_k: int = 4) -> list[RetrievalChunk]:
        term_freqs, idf = self._tfidf_vectors(chunks)

        query_tf = Counter(self._tokenize(question))
        if not query_tf:
            return chunks[:top_k]

        query_vec = {term: count * idf.get(term, 0.0) for term, count in query_tf.items()}
        query_norm = self._norm(query_vec)
        if query_norm == 0:
            return chunks[:top_k]

        scored: list[tuple[float, int]] = []
        for idx, tf in enumerate(term_freqs):
            doc_vec = {term: count * idf.get(term, 0.0) for term, count in tf.items()}
            doc_norm = self._norm(doc_vec)
            if doc_norm == 0:
                continue

            dot = sum(query_vec.get(term, 0.0) * weight for term, weight in doc_vec.items())
            similarity = dot / (query_norm * doc_norm)
            if similarity > 0:
                scored.append((similarity, idx))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return chunks[:top_k]

        selected_indices = [idx for _, idx in scored[: max(1, top_k)]]
        return [chunks[idx] for idx in selected_indices]

    def _build_prompt(self, question: str, contexts: list[RetrievalChunk], conversation_context: str = "") -> list[dict[str, str]]:
        context_blocks = []
        for chunk in contexts:
            context_blocks.append(
                f"Source: {chunk.source.name} (chunk {chunk.chunk_id})\n{chunk.text}"
            )

        system_prompt = (
            "You are a vehicle diagnostics assistant. "
            "Answer only from the provided context. "
            "If context is insufficient, say what is missing and avoid fabricating details."
        )

        conversation_block = ""
        if conversation_context.strip():
            conversation_block = f"Prior conversation context:\n{conversation_context}\n\n"

        user_prompt = (
            "Retrieved knowledge base context:\n\n"
            f"{'\n\n---\n\n'.join(context_blocks)}\n\n"
            f"{conversation_block}"
            f"Question: {question}\n\n"
            "Return a practical, concise answer for a technician."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def ask(self, question: str, top_k: int = 4, conversation_context: str = "") -> RAGResult:
        docs = self._discover_docs()
        chunks = self._build_chunks(docs)
        contexts = self._retrieve(question=question, chunks=chunks, top_k=top_k)

        messages = self._build_prompt(question=question, contexts=contexts, conversation_context=conversation_context)
        answer = self._chat(messages, temperature=0.1).strip()

        return RAGResult(answer=answer, contexts=contexts)


def ask_vehicle_diagnostics(
    question: str,
    docs_dir: str | Path = "knowledge/diagnostics",
    model: str = "granite3.3",
    top_k: int = 4,
    conversation_context: str = "",
) -> RAGResult:
    rag = OllamaDiagnosticsRAG(model=model, docs_dir=docs_dir)
    return rag.ask(question=question, top_k=top_k, conversation_context=conversation_context)
