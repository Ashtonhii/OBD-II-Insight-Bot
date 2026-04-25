from __future__ import annotations

import argparse

from ollama_orchestrator import orchestrate_question


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified multi-agent CLI. Routes question to PAL (data) or RAG (diagnostics)."
    )
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument(
        "--session-id",
        required=True,
        help="Conversation session id used for orchestrator memory",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path for data questions (required if PAL route is needed)",
    )
    parser.add_argument(
        "--docs-dir",
        default="knowledge/diagnostics/fault_codes_database.md",
        help="Directory containing diagnostics .md/.txt docs for RAG",
    )
    parser.add_argument(
        "--router-model",
        default="granite3.3",
        help="Ollama model for orchestration routing",
    )
    parser.add_argument("--pal-model", default="granite-code:8b", help="Ollama model for PAL agent")
    parser.add_argument("--rag-model", default="granite3.3", help="Ollama model for RAG agent")
    parser.add_argument("--top-k", type=int, default=4, help="RAG retrieval chunk count")
    parser.add_argument("--show-route", action="store_true", help="Print route and rationale")
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print PAL code/result preview or RAG retrieved contexts",
    )

    args = parser.parse_args()

    result = orchestrate_question(
        question=args.question,
        csv_path=args.csv,
        session_id=args.session_id,
        router_model=args.router_model,
        pal_model=args.pal_model,
        rag_model=args.rag_model,
        docs_dir=args.docs_dir,
        top_k=max(1, args.top_k),
    )

    if args.show_route:
        if result.decision.rewritten_question != args.question:
            print("\n=== Rewritten question ===")
            print(result.decision.rewritten_question)
        print("\n=== Route ===")
        print(result.decision.route)
        print("\n=== Rationale ===")
        print(result.decision.rationale)

    payload = result.response.payload

    if result.response.agent == "pal":
        if args.show_details:
            print("\n=== Generated pandas code ===")
            print(payload.code)
            print("\n=== Result preview ===")
            print(payload.result_preview)

        print("\n=== Answer ===")
        print(payload.answer)
        return

    if args.show_details:
        print("\n=== Retrieved context ===")
        for chunk in payload.contexts:
            print(f"\n[{chunk.source.name} :: chunk {chunk.chunk_id}]")
            print(chunk.text)

    print("\n=== Answer ===")
    print(payload.answer)


if __name__ == "__main__":
    main()
