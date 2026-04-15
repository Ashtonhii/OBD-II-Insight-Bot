from __future__ import annotations

import argparse

from ollama_rag import ask_vehicle_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask vehicle diagnostics questions using a RAG pipeline with Ollama."
    )
    parser.add_argument("--question", required=True, help="Diagnostics question to answer")
    parser.add_argument(
        "--docs-dir",
        default="knowledge/diagnostics/fault_codes_database.md",
        help="Diagnostics knowledge path: either a .md/.txt file or a directory of .md/.txt files",
    )
    parser.add_argument("--model", default="granite3.3", help="Ollama model name")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks that were used to answer",
    )

    args = parser.parse_args()

    result = ask_vehicle_diagnostics(
        question=args.question,
        docs_dir=args.docs_dir,
        model=args.model,
        top_k=max(1, args.top_k),
    )

    if args.show_context:
        print("\n=== Retrieved context ===")
        for chunk in result.contexts:
            print(f"\n[{chunk.source.name} :: chunk {chunk.chunk_id}]")
            print(chunk.text)

    print("\n=== Answer ===")
    print(result.answer)


if __name__ == "__main__":
    main()
