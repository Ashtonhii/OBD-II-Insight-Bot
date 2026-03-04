from __future__ import annotations

import argparse

from src.ollama_pal import ask_question_on_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over OBD CSV logs using Ollama PAL.")
    parser.add_argument("--csv", required=True, help="Path to the OBD CSV file")
    parser.add_argument("--question", required=True, help="Question about the dataset")
    parser.add_argument("--model", default="granite-code:8b", help="Ollama model name")
    parser.add_argument(
        "--show-code",
        action="store_true",
        help="Print generated pandas code before the final answer",
    )
    args = parser.parse_args()

    result = ask_question_on_csv(csv_path=args.csv, question=args.question, model=args.model)

    if args.show_code:
        print("\n=== Generated pandas code ===")
        print(result.code)

    print("\n=== Answer ===")
    print(result.answer)

    print("\n=== Result preview ===")
    print(result.result_preview)


if __name__ == "__main__":
    main()
