from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ollama_rag import ask_vehicle_diagnostics


DEFAULT_GOLDEN_CSV = PROJECT_ROOT / "data" / "golden" / "rag_golden_dataset.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "golden" / "rag_eval_results.csv"
DEFAULT_DOCS_PATH = PROJECT_ROOT / "knowledge" / "diagnostics" / "fault_codes_database.md"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _clean_optional_text(value: Any) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _score_answer(expected_answer: str, model_answer: str) -> tuple[str, int, str]:
    normalized_expected = _normalize_text(expected_answer)
    normalized_model = _normalize_text(model_answer)

    if not normalized_model:
        return normalized_model, 0, "empty_model_answer"

    if normalized_model == normalized_expected:
        return normalized_model, 1, "exact_match"

    if normalized_expected and normalized_expected in normalized_model:
        return normalized_model, 1, "contains_expected"

    if normalized_model in normalized_expected:
        return normalized_model, 1, "contained_in_expected"

    return normalized_model, 0, "mismatch"


def evaluate_rag(
    golden_csv: Path,
    output_csv: Path,
    docs_path: str | Path = DEFAULT_DOCS_PATH,
    model: str = "granite3.3",
    top_k: int = 4,
    limit: int | None = None,
) -> tuple[float, float, int, int]:
    with golden_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if limit is not None and limit > 0:
        rows = rows[:limit]

    required_columns = {"id", "dtc_code", "question", "expected_answer"}
    missing_columns = required_columns - set(reader.fieldnames or [])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Golden RAG dataset is missing required columns: {missing}")

    results: list[dict[str, Any]] = []

    for row in rows:
        question = str(row.get("question", ""))
        expected_answer = str(row.get("expected_answer", ""))
        row_docs_path = _clean_optional_text(row.get("docs_path")) or str(docs_path)

        record = {
            "id": int(row.get("id", 0)),
            "dtc_code": str(row.get("dtc_code", "")),
            "question": question,
            "expected_answer": expected_answer,
            "model_raw_answer": "",
            "normalized_model_answer": "",
            "docs_path_used": row_docs_path,
            "execution_success": 0,
            "exact_match": 0,
            "notes": "",
        }

        try:
            result = ask_vehicle_diagnostics(
                question=question,
                docs_dir=row_docs_path,
                model=model,
                top_k=top_k,
                answer_only=True,
            )
            record["model_raw_answer"] = result.answer
            record["execution_success"] = 1
            normalized_model, exact_match, note = _score_answer(expected_answer, result.answer)
            record["normalized_model_answer"] = normalized_model
            record["exact_match"] = exact_match
            record["notes"] = note
        except Exception as exc:
            record["notes"] = f"execution_error: {str(exc)[:300]}"

        results.append(record)
        print(
            f"[{record['id']}] exec={record['execution_success']} em={record['exact_match']} code={record['dtc_code']}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "dtc_code",
        "question",
        "expected_answer",
        "model_raw_answer",
        "normalized_model_answer",
        "docs_path_used",
        "execution_success",
        "exact_match",
        "notes",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    total = len(results)
    execution_success_count = sum(int(row["execution_success"]) for row in results)
    exact_match_count = sum(int(row["exact_match"]) for row in results)

    accuracy_all = (exact_match_count / total) if total else 0.0
    accuracy_success_only = (exact_match_count / execution_success_count) if execution_success_count else 0.0
    return accuracy_all, accuracy_success_only, total, execution_success_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on diagnostic code descriptions.")
    parser.add_argument(
        "--golden-csv",
        default=str(DEFAULT_GOLDEN_CSV),
        help="Path to the RAG golden dataset CSV",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to write the RAG evaluation results CSV",
    )
    parser.add_argument(
        "--docs-path",
        default=str(DEFAULT_DOCS_PATH),
        help="Diagnostics knowledge file or directory used by RAG",
    )
    parser.add_argument("--model", default="granite3.3", help="Ollama model for RAG answers")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks to pass into the answerer")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N prompts (0 = all)")
    args = parser.parse_args()

    accuracy_all, accuracy_success_only, total, success = evaluate_rag(
        golden_csv=Path(args.golden_csv),
        output_csv=Path(args.output_csv),
        docs_path=args.docs_path,
        model=args.model,
        top_k=max(1, args.top_k),
        limit=args.limit if args.limit > 0 else None,
    )

    print("\n=== RAG Evaluation Summary ===")
    print(f"Prompts evaluated: {total}")
    print(f"Successful answers: {success}")
    print(f"Accuracy (all rows): {accuracy_all:.4f}")
    print(f"Accuracy (success-only): {accuracy_success_only:.4f}")
    print(f"Results CSV: {Path(args.output_csv)}")


if __name__ == "__main__":
    main()