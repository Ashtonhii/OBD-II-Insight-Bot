from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_pal import ask_question_on_csv


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _score_answer(expected_answer: str, model_answer: str, tolerance: float) -> tuple[str, int, str]:
    expected_number = _to_float(expected_answer)
    if expected_number is not None:
        model_number = _to_float(model_answer)
        if model_number is None:
            return "", 0, "numeric_parse_failed"

        is_match = abs(model_number - expected_number) <= tolerance
        normalized = f"{model_number:.6f}".rstrip("0").rstrip(".")
        note = "numeric_match" if is_match else f"numeric_mismatch(delta={abs(model_number - expected_number):.6f})"
        return normalized, int(is_match), note

    normalized_expected = _normalize_text(expected_answer)
    normalized_model = _normalize_text(model_answer)
    is_match = normalized_model == normalized_expected
    note = "text_match" if is_match else "text_mismatch"
    return normalized_model, int(is_match), note


def evaluate_pal(
    golden_csv: Path,
    output_csv: Path,
    tolerance: float,
    limit: int | None = None,
    row_id: int | None = None,
    row_number: int | None = None,
) -> tuple[float, float, int, int]:
    golden_df = pd.read_csv(golden_csv)

    if row_id is not None:
        filtered = golden_df[golden_df["id"] == row_id]
        if filtered.empty:
            raise ValueError(f"No row found with id={row_id} in {golden_csv}")
        golden_df = filtered

    if row_number is not None:
        if row_number <= 0:
            raise ValueError("row_number must be >= 1")
        if row_number > len(golden_df):
            raise ValueError(
                f"row_number={row_number} is out of range for filtered dataset of size {len(golden_df)}"
            )
        golden_df = golden_df.iloc[[row_number - 1]]

    if limit is not None and limit > 0:
        golden_df = golden_df.head(limit)

    results: list[dict[str, Any]] = []

    for row in golden_df.itertuples(index=False):
        record = {
            "dataset_file": str(row.dataset_file),
            "question": str(row.question),
            "expected_answer": str(row.expected_answer),
            "model_raw_answer": "",
            "pal_code": "",
            "normalized_model_answer": "",
            "execution_success": 0,
            "execution_match": 0,
            "notes": "",
        }

        try:
            result = ask_question_on_csv(
                csv_path=record["dataset_file"],
                question=record["question"],
                answer_only=True,
            )
            record["model_raw_answer"] = result.answer
            record["pal_code"] = result.code
            record["execution_success"] = 1
            normalized_answer, execution_match, note = _score_answer(
                expected_answer=record["expected_answer"],
                model_answer=result.answer,
                tolerance=tolerance,
            )
            record["normalized_model_answer"] = normalized_answer
            record["execution_match"] = execution_match
            record["notes"] = note
        except Exception as exc:
            record["notes"] = f"execution_error: {str(exc)[:300]}"

        results.append(record)
        print(
            f"[{record['dataset_file']}] exec={record['execution_success']} em={record['execution_match']} "
            f"file={record['dataset_file']}"
        )

    result_df = pd.DataFrame(results)
    result_df = result_df[
        [
            "dataset_file",
            "question",
            "expected_answer",
            "model_raw_answer",
            "pal_code",
            "normalized_model_answer",
            "execution_success",
            "execution_match",
            "notes",
        ]
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    total = len(result_df)
    execution_success_count = int(result_df["execution_success"].sum())
    exact_match_count = int(result_df["execution_match"].sum())

    esr = (execution_success_count / total) if total else 0.0
    em = (exact_match_count / execution_success_count) if execution_success_count else 0.0
    return esr, em, total, execution_success_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PAL pipeline and compute ESR/EM.")
    parser.add_argument(
        "--golden-csv",
        default="data/golden/pal_golden_dataset.csv",
        help="Path to golden dataset CSV",
    )
    parser.add_argument(
        "--output-csv",
        default="data/golden/pal_eval_results.csv",
        help="Path to write evaluation results CSV",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Numeric tolerance for exact match")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N questions (0 = all)")
    parser.add_argument(
        "--row-id",
        type=int,
        default=None,
        help="Evaluate only the row with this dataset id value",
    )
    parser.add_argument(
        "--row-number",
        type=int,
        default=None,
        help="Evaluate only this 1-based row position (after optional --row-id filtering)",
    )
    args = parser.parse_args()

    if args.row_id is not None and args.row_number is not None:
        parser.error("Use either --row-id or --row-number, not both.")

    esr, em, total, success = evaluate_pal(
        golden_csv=Path(args.golden_csv),
        output_csv=Path(args.output_csv),
        tolerance=args.tolerance,
        limit=args.limit if args.limit > 0 else None,
        row_id=args.row_id,
        row_number=args.row_number,
    )

    print("\n=== PAL Evaluation Summary ===")
    print(f"Questions evaluated: {total}")
    print(f"Execution successes: {success}")
    print(f"ESR: {esr:.4f}")
    print(f"EM: {em:.4f}")
    print(f"Results CSV: {Path(args.output_csv)}")


if __name__ == "__main__":
    main()
