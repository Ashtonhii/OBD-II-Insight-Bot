from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ollama_orchestrator import OllamaOrchestrator


VALID_ROUTES = {"pal", "rag"}
VALID_CATEGORIES = {"unambiguous", "hybrid", "follow_up"}
DEFAULT_EVAL_CSV = "data/obdiidata/drive1.csv"


def _normalize_route(value: Any) -> str:
    return str(value).strip().lower()


def _clean_optional_text(value: Any) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def evaluate_router(
    golden_csv: Path,
    output_csv: Path,
    router_model: str,
    default_csv: str | None = DEFAULT_EVAL_CSV,
    limit: int | None = None,
) -> tuple[float, float, int, int]:
    df = pd.read_csv(golden_csv, engine="python")
    if limit is not None and limit > 0:
        df = df.head(limit)

    required_columns = {"id", "question", "expected_route"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Golden router dataset is missing required columns: {missing}")

    has_csv_path_column = "csv_path" in df.columns
    has_category_column = "prompt_category" in df.columns
    orchestrator = OllamaOrchestrator(model=router_model)

    rows: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()

        expected_route = _normalize_route(row_dict.get("expected_route"))
        prompt_category = str(row_dict.get("prompt_category", "unambiguous")).strip().lower() if has_category_column else "unambiguous"
        csv_path = (
            _clean_optional_text(row_dict.get("csv_path")) if has_csv_path_column else None
        )
        if not csv_path:
            csv_path = _clean_optional_text(default_csv)
        if not csv_path:
            raise ValueError(
                "Router evaluation requires a CSV for every row. "
                "Provide --default-csv or include csv_path in the golden CSV."
            )

        record = {
            "id": int(row_dict.get("id")),
            "question": str(row_dict.get("question")),
            "expected_route": expected_route,
            "prompt_category": prompt_category,
            "predicted_route": "",
            "csv_path_used": csv_path or "",
            "is_correct": 0,
            "execution_success": 0,
            "router_rationale": "",
            "notes": "",
        }

        if expected_route not in VALID_ROUTES:
            record["notes"] = f"invalid_expected_route:{expected_route}"
            rows.append(record)
            print(f"[{record['id']}] skipped invalid label route={expected_route}")
            continue

        try:
            decision = orchestrator.decide_route(
                question=record["question"],
                csv_path=csv_path,
            )
            record["predicted_route"] = decision.route
            record["router_rationale"] = decision.rationale
            record["execution_success"] = 1
            record["is_correct"] = int(decision.route == expected_route)
            record["notes"] = "ok"
        except Exception as exc:
            record["notes"] = f"router_error:{str(exc)[:300]}"

        rows.append(record)
        print(
            f"[{record['id']}] cat={record['prompt_category']} success={record['execution_success']} "
            f"expected={record['expected_route']} predicted={record['predicted_route'] or '-'} "
            f"correct={record['is_correct']}"
        )

    result_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    total = len(result_df)
    success_count = int(result_df["execution_success"].sum()) if total else 0
    correct_count = int(result_df["is_correct"].sum()) if total else 0

    accuracy_all = (correct_count / total) if total else 0.0
    accuracy_success_only = (correct_count / success_count) if success_count else 0.0
    return accuracy_all, accuracy_success_only, total, success_count


def _print_confusion_table(results_csv: Path) -> None:
    result_df = pd.read_csv(results_csv)
    successful = result_df[result_df["execution_success"] == 1]

    if successful.empty:
        print("\nNo successful routing records to build confusion table.")
        return

    confusion = pd.crosstab(
        successful["expected_route"],
        successful["predicted_route"],
        rownames=["expected"],
        colnames=["predicted"],
        dropna=False,
    )
    print("\n=== Router Confusion Table (successes only) ===")
    print(confusion.to_string())

    # Asymmetric default analysis
    # False positive: expected=rag, predicted=pal  → hard failure (PAL without semantic content)
    # False negative: expected=pal, predicted=rag  → degraded response (RAG answers a data question)
    fp = int(confusion.loc["rag", "pal"]) if ("rag" in confusion.index and "pal" in confusion.columns) else 0
    fn = int(confusion.loc["pal", "rag"]) if ("pal" in confusion.index and "rag" in confusion.columns) else 0
    print("\n=== Asymmetric Default Analysis ===")
    print(f"False positives (RAG→PAL, hard failure):    {fp}")
    print(f"False negatives (PAL→RAG, degraded response): {fn}")
    if fp == 0 and fn == 0:
        print("No misclassifications — default rule not exercised.")
    elif fp <= fn:
        print("Default rule is working as intended: RAG→PAL errors are rarer than PAL→RAG errors.")
    else:
        print("WARNING: RAG→PAL errors exceed PAL→RAG errors — default rule is not suppressing hard failures.")


def _print_category_breakdown(results_csv: Path) -> None:
    result_df = pd.read_csv(results_csv)
    successful = result_df[result_df["execution_success"] == 1].copy()

    if successful.empty:
        print("\nNo successful routing records to break down by category.")
        return

    print("\n=== Accuracy by Prompt Category (successes only) ===")

    categories = ["unambiguous", "hybrid", "follow_up"]
    rows = []
    for cat in categories:
        subset = successful[successful["prompt_category"] == cat]
        if subset.empty:
            rows.append({"category": cat, "total": 0, "correct": 0, "accuracy": "n/a"})
            continue
        total = len(subset)
        correct = int(subset["is_correct"].sum())
        accuracy = correct / total
        rows.append({"category": cat, "total": total, "correct": correct, "accuracy": f"{accuracy:.4f}"})

    breakdown_df = pd.DataFrame(rows)
    print(breakdown_df.to_string(index=False))
    print()
    print("Note: 'hybrid' prompts mix quantitative and qualitative framing.")
    print("      'follow_up' prompts require session context to resolve routing intent.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate orchestrator routing accuracy on labeled prompts.")
    parser.add_argument(
        "--golden-csv",
        default="data/golden/router_golden_dataset.csv",
        help="Path to router labeled prompt dataset CSV",
    )
    parser.add_argument(
        "--output-csv",
        default="data/golden/router_eval_results.csv",
        help="Path to write router evaluation results CSV",
    )
    parser.add_argument(
        "--router-model",
        default="granite3.3",
        help="Ollama model for routing decisions",
    )
    parser.add_argument(
        "--default-csv",
        default=DEFAULT_EVAL_CSV,
        help="CSV path attached to router evaluation rows when csv_path is missing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only first N rows (0 = all)",
    )
    args = parser.parse_args()

    accuracy_all, accuracy_success_only, total, success = evaluate_router(
        golden_csv=Path(args.golden_csv),
        output_csv=Path(args.output_csv),
        router_model=args.router_model,
        default_csv=_clean_optional_text(args.default_csv),
        limit=args.limit if args.limit > 0 else None,
    )

    print("\n=== Router Evaluation Summary ===")
    print(f"Prompts evaluated: {total}")
    print(f"Successful decisions: {success}")
    print(f"Accuracy (all rows): {accuracy_all:.4f}")
    print(f"Accuracy (success-only): {accuracy_success_only:.4f}")
    print(f"Results CSV: {Path(args.output_csv)}")

    _print_confusion_table(Path(args.output_csv))
    _print_category_breakdown(Path(args.output_csv))


if __name__ == "__main__":
    main()
