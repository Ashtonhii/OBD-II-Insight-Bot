from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SOURCE_DOC = PROJECT_ROOT / "knowledge" / "diagnostics" / "fault_codes_database.md"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "golden" / "rag_golden_dataset.csv"
DEFAULT_OUTPUT_TXT = PROJECT_ROOT / "data" / "golden" / "rag_golden_answers.txt"
DEFAULT_TARGET_COUNT = 100
DEFAULT_CODES_TO_USE = 50

CODE_PATTERN = re.compile(r"\b([PCBU][0-9A-F]{4})\b", re.IGNORECASE)

QUESTION_MODES = (
    "code_to_description",
    "description_to_code",
)


def _normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def _parse_code_entries(source_doc: Path) -> list[dict[str, str]]:
    if not source_doc.exists():
        raise FileNotFoundError(f"Diagnostics source document not found: {source_doc}")

    lines = source_doc.read_text(encoding="utf-8", errors="ignore").splitlines()
    entries: list[dict[str, str]] = []

    current_code = ""
    current_desc_parts: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            if current_code:
                description = _normalize_space(" ".join(part for part in current_desc_parts if part.strip()))
                if description:
                    entries.append({"code": current_code, "description": description})
            if entries:
                break
            continue

        match = CODE_PATTERN.match(line)
        if match:
            if current_code:
                description = _normalize_space(" ".join(part for part in current_desc_parts if part.strip()))
                if description:
                    entries.append({"code": current_code, "description": description})

            current_code = match.group(1).upper()
            remainder = line[match.end():].strip(" -:\t")
            current_desc_parts = [remainder] if remainder else []
            continue

        if current_code:
            current_desc_parts.append(line)

    if current_code:
        description = _normalize_space(" ".join(part for part in current_desc_parts if part.strip()))
        if description:
            entries.append({"code": current_code, "description": description})

    if not entries:
        raise ValueError(f"No code descriptions could be parsed from {source_doc}")

    return entries


def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    count = max(1, min(count, total))
    if count == total:
        return list(range(total))

    if count == 1:
        return [0]

    step = (total - 1) / float(count - 1)
    indices: list[int] = []
    seen: set[int] = set()
    for i in range(count):
        idx = int(round(i * step))
        idx = max(0, min(total - 1, idx))
        while idx in seen and idx < total - 1:
            idx += 1
        while idx in seen and idx > 0:
            idx -= 1
        if idx in seen:
            break
        seen.add(idx)
        indices.append(idx)
    return indices


def _build_rows(source_doc: Path, codes_to_use: int, target_count: int) -> list[dict[str, str]]:
    entries = _parse_code_entries(source_doc)
    selected_indices = _evenly_spaced_indices(len(entries), codes_to_use)
    selected_entries = [entries[idx] for idx in selected_indices]

    rows: list[dict[str, str]] = []
    for entry_index, entry in enumerate(selected_entries):
        for mode_index, mode in enumerate(QUESTION_MODES):
            if len(rows) >= target_count:
                return rows

            if mode == "code_to_description":
                question = entry["code"]
                expected_answer = entry["description"]
            else:
                question = entry["description"]
                expected_answer = entry["code"]

            rows.append(
                {
                    "dtc_code": entry["code"],
                    "question": question,
                    "expected_answer": expected_answer,
                    "docs_path": str(source_doc.relative_to(PROJECT_ROOT)),
                    "template_id": str(mode_index + 1),
                    "question_mode": mode,
                    "code_index": str(entry_index + 1),
                }
            )

    if len(rows) < target_count:
        cursor = 0
        while len(rows) < target_count:
            entry = selected_entries[cursor % len(selected_entries)]
            mode = QUESTION_MODES[cursor % len(QUESTION_MODES)]
            if mode == "code_to_description":
                question = entry["code"]
                expected_answer = entry["description"]
            else:
                question = entry["description"]
                expected_answer = entry["code"]

            rows.append(
                {
                    "dtc_code": entry["code"],
                    "question": question,
                    "expected_answer": expected_answer,
                    "docs_path": str(source_doc.relative_to(PROJECT_ROOT)),
                    "template_id": str((cursor % len(QUESTION_MODES)) + 1),
                    "question_mode": mode,
                    "code_index": str((cursor % len(selected_entries)) + 1),
                }
            )
            cursor += 1

    return rows[:target_count]


def generate_rag_golden_dataset(
    *,
    source_doc: Path = DEFAULT_SOURCE_DOC,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    target_count: int = DEFAULT_TARGET_COUNT,
    codes_to_use: int = DEFAULT_CODES_TO_USE,
) -> tuple[Path, Path, int]:
    rows = _build_rows(source_doc, codes_to_use=codes_to_use, target_count=target_count)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "dtc_code",
        "question",
        "expected_answer",
        "docs_path",
        "template_id",
        "question_mode",
        "code_index",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow({"id": idx, **row})

    lines: list[str] = []
    for idx, row in enumerate(rows, start=1):
        lines.append(f"Q{idx}: [{row['dtc_code']}] {row['question']}")
        lines.append(f"A{idx}: {row['expected_answer']}")
        lines.append("")
    output_txt.write_text("\n".join(lines), encoding="utf-8")

    return output_csv, output_txt, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a golden dataset for evaluating the RAG pipeline.")
    parser.add_argument("--source-doc", default=str(DEFAULT_SOURCE_DOC), help="Diagnostics knowledge file to parse")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Path to write the golden CSV")
    parser.add_argument("--output-txt", default=str(DEFAULT_OUTPUT_TXT), help="Path to write the answer key text file")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT, help="Total number of prompts to generate")
    parser.add_argument("--codes-to-use", type=int, default=DEFAULT_CODES_TO_USE, help="How many distinct DTC codes to sample")
    args = parser.parse_args()

    csv_path, txt_path, count = generate_rag_golden_dataset(
        source_doc=Path(args.source_doc),
        output_csv=Path(args.output_csv),
        output_txt=Path(args.output_txt),
        target_count=max(1, args.target_count),
        codes_to_use=max(1, args.codes_to_use),
    )

    print(f"Generated {count} RAG golden QA rows")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")


if __name__ == "__main__":
    main()