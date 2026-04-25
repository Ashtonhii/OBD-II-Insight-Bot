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
DEFAULT_TARGET_COUNT = 120
DEFAULT_CODES_TO_USE = 60

CODE_PATTERN = re.compile(r"\b([PCBU][0-9A-F]{4})\b", re.IGNORECASE)

# Each mode is (mode_name, question_template, expected_field)
# question_template is a callable(code, description) -> str
QUESTION_MODES: list[tuple[str, object, str]] = [
    # Tier 1: exact code lookup — exercises the exact DTC match retrieval path
    ("code_to_description",      lambda code, desc: code,                                         "description"),
    # Tier 2: exact description lookup — exercises TF-IDF path (must find code from prose)
    ("description_to_code",      lambda code, desc: desc,                                         "code"),
    # Tier 3: natural-language "what does X mean" — realistic user phrasing, code in question
    ("nl_what_does_mean",        lambda code, desc: f"What does {code} mean?",                    "description"),
    # Tier 4: natural-language "what is X" — short-form query, code in question
    ("nl_what_is",               lambda code, desc: f"What is {code}?",                           "description"),
    # Tier 5: natural-language "what causes X" — tests whether the description contains the code
    ("nl_what_causes",           lambda code, desc: f"What causes fault code {code}?",            "description"),
    # Tier 6: keyword fragment — retriever must identify the correct code from 2-3 meaningful keywords
    ("keyword_fragment",         lambda code, desc: _keyword_fragment(desc),                       "code"),
]


def _keyword_fragment(text: str) -> str:
    """Extract a meaningful 2-4 word keyword fragment from the description,
    skipping generic circuit/sensor/system words that appear in many entries.
    This produces a partial query that is distinct from the full description
    but still specific enough to identify the correct DTC code via TF-IDF retrieval.
    """
    skip = {
        "circuit", "control", "system", "malfunction", "range", "performance",
        "sensor", "input", "output", "open", "low", "high", "bank", "a", "b",
        "the", "or", "and", "of", "with", "for", "in", "1", "2", "3",
    }
    words = text.split()
    keywords = [w for w in words if w.lower().rstrip("/(,)") not in skip and len(w) > 2]
    if len(keywords) >= 2:
        return " ".join(keywords[:3])
    # Fallback: first 4 words if no meaningful keywords found
    return " ".join(words[:4]) if len(words) >= 4 else text


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
                description = _normalize_space(" ".join(p for p in current_desc_parts if p.strip()))
                if description:
                    entries.append({"code": current_code, "description": description})
            if entries:
                break
            continue

        match = CODE_PATTERN.match(line)
        if match:
            if current_code:
                description = _normalize_space(" ".join(p for p in current_desc_parts if p.strip()))
                if description:
                    entries.append({"code": current_code, "description": description})

            current_code = match.group(1).upper()
            remainder = line[match.end():].strip(" -:\t")
            current_desc_parts = [remainder] if remainder else []
            continue

        if current_code:
            current_desc_parts.append(line)

    if current_code:
        description = _normalize_space(" ".join(p for p in current_desc_parts if p.strip()))
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
        for mode_index, (mode_name, q_fn, expected_field) in enumerate(QUESTION_MODES):
            if len(rows) >= target_count:
                return rows

            question = q_fn(entry["code"], entry["description"])
            expected_answer = entry[expected_field]

            # Skip degenerate keyword fragments that are too short to be meaningful
            if mode_name == "keyword_fragment" and len(question.split()) < 2:
                continue

            rows.append(
                {
                    "dtc_code": entry["code"],
                    "question": question,
                    "expected_answer": expected_answer,
                    "docs_path": str(source_doc.relative_to(PROJECT_ROOT)),
                    "template_id": str(mode_index + 1),
                    "question_mode": mode_name,
                    "code_index": str(entry_index + 1),
                }
            )

    # If we still need more rows, cycle through modes again
    if len(rows) < target_count:
        cursor = 0
        while len(rows) < target_count:
            entry = selected_entries[cursor % len(selected_entries)]
            mode_index = cursor % len(QUESTION_MODES)
            mode_name, q_fn, expected_field = QUESTION_MODES[mode_index]
            question = q_fn(entry["code"], entry["description"])
            expected_answer = entry[expected_field]
            rows.append(
                {
                    "dtc_code": entry["code"],
                    "question": question,
                    "expected_answer": expected_answer,
                    "docs_path": str(source_doc.relative_to(PROJECT_ROOT)),
                    "template_id": str(mode_index + 1),
                    "question_mode": mode_name,
                    "code_index": str((cursor % len(selected_entries)) + 1),
                }
            )
            cursor += 1

    return rows[:target_count]


def generate_rag_golden_dataset(
    *,
    source_doc: Path = DEFAULT_SOURCE_DOC,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    target_count: int = DEFAULT_TARGET_COUNT,
    codes_to_use: int = DEFAULT_CODES_TO_USE,
) -> tuple[Path, int]:
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

    return output_csv, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a golden dataset for evaluating the RAG pipeline.")
    parser.add_argument("--source-doc", default=str(DEFAULT_SOURCE_DOC))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--codes-to-use", type=int, default=DEFAULT_CODES_TO_USE)
    args = parser.parse_args()

    csv_path, count = generate_rag_golden_dataset(
        source_doc=Path(args.source_doc),
        output_csv=Path(args.output_csv),
        target_count=max(1, args.target_count),
        codes_to_use=max(1, args.codes_to_use),
    )

    print(f"Generated {count} RAG golden QA rows")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
