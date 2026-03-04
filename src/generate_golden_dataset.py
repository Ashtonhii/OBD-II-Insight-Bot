from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_and_prep_obd_data


TARGET_FILES = [
    "drive1.csv",
    "idle1.csv",
    "live1.csv",
    "long1.csv",
    "ufpe1.csv",
]


def _format_scalar(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4f}"


def _build_questions_for_file(file_name: str, df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    rows.append(
        {
            "dataset_file": file_name,
            "question": f"How many rows are in {file_name}?",
            "pandas_expression": "len(df)",
            "expected_answer": _format_scalar(len(df)),
        }
    )

    if "engine_rpm" in df.columns:
        rows.append(
            {
                "dataset_file": file_name,
                "question": f"What is the maximum engine rpm in {file_name}?",
                "pandas_expression": "df['engine_rpm'].max()",
                "expected_answer": _format_scalar(float(df["engine_rpm"].max())),
            }
        )
        rows.append(
            {
                "dataset_file": file_name,
                "question": f"What is the average engine rpm in {file_name}?",
                "pandas_expression": "df['engine_rpm'].mean()",
                "expected_answer": _format_scalar(float(df["engine_rpm"].mean())),
            }
        )

    if "vehicle_speed" in df.columns:
        rows.append(
            {
                "dataset_file": file_name,
                "question": f"What is the maximum vehicle speed in {file_name}?",
                "pandas_expression": "df['vehicle_speed'].max()",
                "expected_answer": _format_scalar(float(df["vehicle_speed"].max())),
            }
        )
        rows.append(
            {
                "dataset_file": file_name,
                "question": f"What is the average vehicle speed in {file_name}?",
                "pandas_expression": "df['vehicle_speed'].mean()",
                "expected_answer": _format_scalar(float(df["vehicle_speed"].mean())),
            }
        )
        moving_pct = float((df["vehicle_speed"] > 0).mean() * 100.0)
        rows.append(
            {
                "dataset_file": file_name,
                "question": f"What percentage of rows have vehicle_speed > 0 in {file_name}?",
                "pandas_expression": "(df['vehicle_speed'] > 0).mean() * 100",
                "expected_answer": _format_scalar(moving_pct),
            }
        )

    if "coolant_temperature" in df.columns:
        rows.append(
            {
                "dataset_file": file_name,
                "question": f"What is the highest coolant temperature in {file_name}?",
                "pandas_expression": "df['coolant_temperature'].max()",
                "expected_answer": _format_scalar(float(df["coolant_temperature"].max())),
            }
        )

    return rows


def generate_golden_dataset() -> tuple[Path, Path, int]:
    data_root = PROJECT_ROOT / "data" / "obdiidata"
    golden_root = PROJECT_ROOT / "data" / "golden"
    golden_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []

    for file_name in TARGET_FILES:
        csv_path = data_root / file_name
        df = load_and_prep_obd_data(
            csv_path,
            coerce_numeric=True,
            parse_timestamp=True,
            drop_rows_missing_required=True,
            drop_duplicates=True,
            preserve_empty_cells=True,
        )
        all_rows.extend(_build_questions_for_file(file_name=file_name, df=df))

    golden_df = pd.DataFrame(all_rows).reset_index(drop=True)
    golden_df.insert(0, "id", range(1, len(golden_df) + 1))

    csv_out = golden_root / "pal_golden_dataset.csv"
    txt_out = golden_root / "pal_golden_answers.txt"
    golden_df.to_csv(csv_out, index=False)

    lines: list[str] = []
    for row in golden_df.itertuples(index=False):
        lines.append(f"Q{row.id}: [{row.dataset_file}] {row.question}")
        lines.append(f"A{row.id}: {row.expected_answer}")
        lines.append("")

    txt_out.write_text("\n".join(lines), encoding="utf-8")
    return csv_out, txt_out, len(golden_df)


if __name__ == "__main__":
    csv_path, txt_path, count = generate_golden_dataset()
    print(f"Generated {count} golden QA rows")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")
