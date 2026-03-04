from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_NUMERIC_COLUMNS = (
    "engine_rpm",
    "vehicle_speed",
    "coolant_temperature",
    "throttle_position",
    "intake_air_temperature",
    "intake_manifold_pressure",
    "fuel_level",
    "maf",
)


def _sanitize_column_name(name: str) -> str:
    sanitized = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unnamed"


def _deduplicate_column_names(column_names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []

    for column_name in column_names:
        if column_name not in seen:
            seen[column_name] = 0
            result.append(column_name)
            continue

        seen[column_name] += 1
        result.append(f"{column_name}_{seen[column_name]}")

    return result


def _read_csv_preserve_all_cells(filepath: str | Path) -> pd.DataFrame:
    """Read CSV while preserving every column/cell, even with ragged row lengths."""

    with Path(filepath).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return pd.DataFrame()

    header = [str(value) for value in rows[0]]
    data_rows = rows[1:]
    max_row_len = max((len(row) for row in data_rows), default=0)
    target_width = max(len(header), max_row_len)

    if len(header) < target_width:
        for i in range(target_width - len(header)):
            header.append(f"extra_column_{i + 1}")

    normalized_rows: list[list[str]] = []
    for row in data_rows:
        row_values = [str(value) for value in row]
        if len(row_values) < target_width:
            row_values.extend([""] * (target_width - len(row_values)))
        normalized_rows.append(row_values)

    return pd.DataFrame(normalized_rows, columns=header)


def load_and_prep_obd_data(
    filepath: str | Path,
    *,
    numeric_columns: Iterable[str] = DEFAULT_NUMERIC_COLUMNS,
    required_columns: Iterable[str] = ("engine_rpm", "vehicle_speed"),
    timestamp_column: str = "timestamp",
    coerce_numeric: bool = False,
    parse_timestamp: bool = False,
    drop_rows_missing_required: bool = False,
    drop_duplicates: bool = False,
    preserve_empty_cells: bool = True,
    include_snapshot: bool = False,
    snapshot_rows: int = 10,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load and sanitize an OBD CSV log for downstream LLM analytics.

    Steps:
    1. Read CSV while normalizing common missing-value tokens.
    2. Standardize column names (lowercase + snake_case + unique names).
    3. Trim whitespace in string columns.
    4. Optionally coerce selected OBD metric columns to numeric.
    5. Optionally parse timestamp column.
    6. Preserve all rows/cells by default (drops are opt-in).
    7. Optionally return a first-N-rows all-columns snapshot string.
    """

    if preserve_empty_cells:
        df = _read_csv_preserve_all_cells(filepath)
    else:
        df = pd.read_csv(
            filepath,
            index_col=False,
            na_values=["", "N/A", "NA", "null", "None", "-"],
            keep_default_na=True,
        )

    sanitized = [_sanitize_column_name(col) for col in df.columns]
    df.columns = _deduplicate_column_names(sanitized)

    string_columns = df.select_dtypes(include=["object", "string"]).columns
    for column_name in string_columns:
        df[column_name] = df[column_name].astype("string").str.strip()

    if coerce_numeric:
        for column_name in numeric_columns:
            if column_name in df.columns:
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    if parse_timestamp and timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)

    if drop_rows_missing_required:
        required_existing = [column for column in required_columns if column in df.columns]
        if required_existing:
            df = df.dropna(subset=required_existing)

    if drop_duplicates:
        df = df.drop_duplicates()

    if parse_timestamp and timestamp_column in df.columns:
        df = df.sort_values(timestamp_column)

    df = df.reset_index(drop=True)

    if include_snapshot:
        snapshot = build_dataframe_snapshot(df, rows=snapshot_rows)
        return df, snapshot

    return df


def build_dataframe_snapshot(df: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    """Return a snapshot DataFrame of the first N rows with all columns."""

    if rows <= 0:
        rows = 10

    return df.head(rows).copy()


def export_prepared_obd_outputs(
    df: pd.DataFrame,
    *,
    source_csv_path: str | Path,
    output_dir: str | Path = "data/processed",
    export_snapshot_csv: bool = True,
    snapshot_rows: int = 10,
) -> dict[str, Path]:
    """Export cleaned OBD results to disk for downstream use.

    Exports:
    - <stem>_cleaned.csv: sanitized DataFrame
    - <stem>_snapshot.csv: first-N-row snapshot with all columns (optional)
    """

    source_path = Path(source_csv_path)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv_path = destination_dir / f"{source_path.stem}_cleaned.csv"
    try:
        df.to_csv(cleaned_csv_path, index=False)
    except PermissionError:
        fallback_cleaned_csv_path = (
            destination_dir / f"{source_path.stem}_cleaned_{pd.Timestamp.now('UTC'):%Y%m%d_%H%M%S}.csv"
        )
        df.to_csv(fallback_cleaned_csv_path, index=False)
        cleaned_csv_path = fallback_cleaned_csv_path

    exported_paths: dict[str, Path] = {"cleaned_csv": cleaned_csv_path}

    if export_snapshot_csv:
        snapshot_df = build_dataframe_snapshot(df, rows=snapshot_rows)
        snapshot_path = destination_dir / f"{source_path.stem}_snapshot.csv"
        try:
            snapshot_df.to_csv(snapshot_path, index=False)
        except PermissionError:
            snapshot_path = destination_dir / f"{source_path.stem}_snapshot_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}.csv"
            snapshot_df.to_csv(snapshot_path, index=False)
        exported_paths["snapshot_csv"] = snapshot_path

    return exported_paths


if __name__ == "__main__":
    sample_path = Path("data/obdiidata/drive1.csv")
    clean_df = load_and_prep_obd_data(sample_path)

    exported = export_prepared_obd_outputs(
        clean_df,
        source_csv_path=sample_path,
        export_snapshot_csv=True,
        snapshot_rows=10,
    )
    print(f"Rows: {len(clean_df):,}, Columns: {len(clean_df.columns)}")
    print("Exported files:")
    for name, path in exported.items():
        print(f"- {name}: {path}")