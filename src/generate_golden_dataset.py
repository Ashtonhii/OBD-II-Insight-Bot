from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_and_prep_obd_data


DEFAULT_TARGET_COUNT = 110
DEFAULT_FILES_PER_PREFIX = 3
PREFIXES = ("drive", "idle", "live", "long", "ufpe")
DEFAULT_MAX_QUESTIONS_PER_FILE = 20


def _format_scalar(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4f}"


def _prepare_numeric(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column_name], errors="coerce").dropna()


def _rotating_take(items: list[dict[str, str]], count: int, offset: int) -> list[dict[str, str]]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return items

    ordered = items[offset:] + items[:offset]
    return ordered[:count]


def _build_question_pool_for_file(
    file_name: str,
    df: pd.DataFrame,
    *,
    file_index: int,
    max_questions_per_file: int,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []

    def add_question(question: str, expression: str, value: float | int) -> None:
        candidates.append(
            {
                "dataset_file": file_name,
                "question": question,
                "pandas_expression": expression,
                "expected_answer": _format_scalar(value),
            }
        )

    add_question(f"How many rows are in {file_name}?", "len(df)", len(df))

    engine = _prepare_numeric(df, "engine_rpm")
    speed = _prepare_numeric(df, "vehicle_speed")
    coolant = _prepare_numeric(df, "coolant_temperature")
    throttle = _prepare_numeric(df, "throttle_position")
    maf = _prepare_numeric(df, "maf")

    if not engine.empty:
        engine_mean = float(engine.mean())
        engine_median = float(engine.median())
        engine_q90 = float(engine.quantile(0.9))
        engine_q10 = float(engine.quantile(0.1))

        add_question(f"What is the maximum engine rpm in {file_name}?", "df['engine_rpm'].max()", float(engine.max()))
        add_question(f"What is the minimum engine rpm in {file_name}?", "df['engine_rpm'].min()", float(engine.min()))
        add_question(f"What is the average engine rpm in {file_name}?", "df['engine_rpm'].mean()", engine_mean)
        add_question(f"What is the median engine rpm in {file_name}?", "df['engine_rpm'].median()", engine_median)
        add_question(
            f"What is the 90th percentile of engine rpm in {file_name}?",
            "df['engine_rpm'].quantile(0.9)",
            engine_q90,
        )
        add_question(
            f"What is the interquartile range of engine rpm in {file_name}?",
            "df['engine_rpm'].quantile(0.75) - df['engine_rpm'].quantile(0.25)",
            float(engine.quantile(0.75) - engine.quantile(0.25)),
        )
        add_question(
            f"What percentage of rows have engine_rpm above its own mean in {file_name}?",
            "(df['engine_rpm'] > df['engine_rpm'].mean()).mean() * 100",
            float((engine > engine_mean).mean() * 100.0),
        )
        add_question(
            f"How many rows have engine_rpm in the top 10 percent in {file_name}?",
            "(df['engine_rpm'] >= df['engine_rpm'].quantile(0.9)).sum()",
            int((engine >= engine_q90).sum()),
        )
        add_question(
            f"How many rows have engine_rpm in the lowest 10 percent in {file_name}?",
            "(df['engine_rpm'] <= df['engine_rpm'].quantile(0.1)).sum()",
            int((engine <= engine_q10).sum()),
        )

    if not speed.empty:
        moving_pct = float((speed > 0).mean() * 100.0)
        speed_mean = float(speed.mean())

        add_question(
            f"What is the maximum vehicle speed in {file_name}?",
            "df['vehicle_speed'].max()",
            float(speed.max()),
        )
        add_question(
            f"What is the minimum vehicle speed in {file_name}?",
            "df['vehicle_speed'].min()",
            float(speed.min()),
        )
        add_question(
            f"What is the average vehicle speed in {file_name}?",
            "df['vehicle_speed'].mean()",
            speed_mean,
        )
        add_question(
            f"What is the median vehicle speed in {file_name}?",
            "df['vehicle_speed'].median()",
            float(speed.median()),
        )
        add_question(
            f"What percentage of rows have vehicle_speed > 0 in {file_name}?",
            "(df['vehicle_speed'] > 0).mean() * 100",
            moving_pct,
        )
        add_question(
            f"How many rows have vehicle_speed above the dataset average in {file_name}?",
            "(df['vehicle_speed'] > df['vehicle_speed'].mean()).sum()",
            int((speed > speed_mean).sum()),
        )
        add_question(
            f"What is the speed range (max-min) in {file_name}?",
            "df['vehicle_speed'].max() - df['vehicle_speed'].min()",
            float(speed.max() - speed.min()),
        )
        add_question(
            f"What is the average vehicle speed in km/h for {file_name}?",
            "df['vehicle_speed'].mean() * 1.60934",
            float(speed_mean * 1.60934),
        )

    if not coolant.empty:
        add_question(
            f"What is the highest coolant temperature in {file_name}?",
            "df['coolant_temperature'].max()",
            float(coolant.max()),
        )
        add_question(
            f"What is the average coolant temperature in {file_name}?",
            "df['coolant_temperature'].mean()",
            float(coolant.mean()),
        )
        add_question(
            f"How many rows have coolant_temperature above its 75th percentile in {file_name}?",
            "(df['coolant_temperature'] > df['coolant_temperature'].quantile(0.75)).sum()",
            int((coolant > float(coolant.quantile(0.75))).sum()),
        )

    if not throttle.empty:
        add_question(
            f"What is the maximum throttle position in {file_name}?",
            "df['throttle_position'].max()",
            float(throttle.max()),
        )
        add_question(
            f"What is the average throttle position in {file_name}?",
            "df['throttle_position'].mean()",
            float(throttle.mean()),
        )
        add_question(
            f"What percentage of rows have throttle_position above 20 in {file_name}?",
            "(df['throttle_position'] > 20).mean() * 100",
            float((throttle > 20).mean() * 100.0),
        )

    if not maf.empty:
        add_question(
            f"What is the maximum MAF value in {file_name}?",
            "df['maf'].max()",
            float(maf.max()),
        )
        add_question(
            f"What is the average MAF value in {file_name}?",
            "df['maf'].mean()",
            float(maf.mean()),
        )
        add_question(
            f"What is the median MAF value in {file_name}?",
            "df['maf'].median()",
            float(maf.median()),
        )

    if not engine.empty and not speed.empty:
        joint = (
            df[["engine_rpm", "vehicle_speed"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        joint_engine = joint["engine_rpm"]
        joint_speed = joint["vehicle_speed"]
        moving_mask = joint_speed > 0
        parked_mask = joint_speed == 0
        add_question(
            f"What is the average engine rpm when vehicle_speed > 0 in {file_name}?",
            "df.loc[df['vehicle_speed'] > 0, 'engine_rpm'].mean()",
            float(joint_engine[moving_mask].mean()) if moving_mask.any() else 0,
        )
        add_question(
            f"What is the average engine rpm when vehicle_speed == 0 in {file_name}?",
            "df.loc[df['vehicle_speed'] == 0, 'engine_rpm'].mean()",
            float(joint_engine[parked_mask].mean()) if parked_mask.any() else 0,
        )
        add_question(
            f"How many rows have both engine_rpm > 1500 and vehicle_speed > 0 in {file_name}?",
            "((df['engine_rpm'] > 1500) & (df['vehicle_speed'] > 0)).sum()",
            int(((joint_engine > 1500) & moving_mask).sum()),
        )

    if not engine.empty and not speed.empty:
        corr_joint = (
            df[["engine_rpm", "vehicle_speed"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        correlation = float("nan")
        if len(corr_joint) > 1:
            engine_std = float(corr_joint["engine_rpm"].std(ddof=0))
            speed_std = float(corr_joint["vehicle_speed"].std(ddof=0))
            if engine_std > 0.0 and speed_std > 0.0:
                correlation = float(corr_joint["engine_rpm"].corr(corr_joint["vehicle_speed"]))

        if pd.notna(correlation):
            add_question(
                f"What is the correlation between engine_rpm and vehicle_speed in {file_name}?",
                "df['engine_rpm'].corr(df['vehicle_speed'])",
                correlation,
            )

    offset = file_index % len(candidates)
    return _rotating_take(candidates, count=max_questions_per_file, offset=offset)


def _select_target_files(data_root: Path, files_per_prefix: int) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(data_root.glob("*.csv")):
        stem = path.stem.lower()
        for prefix in PREFIXES:
            if stem.startswith(prefix):
                grouped[prefix].append(path.name)
                break

    target_files: list[str] = []
    for prefix in PREFIXES:
        candidates = grouped.get(prefix, [])
        target_files.extend(candidates[: max(1, files_per_prefix)])

    return target_files


def generate_golden_dataset(
    target_count: int = DEFAULT_TARGET_COUNT,
    files_per_prefix: int = DEFAULT_FILES_PER_PREFIX,
    max_questions_per_file: int = DEFAULT_MAX_QUESTIONS_PER_FILE,
) -> tuple[Path, Path, int]:
    data_root = PROJECT_ROOT / "data" / "obdiidata"
    golden_root = PROJECT_ROOT / "data" / "golden"
    golden_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    target_files = _select_target_files(data_root, files_per_prefix=files_per_prefix)

    for file_index, file_name in enumerate(target_files):
        csv_path = data_root / file_name
        df = load_and_prep_obd_data(
            csv_path,
            coerce_numeric=True,
            parse_timestamp=True,
            drop_rows_missing_required=True,
            drop_duplicates=True,
            preserve_empty_cells=True,
        )
        all_rows.extend(
            _build_question_pool_for_file(
                file_name=file_name,
                df=df,
                file_index=file_index,
                max_questions_per_file=max(1, max_questions_per_file),
            )
        )

    if target_count > 0:
        all_rows = all_rows[:target_count]

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
    parser = argparse.ArgumentParser(description="Generate PAL golden dataset for evaluation.")
    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Maximum number of rows to generate (default: 110)",
    )
    parser.add_argument(
        "--files-per-prefix",
        type=int,
        default=DEFAULT_FILES_PER_PREFIX,
        help="How many files to use per prefix group (drive/idle/live/long/ufpe)",
    )
    parser.add_argument(
        "--max-questions-per-file",
        type=int,
        default=DEFAULT_MAX_QUESTIONS_PER_FILE,
        help="Cap number of generated questions per file (default: 20)",
    )
    args = parser.parse_args()

    csv_path, txt_path, count = generate_golden_dataset(
        target_count=max(1, args.target_count),
        files_per_prefix=max(1, args.files_per_prefix),
        max_questions_per_file=max(1, args.max_questions_per_file),
    )
    print(f"Generated {count} golden QA rows")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")
