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


DEFAULT_TARGET_COUNT = 100
DEFAULT_FILES_PER_PREFIX = 3
PREFIXES = ("drive", "idle", "live", "long", "ufpe")
DEFAULT_MAX_QUESTIONS_PER_FILE = 20

# Extend coercion to all numeric columns present in OBD-II logs
ALL_NUMERIC_COLUMNS = (
    "engine_rpm",
    "vehicle_speed",
    "coolant_temperature",
    "throttle_position",
    "intake_air_temperature",
    "intake_manifold_pressure",
    "fuel_level",
    "maf",
    # additional columns present in this dataset
    "throttle",
    "engine_load",
    "long_term_fuel_trim_bank_1",
    "short_term_fuel_trim_bank_1",
    "fuel_tank",
    "intake_air_temp",
    "timing_advance",
    "catalyst_temperature_bank1_sensor1",
    "catalyst_temperature_bank1_sensor2",
    "control_module_voltage",
    "absolute_barometric_pressure",
    "relative_throttle_position",
    "absolute_throttle_b",
    "commanded_throttle_actuator",
)


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


def _interleave_by_category(
    candidates: list[dict[str, str]], count: int, file_index: int
) -> list[dict[str, str]]:
    """Select up to `count` questions from `candidates` ensuring every question
    category gets at least one representative before any category gets a second.

    Categories are derived from the `pandas_expression` column to avoid relying
    on question phrasing. The file_index is used to rotate which item is chosen
    first within each category, giving different files different question subsets.
    """
    # Group by category (first meaningful token of the pandas expression)
    from collections import defaultdict as _dd
    groups: dict[str, list[dict[str, str]]] = _dd(list)
    for item in candidates:
        expr = item.get("pandas_expression", "")
        # Derive a category key from the expression
        if ".corr(" in expr:
            cat = "correlation"
        elif ".std()" in expr:
            cat = "std_dev"
        elif ".quantile(0.9)" in expr and "engine_rpm" in expr:
            cat = "rpm_q90"
        elif ".quantile(" in expr and "engine_rpm" in expr:
            cat = "rpm_iqr"
        elif ".quantile(" in expr:
            cat = "quantile"
        elif "catalyst_temperature" in expr:
            cat = "catalyst_temp"
        elif "timing_advance" in expr:
            cat = "timing"
        elif "control_module_voltage" in expr:
            cat = "voltage"
        elif "fuel_tank" in expr:
            cat = "fuel_tank"
        elif "engine_load" in expr:
            cat = "engine_load"
        elif "short_term_fuel_trim" in expr:
            cat = "stft"
        elif "long_term_fuel_trim" in expr:
            cat = "ltft"
        elif "intake_air_temp" in expr:
            cat = "iat"
        elif "coolant_temperature" in expr:
            cat = "coolant"
        elif "vehicle_speed" in expr and ("engine_rpm" in expr or "engine_load" in expr):
            cat = "cross_speed_rpm"
        elif "vehicle_speed" in expr and "1.60934" in expr:
            cat = "speed_kmh"
        elif "vehicle_speed" in expr:
            cat = "speed"
        elif "engine_rpm" in expr:
            cat = "rpm"
        elif "len(df)" in expr:
            cat = "rowcount"
        else:
            cat = "other"
        groups[cat].append(item)

    # Rotate within each category by file_index so different files get different items
    rotated: dict[str, list[dict[str, str]]] = {}
    for cat, items in groups.items():
        offset = file_index % len(items)
        rotated[cat] = items[offset:] + items[:offset]

    # Round-robin across categories
    cat_order = sorted(rotated.keys())
    result: list[dict[str, str]] = []
    pointers = {cat: 0 for cat in cat_order}
    while len(result) < count:
        added_this_round = 0
        for cat in cat_order:
            if len(result) >= count:
                break
            ptr = pointers[cat]
            if ptr < len(rotated[cat]):
                result.append(rotated[cat][ptr])
                pointers[cat] += 1
                added_this_round += 1
        if added_this_round == 0:
            break  # all categories exhausted

    return result


def _build_question_pool_for_file(
    file_name: str,
    df: pd.DataFrame,
    *,
    file_index: int,
    max_questions_per_file: int,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []

    def add(question: str, expression: str, value: float | int) -> None:
        candidates.append(
            {
                "dataset_file": file_name,
                "question": question,
                "pandas_expression": expression,
                "expected_answer": _format_scalar(value),
            }
        )

    # ------------------------------------------------------------------ #
    # Basic row-count                                                      #
    # ------------------------------------------------------------------ #
    add(f"How many rows are in {file_name}?", "len(df)", len(df))

    # ------------------------------------------------------------------ #
    # Engine RPM — basic aggregates                                        #
    # ------------------------------------------------------------------ #
    engine = _prepare_numeric(df, "engine_rpm")
    if not engine.empty:
        engine_mean = float(engine.mean())
        engine_median = float(engine.median())
        engine_q90 = float(engine.quantile(0.9))
        engine_q10 = float(engine.quantile(0.1))
        engine_q75 = float(engine.quantile(0.75))
        engine_q25 = float(engine.quantile(0.25))

        add(f"What is the maximum engine rpm in {file_name}?", "df['engine_rpm'].max()", float(engine.max()))
        add(f"What is the minimum engine rpm in {file_name}?", "df['engine_rpm'].min()", float(engine.min()))
        add(f"What is the average engine rpm in {file_name}?", "df['engine_rpm'].mean()", engine_mean)
        add(f"What is the median engine rpm in {file_name}?", "df['engine_rpm'].median()", engine_median)
        add(
            f"What is the 90th percentile of engine rpm in {file_name}?",
            "df['engine_rpm'].quantile(0.9)",
            engine_q90,
        )
        add(
            f"What is the interquartile range of engine rpm in {file_name}?",
            "df['engine_rpm'].quantile(0.75) - df['engine_rpm'].quantile(0.25)",
            float(engine_q75 - engine_q25),
        )
        add(
            f"What percentage of rows have engine_rpm above its own mean in {file_name}?",
            "(df['engine_rpm'] > df['engine_rpm'].mean()).mean() * 100",
            float((engine > engine_mean).mean() * 100.0),
        )
        add(
            f"How many rows have engine_rpm in the top 10 percent in {file_name}?",
            "(df['engine_rpm'] >= df['engine_rpm'].quantile(0.9)).sum()",
            int((engine >= engine_q90).sum()),
        )
        add(
            f"How many rows have engine_rpm in the lowest 10 percent in {file_name}?",
            "(df['engine_rpm'] <= df['engine_rpm'].quantile(0.1)).sum()",
            int((engine <= engine_q10).sum()),
        )
        add(
            f"What is the standard deviation of engine_rpm in {file_name}?",
            "df['engine_rpm'].std()",
            float(engine.std()),
        )

    # ------------------------------------------------------------------ #
    # Vehicle speed                                                        #
    # ------------------------------------------------------------------ #
    speed = _prepare_numeric(df, "vehicle_speed")
    if not speed.empty:
        speed_mean = float(speed.mean())

        add(f"What is the maximum vehicle speed in {file_name}?", "df['vehicle_speed'].max()", float(speed.max()))
        add(f"What is the minimum vehicle speed in {file_name}?", "df['vehicle_speed'].min()", float(speed.min()))
        add(f"What is the average vehicle speed in {file_name}?", "df['vehicle_speed'].mean()", speed_mean)
        add(f"What is the median vehicle speed in {file_name}?", "df['vehicle_speed'].median()", float(speed.median()))
        add(
            f"What percentage of rows have vehicle_speed > 0 in {file_name}?",
            "(df['vehicle_speed'] > 0).mean() * 100",
            float((speed > 0).mean() * 100.0),
        )
        add(
            f"How many rows have vehicle_speed above the dataset average in {file_name}?",
            "(df['vehicle_speed'] > df['vehicle_speed'].mean()).sum()",
            int((speed > speed_mean).sum()),
        )
        add(
            f"What is the speed range (max-min) in {file_name}?",
            "df['vehicle_speed'].max() - df['vehicle_speed'].min()",
            float(speed.max() - speed.min()),
        )
        add(
            f"What is the average vehicle speed in km/h for {file_name}?",
            "df['vehicle_speed'].mean() * 1.60934",
            float(speed_mean * 1.60934),
        )

    # ------------------------------------------------------------------ #
    # Coolant temperature                                                  #
    # ------------------------------------------------------------------ #
    coolant = _prepare_numeric(df, "coolant_temperature")
    if not coolant.empty:
        add(
            f"What is the highest coolant temperature in {file_name}?",
            "df['coolant_temperature'].max()",
            float(coolant.max()),
        )
        add(
            f"What is the average coolant temperature in {file_name}?",
            "df['coolant_temperature'].mean()",
            float(coolant.mean()),
        )
        add(
            f"How many rows have coolant_temperature above its 75th percentile in {file_name}?",
            "(df['coolant_temperature'] > df['coolant_temperature'].quantile(0.75)).sum()",
            int((coolant > float(coolant.quantile(0.75))).sum()),
        )
        add(
            f"What is the minimum coolant temperature in {file_name}?",
            "df['coolant_temperature'].min()",
            float(coolant.min()),
        )

    # ------------------------------------------------------------------ #
    # Engine load                                                          #
    # ------------------------------------------------------------------ #
    load_col = _prepare_numeric(df, "engine_load")
    if not load_col.empty and load_col.std() > 0:
        add(
            f"What is the average engine load in {file_name}?",
            "df['engine_load'].mean()",
            float(load_col.mean()),
        )
        add(
            f"What is the maximum engine load in {file_name}?",
            "df['engine_load'].max()",
            float(load_col.max()),
        )
        add(
            f"What percentage of rows have engine_load above 50 in {file_name}?",
            "(df['engine_load'] > 50).mean() * 100",
            float((load_col > 50).mean() * 100.0),
        )

    # ------------------------------------------------------------------ #
    # Throttle position (column named 'throttle' in this dataset)         #
    # ------------------------------------------------------------------ #
    throttle = _prepare_numeric(df, "throttle")
    if not throttle.empty and throttle.std() > 0:
        add(
            f"What is the maximum throttle position in {file_name}?",
            "df['throttle'].max()",
            float(throttle.max()),
        )
        add(
            f"What is the average throttle position in {file_name}?",
            "df['throttle'].mean()",
            float(throttle.mean()),
        )
        add(
            f"What percentage of rows have throttle above 25 in {file_name}?",
            "(df['throttle'] > 25).mean() * 100",
            float((throttle > 25).mean() * 100.0),
        )

    # ------------------------------------------------------------------ #
    # Short-term fuel trim (STFT)                                          #
    # ------------------------------------------------------------------ #
    stft = _prepare_numeric(df, "short_term_fuel_trim_bank_1")
    if not stft.empty and stft.std() > 0:
        add(
            f"What is the average short-term fuel trim in {file_name}?",
            "df['short_term_fuel_trim_bank_1'].mean()",
            float(stft.mean()),
        )
        add(
            f"What is the maximum short-term fuel trim in {file_name}?",
            "df['short_term_fuel_trim_bank_1'].max()",
            float(stft.max()),
        )
        add(
            f"How many rows have short_term_fuel_trim_bank_1 above 5 in {file_name}?",
            "(df['short_term_fuel_trim_bank_1'] > 5).sum()",
            int((stft > 5).sum()),
        )

    # ------------------------------------------------------------------ #
    # Long-term fuel trim (LTFT)                                           #
    # ------------------------------------------------------------------ #
    ltft = _prepare_numeric(df, "long_term_fuel_trim_bank_1")
    if not ltft.empty and ltft.std() > 0:
        add(
            f"What is the average long-term fuel trim in {file_name}?",
            "df['long_term_fuel_trim_bank_1'].mean()",
            float(ltft.mean()),
        )
        add(
            f"What is the minimum long-term fuel trim in {file_name}?",
            "df['long_term_fuel_trim_bank_1'].min()",
            float(ltft.min()),
        )

    # ------------------------------------------------------------------ #
    # Catalyst temperature                                                 #
    # ------------------------------------------------------------------ #
    cat_temp = _prepare_numeric(df, "catalyst_temperature_bank1_sensor1")
    if not cat_temp.empty and cat_temp.std() > 0:
        add(
            f"What is the average catalyst temperature (bank 1 sensor 1) in {file_name}?",
            "df['catalyst_temperature_bank1_sensor1'].mean()",
            float(cat_temp.mean()),
        )
        add(
            f"What is the maximum catalyst temperature (bank 1 sensor 1) in {file_name}?",
            "df['catalyst_temperature_bank1_sensor1'].max()",
            float(cat_temp.max()),
        )

    # ------------------------------------------------------------------ #
    # Timing advance                                                       #
    # ------------------------------------------------------------------ #
    timing = _prepare_numeric(df, "timing_advance")
    if not timing.empty and timing.std() > 0:
        add(
            f"What is the average timing advance in {file_name}?",
            "df['timing_advance'].mean()",
            float(timing.mean()),
        )
        add(
            f"How many rows have negative timing advance in {file_name}?",
            "(df['timing_advance'] < 0).sum()",
            int((timing < 0).sum()),
        )

    # ------------------------------------------------------------------ #
    # Intake air temperature                                               #
    # ------------------------------------------------------------------ #
    iat = _prepare_numeric(df, "intake_air_temp")
    if not iat.empty and iat.std() > 0:
        add(
            f"What is the average intake air temperature in {file_name}?",
            "df['intake_air_temp'].mean()",
            float(iat.mean()),
        )
        add(
            f"What is the maximum intake air temperature in {file_name}?",
            "df['intake_air_temp'].max()",
            float(iat.max()),
        )

    # ------------------------------------------------------------------ #
    # Control module voltage                                               #
    # ------------------------------------------------------------------ #
    voltage = _prepare_numeric(df, "control_module_voltage")
    if not voltage.empty and voltage.std() > 0:
        add(
            f"What is the average control module voltage in {file_name}?",
            "df['control_module_voltage'].mean()",
            float(voltage.mean()),
        )
        add(
            f"What is the minimum control module voltage in {file_name}?",
            "df['control_module_voltage'].min()",
            float(voltage.min()),
        )

    # ------------------------------------------------------------------ #
    # Fuel tank level                                                      #
    # ------------------------------------------------------------------ #
    fuel_tank = _prepare_numeric(df, "fuel_tank")
    if not fuel_tank.empty and fuel_tank.std() > 0:
        add(
            f"What is the average fuel tank level in {file_name}?",
            "df['fuel_tank'].mean()",
            float(fuel_tank.mean()),
        )

    # ------------------------------------------------------------------ #
    # Cross-column: engine_rpm vs vehicle_speed                           #
    # ------------------------------------------------------------------ #
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

        if moving_mask.any():
            add(
                f"What is the average engine rpm when vehicle_speed > 0 in {file_name}?",
                "df.loc[df['vehicle_speed'] > 0, 'engine_rpm'].mean()",
                float(joint_engine[moving_mask].mean()),
            )
        if parked_mask.any():
            add(
                f"What is the average engine rpm when vehicle_speed == 0 in {file_name}?",
                "df.loc[df['vehicle_speed'] == 0, 'engine_rpm'].mean()",
                float(joint_engine[parked_mask].mean()),
            )
        add(
            f"How many rows have both engine_rpm > 1500 and vehicle_speed > 0 in {file_name}?",
            "((df['engine_rpm'] > 1500) & (df['vehicle_speed'] > 0)).sum()",
            int(((joint_engine > 1500) & moving_mask).sum()),
        )

        if len(joint) > 1:
            engine_std = float(joint_engine.std(ddof=0))
            speed_std = float(joint_speed.std(ddof=0))
            if engine_std > 0.0 and speed_std > 0.0:
                correlation = float(joint_engine.corr(joint_speed))
                add(
                    f"What is the correlation between engine_rpm and vehicle_speed in {file_name}?",
                    "df['engine_rpm'].corr(df['vehicle_speed'])",
                    correlation,
                )

    # ------------------------------------------------------------------ #
    # Cross-column: engine_load vs vehicle_speed                          #
    # ------------------------------------------------------------------ #
    if not load_col.empty and load_col.std() > 0 and not speed.empty:
        joint_ls = (
            df[["engine_load", "vehicle_speed"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        moving_mask_ls = joint_ls["vehicle_speed"] > 0
        if moving_mask_ls.any():
            add(
                f"What is the average engine load when vehicle_speed > 0 in {file_name}?",
                "df.loc[df['vehicle_speed'] > 0, 'engine_load'].mean()",
                float(joint_ls.loc[moving_mask_ls, "engine_load"].mean()),
            )

    # ------------------------------------------------------------------ #
    # Cross-column: STFT behaviour relative to engine rpm                 #
    # ------------------------------------------------------------------ #
    if not stft.empty and stft.std() > 0 and not engine.empty:
        joint_sf = (
            df[["short_term_fuel_trim_bank_1", "engine_rpm"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        high_rpm_mask = joint_sf["engine_rpm"] > 1500
        if high_rpm_mask.any():
            add(
                f"What is the average short-term fuel trim when engine_rpm > 1500 in {file_name}?",
                "df.loc[df['engine_rpm'] > 1500, 'short_term_fuel_trim_bank_1'].mean()",
                float(joint_sf.loc[high_rpm_mask, "short_term_fuel_trim_bank_1"].mean()),
            )

    # ------------------------------------------------------------------ #
    # Cross-column: catalyst temp vs coolant temp correlation             #
    # ------------------------------------------------------------------ #
    if not cat_temp.empty and cat_temp.std() > 0 and not coolant.empty:
        joint_ct = (
            df[["catalyst_temperature_bank1_sensor1", "coolant_temperature"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        if len(joint_ct) > 1:
            ct_std = float(joint_ct["catalyst_temperature_bank1_sensor1"].std(ddof=0))
            cool_std = float(joint_ct["coolant_temperature"].std(ddof=0))
            if ct_std > 0 and cool_std > 0:
                corr_ct = float(
                    joint_ct["catalyst_temperature_bank1_sensor1"].corr(joint_ct["coolant_temperature"])
                )
                add(
                    f"What is the correlation between catalyst_temperature_bank1_sensor1 and coolant_temperature in {file_name}?",
                    "df['catalyst_temperature_bank1_sensor1'].corr(df['coolant_temperature'])",
                    corr_ct,
                )

    return _interleave_by_category(candidates, count=max_questions_per_file, file_index=file_index)


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
            numeric_columns=ALL_NUMERIC_COLUMNS,
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
    golden_df.to_csv(csv_out, index=False)
    return csv_out, len(golden_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PAL golden dataset for evaluation.")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--files-per-prefix", type=int, default=DEFAULT_FILES_PER_PREFIX)
    parser.add_argument("--max-questions-per-file", type=int, default=DEFAULT_MAX_QUESTIONS_PER_FILE)
    args = parser.parse_args()

    csv_path, count = generate_golden_dataset(
        target_count=max(1, args.target_count),
        files_per_prefix=max(1, args.files_per_prefix),
        max_questions_per_file=max(1, args.max_questions_per_file),
    )
    print(f"Generated {count} golden QA rows")
    print(f"CSV: {csv_path}")
