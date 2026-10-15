"""Comprehensive daily activity and data-quality audit.

Reads the existing deduplicated daily series and raw source files.
Writes a consolidated diagnostic CSV without modifying raw data
or forecasting outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASETS, SUPPORTED_FILE_TYPES

OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "forecasting_clean"
)
OUTPUT_FILE = OUTPUT_ROOT / "daily_quality_audit.csv"

TARGETS = {
    "enrolment": ["age_0_5", "age_5_17", "age_18_greater"],
    "demographic": ["demo_age_5_17", "demo_age_17_"],
    "biometric": ["bio_age_5_17", "bio_age_17_"],
}

DATE_FORMAT = "%d-%m-%Y"


def load_raw_dataset(name, directory, targets):
    """Load source partitions and retain file provenance."""
    parts = []

    for ext in SUPPORTED_FILE_TYPES:
        for path in sorted(directory.rglob(f"*{ext}")):
            if path.suffix.lower() == ".csv":
                part = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                part = pd.read_excel(path)
            else:
                continue

            part["_source_file"] = path.name
            parts.append(part)

    if not parts:
        raise FileNotFoundError(f"No supported files found for {name}")

    raw = pd.concat(parts, ignore_index=True)

    required = ["date", *targets]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    raw["date"] = pd.to_datetime(
        raw["date"].astype("string").str.strip(),
        format=DATE_FORMAT,
        errors="coerce",
    )
    raw = raw.dropna(subset=["date"]).copy()

    for col in targets:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Exact duplicate identity excludes provenance so repeated rows
    # across different files are identified as well.
    identity_cols = [
        col for col in raw.columns if col != "_source_file"
    ]
    clean = raw.drop_duplicates(subset=identity_cols).copy()

    return raw, clean



def robust_scores(values):
    median = values.median()
    mad = (values - median).abs().median()

    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=values.index)

    # Preserve direction:
    # Negative = below median; positive = above median.
    return 0.6745 * (values - median) / mad


def audit_dataset(name, directory, targets):
    raw, clean = load_raw_dataset(name, directory, targets)

    # Aggregate raw rows by date.
    raw_daily = raw.groupby("date").agg(
        raw_rows=("date", "size"),
    )
    raw_daily["raw_total_activity"] = (
        raw.groupby("date")[targets].sum().sum(axis=1)
    )

    # Aggregate after exact deduplication.
    clean_daily = clean.groupby("date").agg(
        deduplicated_rows=("date", "size"),
    )
    clean_daily["deduplicated_total_activity"] = (
        clean.groupby("date")[targets].sum().sum(axis=1)
    )

    daily = raw_daily.join(clean_daily, how="outer").fillna(0)
    daily = daily.reset_index().sort_values("date")

    daily["dataset"] = name
    daily["duplicate_rows_removed"] = (
        daily["raw_rows"] - daily["deduplicated_rows"]
    )
    daily["duplicate_activity_difference"] = (
        daily["raw_total_activity"]
        - daily["deduplicated_total_activity"]
    )

    # Keep missing calendar dates as absent observations. Do not
    # manufacture zero-activity records for dates with no source rows.
    expected_dates = pd.date_range(
        daily["date"].min(), daily["date"].max(), freq="D"
    )
    daily["calendar_gap_before"] = (
        daily["date"].diff().dt.days.fillna(1).astype(int) - 1
    )

    daily["raw_activity_per_row"] = np.where(
        daily["raw_rows"] > 0,
        daily["raw_total_activity"] / daily["raw_rows"],
        np.nan,
    )
    daily["deduplicated_activity_per_row"] = np.where(
        daily["deduplicated_rows"] > 0,
        daily["deduplicated_total_activity"] / daily["deduplicated_rows"],
        np.nan,
    )

    daily["activity_robust_score"] = robust_scores(
        daily["deduplicated_total_activity"]
    )
    daily["row_count_robust_score"] = robust_scores(
        daily["deduplicated_rows"].astype(float)
    )

    # Diagnostic flags only; they do not automatically classify
    # observations as invalid.
    daily["low_row_coverage_flag"] = (
        daily["row_count_robust_score"] < -3.5
    )
    daily["unusual_activity_flag"] = (
        daily["activity_robust_score"].abs() > 3.5
    )

    daily["observed_dates_in_dataset"] = len(daily)
    daily["calendar_days_in_range"] = len(expected_dates)
    daily["missing_calendar_days_in_range"] = (
        len(expected_dates) - len(daily)
    )

    return daily


def main():
    reports = []

    for name, directory in DATASETS.items():
        print(f"\nAuditing {name}...")
        daily = audit_dataset(name, directory, TARGETS[name])
        reports.append(daily)

        print(f"Observed dates: {len(daily)}")
        print(
            "Rows removed by exact deduplication:",
            f"{daily['duplicate_rows_removed'].sum():,.0f}",
        )
        print(
            "Dates flagged for low row coverage:",
            int(daily["low_row_coverage_flag"].sum()),
        )
        print(
            "Dates flagged for unusual activity:",
            int(daily["unusual_activity_flag"].sum()),
        )

    combined = pd.concat(reports, ignore_index=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDaily quality audit saved to: {OUTPUT_FILE}")
    print("Raw data and forecasting outputs were not modified.")


if __name__ == "__main__":
    main()