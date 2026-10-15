"""
Aadhaar Insights — Clean Forecasting Data

Responsibilities:
- Parse source dates explicitly.
- Remove exact duplicate records after date normalization.
- Aggregate activity by observed date.
- Preserve missing dates as missing observations.
- Export daily activity and a reproducible cleaning audit.

This module does not modify raw data or train models.
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_all_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_clean"
)

DATE_COLUMN = "date"

ACTIVITY_COLUMNS = {
    "enrolment": [
        "age_0_5",
        "age_5_17",
        "age_18_greater",
    ],
    "demographic": [
        "demo_age_5_17",
        "demo_age_17_",
    ],
    "biometric": [
        "bio_age_5_17",
        "bio_age_17_",
    ],
}


def prepare_daily_series(
    df: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Clean one dataset and aggregate activity by observed date."""

    if dataset_name not in ACTIVITY_COLUMNS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"{dataset_name}: missing '{DATE_COLUMN}' column"
        )

    targets = ACTIVITY_COLUMNS[dataset_name]

    missing_columns = [
        col for col in targets if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing target columns "
            f"{missing_columns}"
        )

    raw_rows = len(df)
    work = df.copy()

    # Parse the known source format explicitly.
    work[DATE_COLUMN] = pd.to_datetime(
        work[DATE_COLUMN].astype("string").str.strip(),
        format="%d-%m-%Y",
        errors="coerce",
    )

    invalid_date_rows = int(work[DATE_COLUMN].isna().sum())

    # Invalid-date records cannot be assigned to a dated series.
    valid = work.loc[work[DATE_COLUMN].notna()].copy()

    # Convert activity values to numeric, but fail rather than
    # silently discard unexpected non-numeric values.
    for col in targets:
        original = valid[col]
        numeric = pd.to_numeric(original, errors="coerce")

        invalid_numeric = (
            original.notna() & numeric.isna()
        )

        if invalid_numeric.any():
            examples = original.loc[
                invalid_numeric
            ].astype(str).head(5).tolist()

            raise ValueError(
                f"{dataset_name}: non-numeric values in "
                f"'{col}'. Examples: {examples}"
            )

        if numeric.isna().any():
            raise ValueError(
                f"{dataset_name}: missing activity values in "
                f"'{col}'. Resolve these before forecasting."
            )

        valid[col] = numeric

    # Deduplicate complete records, not merely geographic keys.
    rows_before_dedup = len(valid)
    cleaned = valid.drop_duplicates().copy()
    duplicate_rows_removed = (
        rows_before_dedup - len(cleaned)
    )

    # A daily observation is the sum of the activity columns
    # across all retained records for that date.
    daily = (
        cleaned.groupby(DATE_COLUMN, as_index=False)[targets]
        .sum()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    daily["total_activity"] = daily[targets].sum(axis=1)

    observed_dates = pd.DatetimeIndex(daily[DATE_COLUMN])

    if len(observed_dates) > 1:
        calendar = pd.date_range(
            observed_dates.min(),
            observed_dates.max(),
            freq="D",
        )
        missing_calendar_days = len(
            calendar.difference(observed_dates)
        )
        largest_gap_days = int(
            observed_dates.to_series()
            .diff()
            .dt.days
            .max()
        )
    else:
        missing_calendar_days = 0
        largest_gap_days = 0

    audit = {
        "dataset": dataset_name,
        "raw_rows": raw_rows,
        "invalid_date_rows": invalid_date_rows,
        "valid_date_rows_before_dedup": rows_before_dedup,
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "retained_rows": len(cleaned),
        "observed_dates": len(daily),
        "first_observed_date": (
            daily[DATE_COLUMN].min().strftime("%Y-%m-%d")
            if not daily.empty else None
        ),
        "last_observed_date": (
            daily[DATE_COLUMN].max().strftime("%Y-%m-%d")
            if not daily.empty else None
        ),
        "missing_calendar_days": missing_calendar_days,
        "largest_gap_days": largest_gap_days,
    }

    # Record totals before and after deduplication for comparison.
    for col in targets:
        audit[f"{col}_total_before_dedup"] = (
            valid[col].sum()
        )
        audit[f"{col}_total_after_dedup"] = (
            cleaned[col].sum()
        )

    audit["total_activity_after_dedup"] = (
        daily["total_activity"].sum()
    )

    return daily, audit


def main() -> None:
    """Build clean daily series and audit files for all datasets."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    audit_rows = []

    for dataset_name, raw_df in datasets.items():
        print(f"\nProcessing {dataset_name}...")

        daily, audit = prepare_daily_series(
            raw_df,
            dataset_name,
        )

        dataset_output_dir = OUTPUT_DIR / dataset_name
        dataset_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        daily_path = (
            dataset_output_dir / "daily_activity.csv"
        )

        daily.to_csv(daily_path, index=False)
        audit_rows.append(audit)

        print(f"Raw rows: {audit['raw_rows']:,}")
        print(
            "Invalid date rows: "
            f"{audit['invalid_date_rows']:,}"
        )
        print(
            "Exact duplicates removed: "
            f"{audit['exact_duplicate_rows_removed']:,}"
        )
        print(
            f"Retained rows: {audit['retained_rows']:,}"
        )
        print(
            f"Observed dates: {audit['observed_dates']}"
        )
        print(
            "Missing calendar days in observed range: "
            f"{audit['missing_calendar_days']}"
        )
        print(f"Saved: {daily_path}")

    audit_df = pd.DataFrame(audit_rows)
    audit_path = OUTPUT_DIR / "cleaning_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    print(f"\nCleaning audit saved: {audit_path}")
    print("Data preparation completed.")


if __name__ == "__main__":
    main()