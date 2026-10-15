"""Prepare validated daily activity series for Aadhaar Insights."""

from typing import Dict, List

import pandas as pd


TARGET_COLUMNS: Dict[str, List[str]] = {
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

RECORD_KEY = ["date", "state", "district", "pincode"]


def build_daily_target_series(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Return daily totals for one Aadhaar activity dataset.

    Duplicate records are removed using the natural record key.
    Missing calendar dates are not automatically filled with zero.
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in TARGET_COLUMNS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    target_columns = TARGET_COLUMNS[dataset_name]
    df = dataframe.copy()

    # Normalize column names consistently with the project.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )

    required_columns = RECORD_KEY + target_columns
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name}: missing required columns: {missing}"
        )

    # Parse dates explicitly because source dates use day-first format.
    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()

    # Rows without a usable date or complete record key cannot be
    # assigned safely to a reporting date/location.
    df = df.dropna(subset=RECORD_KEY).copy()

    # Convert activity counts to numeric values.
    for column in target_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Do not silently treat missing or invalid counts as zero.
    if df[target_columns].isna().any().any():
        raise ValueError(
            f"{dataset_name}: missing/non-numeric activity counts "
            "remain. Inspect and resolve them before forecasting."
        )

    # Counts should not be negative.
    if (df[target_columns] < 0).any().any():
        raise ValueError(
            f"{dataset_name}: negative activity counts detected."
        )

    # All duplicate record keys in the inspected source had identical
    # activity values. Keep one copy before summing across locations.
    df = df.drop_duplicates(subset=RECORD_KEY, keep="first")

    # Sum age-group activity across all locations for each observed date.
    daily = (
        df.groupby("date", as_index=False)[target_columns]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["total_activity"] = daily[target_columns].sum(axis=1)

    daily.insert(0, "dataset", dataset_name)

    return daily


def summarize_date_coverage(
    daily_series: pd.DataFrame,
) -> dict:
    """Summarize observed-date coverage without imputing missing dates."""
    dates = pd.DatetimeIndex(
        pd.to_datetime(daily_series["date"]).dropna().unique()
    ).sort_values()

    gaps = dates.to_series().diff().dt.days.dropna()

    return {
        "observed_dates": len(dates),
        "first_date": dates.min().date().isoformat(),
        "last_date": dates.max().date().isoformat(),
        "missing_calendar_days_between_observations": int(
            ((dates.max() - dates.min()).days + 1) - len(dates)
        ),
        "largest_gap_days": int(gaps.max()) if not gaps.empty else 0,
        "days_with_gaps_over_7": int((gaps > 7).sum()),
    }