"""Data-quality diagnostics for Aadhaar Insights forecasting.

This module flags unusually low reporting volumes for investigation.
It does not modify raw data or exclude observations from forecasting.
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_all_datasets
from src.forecasting_data import (
    build_daily_target_series,
    summarize_date_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT / "data" / "processed" / "forecasting_v2" / "diagnostics"
)

# Initial screening threshold, not an automatic data-cleaning rule.
LOW_VOLUME_RATIO = 0.25

RECORD_KEY = ["date", "state", "district", "pincode"]


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize source column names consistently with forecasting_data.py."""
    df = dataframe.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )
    return df


def build_raw_daily_counts(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Count raw source rows by parsed reporting date before deduplication."""
    df = normalize_columns(dataframe)

    missing = [column for column in RECORD_KEY if column not in df.columns]
    if missing:
        raise ValueError(f"Missing record-key columns: {missing}")

    df["date"] = pd.to_datetime(
        df["date"], dayfirst=True, errors="coerce"
    ).dt.normalize()

    # Count source rows with a valid reporting date.
    # Do not deduplicate: the purpose is to measure source-row volume.
    df = df.dropna(subset=["date"])

    return (
        df.groupby("date")
        .size()
        .rename("raw_source_rows")
        .reset_index()
    )


def build_dataset_diagnostics(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Combine raw row counts with validated, deduplicated daily activity."""
    raw_counts = build_raw_daily_counts(dataframe)

    daily = build_daily_target_series(dataframe, dataset_name)

    report = daily[["date", "total_activity"]].merge(
        raw_counts,
        on="date",
        how="outer",
        validate="one_to_one",
    )

    report["dataset"] = dataset_name
    report = report.sort_values("date").reset_index(drop=True)

    # Calculate medians across observed dates only.
    row_median = report["raw_source_rows"].median()
    activity_median = report["total_activity"].median()

    report["rows_vs_median_ratio"] = (
        report["raw_source_rows"] / row_median
        if pd.notna(row_median) and row_median > 0
        else pd.NA
    )

    report["activity_vs_median_ratio"] = (
        report["total_activity"] / activity_median
        if pd.notna(activity_median) and activity_median > 0
        else pd.NA
    )

    report["low_row_volume_flag"] = (
        report["rows_vs_median_ratio"] < LOW_VOLUME_RATIO
    ).fillna(False)

    report["low_activity_flag"] = (
        report["activity_vs_median_ratio"] < LOW_VOLUME_RATIO
    ).fillna(False)

    report["investigate_flag"] = (
        report["low_row_volume_flag"] | report["low_activity_flag"]
    )

    # Keep reporting gaps explicit; never insert missing dates as zeroes.
    coverage = summarize_date_coverage(daily)

    summary = {
        "dataset": dataset_name,
        **coverage,
        "median_raw_source_rows": float(row_median),
        "median_daily_activity": float(activity_median),
        "low_row_volume_dates": int(report["low_row_volume_flag"].sum()),
        "low_activity_dates": int(report["low_activity_flag"].sum()),
        "dates_flagged_for_investigation": int(
            report["investigate_flag"].sum()
        ),
        "screening_threshold_ratio": LOW_VOLUME_RATIO,
    }

    # Put the most useful columns first.
    columns = [
        "dataset",
        "date",
        "raw_source_rows",
        "total_activity",
        "rows_vs_median_ratio",
        "activity_vs_median_ratio",
        "low_row_volume_flag",
        "low_activity_flag",
        "investigate_flag",
    ]

    return report[columns], summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    all_reports = []
    summaries = []

    for dataset_name, dataframe in datasets.items():
        print(f"\nDiagnosing {dataset_name}...")

        report, summary = build_dataset_diagnostics(
            dataframe, dataset_name
        )

        output_path = (
            OUTPUT_DIR / f"{dataset_name}_data_quality.csv"
        )
        report.to_csv(output_path, index=False)

        all_reports.append(report)
        summaries.append(summary)

        print(f"Saved: {output_path}")
        print(f"Observed dates: {summary['observed_dates']}")
        print(f"Date range: {summary['first_date']} to {summary['last_date']}")
        print(
            "Dates flagged for investigation: "
            f"{summary['dates_flagged_for_investigation']}"
        )

        flagged = report.loc[
            report["investigate_flag"],
            [
                "date",
                "raw_source_rows",
                "total_activity",
                "rows_vs_median_ratio",
                "activity_vs_median_ratio",
            ],
        ]

        if flagged.empty:
            print("No dates crossed the initial screening threshold.")
        else:
            print("\nFlagged dates (review; do not assume invalid):")
            print(flagged.to_string(index=False))

    combined = pd.concat(all_reports, ignore_index=True)
    combined_path = OUTPUT_DIR / "all_datasets_data_quality.csv"
    combined.to_csv(combined_path, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / "data_quality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nCombined report: {combined_path}")
    print(f"Summary report:  {summary_path}")
    print("\nDiagnostics complete. No source data or forecast outputs were changed.")

if __name__ == "__main__":
    main()