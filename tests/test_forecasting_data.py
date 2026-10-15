from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_all_datasets
from src.forecasting_data import (
    build_daily_target_series,
    summarize_date_coverage,
)


def main():
    datasets = load_all_datasets()

    for name, dataframe in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"DATASET: {name.upper()}")
        print(f"{'=' * 60}")

        daily = build_daily_target_series(dataframe, name)
        coverage = summarize_date_coverage(daily)

        print(f"Original rows: {len(dataframe):,}")
        print(f"Observed dates: {coverage['observed_dates']}")
        print(f"Date range: {coverage['first_date']} to "
              f"{coverage['last_date']}")
        print("Largest gap (days):", coverage["largest_gap_days"])
        print("Gaps over 7 days:", coverage["days_with_gaps_over_7"])
        print("Total activity:", f"{daily['total_activity'].sum():,.0f}")

        print("\nFirst 5 observations:")
        print(daily.head().to_string(index=False))

        print("\nLast 5 observations:")
        print(daily.tail().to_string(index=False))

        # Verify that dates are ordered and unique.
        assert daily["date"].is_monotonic_increasing
        assert daily["date"].is_unique
        assert daily["total_activity"].notna().all()
        assert (daily["total_activity"] >= 0).all()

    print("\nAll forecasting data checks passed.")


if __name__ == "__main__":
    main()