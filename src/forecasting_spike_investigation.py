from pathlib import Path

import pandas as pd

from src.config import DATASETS, SUPPORTED_FILE_TYPES

DATE_FORMAT = "%d-%m-%Y"

TARGETS = {
    "enrolment": ["age_0_5", "age_5_17", "age_18_greater"],
    "demographic": ["demo_age_5_17", "demo_age_17_"],
    "biometric": ["bio_age_5_17", "bio_age_17_"],
}

SPIKE_DATES = {
    "enrolment": [
        "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01"
    ],
    "demographic": [
        "2025-03-01", "2025-04-01", "2025-05-01",
        "2025-06-01", "2025-07-01"
    ],
    "biometric": [
        "2025-03-01", "2025-04-01", "2025-05-01",
        "2025-06-01", "2025-07-01"
    ],
}


def main():
    for dataset, directory in DATASETS.items():
        print("\n" + "=" * 85)
        print(dataset.upper())

        targets = TARGETS[dataset]
        files = sorted(
            path
            for ext in SUPPORTED_FILE_TYPES
            for path in directory.rglob(f"*{ext}")
        )

        frames = []

        for path in files:
            if path.suffix.lower() == ".csv":
                part = pd.read_csv(path)
            else:
                part = pd.read_excel(path)

            part["_source_file"] = path.name
            part["date"] = pd.to_datetime(
                part["date"].astype("string").str.strip(),
                format=DATE_FORMAT,
                errors="coerce",
            )

            for col in targets:
                part[col] = pd.to_numeric(
                    part[col], errors="coerce"
                )

            frames.append(part)

        raw = pd.concat(frames, ignore_index=True)
        raw = raw.dropna(subset=["date"])

        # Compare flagged dates with ordinary dates in the same dataset.
        all_daily = raw.groupby("date").agg(
            row_count=("date", "size"),
            **{
                f"{col}_sum": (col, "sum")
                for col in targets
            },
            **{
                f"{col}_max_row": (col, "max")
                for col in targets
            },
        )

        all_daily["total_activity"] = all_daily[
            [f"{col}_sum" for col in targets]
        ].sum(axis=1)

        all_daily["rows_after_exact_dedup"] = (
            raw.drop_duplicates()
            .groupby("date")
            .size()
        )

        print("\nOverall daily row-count distribution:")
        print(all_daily["row_count"].describe().to_string())

        for date_string in SPIKE_DATES[dataset]:
            date = pd.Timestamp(date_string)
            if date not in all_daily.index:
                print(f"\n{date_string}: date not found")
                continue

            print(f"\n--- {date_string} ---")
            print(all_daily.loc[date].to_string())

            subset = raw.loc[raw["date"].eq(date)].copy()

            print("\nRows by source file:")
            print(
                subset.groupby("_source_file")
                .agg(
                    rows=("date", "size"),
                    **{
                        f"{col}_sum": (col, "sum")
                        for col in targets
                    },
                )
                .to_string()
            )

            print("\nGeographic coverage:")
            for col in ["state", "district", "pincode"]:
                if col in subset.columns:
                    print(
                        f"  Unique {col}: "
                        f"{subset[col].nunique(dropna=True):,}"
                    )

            print("\nExact duplicate rows on this date:")
            print(
                f"  {subset.drop(columns=['_source_file']).duplicated().sum():,}"
            )

        print("\nLargest daily totals:")
        print(
            all_daily.nlargest(10, "total_activity")[
                ["row_count", "total_activity"]
            ].to_string()
        )

    print("\nInvestigation complete. No files were modified.")

if __name__ == "__main__":
    main()