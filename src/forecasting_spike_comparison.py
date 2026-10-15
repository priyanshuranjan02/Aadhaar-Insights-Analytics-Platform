from pathlib import Path

import pandas as pd

from src.config import DATASETS, SUPPORTED_FILE_TYPES

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
        print("\n" + "=" * 80)
        print(dataset.upper())

        targets = TARGETS[dataset]
        parts = []

        for ext in SUPPORTED_FILE_TYPES:
            for path in sorted(directory.rglob(f"*{ext}")):
                if path.suffix.lower() == ".csv":
                    part = pd.read_csv(path)
                else:
                    part = pd.read_excel(path)

                part["_source_file"] = path.name
                parts.append(part)

        raw = pd.concat(parts, ignore_index=True)
        raw["date"] = pd.to_datetime(
            raw["date"].astype("string").str.strip(),
            format="%d-%m-%Y",
            errors="coerce",
        )
        raw = raw.dropna(subset=["date"])

        # Keep source provenance for analysis, but exclude it from
        # duplicate identity so identical rows across files are caught.
        identity_cols = [
            col for col in raw.columns if col != "_source_file"
        ]

        clean = raw.drop_duplicates(
            subset=identity_cols
        ).copy()

        for date_string in SPIKE_DATES[dataset]:
            date = pd.Timestamp(date_string)
            before = raw.loc[raw["date"].eq(date)]
            after = clean.loc[clean["date"].eq(date)]

            if before.empty:
                print(f"\n{date_string}: no records")
                continue

            raw_total = before[targets].sum().sum()
            clean_total = after[targets].sum().sum()

            print(f"\nDate: {date_string}")
            print(f"Raw rows: {len(before):,}")
            print(f"Rows after exact deduplication: {len(after):,}")
            print(f"Duplicate rows removed: {len(before) - len(after):,}")
            print(f"Raw total activity: {raw_total:,.0f}")
            print(f"Deduplicated total activity: {clean_total:,.0f}")
            print(
                "Activity difference: "
                f"{raw_total - clean_total:,.0f}"
            )

            print("\nRows by source file:")
            print(
                before.groupby("_source_file")
                .agg(
                    rows=("date", "size"),
                    **{
                        f"{col}_sum": (col, "sum")
                        for col in targets
                    },
                )
                .to_string()
            )

    print("\nComparison complete. No files were modified.")


if __name__ == "__main__":
    main()