"""Consolidate existing forecasting experiment metrics into one report."""

from pathlib import Path

import pandas as pd

from src.forecasting_v2 import OUTPUT_DIR


BASE = OUTPUT_DIR / "sensitivity" / "matched_origin"
REPORT_DIR = OUTPUT_DIR / "consolidated_report"

SOURCES = [
    {
        "file": BASE / "four_scenario_comparison.csv",
        "experiment": "Matched-origin sensitivity",
        "date_column": "Matched Target Dates",
        "notes": "58-date comparison; exclusion scenarios omit flagged dates from training history.",
    },
    {
        "file": BASE / "baseline_comparison" / "baseline_metrics.csv",
        "experiment": "Fixed baseline comparison",
        "date_column": "Target Dates",
        "notes": "60-date one-step-ahead comparison.",
    },
    {
        "file": BASE / "adaptive_selection" / "adaptive_metrics.csv",
        "experiment": "Adaptive selection (14)",
        "date_column": "Target Dates",
        "notes": "60-date one-step-ahead comparison.",
    },
    {
        "file": BASE / "window_experiment" / "window_metrics.csv",
        "experiment": "Adaptive validation-window experiment",
        "date_column": "Target Dates",
        "notes": "Separate row for each model and validation window.",
    },
    {
        "file": BASE / "robust_comparison" / "robust_metrics.csv",
        "experiment": "Robust regression comparison",
        "date_column": "Matched Target Dates",
        "notes": "60-date one-step-ahead comparison.",
    },
]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    missing = []

    for source in SOURCES:
        path = source["file"]

        if not path.exists():
            missing.append(str(path))
            continue

        df = pd.read_csv(path)

        required = {"Model", "MAE", "RMSE", "R2", source["date_column"]}
        absent = required - set(df.columns)

        if absent:
            print(f"Skipping {path}: missing columns {sorted(absent)}")
            continue

        for _, row in df.iterrows():
            model = str(row["Model"])
            window = row["Window"] if "Window" in df.columns else None
            scenario = row["Scenario"] if "Scenario" in df.columns else None

            if pd.isna(window):
                window = None
            if pd.isna(scenario):
                scenario = None

            notes = source["notes"]
            if scenario in {"B_exclusion", "D_persistence_exclusion"}:
                notes += " Dec 22-23 excluded from training history."
            elif scenario in {"A_original", "C_persistence_original"}:
                notes += " Original training history."

            records.append({
                "Experiment": source["experiment"],
                "Scenario": scenario,
                "Model": model,
                "Validation Window": window,
                "Target Dates": int(row[source["date_column"]]),
                "MAE": float(row["MAE"]),
                "RMSE": float(row["RMSE"]),
                "R2": float(row["R2"]),
                "Notes": notes,
                "Source File": str(path),
            })

    if not records:
        raise RuntimeError(
            "No experiment results were loaded. Check the listed file paths."
        )

    report = pd.DataFrame(records)
    report = report.sort_values(
        ["Experiment", "Validation Window", "RMSE"],
        na_position="first",
    ).reset_index(drop=True)

    output = REPORT_DIR / "forecasting_experiment_comparison.csv"
    report.to_csv(output, index=False)

    print("\n=== CONSOLIDATED FORECASTING EXPERIMENTS ===")
    print(
        report[
            [
                "Experiment",
                "Scenario",
                "Model",
                "Validation Window",
                "Target Dates",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].round(3).to_string(index=False)
    )

    print(f"\nRows consolidated: {len(report)}")
    print(f"Report saved to: {output}")

    if missing:
        print("\nSources not found (these experiments may not have run):")
        for path in missing:
            print(f"- {path}")


if __name__ == "__main__":
    main()
