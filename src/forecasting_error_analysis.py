from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_v2"
    / "sensitivity"
    / "matched_origin"
    / "four_scenario_predictions_common.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_v2"
    / "sensitivity"
    / "matched_origin"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "Scenario",
    "Model",
    "Target Date",
    "Actual",
    "Predicted",
    "Absolute Error",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

df["Target Date"] = pd.to_datetime(df["Target Date"])
df["Actual"] = pd.to_numeric(df["Actual"], errors="raise")
df["Predicted"] = pd.to_numeric(df["Predicted"], errors="raise")

# Recalculate errors to validate the stored values.
df["Absolute Error"] = (df["Actual"] - df["Predicted"]).abs()
df["Signed Error"] = df["Predicted"] - df["Actual"]
df["Squared Error"] = df["Signed Error"] ** 2

expected_scenarios = {
    "A_original",
    "B_exclusion",
    "C_persistence_original",
    "D_persistence_exclusion",
}

if set(df["Scenario"].unique()) != expected_scenarios:
    raise ValueError(
        "Unexpected scenario names: "
        f"{sorted(df['Scenario'].unique())}"
    )

# Confirm there is exactly one prediction per scenario and date.
if df.duplicated(["Scenario", "Target Date"]).any():
    raise ValueError("Duplicate scenario/date predictions found.")

# Confirm all scenarios have identical actual values on each date.
actual_counts = df.groupby("Target Date")["Actual"].nunique()

if (actual_counts != 1).any():
    raise ValueError("Actual values differ between scenarios.")

scenario_counts = df.groupby("Target Date")["Scenario"].nunique()

if (scenario_counts != 4).any():
    raise ValueError("Some dates do not contain all four scenarios.")


# --------------------------------------------------
# 3. Overall error metrics
# --------------------------------------------------

summary = (
    df.groupby(["Scenario", "Model"])
    .agg(
        Target_Dates=("Target Date", "nunique"),
        MAE=("Absolute Error", "mean"),
        Median_Absolute_Error=("Absolute Error", "median"),
        RMSE=("Squared Error", lambda x: np.sqrt(x.mean())),
        Mean_Signed_Error=("Signed Error", "mean"),
        Maximum_Absolute_Error=("Absolute Error", "max"),
    )
    .reset_index()
)

summary.to_csv(OUTPUT_DIR / "overall_error_metrics.csv", index=False)


# --------------------------------------------------
# 4. Compare original vs exclusion by target date
# --------------------------------------------------

comparison = df.pivot(
    index="Target Date",
    columns="Scenario",
    values="Absolute Error",
).reset_index()

comparison = comparison.rename(
    columns={
        "A_original": "LR_Original_Error",
        "B_exclusion": "LR_Exclusion_Error",
        "C_persistence_original": "Persistence_Original_Error",
        "D_persistence_exclusion": "Persistence_Exclusion_Error",
    }
)

comparison["LR_Error_Reduction"] = (
    comparison["LR_Original_Error"]
    - comparison["LR_Exclusion_Error"]
)

comparison["Persistence_Error_Reduction"] = (
    comparison["Persistence_Original_Error"]
    - comparison["Persistence_Exclusion_Error"]
)

# Positive reduction means the exclusion scenario had lower error.
comparison["LR_Exclusion_Improved"] = (
    comparison["LR_Error_Reduction"] > 0
)

comparison["Persistence_Exclusion_Improved"] = (
    comparison["Persistence_Error_Reduction"] > 0
)

comparison.to_csv(
    OUTPUT_DIR / "date_level_error_comparison.csv",
    index=False,
)


# --------------------------------------------------
# 5. Improvement consistency
# --------------------------------------------------

def compare_scenarios(
    original_column: str,
    exclusion_column: str,
    model_name: str,
) -> dict:
    original = comparison[original_column]
    exclusion = comparison[exclusion_column]

    original_mae = original.mean()
    exclusion_mae = exclusion.mean()

    original_rmse = np.sqrt(np.mean(original**2))
    exclusion_rmse = np.sqrt(np.mean(exclusion**2))

    mae_reduction_pct = (
        (original_mae - exclusion_mae) / original_mae * 100
        if original_mae != 0
        else np.nan
    )

    rmse_reduction_pct = (
        (original_rmse - exclusion_rmse) / original_rmse * 100
        if original_rmse != 0
        else np.nan
    )

    return {
        "Model": model_name,
        "Target_Dates": len(comparison),
        "Dates_Improved": int((exclusion < original).sum()),
        "Dates_Worsened": int((exclusion > original).sum()),
        "Dates_Unchanged": int((exclusion == original).sum()),
        "Original_MAE": original_mae,
        "Exclusion_MAE": exclusion_mae,
        "MAE_Reduction_Percent": mae_reduction_pct,
        "Original_RMSE": original_rmse,
        "Exclusion_RMSE": exclusion_rmse,
        "RMSE_Reduction_Percent": rmse_reduction_pct,
    }


consistency = pd.DataFrame(
    [
        compare_scenarios(
            "LR_Original_Error",
            "LR_Exclusion_Error",
            "LinearRegression",
        ),
        compare_scenarios(
            "Persistence_Original_Error",
            "Persistence_Exclusion_Error",
            "Persistence",
        ),
    ]
)

consistency.to_csv(
    OUTPUT_DIR / "improvement_consistency.csv",
    index=False,
)


# --------------------------------------------------
# 6. Identify the largest errors
# --------------------------------------------------

top_errors = (
    df.sort_values("Absolute Error", ascending=False)
    .groupby("Scenario", group_keys=False)
    .head(10)
    .sort_values(["Scenario", "Absolute Error"], ascending=[True, False])
)

top_errors.to_csv(
    OUTPUT_DIR / "top_10_errors_per_scenario.csv",
    index=False,
)


# --------------------------------------------------
# 7. Print findings
# --------------------------------------------------

print("\nOverall error metrics:")
print(summary.round(2).to_string(index=False))

print("\nImprovement consistency:")
print(consistency.round(2).to_string(index=False))

print("\nLargest prediction errors:")
print(
    top_errors[
        ["Scenario", "Target Date", "Actual", "Predicted", "Absolute Error"]
    ].round(2).to_string(index=False)
)

print("\nValidation passed:")
print(f"Common target dates: {len(comparison)}")
print("All four scenarios have matching target dates and actual values.")

print(f"\nAnalysis outputs saved to:\n{OUTPUT_DIR}")