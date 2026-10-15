"""Sensitivity experiment for suspected biometric reporting anomalies.

The original forecasting pipeline and outputs remain unchanged.
The exclusion scenario is experimental and does not assert that the
excluded source observations are invalid.
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series
from src.forecasting_v2 import (
    make_features,
    rolling_origin_backtest,
    evaluate_final_holdout,
    forecast_future,
    HORIZON,
    OUTPUT_DIR,
)

ROOT = Path(__file__).resolve().parents[1]
SENSITIVITY_DIR = OUTPUT_DIR / "sensitivity"

SUSPECT_DATES = pd.to_datetime([
    "2025-12-22",
    "2025-12-23",
])


def run_scenario(daily: pd.DataFrame, scenario_name: str) -> dict:
    """Run the existing forecasting evaluation for one daily series."""
    features = make_features(daily)

    backtest_summary, folds, predictions = rolling_origin_backtest(
        features
    )
    selected_model = str(backtest_summary.iloc[0]["Model"])

    final_metrics, holdout, fitted_model, feature_cols = (
        evaluate_final_holdout(features, selected_model)
    )

    future = forecast_future(daily, fitted_model, feature_cols)

    scenario_dir = SENSITIVITY_DIR / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    backtest_summary.to_csv(
        scenario_dir / "rolling_origin_metrics.csv", index=False
    )
    folds.to_csv(
        scenario_dir / "rolling_origin_folds.csv", index=False
    )
    predictions.to_csv(
        scenario_dir / "rolling_origin_predictions.csv", index=False
    )
    final_metrics.to_csv(
        scenario_dir / "final_holdout_metrics.csv", index=False
    )
    holdout.to_csv(
        scenario_dir / "holdout_predictions.csv", index=False
    )
    future.to_csv(
        scenario_dir / "future_forecast.csv", index=False
    )

    return {
        "scenario": scenario_name,
        "observations": len(daily),
        "first_date": daily["date"].min(),
        "last_date": daily["date"].max(),
        "selected_model": selected_model,
        "rolling_origin_rmse": float(
            backtest_summary.iloc[0]["RMSE"]
        ),
        "rolling_origin_mae": float(
            backtest_summary.iloc[0]["MAE"]
        ),
        "holdout_metrics": final_metrics,
        "holdout_predictions": holdout,
        "backtest_predictions": predictions,
    }


def main() -> None:
    SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    daily_original = build_daily_target_series(
        datasets["biometric"], "biometric"
    )

    # Scenario A: existing data-preparation behavior.
    original = run_scenario(
        daily_original.copy(),
        "A_original",
    )

    # Scenario B: experimental exclusion of the two suspect dates.
    daily_excluded = daily_original[
        ~daily_original["date"].isin(SUSPECT_DATES)
    ].copy().reset_index(drop=True)

    excluded = run_scenario(
        daily_excluded,
        "B_exclude_dec22_dec23",
    )

    summary_rows = []
    for result in [original, excluded]:
        summary_rows.append({
            "scenario": result["scenario"],
            "observations": result["observations"],
            "first_date": result["first_date"],
            "last_date": result["last_date"],
            "selected_model": result["selected_model"],
            "rolling_origin_rmse": result["rolling_origin_rmse"],
            "rolling_origin_mae": result["rolling_origin_mae"],
        })

    pd.DataFrame(summary_rows).to_csv(
        SENSITIVITY_DIR / "scenario_summary.csv",
        index=False,
    )

    # Keep the comparison explicit about whether holdout dates match.
    a_dates = pd.to_datetime(
        original["holdout_predictions"]["Date"]
    ).dt.normalize()
    b_dates = pd.to_datetime(
        excluded["holdout_predictions"]["Date"]
    ).dt.normalize()

    date_comparison = pd.DataFrame({
        "original_holdout_date": a_dates,
        "exclusion_holdout_date": b_dates,
    })
    date_comparison["same_date"] = (
        date_comparison["original_holdout_date"]
        == date_comparison["exclusion_holdout_date"]
    )
    date_comparison.to_csv(
        SENSITIVITY_DIR / "holdout_date_comparison.csv",
        index=False,
    )

    print("\nSensitivity experiment complete.")
    print("Output directory:", SENSITIVITY_DIR)
    print("\nScenario summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nHoldout dates:")
    print(date_comparison.to_string(index=False))

    if not date_comparison["same_date"].all():
        print(
            "\nWARNING: Holdout dates differ between scenarios. "
            "Do not directly compare their final-holdout metrics as "
            "if they evaluated the same observations."
        )

    print(
        "\nThis is a sensitivity experiment, not a data correction. "
        "Original source data and baseline forecast outputs were not changed."
    )

if __name__ == "__main__":
    main()