"""Sensitivity experiment for suspected biometric coverage anomalies.

Compares future forecasts using:
1. All observed biometric data.
2. Data excluding Dec 22-23, 2025 before feature generation.

This is a forecast-stability experiment, not proof that either forecast
is more accurate. Original data and baseline outputs remain unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series
from src.forecasting_v2 import (
    OUTPUT_DIR as BASELINE_DIR,
    HORIZON,
    make_features,
    rolling_origin_backtest,
    evaluate_final_holdout,
    forecast_future,
    prepare_observations,
)

ROOT = Path(__file__).resolve().parents[1]
SENSITIVITY_DIR = ROOT / "data" / "processed" / "forecasting_v2_sensitivity"

ANOMALOUS_DATES = pd.to_datetime([
    "2025-12-22",
    "2025-12-23",
])


def forecast_with_history(daily, label):
    """Select a model with rolling-origin validation and forecast future periods."""
    features = make_features(daily)

    summary, _, _ = rolling_origin_backtest(features)
    selected_model = str(summary.iloc[0]["Model"])

    _, _, fitted_model, feature_cols = evaluate_final_holdout(
        features, selected_model
    )

    forecasts = forecast_future(daily, fitted_model, feature_cols)
    forecasts["Scenario"] = label
    forecasts["Selected Model"] = selected_model

    return forecasts, summary


def main():
    datasets = load_all_datasets()
    SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)

    if "biometric" not in datasets:
        raise KeyError(
            "The loaded datasets do not contain a 'biometric' entry."
        )

    raw = datasets["biometric"]
    daily = build_daily_target_series(raw, "biometric")
    daily = prepare_observations(daily)

    baseline_forecast, baseline_scores = forecast_with_history(
        daily, "Baseline: all observations"
    )

    filtered_daily = daily.loc[
        ~daily["date"].isin(ANOMALOUS_DATES)
    ].copy()

    removed = daily.loc[
        daily["date"].isin(ANOMALOUS_DATES),
        ["date", "total_activity"],
    ].copy()

    if len(removed) != len(ANOMALOUS_DATES):
        print(
            "WARNING: Not all suspected dates were present in the "
            "prepared biometric daily series."
        )

    sensitivity_forecast, sensitivity_scores = forecast_with_history(
        filtered_daily, "Sensitivity: exclude Dec 22-23"
    )

    comparison = baseline_forecast[
        ["Forecast Step", "Forecast", "Last Observed Date"]
    ].rename(columns={"Forecast": "Baseline Forecast"})

    comparison["Sensitivity Forecast"] = (
        sensitivity_forecast["Forecast"].to_numpy()
    )

    comparison["Absolute Difference"] = (
        comparison["Sensitivity Forecast"]
        - comparison["Baseline Forecast"]
    ).abs()

    comparison["Difference (%)"] = np.where(
        comparison["Baseline Forecast"] != 0,
        (
            comparison["Sensitivity Forecast"]
            - comparison["Baseline Forecast"]
        ) / comparison["Baseline Forecast"].abs() * 100,
        np.nan,
    )

    comparison.to_csv(
        SENSITIVITY_DIR / "biometric_forecast_comparison.csv",
        index=False,
    )

    baseline_scores.to_csv(
        SENSITIVITY_DIR / "baseline_backtest_metrics.csv",
        index=False,
    )
    sensitivity_scores.to_csv(
        SENSITIVITY_DIR / "sensitivity_backtest_metrics.csv",
        index=False,
    )
    removed.to_csv(
        SENSITIVITY_DIR / "excluded_observations.csv",
        index=False,
    )

    baseline_forecast.to_csv(
        SENSITIVITY_DIR / "baseline_future_forecast.csv",
        index=False,
    )
    sensitivity_forecast.to_csv(
        SENSITIVITY_DIR / "sensitivity_future_forecast.csv",
        index=False,
    )

    print("\nEXCLUDED OBSERVATIONS")
    print(removed.to_string(index=False))

    print("\nBASELINE MODEL RANKING")
    print(baseline_scores.to_string(index=False))

    print("\nSENSITIVITY MODEL RANKING")
    print(sensitivity_scores.to_string(index=False))

    print("\nFUTURE FORECAST COMPARISON")
    print(comparison.to_string(index=False))

    print("\nSaved experiment results to:", SENSITIVITY_DIR)


if __name__ == "__main__":
    main()