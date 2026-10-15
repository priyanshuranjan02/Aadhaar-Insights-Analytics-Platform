"""Matched-target sensitivity experiment for biometric forecasting.

Compares Linear Regression and persistence under original and exclusion
histories on identical target dates. The production forecasting pipeline and
raw data are not modified.

Run from the project root:
    python -m src.forecasting_matched_origin
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series
from src.forecasting_v2 import (
    prepare_observations,
    score_predictions,
    LAGS,
    ROLLING_WINDOW,
    OUTPUT_DIR,
)

OUT_DIR = OUTPUT_DIR / "sensitivity" / "matched_origin"

SUSPECT_DATES = pd.to_datetime(["2025-12-22", "2025-12-23"]).normalize()
MODEL_NAME = "LinearRegression"
MIN_HISTORY_OBSERVATIONS = 24


def _apply_exclusions(data, excluded_dates=None):
    """Remove specified dates without changing the source data."""
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()

    if excluded_dates is not None and len(excluded_dates) > 0:
        dates = pd.to_datetime(list(excluded_dates)).normalize()
        result = result[~result["date"].isin(dates)]

    return (
        result.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _history_and_actual(data, target_date):
    """Return values observed before target_date and its actual value."""
    target_date = pd.Timestamp(target_date).normalize()
    target_rows = data.loc[
        data["date"] == target_date, "total_activity"
    ]

    if target_rows.empty:
        return None, None

    history = data.loc[
        data["date"] < target_date, "total_activity"
    ].to_numpy(dtype=float)

    return history, float(target_rows.iloc[0])


def fit_and_forecast_to_target(
    scenario_data,
    target_date,
    excluded_dates=None,
):
    """Make an expanding-window, one-step-ahead Linear Regression prediction."""
    # Validate the input using the project's established data preparation.
    data = prepare_observations(scenario_data).copy()
    data = _apply_exclusions(data, excluded_dates)

    history, actual = _history_and_actual(data, target_date)
    if history is None:
        return None

    max_lag = max(max(LAGS), ROLLING_WINDOW)
    if len(history) < MIN_HISTORY_OBSERVATIONS or len(history) <= max_lag:
        return None

    X_train = []
    y_train = []

    # Each training row uses only observations preceding its target.
    for i in range(max_lag, len(history)):
        previous = history[:i]
        features = [previous[-lag] for lag in LAGS]
        features.append(float(np.mean(previous[-ROLLING_WINDOW:])))
        X_train.append(features)
        y_train.append(history[i])

    if not X_train:
        return None

    latest_features = [history[-lag] for lag in LAGS]
    latest_features.append(float(np.mean(history[-ROLLING_WINDOW:])))

    model = LinearRegression()
    model.fit(np.asarray(X_train), np.asarray(y_train))
    prediction = float(model.predict(np.asarray([latest_features]))[0])

    return {
        "Target Date": pd.Timestamp(target_date).normalize(),
        "Actual": actual,
        "Predicted": prediction,
        "Absolute Error": abs(actual - prediction),
        "Training Observations": len(history),
        "Forecast Type": "One-step-ahead",
    }


def persistence_prediction(
    scenario_data,
    target_date,
    excluded_dates=None,
):
    """Predict the target using the most recent prior reporting observation."""
    data = prepare_observations(scenario_data).copy()
    data = _apply_exclusions(data, excluded_dates)

    history, actual = _history_and_actual(data, target_date)
    if history is None or len(history) < MIN_HISTORY_OBSERVATIONS:
        return None

    prediction = float(history[-1])

    return {
        "Target Date": pd.Timestamp(target_date).normalize(),
        "Actual": actual,
        "Predicted": prediction,
        "Absolute Error": abs(actual - prediction),
        "Training Observations": len(history),
        "Forecast Type": "One-step-ahead",
    }


def _score_group(group, scenario, model_name):
    """Calculate metrics for a single scenario's selected target dates."""
    if group.empty:
        raise RuntimeError(f"No predictions available for scenario {scenario}.")

    scores = score_predictions(
        group["Actual"].to_numpy(dtype=float),
        group["Predicted"].to_numpy(dtype=float),
    )

    return {
        "Scenario": scenario,
        "Model": model_name,
        "Matched Target Dates": group["Target Date"].nunique(),
        **scores,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    daily = build_daily_target_series(datasets["biometric"], "biometric")

    # Establish the common source series, then construct the exclusion scenario.
    original = prepare_observations(daily)
    original["date"] = pd.to_datetime(original["date"]).dt.normalize()
    excluded = _apply_exclusions(original, SUSPECT_DATES)

    common_targets = sorted(
        (set(original["date"]) & set(excluded["date"])) - set(SUSPECT_DATES)
    )

    scenario_specs = [
        ("A_original", "LinearRegression", original, (), fit_and_forecast_to_target),
        (
            "B_exclusion",
            "LinearRegression",
            original,
            SUSPECT_DATES,
            fit_and_forecast_to_target,
        ),
        (
            "C_persistence_original",
            "Persistence",
            original,
            (),
            persistence_prediction,
        ),
        (
            "D_persistence_exclusion",
            "Persistence",
            original,
            SUSPECT_DATES,
            persistence_prediction,
        ),
    ]

    rows = []
    for target_date in common_targets:
        for scenario, model_name, scenario_data, removed, predictor in scenario_specs:
            result = predictor(
                scenario_data,
                target_date,
                excluded_dates=removed,
            )
            if result is not None:
                rows.append({
                    "Scenario": scenario,
                    "Model": model_name,
                    **result,
                })

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No predictions were generated.")

    predictions.to_csv(
        OUT_DIR / "all_scenario_predictions.csv",
        index=False,
    )

    # Retain only dates that all four scenarios predicted and for which each
    # scenario had the required amount of historical data.
    eligible = predictions[
        predictions["Training Observations"] >= MIN_HISTORY_OBSERVATIONS
    ].copy()

    expected_scenarios = {spec[0] for spec in scenario_specs}
    counts = eligible.groupby("Target Date")["Scenario"].agg(
        lambda values: set(values)
    )
    matched_dates = sorted(
        date for date, scenarios in counts.items()
        if scenarios == expected_scenarios
    )

    if not matched_dates:
        raise RuntimeError(
            "No common target dates have predictions from all four scenarios."
        )

    matched = eligible[eligible["Target Date"].isin(matched_dates)].copy()
    matched = matched.sort_values(["Target Date", "Scenario"]).reset_index(drop=True)

    # Defensive checks: every target must have exactly one row per scenario,
    # and all four scenarios must use the same actual target value.
    per_date_counts = matched.groupby("Target Date")["Scenario"].nunique()
    if not (per_date_counts == len(expected_scenarios)).all():
        raise RuntimeError("Matched-date validation failed: scenario coverage differs.")

    actual_counts = matched.groupby("Target Date")["Actual"].nunique()
    if not (actual_counts == 1).all():
        raise RuntimeError("Matched-date validation failed: actual values differ.")

    matched.to_csv(
        OUT_DIR / "four_scenario_predictions_common.csv",
        index=False,
    )

    summaries = []
    for scenario, model_name, *_ in scenario_specs:
        group = matched[matched["Scenario"] == scenario]
        summaries.append(_score_group(group, scenario, model_name))

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(
        OUT_DIR / "four_scenario_comparison.csv",
        index=False,
    )

    print("\nMatched-target sensitivity experiment complete.")
    print("Output directory:", OUT_DIR)
    print(f"Minimum history requirement: {MIN_HISTORY_OBSERVATIONS}")
    print(f"Common target dates retained for all four scenarios: {len(matched_dates)}")
    print("\nFour-scenario comparison:")
    print(comparison.to_string(index=False))
    print("\nSaved predictions: four_scenario_predictions_common.csv")


if __name__ == "__main__":
    main()