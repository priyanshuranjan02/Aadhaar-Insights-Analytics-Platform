"""Adaptive model selection for biometric activity forecasting.

At each target date, select between Linear Regression and Mean Baseline
using a past-only, rolling validation window.

Run from the project root:
    python -m src.forecasting_adaptive_selection
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series
from src.forecasting_v2 import (
    prepare_observations,
    LAGS,
    ROLLING_WINDOW,
    OUTPUT_DIR,
)

OUT_DIR = (
    OUTPUT_DIR
    / "sensitivity"
    / "matched_origin"
    / "adaptive_selection"
)

MIN_HISTORY_OBSERVATIONS = 24
VALIDATION_WINDOW = 14


def linear_prediction(history):
    """Predict the next observation using historical lag features."""
    max_lag = max(max(LAGS), ROLLING_WINDOW)

    if len(history) <= max_lag:
        raise ValueError("Insufficient history for Linear Regression.")

    X_train = []
    y_train = []

    for i in range(max_lag, len(history)):
        previous = history[:i]

        features = [previous[-lag] for lag in LAGS]
        features.append(float(np.mean(previous[-ROLLING_WINDOW:])))

        X_train.append(features)
        y_train.append(history[i])

    latest_features = [history[-lag] for lag in LAGS]
    latest_features.append(float(np.mean(history[-ROLLING_WINDOW:])))

    model = LinearRegression()
    model.fit(np.asarray(X_train), np.asarray(y_train))

    return float(model.predict(np.asarray([latest_features]))[0])


def mean_prediction(history):
    """Predict using the mean of available historical observations."""
    return float(np.mean(history))



def rolling_validation_mae(history, predictor):
    """Evaluate a predictor using the latest eligible historical targets.

    Each validation prediction uses at least MIN_HISTORY_OBSERVATIONS
    observations before its validation target.
    """
    errors = []

    # Validation targets must each have at least 24 prior observations.
    first_validation_index = MIN_HISTORY_OBSERVATIONS

    # Keep at most the most recent VALIDATION_WINDOW eligible targets.
    start = max(
        first_validation_index,
        len(history) - VALIDATION_WINDOW,
    )

    for target_index in range(start, len(history)):
        past = history[:target_index]
        actual = float(history[target_index])

        prediction = predictor(past)
        errors.append(abs(actual - prediction))

    if not errors:
        return np.inf

    return float(np.mean(errors))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    daily = build_daily_target_series(
        datasets["biometric"],
        "biometric",
    )

    data = prepare_observations(daily).copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    data = (
        data.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    dates = data["date"].tolist()
    values = data["total_activity"].to_numpy(dtype=float)

    predictors = {
        "LinearRegression": linear_prediction,
        "MeanBaseline": mean_prediction,
    }

    rows = []

    # Start after the same minimum history requirement as the
    # established matched-origin experiments.
    for target_index, target_date in enumerate(dates):
        history = values[:target_index]

        if len(history) < MIN_HISTORY_OBSERVATIONS:
            continue

        actual = float(values[target_index])


        validation_scores = {
            name: rolling_validation_mae(history, predictor)
            for name, predictor in predictors.items()
        }

        if not any(np.isfinite(score) for score in validation_scores.values()):
            print(
                f"Warning: no valid selection score for {target_date.date()}; "
                "using the documented fallback."
            )

        valid_scores = {
            name: score
            for name, score in validation_scores.items()
            if np.isfinite(score)
        }

        if valid_scores:
            selected_model = min(valid_scores, key=valid_scores.get)
        else:
            # No valid past-only validation score exists yet.
            # Use Linear Regression as a documented fallback.
            selected_model = "LinearRegression"


        # Evaluate all candidates and the selected rule on the same date.
        predictions = {
            name: predictor(history)
            for name, predictor in predictors.items()
        }

        predictions["AdaptiveSelection"] = predictions[selected_model]

        for model_name, prediction in predictions.items():
            rows.append(
                {
                    "Target Date": target_date,
                    "Model": model_name,
                    "Actual": actual,
                    "Predicted": prediction,
                    "Absolute Error": abs(actual - prediction),
                    "Training Observations": len(history),
                    "Selected Model": selected_model,
                    "LR Validation MAE": validation_scores[
                        "LinearRegression"
                    ],
                    "Mean Validation MAE": validation_scores[
                        "MeanBaseline"
                    ],
                    "Validation Window": VALIDATION_WINDOW,
                    "Forecast Type": "One-step-ahead",
                }
            )

    predictions_df = pd.DataFrame(rows)

    if predictions_df.empty:
        raise RuntimeError("No predictions generated.")

    counts = predictions_df.groupby("Target Date")["Model"].nunique()

    expected_models = {
        "LinearRegression",
        "MeanBaseline",
        "AdaptiveSelection",
    }

    if not (counts == len(expected_models)).all():
        raise RuntimeError("Matched-date validation failed.")

    actual_counts = predictions_df.groupby("Target Date")["Actual"].nunique()

    if not (actual_counts == 1).all():
        raise RuntimeError("Actual values differ between models.")

    predictions_df = predictions_df.sort_values(
        ["Target Date", "Model"]
    ).reset_index(drop=True)

    predictions_df.to_csv(
        OUT_DIR / "adaptive_predictions.csv",
        index=False,
    )

    summary_rows = []

    for model_name, group in predictions_df.groupby("Model"):
        actual = group["Actual"].to_numpy(dtype=float)
        predicted = group["Predicted"].to_numpy(dtype=float)
        errors = predicted - actual

        ss_total = np.sum((actual - actual.mean()) ** 2)
        ss_residual = np.sum(errors**2)

        summary_rows.append(
            {
                "Model": model_name,
                "Target Dates": group["Target Date"].nunique(),
                "MAE": np.mean(np.abs(errors)),
                "RMSE": np.sqrt(np.mean(errors**2)),
                "R2": (
                    1 - ss_residual / ss_total
                    if ss_total > 0
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("RMSE")

    summary.to_csv(
        OUT_DIR / "adaptive_metrics.csv",
        index=False,
    )

    selection_counts = (
        predictions_df.loc[
            predictions_df["Model"] == "AdaptiveSelection",
            "Selected Model",
        ]
        .value_counts()
    )

    print("\nAdaptive model-selection comparison:")
    print(summary.round(3).to_string(index=False))

    print("\nModels selected across target dates:")
    print(selection_counts.to_string())

    print("\nValidation:")
    print(
        "Matched target dates:",
        predictions_df["Target Date"].nunique(),
    )
    print("Validation window:", VALIDATION_WINDOW)
    print("Minimum history:", MIN_HISTORY_OBSERVATIONS)
    print("All models use the same target dates and actual values.")

    print(f"\nOutputs saved to:\n{OUT_DIR}")

if __name__ == "__main__":
    main()