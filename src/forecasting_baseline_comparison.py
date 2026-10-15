"""Compare Linear Regression, Persistence, and Mean Baseline.

Uses identical target dates and expanding historical windows.
Does not modify the production forecasting pipeline or raw data.

Run from the project root:
    python -m src.forecasting_baseline_comparison
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
    / "baseline_comparison"
)

MIN_HISTORY_OBSERVATIONS = 24


def predict_linear(history):
    """Train Linear Regression using only past observations."""
    max_lag = max(max(LAGS), ROLLING_WINDOW)

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


def predict_persistence(history):
    """Use the most recent available observation."""
    return float(history[-1])


def predict_mean(history):
    """Use the mean of all observations available before the target."""
    return float(np.mean(history))


def calculate_metrics(group):
    """Calculate error metrics for a set of matched predictions."""
    actual = group["Actual"].to_numpy(dtype=float)
    predicted = group["Predicted"].to_numpy(dtype=float)
    errors = predicted - actual

    ss_total = np.sum((actual - actual.mean()) ** 2)
    ss_residual = np.sum(errors**2)

    r2 = (
        1 - ss_residual / ss_total
        if ss_total > 0
        else np.nan
    )

    return {
        "Target Dates": len(group),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "R2": float(r2),
        "Mean Signed Error": float(np.mean(errors)),
    }


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
    activity_by_date = dict(
        zip(data["date"], data["total_activity"])
    )

    predictors = {
        "LinearRegression": predict_linear,
        "Persistence": predict_persistence,
        "MeanBaseline": predict_mean,
    }

    rows = []

    # Every prediction uses only observations dated before its target.
    for target_date in dates:
        history_df = data.loc[data["date"] < target_date]
        history = history_df["total_activity"].to_numpy(dtype=float)

        if len(history) < MIN_HISTORY_OBSERVATIONS:
            continue

        actual = float(activity_by_date[target_date])

        for model_name, predictor in predictors.items():
            prediction = predictor(history)

            rows.append(
                {
                    "Target Date": target_date,
                    "Model": model_name,
                    "Actual": actual,
                    "Predicted": prediction,
                    "Absolute Error": abs(actual - prediction),
                    "Training Observations": len(history),
                    "Forecast Type": "One-step-ahead",
                }
            )

    predictions = pd.DataFrame(rows)

    if predictions.empty:
        raise RuntimeError(
            "No predictions generated. Check the observation count "
            "and minimum history requirement."
        )

    # Ensure all models have predictions for exactly the same dates.
    counts = predictions.groupby("Target Date")["Model"].nunique()

    if not (counts == len(predictors)).all():
        raise RuntimeError(
            "Matched-date validation failed: models have different coverage."
        )

    actual_counts = predictions.groupby("Target Date")["Actual"].nunique()

    if not (actual_counts == 1).all():
        raise RuntimeError(
            "Matched-date validation failed: actual values differ."
        )

    predictions = predictions.sort_values(
        ["Target Date", "Model"]
    ).reset_index(drop=True)

    predictions.to_csv(
        OUT_DIR / "baseline_predictions.csv",
        index=False,
    )

    summary_rows = []

    for model_name, group in predictions.groupby("Model"):
        summary_rows.append(
            {
                "Model": model_name,
                **calculate_metrics(group),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("RMSE")

    summary.to_csv(
        OUT_DIR / "baseline_metrics.csv",
        index=False,
    )

    print("\nMatched baseline comparison:")
    print(summary.round(3).to_string(index=False))

    print("\nValidation:")
    print(f"Matched target dates: {predictions['Target Date'].nunique()}")
    print(f"Predictions per model: {len(predictions) // len(predictors)}")
    print("All models use the same actual target values.")
    print("All predictions use historical observations only.")

    print(f"\nOutput directory:\n{OUT_DIR}")


if __name__ == "__main__":
    main()