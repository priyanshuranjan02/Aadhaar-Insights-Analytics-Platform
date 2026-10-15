"""Compare adaptive selection using validation windows of 7, 14 and 21 observations."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series
from src.forecasting_v2 import prepare_observations, OUTPUT_DIR
from src.forecasting_adaptive_selection import (
    MIN_HISTORY_OBSERVATIONS,
    linear_prediction,
    mean_prediction,
    rolling_validation_mae,
)

WINDOWS = (7, 14, 21)

OUT_DIR = (
    OUTPUT_DIR
    / "sensitivity"
    / "matched_origin"
    / "window_experiment"
)


def validation_mae(history, predictor, window):
    """Score a model on recent historical targets, using past data only."""
    errors = []

    # Each validation target must have at least MIN_HISTORY_OBSERVATIONS
    # observations preceding it.
    start = max(
        MIN_HISTORY_OBSERVATIONS,
        len(history) - window,
    )

    for target_index in range(start, len(history)):
        past = history[:target_index]
        actual = float(history[target_index])
        prediction = predictor(past)
        errors.append(abs(actual - prediction))

    return float(np.mean(errors)) if errors else np.inf


def calculate_metrics(group):
    actual = group["Actual"].to_numpy(dtype=float)
    predicted = group["Predicted"].to_numpy(dtype=float)
    errors = predicted - actual

    ss_total = np.sum((actual - actual.mean()) ** 2)
    ss_residual = np.sum(errors ** 2)

    return {
        "Target Dates": group["Target Date"].nunique(),
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "R2": 1 - ss_residual / ss_total if ss_total > 0 else np.nan,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    daily = build_daily_target_series(datasets["biometric"], "biometric")
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

    for target_index, target_date in enumerate(dates):
        history = values[:target_index]

        if len(history) < MIN_HISTORY_OBSERVATIONS:
            continue

        actual = float(values[target_index])

        # Calculate both candidate predictions using identical history.
        candidate_predictions = {
            name: predictor(history)
            for name, predictor in predictors.items()
        }

        for window in WINDOWS:
            scores = {
                name: validation_mae(history, predictor, window)
                for name, predictor in predictors.items()
            }

            valid_scores = {
                name: score
                for name, score in scores.items()
                if np.isfinite(score)
            }

            # Explicit fallback when no validation score is available.
            selected = (
                min(valid_scores, key=valid_scores.get)
                if valid_scores
                else "LinearRegression"
            )

            predictions = {
                **candidate_predictions,
                "AdaptiveSelection": candidate_predictions[selected],
            }

            for model_name, prediction in predictions.items():
                rows.append({
                    "Target Date": target_date,
                    "Window": window,
                    "Model": model_name,
                    "Actual": actual,
                    "Predicted": prediction,
                    "Absolute Error": abs(prediction - actual),
                    "Training Observations": len(history),
                    "Selected Model": selected,
                    "LR Validation MAE": scores["LinearRegression"],
                    "Mean Validation MAE": scores["MeanBaseline"],
                    "Forecast Type": "One-step-ahead",
                })

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No predictions generated.")

    # Confirm that all windows and models use the same target dates.
    date_counts = result.groupby(["Window", "Target Date"])["Model"].nunique()
    if not (date_counts == 3).all():
        raise RuntimeError("Not all windows have three models on every target date.")

    result.to_csv(OUT_DIR / "window_predictions.csv", index=False)

    metric_rows = []
    for (window, model), group in result.groupby(["Window", "Model"]):
        metric_rows.append({
            "Window": window,
            "Model": model,
            **calculate_metrics(group),
        })

    metrics = pd.DataFrame(metric_rows).sort_values(["Window", "RMSE"])
    metrics.to_csv(OUT_DIR / "window_metrics.csv", index=False)

    adaptive = result[result["Model"] == "AdaptiveSelection"]
    selection_counts = (
        adaptive.groupby(["Window", "Selected Model"])
        .size()
        .rename("Dates Selected")
        .reset_index()
    )
    selection_counts.to_csv(OUT_DIR / "selection_counts.csv", index=False)

    print("\n=== METRICS BY VALIDATION WINDOW ===")
    print(metrics.round(3).to_string(index=False))

    print("\n=== SELECTED MODEL COUNTS ===")
    print(selection_counts.to_string(index=False))

    print(f"\nMatched target dates: {result['Target Date'].nunique()}")
    print(f"Windows tested: {WINDOWS}")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
