"""Compare ordinary and robust regression on matched biometric target dates."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series
from src.forecasting_v2 import (
    prepare_observations,
    score_predictions,
    LAGS,
    ROLLING_WINDOW,
    OUTPUT_DIR,
)

OUT_DIR = (
    OUTPUT_DIR
    / "sensitivity"
    / "matched_origin"
    / "robust_comparison"
)

MIN_HISTORY_OBSERVATIONS = 24


def make_features(history):
    """Create lag and rolling-mean features using past observations only."""
    max_lag = max(max(LAGS), ROLLING_WINDOW)
    X, y = [], []

    for i in range(max_lag, len(history)):
        previous = history[:i]
        X.append(
            [previous[-lag] for lag in LAGS]
            + [float(np.mean(previous[-ROLLING_WINDOW:]))]
        )
        y.append(float(history[i]))

    latest = (
        [history[-lag] for lag in LAGS]
        + [float(np.mean(history[-ROLLING_WINDOW:]))]
    )

    return np.asarray(X), np.asarray(y), np.asarray([latest])


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

    rows = []

    for target_index, target_date in enumerate(dates):
        history = values[:target_index]

        if len(history) < MIN_HISTORY_OBSERVATIONS:
            continue

        actual = float(values[target_index])
        X, y, latest = make_features(history)

        models = {
            "LinearRegression": LinearRegression(),
            "HuberRegression": HuberRegressor(max_iter=2000),
        }

        for model_name, model in models.items():
            try:
                model.fit(X, y)
                prediction = float(model.predict(latest)[0])
            except (ValueError, FloatingPointError) as exc:
                print(f"Skipping {model_name} for {target_date.date()}: {exc}")
                continue

            rows.append({
                "Target Date": target_date,
                "Model": model_name,
                "Actual": actual,
                "Predicted": prediction,
                "Absolute Error": abs(prediction - actual),
                "Training Observations": len(history),
                "Forecast Type": "One-step-ahead",
            })

    predictions = pd.DataFrame(rows)

    if predictions.empty:
        raise RuntimeError("No predictions generated.")

    # Compare models only on target dates where both produced predictions.
    coverage = predictions.groupby("Target Date")["Model"].nunique()
    matched_dates = coverage[coverage == 2].index
    matched = predictions[predictions["Target Date"].isin(matched_dates)].copy()

    if matched.empty:
        raise RuntimeError("No matched target dates across both models.")

    matched.to_csv(OUT_DIR / "robust_predictions.csv", index=False)

    summary = []
    for model_name, group in matched.groupby("Model"):
        scores = score_predictions(
            group["Actual"].to_numpy(dtype=float),
            group["Predicted"].to_numpy(dtype=float),
        )
        summary.append({
            "Model": model_name,
            "Matched Target Dates": group["Target Date"].nunique(),
            **scores,
        })

    metrics = pd.DataFrame(summary).sort_values("RMSE")
    metrics.to_csv(OUT_DIR / "robust_metrics.csv", index=False)

    print("\n=== ROBUST REGRESSION COMPARISON ===")
    print(metrics.round(3).to_string(index=False))
    print(f"\nMatched target dates: {matched['Target Date'].nunique()}")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
