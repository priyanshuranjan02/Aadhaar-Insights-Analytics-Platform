"""Rolling-origin sensitivity analysis for biometric coverage anomalies."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series
from src.forecasting_v2 import (
    LAGS,
    ROLLING_WINDOW,
    RECENT_START,
    make_model_factories,
    prepare_observations,
    score_predictions,
    recursive_predict,
    feature_columns,
)

ANOMALOUS_DATES = pd.to_datetime([
    "2025-12-22",
    "2025-12-23",
])

HORIZON = 6
INITIAL_TRAIN_FEATURES = 24
N_FOLDS = 3


def build_features(daily):
    """Build lag features from the supplied reporting history."""
    daily = daily.sort_values("date").reset_index(drop=True)
    values = daily["total_activity"].to_numpy(dtype=float)
    rows = []

    for i in range(ROLLING_WINDOW, len(daily)):
        row = {
            "date": daily.loc[i, "date"],
            "target": values[i],
        }

        for lag in LAGS:
            row[f"lag_observation_{lag}"] = values[i - lag]

        row["rolling_mean_previous_4"] = float(
            np.mean(values[i - ROLLING_WINDOW:i])
        )
        rows.append(row)

    return pd.DataFrame(rows)


def fit_and_forecast(history, test, model_name, factory):
    """Fit on training features and recursively predict the test horizon."""
    train_features = build_features(history)
    cols = feature_columns()

    if len(train_features) < INITIAL_TRAIN_FEATURES:
        raise ValueError(
            f"Insufficient training features: {len(train_features)}"
        )

    model = factory()
    model.fit(train_features[cols], train_features["target"])

    initial_history = history["total_activity"].tail(ROLLING_WINDOW).tolist()

    predicted = recursive_predict(
        model,
        initial_history,
        len(test),
        cols,
    )

    actual = test["total_activity"].to_numpy(dtype=float)

    return {
        "Model": model_name,
        **score_predictions(actual, predicted),
        "Train observations": len(train_features),
        "Test observations": len(test),
    }


def main():
    datasets = load_all_datasets()
    daily = build_daily_target_series(datasets["biometric"], "biometric")
    daily = prepare_observations(daily)

    # Select 18 non-anomalous reporting dates for three consecutive folds.
    eligible = daily.loc[
        ~daily["date"].isin(ANOMALOUS_DATES)
    ].reset_index(drop=True)

    test_dates = eligible["date"].tail(HORIZON * N_FOLDS).tolist()

    if len(test_dates) != HORIZON * N_FOLDS:
        raise ValueError("Not enough non-anomalous observations.")

    results = []

    for fold_number in range(N_FOLDS):
        fold_dates = test_dates[
            fold_number * HORIZON:(fold_number + 1) * HORIZON
        ]
        test_start = min(fold_dates)
        test_end = max(fold_dates)

        # The test targets are taken from the original daily series.
        test = daily.loc[
            daily["date"].isin(fold_dates)
        ].sort_values("date")

        # Baseline training history retains the anomalous observations
        # only when they occur before the fold's test start.
        baseline_history = daily.loc[
            daily["date"] < test_start
        ].copy()

        filtered_history = baseline_history.loc[
            ~baseline_history["date"].isin(ANOMALOUS_DATES)
        ].copy()

        for scenario, history in [
            ("Baseline", baseline_history),
            ("Exclude Dec 22-23", filtered_history),
        ]:
            for model_name, factory in make_model_factories().items():
                row = fit_and_forecast(
                    history,
                    test,
                    model_name,
                    factory,
                )
                row.update({
                    "Scenario": scenario,
                    "Fold": fold_number + 1,
                    "Test Start": test_start,
                    "Test End": test_end,
                })
                results.append(row)

    results_df = pd.DataFrame(results)

    summary = (
        results_df.groupby(["Scenario", "Model"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            R2=("R2", "mean"),
            Folds=("Fold", "nunique"),
        )
        .sort_values(["Scenario", "RMSE"])
    )

    output_dir = Path(
        "data/processed/forecasting_v2_sensitivity/rolling_origin"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    print("\nFOLD WINDOWS")
    print(
        results_df[
            ["Scenario", "Fold", "Test Start", "Test End"]
        ].drop_duplicates().to_string(index=False)
    )

    print("\nFOLD METRICS")
    print(results_df.to_string(index=False))

    print("\nSUMMARY METRICS")
    print(summary.to_string(index=False))
    print("\nSaved to:", output_dir.resolve())

if __name__ == "__main__":
    main()