"""Observation-based forecasting pipeline for Aadhaar Insights.

Forecasts the next 12 reporting periods, not the next 12 calendar days.
No missing dates are interpolated or replaced with zero.

Model selection uses rolling-origin backtesting on historical windows that
precede a separate final 12-observation holdout. The winning model is then
refitted on all eligible observations for the experimental future projection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "forecasting_v2"

RECENT_START = pd.Timestamp("2025-09-01")
HORIZON = 12
LAGS = (1, 2, 3, 4)
ROLLING_WINDOW = 4
BACKTEST_FOLDS = 3
MIN_TRAIN_FEATURES = 24


class PersistenceRegressor:
    """Predict the next reporting value using the latest observation."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag_observation_1"].to_numpy(dtype=float)


def make_model_factories():
    """Return fresh model instances for each fold to prevent state leakage."""
    return {
        "Persistence": PersistenceRegressor,
        "MeanBaseline": lambda: DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression,
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


def prepare_observations(series):
    """Return actual reporting observations without interpolation."""
    data = series[["date", "total_activity"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["total_activity"] = pd.to_numeric(
        data["total_activity"], errors="coerce"
    )

    data = data.dropna(subset=["date", "total_activity"])
    data = data[data["date"] >= RECENT_START]
    data = data.sort_values("date")

    if data["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found in the daily target series. "
            "Resolve aggregation before forecasting."
        )

    if len(data) < MIN_TRAIN_FEATURES + ROLLING_WINDOW:
        raise ValueError(
            f"Only {len(data)} reporting observations are available since "
            f"{RECENT_START.date()}; more observations are required."
        )

    if (data["total_activity"] < 0).any():
        raise ValueError("Negative activity values were found.")

    return data.reset_index(drop=True)


def make_features(series):
    """Create features from previous reporting observations only."""
    data = prepare_observations(series)
    values = data["total_activity"].to_numpy(dtype=float)
    rows = []

    for i in range(ROLLING_WINDOW, len(data)):
        row = {
            "date": data.loc[i, "date"],
            "target": values[i],
        }

        for lag in LAGS:
            row[f"lag_observation_{lag}"] = values[i - lag]

        row["rolling_mean_previous_4"] = float(
            np.mean(values[i - ROLLING_WINDOW:i])
        )
        rows.append(row)

    features = pd.DataFrame(rows).reset_index(drop=True)

    minimum_required = (
        MIN_TRAIN_FEATURES + HORIZON * (BACKTEST_FOLDS + 1)
    )
    if len(features) < minimum_required:
        raise ValueError(
            f"Only {len(features)} feature rows are available. "
            f"At least {minimum_required} are required for "
            f"{BACKTEST_FOLDS} rolling-origin folds, a separate "
            f"{HORIZON}-observation final holdout, and "
            f"{MIN_TRAIN_FEATURES} initial training rows."
        )

    return features


def feature_columns():
    return [
        *(f"lag_observation_{lag}" for lag in LAGS),
        "rolling_mean_previous_4",
    ]


def recursive_predict(model, initial_history, steps, feature_cols):
    """Forecast sequentially without feeding actual test targets to history."""
    history = [float(value) for value in initial_history]
    predictions = []

    for _ in range(steps):
        if len(history) < ROLLING_WINDOW:
            raise ValueError("Not enough history to create forecast features.")

        row = {
            f"lag_observation_{lag}": history[-lag]
            for lag in LAGS
        }
        row["rolling_mean_previous_4"] = float(
            np.mean(history[-ROLLING_WINDOW:])
        )

        X_next = pd.DataFrame([row], columns=feature_cols)
        prediction = max(0.0, float(model.predict(X_next)[0]))
        predictions.append(prediction)
        history.append(prediction)

    return np.asarray(predictions, dtype=float)


def score_predictions(actual, predicted):
    """Calculate common regression metrics for one evaluation window."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = mean_squared_error(actual, predicted)

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(actual, predicted)),
    }


def rolling_origin_backtest(features):
    """Evaluate all models over three windows before the final holdout."""
    feature_cols = feature_columns()
    n_rows = len(features)

    # Keep the final HORIZON rows untouched by model selection. The rolling
    # backtest windows all end before that final holdout starts.
    final_holdout_start = n_rows - HORIZON
    fold_rows = []
    fold_predictions = []

    factories = make_model_factories()

    # Three consecutive 12-observation windows immediately before the final
    # holdout. Each fold trains only on feature rows dated before its test.
    for fold_number in range(1, BACKTEST_FOLDS + 1):
        test_end = final_holdout_start - (
            BACKTEST_FOLDS - fold_number
        ) * HORIZON
        test_start = test_end - HORIZON

        train = features.iloc[:test_start].copy()
        test = features.iloc[test_start:test_end].copy()

        if len(train) < MIN_TRAIN_FEATURES or len(test) != HORIZON:
            raise ValueError(
                f"Invalid fold {fold_number}: train={len(train)}, "
                f"test={len(test)}. Check the available reporting history."
            )

        first_test = test.iloc[0]
        initial_history = [
            float(first_test[f"lag_observation_{lag}"])
            for lag in reversed(LAGS)
        ]

        for model_name, factory in factories.items():
            model = factory()
            model.fit(train[feature_cols], train["target"])

            predicted = recursive_predict(
                model=model,
                initial_history=initial_history,
                steps=len(test),
                feature_cols=feature_cols,
            )
            actual = test["target"].to_numpy(dtype=float)
            scores = score_predictions(actual, predicted)

            fold_rows.append({
                "Fold": fold_number,
                "Train End Date": train["date"].iloc[-1],
                "Test Start Date": test["date"].iloc[0],
                "Test End Date": test["date"].iloc[-1],
                "Model": model_name,
                **scores,
                "Train observations": len(train),
                "Test observations": len(test),
            })

            for date, actual_value, predicted_value in zip(
                test["date"], actual, predicted
            ):
                fold_predictions.append({
                    "Fold": fold_number,
                    "Date": date,
                    "Model": model_name,
                    "Actual": actual_value,
                    "Predicted": predicted_value,
                    "Absolute Error": abs(actual_value - predicted_value),
                })

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby("Model", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            R2=("R2", "mean"),
            **{
                "MAE Std": ("MAE", "std"),
                "RMSE Std": ("RMSE", "std"),
                "Backtest Folds": ("Fold", "nunique"),
            },
        )
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )
    summary["Evaluation"] = (
        f"Rolling-origin recursive backtest ({BACKTEST_FOLDS} folds)"
    )

    return summary, fold_metrics, pd.DataFrame(fold_predictions)



def evaluate_final_holdout(features, selected_model_name):
    """Compare the selected model and simple baselines on one final holdout."""
    feature_cols = feature_columns()
    train = features.iloc[:-HORIZON].copy()
    test = features.iloc[-HORIZON:].copy()

    first_test = test.iloc[0]
    initial_history = [
        float(first_test[f"lag_observation_{lag}"])
        for lag in reversed(LAGS)
    ]
    actual = test["target"].to_numpy(dtype=float)

    factories = make_model_factories()
    model_names = [
        "MeanBaseline",
        "Persistence",
        selected_model_name,
    ]
    # Preserve order while avoiding duplicate names.
    model_names = list(dict.fromkeys(model_names))

    metrics_rows = []
    predictions_by_model = {}

    for model_name in model_names:
        model = factories[model_name]()
        model.fit(train[feature_cols], train["target"])

        predicted = recursive_predict(
            model=model,
            initial_history=initial_history,
            steps=len(test),
            feature_cols=feature_cols,
        )

        scores = score_predictions(actual, predicted)
        metrics_rows.append({
            "Model": model_name,
            **scores,
            "Train observations": len(train),
            "Test observations": len(test),
            "Evaluation": "Final recursive 12-observation holdout",
            "Selected by rolling backtest": (
                model_name == selected_model_name
            ),
        })
        predictions_by_model[model_name] = predicted

    holdout = pd.DataFrame({
        "Date": test["date"].to_numpy(),
        "Actual": actual,
    })

    for model_name, predicted in predictions_by_model.items():
        holdout[f"{model_name} Predicted"] = predicted

    # Keep the existing dashboard's Actual/Predicted columns compatible.
    holdout["Predicted"] = predictions_by_model[selected_model_name]
    holdout["Model"] = selected_model_name
    holdout["Absolute Error"] = (
        holdout["Actual"] - holdout["Predicted"]
    ).abs()

    final_metrics = pd.DataFrame(metrics_rows).sort_values(
        "RMSE"
    ).reset_index(drop=True)

    # Refit the selected model on all eligible observations for forecasting.
    fitted_for_forecast = factories[selected_model_name]()
    fitted_for_forecast.fit(features[feature_cols], features["target"])

    return final_metrics, holdout, fitted_for_forecast, feature_cols


def forecast_future(series, model, feature_cols):
    """Recursively forecast the next 12 reporting periods."""
    data = prepare_observations(series)
    history = data["total_activity"].astype(float).to_list()
    last_observed_date = data["date"].max()

    predictions = recursive_predict(
        model=model,
        initial_history=history,
        steps=HORIZON,
        feature_cols=feature_cols,
    )

    return pd.DataFrame({
        "Forecast Step": range(1, HORIZON + 1),
        "Forecast": predictions,
        "Last Observed Date": last_observed_date,
        "Horizon Definition": "Next reporting period",
    })


def main():
    datasets = load_all_datasets()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, raw in datasets.items():
        print(f"\nProcessing {name}...")

        daily = build_daily_target_series(raw, name)
        features = make_features(daily)

        backtest_summary, backtest_folds, backtest_predictions = (
            rolling_origin_backtest(features)
        )
        selected_model_name = str(backtest_summary.iloc[0]["Model"])

        final_metrics, holdout, best_model, feature_cols = (
            evaluate_final_holdout(features, selected_model_name)
        )
        forecasts = forecast_future(daily, best_model, feature_cols)

        dataset_dir = OUTPUT_DIR / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the filenames consumed by the current dashboard.
        # model_metrics.csv now contains the rolling-origin model ranking.
        backtest_summary.to_csv(
            dataset_dir / "model_metrics.csv", index=False
        )
        holdout.to_csv(
            dataset_dir / "holdout_predictions.csv", index=False
        )
        forecasts.to_csv(
            dataset_dir / "future_forecast.csv", index=False
        )

        # Additional audit files for fold-level review and the final holdout.
        backtest_folds.to_csv(
            dataset_dir / "rolling_origin_folds.csv", index=False
        )
        backtest_predictions.to_csv(
            dataset_dir / "rolling_origin_predictions.csv", index=False
        )
        final_metrics.to_csv(
            dataset_dir / "final_holdout_metrics.csv", index=False
        )

        print("Last observed date:", daily["date"].max().date())
        print("Selected model by mean rolling-origin RMSE:", selected_model_name)
        print("\nRolling-origin model comparison:")
        print(backtest_summary.to_string(index=False))
        print("\nFinal holdout metrics (not used for model selection):")
        print(final_metrics.to_string(index=False))
        print("Saved results to:", dataset_dir)


if __name__ == "__main__":
    main()