"""
Aadhaar Insights — Clean Chronological Baseline Evaluation

Evaluates observed-date activity totals using a chronological holdout.
Missing calendar dates are not imputed. Forecast horizons are measured
in observed reporting periods, not assumed calendar days.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "forecasting_clean"
OUTPUT_DIR = INPUT_DIR / "evaluation"

DATASETS = ["enrolment", "demographic", "biometric"]
TARGETS = {
    "enrolment": [
        "age_0_5", "age_5_17", "age_18_greater", "total_activity"
    ],
    "demographic": [
        "demo_age_5_17", "demo_age_17_", "total_activity"
    ],
    "biometric": [
        "bio_age_5_17", "bio_age_17_", "total_activity"
    ],
}

TEST_FRACTION = 0.20
MIN_OBSERVATIONS = 20


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) >= 2 else np.nan
        ),
    }


def evaluate_target(df, target, dataset_name):
    df = df.sort_values("date").reset_index(drop=True).copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")

    if df[target].isna().any():
        raise ValueError(
            f"{dataset_name}/{target}: missing or non-numeric target"
        )

    if not np.isfinite(df[target].to_numpy(dtype=float)).all():
        raise ValueError(
            f"{dataset_name}/{target}: non-finite target values"
        )

    n = len(df)
    test_size = max(1, int(np.ceil(n * TEST_FRACTION)))
    split = n - test_size

    if split < MIN_OBSERVATIONS or test_size < 2:
        raise ValueError(
            f"Insufficient observations for {dataset_name}/{target}: "
            f"n={n}, train={split}, test={test_size}"
        )

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)

    # Use actual elapsed calendar time as the trend input.
    origin = df["date"].min()
    x_train = (
        (train["date"] - origin).dt.total_seconds().to_numpy()
        / 86400.0
    ).reshape(-1, 1)
    x_test = (
        (test["date"] - origin).dt.total_seconds().to_numpy()
        / 86400.0
    ).reshape(-1, 1)

    predictions = {
        "TrainingMean": np.full(len(test), y_train.mean()),
        "LastObservedValue": np.full(len(test), y_train[-1]),
    }

    trend_model = LinearRegression()
    trend_model.fit(x_train, y_train)
    predictions["LinearCalendarTrend"] = trend_model.predict(x_test)

    metric_rows = []
    prediction_frames = []

    for model_name, predicted in predictions.items():
        metrics = calculate_metrics(y_test, predicted)

        metric_rows.append({
            "dataset": dataset_name,
            "target": target,
            "model": model_name,
            "train_observations": len(train),
            "test_observations": len(test),
            "train_end_date": train["date"].max().date().isoformat(),
            "test_start_date": test["date"].min().date().isoformat(),
            "test_end_date": test["date"].max().date().isoformat(),
            **metrics,
        })

        prediction_frames.append(pd.DataFrame({
            "dataset": dataset_name,
            "target": target,
            "model": model_name,
            "date": test["date"].to_numpy(),
            "actual": y_test,
            "predicted": predicted,
            "absolute_error": np.abs(y_test - predicted),
        }))

    return metric_rows, prediction_frames


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_predictions = []

    for dataset_name in DATASETS:
        path = INPUT_DIR / dataset_name / "daily_activity.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing cleaned data: {path}. "
                "Run src.forecasting_data_clean first."
            )

        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if df["date"].duplicated().any():
            raise ValueError(
                f"{dataset_name}: duplicate dates found"
            )

        print(f"\n{dataset_name.upper()} — {len(df)} observations")

        for target in TARGETS[dataset_name]:
            metrics, predictions = evaluate_target(
                df, target, dataset_name
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)

            print(f"  Evaluated target: {target}")

    metrics_df = pd.DataFrame(all_metrics).sort_values(
        ["dataset", "target", "RMSE"]
    )
    predictions_df = pd.concat(
        all_predictions, ignore_index=True
    )

    metrics_path = OUTPUT_DIR / "baseline_metrics.csv"
    predictions_path = OUTPUT_DIR / "baseline_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nBASELINE METRICS")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print("\nEvaluation completed. Existing forecasts were not modified.")


if __name__ == "__main__":
    main()