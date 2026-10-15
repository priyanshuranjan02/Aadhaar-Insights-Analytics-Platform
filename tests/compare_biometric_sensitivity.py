"""Compare biometric models on identical non-anomalous holdout dates."""

import pandas as pd

from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series
from src.forecasting_v2 import (
    HORIZON,
    make_features,
    make_model_factories,
    prepare_observations,
    recursive_predict,
    score_predictions,
    feature_columns,
)

ANOMALOUS_DATES = pd.to_datetime([
    "2025-12-22",
    "2025-12-23",
])


def evaluate_history(daily, holdout_dates):
    """Evaluate each model on the specified dates."""
    features = make_features(daily)
    feature_cols = feature_columns()

    test = features[features["date"].isin(holdout_dates)].copy()
    train = features[~features["date"].isin(holdout_dates)].copy()

    if len(test) != len(holdout_dates):
        raise ValueError(
            f"Expected {len(holdout_dates)} holdout rows, "
            f"found {len(test)}."
        )

    # Require the training rows to precede the first test date.
    test_start = test["date"].min()
    train = train[train["date"] < test_start]

    if train.empty:
        raise ValueError("No training rows precede the holdout.")

    first_test = test.iloc[0]
    initial_history = [
        float(first_test[f"lag_observation_{lag}"])
        for lag in (4, 3, 2, 1)
    ]

    actual = test["target"].to_numpy(dtype=float)
    factories = make_model_factories()
    results = []

    for model_name, factory in factories.items():
        model = factory()
        model.fit(train[feature_cols], train["target"])

        predicted = recursive_predict(
            model,
            initial_history,
            len(test),
            feature_cols,
        )

        results.append({
            "Model": model_name,
            **score_predictions(actual, predicted),
            "Train rows": len(train),
            "Test rows": len(test),
        })

    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)


def main():
    datasets = load_all_datasets()
    daily = build_daily_target_series(datasets["biometric"], "biometric")
    daily = prepare_observations(daily)

    original_features = make_features(daily)

    # Select the final 12 feature dates, then exclude the two anomalies
    # from the evaluation window in BOTH scenarios.
    original_holdout_dates = original_features["date"].tail(HORIZON)
    holdout_dates = [
        date for date in original_holdout_dates
        if date not in set(ANOMALOUS_DATES)
    ]

    baseline = evaluate_history(daily, holdout_dates)

    filtered = daily.loc[
        ~daily["date"].isin(ANOMALOUS_DATES)
    ].copy()

    sensitivity = evaluate_history(filtered, holdout_dates)

    print("\nCOMMON EVALUATION DATES")
    print([pd.Timestamp(d).strftime("%Y-%m-%d") for d in holdout_dates])

    print("\nBASELINE: ORIGINAL HISTORY")
    print(baseline.to_string(index=False))

    print("\nSENSITIVITY: EXCLUDING DEC 22-23 FROM HISTORY")
    print(sensitivity.to_string(index=False))

    print(
        "\nInterpretation: both scenarios use the same non-anomalous "
        "evaluation dates. The filtered scenario also rebuilds features "
        "without the two suspected coverage anomalies."
    )


if __name__ == "__main__":
    main()