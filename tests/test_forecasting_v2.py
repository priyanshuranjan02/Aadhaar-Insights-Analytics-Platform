import numpy as np
import pandas as pd
import pytest

from src.forecasting_v2 import (
    feature_columns,
    make_features,
    prepare_observations,
    recursive_predict,
    PersistenceRegressor,
    ROLLING_WINDOW,
)


def make_daily_series(n=80):
    """Create a deterministic series with observed reporting dates."""
    dates = pd.date_range("2025-09-01", periods=n, freq="D")
    values = np.arange(100, 100 + n, dtype=float)

    return pd.DataFrame({
        "date": dates,
        "total_activity": values,
    })


def test_prepare_observations_sorts_dates():
    daily = make_daily_series(80).sample(
        frac=1,
        random_state=42,
    )

    prepared = prepare_observations(daily)

    assert prepared["date"].is_monotonic_increasing
    assert not prepared["date"].duplicated().any()


def test_make_features_uses_only_previous_observations():
    daily = make_daily_series(80)
    features = make_features(daily)

    first = features.iloc[0]

    assert first["target"] == daily.iloc[ROLLING_WINDOW]["total_activity"]
    assert first["lag_observation_1"] == daily.iloc[ROLLING_WINDOW - 1]["total_activity"]
    assert first["lag_observation_2"] == daily.iloc[ROLLING_WINDOW - 2]["total_activity"]
    assert first["rolling_mean_previous_4"] == pytest.approx(
        daily.iloc[:ROLLING_WINDOW]["total_activity"].mean()
    )


def test_recursive_forecast_updates_history():
    daily = make_daily_series(80)
    features = make_features(daily)

    model = PersistenceRegressor()
    model.fit(features[feature_columns()], features["target"])

    initial_history = daily["total_activity"].tail(ROLLING_WINDOW).tolist()

    predictions = recursive_predict(
        model=model,
        initial_history=initial_history,
        steps=5,
        feature_cols=feature_columns(),
    )

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()

    # Persistence should keep predicting the latest value.
    assert np.allclose(predictions, initial_history[-1])


def test_prepare_observations_rejects_duplicate_dates():
    daily = make_daily_series(80)
    duplicate = pd.concat([daily, daily.iloc[[10]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate dates"):
        prepare_observations(duplicate)


def test_prepare_observations_rejects_negative_activity():
    daily = make_daily_series(80)
    daily.loc[10, "total_activity"] = -1

    with pytest.raises(ValueError, match="Negative activity"):
        prepare_observations(daily)