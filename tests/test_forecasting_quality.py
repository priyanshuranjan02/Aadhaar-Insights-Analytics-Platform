import numpy as np
import pandas as pd
import pytest

from src.forecasting_outlier_audit import robust_scores

def test_robust_scores_preserve_direction():
    values = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18])

    scores = robust_scores(values)

    assert scores.iloc[0] < 0
    assert scores.iloc[-1] > 0
    assert scores.iloc[4] == pytest.approx(0.0)


def test_robust_scores_return_nan_when_mad_is_zero():
    values = pd.Series([5, 5, 5, 5, 100])

    scores = robust_scores(values)

    assert scores.isna().all()


def test_daily_aggregation_preserves_observed_dates():
    dates = pd.to_datetime(
        ["2025-12-22", "2025-12-22", "2025-12-24"]
    )

    data = pd.DataFrame({
        "date": dates,
        "age_0_5": [10, 20, 5],
        "age_5_17": [2, 3, 1],
        "age_18_greater": [1, 2, 1],
    })

    target_columns = [
        "age_0_5",
        "age_5_17",
        "age_18_greater",
    ]

    daily = (
        data.groupby("date")[target_columns]
        .sum()
        .sum(axis=1)
        .rename("total_activity")
        .reset_index()
    )

    assert len(daily) == 2
    assert daily["date"].is_unique
    assert daily["date"].is_monotonic_increasing
    assert daily.loc[0, "total_activity"] == 38
    assert daily.loc[1, "total_activity"] == 7
    assert pd.Timestamp("2025-12-23") not in set(daily["date"])


def test_low_coverage_scores_are_negative():
    values = pd.Series([100, 101, 99, 102, 100, 1])

    scores = robust_scores(values)

    assert scores.iloc[-1] < -3.5