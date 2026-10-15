# Forecasting Evaluation — Aadhaar Insights Analytics Platform

## 1. Purpose

This document records the biometric activity forecasting experiments,
their evaluation results, data-quality observations, and current model
selection decision.

The experiments are exploratory. Their results should not be interpreted
as evidence that future biometric activity can be predicted reliably.

## 2. Forecasting Target

The target is `total_activity` derived from the biometric dataset.

The experiments use chronological, one-step-ahead predictions: for each
target date, the model uses observations available before that date.

The evaluated period contains 60 target dates for the principal model
comparisons. A separate sensitivity experiment uses 58 matched target
dates.

## 3. Principal Model Comparison

The following models were evaluated on the same 60 target dates.

| Model | Target Dates | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 60 | 104009.688 | 137552.624 | -0.039 |
| Adaptive Selection (14) | 60 | 104111.037 | 134619.121 | 0.005 |
| Mean Baseline | 60 | 108699.748 | 136422.585 | -0.022 |
| Persistence | 60 | 112072.983 | 157804.855 | -0.367 |
| Huber Regression | 60 | 111101.403 | 147404.955 | -0.193 |

Lower MAE and RMSE indicate smaller prediction errors. Higher R² is
generally preferable.

### Interpretation

- Linear Regression achieved the lowest MAE among these models.
- Adaptive Selection (14) achieved the lowest RMSE, but its MAE was
  slightly higher than that of fixed Linear Regression.
- Mean Baseline provides a useful reference for judging model performance.
- Huber Regression and Persistence performed worse than Linear Regression
  on both MAE and RMSE.
- R² values are close to zero or negative, indicating limited predictive
  performance in this evaluation.

## 4. Adaptive Validation-Window Experiment

The adaptive selector was tested with validation windows of 7, 14, and
21 observations, using the same 60 target dates.

| Window | Model | MAE | RMSE | R² |
|---:|---|---:|---:|---:|
| 7 | Adaptive Selection | 105295.495 | 132232.418 | 0.040 |
| 14 | Adaptive Selection | 104111.037 | 134619.121 | 0.005 |
| 21 | Adaptive Selection | 106112.096 | 138968.291 | -0.060 |

The 7-observation window achieved the lowest RMSE among the adaptive
configurations, while the 14-observation window achieved the lowest MAE.
The 21-observation window performed worst among these adaptive
configurations by both MAE and RMSE.

No window was uniformly best across all metrics.

## 5. Sensitivity Analysis: Flagged Dates

December 22 and December 23, 2025, were flagged for exceptionally low
recorded biometric activity.

A separate matched-origin experiment compared original training history
with training history excluding these dates. Both scenarios were evaluated
on the same 58 target dates.

| Scenario | Model | Dates | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Original history | Linear Regression | 58 | 97847.402 | 129639.964 | -0.084 |
| Exclusion history | Linear Regression | 58 | 95647.312 | 124140.800 | 0.006 |
| Original history | Persistence | 58 | 112755.603 | 158667.637 | -0.623 |
| Exclusion history | Persistence | 58 | 109582.259 | 150185.348 | -0.454 |

Excluding the flagged dates from training modestly improved the reported
metrics in this experiment. This is a sensitivity result only. It does
not establish that the records are invalid or explain why activity was
low.

The flagged dates were not removed from the original source data, and
their actual values were not replaced.

The 58-date sensitivity results should not be ranked directly against
the principal 60-date results because the evaluation setup differs.

## 6. Data-Quality Observations

The biometric series includes abrupt drops and rebounds. Examples include
very low recorded activity on December 22–23 and substantially higher
activity on nearby dates.

These patterns may reflect reporting-volume, collection, extraction, or
genuine activity changes. The available observations alone do not establish
the cause.

Forecast errors occur in both directions: some sharp drops are
overpredicted, while some rebounds are underpredicted.

## 7. Current Model Decision

Retain Linear Regression as the primary fixed forecasting candidate and
Mean Baseline as a benchmark.

Adaptive Selection remains an experimental alternative because its
relative performance depends on the metric and validation window.
Huber Regression is not selected because it performed worse than Linear
Regression in the reported comparison.

This is a provisional decision based on the experiments conducted to
date, not a claim that Linear Regression is reliably predictive in every
period.

## 8. Limitations

- The evaluation contains only 60 target dates in the principal comparison.
- The target series is volatile and includes extreme activity changes.
- Near-zero or negative R² values indicate weak explanatory performance.
- Results may not generalize to future reporting periods.
- Sensitivity experiments do not establish the cause or validity of flagged
  observations.
- One-step-ahead results do not establish the accuracy of a multi-step
  future forecast.

## 9. Reproducibility

The following scripts and outputs record the experiments:

- `src/forecasting_baseline_comparison.py`
- `src/forecasting_adaptive_selection.py`
- `src/forecasting_window_experiment.py`
- `src/forecasting_robust_comparison.py`
- `src/forecasting_matched_origin.py`
- `src/forecasting_consolidated_report.py`
- `data/processed/forecasting_v2/consolidated_report/forecasting_experiment_comparison.csv`

Generated outputs are stored under
`data/processed/forecasting_v2/sensitivity/matched_origin/`.

Re-run the relevant experiment scripts to regenerate their outputs.
