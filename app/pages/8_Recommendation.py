import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, APP_DIR)
sys.path.insert(0, ROOT_DIR)

from src.data_loader import load_analysis, load_forecasting_results


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Recommendations",
    page_icon="💡",
    layout="wide",
)

st.title("💡 AI Recommendations & Insights")
st.markdown(
    """
    Data-driven recommendations based on dataset analysis, trend summaries,
    anomaly detection, and the available model evaluation results.
    """
)

DATASETS = ["enrolment", "demographic", "biometric"]

st.sidebar.header("Configuration")
dataset = st.sidebar.selectbox("Select Dataset", DATASETS)


# ---------------------------------------------------------
# SAFE DATA LOADING
# ---------------------------------------------------------

def as_dataframe(value):
    """Convert a supported value into a DataFrame safely."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except (TypeError, ValueError):
            return pd.DataFrame()
    return pd.DataFrame()


def safe_count(value):
    """Count records only when the supplied object supports counting."""
    try:
        return len(value) if value is not None else 0
    except TypeError:
        return 0


def numeric_column(frame, column):
    """Return a numeric Series or an empty Series."""
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


try:
    analysis = load_analysis(dataset) or {}
    forecast_results = load_forecasting_results(dataset) or {}
except Exception as exc:
    st.error(f"Could not load analysis results for {dataset}: {exc}")
    st.stop()

if not isinstance(analysis, dict):
    st.error("The analysis loader returned an unexpected data structure.")
    st.stop()

if not isinstance(forecast_results, dict):
    st.error("The forecasting loader returned an unexpected data structure.")
    st.stop()

metrics_df = as_dataframe(forecast_results.get("metrics"))
anomalies = analysis.get("anomalies", [])
yearly_trend = as_dataframe(analysis.get("yearly_trend"))
corr_matrix = as_dataframe(analysis.get("correlation"))

anomaly_count = safe_count(anomalies)
trend_points = len(yearly_trend)

# Count unique correlated pairs, excluding the diagonal.
strong_corr = 0
if not corr_matrix.empty:
    correlation_values = corr_matrix.apply(pd.to_numeric, errors="coerce")
    valid_columns = [
        col for col in correlation_values.columns
        if col in correlation_values.index
    ]
    correlation_values = correlation_values.loc[
        valid_columns, valid_columns
    ]
    matrix = correlation_values.to_numpy(dtype=float)

    if matrix.size:
        upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
        strong_corr = int(
            np.sum(np.isfinite(upper_triangle) & (np.abs(upper_triangle) > 0.7))
        )


# ---------------------------------------------------------
# MODEL METRICS AND LIMITATIONS
# ---------------------------------------------------------

required_metrics = {"Model", "R2", "MAE", "RMSE"}
missing_metrics = required_metrics - set(metrics_df.columns)

best_model = None
r2 = np.nan
mae = np.nan
rmse = np.nan

if not metrics_df.empty and not missing_metrics:
    metrics_df = metrics_df.copy()

    for column in ["R2", "MAE", "RMSE"]:
        metrics_df[column] = pd.to_numeric(
            metrics_df[column], errors="coerce"
        )

    usable_metrics = metrics_df.dropna(subset=["R2", "MAE", "RMSE"])

    if not usable_metrics.empty:
        # This selects the highest-R² row from the available metrics.
        # It does not prove that the model is suitable for deployment.
        best_model = usable_metrics.sort_values(
            "R2", ascending=False
        ).iloc[0]

        r2 = float(best_model["R2"])
        mae = float(best_model["MAE"])
        rmse = float(best_model["RMSE"])


# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", dataset.title())

with col2:
    st.metric("Anomalies Reported", f"{anomaly_count:,}")

with col3:
    st.metric("Trend Data Points", f"{trend_points:,}")

st.caption(
    "Anomaly counts and trend points describe the loaded analysis outputs. "
    "They do not independently establish overall data quality."
)

st.divider()


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

st.subheader("🤖 Model Evaluation")

if best_model is None:
    st.warning(
        "Usable model metrics are unavailable. Check that the forecasting "
        "metrics file contains Model, R2, MAE, and RMSE columns with valid values."
    )
    st.info(
        "Model-based recommendations are withheld until valid evaluation "
        "results are available."
    )
else:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Highest-R² Model", str(best_model["Model"]))

    with c2:
        st.metric("R²", f"{r2:.4f}")

    with c3:
        st.metric("MAE", f"{mae:,.2f}")

    with c4:
        st.metric("RMSE", f"{rmse:,.2f}")

    if r2 < 0:
        st.error(
            "The selected model has a negative R² score on the supplied "
            "evaluation results. It performs worse than predicting the "
            "evaluation target's mean under the standard R² definition."
        )
    elif r2 < 0.3:
        st.warning(
            "The selected model's R² is low. Treat its predictions as "
            "experimental and avoid relying on them for operational decisions."
        )
    else:
        st.info(
            "R² alone is not sufficient to establish forecasting reliability. "
            "Review the evaluation design, baseline comparisons, and error sizes "
            "before using predictions for planning."
        )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )

st.divider()


# ---------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------

st.subheader("📋 Executive Summary")

summary_lines = [
    f"**Dataset:** {dataset.title()}",
    f"**Trend data points:** {trend_points:,}",
    f"**Reported anomalies:** {anomaly_count:,}",
    f"**Strong correlation pairs (|r| > 0.7):** {strong_corr:,}",
]

if best_model is not None:
    summary_lines.extend(
        [
            f"**Highest-R² model in loaded metrics:** {best_model['Model']}",
            f"**R²:** {r2:.4f}",
            f"**MAE:** {mae:,.2f}",
            f"**RMSE:** {rmse:,.2f}",
        ]
    )
else:
    summary_lines.append("**Model evaluation:** unavailable or incomplete")

st.info("\n\n".join(summary_lines))

st.divider()


# ---------------------------------------------------------
# KEY FINDINGS
# ---------------------------------------------------------

st.subheader("🔍 Key Findings")

left, right = st.columns(2)

with left:
    st.markdown("### 📈 Trend Analysis")
    if trend_points == 0:
        st.info("No trend observations are available in the loaded analysis.")
    else:
        st.write(
            f"The loaded trend output contains {trend_points:,} time periods. "
            "Inspect the underlying values before concluding that growth is positive."
        )

    st.markdown("### 🔗 Correlation Analysis")
    if strong_corr:
        st.write(
            f"{strong_corr:,} unique feature pairs have an absolute correlation "
            "greater than 0.7. Correlation does not establish causation."
        )
    else:
        st.write(
            "No strong feature pairs were identified in the available "
            "correlation matrix, or the matrix is unavailable."
        )

with right:
    st.markdown("### ⚠ Anomaly Detection")
    if anomaly_count:
        st.write(
            f"The analysis reports {anomaly_count:,} anomalous records. "
            "Review them in context before deciding whether they represent "
            "errors, unusual activity, or reporting issues."
        )
    else:
        st.write(
            "No anomalies were reported by the loaded analysis. This does not "
            "guarantee that the dataset is free from data-quality problems."
        )

    st.markdown("### 🤖 Forecasting")
    if best_model is None:
        st.write("Forecasting conclusions are unavailable without valid metrics.")
    elif r2 < 0.3:
        st.write(
            "The available R² indicates limited predictive evidence. Treat "
            "forecasts as experimental and compare them with simple baselines "
            "before considering any operational use."
        )
    else:
        st.write(
            "Review MAE, RMSE, baseline performance, and out-of-sample validation "
            "before deciding whether the model is suitable for a specific use."
        )

st.divider()


# ---------------------------------------------------------
# DATA-DRIVEN RECOMMENDATIONS
# ---------------------------------------------------------

st.subheader("📈 Recommended Actions")

recommendations = []

if anomaly_count > 0:
    recommendations.append(
        {
            "Priority": "High",
            "Recommendation": "Review reported anomalous records",
            "Reason": f"The analysis reports {anomaly_count:,} anomalies.",
            "Next step": "Inspect dates, regions, counts, and data provenance before changing data.",
        }
    )
else:
    recommendations.append(
        {
            "Priority": "Medium",
            "Recommendation": "Maintain data-quality checks",
            "Reason": "The loaded analysis reports no anomalies.",
            "Next step": "Continue validation; do not treat zero detected anomalies as proof of perfect quality.",
        }
    )

if trend_points == 0:
    recommendations.append(
        {
            "Priority": "Medium",
            "Recommendation": "Verify trend data availability",
            "Reason": "No trend points were loaded.",
            "Next step": "Check source dates, aggregation, and analysis outputs.",
        }
    )

if best_model is None:
    recommendations.append(
        {
            "Priority": "High",
            "Recommendation": "Validate model evaluation outputs",
            "Reason": "Usable model metrics are unavailable.",
            "Next step": "Regenerate metrics and verify their columns and numeric values.",
        }
    )
elif r2 < 0.3:
    recommendations.append(
        {
            "Priority": "High",
            "Recommendation": "Avoid relying on forecasts for operational decisions",
            "Reason": f"The highest-R² available model has R² = {r2:.4f}.",
            "Next step": "Compare against baselines and improve out-of-sample validation.",
        }
    )
else:
    recommendations.append(
        {
            "Priority": "Medium",
            "Recommendation": "Validate forecast performance for the intended use",
            "Reason": "A positive R² does not alone establish deployment readiness.",
            "Next step": "Check baseline comparisons, temporal validation, and error magnitude.",
        }
    )

if strong_corr > 0:
    recommendations.append(
        {
            "Priority": "Low",
            "Recommendation": "Investigate strong feature relationships",
            "Reason": f"{strong_corr:,} pairs exceed the |r| > 0.7 threshold.",
            "Next step": "Check domain meaning, redundant variables, and possible leakage.",
        }
    )

recommendations.append(
    {
        "Priority": "Medium",
        "Recommendation": "Document data coverage and reporting gaps",
        "Reason": "The interpretation of trends depends on the dates and coverage available.",
        "Next step": "Record missing periods and investigate unusual reporting volumes without silently altering raw data.",
    }
)

recommendation_df = pd.DataFrame(recommendations)
st.dataframe(
    recommendation_df,
    use_container_width=True,
    hide_index=True,
)

st.divider()


# ---------------------------------------------------------
# FUTURE ENHANCEMENTS
# ---------------------------------------------------------

st.subheader("🚀 Future Enhancements")

future_df = pd.DataFrame(
    {
        "Enhancement": [
            "Improved data-quality diagnostics",
            "Regional coverage and reporting-gap analysis",
            "Forecast baseline comparison",
            "Temporal backtesting and model monitoring",
            "GIS heatmap visualization",
            "Automated anomaly alerts",
            "Live API integration",
            "Mobile dashboard",
        ],
        "Status": [
            "Recommended",
            "Recommended",
            "In Progress / Experimental",
            "Recommended",
            "Future Work",
            "Future Work",
            "Future Work",
            "Future Work",
        ],
    }
)

st.dataframe(
    future_df,
    use_container_width=True,
    hide_index=True,
)

st.divider()


# ---------------------------------------------------------
# PROJECT CONCLUSION
# ---------------------------------------------------------

st.subheader("📌 Conclusion")

st.markdown(
    f"""
The **{dataset.title()}** view presents the available outputs for data
exploration, trend analysis, correlation analysis, anomaly detection, and
model evaluation.

These outputs can support further investigation and planning, but they do not
automatically establish data completeness, causation, or forecasting reliability.
Review reporting gaps and model performance before using results to guide
operational decisions.
"""
)

st.divider()


# ---------------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------------

st.subheader("📥 Download Recommendation Report")

report_lines = [
    "AADHAAR INSIGHTS ANALYTICS PLATFORM",
    "Recommendation Report",
    "",
    f"Dataset: {dataset.title()}",
    f"Trend data points: {trend_points}",
    f"Reported anomalies: {anomaly_count}",
    f"Strong correlation pairs (|r| > 0.7): {strong_corr}",
    "",
]

if best_model is not None:
    report_lines.extend(
        [
            f"Highest-R² model in loaded metrics: {best_model['Model']}",
            f"R²: {r2:.4f}",
            f"MAE: {mae:.4f}",
            f"RMSE: {rmse:.4f}",
        ]
    )
else:
    report_lines.append("Model metrics: unavailable or invalid")

report_lines.extend(
    [
        "",
        "Recommendations:",
        *[
            f"{row['Priority']}: {row['Recommendation']} — {row['Next step']}"
            for row in recommendations
        ],
        "",
        "Note: Metrics describe the loaded evaluation outputs. They do not, by themselves, establish forecasting suitability.",
        "",
        "Generated by Aadhaar Insights Analytics Platform",
    ]
)

report = "\n".join(report_lines)

st.download_button(
    label="📄 Download Recommendation Report",
    data=report,
    file_name=f"{dataset}_recommendation_report.txt",
    mime="text/plain",
)

st.caption("Recommendations are based on the analysis outputs currently available.")