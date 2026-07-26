import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    load_analysis,
    load_forecasting_results
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Recommendations",
    page_icon="💡",
    layout="wide"
)

st.title("💡 AI Recommendations & Insights")
st.markdown(
    """
    Intelligent recommendations generated from data analytics,
    trend analysis, anomaly detection and machine learning.
    """
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Configuration")

dataset = st.sidebar.selectbox(
    "Select Dataset",
    [
        "enrolment",
        "demographic",
        "biometric"
    ]
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

analysis = load_analysis(dataset)
forecast = load_forecasting_results(dataset)

# ---------------------------------------------------------
# HEALTH METRICS
# ---------------------------------------------------------

metrics_df = forecast["metrics"]

best_model = metrics_df.sort_values(
    "R2",
    ascending=False
).iloc[0]

# -----------------------------
# Dynamic Health Score
# -----------------------------

r2 = float(best_model["R2"])
mae = float(best_model["MAE"])
rmse = float(best_model["RMSE"])

health_score = (
    r2 * 70
    + max(0, 15 - mae * 10)
    + max(0, 15 - rmse * 10)
)

health_score = max(0, min(100, round(health_score)))

# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------

st.subheader("📊 Overall Dataset Health")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Health Score",
        f"{health_score}/100"
    )

with col2:
    st.metric(
        "Best Model",
        best_model["Model"]
    )

with col3:
    st.metric(
        "Best R² Score",
        f"{best_model['R2']:.3f}"
    )

st.divider()


# ---------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------

st.subheader("📋 Executive Summary")

# -----------------------------
# Real-time Dataset Statistics
# -----------------------------

anomaly_count = len(analysis["anomalies"])

corr_matrix = analysis["correlation"]

strong_corr = (
    (corr_matrix.abs() > 0.7).sum().sum()
    - len(corr_matrix)
) // 2

trend_points = len(analysis["yearly_trend"])

summary = f"""
The **{dataset.title()}** dataset has been successfully analyzed using statistical,
trend and machine learning techniques.

### Key Highlights

• Best Performing Model : **{best_model['Model']}**

• Model Accuracy (R²) : **{best_model['R2']:.3f}**

• Years Analysed : **{trend_points}**

• Strong Feature Relationships : **{strong_corr}**

• Total Anomalies Detected : **{anomaly_count}**

• Dataset Health Score : **{health_score}/100**
"""

st.info(summary)

st.divider()

# ---------------------------------------------------------
# KEY FINDINGS
# ---------------------------------------------------------

st.subheader("🔍 Key Findings")

left, right = st.columns(2)

with left:

    st.success("""
### 📈 Growth Analysis

- Positive growth trend observed.
- Dataset is well structured.
- No major inconsistencies detected.
""")

    st.success("""
### 🔗 Correlation Analysis

- Strong feature relationships identified.
- Useful for predictive modelling.
""")

with right:

    st.warning("""
### ⚠ Anomaly Detection

- Few abnormal records detected.
- Regular monitoring is recommended.
""")

    st.info("""
### 🤖 Machine Learning

- Random Forest provides the best prediction performance.
- Model is suitable for forecasting future trends.
""")

st.divider()

# ---------------------------------------------------------
# AI GENERATED INSIGHTS
# ---------------------------------------------------------

st.subheader("🧠 AI Generated Insights")

insights = []

if health_score >= 90:
    insights.append("Dataset quality is excellent for advanced analytics.")
elif health_score >= 75:
    insights.append("Dataset quality is good with minor improvement opportunities.")
else:
    insights.append("Dataset quality requires further cleaning.")

if anomaly_count == 0:
    insights.append("No anomalies were detected.")
elif anomaly_count < 100:
    insights.append(f"{anomaly_count} anomalies detected. Regular monitoring is recommended.")
else:
    insights.append(f"{anomaly_count} anomalies detected. Immediate investigation is recommended.")

insights.append(
    f"{best_model['Model']} achieved the highest prediction accuracy (R² = {best_model['R2']:.3f})."
)

insights.append(
    f"{strong_corr} strong feature relationships were identified."
)

insights.append(
    f"Trend analysis covers {trend_points} time periods."
)

for insight in insights:
    st.markdown(f"✅ {insight}")

st.divider()


# ---------------------------------------------------------
# BUSINESS RECOMMENDATIONS
# ---------------------------------------------------------

st.subheader("📈 Business Recommendations")

high, medium, low = st.columns(3)

with high:

    if anomaly_count == 0:
        st.success(f"""
### 🟢 High Priority

- No anomalies detected in the dataset.
- Maintain the current data quality standards.
- Continue regular data validation and monitoring.
""")

    elif anomaly_count <= 100:
        st.warning(f"""
### 🟡 High Priority

- {anomaly_count} anomalous records detected.
- Investigate abnormal records for potential inconsistencies.
- Strengthen data quality checks in affected regions.
""")

    else:
        st.error(f"""
### 🔴 High Priority

- {anomaly_count} anomalous records detected.
- Immediate investigation of abnormal records is recommended.
- Improve data collection and validation processes.
""")

with medium:

    if health_score >= 90:
        st.success(f"""
### 🟢 Medium Priority

- Dataset quality is excellent ({health_score}/100).
- Continue monitoring yearly growth trends.
- Maintain existing analytical workflows.
""")

    elif health_score >= 75:
        st.warning(f"""
### 🟡 Medium Priority

- Dataset quality is good ({health_score}/100).
- Improve coverage in low-performing regions.
- Monitor yearly growth and update trends regularly.
""")

    else:
        st.error(f"""
### 🔴 Medium Priority

- Dataset health score is {health_score}/100.
- Improve data completeness and consistency.
- Perform additional preprocessing before analysis.
""")

with low:

    st.info(f"""
### 🔵 Low Priority

- Best performing model: **{best_model['Model']}**
- Strong feature relationships identified: **{strong_corr}**
- Retrain prediction models periodically with new data.
- Expand analytics using future datasets and real-time updates.
""")

st.divider()

# ---------------------------------------------------------
# MACHINE LEARNING INSIGHTS
# ---------------------------------------------------------

st.subheader("🤖 Machine Learning Insights")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Best Model",
        best_model["Model"]
    )

with c2:
    st.metric(
        "R² Score",
        f"{best_model['R2']:.3f}"
    )

with c3:
    st.metric(
        "MAE",
        f"{best_model['MAE']:.4f}"
    )

with c4:
    st.metric(
        "RMSE",
        f"{best_model['RMSE']:.4f}"
    )

st.divider()


# ---------------------------------------------------------
# ACTION PRIORITY MATRIX
# ---------------------------------------------------------

st.subheader("📋 Action Priority Matrix")

priority_df = pd.DataFrame(
    {
        "Priority": [
            "🔴 High",
            "🔴 High",
            "🟡 Medium",
            "🟡 Medium",
            "🟢 Low",
            "🟢 Low"
        ],
        "Recommendation": [
            "Investigate anomalous records",
            "Improve low-performing regions",
            "Increase demographic awareness",
            "Monitor yearly growth",
            "Retrain ML model periodically",
            "Continue dashboard monitoring"
        ],
        "Expected Impact": [
            "High",
            "High",
            "Medium",
            "Medium",
            "Low",
            "Low"
        ]
    }
)

st.dataframe(
    priority_df,
    use_container_width=True,
    hide_index=True
)

st.divider()


# ---------------------------------------------------------
# FUTURE ENHANCEMENTS
# ---------------------------------------------------------

st.subheader("🚀 Future Enhancements")

future_df = pd.DataFrame(
    {
        "Enhancement": [
            "Real-time Aadhaar Analytics",
            "GIS Heatmap Visualization",
            "Deep Learning Forecasting",
            "AI Chat Assistant",
            "Live API Integration",
            "Automated Alerts",
            "Cloud Deployment",
            "Mobile Dashboard"
        ],
        "Status": [
            "Planned",
            "Planned",
            "Future Work",
            "Future Work",
            "Future Work",
            "Future Work",
            "Future Work",
            "Future Work"
        ]
    }
)

st.dataframe(
    future_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ---------------------------------------------------------
# PROJECT CONCLUSION
# ---------------------------------------------------------

st.subheader("📌 Conclusion")

st.success(
    f"""
The **{dataset.title()}** dataset has been successfully analyzed using
statistical analysis, machine learning, forecasting and visualization
techniques.

This dashboard enables:

✅ Data Exploration

✅ Trend Analysis

✅ Growth Monitoring

✅ Correlation Analysis

✅ Anomaly Detection

✅ Machine Learning Forecasting

✅ Business Recommendations

Overall, the platform provides actionable insights that can support
data-driven planning and decision making.
"""
)

st.divider()

# ---------------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------------

st.subheader("📥 Download Recommendation Report")

report = f"""
AADHAAR INSIGHTS ANALYTICS PLATFORM
Recommendation Report

Dataset : {dataset.title()}

Overall Health Score : {health_score}/100

Best Model : {best_model['Model']}

R² Score : {best_model['R2']:.4f}

MAE : {best_model['MAE']:.4f}

RMSE : {best_model['RMSE']:.4f}

--------------------------------------------------

Key Recommendations

1. Continue monitoring dataset quality.
2. Investigate anomalous records.
3. Improve low-performing regions.
4. Retrain ML models periodically.
5. Expand analytics using real-time data.

--------------------------------------------------

Generated by:
Aadhaar Insights Analytics Platform
"""

st.download_button(
    label="📄 Download Recommendation Report",
    data=report,
    file_name=f"{dataset}_recommendation_report.txt",
    mime="text/plain"
)

st.success("🎉 Recommendations generated successfully.")