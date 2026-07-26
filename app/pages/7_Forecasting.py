"""
===============================================================================
Forecasting
===============================================================================
Displays machine learning forecasting results generated during analytics.
===============================================================================
"""
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PROCESSED_DATA_DIR

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Forecasting")
st.caption(
    "Machine Learning Forecasting and Predictive Analytics"
)

# =============================================================================
# Constants
# =============================================================================

ANALYTICS_DIR = (
    PROCESSED_DATA_DIR /
    "analytics"
)

DATASETS = [
    "enrolment",
    "demographic",
    "biometric",
]

# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.header("Forecast Settings")

dataset = st.sidebar.selectbox(
    "Dataset",
    DATASETS,
)

dataset_dir = ANALYTICS_DIR / dataset

# =============================================================================
# Load Data
# =============================================================================

def load_csv(filename: str) -> pd.DataFrame:

    filepath = dataset_dir / filename

    if filepath.exists():
        return pd.read_csv(filepath)

    return pd.DataFrame()


metrics_df = load_csv(
    "model_metrics.csv"
)

forecast_df = load_csv(
    "forecast_month.csv"
)

importance_df = load_csv(
    "feature_importance.csv"
)

prediction_df = load_csv(
    "actual_vs_predicted.csv"
)

# =============================================================================
# Check Availability
# =============================================================================

if metrics_df.empty:

    st.warning(
        "Forecasting results not found.\n\n"
        "Run forecasting.py first."
    )

    st.stop()

# =============================================================================
# KPI Cards
# =============================================================================

best = metrics_df.iloc[0]

best_model = best["Model"]

best_r2 = best["R2"]

best_rmse = best["RMSE"]

forecast_horizon = len(
    forecast_df
)

card1, card2, card3, card4 = st.columns(4)

with card1:

    st.metric(
        "🏆 Best Model",
        best_model,
    )

with card2:

    st.metric(
        "📈 Best R²",
        f"{best_r2:.4f}",
    )

with card3:

    st.metric(
        "📉 Lowest RMSE",
        f"{best_rmse:.6f}",
    )

with card4:

    st.metric(
        "📅 Forecast Horizon",
        forecast_horizon,
    )

st.divider()


# =============================================================================
# Model Performance
# =============================================================================

st.subheader("📊 Model Performance")

styled_metrics = (
    metrics_df.style
    .highlight_max(
        subset=["R2"],
        color="#d4edda",
    )
    .highlight_min(
        subset=["RMSE"],
        color="#d4edda",
    )
    .format(
        {
            "MAE": "{:.6f}",
            "MSE": "{:.6f}",
            "RMSE": "{:.6f}",
            "MAPE": "{:.2f}",
            "R2": "{:.4f}",
        }
    )
)

st.dataframe(
    styled_metrics,
    use_container_width=True,
)

st.divider()

# =============================================================================
# Actual vs Predicted
# =============================================================================

st.subheader("🎯 Actual vs Predicted")

if prediction_df.empty:

    st.info(
        "Prediction results not available."
    )

else:

    plot_df = prediction_df.copy()

    # Downsample large datasets
    MAX_POINTS = 2000

    if len(plot_df) > MAX_POINTS:

        step = max(
            len(plot_df) // MAX_POINTS,
            1,
        )

        plot_df = plot_df.iloc[::step]

    plot_df = plot_df.reset_index(
        drop=True
    )

    plot_df["Sample"] = plot_df.index + 1

    fig = px.line(
        plot_df,
        x="Sample",
        y=[
            "Actual",
            "Predicted",
        ],
        title="Actual vs Predicted Values",
    )

    fig.update_layout(
        xaxis_title="Samples",
        yaxis_title="Value",
        hovermode="x unified",
        legend_title="Series",
        height=550,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        f"Displaying {len(plot_df):,} sampled observations "
        f"from {len(prediction_df):,} total predictions."
    )

st.divider()


# =============================================================================
# Future Forecast
# =============================================================================

st.subheader("📈 Future Forecast")

if forecast_df.empty:

    st.info(
        "Forecast data not available."
    )

else:

    fig = px.line(
        forecast_df,
        x="date_month",
        y="prediction",
        markers=True,
        title="Future Forecast",
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Forecast Value",
        hovermode="x unified",
        height=500,
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True,
    )

st.divider()

# =============================================================================
# Feature Importance
# =============================================================================

st.subheader("📌 Top Feature Importance")

if importance_df.empty:

    st.info(
        "Feature importance not available."
    )

else:

    top_features = (
        importance_df
        .head(20)
        .sort_values(
            "Importance",
            ascending=True,
        )
    )

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 20 Most Important Features",
    )

    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=700,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

st.divider()

# =============================================================================
# Error Analysis
# =============================================================================

st.subheader("📉 Error Analysis")

if prediction_df.empty:

    st.info(
        "Prediction data not available."
    )

else:

    fig = px.histogram(
        prediction_df,
        x="Residual",
        nbins=50,
        title="Residual Distribution",
    )

    fig.update_layout(
        xaxis_title="Residual",
        yaxis_title="Frequency",
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    residual_mean = prediction_df[
        "Residual"
    ].mean()

    residual_std = prediction_df[
        "Residual"
    ].std()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Mean Residual",
            f"{residual_mean:.6f}",
        )

    with col2:

        st.metric(
            "Residual Std Dev",
            f"{residual_std:.6f}",
        )

st.divider()


# =============================================================================
# Download Results
# =============================================================================

st.subheader("📥 Download Results")

download_col1, download_col2 = st.columns(2)

with download_col1:

    st.download_button(
        label="📊 Download Model Metrics",
        data=metrics_df.to_csv(index=False),
        file_name=f"{dataset}_model_metrics.csv",
        mime="text/csv",
    )

    st.download_button(
        label="📈 Download Forecast",
        data=forecast_df.to_csv(index=False),
        file_name=f"{dataset}_forecast.csv",
        mime="text/csv",
    )

with download_col2:

    st.download_button(
        label="📌 Download Feature Importance",
        data=importance_df.to_csv(index=False),
        file_name=f"{dataset}_feature_importance.csv",
        mime="text/csv",
    )

    if not prediction_df.empty:

        st.download_button(
            label="🎯 Download Predictions",
            data=prediction_df.to_csv(index=False),
            file_name=f"{dataset}_actual_vs_predicted.csv",
            mime="text/csv",
        )

st.divider()

# =============================================================================
# Model Insights
# =============================================================================

st.subheader("💡 Model Insights")

best = metrics_df.iloc[0]

model = best["Model"]
r2 = float(best["R2"])
rmse = float(best["RMSE"])
mape = float(best["MAPE"])

if r2 >= 0.90:
    performance = "Excellent"

elif r2 >= 0.75:
    performance = "Very Good"

elif r2 >= 0.50:
    performance = "Good"

elif r2 >= 0.25:
    performance = "Moderate"

else:
    performance = "Needs Improvement"

st.success(
    f"""
**Best Model:** {model}

**Overall Performance:** {performance}

• R² Score: **{r2:.4f}**

• RMSE: **{rmse:.6f}**

• MAPE: **{mape:.2f}%**

The forecasting pipeline selected **{model}** as the best-performing
model based on evaluation metrics. Feature importance analysis highlights
the variables contributing most to the predictions, while the residual
distribution provides insight into model prediction errors.
"""
)

st.divider()

# =============================================================================
# Footer
# =============================================================================

st.markdown(
    """
---
<center>

**Aadhaar Insights Analytics Platform**

Forecasting Dashboard • Machine Learning • Predictive Analytics

Developed using **Streamlit**, **Scikit-Learn**, **Pandas**, and **Plotly**

</center>
""",
    unsafe_allow_html=True,
)