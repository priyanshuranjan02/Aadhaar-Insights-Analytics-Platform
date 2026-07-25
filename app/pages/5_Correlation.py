"""
===============================================================================
Aadhaar Insights Analytics Platform
Correlation Analysis
===============================================================================
Author : Priyanshu Ranjan
===============================================================================
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_correlation
from components.metrics import metric_row


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Correlation Analysis",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Correlation Analysis")

st.markdown(
"""
Explore relationships between engineered features using
an interactive Pearson Correlation Matrix.

Correlation values range between **-1** and **1**.

- **+1** → Perfect Positive Correlation
- **0** → No Correlation
- **-1** → Perfect Negative Correlation
"""
)

st.divider()


# =============================================================================
# DATASET SELECTION
# =============================================================================

dataset = st.selectbox(
    "Select Dataset",
    (
        "enrolment",
        "demographic",
        "biometric",
    ),
)

st.divider()


# =============================================================================
# LOAD DATA
# =============================================================================

correlation = load_correlation(dataset)

# Remove NaN values (date_year generally becomes NaN)
correlation = correlation.fillna(0)
corr_matrix = correlation.copy()
corr_matrix = correlation

# =============================================================================
# KPI CALCULATIONS
# =============================================================================

corr_matrix = correlation.copy()

matrix = corr_matrix.values

upper = matrix[
    np.triu_indices_from(
        matrix,
        k=1,
    )
]

strong_positive = np.sum(
    upper >= 0.70
)

strong_negative = np.sum(
    upper <= -0.70
)

average_corr = np.mean(
    np.abs(upper)
)

metric_row([
    {
        "title": "Features",
        "value": corr_matrix.shape[0],
    },
    {
        "title": "Strong Positive",
        "value": int(strong_positive),
    },
    {
        "title": "Strong Negative",
        "value": int(strong_negative),
    },
    {
        "title": "Average Correlation",
        "value": f"{average_corr:.2f}",
    },
])

st.divider()


# =============================================================================
# CORRELATION HEATMAP
# =============================================================================

st.subheader("🔥 Correlation Heatmap")

fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
)

fig.update_layout(
    height=750,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
    coloraxis_colorbar=dict(
        title="Correlation"
    ),
)

fig.update_xaxes(
    side="bottom"
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()


# =============================================================================
# TOP POSITIVE & NEGATIVE CORRELATIONS
# =============================================================================

corr_pairs = (
    corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )
    .stack()
    .reset_index()
)

corr_pairs.columns = [
    "Feature 1",
    "Feature 2",
    "Correlation"
]

positive_corr = (
    corr_pairs
    .sort_values(
        by="Correlation",
        ascending=False
    )
    .head(10)
)

negative_corr = (
    corr_pairs
    .sort_values(
        by="Correlation",
        ascending=True
    )
    .head(10)
)

left, right = st.columns(2)

# =============================================================================
# TOP POSITIVE
# =============================================================================

with left:

    st.subheader("🟢 Top Positive Correlations")

    positive_corr["Pair"] = (
        positive_corr["Feature 1"]
        + " ↔ "
        + positive_corr["Feature 2"]
    )

    fig = px.bar(
        positive_corr,
        x="Correlation",
        y="Pair",
        orientation="h",
        text="Correlation",
        color="Correlation",
        color_continuous_scale="Greens",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b>"
            "<br>Correlation: %{x:.2f}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        height=500,
        xaxis_title="Correlation",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# =============================================================================
# TOP NEGATIVE
# =============================================================================

with right:

    st.subheader("🔴 Top Negative Correlations")

    negative_corr["Pair"] = (
        negative_corr["Feature 1"]
        + " ↔ "
        + negative_corr["Feature 2"]
    )

    fig = px.bar(
        negative_corr,
        x="Correlation",
        y="Pair",
        orientation="h",
        text="Correlation",
        color="Correlation",
        color_continuous_scale="Reds_r",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b>"
            "<br>Correlation: %{x:.2f}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        height=500,
        xaxis_title="Correlation",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

st.divider()


# =============================================================================
# FEATURE CORRELATION EXPLORER
# =============================================================================

st.subheader("🎯 Feature Correlation Explorer")

selected_feature = st.selectbox(
    "Select Feature",
    corr_matrix.columns,
)

feature_corr = (
    corr_matrix[selected_feature]
    .drop(selected_feature)
    .sort_values(
        ascending=False
    )
    .reset_index()
)

feature_corr.columns = [
    "Feature",
    "Correlation",
]

feature_corr["Direction"] = feature_corr[
    "Correlation"
].apply(
    lambda x: "Positive"
    if x >= 0
    else "Negative"
)

fig = px.bar(
    feature_corr,
    x="Correlation",
    y="Feature",
    orientation="h",
    text="Correlation",
    color="Direction",
    color_discrete_map={
        "Positive": "#2E8B57",
        "Negative": "#DC143C",
    },
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
    hovertemplate=(
        "<b>%{y}</b>"
        "<br>Correlation: %{x:.2f}"
        "<extra></extra>"
    ),
)

fig.update_layout(
    height=650,
    xaxis_title="Correlation",
    yaxis_title="",
    legend_title="Direction",
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
)

fig.add_vline(
    x=0,
    line_dash="dash",
    line_color="black",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()


# =============================================================================
# CORRELATION INSIGHTS
# =============================================================================

st.subheader("💡 Correlation Insights")

left, right = st.columns(2)

with left:

    strongest_positive = positive_corr.iloc[0]

    st.success(
        f"""
### Strongest Positive Relationship

**{strongest_positive['Feature 1']}**

↕

**{strongest_positive['Feature 2']}**

Correlation = **{strongest_positive['Correlation']:.2f}**
"""
    )

with right:

    strongest_negative = negative_corr.iloc[0]

    st.error(
        f"""
### Strongest Negative Relationship

**{strongest_negative['Feature 1']}**

↕

**{strongest_negative['Feature 2']}**

Correlation = **{strongest_negative['Correlation']:.2f}**
"""
    )

st.divider()


# =============================================================================
# CORRELATION GUIDE
# =============================================================================

st.subheader("📖 Correlation Strength Guide")

guide = pd.DataFrame(
    {
        "Correlation Range": [
            "0.90 – 1.00",
            "0.70 – 0.89",
            "0.40 – 0.69",
            "0.10 – 0.39",
            "0.00 – 0.09",
        ],
        "Interpretation": [
            "Very Strong",
            "Strong",
            "Moderate",
            "Weak",
            "Negligible",
        ],
    }
)

st.dataframe(
    guide,
    use_container_width=True,
    hide_index=True,
)

st.divider()


# =============================================================================
# RAW CORRELATION MATRIX
# =============================================================================

st.subheader("📄 Correlation Matrix")

with st.expander("View Correlation Matrix"):

    st.dataframe(
        corr_matrix,
        use_container_width=True,
    )

st.divider()


# =============================================================================
# DOWNLOAD
# =============================================================================

st.subheader("📥 Download Correlation Matrix")

st.download_button(
    label="📄 Download CSV",
    data=corr_matrix.to_csv(),
    file_name="correlation_matrix.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()


# =============================================================================
# KEY OBSERVATIONS
# =============================================================================

st.subheader("📌 Key Observations")

st.info(
    f"""
• Total Features Analysed: **{corr_matrix.shape[0]}**

• Strong Positive Correlations: **{strong_positive}**

• Strong Negative Correlations: **{strong_negative}**

• Average Absolute Correlation: **{average_corr:.2f}**

• Pearson Correlation values range between **-1** and **1**.

• Correlation indicates statistical association and should not be interpreted as causation.
"""
)

st.divider()


# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    "Aadhaar Insights Analytics Platform • Correlation Analysis"
)