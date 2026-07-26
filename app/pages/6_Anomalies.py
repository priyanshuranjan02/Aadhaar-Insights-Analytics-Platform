###############################################################################
# Anomaly Detection
###############################################################################
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

###############################################################################
# Page Configuration
###############################################################################

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="⚠️",
    layout="wide",
)

st.title("⚠️ Anomaly Detection")

st.markdown(
    """
    Detect and explore unusual records identified during
    statistical anomaly analysis. This page highlights the
    most significant anomalies, their severity distribution,
    and overall anomaly statistics.
    """
)

###############################################################################
# Data Loader
###############################################################################

@st.cache_data(show_spinner=False)
def load_anomaly_data(dataset: str):

    base_path = (
        Path(PROCESSED_DATA_DIR)
        / "analytics"
        / dataset.lower()
    )

    summary = pd.read_csv(
        base_path / "anomalies_summary.csv"
    )

    severity = pd.read_csv(
        base_path / "anomalies_severity.csv"
    )

    top100 = pd.read_csv(
        base_path / "anomalies_top100.csv"
    )

    return summary, severity, top100

###############################################################################
# Dataset Selection
###############################################################################

datasets = [
    "Enrolment",
    "Demographic",
    "Biometric",
]

selected_dataset = st.selectbox(
    "Select Dataset",
    datasets,
)

summary, severity, top100 = load_anomaly_data(
    selected_dataset
)

###############################################################################
# KPI Cards
###############################################################################

st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Anomalies",
        f"{int(summary.loc[0,'total_anomalies']):,}"
    )

with col2:

    st.metric(
        "Critical",
        f"{int(summary.loc[0,'critical']):,}"
    )

with col3:

    st.metric(
        "Average Score",
        f"{summary.loc[0,'avg_score']:.2f}"
    )

with col4:

    st.metric(
        "Maximum Score",
        f"{summary.loc[0,'max_score']:.2f}"
    )

###############################################################################
# Severity Distribution
###############################################################################

st.markdown("---")

left, right = st.columns([2, 1])

with left:

    st.subheader("Severity Distribution")

    fig = px.pie(
        severity,
        names="severity",
        values="count",
        hole=0.55,
        color="severity",
        color_discrete_map={
            "Low": "#2ECC71",
            "Medium": "#F1C40F",
            "High": "#E67E22",
            "Critical": "#E74C3C",
        },
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig.update_layout(
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
        legend_title="Severity",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with right:

    st.subheader("Severity Breakdown")

    severity_display = severity.copy()

    severity_display.columns = [
        "Severity",
        "Count",
    ]

    st.dataframe(
        severity_display,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        """
        **Severity Levels**

        🔴 Critical

        🟠 High

        🟡 Medium

        🟢 Low
        """
    )

st.markdown("---")


###############################################################################
# Top Anomalies
###############################################################################

st.subheader("📈 Top 10 Highest Anomalies")

top10 = (
    top100
    .nlargest(10, "anomaly_score")
    .copy()
)

top10["Record"] = (
    top10.index.astype(str)
)

fig = px.bar(
    top10,
    x="anomaly_score",
    y="Record",
    orientation="h",
    color="anomaly_score",
    color_continuous_scale="Reds",
    text="anomaly_score",
)

fig.update_layout(
    height=500,
    yaxis_title="Record",
    xaxis_title="Anomaly Score",
    coloraxis_showscale=False,
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

###############################################################################
# Top 100 Anomalies
###############################################################################

st.markdown("---")

st.subheader("📋 Top 100 Critical Anomalies")

###############################################################################
# Search
###############################################################################

search = st.text_input(
    "🔍 Search",
    placeholder="Search state, district or pincode...",
)

filtered = top100.copy()

###############################################################################
# Search by object columns
###############################################################################

if search:

    object_columns = filtered.select_dtypes(
        include="object"
    ).columns

    mask = pd.Series(
        False,
        index=filtered.index,
    )

    for column in object_columns:

        mask |= (
            filtered[column]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
            )
        )

    filtered = filtered.loc[mask]

###############################################################################
# Sort
###############################################################################

filtered = filtered.sort_values(
    "anomaly_score",
    ascending=False,
)

###############################################################################
# Display Columns
###############################################################################

preferred_columns = [

    "state",

    "district",

    "pincode",

    "severity",

    "anomaly_score",

]

display_columns = [

    column

    for column in preferred_columns

    if column in filtered.columns

]

remaining_columns = [

    column

    for column in filtered.columns

    if column not in display_columns

]

display_columns.extend(remaining_columns)

###############################################################################
# Data Table
###############################################################################

st.dataframe(

    filtered[display_columns],

    use_container_width=True,

    hide_index=True,

    height=500,

)

###############################################################################
# Download
###############################################################################

csv = filtered.to_csv(
    index=False,
).encode("utf-8")

st.download_button(

    label="⬇ Download Top 100 Anomalies",

    data=csv,

    file_name=f"{selected_dataset.lower()}_top100_anomalies.csv",

    mime="text/csv",

)

###############################################################################
# Summary Statistics
###############################################################################

st.markdown("---")

st.subheader("📊 Summary Statistics")

col1, col2 = st.columns(2)

with col1:

    st.dataframe(
        summary.T.rename(columns={0: "Value"}),
        use_container_width=True,
    )

with col2:

    score_stats = (
        top100["anomaly_score"]
        .describe()
        .round(2)
        .rename("Value")
    )

    st.dataframe(
        score_stats.to_frame(),
        use_container_width=True,
    )

###############################################################################
# Key Insights
###############################################################################

st.markdown("---")

st.subheader("💡 Key Insights")

highest = top100.iloc[0]

state = (
    highest["state"]
    if "state" in highest.index
    else "N/A"
)

district = (
    highest["district"]
    if "district" in highest.index
    else "N/A"
)

score = highest["anomaly_score"]

severity_level = (
    highest["severity"]
    if "severity" in highest.index
    else "N/A"
)

insight1, insight2 = st.columns(2)

with insight1:

    st.success(
        f"""
### Highest Ranked Anomaly

**State:** {state}

**District:** {district}

**Severity:** {severity_level}

**Score:** {score:.2f}
"""
    )

with insight2:

    st.info(
        f"""
### Overall Summary

• Total anomalies detected:
**{int(summary.loc[0,'total_anomalies']):,}**

• Average anomaly score:
**{summary.loc[0,'avg_score']:.2f}**

• Maximum anomaly score:
**{summary.loc[0,'max_score']:.2f}**
"""
    )

###############################################################################
# Score Distribution
###############################################################################

st.markdown("---")

st.subheader("📈 Distribution of Top 100 Scores")

hist = px.histogram(
    top100,
    x="anomaly_score",
    nbins=20,
    title="Anomaly Score Distribution",
)

hist.update_layout(
    height=450,
    bargap=0.08,
)

st.plotly_chart(
    hist,
    use_container_width=True,
)

###############################################################################
# Expandable Raw Dataset
###############################################################################

with st.expander(
    "📄 View Complete Top 100 Dataset"
):

    st.dataframe(
        top100,
        use_container_width=True,
        hide_index=True,
    )

###############################################################################
# Notes
###############################################################################

st.markdown("---")

st.subheader("📝 Interpretation")

st.markdown(
    """
- **Anomaly Score** indicates how unusual a record is compared with the overall dataset.

- Higher scores represent records that deviate more strongly from normal statistical patterns.

- These anomalies may indicate:
    - Data quality issues
    - Exceptional enrollment activity
    - Sudden demographic changes
    - Potential operational irregularities

- An anomaly does **not automatically imply an error or fraud**. It is a signal that the record deserves further investigation.
"""
)

###############################################################################
# Footer
###############################################################################

st.markdown("---")

st.caption(
    """
Aadhaar Insights Analytics Platform •
Anomaly Detection Module •
Powered by Statistical Z-Score Analysis
"""
)