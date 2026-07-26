"""
===============================================================================
Aadhaar Insights Analytics Platform
Dashboard
===============================================================================
Author : Priyanshu Ranjan
===============================================================================
"""
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import streamlit as st
import plotly.express as px
import pandas as pd

from src.data_loader import (
    load_dashboard_metrics,
    load_top_states,
    load_top_districts,
    load_executive_dashboard,
    load_state_summary,
    load_bottom_districts,
    load_monthly_trend
)

from components.metrics import metric_row


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dashboard")

st.markdown(
"""
Explore interactive analytics across the Aadhaar datasets.

Select a dataset below to dynamically update all dashboard
visualizations.
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

metrics = load_dashboard_metrics()

top_states = load_top_states(dataset)

top_districts = load_top_districts(dataset)

bottom_districts = load_bottom_districts(dataset)

monthly_trend = load_monthly_trend(dataset)

executive = load_executive_dashboard()

state_summary = load_state_summary(dataset)

executive = executive[
    executive["dataset"].str.lower() == dataset
].iloc[0]


# =============================================================================
# KPI CARDS
# =============================================================================

current = metrics[
    metrics["dataset"].str.lower() == dataset
].iloc[0]

metric_row([
    {
        "title": "Records",
        "value": f"{int(current['total_records']):,}"
    },
    {
        "title": "Features",
        "value": int(current["total_columns"])
    },
    {
        "title": "States",
        "value": int(current["unique_states"])
    },
    {
        "title": "Districts",
        "value": int(current["unique_districts"])
    },
])

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

st.divider()

st.subheader("📋 Executive Summary")

left, right = st.columns(2)

with left:

    st.markdown(
        f"""
**Dataset**

{executive["dataset"].title()}

**Records**

{executive["records"]:,}

**Features**

{executive["features"]:,}
"""
    )

with right:

    st.markdown(
        f"""
**States**

{executive["states"]}

**Districts**

{executive["districts"]}

**Missing Values**

{executive["missing_values"]}

**Duplicate Rows**

{executive["duplicate_rows"]}
"""
    )

st.divider()


# =============================================================================
# PREPARE DATA FOR CHARTS
# =============================================================================

top_states_chart = (
    top_states[
        ["state", "state_record_count"]
    ]
    .sort_values(
        by="state_record_count",
        ascending=False
    )
)

top_districts_chart = (
    top_districts[
        ["district", "district_record_count"]
    ]
    .sort_values(
        by="district_record_count",
        ascending=False
    )
)

bottom_districts_chart = (
    bottom_districts[
        [
            "district",
            "district_record_count",
        ]
    ]
    .sort_values(
        by="district_record_count",
        ascending=True,
    )
)

state_contribution = (
    state_summary[
        [
            "state",
            "contribution_percent"
        ]
    ]
    .sort_values(
        by="contribution_percent",
        ascending=False
    )
)

# =============================================================================
# STATE ANALYSIS
# =============================================================================

left, right = st.columns(2)

with left:

    st.subheader("🏆 Top 10 States by Aadhaar Records")

    fig = px.bar(
        top_states_chart,
        x="state_record_count",
        y="state",
        orientation="h",
        text="state_record_count",
        color="state_record_count",
        color_continuous_scale="Blues",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Records: %{x:,}<extra></extra>",
    )

    fig.update_layout(
        height=500,
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Records",
        yaxis_title="State",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with right:

    st.subheader("🥧 State Contribution")

    fig = px.pie(
        state_summary,
        names="state",
        values="contribution_percent",
        hole=0.55,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
    )

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

st.divider()

# =============================================================================
# DISTRICT ANALYSIS
# =============================================================================

left, right = st.columns(2)

with left:

    st.subheader("🏆 Top 10 Districts by Aadhaar Records")

    fig = px.bar(
        top_districts_chart,
        x="district_record_count",
        y="district",
        orientation="h",
        text="district_record_count",
        color="district_record_count",
        color_continuous_scale="Greens",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Records: %{x:,}<extra></extra>",
    )

    fig.update_layout(
        height=500,
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Records",
        yaxis_title="District",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with right:

    st.subheader("📉 Bottom 10 Districts")

    fig = px.bar(
        bottom_districts_chart,
        x="district_record_count",
        y="district",
        orientation="h",
        text="district_record_count",
        color="district_record_count",
        color_continuous_scale="Reds",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Records: %{x:,}<extra></extra>",
    )

    fig.update_layout(
        height=500,
        yaxis=dict(categoryorder="total descending"),
        xaxis_title="Records",
        yaxis_title="District",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )