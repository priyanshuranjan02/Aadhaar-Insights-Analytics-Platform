"""
===============================================================================
Aadhaar Insights Analytics Platform
Growth Analysis
===============================================================================
Author : Priyanshu Ranjan
===============================================================================
"""
import os
import sys

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, APP_DIR)
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import plotly.express as px
import pandas as pd

from src.data_loader import (
    load_yearly_growth,
    load_monthly_growth,
    load_quarterly_growth,
)

from components.metrics import metric_row


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Growth Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Growth Analysis")

st.markdown(
"""
Explore Aadhaar record growth across different
time dimensions.

This page highlights percentage growth trends,
allowing comparison of month-wise, quarter-wise,
and year-wise performance.
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

yearly = load_yearly_growth(dataset)

monthly = load_monthly_growth(dataset)

quarterly = load_quarterly_growth(dataset)


# =============================================================================
# DATA PREPARATION
# =============================================================================

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

monthly["date_month_name"] = pd.Categorical(
    monthly["date_month_name"],
    categories=month_order,
    ordered=True,
)

monthly = monthly.sort_values(
    "date_month_name"
)

quarterly = quarterly.sort_values(
    "date_quarter"
)

yearly = yearly.sort_values(
    "date_year"
)

# Replace NaN growth values (first period)
monthly["growth_percent"] = (
    monthly["growth_percent"].fillna(0)
)

quarterly["growth_percent"] = (
    quarterly["growth_percent"].fillna(0)
)

yearly["growth_percent"] = (
    yearly["growth_percent"].fillna(0)
)

# Quarter labels
quarterly["quarter"] = (
    "Q" + quarterly["date_quarter"].astype(str)
)


# =============================================================================
# KPI CARDS
# =============================================================================

highest_growth = monthly["growth_percent"].max()

lowest_growth = monthly["growth_percent"].min()

metric_row([
    {
        "title": "Years",
        "value": yearly.shape[0],
    },
    {
        "title": "Months",
        "value": monthly.shape[0],
    },
    {
        "title": "Highest Growth",
        "value": f"{highest_growth:.2f}%",
    },
    {
        "title": "Lowest Growth",
        "value": f"{lowest_growth:.2f}%",
    },
])

st.divider()


# =============================================================================
# MONTHLY GROWTH
# =============================================================================

st.subheader("📅 Monthly Growth Trend")

fig = px.line(
    monthly,
    x="date_month_name",
    y="growth_percent",
    markers=True,
)

fig.update_traces(
    line=dict(
        width=4,
        color="green",
    ),
    marker=dict(
        size=10,
    ),
    hovertemplate=(
        "<b>%{x}</b>"
        "<br>Growth: %{y:.2f}%"
        "<extra></extra>"
    ),
)

fig.update_layout(
    height=500,
    xaxis_title="Month",
    yaxis_title="Growth (%)",
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
)

# Zero growth reference line
fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="red",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()


# =============================================================================
# QUARTERLY & YEARLY GROWTH
# =============================================================================

left, right = st.columns(2)

# -----------------------------------------------------------------------------
# QUARTERLY GROWTH
# -----------------------------------------------------------------------------

with left:

    st.subheader("📊 Quarterly Growth")

    quarterly["color"] = quarterly["growth_percent"].apply(
        lambda x: "Positive" if x >= 0 else "Negative"
    )

    fig = px.bar(
        quarterly,
        x="quarter",
        y="growth_percent",
        text="growth_percent",
        color="color",
        color_discrete_map={
            "Positive": "#2E8B57",
            "Negative": "#DC143C",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        height=450,
        xaxis_title="Quarter",
        yaxis_title="Growth (%)",
        legend_title="Growth",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# YEARLY GROWTH
# -----------------------------------------------------------------------------

with right:

    st.subheader("📅 Yearly Growth")

    if len(yearly) == 1:

        st.metric(
            "Available Year",
            int(yearly.iloc[0]["date_year"]),
        )

        st.metric(
            "Growth",
            f"{yearly.iloc[0]['growth_percent']:.2f}%",
        )

    else:

        yearly["color"] = yearly["growth_percent"].apply(
            lambda x: "Positive" if x >= 0 else "Negative"
        )

        fig = px.bar(
            yearly,
            x="date_year",
            y="growth_percent",
            text="growth_percent",
            color="color",
            color_discrete_map={
                "Positive": "#2E8B57",
                "Negative": "#DC143C",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>",
        )

        fig.update_layout(
            height=450,
            xaxis_title="Year",
            yaxis_title="Growth (%)",
            legend_title="Growth",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


st.divider()


# =============================================================================
# GROWTH COMPARISON
# =============================================================================

st.subheader("📈 Monthly Growth Comparison")

comparison = monthly.copy()

comparison["Direction"] = comparison["growth_percent"].apply(
    lambda x: "Increase" if x >= 0 else "Decrease"
)

fig = px.bar(
    comparison,
    x="date_month_name",
    y="growth_percent",
    color="Direction",
    text="growth_percent",
    color_discrete_map={
        "Increase": "#2E8B57",
        "Decrease": "#DC143C",
    },
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>",
)

fig.update_layout(
    height=500,
    xaxis_title="Month",
    yaxis_title="Growth (%)",
    legend_title="Direction",
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
)

fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="black",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()


# =============================================================================
# GROWTH SUMMARY
# =============================================================================

st.subheader("📋 Growth Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    highest_month = monthly.loc[
        monthly["growth_percent"].idxmax()
    ]

    highest_quarter = quarterly.loc[
        quarterly["growth_percent"].idxmax()
    ]

    st.markdown(
        f"""
### 📈 Highest Growth

- **Month:** {highest_month['date_month_name']}
- **Growth:** {highest_month['growth_percent']:.2f}%

- **Quarter:** Q{int(highest_quarter['date_quarter'])}
- **Growth:** {highest_quarter['growth_percent']:.2f}%
"""
    )

with summary_col2:

    lowest_month = monthly.loc[
        monthly["growth_percent"].idxmin()
    ]

    lowest_quarter = quarterly.loc[
        quarterly["growth_percent"].idxmin()
    ]

    st.markdown(
        f"""
### 📉 Lowest Growth

- **Month:** {lowest_month['date_month_name']}
- **Growth:** {lowest_month['growth_percent']:.2f}%

- **Quarter:** Q{int(lowest_quarter['date_quarter'])}
- **Growth:** {lowest_quarter['growth_percent']:.2f}%
"""
    )


st.divider()


# =============================================================================
# RAW DATA
# =============================================================================

st.subheader("📄 Growth Data")

with st.expander("View Growth Tables"):

    tab1, tab2, tab3 = st.tabs(
        [
            "Monthly",
            "Quarterly",
            "Yearly",
        ]
    )

    with tab1:

        st.dataframe(
            monthly,
            use_container_width=True,
            hide_index=True,
        )

    with tab2:

        st.dataframe(
            quarterly,
            use_container_width=True,
            hide_index=True,
        )

    with tab3:

        st.dataframe(
            yearly,
            use_container_width=True,
            hide_index=True,
        )


st.divider()


# =============================================================================
# DOWNLOADS
# =============================================================================

st.subheader("📥 Download Growth Data")

col1, col2, col3 = st.columns(3)

with col1:

    st.download_button(
        label="📄 Monthly Growth CSV",
        data=monthly.to_csv(index=False),
        file_name="monthly_growth.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:

    st.download_button(
        label="📄 Quarterly Growth CSV",
        data=quarterly.to_csv(index=False),
        file_name="quarterly_growth.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col3:

    st.download_button(
        label="📄 Yearly Growth CSV",
        data=yearly.to_csv(index=False),
        file_name="yearly_growth.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()


# =============================================================================
# KEY OBSERVATIONS
# =============================================================================

st.subheader("💡 Key Observations")

positive_months = (monthly["growth_percent"] > 0).sum()
negative_months = (monthly["growth_percent"] < 0).sum()

st.info(
    f"""
• **Positive Growth Months:** {positive_months}

• **Negative Growth Months:** {negative_months}

• Growth percentages are calculated using period-over-period percentage change.

• The first period shows **0% growth** because no previous period exists for comparison.
"""
)


st.divider()


# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    "Aadhaar Insights Analytics Platform • Growth Analysis"
)