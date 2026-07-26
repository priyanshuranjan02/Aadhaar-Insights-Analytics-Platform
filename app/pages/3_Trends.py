"""
===============================================================================
Aadhaar Insights Analytics Platform
Trend Analysis
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
    load_yearly_trend,
    load_monthly_trend,
    load_quarterly_trend,
    load_weekly_trend,
    load_weekend_trend,
)

from components.metrics import metric_row


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Trend Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trend Analysis")

st.markdown(
"""
Explore historical Aadhaar activity across different
time dimensions.

The charts below help identify seasonal behaviour,
weekly patterns, quarterly variations and yearly trends.
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

yearly = load_yearly_trend(dataset)

monthly = load_monthly_trend(dataset)

quarterly = load_quarterly_trend(dataset)

weekly = load_weekly_trend(dataset)

weekend = load_weekend_trend(dataset)


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

weekly = weekly.sort_values(
    "date_week"
)

yearly = yearly.sort_values(
    "date_year"
)


weekend["day_type"] = weekend[
    "date_is_weekend"
].map(
    {
        True: "Weekend",
        False: "Weekday",
        "True": "Weekend",
        "False": "Weekday",
        1: "Weekend",
        0: "Weekday",
    }
)

weekend["day_type"] = weekend[
    "day_type"
].fillna(
    weekend["date_is_weekend"].astype(str)
)


# =============================================================================
# KPI CARDS
# =============================================================================

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
        "title": "Quarters",
        "value": quarterly.shape[0],
    },
    {
        "title": "Weeks",
        "value": weekly.shape[0],
    },
])

st.divider()


# =============================================================================
# MONTHLY TREND
# =============================================================================

st.subheader("📅 Monthly Aadhaar Trend")

fig = px.line(
    monthly,
    x="date_month_name",
    y="record_count",
    markers=True,
)

fig.update_traces(
    line=dict(width=4),
    marker=dict(size=10),
    hovertemplate="<b>%{x}</b><br>Records: %{y:,}<extra></extra>",
)

fig.update_layout(
    height=500,
    xaxis_title="Month",
    yaxis_title="Records",
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
# QUARTERLY TREND
# =============================================================================

left, right = st.columns(2)

with left:

    st.subheader("📊 Quarterly Trend")

    fig = px.bar(
        quarterly,
        x="date_quarter",
        y="record_count",
        text="record_count",
        color="record_count",
        color_continuous_scale="Blues",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Records: %{y:,}<extra></extra>",
    )

    fig.update_layout(
        height=450,
        xaxis_title="Quarter",
        yaxis_title="Records",
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
# WEEKLY TREND
# =============================================================================

with right:

    st.subheader("📅 Weekly Trend")

    fig = px.area(
        weekly,
        x="date_week",
        y="record_count",
    )

    fig.update_traces(
        hovertemplate="<b>Week %{x}</b><br>Records: %{y:,}<extra></extra>",
    )

    fig.update_layout(
        height=450,
        xaxis_title="Week Number",
        yaxis_title="Records",
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
# WEEKEND DISTRIBUTION
# =============================================================================

left, right = st.columns(2)

with left:

    st.subheader("🌤️ Weekday vs Weekend")

    fig = px.pie(
        weekend,
        names="day_type",
        values="record_count",
        hole=0.55,
        color_discrete_sequence=px.colors.sequential.Teal,
    )

    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Records: %{value:,}<extra></extra>",
    )

    fig.update_layout(
        height=450,
        showlegend=False,
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
# YEAR SUMMARY
# =============================================================================

with right:

    st.subheader("📅 Year Summary")

    if len(yearly) == 1:

        year = int(yearly.iloc[0]["date_year"])
        records = int(yearly.iloc[0]["record_count"])

        st.metric(
            label="Available Year",
            value=year,
        )

        st.metric(
            label="Total Records",
            value=f"{records:,}",
        )

    else:

        fig = px.bar(
            yearly,
            x="date_year",
            y="record_count",
            text="record_count",
            color="record_count",
            color_continuous_scale="Purples",
        )

        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Records: %{y:,}<extra></extra>",
        )

        fig.update_layout(
            height=450,
            xaxis_title="Year",
            yaxis_title="Records",
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
# TREND SUMMARY
# =============================================================================

st.subheader("📋 Trend Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    highest_month = monthly.loc[
        monthly["record_count"].idxmax()
    ]

    highest_quarter = quarterly.loc[
        quarterly["record_count"].idxmax()
    ]

    st.markdown(
        f"""
### Highlights

- 📈 **Highest Activity Month:** {highest_month['date_month_name']}
- 📊 **Records:** {int(highest_month['record_count']):,}

- 🏆 **Highest Activity Quarter:** Q{int(highest_quarter['date_quarter'])}
- 📊 **Records:** {int(highest_quarter['record_count']):,}
"""
    )


with summary_col2:

    highest_week = weekly.loc[
        weekly["record_count"].idxmax()
    ]

    weekday = weekend.loc[
        weekend["record_count"].idxmax()
    ]

    st.markdown(
        f"""
### Additional Insights

- 📅 **Highest Activity Week:** {int(highest_week['date_week'])}
- 📊 **Records:** {int(highest_week['record_count']):,}

- 🌤️ **Dominant Category:** {weekday['day_type']}
- 📊 **Records:** {int(weekday['record_count']):,}
"""
    )


st.divider()


# =============================================================================
# RAW DATA
# =============================================================================

with st.expander("📄 View Trend Data"):

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Monthly",
            "Quarterly",
            "Weekly",
            "Weekend",
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
            weekly,
            use_container_width=True,
            hide_index=True,
        )

    with tab4:
        st.dataframe(
            weekend,
            use_container_width=True,
            hide_index=True,
        )

    with tab5:
        st.dataframe(
            yearly,
            use_container_width=True,
            hide_index=True,
        )


st.divider()


# =============================================================================
# DOWNLOADS
# =============================================================================

st.subheader("📥 Download Trend Data")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.download_button(
        "Monthly CSV",
        monthly.to_csv(index=False),
        "monthly_trend.csv",
        "text/csv",
    )

with col2:
    st.download_button(
        "Quarterly CSV",
        quarterly.to_csv(index=False),
        "quarterly_trend.csv",
        "text/csv",
    )

with col3:
    st.download_button(
        "Weekly CSV",
        weekly.to_csv(index=False),
        "weekly_trend.csv",
        "text/csv",
    )

with col4:
    st.download_button(
        "Weekend CSV",
        weekend.to_csv(index=False),
        "weekend_distribution.csv",
        "text/csv",
    )

with col5:
    st.download_button(
        "Yearly CSV",
        yearly.to_csv(index=False),
        "yearly_trend.csv",
        "text/csv",
    )


st.divider()


# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    "Aadhaar Insights Analytics Platform • Trend Analysis"
)