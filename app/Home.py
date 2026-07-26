"""
===============================================================================
Aadhaar Insights Analytics Platform
Home Page
===============================================================================
Author : Priyanshu Ranjan
===============================================================================
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.data_loader import (
    load_state_summary,
    load_district_summary,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aadhaar Insights Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Aadhaar Insights")

    st.caption("End-to-End Analytics Platform")

    st.divider()

    st.markdown("""
### 📊 Project

**Datasets:** 3

**Pages:** 9

**ML Models:** 4

**Framework:** Streamlit

**Developer:** Priyanshu Ranjan
""")

    st.divider()

    st.caption("Version 1.0")
    st.caption("© 2026 Priyanshu Ranjan")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image("app/assets/AadhaarInsights_Logo.jpeg", width=130)

with col_title:
    st.title("Aadhaar Insights Analytics Platform")
    st.subheader(
        "End-to-End Analytics, Business Intelligence & Machine Learning Platform"
    )

st.write("""
An integrated analytics platform for exploring Aadhaar enrolment,
demographic updates, biometric updates, forecasting future trends,
and generating intelligent recommendations.
""")

st.divider()

# ---------------------------------------------------------
# Dynamic KPIs
# ---------------------------------------------------------

datasets = [
    "enrolment",
    "demographic",
    "biometric"
]

total_records = 0
states = set()
districts = set()

for ds in datasets:
    s = load_state_summary(ds)
    d = load_district_summary(ds)

    total_records += s["state_record_count"].sum()

    states.update(
        s["state"]
          .str.strip()
          .str.title()
          .dropna()
          .unique()
    )

    districts.update(d["district"].dropna().unique())

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{int(total_records):,}"
)

col2.metric(
    "States Covered",
    len(states)
)

col3.metric(
    "Districts Covered",
    len(districts)
)

col4.metric(
    "ML Models",
    "4",
    delta="Best: Random Forest"
)

st.divider()

# -----------------------------------------------------------------------------
# Project Overview
# -----------------------------------------------------------------------------
st.header("📌 Project Overview")

st.write("""
The Aadhaar Insights Analytics Platform provides an end-to-end analytics
solution for analysing Aadhaar enrolment, demographic updates and biometric
updates across India.

The platform combines:

- 📊 Interactive Analytics Dashboard
- 📈 Trend Analysis
- 📉 Growth Analysis
- 🔗 Correlation Analysis
- 🚨 Anomaly Detection
- 🤖 Machine Learning Forecasting
- 💡 AI-powered Recommendations
- 📄 Executive Reports
""")

st.divider()

# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
st.header("✨ Platform Features")

left, right = st.columns(2)

with left:
    st.success("""
### 📊 Analytics

✅ Dashboard

✅ Trends

✅ Growth

✅ Correlation

✅ Anomaly Detection
""")

with right:
    st.info("""
### 🤖 Machine Learning

✅ Forecasting

✅ Model Comparison

✅ AI Recommendations

✅ Executive Reports

✅ Interactive Visualizations
""")

st.divider()

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
st.info(
    "👈 Use the navigation menu in the left sidebar to explore each module of the Aadhaar Insights Analytics Platform."
)