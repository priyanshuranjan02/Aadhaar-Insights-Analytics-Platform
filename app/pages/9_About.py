import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About Aadhaar Insights Analytics Platform")
st.caption(
    "An AI-powered analytics platform for exploring Aadhaar enrolment, demographic updates, "
    "biometric updates and forecasting future trends."
)

st.divider()

# ---------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------

st.header("📌 Project Overview")

st.markdown("""
The **Aadhaar Insights Analytics Platform** is an end-to-end data analytics and
machine learning application developed to analyse large-scale Aadhaar datasets
published by UIDAI.

The platform transforms raw enrolment, demographic update and biometric update
datasets into interactive dashboards, predictive models and actionable
recommendations.

It combines statistical analysis, data visualization and machine learning to
help users discover hidden patterns, identify anomalies and forecast future
trends for informed decision making.
""")

st.divider()

# ---------------------------------------------------------
# PROJECT OBJECTIVES
# ---------------------------------------------------------

st.header("🎯 Project Objectives")

col1, col2 = st.columns(2)

with col1:

    st.success("""
### Analytics Objectives

- Explore large-scale Aadhaar datasets

- Perform descriptive analytics

- Identify trends and growth patterns

- Detect anomalies

- Study feature correlations
""")

with col2:

    st.info("""
### Machine Learning Objectives

- Build predictive models

- Compare ML algorithms

- Forecast future trends

- Generate AI-powered recommendations

- Support data-driven decision making
""")

st.divider()


# ---------------------------------------------------------
# TECHNOLOGY STACK
# ---------------------------------------------------------

st.header("🛠 Technology Stack")

tech1, tech2, tech3 = st.columns(3)

with tech1:

    st.markdown("""
### 💻 Programming

- Python 3.14.6
- Pandas
- NumPy
- Scikit-learn
""")

with tech2:

    st.markdown("""
### 📊 Visualization

- Streamlit
- Plotly
- Matplotlib
- Seaborn
""")

with tech3:

    st.markdown("""
### 🤖 Machine Learning

- Linear Regression
- Random Forest
- Decision Tree
- Gradient Boosting
""")

st.divider()

# ---------------------------------------------------------
# DATASETS
# ---------------------------------------------------------

st.header("📂 Dataset Information")

st.markdown("""
The platform analyses three major UIDAI datasets that capture different aspects
of the Aadhaar ecosystem.
""")

dataset_df = pd.DataFrame(
    {
        "Dataset": [
            "Enrolment",
            "Demographic Updates",
            "Biometric Updates"
        ],
        "Description": [
            "Information related to Aadhaar enrolment across India.",
            "Records of demographic information updates such as name, address and date of birth.",
            "Records of biometric updates including fingerprints and iris authentication."
        ],
        "Purpose": [
            "Study enrolment trends and regional distribution.",
            "Analyse demographic update behaviour and growth.",
            "Understand biometric update patterns and forecasting."
        ]
    }
)

st.dataframe(
    dataset_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ---------------------------------------------------------
# PROJECT FEATURES
# ---------------------------------------------------------

st.header("✨ Key Features")

left, right = st.columns(2)

with left:

    st.success("""
### 📊 Data Analytics

✅ Interactive Dashboard

✅ Trend Analysis

✅ Growth Analysis

✅ Correlation Analysis

✅ Anomaly Detection
""")

with right:

    st.info("""
### 🤖 AI & Machine Learning

✅ Forecasting Models

✅ Model Comparison

✅ Feature Importance

✅ AI Recommendations

✅ Downloadable Reports
""")

st.divider()


# ---------------------------------------------------------
# MACHINE LEARNING WORKFLOW
# ---------------------------------------------------------

st.header("🧠 Machine Learning Workflow")

st.markdown("""
The forecasting pipeline follows a structured machine learning workflow to
generate accurate predictions and evaluate model performance.
""")

workflow = [
    "📥 Data Collection",
    "🧹 Data Cleaning & Preprocessing",
    "📊 Exploratory Data Analysis (EDA)",
    "⚙ Feature Engineering",
    "🤖 Model Training",
    "📈 Model Evaluation",
    "🔮 Future Forecasting",
    "💡 AI-powered Recommendations"
]

for step in workflow:
    st.markdown(f"- {step}")

st.divider()

# ---------------------------------------------------------
# FORECASTING MODELS
# ---------------------------------------------------------

st.header("📈 Machine Learning Models")

models_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Gradient Boosting Regressor"
    ],
    "Purpose": [
        "Baseline prediction model",
        "Tree-based regression analysis",
        "High accuracy ensemble learning",
        "Boosted ensemble forecasting"
    ],
    "Evaluation Metrics": [
        "MAE, RMSE, MAPE, R²",
        "MAE, RMSE, MAPE, R²",
        "MAE, RMSE, MAPE, R²",
        "MAE, RMSE, MAPE, R²"
    ]
})

st.dataframe(
    models_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ---------------------------------------------------------
# PROJECT ARCHITECTURE
# ---------------------------------------------------------

st.header("🏗 Project Architecture")

st.code("""
                 Raw UIDAI Datasets
                        │
                        ▼
          Data Loading & Validation
                        │
                        ▼
        Data Cleaning & Preprocessing
                        │
                        ▼
         Exploratory Data Analysis
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Analytics      Machine Learning   Forecasting
        │               │               │
        └───────────────┼───────────────┘
                        ▼
            Streamlit Interactive Dashboard
                        ▼
            AI Recommendations & Reports
""", language="text")

st.divider()

# ---------------------------------------------------------
# PROJECT STRUCTURE
# ---------------------------------------------------------

st.header("📂 Project Structure")

st.code("""
Aadhaar Insights/
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── Dashboard
│       ├── Trends
│       ├── Growth
│       ├── Correlation
│       ├── Anomalies
│       ├── Forecasting
│       ├── Recommendations
│       └── About
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── analytics.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── preprocessing.py
│   └── utils.py
│
├── reports/
│
└── requirements.txt
""", language="text")

st.divider()


# ---------------------------------------------------------
# DEVELOPER INFORMATION
# ---------------------------------------------------------

st.header("👨‍💻 Developer")

developer_col1, developer_col2 = st.columns([1, 3])

with developer_col1:
    st.image(
        "app/assets/my-profileImage.png",
        width=300
    )

with developer_col2:
    st.markdown("""
### Priyanshu Ranjan
**Aspiring Data Analyst**

**B.Tech Computer Science & Engineering (AI & ML)**  
VIT Bhopal University

**Project:** Aadhaar Insights Analytics Platform

This project was developed to demonstrate advanced skills in data analytics,
machine learning, visualization and interactive dashboard development using
real-world UIDAI datasets.
""")

st.divider()

# ---------------------------------------------------------
# PROJECT STATISTICS
# ---------------------------------------------------------

st.header("📊 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Datasets", "3")

with col2:
    st.metric("Dashboard Pages", "9")

with col3:
    st.metric("ML Models", "4")

with col4:
    st.metric("Analytics Modules", "8+")

st.divider()

# ---------------------------------------------------------
# ACKNOWLEDGEMENTS
# ---------------------------------------------------------

st.header("🙏 Acknowledgements")

st.markdown("""
The development of this project was made possible with the support of:

- **UIDAI (Unique Identification Authority of India)** for providing publicly available Aadhaar datasets.
- **Open-source Python Community** for libraries including Pandas, NumPy, Scikit-learn, Plotly and Streamlit.
""")

st.divider()

# ---------------------------------------------------------
# REFERENCES
# ---------------------------------------------------------

st.header("📚 References")

st.markdown("""
- UIDAI Open Data Portal
- Streamlit Documentation
- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
- Plotly Documentation
""")

st.divider()

# ---------------------------------------------------------
# PROJECT HIGHLIGHTS
# ---------------------------------------------------------

st.header("🏆 Project Highlights")

highlights = [
    "📊 Interactive multi-page analytics dashboard",
    "📈 Trend and growth analysis",
    "🔗 Correlation analysis",
    "🚨 Anomaly detection",
    "🤖 Machine learning forecasting",
    "📉 Model comparison using multiple evaluation metrics",
    "💡 AI-powered recommendation engine",
    "📄 Downloadable executive reports",
    "⚡ Dynamic insights generated from real data"
]

for item in highlights:
    st.markdown(f"✅ {item}")

st.divider()

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.success("""
### 🎉 Project Successfully Completed

Thank you for exploring the **Aadhaar Insights Analytics Platform**.

This application demonstrates how modern Data Analytics, Machine Learning and
Interactive Visualization techniques can transform raw government datasets into
meaningful insights for informed decision-making.

**© 2026 Priyanshu Ranjan | Aspiring Data Analyst**
""")