# File Tree: Aadhaar Insights Deployment

**Generated:** 7/25/2026, 11:20:21 PM
**Root Path:** `/Users/priyanshuranjan02/Projects and Courses/Aadhaar Insights Deployment`

```
├── 📁 app
│   ├── 📁 assets
│   │   ├── 🖼️ AadhaarInsights_Logo.jpeg
│   │   └── 🖼️ my-profileImage.png
│   ├── 📁 components
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 cards.py
│   │   ├── 🐍 charts.py
│   │   ├── 🐍 footer.py
│   │   ├── 🐍 metrics.py
│   │   ├── 🐍 sidebar.py
│   │   ├── 🐍 tables.py
│   │   └── 🐍 theme.py
│   ├── 📁 pages
│   │   ├── 🐍 2_Dashboard.py
│   │   ├── 🐍 3_Trends.py
│   │   ├── 🐍 4_Growth.py
│   │   ├── 🐍 5_Correlation.py
│   │   ├── 🐍 6_Anomalies.py
│   │   ├── 🐍 7_Forecasting.py
│   │   ├── 🐍 8_Recommendation.py
│   │   └── 🐍 9_About.py
│   ├── 📁 utils
│   ├── 🐍 Home.py
│   └── 🐍 __init__.py
├── 📁 dashboard
│   ├── 📁 assets
│   │   ├── 📁 icons
│   │   ├── 📁 logos
│   │   ├── 📁 screenshots
│   │   │   ├── 🖼️ biometric_analysis.png
│   │   │   ├── 🖼️ demographic_analysis.png
│   │   │   ├── 🖼️ executive_overview.png
│   │   │   └── 🖼️ overall_analytics.png
│   │   └── 📁 theme
│   │       └── ⚙️ aadhaar_theme.json
│   └── 📁 powerbi
│       ├── 📕 Aadhaar_Insights_Dashboard.pdf
│       └── 📝 README.md
├── 📁 data
│   └── 📁 processed
│       └── 📁 analytics
│           ├── 📁 biometric
│           │   ├── 📄 actual_vs_predicted.csv
│           │   ├── 📄 anomalies_severity.csv
│           │   ├── 📄 anomalies_summary.csv
│           │   ├── 📄 anomalies_top100.csv
│           │   ├── 📄 bottom_districts.csv
│           │   ├── 📄 bottom_states.csv
│           │   ├── 📄 correlation.csv
│           │   ├── 📄 date_month_growth.csv
│           │   ├── 📄 date_month_name_growth.csv
│           │   ├── 📄 date_quarter_growth.csv
│           │   ├── 📄 date_year_growth.csv
│           │   ├── 📄 district_summary.csv
│           │   ├── 📄 feature_importance.csv
│           │   ├── 📄 forecast_month.csv
│           │   ├── 📄 forecast_summary.csv
│           │   ├── 📄 model_metrics.csv
│           │   ├── 📄 state_summary.csv
│           │   ├── 📄 top_districts.csv
│           │   ├── 📄 top_states.csv
│           │   ├── 📄 trend_date_is_weekend.csv
│           │   ├── 📄 trend_date_month.csv
│           │   ├── 📄 trend_date_month_name.csv
│           │   ├── 📄 trend_date_quarter.csv
│           │   ├── 📄 trend_date_week.csv
│           │   └── 📄 trend_date_year.csv
│           ├── 📁 demographic
│           │   ├── 📄 actual_vs_predicted.csv
│           │   ├── 📄 anomalies_severity.csv
│           │   ├── 📄 anomalies_summary.csv
│           │   ├── 📄 anomalies_top100.csv
│           │   ├── 📄 bottom_districts.csv
│           │   ├── 📄 bottom_states.csv
│           │   ├── 📄 correlation.csv
│           │   ├── 📄 date_month_growth.csv
│           │   ├── 📄 date_month_name_growth.csv
│           │   ├── 📄 date_quarter_growth.csv
│           │   ├── 📄 date_year_growth.csv
│           │   ├── 📄 district_summary.csv
│           │   ├── 📄 feature_importance.csv
│           │   ├── 📄 forecast_month.csv
│           │   ├── 📄 forecast_summary.csv
│           │   ├── 📄 model_metrics.csv
│           │   ├── 📄 state_summary.csv
│           │   ├── 📄 top_districts.csv
│           │   ├── 📄 top_states.csv
│           │   ├── 📄 trend_date_is_weekend.csv
│           │   ├── 📄 trend_date_month.csv
│           │   ├── 📄 trend_date_month_name.csv
│           │   ├── 📄 trend_date_quarter.csv
│           │   ├── 📄 trend_date_week.csv
│           │   ├── 📄 trend_date_week_ratio.csv
│           │   ├── 📄 trend_date_year.csv
│           │   └── 📄 trend_date_year_ratio.csv
│           ├── 📁 enrolment
│           │   ├── 📄 actual_vs_predicted.csv
│           │   ├── 📄 anomalies_severity.csv
│           │   ├── 📄 anomalies_summary.csv
│           │   ├── 📄 anomalies_top100.csv
│           │   ├── 📄 bottom_districts.csv
│           │   ├── 📄 bottom_states.csv
│           │   ├── 📄 correlation.csv
│           │   ├── 📄 date_month_growth.csv
│           │   ├── 📄 date_month_name_growth.csv
│           │   ├── 📄 date_quarter_growth.csv
│           │   ├── 📄 date_year_growth.csv
│           │   ├── 📄 district_summary.csv
│           │   ├── 📄 feature_importance.csv
│           │   ├── 📄 forecast_month.csv
│           │   ├── 📄 forecast_summary.csv
│           │   ├── 📄 model_metrics.csv
│           │   ├── 📄 state_summary.csv
│           │   ├── 📄 top_districts.csv
│           │   ├── 📄 top_states.csv
│           │   ├── 📄 trend_date_is_weekend.csv
│           │   ├── 📄 trend_date_month.csv
│           │   ├── 📄 trend_date_month_name.csv
│           │   ├── 📄 trend_date_quarter.csv
│           │   ├── 📄 trend_date_week.csv
│           │   └── 📄 trend_date_year.csv
│           ├── ⚙️ analytics_manifest.json
│           ├── 📄 analytics_summary.csv
│           ├── ⚙️ dashboard_config.json
│           ├── 📄 dashboard_metrics.csv
│           └── 📄 executive_dashboard.csv
├── 📁 documentation
│   ├── 📘 UIDAI Hackathon Report.docx
│   ├── 📕 UIDAI Hackathon Submission Report.pdf
│   └── 📕 UIDAI_9222.pdf
├── 📁 images
├── 📁 models
├── 📁 notebooks
│   ├── 📄 01_enrolment_eda.ipynb
│   ├── 📄 02_demographic_eda.ipynb
│   └── 📄 03_biometric_eda.ipynb
├── 📁 reports
│   ├── 📁 figures
│   └── 📁 generated
├── 📁 src
│   ├── 🐍 __init__.py
│   ├── 🐍 analytics.py
│   ├── 🐍 config.py
│   ├── 🐍 data_loader.py
│   ├── 🐍 feature_engineering.py
│   ├── 🐍 forecasting.py
│   ├── 🐍 preprocessing.py
│   ├── 🐍 recommendation_engine.py
│   └── 🐍 utils.py
├── 📁 tests
├── ⚙️ .gitattributes
├── ⚙️ .gitignore
├── 📝 Project Structure.md
├── 📝 README.md
└── 📄 requirements.txt
```

---
*Generated by FileTree Pro Extension*