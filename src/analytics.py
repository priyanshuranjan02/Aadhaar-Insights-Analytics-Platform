"""
===============================================================================
Project      : Aadhaar Insights
File         : analytics.py
Author       : Priyanshu Ranjan

Description:
    Analytical engine for generating KPIs, insights, rankings,
    trends, anomaly detection, and dashboard metrics.

===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import LOGGER, PROCESSED_DATA_DIR
from src.feature_engineering import (
    feature_engineer_all_datasets,
)
from src.preprocessing import preprocess_all_datasets
from src.data_loader import load_all_datasets


###############################################################################
# Helper Utilities
###############################################################################

def get_numeric_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return all numeric columns.
    """

    return list(
        df.select_dtypes(
            include=np.number
        ).columns
    )


###############################################################################

def get_categorical_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return all categorical columns.
    """

    return list(
        df.select_dtypes(
            exclude=np.number
        ).columns
    )


###############################################################################

def safe_sum(
    df: pd.DataFrame,
    column: str,
) -> float:
    """
    Safely calculate column sum.
    """

    if column not in df.columns:

        return 0.0

    return float(
        df[column].sum(skipna=True)
    )


###############################################################################

def safe_mean(
    df: pd.DataFrame,
    column: str,
) -> float:
    """
    Safely calculate column mean.
    """

    if column not in df.columns:

        return 0.0

    return float(
        df[column].mean(skipna=True)
    )


###############################################################################

def safe_count(
    df: pd.DataFrame,
    column: str,
) -> int:
    """
    Safely count non-null values.
    """

    if column not in df.columns:

        return 0

    return int(
        df[column].count()
    )


###############################################################################
# KPI Calculation
###############################################################################

def calculate_kpis(
    dataset_name: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate high-level KPIs for a dataset.
    """

    numeric_columns = get_numeric_columns(df)

    kpis = {

        "dataset": dataset_name,

        "rows": len(df),

        "columns": len(df.columns),

        "numeric_columns": len(
            numeric_columns
        ),

        "categorical_columns": len(
            get_categorical_columns(df)
        ),

        "missing_values": int(
            df.isna().sum().sum()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "memory_usage_mb": round(

            df.memory_usage(
                deep=True
            ).sum()

            / 1024 ** 2,

            2,

        ),

        "total_numeric_sum": round(

            df[numeric_columns]
            .sum()
            .sum(),

            2,

        ) if numeric_columns else 0,

        "average_numeric_value": round(

            df[numeric_columns]
            .mean()
            .mean(),

            2,

        ) if numeric_columns else 0,

    }

    LOGGER.info(
        "KPIs generated for %s",
        dataset_name,
    )

    return kpis


###############################################################################
# Dataset Statistics
###############################################################################

def dataset_statistics(
    dataset_name: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate descriptive statistics.
    """

    numeric_columns = get_numeric_columns(df)

    statistics = {

        "dataset": dataset_name,

        "shape": df.shape,

        "rows": len(df),

        "columns": len(df.columns),

        "numeric_features": len(
            numeric_columns
        ),

        "categorical_features": len(
            get_categorical_columns(df)
        ),

        "missing_percentage": round(

            (

                df.isna()

                .sum()

                .sum()

                /

                (df.shape[0] * df.shape[1])

            )

            * 100,

            2,

        ),

        "duplicate_rows": int(

            df.duplicated().sum()

        ),

        "memory_usage_mb": round(

            df.memory_usage(

                deep=True

            ).sum()

            / 1024 ** 2,

            2,

        ),

    }

    LOGGER.info(
        "Statistics generated for %s",
        dataset_name,
    )

    return statistics


###############################################################################
# Numerical Summary
###############################################################################

def numerical_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate descriptive statistics
    for all numeric columns.
    """

    numeric_columns = get_numeric_columns(df)

    if not numeric_columns:

        return pd.DataFrame()

    return (

        df[numeric_columns]

        .describe()

        .transpose()

    )


###############################################################################
# Missing Value Summary
###############################################################################

def missing_value_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize missing values.
    """

    report = pd.DataFrame({

        "missing_values": df.isna().sum(),

        "percentage":

            round(

                df.isna().mean() * 100,

                2,

            ),

    })

    report = report.sort_values(

        by="missing_values",

        ascending=False,

    )

    return report


###############################################################################
# Dataset Overview
###############################################################################

def dataset_overview(
    dataset_name: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Combine all basic analytics into one object.
    """

    overview = {

        "kpis":

            calculate_kpis(

                dataset_name,

                df,

            ),

        "statistics":

            dataset_statistics(

                dataset_name,

                df,

            ),

        "numeric_summary":

            numerical_summary(df),

        "missing_summary":

            missing_value_summary(df),

    }

    LOGGER.info(
        "Overview generated for %s",
        dataset_name,
    )

    return overview


###############################################################################
# End of Part 1
###############################################################################

###############################################################################
# State Analysis
###############################################################################

def state_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if "state" not in df.columns:
        LOGGER.warning("'state' column not found.")
        return pd.DataFrame()

    summary = (
        df.groupby("state")
        .size()
        .reset_index(name="record_count")
    )

    total_records = summary["record_count"].sum()

    summary["contribution_percent"] = (
        summary["record_count"] / total_records * 100
    ).round(2)

    summary = summary.sort_values(
        by="record_count",
        ascending=False
    ).reset_index(drop=True)

    LOGGER.info("State analysis completed.")

    return summary

###############################################################################
# District Analysis
###############################################################################

def district_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if "district" not in df.columns:
        LOGGER.warning("'district' column not found.")
        return pd.DataFrame()

    summary = (
        df.groupby("district")
        .size()
        .reset_index(name="record_count")
    )

    summary = summary.sort_values(
        by="record_count",
        ascending=False
    ).reset_index(drop=True)

    LOGGER.info("District analysis completed.")

    return summary


###############################################################################
# State Rankings
###############################################################################

def state_rankings(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank states based on activity.
    """

    if state_df.empty:
        return state_df

    ranked = state_df.copy()

    ranked["state_rank"] = (
        ranked["record_count"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    max_records = ranked["record_count"].max()

    ranked["performance_score"] = round(
        ranked["record_count"] / max_records * 100,
        2,
    )

    ranked = ranked.rename(
        columns={
            "record_count": "state_record_count"
        }
    )

    ranked = ranked.sort_values(
        by="state_rank"
    ).reset_index(drop=True)

    LOGGER.info("State rankings generated.")

    return ranked


###############################################################################
# District Rankings
###############################################################################

def district_rankings(district_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank districts based on activity.
    """

    if district_df.empty:
        return district_df

    ranked = district_df.copy()

    ranked["district_rank"] = (
        ranked["record_count"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    max_records = ranked["record_count"].max()

    ranked["performance_score"] = round(
        ranked["record_count"] / max_records * 100,
        2,
    )

    ranked = ranked.rename(
        columns={
            "record_count": "district_record_count"
        }
    )

    ranked = ranked.sort_values(
        by="district_rank"
    ).reset_index(drop=True)

    LOGGER.info("District rankings generated.")

    return ranked


###############################################################################
# Top / Bottom Regions
###############################################################################

def top_states(state_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if state_df.empty:
        return state_df

    return (
        state_df
        .sort_values(
            by="state_record_count",
            ascending=False,
        )
        .head(n)
    )


###############################################################################

def bottom_states(state_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if state_df.empty:
        return state_df

    return (
        state_df
        .sort_values(
            by="state_record_count",
            ascending=True,
        )
        .head(n)
    )


###############################################################################

def top_districts(district_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if district_df.empty:
        return district_df

    return (
        district_df
        .sort_values(
            by="district_record_count",
            ascending=False,
        )
        .head(n)
    )


###############################################################################

def bottom_districts(district_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if district_df.empty:
        return district_df

    return (
        district_df
        .sort_values(
            by="district_record_count",
            ascending=True,
        )
        .head(n)
    )

###############################################################################
# Regional Analytics
###############################################################################

def regional_analysis(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Complete regional analytics.
    """

    states = state_analysis(df)

    districts = district_analysis(df)

    states = state_rankings(states)

    districts = district_rankings(districts)

    regional_results = {

        "state_summary": states,

        "district_summary": districts,

        "top_states": top_states(states),

        "bottom_states": bottom_states(states),

        "top_districts": top_districts(districts),

        "bottom_districts": bottom_districts(districts),

    }

    LOGGER.info(
        "Regional analytics completed."
    )

    return regional_results


###############################################################################
# End of Part 2
###############################################################################

###############################################################################
# Trend Analysis
###############################################################################

def trend_analysis(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Generate dashboard-ready trend datasets.
    Each exported dataframe contains only the
    columns required for visualization.
    """

    trends = {}

    # --------------------------------------------------
    # Yearly Trend
    # --------------------------------------------------

    IGNORE = [
        "ratio",
        "lag",
        "rolling",
        "ema",
        "ewm",
        "cumulative",
        "interaction",
        "normalized",
        "score",
        "activity"
    ]

    year_cols = [
        c for c in df.columns
        if "year" in c.lower()
        and not any(k in c.lower() for k in IGNORE)
    ]

    for col in year_cols:

        yearly = (
            df.groupby(col)
              .size()
              .reset_index(name="record_count")
              .sort_values(col)
        )

        trends[col] = yearly

    # --------------------------------------------------
    # Monthly Trend
    # --------------------------------------------------

    month_cols = [
        c for c in df.columns
        if "month" in c.lower()
        and not any(k in c.lower() for k in IGNORE)
    ]

    month_order = [
        "January","February","March","April",
        "May","June","July","August",
        "September","October","November","December"
    ]

    for col in month_cols:

        monthly = (
            df.groupby(col)
              .size()
              .reset_index(name="record_count")
        )

        if monthly[col].dtype == object:

            monthly[col] = pd.Categorical(
                monthly[col],
                categories=month_order,
                ordered=True,
            )

            monthly = monthly.sort_values(col)

        else:

            monthly = monthly.sort_values(col)

        trends[col] = monthly.reset_index(drop=True)

    # --------------------------------------------------
    # Quarterly Trend
    # --------------------------------------------------

    quarter_cols = [
        c for c in df.columns
        if "quarter" in c.lower()
        and not any(k in c.lower() for k in IGNORE)
    ]

    for col in quarter_cols:

        quarterly = (
            df.groupby(col)
              .size()
              .reset_index(name="record_count")
              .sort_values(col)
        )

        trends[col] = quarterly

    # --------------------------------------------------
    # Weekly Trend
    # --------------------------------------------------

    week_cols = [
        c for c in df.columns
        if "week" in c.lower()
        and not any(k in c.lower() for k in IGNORE)
    ]

    weekday_order = [
        "Monday","Tuesday","Wednesday",
        "Thursday","Friday","Saturday","Sunday"
    ]

    for col in week_cols:

        weekly = (
            df.groupby(col)
              .size()
              .reset_index(name="record_count")
        )

        if weekly[col].dtype == object:

            weekly[col] = pd.Categorical(
                weekly[col],
                categories=weekday_order,
                ordered=True,
            )

            weekly = weekly.sort_values(col)

        else:

            weekly = weekly.sort_values(col)

        trends[col] = weekly.reset_index(drop=True)

    LOGGER.info(
        "Dashboard-ready trend analysis completed."
    )

    return trends


###############################################################################
# Growth Analysis
###############################################################################

def growth_analysis(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Generate dashboard-ready growth datasets.
    """

    growth_results = {}

    IGNORE = [
        "ratio",
        "lag",
        "rolling",
        "ema",
        "ewm",
        "cumulative",
        "interaction",
        "normalized",
        "score",
        "activity"
    ]   
    # ======================================================
    # YEARLY GROWTH
    # ======================================================

    year_cols = [
        c for c in df.columns
        if "year" in c.lower()
        and not any(k in c.lower() for k in IGNORE)
    ]

    for col in year_cols:

        yearly = (
            df.groupby(col)
              .size()
              .reset_index(name="record_count")
              .sort_values(col)
        )

        yearly["growth_percent"] = (
            yearly["record_count"]
            .pct_change()
            .mul(100)
            .round(2)
        )

        growth_results[f"{col}_growth"] = yearly

    # ======================================================
    # MONTHLY GROWTH
    # ======================================================

    month_cols = [
        c for c in df.columns
        if "month" in c.lower()
        and not any(k in c.lower() for k in IGNORE)
    ]

    month_order = [
        "January","February","March","April",
        "May","June","July","August",
        "September","October","November","December"
    ]

    for col in month_cols:

        monthly = (
            df.groupby(col)
              .size()
              .reset_index(name="record_count")
        )

        if monthly[col].dtype == object:

            monthly[col] = pd.Categorical(
                monthly[col],
                categories=month_order,
                ordered=True
            )

        monthly = monthly.sort_values(col)

        monthly["growth_percent"] = (
            monthly["record_count"]
            .pct_change()
            .mul(100)
            .round(2)
        )

        growth_results[f"{col}_growth"] = (
            monthly.reset_index(drop=True)
        )

    # ======================================================
    # QUARTERLY GROWTH
    # ======================================================

    quarter_cols = [
        c for c in df.columns
        if "quarter" in c.lower()
        and not any(k in c.lower() for k in IGNORE)
    ]

    for col in quarter_cols:

        quarterly = (
            df.groupby(col)
              .size()
              .reset_index(name="record_count")
              .sort_values(col)
        )

        quarterly["growth_percent"] = (
            quarterly["record_count"]
            .pct_change()
            .mul(100)
            .round(2)
        )

        growth_results[f"{col}_growth"] = quarterly

    LOGGER.info(
        "Growth analysis completed."
    )

    return growth_results

###############################################################################
# Correlation Analysis
###############################################################################

def correlation_analysis(
    df: pd.DataFrame,
    max_features: int = 25,
) -> pd.DataFrame:
    """
    Generate dashboard-ready correlation matrix.
    """

    numeric_columns = get_numeric_columns(df)

    if len(numeric_columns) < 2:
        return pd.DataFrame()

    # ---------------------------------------------------
    # Remove engineered columns
    # ---------------------------------------------------

    ignore_keywords = [

        "lag",

        "rolling",

        "ema",

        "ewm",

        "cumulative",

        "normalized",

        "score",

        "activity",

        "interaction",

        "ratio",

    ]

    filtered_columns = [

        column

        for column in numeric_columns

        if not any(
            keyword in column.lower()
            for keyword in ignore_keywords
        )

    ]

    # ---------------------------------------------------
    # Keep first N useful features
    # ---------------------------------------------------

    if not filtered_columns:
        filtered_columns = numeric_columns.copy()

    variance = (
        df[filtered_columns]
        .var()
        .sort_values(ascending=False)
    )

    filtered_columns = list(
        variance.head(max_features).index
    )

    if len(filtered_columns) < 2:

        filtered_columns = numeric_columns[:max_features]

    correlation = (

        df[filtered_columns]

        .corr(method="pearson")

        .round(3)

    )

    LOGGER.info(
        "Dashboard correlation generated."
    )

    return correlation

###############################################################################
# Anomaly Detection
###############################################################################

def anomaly_detection(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect anomalies using Z-score and export
    only anomalous records.
    """

    numeric_columns = get_numeric_columns(df)

    if not numeric_columns:
        return pd.DataFrame()

    anomaly_mask = pd.Series(False, index=df.index)

    for column in numeric_columns:

        std = df[column].std()

        if std == 0 or pd.isna(std):
            continue

        z_score = (
            df[column] - df[column].mean()
        ) / std

        mask = z_score.abs() > z_threshold

        anomaly_mask |= mask

    if not anomaly_mask.any():

        LOGGER.info("No anomalies detected.")

        return pd.DataFrame()

    anomalies = df.loc[anomaly_mask].copy()

    anomalies["anomaly_score"] = 0

    for column in numeric_columns:

        std = df[column].std()

        if std == 0 or pd.isna(std):
            continue

        anomalies["anomaly_score"] += (
            (
                anomalies[column]
                - df[column].mean()
            ).abs() / std
        )

    anomalies["anomaly_score"] = (
        anomalies["anomaly_score"]
        .round(2)
    )

    anomalies = anomalies.sort_values(
        "anomaly_score",
        ascending=False,
    )

    LOGGER.info(
        "%d anomalies detected.",
        len(anomalies)
    )

    anomalies["severity"] = pd.cut(
        anomalies["anomaly_score"],
        bins=[0,5,10,20,np.inf],

        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ]
    )

    return anomalies.reset_index(drop=True)


###############################################################################
# Forecast Analysis
###############################################################################

def forecast_analysis(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Generate forecasting datasets for dashboard.

    NOTE:
    Replace prediction generation with your trained
    Random Forest model.
    """

    if "date_month" not in df.columns:
        return {}

    monthly = (
        df.groupby("date_month")
        .size()
        .reset_index(name="actual")
        .sort_values("date_month")
    )

    ###########################################################################
    # TODO:
    # Replace this block with your Random Forest predictions
    ###########################################################################

    monthly["prediction"] = monthly["actual"]

    ###########################################################################

    monthly["error"] = (
        monthly["actual"]
        - monthly["prediction"]
    )

    summary = pd.DataFrame([{

        "model": "Random Forest",

        "months": len(monthly),

        "actual_total": int(
            monthly["actual"].sum()
        ),

        "predicted_total": int(
            monthly["prediction"].sum()
        ),

        "average_prediction": round(
            monthly["prediction"].mean(),
            2
        )

    }])

    return {

        "forecast_month": monthly,

        "forecast_summary": summary,

    }


###############################################################################
# Dashboard Metrics
###############################################################################

def dashboard_metrics(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate dashboard-ready KPIs.
    """

    metrics = {

        "total_records": len(df),

        "total_columns": len(df.columns),

        "missing_values": int(

            df.isna().sum().sum()

        ),

        "duplicate_rows": int(

            df.duplicated().sum()

        ),

        "numeric_columns": len(

            get_numeric_columns(df)

        ),

        "categorical_columns": len(

            get_categorical_columns(df)

        ),

    }

    if "state" in df.columns:

        metrics["unique_states"] = (

            df["state"]

            .nunique()

        )

    if "district" in df.columns:

        metrics["unique_districts"] = (

            df["district"]

            .nunique()

        )

    LOGGER.info(
        "Dashboard metrics generated."
    )

    return metrics


###############################################################################
# Executive Summary
###############################################################################

def executive_summary(
    dataset_name: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate a concise executive summary.
    """

    summary = {

        "dataset": dataset_name,

        "records": len(df),

        "features": len(df.columns),

        "missing_values": int(

            df.isna().sum().sum()

        ),

        "duplicate_rows": int(

            df.duplicated().sum()

        ),

        "numeric_features": len(

            get_numeric_columns(df)

        ),

        "categorical_features": len(

            get_categorical_columns(df)

        ),

    }

    if "state" in df.columns:

        summary["states"] = (

            df["state"]

            .nunique()

        )

    if "district" in df.columns:

        summary["districts"] = (

            df["district"]

            .nunique()

        )

    LOGGER.info(
        "Executive summary generated for %s",
        dataset_name,
    )

    return summary


###############################################################################
# Complete Analytics for One Dataset
###############################################################################

def analytical_insights(
    dataset_name: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Execute all analytical functions
    for a single dataset.
    """

    insights = {

        "overview":

            dataset_overview(

                dataset_name,

                df,

            ),

        "regional":

            regional_analysis(df),

        "trends":

            trend_analysis(df),

        "growth":

            growth_analysis(df),

        "correlation":

            correlation_analysis(df),

        "anomalies":

            anomaly_detection(df),

        "forecast":

            forecast_analysis(df),

        "dashboard":

            dashboard_metrics(df),

        "summary":

            executive_summary(

                dataset_name,

                df,

            ),

    }

    LOGGER.info(
        "Analytical insights generated for %s",
        dataset_name,
    )

    return insights


###############################################################################
# End of Part 3
###############################################################################

###############################################################################
# Analytics Pipeline for All Datasets
###############################################################################

def analyze_all_datasets(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Execute analytics for every engineered dataset.
    """

    results = {}

    LOGGER.info("=" * 80)
    LOGGER.info("Starting Analytics Pipeline")
    LOGGER.info("=" * 80)

    for dataset_name, dataframe in datasets.items():

        try:

            LOGGER.info(
                "Analyzing %s",
                dataset_name,
            )

            results[dataset_name] = analytical_insights(
                dataset_name,
                dataframe,
            )

        except Exception as error:

            LOGGER.exception(
                "Analytics failed for %s",
                dataset_name,
            )

            raise error

    LOGGER.info("=" * 80)
    LOGGER.info("Analytics Pipeline Completed")
    LOGGER.info("=" * 80)

    return results


###############################################################################
# Export Utilities
###############################################################################

def export_dataframe(
    dataframe,
    output_path,
):

    if dataframe.empty:

        LOGGER.warning(
            "%s skipped (empty)",
            output_path.name
        )

        return False

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    LOGGER.info(
        "%s exported (%d rows)",
        output_path.name,
        len(dataframe),
    )

    return True

###############################################################################
# Manifest
###############################################################################

import json
from datetime import datetime


def generate_manifest():

    return {

        "project": "Aadhaar Insights",

        "version": "2.0",

        "generated_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "datasets": {},

        "statistics": {

            "datasets": 0,

            "files": 0,

            "rows": 0,

        },

    }

###############################################################################
# Manifest Helper
###############################################################################

def register_export(
    manifest: Dict[str, Any],
    dataset_name: str,
    export_name: str,
    dataframe: pd.DataFrame,
    output_path: Path,
):
    """
    Register an exported dataframe in the manifest.
    """

    manifest.setdefault("datasets", {})
    manifest["datasets"].setdefault(dataset_name, {})

    manifest["datasets"][dataset_name][export_name] = {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "file": output_path.name,
    }

    manifest["statistics"]["files"] += 1
    manifest["statistics"]["rows"] += len(dataframe)

###############################################################################

def export_anomaly_dashboard_files(
    anomalies: pd.DataFrame,
    dataset_dir: Path,
):
    """
    Export lightweight anomaly files for the Streamlit dashboard.
    """

    if anomalies.empty:
        return

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    summary = pd.DataFrame([{
        "total_anomalies": len(anomalies),
        "critical": (anomalies["severity"] == "Critical").sum(),
        "high": (anomalies["severity"] == "High").sum(),
        "medium": (anomalies["severity"] == "Medium").sum(),
        "low": (anomalies["severity"] == "Low").sum(),
        "max_score": anomalies["anomaly_score"].max(),
        "avg_score": round(
            anomalies["anomaly_score"].mean(),
            2
        )
    }])

    export_dataframe(
        summary,
        dataset_dir / "anomalies_summary.csv"
    )

    # ----------------------------------------------------
    # Severity Distribution
    # ----------------------------------------------------

    severity = (
        anomalies["severity"]
        .value_counts()
        .rename_axis("severity")
        .reset_index(name="count")
    )

    export_dataframe(
        severity,
        dataset_dir / "anomalies_severity.csv"
    )

    # ----------------------------------------------------
    # Top 100 anomalies
    # ----------------------------------------------------

    export_dataframe(
        anomalies.head(100),
        dataset_dir / "anomalies_top100.csv"
    )



def export_analysis(
    analytics_results: Dict[str, Any],
):
    """
    Export all analytics outputs and generate manifest.
    """

    output_dir = Path(PROCESSED_DATA_DIR) / "analytics"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = generate_manifest()

    for dataset_name, analysis in analytics_results.items():

        dataset_dir = output_dir / dataset_name.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)

        #######################################################################
        # Regional Analysis
        #######################################################################

        for name, dataframe in analysis["regional"].items():

            if not isinstance(dataframe, pd.DataFrame):
                continue

            output_path = dataset_dir / f"{name}.csv"

            if export_dataframe(dataframe, output_path):
                register_export(
                    manifest,
                    dataset_name,
                    name,
                    dataframe,
                    output_path,
                )

        #######################################################################
        # Trend Analysis
        #######################################################################

        for name, dataframe in analysis["trends"].items():

            output_path = dataset_dir / f"trend_{name}.csv"

            if export_dataframe(dataframe, output_path):
                register_export(
                    manifest,
                    dataset_name,
                    f"trend_{name}",
                    dataframe,
                    output_path,
                )

        #######################################################################
        # Growth Analysis
        #######################################################################

        for name, dataframe in analysis["growth"].items():

            output_path = dataset_dir / f"{name}.csv"

            if export_dataframe(dataframe, output_path):
                register_export(
                    manifest,
                    dataset_name,
                    name,
                    dataframe,
                    output_path,
                )

        #######################################################################
        # Correlation
        #######################################################################

        output_path = dataset_dir / "correlation.csv"

        analysis["correlation"].to_csv(
            output_path,
            index=True,
        )
        
        register_export(
            manifest,
            dataset_name,
            "correlation",
            analysis["correlation"],
            output_path,
        )


        ###########################################################################
        # Forecast
        ###########################################################################

        for name, dataframe in analysis["forecast"].items():
        
            output_path = dataset_dir / f"{name}.csv"

            if export_dataframe(
                dataframe,
                output_path,
            ):
                register_export(
                    manifest,
                    dataset_name,
                    name,
                    dataframe,
                    output_path,
                )

        ###########################################################################
        # Anomalies
        ###########################################################################
        
        output_path = dataset_dir / "anomalies.csv"
        
        if export_dataframe(
            analysis["anomalies"],
            output_path,
        ):
            register_export(
                manifest,
                dataset_name,
                "anomalies",
                analysis["anomalies"],
                output_path,
            )
        
            # NEW
            export_anomaly_dashboard_files(
                analysis["anomalies"],
                dataset_dir,
            )

    ###########################################################################
    # Save Manifest
    ###########################################################################

    manifest["statistics"]["datasets"] = len(
            analytics_results
        )
    
    manifest_file = output_dir / "analytics_manifest.json"

    with open(
        manifest_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    LOGGER.info("Analytics exported successfully.")
    LOGGER.info("Analytics manifest exported.")


###############################################################################
# Executive Report
###############################################################################

def generate_summary_report(
    analytics_results: Dict[str, Any],
) -> pd.DataFrame:
    """
    Create overall summary report.
    """

    report = []

    for dataset_name, analysis in analytics_results.items():

        summary = analysis["summary"]

        report.append(summary)

    return pd.DataFrame(report)


###############################################################################

def save_summary_report(
    report: pd.DataFrame,
):
    """
    Save analytics summary report.
    """

    report_file = (

        Path(PROCESSED_DATA_DIR)

        / "analytics"

        / "analytics_summary.csv"

    )

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(

        report_file,

        index=False,

    )

    LOGGER.info(
        "Analytics summary saved."
    )

###############################################################################
# Dashboard Export Utilities
###############################################################################

def export_dashboard_metrics(
    analytics_results: Dict[str, Any],
):
    """
    Export dashboard KPIs for Power BI.
    """

    rows = []

    for dataset_name, analysis in analytics_results.items():

        metrics = analysis["dashboard"].copy()
        metrics["dataset"] = dataset_name

        rows.append(metrics)

    dashboard_metrics = pd.DataFrame(rows)

    output_file = (
        Path(PROCESSED_DATA_DIR)
        / "analytics"
        / "dashboard_metrics.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dashboard_metrics.to_csv(
        output_file,
        index=False,
    )

    LOGGER.info(
        "Dashboard metrics exported."
    )

def export_executive_dashboard(
    summary_report: pd.DataFrame,
):
    """
    Export executive dashboard table.
    """

    output_file = (
        Path(PROCESSED_DATA_DIR)
        / "analytics"
        / "executive_dashboard.csv"
    )

    summary_report.to_csv(
        output_file,
        index=False,
    )

    LOGGER.info(
        "Executive dashboard exported."
    )

import json
from datetime import datetime

def export_dashboard_config():

    config = {

        "project": "Aadhaar Insights Analytics Platform",

        "generated_on":
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "forecast_available": True,

        "analytics_available": True,

        "dashboard_version": "1.0",

    }

    output_file = (
        Path(PROCESSED_DATA_DIR)
        / "analytics"
        / "dashboard_config.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    LOGGER.info(
        "Dashboard configuration exported."
    )



###############################################################################
# Console Summary
###############################################################################

def print_summary(
    report: pd.DataFrame,
):
    """
    Display analytics summary.
    """

    print()

    print("=" * 80)

    print("AADHAAR INSIGHTS ANALYTICS SUMMARY")

    print("=" * 80)

    print(report)

    print("=" * 80)

    print()


###############################################################################
# Main
###############################################################################

def main():
    """
    Execute complete analytics workflow.
    """

    LOGGER.info("=" * 80)
    LOGGER.info("AADHAAR INSIGHTS ANALYTICS")
    LOGGER.info("=" * 80)

    try:

        #######################################################################
        # Load Raw Data
        #######################################################################

        raw_datasets = load_all_datasets()

        #######################################################################
        # Preprocessing
        #######################################################################

        processed_datasets, _ = (

            preprocess_all_datasets(

                raw_datasets

            )

        )

        #######################################################################
        # Feature Engineering
        #######################################################################

        engineered_datasets, _ = (

            feature_engineer_all_datasets(

                processed_datasets

            )

        )

        #######################################################################
        # Analytics
        #######################################################################

        analytics_results = (

            analyze_all_datasets(

                engineered_datasets

            )

        )

        #######################################################################
        # Export Results
        #######################################################################

        export_analysis(

            analytics_results

        )

        #######################################################################
        # Save Summary
        #######################################################################

        summary_report = (

            generate_summary_report(

                analytics_results

            )

        )

        save_summary_report(

            summary_report

        )

        export_dashboard_metrics(
            analytics_results
        )

        export_executive_dashboard(
            summary_report
        )

        export_dashboard_config()

        #######################################################################
        # Display Summary
        #######################################################################

        print_summary(

            summary_report

        )

        LOGGER.info(
            "Analytics completed successfully."
        )

    except Exception as error:

        LOGGER.exception(
            "Analytics pipeline failed."
        )

        raise error

###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()