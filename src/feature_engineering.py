"""
===============================================================================
Project      : Aadhaar Insights
File         : feature_engineering.py
Author       : Priyanshu Ranjan

Description:
    Feature Engineering Pipeline for UIDAI datasets.

Responsibilities
----------------
1. Time Features
2. Age Features
3. Regional Features
4. Ratio Features
5. Growth Features
6. Ranking Features
7. ML Features
8. Dashboard Features
9. Save Engineered Datasets
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import LOGGER, PROCESSED_DATA_DIR
from src.preprocessing import preprocess_all_datasets
from src.data_loader import load_all_datasets


###############################################################################
# Helper Functions
###############################################################################

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate calendar-based features.
    """

    date_columns = [
        col for col in df.columns
        if "date" in col.lower()
        or "time" in col.lower()
    ]

    for col in date_columns:

        try:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
            )

            prefix = col.replace("_date", "")

            df[f"{prefix}_year"] = df[col].dt.year

            df[f"{prefix}_month"] = df[col].dt.month

            df[f"{prefix}_month_name"] = (
                df[col].dt.month_name()
            )

            df[f"{prefix}_quarter"] = (
                df[col].dt.quarter
            )

            df[f"{prefix}_week"] = (
                df[col].dt.isocalendar().week
            )

            df[f"{prefix}_day"] = (
                df[col].dt.day
            )

            df[f"{prefix}_day_name"] = (
                df[col].dt.day_name()
            )

            df[f"{prefix}_is_weekend"] = (
                df[col].dt.weekday >= 5
            )

        except Exception:

            LOGGER.warning(
                "Unable to create time features for %s",
                col,
            )

    LOGGER.info("Time features created.")

    return df


###############################################################################

def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age groups.
    """

    if "age" not in df.columns:

        return df

    bins = [0, 5, 18, 30, 45, 60, 120]

    labels = [
        "0-5",
        "6-18",
        "19-30",
        "31-45",
        "46-60",
        "60+",
    ]

    df["age_group"] = pd.cut(

        df["age"],

        bins=bins,

        labels=labels,

        include_lowest=True,

    )

    LOGGER.info("Age groups created.")

    return df


###############################################################################

def create_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create state and district statistics.
    """

    if "state" in df.columns:

        df["state_record_count"] = (
            df.groupby("state")["state"]
            .transform("count")
        )

        df["state_rank"] = (
            df["state_record_count"]
            .rank(
                ascending=False,
                method="dense",
            )
        )

    if "district" in df.columns:

        df["district_record_count"] = (
            df.groupby("district")["district"]
            .transform("count")
        )

    LOGGER.info(
        "Regional features created."
    )

    return df


###############################################################################

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create percentage and ratio features.
    """

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) == 0:

        return df

    total = df[numeric_columns].sum(axis=1)

    total = total.replace(0, np.nan)

    for column in numeric_columns:

        df[f"{column}_ratio"] = (

            df[column] / total

        )

    LOGGER.info(
        "Ratio features created."
    )

    return df


###############################################################################

def create_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cumulative and rolling features.
    """

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        df[f"{column}_cumulative"] = (

            df[column].cumsum()

        )

        df[f"{column}_rolling_mean"] = (

            df[column]

            .rolling(
                window=3,
                min_periods=1,
            )

            .mean()

        )

        df[f"{column}_rolling_std"] = (

            df[column]

            .rolling(
                window=3,
                min_periods=1,
            )

            .std()

        )

    LOGGER.info(
        "Growth features created."
    )

    return df


###############################################################################
# End of Part 1
###############################################################################

###############################################################################
# Ranking Features
###############################################################################

def create_ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ranking-based analytical features.
    """

    if "state_record_count" in df.columns:

        df["state_rank"] = (

            df["state_record_count"]

            .rank(

                method="dense",

                ascending=False,

            )

            .astype(int)

        )

    if "district_record_count" in df.columns:

        df["district_rank"] = (

            df["district_record_count"]

            .rank(

                method="dense",

                ascending=False,

            )

            .astype(int)

        )

    LOGGER.info(
        "Ranking features created."
    )

    return df


###############################################################################
# Machine Learning Features
###############################################################################

def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate lag and moving-average features
    for forecasting models.
    """

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        df[f"{column}_lag_1"] = (
            df[column].shift(1)
        )

        df[f"{column}_lag_3"] = (
            df[column].shift(3)
        )

        df[f"{column}_lag_6"] = (
            df[column].shift(6)
        )

        df[f"{column}_lag_12"] = (
            df[column].shift(12)
        )

        df[f"{column}_ema"] = (

            df[column]

            .ewm(
                span=5,
                adjust=False,
            )

            .mean()

        )

    LOGGER.info(
        "Machine Learning features created."
    )

    return df


###############################################################################
# Dashboard Features
###############################################################################

def create_dashboard_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate KPIs for dashboards.
    """

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) == 0:

        return df

    df["total_activity"] = (

        df[numeric_columns]

        .sum(axis=1)

    )

    df["average_activity"] = (

        df[numeric_columns]

        .mean(axis=1)

    )

    df["maximum_activity"] = (

        df[numeric_columns]

        .max(axis=1)

    )

    df["minimum_activity"] = (

        df[numeric_columns]

        .min(axis=1)

    )

    df["activity_score"] = (

        df["average_activity"]

        /

        df["maximum_activity"]

        .replace(0, np.nan)

    )

    LOGGER.info(
        "Dashboard features created."
    )

    return df


###############################################################################
# Dataset Statistics
###############################################################################

def generate_feature_statistics(
    df: pd.DataFrame
) -> Dict:
    """
    Generate statistics after feature engineering.
    """

    return {

        "rows": len(df),

        "columns": len(df.columns),

        "numeric_columns": len(

            df.select_dtypes(
                include=np.number
            ).columns

        ),

        "categorical_columns": len(

            df.select_dtypes(
                exclude=np.number
            ).columns

        ),

        "missing_values": int(

            df.isna().sum().sum()

        ),

        "memory_usage_mb": round(

            df.memory_usage(
                deep=True
            ).sum()

            /

            1024**2,

            2,

        ),

    }


###############################################################################
# Feature Engineering Pipeline
###############################################################################

def feature_engineering_pipeline(
    dataset_name: str,
    df: pd.DataFrame,
):
    """
    Complete feature engineering pipeline.
    """

    LOGGER.info(
        "Starting feature engineering for %s",
        dataset_name,
    )

    df = create_time_features(df)

    df = create_age_features(df)

    df = create_region_features(df)

    df = create_ratio_features(df)

    df = create_growth_features(df)

    df = create_ranking_features(df)

    df = create_ml_features(df)

    df = create_dashboard_features(df)

    statistics = generate_feature_statistics(df)

    LOGGER.info(
        "Completed feature engineering for %s",
        dataset_name,
    )

    return df, statistics


###############################################################################
# End of Part 2
###############################################################################

###############################################################################
# Feature Engineering for All Datasets
###############################################################################

def feature_engineer_all_datasets(
    datasets: Dict[str, pd.DataFrame]
):
    """
    Apply feature engineering to all datasets.
    """

    engineered_datasets = {}

    statistics = {}

    LOGGER.info("=" * 80)
    LOGGER.info("Starting Feature Engineering Pipeline")
    LOGGER.info("=" * 80)

    for dataset_name, dataframe in datasets.items():

        try:

            engineered_df, stats = feature_engineering_pipeline(
                dataset_name,
                dataframe,
            )

            engineered_datasets[
                dataset_name
            ] = engineered_df

            statistics[
                dataset_name
            ] = stats

            LOGGER.info(
                "Completed %s",
                dataset_name,
            )

        except Exception as error:

            LOGGER.exception(
                "Feature engineering failed for %s",
                dataset_name,
            )

            raise error

    LOGGER.info("=" * 80)
    LOGGER.info("Feature Engineering Finished")
    LOGGER.info("=" * 80)

    return engineered_datasets, statistics


###############################################################################
# Save Engineered Datasets
###############################################################################

def save_engineered_datasets(
    datasets: Dict[str, pd.DataFrame]
):
    """
    Save engineered datasets.
    """

    output_dir = Path(PROCESSED_DATA_DIR) / "engineered"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset_name, dataframe in datasets.items():

        output_file = (
            output_dir /
            f"{dataset_name.lower()}_engineered.csv"
        )

        dataframe.to_csv(
            output_file,
            index=False,
        )

        LOGGER.info(
            "Saved %s",
            output_file.name,
        )


###############################################################################
# Generate Feature Report
###############################################################################

def generate_feature_report(
    statistics: Dict
) -> pd.DataFrame:
    """
    Create feature engineering report.
    """

    report = pd.DataFrame.from_dict(

        statistics,

        orient="index",

    )

    report.index.name = "dataset"

    return report


###############################################################################
# Save Feature Report
###############################################################################

def save_feature_report(
    report: pd.DataFrame
):
    """
    Save feature report.
    """

    output_dir = Path(PROCESSED_DATA_DIR)

    report_file = (
        output_dir /
        "feature_engineering_report.csv"
    )

    report.to_csv(
        report_file
    )

    LOGGER.info(
        "Feature report saved."
    )


###############################################################################
# Summary
###############################################################################

def print_summary(
    report: pd.DataFrame
):
    """
    Print pipeline summary.
    """

    print()

    print("=" * 80)

    print(
        "FEATURE ENGINEERING SUMMARY"
    )

    print("=" * 80)

    print(report)

    print("=" * 80)

    print()


###############################################################################
# Main
###############################################################################

def main():
    """
    Execute complete feature engineering pipeline.
    """

    LOGGER.info("=" * 80)
    LOGGER.info("AADHAAR INSIGHTS FEATURE ENGINEERING")
    LOGGER.info("=" * 80)

    try:

        #######################################################################
        # Load raw datasets
        #######################################################################

        raw_datasets = load_all_datasets()

        #######################################################################
        # Preprocess
        #######################################################################

        processed_datasets, _ = preprocess_all_datasets(
            raw_datasets
        )

        #######################################################################
        # Feature Engineering
        #######################################################################

        engineered_datasets, statistics = (
            feature_engineer_all_datasets(
                processed_datasets
            )
        )

        #######################################################################
        # Save datasets
        #######################################################################

        save_engineered_datasets(
            engineered_datasets
        )

        #######################################################################
        # Save report
        #######################################################################

        report = generate_feature_report(
            statistics
        )

        save_feature_report(
            report
        )

        #######################################################################
        # Display summary
        #######################################################################

        print_summary(
            report
        )

        LOGGER.info(
            "Feature Engineering Completed Successfully."
        )

    except Exception as error:

        LOGGER.exception(
            "Pipeline Failed."
        )

        raise error


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()