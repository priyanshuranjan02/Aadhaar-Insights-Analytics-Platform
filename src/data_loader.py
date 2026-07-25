"""
===============================================================================
Project      : Aadhaar Insights
File         : data_loader.py
Author       : Priyanshu Ranjan

Description:
    Generic data loading module for the Aadhaar Insights project.

Responsibilities:
    - Load raw datasets
    - Validate dataset folders
    - Read multiple CSV/Excel files
    - Merge datasets
    - Return pandas DataFrames
===============================================================================
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import (
    DATASETS,
    SUPPORTED_FILE_TYPES,
    LOGGER,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_dataset_files(dataset_dir: Path) -> List[Path]:
    """
    Return all supported dataset files inside a directory
    and all of its subdirectories.
    """

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{dataset_dir}"
        )

    files = []

    for extension in SUPPORTED_FILE_TYPES:
        files.extend(dataset_dir.rglob(f"*{extension}"))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"No supported dataset files found in:\n{dataset_dir}"
        )

    return files


def _read_file(file_path: Path) -> pd.DataFrame:
    """
    Read a CSV or Excel file.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    raise ValueError(
        f"Unsupported file format: {suffix}"
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Perform basic validation.
    """

    if df.empty:
        raise ValueError(
            f"{dataset_name} dataset is empty."
        )

    if df.shape[1] == 0:
        raise ValueError(
            f"{dataset_name} dataset has no columns."
        )


# =============================================================================
# GENERIC DATA LOADER
# =============================================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load all files belonging to a dataset and merge them.

    Parameters
    ----------
    dataset_name : str
        enrolment | demographic | biometric

    Returns
    -------
    pd.DataFrame
    """

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    dataset_dir = DATASETS[dataset_name]

    LOGGER.info(f"Loading {dataset_name} dataset...")

    files = _get_dataset_files(dataset_dir)

    dataframes = []

    for file in files:

        LOGGER.info(f"Reading {file.name}")

        df = _read_file(file)

        dataframes.append(df)

    merged_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    validate_dataset(
        merged_df,
        dataset_name
    )

    LOGGER.info(
        f"{dataset_name} loaded successfully "
        f"({merged_df.shape[0]:,} rows)"
    )

    return merged_df


# =============================================================================
# SPECIFIC LOADERS
# =============================================================================

def load_enrolment_data() -> pd.DataFrame:
    """
    Load Aadhaar Enrolment dataset.
    """

    return load_dataset("enrolment")


def load_demographic_data() -> pd.DataFrame:
    """
    Load Aadhaar Demographic dataset.
    """

    return load_dataset("demographic")


def load_biometric_data() -> pd.DataFrame:
    """
    Load Aadhaar Biometric dataset.
    """

    return load_dataset("biometric")


# =============================================================================
# LOAD ALL DATASETS
# =============================================================================

def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load every dataset.

    Returns
    -------
    dict
    """

    return {
        "enrolment": load_enrolment_data(),
        "demographic": load_demographic_data(),
        "biometric": load_biometric_data(),
    }


# =============================================================================
# DATASET INFORMATION
# =============================================================================

def get_dataset_info(df: pd.DataFrame) -> Dict:
    """
    Return summary statistics for a dataset.
    """

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(memory_mb, 2),
    }


# =============================================================================
# STREAMLIT DASHBOARD LOADERS
# =============================================================================

from pathlib import Path

# Root directory of project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYTICS_DIR = PROJECT_ROOT / "data" / "processed" / "analytics"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _analytics_path(dataset: str, filename: str) -> Path:
    """
    Returns path to an analytics CSV.

    Example
    -------
    analytics/enrolment/top_states.csv
    """

    return ANALYTICS_DIR / dataset / filename


def load_dashboard_metrics() -> pd.DataFrame:
    """
    Load dashboard metrics.
    """

    return pd.read_csv(
        ANALYTICS_DIR / "dashboard_metrics.csv"
    )


def load_executive_dashboard() -> pd.DataFrame:
    """
    Load executive dashboard summary.
    """

    return pd.read_csv(
        ANALYTICS_DIR / "executive_dashboard.csv"
    )


# =============================================================================
# ANALYTICS LOADERS
# =============================================================================

def load_top_states(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        _analytics_path(dataset, "top_states.csv")
    )


def load_bottom_states(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        _analytics_path(dataset, "bottom_states.csv")
    )


def load_top_districts(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        _analytics_path(dataset, "top_districts.csv")
    )


def load_bottom_districts(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        _analytics_path(dataset, "bottom_districts.csv")
    )


def load_state_summary(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        _analytics_path(dataset, "state_summary.csv")
    )


def load_district_summary(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        _analytics_path(dataset, "district_summary.csv")
    )


# def load_growth(dataset: str) -> pd.DataFrame:
#     return pd.read_csv(
#         _analytics_path(dataset, "growth.csv")
#     )


def load_correlation(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        _analytics_path(dataset, "correlation.csv"),
        index_col=0,
    )


def load_anomalies(dataset):
    return pd.read_csv(
        _analytics_path(dataset, "anomalies_top100.csv")
    )


def load_yearly_trend(dataset: str) -> pd.DataFrame:
    """
    Year-wise trend.
    """
    return pd.read_csv(
        _analytics_path(dataset, "trend_date_year.csv")
    )


def load_monthly_trend(dataset: str) -> pd.DataFrame:
    """
    Month-wise trend.
    """

    return pd.read_csv(
        _analytics_path(dataset, "trend_date_month_name.csv")
    )


def load_quarterly_trend(dataset: str) -> pd.DataFrame:
    """
    Quarter-wise trend.
    """
    return pd.read_csv(
        _analytics_path(dataset, "trend_date_quarter.csv")
    )


def load_weekly_trend(dataset: str) -> pd.DataFrame:
    """
    Week-wise trend.
    """
    return pd.read_csv(
        _analytics_path(dataset, "trend_date_week.csv")
    )

def load_yearly_growth(dataset: str) -> pd.DataFrame:
    """
    Year-wise growth.
    """
    return pd.read_csv(
        _analytics_path(dataset, "date_year_growth.csv")
    )


def load_monthly_growth(dataset: str) -> pd.DataFrame:
    """
    Month-wise growth.
    """
    return pd.read_csv(
        _analytics_path(dataset, "date_month_name_growth.csv")
    )


def load_quarterly_growth(dataset: str) -> pd.DataFrame:
    """
    Quarter-wise growth.
    """
    return pd.read_csv(
        _analytics_path(dataset, "date_quarter_growth.csv")
    )


def load_weekend_trend(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        _analytics_path(dataset, "trend_date_is_weekend.csv")
    )

# =============================================================================
# RECOMMENDATION PAGE LOADERS
# =============================================================================

def load_analysis(dataset: str):
    """
    Load analytics required for Recommendations page.
    """

    return {
        "top_states": load_top_states(dataset),
        "bottom_states": load_bottom_states(dataset),
        "state_summary": load_state_summary(dataset),
        "district_summary": load_district_summary(dataset),
        "correlation": load_correlation(dataset),
        "anomalies": load_anomalies(dataset),
        "yearly_trend": load_yearly_trend(dataset),
        "yearly_growth": load_yearly_growth(dataset),
    }


def load_forecasting_results(dataset: str):
    """
    Load forecasting outputs for Recommendations page.
    """

    analytics_dir = ANALYTICS_DIR / dataset

    return {
        "metrics": pd.read_csv(
            analytics_dir / "model_metrics.csv"
        ),
        "feature_importance": pd.read_csv(
            analytics_dir / "feature_importance.csv"
        ),
        "forecast_summary": pd.read_csv(
            analytics_dir / "forecast_summary.csv"
        ),
        "forecast_month": pd.read_csv(
            analytics_dir / "forecast_month.csv"
        ),
        "actual_vs_predicted": pd.read_csv(
            analytics_dir / "actual_vs_predicted.csv"
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    datasets = load_all_datasets()

    print("\n")

    for name, dataframe in datasets.items():

        print("=" * 60)
        print(name.upper())

        info = get_dataset_info(dataframe)

        for key, value in info.items():
            print(f"{key:20}: {value}")

        print("=" * 60)