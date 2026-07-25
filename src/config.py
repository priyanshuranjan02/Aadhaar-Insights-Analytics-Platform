"""
===============================================================================
Project      : Aadhaar Insights
File         : config.py
Author       : Priyanshu Ranjan
Description  : Central configuration file for the Aadhaar Insights project.
               This file contains project metadata, directory paths,
               dataset configuration, visualization settings,
               machine learning settings, logging configuration,
               and global constants used throughout the project.
===============================================================================
"""

from pathlib import Path
import logging

# =============================================================================
# PROJECT INFORMATION
# =============================================================================

PROJECT_NAME = "Aadhaar Insights"
PROJECT_VERSION = "1.0.0"

AUTHOR = "Priyanshu Ranjan"

DESCRIPTION = (
    "End-to-End Analytics Platform for Aadhaar Enrolment, "
    "Demographic Updates, Biometric Updates and Demand Forecasting."
)

# =============================================================================
# PROJECT ROOT
# =============================================================================

# src/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# DIRECTORY PATHS
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
MODEL_ARTIFACTS_DIR = MODELS_DIR / "artifacts"

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
POWERBI_DIR = DASHBOARD_DIR / "powerbi"

APP_DIR = PROJECT_ROOT / "app"

IMAGES_DIR = PROJECT_ROOT / "images"

DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"

TESTS_DIR = PROJECT_ROOT / "tests"

# =============================================================================
# RAW DATASET DIRECTORIES
# =============================================================================

ENROLMENT_DATA_DIR = RAW_DATA_DIR / "enrolment"
DEMOGRAPHIC_DATA_DIR = RAW_DATA_DIR / "demographic"
BIOMETRIC_DATA_DIR = RAW_DATA_DIR / "biometric"

DATASETS = {
    "enrolment": ENROLMENT_DATA_DIR,
    "demographic": DEMOGRAPHIC_DATA_DIR,
    "biometric": BIOMETRIC_DATA_DIR,
}

# =============================================================================
# PROCESSED DATA FILES
# =============================================================================

ENROLMENT_PROCESSED = PROCESSED_DATA_DIR / "enrolment_clean.csv"

DEMOGRAPHIC_PROCESSED = PROCESSED_DATA_DIR / "demographic_clean.csv"

BIOMETRIC_PROCESSED = PROCESSED_DATA_DIR / "biometric_clean.csv"

ANALYTICS_SUMMARY = PROCESSED_DATA_DIR / "analytics_summary.csv"

# =============================================================================
# FILE FORMATS
# =============================================================================

SUPPORTED_FILE_TYPES = [
    ".csv",
    ".xlsx",
    ".xls",
]

CSV_EXTENSION = ".csv"

# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

RANDOM_SEED = 42

TEST_SIZE = 0.20

FORECAST_HORIZON = 12

CROSS_VALIDATION_FOLDS = 5

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

FIGURE_SIZE = (12, 6)

DPI = 120

STYLE = "default"

COLOR_PALETTE = "viridis"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = logging.INFO

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# =============================================================================
# DATE SETTINGS
# =============================================================================

DATE_FORMAT = "%Y-%m-%d"

# =============================================================================
# COLUMN STANDARDIZATION
# =============================================================================

COLUMN_MAPPINGS = {
    "State": "state",
    "STATE": "state",
    "state": "state",

    "District": "district",
    "DISTRICT": "district",
    "district": "district",

    "Date": "date",
    "DATE": "date",
    "date": "date",

    "Month": "month",
    "MONTH": "month",

    "Year": "year",
    "YEAR": "year",

    "Age": "age",
    "AGE": "age",
}

# =============================================================================
# OUTPUT FILES
# =============================================================================

FORECAST_OUTPUT = GENERATED_REPORTS_DIR / "forecast_results.csv"

MODEL_METRICS = MODEL_ARTIFACTS_DIR / "model_metrics.json"

FEATURE_COLUMNS = MODEL_ARTIFACTS_DIR / "feature_columns.json"

# =============================================================================
# CREATE DIRECTORIES IF THEY DO NOT EXIST
# =============================================================================

DIRECTORIES = [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    GENERATED_REPORTS_DIR,
    MODELS_DIR,
    TRAINED_MODELS_DIR,
    MODEL_ARTIFACTS_DIR,
    DASHBOARD_DIR,
    POWERBI_DIR,
    APP_DIR,
    IMAGES_DIR,
    DOCUMENTATION_DIR,
    TESTS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOGGING INITIALIZATION
# =============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
)

LOGGER = logging.getLogger(PROJECT_NAME)

# =============================================================================
# END OF CONFIG
# =============================================================================