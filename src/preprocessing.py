"""
===============================================================================
Project      : Aadhaar Insights
File         : preprocessing.py
Author       : Priyanshu Ranjan

Description:
    End-to-End preprocessing pipeline for UIDAI datasets.

Responsibilities
----------------
1. Standardize column names
2. Clean text columns
3. Convert datatypes
4. Parse dates
5. Handle missing values
6. Remove duplicates
7. Dataset validation
8. Dataset specific preprocessing
9. Save processed datasets
10. Generate preprocessing statistics

===============================================================================
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    LOGGER,
    PROCESSED_DATA_DIR,
)

from src.data_loader import (
    load_all_datasets,
)

###############################################################################
# Helper Functions
###############################################################################

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe column names.

    Example
    -------
    'State Name' -> 'state_name'
    'District(Name)' -> 'district_name'
    """

    columns = []

    for col in df.columns:

        col = str(col)

        col = col.strip()

        col = col.lower()

        col = re.sub(r"[^\w\s]", "", col)

        col = re.sub(r"\s+", "_", col)

        col = re.sub(r"_+", "_", col)

        columns.append(col)

    df.columns = columns

    LOGGER.info("Column names standardized.")

    return df


###############################################################################

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all string columns.
    """

    object_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in object_columns:

        df[col] = (

            df[col]

            .astype("string")

            .str.strip()

            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )

            .replace(
                {
                    "": pd.NA,
                    "NULL": pd.NA,
                    "null": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                }
            )

        )

    LOGGER.info(
        "Cleaned %d text columns.",
        len(object_columns),
    )

    return df


###############################################################################

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns into numeric wherever possible.
    """

    converted = 0

    for col in df.columns:

        if df[col].dtype != "object":

            continue

        try:

            converted_series = pd.to_numeric(
                df[col]
            )

            df[col] = converted_series

            converted += 1

        except Exception:

            continue

    LOGGER.info(
        "Converted %d columns to numeric.",
        converted,
    )

    return df


###############################################################################

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically parse date columns.
    """

    keywords = [

        "date",

        "time",

        "month",

        "year",

    ]

    parsed = 0

    for col in df.columns:

        if any(

            keyword in col.lower()

            for keyword in keywords

        ):

            try:

                df[col] = pd.to_datetime(

                    df[col],

                    errors="coerce",

                )

                parsed += 1

            except Exception:

                LOGGER.warning(

                    "Unable to parse %s",

                    col,

                )

    LOGGER.info(

        "Parsed %d date columns.",

        parsed,

    )

    return df


###############################################################################

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows.
    """

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    LOGGER.info(

        "Removed %d duplicate rows.",

        removed,

    )

    return df, removed


###############################################################################

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values intelligently.
    """

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            if df[col].isna().any():

                median = df[col].median()

                df[col] = df[col].fillna(median)

        else:

            if df[col].isna().any():

                mode = df[col].mode()

                if not mode.empty:

                    df[col] = df[col].fillna(

                        mode.iloc[0]

                    )

    LOGGER.info(

        "Missing value handling completed."

    )

    return df


###############################################################################

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate processed dataframe.
    """

    if df.empty:

        raise ValueError(

            "Dataset is empty."

        )

    if df.columns.duplicated().any():

        raise ValueError(

            "Duplicate columns detected."

        )

    if df.duplicated().any():

        LOGGER.warning(

            "Duplicate rows still exist."

        )

    empty_columns = [

        col

        for col in df.columns

        if df[col].isna().all()

    ]

    if empty_columns:

        LOGGER.warning(

            "Empty columns detected: %s",

            empty_columns,

        )

###############################################################################
# End of Part 1
###############################################################################

###############################################################################
# UIDAI Dataset Specific Preprocessing
###############################################################################

def preprocess_enrolment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-specific preprocessing for Aadhaar Enrolment data.
    """

    LOGGER.info("Applying Enrolment dataset preprocessing...")

    # Normalize state names
    if "state" in df.columns:
        df["state"] = (
            df["state"]
            .astype("string")
            .str.title()
            .str.strip()
        )

    # Normalize district names
    if "district" in df.columns:
        df["district"] = (
            df["district"]
            .astype("string")
            .str.title()
            .str.strip()
        )

    # Remove impossible age values
    if "age" in df.columns:
        df.loc[
            (df["age"] < 0) |
            (df["age"] > 120),
            "age"
        ] = np.nan

    return df


###############################################################################

def preprocess_demographic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-specific preprocessing for Demographic Update data.
    """

    LOGGER.info("Applying Demographic dataset preprocessing...")

    if "state" in df.columns:
        df["state"] = (
            df["state"]
            .astype("string")
            .str.title()
        )

    if "district" in df.columns:
        df["district"] = (
            df["district"]
            .astype("string")
            .str.title()
        )

    return df


###############################################################################

def preprocess_biometric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-specific preprocessing for Biometric Update data.
    """

    LOGGER.info("Applying Biometric dataset preprocessing...")

    if "state" in df.columns:
        df["state"] = (
            df["state"]
            .astype("string")
            .str.title()
        )

    if "district" in df.columns:
        df["district"] = (
            df["district"]
            .astype("string")
            .str.title()
        )

    return df


###############################################################################
# Statistics
###############################################################################

def generate_statistics(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    duplicates_removed: int
) -> Dict:
    """
    Generate preprocessing statistics.
    """

    statistics = {

        "rows_before": len(before_df),

        "rows_after": len(after_df),

        "columns": len(after_df.columns),

        "duplicates_removed": duplicates_removed,

        "missing_values_remaining": int(
            after_df.isna().sum().sum()
        ),

        "memory_usage_mb": round(

            after_df.memory_usage(
                deep=True
            ).sum() / 1024 ** 2,

            2,

        ),

    }

    return statistics


###############################################################################
# Generic Dataset Pipeline
###############################################################################

def preprocess_dataset(
    dataset_name: str,
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline.
    """

    LOGGER.info(
        "Starting preprocessing for %s dataset...",
        dataset_name,
    )

    before_df = df.copy()

    ###########################################################################
    # Generic preprocessing
    ###########################################################################

    df = standardize_column_names(df)

    df = clean_text_columns(df)

    df = convert_data_types(df)

    df = parse_dates(df)

    df = handle_missing_values(df)

    df, duplicates_removed = remove_duplicates(df)

    ###########################################################################
    # Dataset-specific preprocessing
    ###########################################################################

    if dataset_name.lower() == "enrolment":

        df = preprocess_enrolment(df)

    elif dataset_name.lower() == "demographic":

        df = preprocess_demographic(df)

    elif dataset_name.lower() == "biometric":

        df = preprocess_biometric(df)

    ###########################################################################
    # Validation
    ###########################################################################

    validate_dataset(df)

    ###########################################################################
    # Statistics
    ###########################################################################

    statistics = generate_statistics(

        before_df,

        df,

        duplicates_removed,

    )

    LOGGER.info(

        "Completed preprocessing for %s.",

        dataset_name,

    )

    return df, statistics


###############################################################################
# Processing Report
###############################################################################

def generate_processing_report(
    statistics: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Convert statistics dictionary into a report dataframe.
    """

    report = pd.DataFrame.from_dict(

        statistics,

        orient="index",

    )

    report.index.name = "dataset"

    return report


###############################################################################
# End of Part 2
###############################################################################


###############################################################################
# Complete Pipeline
###############################################################################

def preprocess_all_datasets(
    datasets: Dict[str, pd.DataFrame]
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Dict]
]:
    """
    Preprocess all datasets.

    Parameters
    ----------
    datasets : dict
        Dictionary containing all loaded datasets.

    Returns
    -------
    processed_datasets
    preprocessing_statistics
    """

    processed_datasets = {}

    statistics = {}

    LOGGER.info(
        "=" * 70
    )

    LOGGER.info(
        "Starting preprocessing pipeline..."
    )

    LOGGER.info(
        "=" * 70
    )

    for dataset_name, dataframe in datasets.items():

        try:

            LOGGER.info(
                "Processing dataset: %s",
                dataset_name,
            )

            processed_df, stats = preprocess_dataset(
                dataset_name,
                dataframe,
            )

            processed_datasets[
                dataset_name
            ] = processed_df

            statistics[
                dataset_name
            ] = stats

            LOGGER.info(
                "Finished processing %s",
                dataset_name,
            )

        except Exception as error:

            LOGGER.exception(
                "Failed preprocessing dataset %s",
                dataset_name,
            )

            raise error

    LOGGER.info(
        "=" * 70
    )

    LOGGER.info(
        "All datasets processed successfully."
    )

    LOGGER.info(
        "=" * 70
    )

    return processed_datasets, statistics


###############################################################################
# Save Processed Files
###############################################################################

def save_processed_data(
    datasets: Dict[str, pd.DataFrame]
) -> None:
    """
    Save processed datasets.
    """

    output_directory = Path(
        PROCESSED_DATA_DIR
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset_name, dataframe in datasets.items():

        output_path = (
            output_directory /
            f"{dataset_name.lower()}_processed.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        LOGGER.info(
            "Saved: %s",
            output_path.name,
        )


###############################################################################
# Save Processing Report
###############################################################################

def save_processing_report(
    report: pd.DataFrame
) -> None:
    """
    Save preprocessing report.
    """

    output_directory = Path(
        PROCESSED_DATA_DIR
    )

    report_path = (
        output_directory /
        "preprocessing_report.csv"
    )

    report.to_csv(
        report_path
    )

    LOGGER.info(
        "Saved preprocessing report."
    )


###############################################################################
# Display Summary
###############################################################################

def print_summary(
    report: pd.DataFrame
) -> None:
    """
    Display preprocessing summary.
    """

    print("\n")

    print("=" * 80)

    print(
        "PREPROCESSING SUMMARY"
    )

    print("=" * 80)

    print(report)

    print("=" * 80)

    print("\n")


###############################################################################
# Main
###############################################################################

def main() -> None:
    """
    Main preprocessing pipeline.
    """

    LOGGER.info(
        "=" * 80
    )

    LOGGER.info(
        "AADHAAR INSIGHTS PREPROCESSING PIPELINE"
    )

    LOGGER.info(
        "=" * 80
    )

    try:

        #######################################################################
        # Load datasets
        #######################################################################

        datasets = load_all_datasets()

        #######################################################################
        # Preprocess
        #######################################################################

        processed_datasets, statistics = preprocess_all_datasets(
            datasets
        )

        #######################################################################
        # Save processed datasets
        #######################################################################

        save_processed_data(
            processed_datasets
        )

        #######################################################################
        # Generate report
        #######################################################################

        report = generate_processing_report(
            statistics
        )

        save_processing_report(
            report
        )

        print_summary(
            report
        )

        LOGGER.info(
            "Preprocessing completed successfully."
        )

    except Exception as error:

        LOGGER.exception(
            "Pipeline failed."
        )

        raise error


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()