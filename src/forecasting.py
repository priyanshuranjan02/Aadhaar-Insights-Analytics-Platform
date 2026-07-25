"""
===============================================================================
Project      : Aadhaar Insights
File         : forecasting.py
Author       : Priyanshu Ranjan

Description
-----------
Machine Learning forecasting engine for Aadhaar Insights.

Responsibilities
----------------
1. Dataset preparation
2. Feature selection
3. Data scaling
4. Train/Test split
5. Forecast model training
6. Model evaluation
7. Future prediction
8. Model persistence

===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.analytics import analyze_all_datasets
from src.config import LOGGER
from src.data_loader import load_all_datasets
from src.feature_engineering import feature_engineer_all_datasets
from src.preprocessing import preprocess_all_datasets
from src.config import PROCESSED_DATA_DIR


###############################################################################
# Helper Functions
###############################################################################

def numeric_columns(
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Return numeric columns.
    """

    return list(

        dataframe

        .select_dtypes(

            include=np.number

        )

        .columns

    )


###############################################################################

def categorical_columns(
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Return categorical columns.
    """

    return list(

        dataframe

        .select_dtypes(

            exclude=np.number

        )

        .columns

    )


###############################################################################

from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
)

def prepare_forecasting_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare engineered dataset for forecasting.
    """

    LOGGER.info("Preparing forecasting dataset.")

    df = dataframe.copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    # Handle numeric columns
    numeric = numeric_columns(df)

    for column in numeric:
        if is_numeric_dtype(df[column]):
            df[column] = (
                pd.to_numeric(df[column], errors="coerce")
                .astype("float64")
                .fillna(df[column].median())
            )

    # Handle datetime columns
    for column in list(df.columns):   # <-- iterate over a copy
        if is_datetime64_any_dtype(df[column]):
            df[f"{column}_year"] = df[column].dt.year
            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_day"] = df[column].dt.day
            df[f"{column}_dayofweek"] = df[column].dt.dayofweek

            df.drop(columns=[column], inplace=True)

    # Handle categorical columns
    categorical = categorical_columns(df)

    for column in categorical:
        df[column] = df[column].fillna("Unknown")

    LOGGER.info("Forecast dataset prepared.")

    print("\nRemaining datetime columns:")
    print(df.select_dtypes(include=["datetime", "datetimetz"]).dtypes)

    return df


###############################################################################
# Feature Selection
###############################################################################

from pandas.api.types import is_datetime64_any_dtype


def select_features(
    dataframe: pd.DataFrame,
    target_column: str,
):
    """
    Select features for prediction.
    """

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    features = dataframe.drop(columns=[target_column])

    target = dataframe[target_column]

    # Convert datetime columns into numeric components
    for column in features.columns:

        if is_datetime64_any_dtype(features[column]):

            features[f"{column}_year"] = features[column].dt.year
            features[f"{column}_month"] = features[column].dt.month
            features[f"{column}_day"] = features[column].dt.day
            features[f"{column}_dayofweek"] = features[column].dt.dayofweek

            features.drop(columns=[column], inplace=True)

    features = pd.get_dummies(
        features,
        drop_first=True,
    )

    LOGGER.info(
        "Feature selection completed."
    )

    return features, target


###############################################################################
# Train Test Split
###############################################################################

def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split dataset.
    """

    LOGGER.info(
        "Splitting dataset."
    )

    return train_test_split(

        features,

        target,

        test_size=test_size,

        random_state=random_state,

    )


###############################################################################
# Feature Scaling
###############################################################################

from sklearn.preprocessing import StandardScaler

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Scale numerical features.
    """

    # Select only numeric columns
    numeric_cols = X_train.select_dtypes(
        include=["number"]
    ).columns

    if len(numeric_cols) == 0:
        raise ValueError(
            "No numeric columns found for scaling."
        )

    # Ensure train and test have the same numeric columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    LOGGER.info(
        f"Feature scaling completed ({len(numeric_cols)} features)."
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler,
        list(numeric_cols),
    )


###############################################################################
# Complete Data Preparation
###############################################################################

def prepare_training_data(
    dataframe: pd.DataFrame,
    target_column: str,
):
    """
    Execute complete preparation pipeline.
    """

    dataframe = prepare_forecasting_data(
        dataframe
    )

    X, y = select_features(

        dataframe,

        target_column,

    )

    (

        X_train,

        X_test,

        y_train,

        y_test,

    ) = split_dataset(

        X,

        y,

    )

    (
        X_train,
        X_test,
        scaler,
        feature_names,
    ) = scale_features(
        X_train,
        X_test,
    )

    LOGGER.info(
        "Training dataset prepared."
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names,
    )


###############################################################################
# Dataset Summary
###############################################################################

def forecasting_summary(
    dataframe: pd.DataFrame,
):
    """
    Generate dataset summary.
    """

    summary = {

        "rows": len(dataframe),

        "columns": len(dataframe.columns),

        "numeric_columns": len(

            numeric_columns(dataframe)

        ),

        "categorical_columns": len(

            categorical_columns(dataframe)

        ),

        "missing_values": int(

            dataframe

            .isna()

            .sum()

            .sum()

        ),

        "memory_usage_mb": round(

            dataframe

            .memory_usage(

                deep=True

            )

            .sum()

            /

            1024**2,

            2,

        ),

    }

    LOGGER.info(
        "Forecast summary generated."
    )

    return summary


###############################################################################
# End of Part 1
###############################################################################

###############################################################################
# Model Imports
###############################################################################

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


###############################################################################
# Baseline Model
###############################################################################

def train_baseline_model(
    X_train,
    y_train,
):
    """
    Train baseline regression model.
    """

    LOGGER.info(
        "Training baseline model."
    )

    model = DummyRegressor(
        strategy="mean"
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


###############################################################################
# Linear Regression
###############################################################################

def train_linear_regression(
    X_train,
    y_train,
):
    """
    Train Linear Regression model.
    """

    LOGGER.info(
        "Training Linear Regression."
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    return model


###############################################################################
# Random Forest
###############################################################################

def train_random_forest(
    X_train,
    y_train,
):
    """
    Train Random Forest model.
    """

    LOGGER.info(
        "Training Random Forest."
    )

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


###############################################################################
# Prediction Utility
###############################################################################

def predict(
    model,
    X_test,
):
    """
    Generate predictions.
    """

    return model.predict(
        X_test
    )


###############################################################################
# Forecast DataFrame
###############################################################################

def prediction_dataframe(
    actual,
    predicted,
):
    """
    Build prediction dataframe.
    """

    prediction_df = pd.DataFrame({

        "Actual": actual,

        "Predicted": predicted,

    })

    prediction_df["Residual"] = (

        prediction_df["Actual"]

        -

        prediction_df["Predicted"]

    )

    prediction_df["Absolute Error"] = (

        prediction_df["Residual"]

        .abs()

    )

    return prediction_df


###############################################################################
# Complete Model Training
###############################################################################

def train_models(
    X_train,
    y_train,
):
    """
    Train every supported model.
    """

    models = {}

    models["Baseline"] = train_baseline_model(
        X_train,
        y_train,
    )

    models["LinearRegression"] = train_linear_regression(
        X_train,
        y_train,
    )

    models["RandomForest"] = train_random_forest(
        X_train,
        y_train,
    )

    LOGGER.info(
        "%d models trained.",
        len(models),
    )

    return models


###############################################################################
# Predict Using All Models
###############################################################################

def predict_all_models(
    models,
    X_test,
    y_test,
):
    """
    Generate predictions for every model.
    """

    predictions = {}

    for model_name, model in models.items():

        y_pred = predict(
            model,
            X_test,
        )

        predictions[model_name] = prediction_dataframe(

            y_test,

            y_pred,

        )

    LOGGER.info(
        "Predictions completed."
    )

    return predictions


###############################################################################
# End of Part 2
###############################################################################

###############################################################################
# Evaluation Imports
###############################################################################

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


###############################################################################
# Model Evaluation
###############################################################################

def evaluate_model(
    model_name: str,
    actual,
    predicted,
):
    """
    Evaluate a regression model.
    """

    mse = mean_squared_error(
        actual,
        predicted,
    )

    metrics = {

        "Model": model_name,

        "MAE": round(
            mean_absolute_error(
                actual,
                predicted,
            ),
            4,
        ),

        "MSE": round(
            mse,
            4,
        ),

        "RMSE": round(
            np.sqrt(mse),
            4,
        ),

        "MAPE": round(
            mean_absolute_percentage_error(
                actual,
                predicted,
            ) * 100,
            2,
        ),

        "R2": round(
            r2_score(
                actual,
                predicted,
            ),
            4,
        ),

    }

    return metrics


###############################################################################
# Evaluate All Models
###############################################################################

def evaluate_all_models(
    models,
    X_test,
    y_test,
):
    """
    Evaluate every trained model.
    """

    evaluation = []

    for model_name, model in models.items():

        prediction = model.predict(
            X_test
        )

        evaluation.append(

            evaluate_model(

                model_name,

                y_test,

                prediction,

            )

        )

    report = pd.DataFrame(
        evaluation
    )

    # report = report.sort_values(

    #     by="RMSE",

    #     ascending=True,

    # ).reset_index(drop=True)

    report = report.sort_values(
        by="R2",
        ascending=False,
    ).reset_index(drop=True)

    LOGGER.info(
        "Evaluation completed."
    )

    return report


###############################################################################
# Best Model
###############################################################################

def best_model(
    models,
    evaluation_report,
):
    """
    Return best-performing model.
    """

    best = evaluation_report.iloc[0]

    model_name = best["Model"]

    LOGGER.info(
        "Best model: %s",
        model_name,
    )

    return (
        model_name,
        models[model_name],
    )


###############################################################################
# Future Prediction
###############################################################################

def forecast_future(
    model,
    feature_dataframe,
    periods: int = 12,
):
    """
    Forecast future observations.

    Uses the latest available feature vector
    as the basis for iterative forecasting.
    """

    if len(feature_dataframe) == 0:

        return pd.DataFrame()

    latest = feature_dataframe.iloc[-1].copy()

    forecasts = []

    for step in range(1, periods + 1):

        prediction = model.predict(

            latest.values.reshape(1, -1)

        )[0]

        forecasts.append({

            "Step": step,

            "Forecast": prediction,

        })

    LOGGER.info(
        "%d future predictions generated.",
        periods,
    )

    return pd.DataFrame(
        forecasts
    )


###############################################################################
# Feature Importance
###############################################################################

def feature_importance(
    model,
    feature_names,
):
    """
    Extract feature importance
    if supported.
    """

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance = model.feature_importances_

    elif hasattr(
        model,
        "coef_",
    ):

        importance = np.abs(
            model.coef_
        )

    else:

        return pd.DataFrame()

    report = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance,

    })

    report = report.sort_values(

        by="Importance",

        ascending=False,

    ).reset_index(drop=True)

    LOGGER.info(
        "Feature importance generated."
    )

    return report


###############################################################################
# Forecast Report
###############################################################################

def forecast_report(
    evaluation_report,
    future_forecast,
):
    """
    Build overall forecasting report.
    """

    return {

        "evaluation": evaluation_report,

        "future_forecast": future_forecast,

    }


###############################################################################
# Complete Forecasting
###############################################################################

def forecasting_pipeline(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
):
    """
    Execute complete forecasting workflow.
    """

    models = train_models(

        X_train,

        y_train,

    )

    predictions = predict_all_models(
        models,
        X_test,
        y_test,
    )

    evaluation = evaluate_all_models(

        models,

        X_test,

        y_test,

    )

    model_name, model = best_model(

        models,

        evaluation,

    )

    future = forecast_future(

        model,

        pd.DataFrame(

            X_test,

            columns=feature_names,

        ),

    )

    importance = feature_importance(

        model,

        feature_names,

    )

    report = forecast_report(

        evaluation,

        future,

    )

    LOGGER.info(
        "Forecasting pipeline completed."
    )

    return {

        "models": models,

        "best_model_name": model_name,

        "best_model": model,

        "evaluation": evaluation,

        "future_forecast": future,

        "feature_importance": importance,

        "report": report,

        "predictions": predictions

    }


###############################################################################
# End of Part 3
###############################################################################

###############################################################################
# Model Persistence
###############################################################################

MODEL_DIR = Path("models") / "trained"
ARTIFACT_DIR = Path("models") / "artifacts"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_model(
    model,
    model_name: str,
    dataset_name: str
):
    """
    Save trained model.
    """

    filepath = MODEL_DIR / f"{dataset_name.lower()}_{model_name}.pkl"

    joblib.dump(
        model,
        filepath,
    )

    LOGGER.info(
        "Saved model: %s",
        filepath,
    )


###############################################################################

def save_scaler(
    scaler,
    dataset_name: str,
):
    """
    Save fitted scaler.
    """

    filepath = MODEL_DIR / f"{dataset_name.lower()}_scaler.pkl"

    joblib.dump(
        scaler,
        filepath,
    )

    LOGGER.info(
        "Scaler saved."
    )


###############################################################################

def load_model(
    model_name: str,
    dataset_name: str,
):
    """
    Load trained model.
    """

    filepath = MODEL_DIR / f"{dataset_name.lower()}_{model_name}.pkl"

    return joblib.load(filepath)


###############################################################################
# Export Utilities
###############################################################################

def export_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
):
    """
    Export dataframe as CSV.
    """

    filepath = ARTIFACT_DIR / filename

    dataframe.to_csv(
        filepath,
        index=False,
    )

    LOGGER.info(
        "Saved %s",
        filepath.name,
    )


###############################################################################

def export_forecasting_results(
    forecasting_results,
    dataset_name,
):
    """
    Export forecasting artifacts.
    """

    # ------------------------------------------------------------------
    # Existing exports (models/artifacts)
    # ------------------------------------------------------------------

    export_dataframe(
        forecasting_results["evaluation"],
        "evaluation.csv",
    )

    export_dataframe(
        forecasting_results["future_forecast"],
        "future_forecast.csv",
    )

    if not forecasting_results["feature_importance"].empty:

        export_dataframe(
            forecasting_results["feature_importance"],
            "feature_importance.csv",
        )

    # ------------------------------------------------------------------
    # Dashboard exports
    # ------------------------------------------------------------------

    analytics_dir = (
        PROCESSED_DATA_DIR
        / "analytics"
        / dataset_name.lower()
    )

    analytics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasting_results["evaluation"].to_csv(
        analytics_dir / "model_metrics.csv",
        index=False,
    )

    forecasting_results["future_forecast"].to_csv(
        analytics_dir / "forecast_month.csv",
        index=False,
    )

    if not forecasting_results["feature_importance"].empty:

        forecasting_results["feature_importance"].to_csv(
            analytics_dir / "feature_importance.csv",
            index=False,
        )

    LOGGER.info(
        "Forecast dashboard files exported to %s",
        analytics_dir,
    )

    best_model = (
    forecasting_results["best_model_name"]
    )

    forecasting_results["predictions"][
        best_model
    ].to_csv(

        analytics_dir
        / "actual_vs_predicted.csv",

        index=False,
    )


###############################################################################
# Save All Models
###############################################################################

def save_all_models(
    models,
    scaler,
    dataset_name
):
    """
    Save every trained model.
    """

    for model_name, model in models.items():

        filename = (

            model_name

            .lower()

            .replace(" ", "_")

        )

        save_model(

            model,

            filename,
            dataset_name

        )

    save_scaler(
        scaler,
        dataset_name
    )

    LOGGER.info(
        "All models saved."
    )


###############################################################################
# Forecast Summary
###############################################################################

def forecasting_summary_report(
    forecasting_results,
):
    """
    Print concise forecasting summary.
    """

    print()

    print("=" * 80)

    print("FORECASTING SUMMARY")

    print("=" * 80)

    print()

    print(

        "Best Model:",

        forecasting_results["best_model_name"],

    )

    print()

    print(

        forecasting_results["evaluation"]

    )

    print()

    print("=" * 80)

    print()


###############################################################################
# Complete Pipeline
###############################################################################

def run_forecasting_pipeline(
    dataframe: pd.DataFrame,
    target_column: str,
    dataset_name: str,
):
    """
    Execute end-to-end forecasting pipeline.
    """

    (

        X_train,

        X_test,

        y_train,

        y_test,

        scaler,

        feature_names,

    ) = prepare_training_data(

        dataframe,

        target_column,

    )

    forecasting_results = forecasting_pipeline(

        X_train,

        X_test,

        y_train,

        y_test,

        feature_names,

    )

    save_all_models(

        forecasting_results["models"],
        scaler,
        dataset_name

    )

    export_forecasting_results(

        forecasting_results,
        dataset_name,

    )

    forecasting_summary_report(

        forecasting_results,

    )

    return forecasting_results


###############################################################################
# Main
###############################################################################

def main():
    """
    Execute forecasting workflow.
    """

    LOGGER.info("=" * 80)
    LOGGER.info("AADHAAR INSIGHTS FORECASTING")
    LOGGER.info("=" * 80)

    raw = load_all_datasets()

    processed, _ = preprocess_all_datasets(
        raw
    )

    engineered, _ = feature_engineer_all_datasets(
        processed
    )

    ########################################################################
    # Sample large datasets
    ########################################################################

    sampled = {}

    for dataset_name, dataframe in engineered.items():

        if len(dataframe) > 100000:

            LOGGER.info(
                "Sampling %s from %d rows to 100000 rows.",
                dataset_name,
                len(dataframe),
            )

            sampled[dataset_name] = dataframe.sample(
                n=100000,
                random_state=42,
            )

        else:

            sampled[dataset_name] = dataframe

    ########################################################################
    # Analytics
    ########################################################################

    analyze_all_datasets(
        sampled
    )

    ########################################################################
    # Forecasting
    ########################################################################

    for dataset_name, dataframe in sampled.items():

        LOGGER.info("=" * 80)
        LOGGER.info("Processing dataset: %s", dataset_name)

        numeric = numeric_columns(dataframe)

        if len(numeric) == 0:
            LOGGER.warning(
                "Skipping %s (no numeric target found).",
                dataset_name,
            )
            continue

        target = numeric[-1]

        LOGGER.info(
            "Forecast Target: %s",
            target,
        )

        run_forecasting_pipeline(
            dataframe,
            target,
            dataset_name,
        )

    LOGGER.info(
        "Forecasting completed successfully."
    )


###############################################################################
# Entry Point
###############################################################################

if __name__ == "__main__":

    main()