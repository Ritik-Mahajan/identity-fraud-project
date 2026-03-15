"""
train_baseline_models.py

Train baseline fraud detection models using structured features only.
This is Stage 1 of the two-stage fraud detection system.

Models trained:
1. Logistic Regression (interpretable baseline)
2. LightGBM (gradient boosting)
3. XGBoost (optional comparison)

Usage:
    python src/train_baseline_models.py
    
Or import and use:
    from src.train_baseline_models import train_all_models
    results = train_all_models()
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

# Input file
INPUT_PARQUET = DATA_PROCESSED / "structured_features.parquet"
INPUT_CSV = DATA_PROCESSED / "structured_features.csv"

# Output files
OUTPUT_PREDICTIONS = DATA_PROCESSED / "baseline_predictions.parquet"
OUTPUT_METRICS = TABLES_DIR / "baseline_metrics.csv"

# Model files
LR_MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
LGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Numeric features for modeling
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "months_at_address",
    "months_at_employer",
    "num_prev_apps_same_device_7d",
    "num_prev_apps_same_email_30d",
    "num_prev_apps_same_phone_30d",
    "num_prev_apps_same_address_30d",
    "zip_ip_distance_proxy",
    "application_hour",
    "name_email_match_score",
    "income_age_ratio",
    "tenure_min",
]

# Binary features (treated as numeric)
BINARY_FEATURES = [
    "is_free_email_domain",
    "document_uploaded",
    "thin_file_flag",
    "night_application_flag",
    "high_device_velocity_flag",
    "high_identity_reuse_flag",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "state",
    "housing_status",
    "ip_region",
    "employer_industry",
]

# Target column
TARGET = "fraud_label"

# Metadata columns (not used for training)
META_COLUMNS = [
    "application_id",
    "application_date",
    "application_month",
    "fraud_label",
    "fraud_type",
    "difficulty_level",
    "generated_signal_score",
]

# All modeling features
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_feature_table() -> pd.DataFrame:
    """
    Load the structured feature table from Phase 6.
    
    Returns:
        DataFrame with features and metadata
    """
    if INPUT_PARQUET.exists():
        print(f"Loading from: {INPUT_PARQUET}")
        df = pd.read_parquet(INPUT_PARQUET)
    elif INPUT_CSV.exists():
        print(f"Loading from: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
    else:
        raise FileNotFoundError(f"Feature table not found at {INPUT_PARQUET} or {INPUT_CSV}")
    
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# TIME-BASED SPLIT
# =============================================================================

def create_time_based_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create train/validation/test split based on application month.
    
    Split design:
    - Train: First 70% of months (Jan-Aug 2024, 8 months)
    - Validation: Next 15% of months (Sep-Oct 2024, 2 months)
    - Test: Last 15% of months (Nov-Dec 2024, 2 months)
    
    Args:
        df: DataFrame with application_month column
        
    Returns:
        DataFrame with split_label column added
    """
    print("\n" + "=" * 60)
    print("CREATING TIME-BASED SPLIT")
    print("=" * 60)
    
    # Get sorted unique months
    months = sorted(df["application_month"].unique())
    n_months = len(months)
    
    print(f"Total months: {n_months}")
    print(f"Date range: {months[0]} to {months[-1]}")
    
    # Calculate split points
    # 70% train, 15% val, 15% test
    train_end_idx = int(n_months * 0.70)  # 8 months for train
    val_end_idx = int(n_months * 0.85)    # 2 months for val
    
    train_months = months[:train_end_idx]
    val_months = months[train_end_idx:val_end_idx]
    test_months = months[val_end_idx:]
    
    print(f"\nTrain months ({len(train_months)}): {train_months[0]} to {train_months[-1]}")
    print(f"Val months ({len(val_months)}): {val_months[0]} to {val_months[-1]}")
    print(f"Test months ({len(test_months)}): {test_months[0]} to {test_months[-1]}")
    
    # Assign split labels
    df = df.copy()
    df["split_label"] = "unknown"
    df.loc[df["application_month"].isin(train_months), "split_label"] = "train"
    df.loc[df["application_month"].isin(val_months), "split_label"] = "val"
    df.loc[df["application_month"].isin(test_months), "split_label"] = "test"
    
    # Print split statistics
    print("\n--- Split Statistics ---")
    for split in ["train", "val", "test"]:
        split_df = df[df["split_label"] == split]
        fraud_rate = split_df[TARGET].mean()
        print(f"{split:5s}: {len(split_df):,} rows ({100*len(split_df)/len(df):.1f}%), "
              f"fraud rate: {100*fraud_rate:.1f}%")
    
    return df


def prepare_feature_matrices(df: pd.DataFrame):
    """
    Prepare X and y matrices for each split.
    
    Returns:
        Dictionary with train/val/test X and y arrays
    """
    print("\n" + "=" * 60)
    print("PREPARING FEATURE MATRICES")
    print("=" * 60)
    
    data = {}
    
    for split in ["train", "val", "test"]:
        split_df = df[df["split_label"] == split]
        
        X = split_df[ALL_FEATURES].copy()
        y = split_df[TARGET].values
        
        data[f"X_{split}"] = X
        data[f"y_{split}"] = y
        
        print(f"{split}: X shape = {X.shape}, y shape = {y.shape}")
    
    return data


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

def create_preprocessor():
    """
    Create a preprocessing pipeline for the features.
    
    - Numeric features: StandardScaler
    - Categorical features: OneHotEncoder
    
    Returns:
        ColumnTransformer for preprocessing
    """
    numeric_cols = NUMERIC_FEATURES + BINARY_FEATURES
    categorical_cols = CATEGORICAL_FEATURES
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop"
    )
    
    return preprocessor


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    
    Uses sklearn Pipeline with preprocessing.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels array
        
    Returns:
        Trained pipeline (preprocessor + model)
    """
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",  # Handle class imbalance
            solver="lbfgs",
        ))
    ])
    
    # Fit the pipeline
    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # Get training score
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Training ROC-AUC: {train_auc:.4f}")
    
    return pipeline


def train_lightgbm(X_train, y_train, X_val=None, y_val=None):
    """
    Train a LightGBM model.
    
    Uses early stopping if validation data is provided.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels array
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        
    Returns:
        Dictionary with preprocessor and model
    """
    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM")
    print("=" * 60)
    
    # Create preprocessor and transform data
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Get feature names after preprocessing
    num_features = NUMERIC_FEATURES + BINARY_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = num_features + cat_feature_names
    
    # LightGBM parameters (beginner-friendly defaults)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": RANDOM_SEED,
        "is_unbalance": True,  # Handle class imbalance
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train_processed, label=y_train, feature_name=feature_names)
    
    callbacks = [lgb.log_evaluation(period=0)]  # Suppress verbose output
    
    if X_val is not None and y_val is not None:
        X_val_processed = preprocessor.transform(X_val)
        val_data = lgb.Dataset(X_val_processed, label=y_val, reference=train_data)
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
        
        print("Training with early stopping...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
    else:
        print("Training without validation...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            callbacks=callbacks,
        )
    
    # Get training score
    train_proba = model.predict(X_train_processed)
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Training ROC-AUC: {train_auc:.4f}")
    print(f"Best iteration: {model.best_iteration}")
    
    return {"preprocessor": preprocessor, "model": model}


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels array
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        
    Returns:
        Dictionary with preprocessor and model
    """
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)
    
    # Create preprocessor and transform data
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Calculate scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    # XGBoost parameters (beginner-friendly defaults)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "verbosity": 0,
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_processed, label=y_train)
    
    evals = [(dtrain, "train")]
    
    if X_val is not None and y_val is not None:
        X_val_processed = preprocessor.transform(X_val)
        dval = xgb.DMatrix(X_val_processed, label=y_val)
        evals.append((dval, "val"))
        
        print("Training with early stopping...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False,
        )
    else:
        print("Training without validation...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            verbose_eval=False,
        )
    
    # Get training score
    train_proba = model.predict(dtrain)
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Training ROC-AUC: {train_auc:.4f}")
    print(f"Best iteration: {model.best_iteration}")
    
    return {"preprocessor": preprocessor, "model": model}


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def predict_with_model(model_obj, X, model_type="sklearn"):
    """
    Get predictions from a model.
    
    Args:
        model_obj: Model object (pipeline or dict with preprocessor+model)
        X: Features DataFrame
        model_type: One of "sklearn", "lightgbm", "xgboost"
        
    Returns:
        Tuple of (probabilities, predictions)
    """
    if model_type == "sklearn":
        proba = model_obj.predict_proba(X)[:, 1]
    elif model_type == "lightgbm":
        X_processed = model_obj["preprocessor"].transform(X)
        proba = model_obj["model"].predict(X_processed)
    elif model_type == "xgboost":
        X_processed = model_obj["preprocessor"].transform(X)
        dmatrix = xgb.DMatrix(X_processed)
        proba = model_obj["model"].predict(dmatrix)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    pred = (proba >= 0.5).astype(int)
    return proba, pred


def evaluate_model(y_true, y_proba, y_pred, model_name, split_name):
    """
    Evaluate a model and return metrics.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        y_pred: Predicted labels
        model_name: Name of the model
        split_name: Name of the data split
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "model": model_name,
        "split": split_name,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    
    return metrics


def print_evaluation(metrics):
    """Print evaluation metrics in a readable format."""
    print(f"\n--- {metrics['model']} on {metrics['split']} ---")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Confusion: TN={metrics['true_negatives']}, FP={metrics['false_positives']}, "
          f"FN={metrics['false_negatives']}, TP={metrics['true_positives']}")


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_model(model_obj, path, model_type="sklearn"):
    """Save a model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f"Saved model to: {path}")


def save_metrics_table(metrics_list):
    """Save metrics to CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    df.to_csv(OUTPUT_METRICS, index=False)
    print(f"\nSaved metrics to: {OUTPUT_METRICS}")
    return df


def save_predictions(df, predictions_dict):
    """
    Save predictions to parquet.
    
    Args:
        df: Original DataFrame with split_label
        predictions_dict: Dict mapping column names to prediction arrays
    """
    # Create output DataFrame with metadata
    output_df = df[META_COLUMNS + ["split_label"]].copy()
    
    # Add predictions
    for col_name, values in predictions_dict.items():
        output_df[col_name] = values
    
    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(OUTPUT_PREDICTIONS, index=False)
    print(f"\nSaved predictions to: {OUTPUT_PREDICTIONS}")
    
    return output_df


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_all_models():
    """
    Main function to train all baseline models.
    
    Returns:
        Dictionary with models, metrics, and predictions
    """
    print("\n" + "=" * 60)
    print("PHASE 7: BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_feature_table()
    
    # Step 2: Create time-based split
    df = create_time_based_split(df)
    
    # Step 3: Prepare feature matrices
    data = prepare_feature_matrices(df)
    
    # Initialize storage
    models = {}
    all_metrics = []
    predictions = {
        "logistic_regression_score": np.zeros(len(df)),
        "logistic_regression_pred": np.zeros(len(df)),
        "lightgbm_score": np.zeros(len(df)),
        "lightgbm_pred": np.zeros(len(df)),
        "xgboost_score": np.zeros(len(df)),
        "xgboost_pred": np.zeros(len(df)),
    }
    
    # ==========================================================================
    # Train Logistic Regression
    # ==========================================================================
    lr_model = train_logistic_regression(data["X_train"], data["y_train"])
    models["logistic_regression"] = lr_model
    save_model(lr_model, LR_MODEL_PATH)
    
    # Evaluate on all splits
    for split in ["train", "val", "test"]:
        proba, pred = predict_with_model(lr_model, data[f"X_{split}"], "sklearn")
        metrics = evaluate_model(data[f"y_{split}"], proba, pred, "Logistic Regression", split)
        all_metrics.append(metrics)
        print_evaluation(metrics)
        
        # Store predictions
        mask = df["split_label"] == split
        predictions["logistic_regression_score"][mask] = proba
        predictions["logistic_regression_pred"][mask] = pred
    
    # ==========================================================================
    # Train LightGBM
    # ==========================================================================
    lgbm_model = train_lightgbm(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"]
    )
    models["lightgbm"] = lgbm_model
    save_model(lgbm_model, LGBM_MODEL_PATH)
    
    # Evaluate on all splits
    for split in ["train", "val", "test"]:
        proba, pred = predict_with_model(lgbm_model, data[f"X_{split}"], "lightgbm")
        metrics = evaluate_model(data[f"y_{split}"], proba, pred, "LightGBM", split)
        all_metrics.append(metrics)
        print_evaluation(metrics)
        
        # Store predictions
        mask = df["split_label"] == split
        predictions["lightgbm_score"][mask] = proba
        predictions["lightgbm_pred"][mask] = pred
    
    # ==========================================================================
    # Train XGBoost
    # ==========================================================================
    xgb_model = train_xgboost(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"]
    )
    models["xgboost"] = xgb_model
    save_model(xgb_model, XGB_MODEL_PATH)
    
    # Evaluate on all splits
    for split in ["train", "val", "test"]:
        proba, pred = predict_with_model(xgb_model, data[f"X_{split}"], "xgboost")
        metrics = evaluate_model(data[f"y_{split}"], proba, pred, "XGBoost", split)
        all_metrics.append(metrics)
        print_evaluation(metrics)
        
        # Store predictions
        mask = df["split_label"] == split
        predictions["xgboost_score"][mask] = proba
        predictions["xgboost_pred"][mask] = pred
    
    # ==========================================================================
    # Save outputs
    # ==========================================================================
    metrics_df = save_metrics_table(all_metrics)
    predictions_df = save_predictions(df, predictions)
    
    # ==========================================================================
    # Print final summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - TEST SET COMPARISON")
    print("=" * 60)
    
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    print("\n" + test_metrics[["model", "roc_auc", "pr_auc", "recall", "f1"]].to_string(index=False))
    
    # Find best model
    best_model = test_metrics.loc[test_metrics["roc_auc"].idxmax(), "model"]
    best_auc = test_metrics["roc_auc"].max()
    print(f"\nBest model by ROC-AUC: {best_model} ({best_auc:.4f})")
    
    return {
        "models": models,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "data": data,
        "df": df,
    }


def main():
    """Entry point for command-line execution."""
    results = train_all_models()
    return results


if __name__ == "__main__":
    main()
