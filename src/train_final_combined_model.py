"""
Phase 10: Final Combined Model Training

This module trains the final combined model that merges:
- Stage 1 structured model scores
- Stage 2 text/encoder features

It also performs an ablation study comparing:
1. Structured model only (baseline)
2. Text features only
3. Structured + text on all cases
4. Structured + text only on borderline cases (routed approach)

Author: Identity Fraud Detection Project
Phase: 10 - Final Combined Model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

# Text features to use in combined model
TEXT_FEATURES = [
    'application_ocr_similarity',
    'employment_consistency_score',
    'address_consistency_score',
    'verification_note_length',
    'ocr_text_length',
    'suspicious_keyword_count_total',
]

# Borderline band thresholds (from Phase 8)
BORDERLINE_LOW = 0.01
BORDERLINE_HIGH = 0.99


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_inputs():
    """
    Load all required input files for Phase 10.
    
    Returns:
        tuple: (baseline_predictions, text_encoder_features, structured_features)
    """
    print("Loading input files...")
    
    # Load baseline predictions (has Stage 1 scores and split labels)
    baseline_path = PROCESSED_DIR / "baseline_predictions.parquet"
    if baseline_path.exists():
        baseline_predictions = pd.read_parquet(baseline_path)
    else:
        baseline_predictions = pd.read_csv(PROCESSED_DIR / "baseline_predictions.csv")
    print(f"  Baseline predictions: {baseline_predictions.shape}")
    
    # Load text encoder features (has text-derived features for borderline cases)
    text_path = PROCESSED_DIR / "text_encoder_features.parquet"
    if text_path.exists():
        text_encoder_features = pd.read_parquet(text_path)
    else:
        text_encoder_features = pd.read_csv(PROCESSED_DIR / "text_encoder_features.csv")
    print(f"  Text encoder features: {text_encoder_features.shape}")
    
    # Load structured features (for text-only model training)
    struct_path = PROCESSED_DIR / "structured_features.parquet"
    if struct_path.exists():
        structured_features = pd.read_parquet(struct_path)
    else:
        structured_features = pd.read_csv(PROCESSED_DIR / "structured_features.csv")
    print(f"  Structured features: {structured_features.shape}")
    
    return baseline_predictions, text_encoder_features, structured_features


def load_cleaned_dataset():
    """
    Load the cleaned dataset to get text fields for all rows.
    
    Returns:
        pd.DataFrame: Cleaned dataset with text fields
    """
    cleaned_path = PROCESSED_DIR / "applications_cleaned.parquet"
    if cleaned_path.exists():
        return pd.read_parquet(cleaned_path)
    else:
        return pd.read_csv(PROCESSED_DIR / "applications_cleaned.csv")


# =============================================================================
# DATA MERGING FUNCTIONS
# =============================================================================

def merge_model_inputs(baseline_predictions, text_encoder_features):
    """
    Merge baseline predictions with text encoder features.
    
    For rows NOT in the text_encoder_features table (non-borderline),
    we fill text features with default values.
    
    Args:
        baseline_predictions: DataFrame with Stage 1 scores
        text_encoder_features: DataFrame with text features (borderline only)
    
    Returns:
        pd.DataFrame: Merged dataset with all rows
    """
    print("\nMerging model inputs...")
    
    # Select only the text features we need from the encoder features table
    text_cols = ['application_id'] + TEXT_FEATURES + ['borderline_flag']
    text_subset = text_encoder_features[text_cols].copy()
    
    # Merge with baseline predictions
    merged = baseline_predictions.merge(
        text_subset,
        on='application_id',
        how='left'
    )
    
    # Fill borderline_flag for non-borderline cases
    merged['borderline_flag'] = merged['borderline_flag'].fillna(0).astype(int)
    
    # For non-borderline cases, fill text features with neutral values
    # Using median values from borderline cases as a reasonable default
    for col in TEXT_FEATURES:
        if col in merged.columns:
            # Fill missing with median of available values
            median_val = text_encoder_features[col].median()
            merged[col] = merged[col].fillna(median_val)
    
    print(f"  Merged dataset: {merged.shape}")
    print(f"  Borderline cases: {merged['borderline_flag'].sum()}")
    
    return merged


def choose_primary_stage1_model(baseline_predictions):
    """
    Choose the best Stage 1 model based on test set performance.
    
    From Phase 7, LightGBM was the best model.
    
    Args:
        baseline_predictions: DataFrame with model scores
    
    Returns:
        tuple: (model_name, score_column, pred_column)
    """
    print("\nChoosing primary Stage 1 model...")
    
    # LightGBM was the best model from Phase 7
    # Test ROC-AUC: 0.995, PR-AUC: 0.974
    model_name = "lightgbm"
    score_col = "lightgbm_score"
    pred_col = "lightgbm_pred"
    
    print(f"  Selected: {model_name}")
    print(f"  Score column: {score_col}")
    
    return model_name, score_col, pred_col


# =============================================================================
# DATASET PREPARATION FUNCTIONS
# =============================================================================

def prepare_text_only_dataset(merged_df, text_encoder_features, cleaned_dataset):
    """
    Prepare a dataset for training a text-only model.
    
    We need to compute text features for ALL rows (not just borderline).
    For simplicity, we'll use the borderline cases for training/evaluation
    since that's where we have the features computed.
    
    For a more complete approach, we would need to run the encoder on all rows.
    Here, we'll train on borderline cases and evaluate on borderline cases.
    
    Args:
        merged_df: Merged dataset
        text_encoder_features: Text features for borderline cases
        cleaned_dataset: Full cleaned dataset with text fields
    
    Returns:
        pd.DataFrame: Dataset ready for text-only modeling
    """
    print("\nPreparing text-only dataset...")
    
    # For text-only model, we use the text encoder features table
    # which has text features computed for borderline cases
    text_df = text_encoder_features.copy()
    
    # Add split_label from merged_df if not already present
    if 'split_label' not in text_df.columns:
        split_info = merged_df[['application_id', 'split_label']].drop_duplicates()
        text_df = text_df.merge(split_info, on='application_id', how='left')
    
    print(f"  Text-only dataset: {text_df.shape}")
    if 'split_label' in text_df.columns:
        print(f"  Split distribution: {text_df['split_label'].value_counts().to_dict()}")
    else:
        print("  Warning: split_label not available")
    
    return text_df


def prepare_combined_dataset(merged_df, stage1_score_col):
    """
    Prepare a dataset for the combined model (Stage 1 score + text features).
    
    Args:
        merged_df: Merged dataset with all features
        stage1_score_col: Name of the Stage 1 score column
    
    Returns:
        pd.DataFrame: Dataset ready for combined modeling
    """
    print("\nPreparing combined dataset...")
    
    # Create a copy with the features we need
    combined_df = merged_df.copy()
    
    # Rename stage1 score for clarity
    combined_df['stage1_score'] = combined_df[stage1_score_col]
    
    # Features for combined model
    combined_features = ['stage1_score', 'borderline_flag'] + TEXT_FEATURES
    
    print(f"  Combined dataset: {combined_df.shape}")
    print(f"  Features: {combined_features}")
    
    return combined_df, combined_features


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_logistic_combiner(X_train, y_train, feature_names):
    """
    Train a Logistic Regression model as the combiner.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
    
    Returns:
        tuple: (trained_model, scaler)
    """
    print("\nTraining Logistic Regression combiner...")
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train_scaled, y_train)
    
    # Print feature importance (coefficients)
    print("\n  Feature coefficients:")
    for name, coef in sorted(zip(feature_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name}: {coef:.4f}")
    
    return model, scaler


def train_text_only_model(text_df, text_features):
    """
    Train a model using only text features.
    
    Note: Due to the small number of borderline cases in the training split,
    we may need to use train+val data for training the text-only model.
    
    Args:
        text_df: Dataset with text features
        text_features: List of text feature names
    
    Returns:
        tuple: (model, scaler, train_df, val_df, test_df)
    """
    print("\nTraining text-only model...")
    
    # Split by split_label
    train_df = text_df[text_df['split_label'] == 'train'].copy()
    val_df = text_df[text_df['split_label'] == 'val'].copy()
    test_df = text_df[text_df['split_label'] == 'test'].copy()
    
    print(f"  Train: {len(train_df)} (fraud: {train_df['fraud_label'].sum()})")
    print(f"  Val: {len(val_df)} (fraud: {val_df['fraud_label'].sum()})")
    print(f"  Test: {len(test_df)} (fraud: {test_df['fraud_label'].sum()})")
    
    # Check if training data has both classes
    train_fraud_count = train_df['fraud_label'].sum()
    train_legit_count = len(train_df) - train_fraud_count
    
    if train_fraud_count == 0 or train_legit_count == 0:
        print("  WARNING: Training set has only one class!")
        print("  Using train+val data for text-only model training...")
        
        # Combine train and val for training
        train_combined = pd.concat([train_df, val_df], ignore_index=True)
        X_train = train_combined[text_features].values
        y_train = train_combined['fraud_label'].values
        
        print(f"  Combined train+val: {len(train_combined)} (fraud: {y_train.sum()})")
    else:
        X_train = train_df[text_features].values
        y_train = train_df['fraud_label'].values
    
    # Check if we still have both classes
    if len(np.unique(y_train)) < 2:
        print("  ERROR: Still only one class after combining. Cannot train text-only model.")
        print("  Returning a dummy model that predicts based on class prior.")
        
        # Create a dummy model that predicts the majority class
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Return None for model to indicate failure
        return None, scaler, train_df, val_df, test_df
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    print("\n  Feature coefficients:")
    for name, coef in sorted(zip(text_features, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name}: {coef:.4f}")
    
    return model, scaler, train_df, val_df, test_df


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_setup(y_true, y_score, y_pred, setup_name, split_name):
    """
    Evaluate a model setup and return metrics.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        y_pred: Predicted labels
        setup_name: Name of the setup (e.g., "structured_only")
        split_name: Name of the split (e.g., "test")
    
    Returns:
        dict: Dictionary of metrics
    """
    # Handle edge cases
    if len(y_true) == 0:
        return {
            'setup': setup_name,
            'split': split_name,
            'n_samples': 0,
            'n_fraud': 0,
            'roc_auc': np.nan,
            'pr_auc': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1': np.nan,
            'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0
        }
    
    # Compute metrics
    metrics = {
        'setup': setup_name,
        'split': split_name,
        'n_samples': len(y_true),
        'n_fraud': int(y_true.sum()),
        'roc_auc': roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        'pr_auc': average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Confusion matrix
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics.update({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
    
    return metrics


def build_borderline_only_decision_logic(merged_df, stage1_score_col, stage1_pred_col,
                                          combined_model, combined_scaler, combined_features,
                                          borderline_low=BORDERLINE_LOW, borderline_high=BORDERLINE_HIGH):
    """
    Build the borderline-only routing decision logic.
    
    Logic:
    - For cases OUTSIDE the borderline band: use Stage 1 prediction directly
    - For cases INSIDE the borderline band: use the combined Stage 2 model
    
    Args:
        merged_df: Merged dataset with all features
        stage1_score_col: Stage 1 score column name
        stage1_pred_col: Stage 1 prediction column name
        combined_model: Trained combined model
        combined_scaler: Scaler for combined model
        combined_features: List of feature names for combined model
        borderline_low: Lower threshold for borderline band
        borderline_high: Upper threshold for borderline band
    
    Returns:
        pd.DataFrame: Dataset with routed predictions
    """
    print("\nBuilding borderline-only routing logic...")
    
    df = merged_df.copy()
    
    # Identify borderline cases
    df['is_borderline'] = (
        (df[stage1_score_col] >= borderline_low) & 
        (df[stage1_score_col] <= borderline_high)
    ).astype(int)
    
    # Get combined model predictions for all rows
    X_combined = df[combined_features].values
    X_combined_scaled = combined_scaler.transform(X_combined)
    combined_scores = combined_model.predict_proba(X_combined_scaled)[:, 1]
    combined_preds = (combined_scores >= 0.5).astype(int)
    
    # Apply routing logic
    # Default: use Stage 1 predictions
    df['routed_score'] = df[stage1_score_col].copy()
    df['routed_pred'] = df[stage1_pred_col].copy()
    
    # For borderline cases: use combined model
    borderline_mask = df['is_borderline'] == 1
    df.loc[borderline_mask, 'routed_score'] = combined_scores[borderline_mask]
    df.loc[borderline_mask, 'routed_pred'] = combined_preds[borderline_mask]
    
    print(f"  Total rows: {len(df)}")
    print(f"  Borderline rows (routed to Stage 2): {borderline_mask.sum()}")
    print(f"  Non-borderline rows (Stage 1 only): {(~borderline_mask).sum()}")
    
    return df


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_outputs(predictions_df, metrics_list, combined_model, combined_scaler):
    """
    Save all outputs from Phase 10.
    
    Args:
        predictions_df: DataFrame with all predictions
        metrics_list: List of metric dictionaries
        combined_model: Trained combined model
        combined_scaler: Scaler for combined model
    """
    print("\nSaving outputs...")
    
    # Save predictions
    pred_path = PROCESSED_DIR / "final_model_predictions.parquet"
    predictions_df.to_parquet(pred_path, index=False)
    print(f"  Saved: {pred_path}")
    
    # Also save as CSV for easy inspection
    csv_path = PROCESSED_DIR / "final_model_predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Save metrics table
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = TABLES_DIR / "final_model_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")
    
    # Save combined model
    model_path = MODELS_DIR / "final_combined_logistic_regression.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': combined_model,
            'scaler': combined_scaler,
            'features': ['stage1_score', 'borderline_flag'] + TEXT_FEATURES
        }, f)
    print(f"  Saved: {model_path}")


def print_ablation_summary(metrics_list):
    """
    Print a summary of the ablation study results.
    
    Args:
        metrics_list: List of metric dictionaries
    """
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Filter to test set for main comparison
    test_metrics = metrics_df[metrics_df['split'] == 'test'].copy()
    
    if len(test_metrics) > 0:
        print("\nTest Set Results:")
        print("-"*70)
        print(f"{'Setup':<40} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10}")
        print("-"*70)
        
        for _, row in test_metrics.iterrows():
            print(f"{row['setup']:<40} {row['roc_auc']:>10.4f} {row['pr_auc']:>10.4f} {row['f1']:>10.4f}")
    
    # Borderline-only results
    borderline_metrics = metrics_df[metrics_df['split'] == 'test_borderline'].copy()
    
    if len(borderline_metrics) > 0:
        print("\nBorderline Cases Only (Test Set):")
        print("-"*70)
        print(f"{'Setup':<40} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10}")
        print("-"*70)
        
        for _, row in borderline_metrics.iterrows():
            roc = f"{row['roc_auc']:.4f}" if not np.isnan(row['roc_auc']) else "N/A"
            pr = f"{row['pr_auc']:.4f}" if not np.isnan(row['pr_auc']) else "N/A"
            f1 = f"{row['f1']:.4f}" if not np.isnan(row['f1']) else "N/A"
            print(f"{row['setup']:<40} {roc:>10} {pr:>10} {f1:>10}")
    
    print("="*70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to run Phase 10: Final Combined Model.
    
    This function:
    1. Loads all inputs
    2. Merges datasets
    3. Trains models for ablation study
    4. Evaluates all setups
    5. Saves outputs
    """
    print("="*70)
    print("PHASE 10: FINAL COMBINED MODEL")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load inputs
    # -------------------------------------------------------------------------
    baseline_predictions, text_encoder_features, structured_features = load_inputs()
    cleaned_dataset = load_cleaned_dataset()
    
    # -------------------------------------------------------------------------
    # Step 2: Choose primary Stage 1 model
    # -------------------------------------------------------------------------
    model_name, stage1_score_col, stage1_pred_col = choose_primary_stage1_model(baseline_predictions)
    
    # -------------------------------------------------------------------------
    # Step 3: Merge datasets
    # -------------------------------------------------------------------------
    merged_df = merge_model_inputs(baseline_predictions, text_encoder_features)
    
    # -------------------------------------------------------------------------
    # Step 4: Prepare datasets for ablation study
    # -------------------------------------------------------------------------
    
    # 4a. Text-only dataset (borderline cases only, since that's where we have features)
    text_df = prepare_text_only_dataset(merged_df, text_encoder_features, cleaned_dataset)
    
    # 4b. Combined dataset (all cases)
    combined_df, combined_features = prepare_combined_dataset(merged_df, stage1_score_col)
    
    # -------------------------------------------------------------------------
    # Step 5: Train models
    # -------------------------------------------------------------------------
    
    # 5a. Train text-only model (on borderline cases)
    text_model, text_scaler, text_train, text_val, text_test = train_text_only_model(
        text_df, TEXT_FEATURES
    )
    
    # 5b. Train combined model (on all training data)
    train_mask = combined_df['split_label'] == 'train'
    X_train_combined = combined_df.loc[train_mask, combined_features].values
    y_train_combined = combined_df.loc[train_mask, 'fraud_label'].values
    
    combined_model, combined_scaler = train_logistic_combiner(
        X_train_combined, y_train_combined, combined_features
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Generate predictions for all setups
    # -------------------------------------------------------------------------
    print("\nGenerating predictions for all setups...")
    
    # Stage 1 predictions (already in merged_df)
    merged_df['stage1_score'] = merged_df[stage1_score_col]
    merged_df['stage1_pred'] = merged_df[stage1_pred_col]
    
    # Text-only predictions (only for borderline cases)
    # For non-borderline, we'll use NaN
    merged_df['text_only_score'] = np.nan
    merged_df['text_only_pred'] = np.nan
    
    # Get text-only predictions for borderline cases (if model was trained)
    if text_model is not None:
        borderline_ids = text_df['application_id'].values
        X_text = text_df[TEXT_FEATURES].values
        X_text_scaled = text_scaler.transform(X_text)
        text_scores = text_model.predict_proba(X_text_scaled)[:, 1]
        text_preds = (text_scores >= 0.5).astype(int)
        
        for i, app_id in enumerate(borderline_ids):
            mask = merged_df['application_id'] == app_id
            merged_df.loc[mask, 'text_only_score'] = text_scores[i]
            merged_df.loc[mask, 'text_only_pred'] = text_preds[i]
    else:
        print("  Text-only model not available, skipping text-only predictions")
    
    # Combined model predictions (all cases)
    X_combined_all = merged_df[combined_features].values
    X_combined_all_scaled = combined_scaler.transform(X_combined_all)
    merged_df['combined_all_score'] = combined_model.predict_proba(X_combined_all_scaled)[:, 1]
    merged_df['combined_all_pred'] = (merged_df['combined_all_score'] >= 0.5).astype(int)
    
    # Borderline-routed predictions
    routed_df = build_borderline_only_decision_logic(
        merged_df, stage1_score_col, stage1_pred_col,
        combined_model, combined_scaler, combined_features
    )
    merged_df['final_borderline_routed_score'] = routed_df['routed_score']
    merged_df['final_borderline_routed_pred'] = routed_df['routed_pred']
    
    # -------------------------------------------------------------------------
    # Step 7: Evaluate all setups (Ablation Study)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("ABLATION STUDY: EVALUATING ALL SETUPS")
    print("="*70)
    
    metrics_list = []
    
    # Define splits to evaluate
    splits = ['val', 'test']
    
    for split in splits:
        split_mask = merged_df['split_label'] == split
        split_df = merged_df[split_mask]
        
        y_true = split_df['fraud_label'].values
        
        # Setup 1: Structured model only
        y_score_s1 = split_df['stage1_score'].values
        y_pred_s1 = split_df['stage1_pred'].values
        metrics_list.append(evaluate_setup(y_true, y_score_s1, y_pred_s1, 
                                           "1_structured_only", split))
        
        # Setup 2: Text features only (borderline cases only)
        borderline_split = split_df[split_df['borderline_flag'] == 1]
        if len(borderline_split) > 0:
            y_true_bl = borderline_split['fraud_label'].values
            y_score_text = borderline_split['text_only_score'].values
            y_pred_text = borderline_split['text_only_pred'].values
            # Remove NaN values
            valid_mask = ~np.isnan(y_score_text)
            if valid_mask.sum() > 0:
                metrics_list.append(evaluate_setup(
                    y_true_bl[valid_mask], y_score_text[valid_mask], 
                    y_pred_text[valid_mask].astype(int),
                    "2_text_only_borderline", split
                ))
        
        # Setup 3: Structured + text on all cases
        y_score_comb = split_df['combined_all_score'].values
        y_pred_comb = split_df['combined_all_pred'].values
        metrics_list.append(evaluate_setup(y_true, y_score_comb, y_pred_comb,
                                           "3_combined_all_cases", split))
        
        # Setup 4: Structured + text only on borderline cases (routed)
        y_score_routed = split_df['final_borderline_routed_score'].values
        y_pred_routed = split_df['final_borderline_routed_pred'].values
        metrics_list.append(evaluate_setup(y_true, y_score_routed, y_pred_routed,
                                           "4_borderline_routed", split))
        
        # Also evaluate on borderline subset specifically
        if len(borderline_split) > 0:
            y_true_bl = borderline_split['fraud_label'].values
            
            # Structured only on borderline
            y_score_s1_bl = borderline_split['stage1_score'].values
            y_pred_s1_bl = borderline_split['stage1_pred'].values
            metrics_list.append(evaluate_setup(y_true_bl, y_score_s1_bl, y_pred_s1_bl,
                                               "1_structured_only", f"{split}_borderline"))
            
            # Combined on borderline
            y_score_comb_bl = borderline_split['combined_all_score'].values
            y_pred_comb_bl = borderline_split['combined_all_pred'].values
            metrics_list.append(evaluate_setup(y_true_bl, y_score_comb_bl, y_pred_comb_bl,
                                               "3_combined_all_cases", f"{split}_borderline"))
            
            # Routed on borderline (should be same as combined for borderline)
            y_score_routed_bl = borderline_split['final_borderline_routed_score'].values
            y_pred_routed_bl = borderline_split['final_borderline_routed_pred'].values
            metrics_list.append(evaluate_setup(y_true_bl, y_score_routed_bl, y_pred_routed_bl,
                                               "4_borderline_routed", f"{split}_borderline"))
    
    # Print summary
    print_ablation_summary(metrics_list)
    
    # -------------------------------------------------------------------------
    # Step 8: Prepare final predictions table
    # -------------------------------------------------------------------------
    print("\nPreparing final predictions table...")
    
    # Select columns for output
    output_cols = [
        'application_id',
        'fraud_label',
        'fraud_type',
        'difficulty_level',
        'split_label',
        'stage1_score',
        'stage1_pred',
        'text_only_score',
        'text_only_pred',
        'combined_all_score',
        'combined_all_pred',
        'final_borderline_routed_score',
        'final_borderline_routed_pred',
        'borderline_flag'
    ]
    
    predictions_df = merged_df[output_cols].copy()
    
    # -------------------------------------------------------------------------
    # Step 9: Save outputs
    # -------------------------------------------------------------------------
    save_outputs(predictions_df, metrics_list, combined_model, combined_scaler)
    
    print("\n" + "="*70)
    print("PHASE 10 COMPLETE")
    print("="*70)
    
    return predictions_df, metrics_list


if __name__ == "__main__":
    predictions_df, metrics_list = main()
