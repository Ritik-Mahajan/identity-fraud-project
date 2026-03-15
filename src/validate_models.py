"""
Phase 11: Model Validation

This module performs comprehensive validation of the fraud detection system:
1. Standard model evaluation (ROC-AUC, PR-AUC, precision, recall, F1)
2. Borderline-case evaluation (the most important section)
3. Threshold analysis
4. Calibration analysis
5. Ablation study validation
6. Stability over time
7. Runtime/practicality assessment

Author: Identity Fraud Detection Project
Phase: 11 - Model Validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve,
    roc_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Thresholds to analyze
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_validation_inputs():
    """
    Load all inputs needed for validation.
    
    Returns:
        dict: Dictionary containing all loaded dataframes
    """
    print("Loading validation inputs...")
    
    inputs = {}
    
    # Final model predictions (from Phase 10)
    inputs['final_predictions'] = pd.read_parquet(PROCESSED_DIR / "final_model_predictions.parquet")
    print(f"  Final predictions: {inputs['final_predictions'].shape}")
    
    # Baseline predictions (from Phase 7) - has application_date and month
    inputs['baseline_predictions'] = pd.read_parquet(PROCESSED_DIR / "baseline_predictions.parquet")
    print(f"  Baseline predictions: {inputs['baseline_predictions'].shape}")
    
    # Text encoder features (from Phase 9)
    inputs['text_features'] = pd.read_parquet(PROCESSED_DIR / "text_encoder_features.parquet")
    print(f"  Text features: {inputs['text_features'].shape}")
    
    # Phase 7 metrics
    inputs['baseline_metrics'] = pd.read_csv(TABLES_DIR / "baseline_metrics.csv")
    print(f"  Baseline metrics: {inputs['baseline_metrics'].shape}")
    
    # Phase 10 comparison
    inputs['final_comparison'] = pd.read_csv(TABLES_DIR / "final_model_comparison.csv")
    print(f"  Final comparison: {inputs['final_comparison'].shape}")
    
    # Merge to get application_date and application_month
    date_cols = ['application_id', 'application_date', 'application_month']
    inputs['final_predictions'] = inputs['final_predictions'].merge(
        inputs['baseline_predictions'][date_cols],
        on='application_id',
        how='left'
    )
    
    return inputs


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def compute_classification_metrics(y_true, y_score, y_pred, setup_name, split_name, subset_name='overall'):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        y_pred: Predicted labels
        setup_name: Name of the setup
        split_name: Name of the split (train/val/test)
        subset_name: Name of the subset (overall/borderline_only)
    
    Returns:
        dict: Dictionary of metrics
    """
    n_samples = len(y_true)
    n_fraud = int(y_true.sum())
    
    # Handle edge cases
    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return {
            'setup': setup_name,
            'split': split_name,
            'subset': subset_name,
            'n_samples': n_samples,
            'n_fraud': n_fraud,
            'fraud_rate': n_fraud / n_samples if n_samples > 0 else 0,
            'roc_auc': np.nan,
            'pr_auc': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1': np.nan,
            'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
            'error_count': 0,
            'error_rate': 0
        }
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    error_count = fp + fn
    error_rate = error_count / n_samples
    
    metrics = {
        'setup': setup_name,
        'split': split_name,
        'subset': subset_name,
        'n_samples': n_samples,
        'n_fraud': n_fraud,
        'fraud_rate': n_fraud / n_samples,
        'roc_auc': roc_auc_score(y_true, y_score),
        'pr_auc': average_precision_score(y_true, y_score),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'error_count': error_count,
        'error_rate': error_rate
    }
    
    return metrics


def build_confusion_summary(y_true, y_pred):
    """
    Build a confusion matrix summary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Confusion matrix values
    """
    if len(y_true) == 0:
        return {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


# =============================================================================
# SETUP EVALUATION
# =============================================================================

def evaluate_setup(df, score_col, pred_col, setup_name, split_name, subset_mask=None):
    """
    Evaluate a single setup on a given split.
    
    Args:
        df: DataFrame with predictions
        score_col: Column name for scores
        pred_col: Column name for predictions
        setup_name: Name of the setup
        split_name: Name of the split
        subset_mask: Optional boolean mask for subset evaluation
    
    Returns:
        dict: Metrics dictionary
    """
    if subset_mask is not None:
        eval_df = df[subset_mask]
        subset_name = 'borderline_only'
    else:
        eval_df = df
        subset_name = 'overall'
    
    y_true = eval_df['fraud_label'].values
    y_score = eval_df[score_col].values
    y_pred = eval_df[pred_col].values
    
    # Handle NaN scores (text-only model only has scores for borderline cases)
    valid_mask = ~np.isnan(y_score)
    if valid_mask.sum() < len(y_score):
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]
        y_pred = y_pred[valid_mask].astype(int)
    
    return compute_classification_metrics(y_true, y_score, y_pred, setup_name, split_name, subset_name)


def evaluate_borderline_subset(df, baseline_score_col, baseline_pred_col, setups_config):
    """
    Evaluate all setups specifically on the borderline subset.
    
    Args:
        df: DataFrame with predictions
        baseline_score_col: Column name for baseline scores
        baseline_pred_col: Column name for baseline predictions
        setups_config: List of (setup_name, score_col, pred_col) tuples
    
    Returns:
        list: List of metrics dictionaries with error reduction calculations
    """
    print("\nEvaluating borderline subset...")
    
    # Get borderline cases
    borderline_mask = df['borderline_flag'] == 1
    borderline_df = df[borderline_mask]
    
    print(f"  Borderline cases: {len(borderline_df)}")
    print(f"  Fraud rate in borderline: {borderline_df['fraud_label'].mean():.1%}")
    
    results = []
    
    # Get baseline metrics for comparison
    y_true = borderline_df['fraud_label'].values
    y_score_baseline = borderline_df[baseline_score_col].values
    y_pred_baseline = borderline_df[baseline_pred_col].values
    
    baseline_cm = build_confusion_summary(y_true, y_pred_baseline)
    baseline_errors = baseline_cm['fp'] + baseline_cm['fn']
    baseline_fp = baseline_cm['fp']
    baseline_fn = baseline_cm['fn']
    
    for setup_name, score_col, pred_col in setups_config:
        y_score = borderline_df[score_col].values
        y_pred = borderline_df[pred_col].values
        
        # Handle NaN
        valid_mask = ~np.isnan(y_score)
        if valid_mask.sum() == 0:
            continue
            
        y_true_valid = y_true[valid_mask]
        y_score_valid = y_score[valid_mask]
        y_pred_valid = y_pred[valid_mask].astype(int)
        
        metrics = compute_classification_metrics(
            y_true_valid, y_score_valid, y_pred_valid,
            setup_name, 'test', 'borderline_only'
        )
        
        # Compute error reductions vs baseline
        candidate_errors = metrics['fp'] + metrics['fn']
        candidate_fp = metrics['fp']
        candidate_fn = metrics['fn']
        
        # Safe division
        metrics['error_reduction'] = (baseline_errors - candidate_errors) / baseline_errors if baseline_errors > 0 else 0
        metrics['fp_reduction'] = (baseline_fp - candidate_fp) / baseline_fp if baseline_fp > 0 else 0
        metrics['fn_reduction'] = (baseline_fn - candidate_fn) / baseline_fn if baseline_fn > 0 else 0
        
        results.append(metrics)
    
    return results


# =============================================================================
# THRESHOLD ANALYSIS
# =============================================================================

def run_threshold_analysis(df, setups_config, thresholds=THRESHOLDS, split='test'):
    """
    Analyze performance at different thresholds.
    
    Args:
        df: DataFrame with predictions
        setups_config: List of (setup_name, score_col) tuples
        thresholds: List of thresholds to test
        split: Split to evaluate on
    
    Returns:
        pd.DataFrame: Threshold analysis results
    """
    print(f"\nRunning threshold analysis on {split} set...")
    
    split_df = df[df['split_label'] == split]
    y_true = split_df['fraud_label'].values
    n_total = len(y_true)
    
    results = []
    
    for setup_name, score_col in setups_config:
        y_score = split_df[score_col].values
        
        # Handle NaN
        valid_mask = ~np.isnan(y_score)
        if valid_mask.sum() == 0:
            continue
        
        y_true_valid = y_true[valid_mask]
        y_score_valid = y_score[valid_mask]
        n_valid = len(y_true_valid)
        
        for threshold in thresholds:
            y_pred = (y_score_valid >= threshold).astype(int)
            
            # Compute metrics
            flagged = y_pred.sum()
            review_rate = flagged / n_valid if n_valid > 0 else 0
            
            if flagged > 0:
                precision = precision_score(y_true_valid, y_pred, zero_division=0)
                recall = recall_score(y_true_valid, y_pred, zero_division=0)
                f1 = f1_score(y_true_valid, y_pred, zero_division=0)
            else:
                precision = 0
                recall = 0
                f1 = 0
            
            results.append({
                'setup': setup_name,
                'threshold': threshold,
                'n_flagged': flagged,
                'review_rate': review_rate,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
    
    return pd.DataFrame(results)


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def compute_calibration_metrics(df, setups_config, split='test', n_bins=10):
    """
    Compute calibration metrics for each setup.
    
    Args:
        df: DataFrame with predictions
        setups_config: List of (setup_name, score_col) tuples
        split: Split to evaluate on
        n_bins: Number of bins for calibration curve
    
    Returns:
        tuple: (calibration_df, calibration_curves)
    """
    print(f"\nComputing calibration metrics on {split} set...")
    
    split_df = df[df['split_label'] == split]
    y_true = split_df['fraud_label'].values
    
    results = []
    curves = {}
    
    for setup_name, score_col in setups_config:
        y_score = split_df[score_col].values
        
        # Handle NaN
        valid_mask = ~np.isnan(y_score)
        if valid_mask.sum() == 0:
            continue
        
        y_true_valid = y_true[valid_mask]
        y_score_valid = y_score[valid_mask]
        
        # Brier score
        brier = brier_score_loss(y_true_valid, y_score_valid)
        
        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_true_valid, y_score_valid, n_bins=n_bins, strategy='uniform')
            curves[setup_name] = (prob_true, prob_pred)
            
            # Mean absolute calibration error
            mace = np.mean(np.abs(prob_true - prob_pred))
        except:
            mace = np.nan
            curves[setup_name] = (None, None)
        
        results.append({
            'setup': setup_name,
            'brier_score': brier,
            'mean_abs_calibration_error': mace,
            'n_samples': len(y_true_valid)
        })
    
    return pd.DataFrame(results), curves


# =============================================================================
# STABILITY OVER TIME
# =============================================================================

def evaluate_monthly_stability(df, setups_config, split='test'):
    """
    Evaluate performance stability over time (by month).
    
    Args:
        df: DataFrame with predictions
        setups_config: List of (setup_name, score_col, pred_col) tuples
        split: Split to evaluate on
    
    Returns:
        pd.DataFrame: Monthly metrics
    """
    print(f"\nEvaluating monthly stability on {split} set...")
    
    split_df = df[df['split_label'] == split].copy()
    
    # Ensure application_month is available
    if 'application_month' not in split_df.columns:
        print("  Warning: application_month not available")
        return pd.DataFrame()
    
    months = sorted(split_df['application_month'].unique())
    print(f"  Months in {split}: {months}")
    
    results = []
    
    for month in months:
        month_df = split_df[split_df['application_month'] == month]
        n_samples = len(month_df)
        n_fraud = month_df['fraud_label'].sum()
        fraud_rate = n_fraud / n_samples if n_samples > 0 else 0
        
        for setup_name, score_col, pred_col in setups_config:
            y_true = month_df['fraud_label'].values
            y_score = month_df[score_col].values
            y_pred = month_df[pred_col].values
            
            # Handle NaN
            valid_mask = ~np.isnan(y_score)
            if valid_mask.sum() == 0:
                continue
            
            y_true_valid = y_true[valid_mask]
            y_score_valid = y_score[valid_mask]
            y_pred_valid = y_pred[valid_mask].astype(int)
            
            # Compute metrics (may fail for small samples)
            try:
                if len(np.unique(y_true_valid)) > 1:
                    roc_auc = roc_auc_score(y_true_valid, y_score_valid)
                else:
                    roc_auc = np.nan
            except:
                roc_auc = np.nan
            
            avg_score = np.mean(y_score_valid)
            fraud_captured = y_pred_valid[y_true_valid == 1].sum() if n_fraud > 0 else 0
            capture_rate = fraud_captured / n_fraud if n_fraud > 0 else 0
            
            results.append({
                'month': month,
                'setup': setup_name,
                'n_samples': n_samples,
                'n_fraud': n_fraud,
                'fraud_rate': fraud_rate,
                'avg_score': avg_score,
                'roc_auc': roc_auc,
                'fraud_captured': fraud_captured,
                'capture_rate': capture_rate
            })
    
    return pd.DataFrame(results)


# =============================================================================
# RUNTIME MEASUREMENT
# =============================================================================

def measure_runtime(inputs):
    """
    Measure approximate runtimes for different stages.
    
    Args:
        inputs: Dictionary of loaded inputs
    
    Returns:
        pd.DataFrame: Runtime summary
    """
    print("\nMeasuring runtimes...")
    
    results = []
    n_samples = len(inputs['final_predictions'])
    
    # Stage 1: Load model and predict
    try:
        model_path = MODELS_DIR / "lightgbm_model.pkl"
        
        start = time.time()
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        load_time = time.time() - start
        
        results.append({
            'stage': 'Stage 1 Model Load',
            'time_seconds': load_time,
            'time_per_sample_ms': (load_time / n_samples) * 1000 if n_samples > 0 else 0,
            'notes': 'LightGBM model loading'
        })
    except Exception as e:
        results.append({
            'stage': 'Stage 1 Model Load',
            'time_seconds': np.nan,
            'time_per_sample_ms': np.nan,
            'notes': f'Error: {str(e)}'
        })
    
    # Stage 2: Combined model load and predict
    try:
        model_path = MODELS_DIR / "final_combined_logistic_regression.pkl"
        
        start = time.time()
        with open(model_path, 'rb') as f:
            combined_data = pickle.load(f)
        load_time = time.time() - start
        
        results.append({
            'stage': 'Stage 2 Combined Model Load',
            'time_seconds': load_time,
            'time_per_sample_ms': (load_time / n_samples) * 1000 if n_samples > 0 else 0,
            'notes': 'Logistic Regression combiner loading'
        })
    except Exception as e:
        results.append({
            'stage': 'Stage 2 Combined Model Load',
            'time_seconds': np.nan,
            'time_per_sample_ms': np.nan,
            'notes': f'Error: {str(e)}'
        })
    
    # Encoder feature generation (estimate from Phase 9)
    # Note: This is an estimate based on typical sentence-transformer performance
    n_borderline = inputs['final_predictions']['borderline_flag'].sum()
    encoder_time_estimate = 0.05 * n_borderline  # ~50ms per sample for MiniLM
    
    results.append({
        'stage': 'Encoder Feature Generation (estimate)',
        'time_seconds': encoder_time_estimate,
        'time_per_sample_ms': 50.0,
        'notes': f'Estimated for {n_borderline} borderline cases using MiniLM'
    })
    
    # Total Stage 2 overhead
    total_stage2 = encoder_time_estimate + (results[1]['time_seconds'] if len(results) > 1 else 0)
    results.append({
        'stage': 'Total Stage 2 Overhead (estimate)',
        'time_seconds': total_stage2,
        'time_per_sample_ms': (total_stage2 / n_borderline) * 1000 if n_borderline > 0 else 0,
        'notes': 'Encoder + combined model for borderline cases'
    })
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_validation_figures(df, inputs, setups_config):
    """
    Create validation figures.
    
    Args:
        df: DataFrame with predictions
        inputs: Dictionary of loaded inputs
        setups_config: Configuration for setups
    """
    print("\nCreating validation figures...")
    
    test_df = df[df['split_label'] == 'test']
    y_true = test_df['fraud_label'].values
    
    # Figure 1: Score distribution by label - Structured model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    legit_scores = test_df[test_df['fraud_label'] == 0]['stage1_score'].values
    fraud_scores = test_df[test_df['fraud_label'] == 1]['stage1_score'].values
    ax1.hist(legit_scores, bins=50, alpha=0.7, label='Legitimate', color='steelblue')
    ax1.hist(fraud_scores, bins=50, alpha=0.7, label='Fraud', color='crimson')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Stage 1 (LightGBM) Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Figure 2: Score distribution - Routed system
    ax2 = axes[1]
    legit_scores = test_df[test_df['fraud_label'] == 0]['final_borderline_routed_score'].values
    fraud_scores = test_df[test_df['fraud_label'] == 1]['final_borderline_routed_score'].values
    ax2.hist(legit_scores, bins=50, alpha=0.7, label='Legitimate', color='steelblue')
    ax2.hist(fraud_scores, bins=50, alpha=0.7, label='Fraud', color='crimson')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Borderline-Routed System Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'score_distributions.png'}")
    
    # Figure 3: Threshold vs Precision/Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for setup_name, score_col in [('Stage 1 (LightGBM)', 'stage1_score'), 
                                   ('Borderline Routed', 'final_borderline_routed_score')]:
        y_score = test_df[score_col].values
        precision_vals = []
        recall_vals = []
        
        for thresh in np.linspace(0.1, 0.9, 17):
            y_pred = (y_score >= thresh).astype(int)
            precision_vals.append(precision_score(y_true, y_pred, zero_division=0))
            recall_vals.append(recall_score(y_true, y_pred, zero_division=0))
        
        ax.plot(np.linspace(0.1, 0.9, 17), precision_vals, label=f'{setup_name} Precision', linestyle='-')
        ax.plot(np.linspace(0.1, 0.9, 17), recall_vals, label=f'{setup_name} Recall', linestyle='--')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'threshold_analysis.png'}")
    
    # Figure 4: Calibration curves
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    for setup_name, score_col in [('Stage 1 (LightGBM)', 'stage1_score'),
                                   ('Borderline Routed', 'final_borderline_routed_score')]:
        y_score = test_df[score_col].values
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy='uniform')
            ax.plot(prob_pred, prob_true, marker='o', label=setup_name)
        except:
            pass
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'calibration_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'calibration_curve.png'}")
    
    # Figure 5: Monthly fraud rate and capture
    if 'application_month' in test_df.columns:
        monthly_data = test_df.groupby('application_month').agg({
            'fraud_label': ['count', 'sum', 'mean'],
            'stage1_score': 'mean',
            'final_borderline_routed_score': 'mean'
        }).reset_index()
        monthly_data.columns = ['month', 'n_samples', 'n_fraud', 'fraud_rate', 'avg_stage1_score', 'avg_routed_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Fraud rate by month
        ax1 = axes[0]
        ax1.bar(range(len(monthly_data)), monthly_data['fraud_rate'], color='crimson', alpha=0.7)
        ax1.set_xticks(range(len(monthly_data)))
        ax1.set_xticklabels(monthly_data['month'], rotation=45, ha='right')
        ax1.set_ylabel('Fraud Rate')
        ax1.set_title('Fraud Rate by Month')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Average score by month
        ax2 = axes[1]
        x = range(len(monthly_data))
        width = 0.35
        ax2.bar([i - width/2 for i in x], monthly_data['avg_stage1_score'], width, label='Stage 1', alpha=0.7)
        ax2.bar([i + width/2 for i in x], monthly_data['avg_routed_score'], width, label='Routed', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(monthly_data['month'], rotation=45, ha='right')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Score by Month')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Sample size by month
        ax3 = axes[2]
        ax3.bar(range(len(monthly_data)), monthly_data['n_samples'], color='steelblue', alpha=0.7)
        ax3.set_xticks(range(len(monthly_data)))
        ax3.set_xticklabels(monthly_data['month'], rotation=45, ha='right')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Samples by Month')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'monthly_stability.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'monthly_stability.png'}")
    
    # Figure 6: Borderline subset comparison
    borderline_df = test_df[test_df['borderline_flag'] == 1]
    if len(borderline_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        setups = ['Stage 1\n(Structured)', 'Combined\n(All Cases)', 'Borderline\nRouted']
        
        # Compute metrics for each setup on borderline
        y_true_bl = borderline_df['fraud_label'].values
        
        metrics_data = []
        for setup_name, score_col, pred_col in [
            ('Stage 1', 'stage1_score', 'stage1_pred'),
            ('Combined', 'combined_all_score', 'combined_all_pred'),
            ('Routed', 'final_borderline_routed_score', 'final_borderline_routed_pred')
        ]:
            y_score = borderline_df[score_col].values
            y_pred = borderline_df[pred_col].values
            
            if len(np.unique(y_true_bl)) > 1:
                roc = roc_auc_score(y_true_bl, y_score)
                pr = average_precision_score(y_true_bl, y_score)
                f1 = f1_score(y_true_bl, y_pred)
            else:
                roc, pr, f1 = 0, 0, 0
            
            metrics_data.append([roc, pr, f1])
        
        metrics_data = np.array(metrics_data)
        
        x = np.arange(len(setups))
        width = 0.25
        
        ax.bar(x - width, metrics_data[:, 0], width, label='ROC-AUC', color='steelblue')
        ax.bar(x, metrics_data[:, 1], width, label='PR-AUC', color='darkorange')
        ax.bar(x + width, metrics_data[:, 2], width, label='F1', color='forestgreen')
        
        ax.set_ylabel('Score')
        ax.set_title('Borderline Cases: Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(setups)
        ax.legend()
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (roc, pr, f1) in enumerate(metrics_data):
            ax.annotate(f'{roc:.3f}', xy=(i - width, roc), xytext=(0, 3),
                       textcoords='offset points', ha='center', fontsize=8)
            ax.annotate(f'{pr:.3f}', xy=(i, pr), xytext=(0, 3),
                       textcoords='offset points', ha='center', fontsize=8)
            ax.annotate(f'{f1:.3f}', xy=(i + width, f1), xytext=(0, 3),
                       textcoords='offset points', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'borderline_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'borderline_comparison.png'}")
    
    # Figure 7: ROC curves comparison
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for setup_name, score_col in [('Stage 1 (LightGBM)', 'stage1_score'),
                                   ('Combined (All)', 'combined_all_score'),
                                   ('Borderline Routed', 'final_borderline_routed_score')]:
        y_score = test_df[score_col].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f'{setup_name} (AUC={auc:.4f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'roc_curves.png'}")


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_outputs(validation_metrics, threshold_df, calibration_df, runtime_df, monthly_df):
    """
    Save all validation outputs.
    
    Args:
        validation_metrics: List of validation metric dictionaries
        threshold_df: Threshold analysis DataFrame
        calibration_df: Calibration metrics DataFrame
        runtime_df: Runtime summary DataFrame
        monthly_df: Monthly stability DataFrame
    """
    print("\nSaving outputs...")
    
    # Validation metrics summary
    metrics_df = pd.DataFrame(validation_metrics)
    metrics_path = TABLES_DIR / "validation_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")
    
    # Threshold analysis
    threshold_path = TABLES_DIR / "threshold_analysis.csv"
    threshold_df.to_csv(threshold_path, index=False)
    print(f"  Saved: {threshold_path}")
    
    # Calibration summary
    calibration_path = TABLES_DIR / "calibration_summary.csv"
    calibration_df.to_csv(calibration_path, index=False)
    print(f"  Saved: {calibration_path}")
    
    # Runtime summary
    runtime_path = TABLES_DIR / "runtime_summary.csv"
    runtime_df.to_csv(runtime_path, index=False)
    print(f"  Saved: {runtime_path}")
    
    # Monthly stability (if available)
    if len(monthly_df) > 0:
        monthly_path = TABLES_DIR / "monthly_stability.csv"
        monthly_df.to_csv(monthly_path, index=False)
        print(f"  Saved: {monthly_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to run Phase 11: Model Validation.
    """
    print("="*70)
    print("PHASE 11: MODEL VALIDATION")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load inputs
    # -------------------------------------------------------------------------
    inputs = load_validation_inputs()
    df = inputs['final_predictions']
    
    # -------------------------------------------------------------------------
    # Step 2: Define setups to evaluate
    # -------------------------------------------------------------------------
    
    # Add logistic regression scores from baseline predictions first
    baseline = inputs['baseline_predictions']
    df = df.merge(
        baseline[['application_id', 'logistic_regression_score', 'logistic_regression_pred']],
        on='application_id',
        how='left'
    )
    
    # Setups with (name, score_col, pred_col)
    # Note: stage1_score in final_predictions is LightGBM score
    setups_full = [
        ('1_logistic_regression', 'logistic_regression_score', 'logistic_regression_pred'),
        ('2_lightgbm_structured', 'stage1_score', 'stage1_pred'),
        ('3_text_only', 'text_only_score', 'text_only_pred'),
        ('4_combined_all', 'combined_all_score', 'combined_all_pred'),
        ('5_borderline_routed', 'final_borderline_routed_score', 'final_borderline_routed_pred'),
    ]
    
    # Setups for threshold/calibration (score_col only)
    setups_score = [
        ('LightGBM (Stage 1)', 'stage1_score'),
        ('Combined (All)', 'combined_all_score'),
        ('Borderline Routed', 'final_borderline_routed_score'),
    ]
    
    # -------------------------------------------------------------------------
    # Step 3: Standard model evaluation
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("11.1 STANDARD MODEL EVALUATION")
    print("="*70)
    
    validation_metrics = []
    
    for split in ['val', 'test']:
        split_df = df[df['split_label'] == split]
        
        for setup_name, score_col, pred_col in setups_full:
            # Skip if column doesn't exist
            if score_col not in split_df.columns:
                continue
            
            # Overall evaluation
            metrics = evaluate_setup(split_df, score_col, pred_col, setup_name, split)
            validation_metrics.append(metrics)
            
            # Borderline-only evaluation
            borderline_mask = split_df['borderline_flag'] == 1
            if borderline_mask.sum() > 0:
                metrics_bl = evaluate_setup(split_df, score_col, pred_col, setup_name, split, borderline_mask)
                validation_metrics.append(metrics_bl)
    
    # Print summary
    print("\nTest Set Overall Results:")
    print("-"*70)
    test_overall = [m for m in validation_metrics if m['split'] == 'test' and m['subset'] == 'overall']
    for m in test_overall:
        print(f"  {m['setup']:<25} ROC-AUC: {m['roc_auc']:.4f}  PR-AUC: {m['pr_auc']:.4f}  F1: {m['f1']:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Borderline-case evaluation
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("11.2 BORDERLINE-CASE EVALUATION")
    print("="*70)
    
    test_df = df[df['split_label'] == 'test']
    
    # Use stage1_score as baseline for comparison (which is LightGBM)
    borderline_results = evaluate_borderline_subset(
        test_df,
        'stage1_score', 'stage1_pred',
        [
            ('LightGBM (Baseline)', 'stage1_score', 'stage1_pred'),
            ('Combined (All)', 'combined_all_score', 'combined_all_pred'),
            ('Borderline Routed', 'final_borderline_routed_score', 'final_borderline_routed_pred'),
        ]
    )
    
    print("\nBorderline Subset Results (Test Set):")
    print("-"*70)
    for m in borderline_results:
        print(f"  {m['setup']:<25} ROC-AUC: {m['roc_auc']:.4f}  Error Reduction: {m['error_reduction']:+.1%}")
    
    # -------------------------------------------------------------------------
    # Step 5: Threshold analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("11.3 THRESHOLD ANALYSIS")
    print("="*70)
    
    threshold_df = run_threshold_analysis(df, setups_score)
    
    print("\nThreshold Analysis (Test Set):")
    print(threshold_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # Step 6: Calibration analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("11.4 CALIBRATION ANALYSIS")
    print("="*70)
    
    calibration_df, calibration_curves = compute_calibration_metrics(df, setups_score)
    
    print("\nCalibration Metrics (Test Set):")
    print(calibration_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # Step 7: Stability over time
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("11.6 STABILITY OVER TIME")
    print("="*70)
    
    monthly_df = evaluate_monthly_stability(
        df,
        [
            ('LightGBM (Stage 1)', 'stage1_score', 'stage1_pred'),
            ('Borderline Routed', 'final_borderline_routed_score', 'final_borderline_routed_pred'),
        ]
    )
    
    if len(monthly_df) > 0:
        print("\nMonthly Stability (Test Set):")
        pivot = monthly_df[monthly_df['setup'] == 'LightGBM (Stage 1)'][['month', 'fraud_rate', 'avg_score', 'capture_rate']]
        print(pivot.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # Step 8: Runtime measurement
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("11.7 RUNTIME / PRACTICALITY")
    print("="*70)
    
    runtime_df = measure_runtime(inputs)
    
    print("\nRuntime Summary:")
    print(runtime_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # Step 9: Create figures
    # -------------------------------------------------------------------------
    create_validation_figures(df, inputs, setups_full)
    
    # -------------------------------------------------------------------------
    # Step 10: Save outputs
    # -------------------------------------------------------------------------
    save_outputs(validation_metrics, threshold_df, calibration_df, runtime_df, monthly_df)
    
    print("\n" + "="*70)
    print("PHASE 11 COMPLETE")
    print("="*70)
    
    return validation_metrics, threshold_df, calibration_df, runtime_df, monthly_df


if __name__ == "__main__":
    results = main()
