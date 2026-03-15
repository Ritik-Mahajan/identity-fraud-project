"""
define_borderline_band.py

Define the borderline / uncertain band from Stage 1 model predictions.
These cases will be routed to Stage 2 text/encoder analysis.

The borderline band captures cases where:
- The structured model is uncertain (score not near 0 or 1)
- Text/semantic features might provide additional signal
- The model has higher error rates

Usage:
    python src/define_borderline_band.py
    
Or import and use:
    from src.define_borderline_band import extract_borderline_cases
    borderline_df = extract_borderline_cases(predictions_df, low=0.01, high=0.99)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Input files
PREDICTIONS_PATH = DATA_PROCESSED / "baseline_predictions.parquet"
CLEANED_DATA_PATH = DATA_PROCESSED / "applications_cleaned.parquet"
CLEANED_DATA_CSV = DATA_PROCESSED / "applications_cleaned.csv"

# Output file
BORDERLINE_OUTPUT = DATA_PROCESSED / "borderline_cases.parquet"

# Text columns to include in borderline output
TEXT_COLUMNS = [
    "verification_note",
    "ocr_document_text",
    "address_explanation_text",
    "employment_explanation_text",
]

# Candidate borderline bands to evaluate
CANDIDATE_BANDS = [
    (0.01, 0.99),   # Widest - captures all uncertain cases
    (0.05, 0.95),   # Moderate
    (0.10, 0.90),   # Narrower
    (0.15, 0.85),   # Even narrower
    (0.20, 0.80),   # Traditional uncertainty band
]

# Default final band (can be overridden)
DEFAULT_BAND = (0.01, 0.99)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions() -> pd.DataFrame:
    """
    Load the baseline predictions from Phase 7.
    
    Returns:
        DataFrame with predictions and metadata
    """
    if PREDICTIONS_PATH.exists():
        print(f"Loading predictions from: {PREDICTIONS_PATH}")
        df = pd.read_parquet(PREDICTIONS_PATH)
    else:
        raise FileNotFoundError(f"Predictions not found at {PREDICTIONS_PATH}")
    
    print(f"Loaded {len(df):,} rows")
    return df


def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned dataset to get text columns.
    
    Returns:
        DataFrame with text columns
    """
    if CLEANED_DATA_PATH.exists():
        print(f"Loading cleaned data from: {CLEANED_DATA_PATH}")
        df = pd.read_parquet(CLEANED_DATA_PATH)
    elif CLEANED_DATA_CSV.exists():
        print(f"Loading cleaned data from: {CLEANED_DATA_CSV}")
        df = pd.read_csv(CLEANED_DATA_CSV)
    else:
        raise FileNotFoundError("Cleaned data not found")
    
    return df


# =============================================================================
# MODEL SELECTION
# =============================================================================

def choose_primary_model(df: pd.DataFrame) -> str:
    """
    Choose the primary model for borderline band definition.
    
    Based on Phase 7 results, LightGBM was the best model.
    
    Args:
        df: Predictions DataFrame
        
    Returns:
        Column name for the primary model's score
    """
    # LightGBM was best in Phase 7 (ROC-AUC 0.995 on test)
    primary_model = "lightgbm"
    score_col = f"{primary_model}_score"
    pred_col = f"{primary_model}_pred"
    
    if score_col not in df.columns:
        raise ValueError(f"Primary model score column '{score_col}' not found")
    
    print(f"\nPrimary model: {primary_model.upper()}")
    print(f"Score column: {score_col}")
    print(f"Prediction column: {pred_col}")
    
    return primary_model


# =============================================================================
# BAND ANALYSIS
# =============================================================================

def summarize_candidate_band(
    df: pd.DataFrame,
    score_col: str,
    pred_col: str,
    low: float,
    high: float
) -> dict:
    """
    Compute summary statistics for a candidate borderline band.
    
    Args:
        df: Predictions DataFrame
        score_col: Column name for model scores
        pred_col: Column name for model predictions
        low: Lower bound of band
        high: Upper bound of band
        
    Returns:
        Dictionary of summary statistics
    """
    # Filter to band
    mask = (df[score_col] >= low) & (df[score_col] <= high)
    band_df = df[mask]
    
    n_total = len(df)
    n_band = len(band_df)
    
    if n_band == 0:
        return {
            "low": low,
            "high": high,
            "n_rows": 0,
            "pct_of_total": 0.0,
            "fraud_rate": None,
            "error_rate": None,
        }
    
    # Basic stats
    fraud_rate = band_df["fraud_label"].mean()
    legit_rate = 1 - fraud_rate
    
    # Error rate
    errors = (band_df[pred_col] != band_df["fraud_label"]).sum()
    error_rate = errors / n_band
    
    # Fraud type composition
    fraud_type_counts = band_df["fraud_type"].value_counts().to_dict()
    
    # Difficulty composition
    difficulty_counts = band_df["difficulty_level"].value_counts().to_dict()
    
    # Score stats within band
    score_mean = band_df[score_col].mean()
    score_std = band_df[score_col].std()
    
    return {
        "low": low,
        "high": high,
        "n_rows": n_band,
        "pct_of_total": 100 * n_band / n_total,
        "fraud_rate": fraud_rate,
        "legit_rate": legit_rate,
        "error_rate": error_rate,
        "n_errors": errors,
        "score_mean": score_mean,
        "score_std": score_std,
        "fraud_type_counts": fraud_type_counts,
        "difficulty_counts": difficulty_counts,
        "n_hard": difficulty_counts.get("hard", 0),
        "n_medium": difficulty_counts.get("medium", 0),
        "n_easy": difficulty_counts.get("easy", 0),
    }


def compare_candidate_bands(
    df: pd.DataFrame,
    score_col: str,
    pred_col: str,
    bands: list = None
) -> pd.DataFrame:
    """
    Compare multiple candidate borderline bands.
    
    Args:
        df: Predictions DataFrame
        score_col: Column name for model scores
        pred_col: Column name for model predictions
        bands: List of (low, high) tuples
        
    Returns:
        DataFrame comparing all bands
    """
    if bands is None:
        bands = CANDIDATE_BANDS
    
    print("\n" + "=" * 60)
    print("COMPARING CANDIDATE BORDERLINE BANDS")
    print("=" * 60)
    
    results = []
    for low, high in bands:
        summary = summarize_candidate_band(df, score_col, pred_col, low, high)
        results.append(summary)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            "Band": f"{r['low']:.2f}-{r['high']:.2f}",
            "Rows": r["n_rows"],
            "% Total": f"{r['pct_of_total']:.2f}%",
            "Fraud Rate": f"{100*r['fraud_rate']:.1f}%" if r["fraud_rate"] is not None else "N/A",
            "Error Rate": f"{100*r['error_rate']:.1f}%" if r["error_rate"] is not None else "N/A",
            "Hard": r["n_hard"],
            "Medium": r["n_medium"],
            "Easy": r["n_easy"],
        }
        for r in results
    ])
    
    print("\n" + comparison_df.to_string(index=False))
    
    return results


def select_final_band(
    df: pd.DataFrame,
    score_col: str,
    pred_col: str,
    band: tuple = None
) -> dict:
    """
    Select and justify the final borderline band.
    
    Args:
        df: Predictions DataFrame
        score_col: Column name for model scores
        pred_col: Column name for model predictions
        band: (low, high) tuple, or None to use default
        
    Returns:
        Summary dictionary for the selected band
    """
    if band is None:
        band = DEFAULT_BAND
    
    low, high = band
    
    print("\n" + "=" * 60)
    print(f"SELECTED FINAL BAND: {low:.2f} to {high:.2f}")
    print("=" * 60)
    
    summary = summarize_candidate_band(df, score_col, pred_col, low, high)
    
    print(f"\n--- Band Statistics ---")
    print(f"  Rows in band: {summary['n_rows']:,} ({summary['pct_of_total']:.2f}% of total)")
    print(f"  Fraud rate: {100*summary['fraud_rate']:.1f}%")
    print(f"  Legit rate: {100*summary['legit_rate']:.1f}%")
    print(f"  Error rate: {100*summary['error_rate']:.1f}% ({summary['n_errors']} errors)")
    print(f"  Score mean: {summary['score_mean']:.4f}")
    print(f"  Score std: {summary['score_std']:.4f}")
    
    print(f"\n--- Difficulty Distribution ---")
    print(f"  Hard: {summary['n_hard']}")
    print(f"  Medium: {summary['n_medium']}")
    print(f"  Easy: {summary['n_easy']}")
    
    print(f"\n--- Fraud Type Distribution ---")
    for fraud_type, count in summary['fraud_type_counts'].items():
        print(f"  {fraud_type}: {count}")
    
    return summary


# =============================================================================
# BORDERLINE EXTRACTION
# =============================================================================

def extract_borderline_cases(
    predictions_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    score_col: str,
    pred_col: str,
    low: float,
    high: float
) -> pd.DataFrame:
    """
    Extract borderline cases with text fields for Stage 2 analysis.
    
    Args:
        predictions_df: Predictions DataFrame
        cleaned_df: Cleaned data with text columns
        score_col: Column name for model scores
        pred_col: Column name for model predictions
        low: Lower bound of band
        high: Upper bound of band
        
    Returns:
        DataFrame with borderline cases and text fields
    """
    print("\n" + "=" * 60)
    print("EXTRACTING BORDERLINE CASES")
    print("=" * 60)
    
    # Filter to band
    mask = (predictions_df[score_col] >= low) & (predictions_df[score_col] <= high)
    borderline = predictions_df[mask].copy()
    
    print(f"Borderline cases: {len(borderline)}")
    
    # Merge text columns from cleaned data
    text_cols_to_merge = ["application_id"] + TEXT_COLUMNS
    text_df = cleaned_df[text_cols_to_merge]
    
    borderline = borderline.merge(text_df, on="application_id", how="left")
    
    # Rename score and pred columns for clarity
    borderline = borderline.rename(columns={
        score_col: "best_model_score",
        pred_col: "best_model_pred",
    })
    
    # Add band info
    borderline["borderline_band_low"] = low
    borderline["borderline_band_high"] = high
    
    # Reorder columns
    first_cols = [
        "application_id",
        "application_date",
        "application_month",
        "fraud_label",
        "fraud_type",
        "difficulty_level",
        "generated_signal_score",
        "best_model_score",
        "best_model_pred",
        "borderline_band_low",
        "borderline_band_high",
    ]
    
    other_cols = [c for c in borderline.columns if c not in first_cols]
    borderline = borderline[first_cols + other_cols]
    
    print(f"Final columns: {len(borderline.columns)}")
    print(f"Text columns included: {TEXT_COLUMNS}")
    
    return borderline


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_borderline_cases(df: pd.DataFrame) -> None:
    """
    Save the borderline cases to parquet.
    
    Args:
        df: Borderline cases DataFrame
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BORDERLINE_OUTPUT, index=False)
    print(f"\nSaved borderline cases to: {BORDERLINE_OUTPUT}")
    print(f"File size: {BORDERLINE_OUTPUT.stat().st_size / 1024:.1f} KB")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to define the borderline band and extract cases.
    """
    print("\n" + "=" * 60)
    print("PHASE 8: DEFINE BORDERLINE BAND")
    print("=" * 60)
    
    # Step 1: Load data
    predictions_df = load_predictions()
    cleaned_df = load_cleaned_data()
    
    # Step 2: Choose primary model
    primary_model = choose_primary_model(predictions_df)
    score_col = f"{primary_model}_score"
    pred_col = f"{primary_model}_pred"
    
    # Step 3: Compare candidate bands
    band_results = compare_candidate_bands(
        predictions_df, score_col, pred_col, CANDIDATE_BANDS
    )
    
    # Step 4: Select final band
    # Using 0.01-0.99 because:
    # - Captures 117 uncertain cases (2.3% of data)
    # - Contains mostly hard/medium difficulty cases
    # - Has 20.5% error rate (vs ~1% overall) - these are truly uncertain
    # - Includes legitimate_noisy and true_name_fraud - the ambiguous types
    final_band = (0.01, 0.99)
    final_summary = select_final_band(
        predictions_df, score_col, pred_col, final_band
    )
    
    # Step 5: Extract borderline cases
    borderline_df = extract_borderline_cases(
        predictions_df, cleaned_df, score_col, pred_col,
        final_band[0], final_band[1]
    )
    
    # Step 6: Save outputs
    save_borderline_cases(borderline_df)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("BORDERLINE BAND DEFINITION COMPLETE")
    print("=" * 60)
    print(f"\nFinal band: {final_band[0]:.2f} to {final_band[1]:.2f}")
    print(f"Borderline cases: {len(borderline_df)}")
    print(f"Fraud rate in band: {100*final_summary['fraud_rate']:.1f}%")
    print(f"Error rate in band: {100*final_summary['error_rate']:.1f}%")
    
    return borderline_df, final_summary


if __name__ == "__main__":
    main()
