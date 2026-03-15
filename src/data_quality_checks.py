"""
data_quality_checks.py

Data quality checks and light cleaning for the Identity Fraud Detection project.
This module provides reusable functions for EDA and data validation.

Usage:
    python src/data_quality_checks.py

Or import functions:
    from src.data_quality_checks import load_dataset, basic_quality_report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Valid values for validation
VALID_FRAUD_TYPES = [
    "legitimate_clean",
    "legitimate_noisy", 
    "synthetic_identity",
    "true_name_fraud",
    "coordinated_attack",
]

VALID_DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Text columns to check
TEXT_COLUMNS = [
    "verification_note",
    "ocr_document_text",
    "address_explanation_text",
    "employment_explanation_text",
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(prefer_parquet: bool = True) -> pd.DataFrame:
    """
    Load the raw dataset, preferring parquet format.
    
    Args:
        prefer_parquet: If True, try parquet first, then CSV
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    parquet_path = DATA_RAW / "applications_prototype.parquet"
    csv_path = DATA_RAW / "applications_prototype.csv"
    
    if prefer_parquet and parquet_path.exists():
        print(f"Loading from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"Loading from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No dataset found. Expected at:\n  {parquet_path}\n  or {csv_path}"
        )
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# QUALITY CHECKS
# =============================================================================

def basic_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a basic quality report for the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Quality metrics
    """
    report = {}
    
    # Row counts
    report["total_rows"] = len(df)
    report["total_columns"] = len(df.columns)
    
    # Duplicate checks
    report["duplicate_rows"] = df.duplicated().sum()
    report["duplicate_application_ids"] = df["application_id"].duplicated().sum()
    
    # Missing values
    missing_counts = df.isnull().sum()
    report["columns_with_missing"] = (missing_counts > 0).sum()
    report["total_missing_values"] = missing_counts.sum()
    
    # Fraud rate
    report["fraud_count"] = df["fraud_label"].sum()
    report["fraud_rate"] = df["fraud_label"].mean() * 100
    
    return report


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of missing values by column.
    
    Args:
        df: Input DataFrame
    
    Returns:
        pd.DataFrame: Missing value summary
    """
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    
    summary = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_pct": missing_pct,
    })
    
    # Only show columns with missing values
    summary = summary[summary["missing_count"] > 0]
    
    if len(summary) == 0:
        print("No missing values found!")
        return pd.DataFrame()
    
    return summary.sort_values("missing_count", ascending=False)


def validate_allowed_values(df: pd.DataFrame) -> dict:
    """
    Validate that categorical columns contain only allowed values.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Validation results
    """
    results = {}
    
    # Check fraud_label
    fraud_label_values = set(df["fraud_label"].unique())
    invalid_fraud_labels = fraud_label_values - {0, 1}
    results["fraud_label_valid"] = len(invalid_fraud_labels) == 0
    results["fraud_label_invalid_values"] = list(invalid_fraud_labels)
    
    # Check fraud_type
    fraud_type_values = set(df["fraud_type"].unique())
    invalid_fraud_types = fraud_type_values - set(VALID_FRAUD_TYPES)
    results["fraud_type_valid"] = len(invalid_fraud_types) == 0
    results["fraud_type_invalid_values"] = list(invalid_fraud_types)
    
    # Check difficulty_level
    difficulty_values = set(df["difficulty_level"].unique())
    invalid_difficulty = difficulty_values - set(VALID_DIFFICULTY_LEVELS)
    results["difficulty_level_valid"] = len(invalid_difficulty) == 0
    results["difficulty_level_invalid_values"] = list(invalid_difficulty)
    
    return results


def summarize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize text field quality.
    
    Args:
        df: Input DataFrame
    
    Returns:
        pd.DataFrame: Text field summary
    """
    summaries = []
    
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            continue
        
        # Count nulls and empty strings
        null_count = df[col].isnull().sum()
        empty_count = (df[col] == "").sum() if df[col].dtype == object else 0
        
        # Text length stats (for non-null values)
        lengths = df[col].dropna().astype(str).str.len()
        
        summaries.append({
            "column": col,
            "null_count": null_count,
            "empty_count": empty_count,
            "min_length": lengths.min() if len(lengths) > 0 else 0,
            "max_length": lengths.max() if len(lengths) > 0 else 0,
            "mean_length": round(lengths.mean(), 1) if len(lengths) > 0 else 0,
        })
    
    return pd.DataFrame(summaries)


def summarize_suspicious_patterns(df: pd.DataFrame) -> dict:
    """
    Check whether fraud signals show expected directional patterns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Pattern analysis results
    """
    results = {}
    
    # Split by fraud label
    legit = df[df["fraud_label"] == 0]
    fraud = df[df["fraud_label"] == 1]
    
    # Device reuse
    results["device_reuse_legit_mean"] = legit["num_prev_apps_same_device_7d"].mean()
    results["device_reuse_fraud_mean"] = fraud["num_prev_apps_same_device_7d"].mean()
    results["device_reuse_higher_in_fraud"] = (
        results["device_reuse_fraud_mean"] > results["device_reuse_legit_mean"]
    )
    
    # Email match score (lower in fraud expected)
    results["email_match_legit_mean"] = legit["name_email_match_score"].mean()
    results["email_match_fraud_mean"] = fraud["name_email_match_score"].mean()
    results["email_match_lower_in_fraud"] = (
        results["email_match_fraud_mean"] < results["email_match_legit_mean"]
    )
    
    # ZIP/IP distance (higher in fraud expected)
    results["zip_ip_dist_legit_mean"] = legit["zip_ip_distance_proxy"].mean()
    results["zip_ip_dist_fraud_mean"] = fraud["zip_ip_distance_proxy"].mean()
    results["zip_ip_dist_higher_in_fraud"] = (
        results["zip_ip_dist_fraud_mean"] > results["zip_ip_dist_legit_mean"]
    )
    
    # Thin file flag (more common in fraud expected)
    results["thin_file_legit_rate"] = legit["thin_file_flag"].mean()
    results["thin_file_fraud_rate"] = fraud["thin_file_flag"].mean()
    results["thin_file_higher_in_fraud"] = (
        results["thin_file_fraud_rate"] > results["thin_file_legit_rate"]
    )
    
    # Months at address (lower in fraud expected)
    results["months_addr_legit_mean"] = legit["months_at_address"].mean()
    results["months_addr_fraud_mean"] = fraud["months_at_address"].mean()
    results["months_addr_lower_in_fraud"] = (
        results["months_addr_fraud_mean"] < results["months_addr_legit_mean"]
    )
    
    # Months at employer (lower in fraud expected)
    results["months_emp_legit_mean"] = legit["months_at_employer"].mean()
    results["months_emp_fraud_mean"] = fraud["months_at_employer"].mean()
    results["months_emp_lower_in_fraud"] = (
        results["months_emp_fraud_mean"] < results["months_emp_legit_mean"]
    )
    
    return results


def summarize_borderline_cases(df: pd.DataFrame) -> dict:
    """
    Analyze borderline/ambiguous cases in the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Borderline case analysis
    """
    results = {}
    
    # Hard cases
    hard_cases = df[df["difficulty_level"] == "hard"]
    results["total_hard_cases"] = len(hard_cases)
    results["hard_cases_pct"] = round(len(hard_cases) / len(df) * 100, 1)
    
    # Hard by fraud label
    results["hard_legit_count"] = len(hard_cases[hard_cases["fraud_label"] == 0])
    results["hard_fraud_count"] = len(hard_cases[hard_cases["fraud_label"] == 1])
    
    # Hard by fraud type
    results["hard_by_fraud_type"] = hard_cases["fraud_type"].value_counts().to_dict()
    
    # Legitimate noisy (borderline legitimate)
    results["legitimate_noisy_count"] = len(df[df["fraud_type"] == "legitimate_noisy"])
    
    # True name fraud hard cases (hardest fraud to detect)
    true_name_hard = df[
        (df["fraud_type"] == "true_name_fraud") & 
        (df["difficulty_level"] == "hard")
    ]
    results["true_name_fraud_hard_count"] = len(true_name_hard)
    
    # Signal score distribution (middle range = ambiguous)
    signal_scores = df["generated_signal_score"]
    middle_range = df[
        (signal_scores >= 0.3) & (signal_scores <= 0.6)
    ]
    results["middle_signal_score_count"] = len(middle_range)
    results["middle_signal_score_pct"] = round(len(middle_range) / len(df) * 100, 1)
    
    return results


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def create_cleaned_dataset(df: pd.DataFrame) -> tuple:
    """
    Apply light cleaning to the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        tuple: (cleaned DataFrame, list of cleaning actions taken)
    """
    df_clean = df.copy()
    actions = []
    
    # 1. Remove exact duplicate rows
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    n_removed = n_before - len(df_clean)
    if n_removed > 0:
        actions.append(f"Removed {n_removed} exact duplicate rows")
    
    # 2. Remove duplicate application_ids (keep first)
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["application_id"], keep="first")
    n_removed = n_before - len(df_clean)
    if n_removed > 0:
        actions.append(f"Removed {n_removed} duplicate application_id rows")
    
    # 3. Normalize whitespace in text columns
    for col in TEXT_COLUMNS:
        if col in df_clean.columns:
            # Strip leading/trailing whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Replace multiple spaces with single space
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
    actions.append("Normalized whitespace in text columns")
    
    # 4. Ensure application_date is datetime-like string
    if "application_date" in df_clean.columns:
        # Verify format is YYYY-MM-DD
        try:
            pd.to_datetime(df_clean["application_date"])
            actions.append("Verified application_date format")
        except Exception as e:
            actions.append(f"WARNING: application_date format issue: {e}")
    
    # 5. Verify application_month alignment
    if "application_date" in df_clean.columns and "application_month" in df_clean.columns:
        # Check that application_month matches application_date
        derived_month = pd.to_datetime(df_clean["application_date"]).dt.strftime("%Y-%m")
        mismatches = (df_clean["application_month"] != derived_month).sum()
        if mismatches > 0:
            df_clean["application_month"] = derived_month
            actions.append(f"Fixed {mismatches} application_month mismatches")
        else:
            actions.append("Verified application_month alignment")
    
    # 6. Ensure numeric columns are numeric
    numeric_cols = [
        "age", "annual_income", "months_at_address", "months_at_employer",
        "application_hour", "num_prev_apps_same_device_7d", 
        "num_prev_apps_same_email_30d", "num_prev_apps_same_phone_30d",
        "num_prev_apps_same_address_30d", "fraud_label", "thin_file_flag",
        "is_free_email_domain", "document_uploaded",
    ]
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    actions.append("Ensured numeric columns are numeric type")
    
    # 7. Handle any NaN in text columns (fill with placeholder if needed)
    for col in TEXT_COLUMNS:
        if col in df_clean.columns:
            null_count = df_clean[col].isnull().sum()
            if null_count > 0:
                df_clean[col] = df_clean[col].fillna("No information available")
                actions.append(f"Filled {null_count} null values in {col}")
    
    if len(actions) == 0:
        actions.append("No cleaning actions needed")
    
    return df_clean, actions


def save_cleaned_outputs(df: pd.DataFrame, output_name: str = "applications_cleaned") -> dict:
    """
    Save the cleaned dataset.
    
    Args:
        df: Cleaned DataFrame
        output_name: Base name for output files
    
    Returns:
        dict: Paths to saved files
    """
    # Ensure output directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Save files
    parquet_path = DATA_PROCESSED / f"{output_name}.parquet"
    csv_path = DATA_PROCESSED / f"{output_name}.csv"
    
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    
    print(f"Saved: {parquet_path}")
    print(f"Saved: {csv_path}")
    
    return {
        "parquet": parquet_path,
        "csv": csv_path,
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_quality_summary(df: pd.DataFrame) -> None:
    """Print a concise quality summary to console."""
    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)
    
    # Basic report
    report = basic_quality_report(df)
    print(f"\nTotal rows: {report['total_rows']}")
    print(f"Total columns: {report['total_columns']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Duplicate application_ids: {report['duplicate_application_ids']}")
    print(f"Columns with missing: {report['columns_with_missing']}")
    print(f"Fraud rate: {report['fraud_rate']:.1f}%")
    
    # Validation
    validation = validate_allowed_values(df)
    print(f"\nValidation:")
    print(f"  fraud_label valid: {validation['fraud_label_valid']}")
    print(f"  fraud_type valid: {validation['fraud_type_valid']}")
    print(f"  difficulty_level valid: {validation['difficulty_level_valid']}")
    
    # Suspicious patterns
    patterns = summarize_suspicious_patterns(df)
    print(f"\nFraud Signal Patterns (expected direction):")
    print(f"  Device reuse higher in fraud: {patterns['device_reuse_higher_in_fraud']}")
    print(f"  Email match lower in fraud: {patterns['email_match_lower_in_fraud']}")
    print(f"  ZIP/IP dist higher in fraud: {patterns['zip_ip_dist_higher_in_fraud']}")
    print(f"  Thin file higher in fraud: {patterns['thin_file_higher_in_fraud']}")
    print(f"  Months at address lower in fraud: {patterns['months_addr_lower_in_fraud']}")
    print(f"  Months at employer lower in fraud: {patterns['months_emp_lower_in_fraud']}")
    
    # Borderline cases
    borderline = summarize_borderline_cases(df)
    print(f"\nBorderline Cases:")
    print(f"  Hard cases: {borderline['total_hard_cases']} ({borderline['hard_cases_pct']}%)")
    print(f"  Legitimate noisy: {borderline['legitimate_noisy_count']}")
    print(f"  True name fraud (hard): {borderline['true_name_fraud_hard_count']}")
    
    print("=" * 60)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run data quality checks and create cleaned dataset."""
    print("=" * 60)
    print("IDENTITY FRAUD DETECTION - DATA QUALITY CHECKS")
    print("=" * 60)
    
    # Load dataset
    df = load_dataset()
    
    # Print quality summary
    print_quality_summary(df)
    
    # Create cleaned dataset
    print("\nApplying light cleaning...")
    df_clean, actions = create_cleaned_dataset(df)
    
    print("\nCleaning actions:")
    for action in actions:
        print(f"  - {action}")
    
    print(f"\nRows before cleaning: {len(df)}")
    print(f"Rows after cleaning: {len(df_clean)}")
    
    # Save outputs
    print("\nSaving cleaned dataset...")
    save_cleaned_outputs(df_clean)
    
    print("\nData quality checks complete!")
    
    return df_clean


if __name__ == "__main__":
    main()
