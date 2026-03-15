"""
feature_engineering.py

Feature engineering module for Stage 1 structured fraud model.
Creates numeric, binary, and engineered features from the cleaned dataset.

This module does NOT:
- Use text embeddings (that's Stage 2)
- Train models (that's Phase 7)
- Do heavy normalization (kept simple for beginners)

Usage:
    python src/feature_engineering.py
    
Or import and use:
    from src.feature_engineering import create_feature_table
    df_features = create_feature_table(df_cleaned)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Input files (prefer parquet, fall back to csv)
INPUT_PARQUET = DATA_PROCESSED / "applications_cleaned.parquet"
INPUT_CSV = DATA_PROCESSED / "applications_cleaned.csv"

# Output files
OUTPUT_PARQUET = DATA_PROCESSED / "structured_features.parquet"
OUTPUT_CSV = DATA_PROCESSED / "structured_features.csv"

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Numeric features to include directly from the cleaned data
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
]

# Binary features to include directly
BINARY_FEATURES = [
    "is_free_email_domain",
    "document_uploaded",
    "thin_file_flag",
]

# Pre-computed feature from data generation
PRECOMPUTED_FEATURES = [
    "name_email_match_score",
]

# Categorical features (will remain as-is, not one-hot encoded)
CATEGORICAL_FEATURES = [
    "state",
    "housing_status",
    "ip_region",
    "employer_industry",
]

# Columns to preserve for later phases (not modeling features)
META_COLUMNS = [
    "application_id",
    "application_date",
    "application_month",
    "fraud_label",
    "fraud_type",
    "difficulty_level",
    "generated_signal_score",
]

# Engineered feature names (created in this module)
ENGINEERED_FEATURES = [
    "income_age_ratio",
    "tenure_min",
    "night_application_flag",
    "high_device_velocity_flag",
    "high_identity_reuse_flag",
]

# =============================================================================
# THRESHOLDS FOR ENGINEERED FEATURES
# =============================================================================

# Night hours: applications submitted between midnight and 5 AM
NIGHT_HOURS = [0, 1, 2, 3, 4, 5]

# High device velocity: 3+ applications from same device in 7 days
HIGH_DEVICE_VELOCITY_THRESHOLD = 3

# High identity reuse: any reuse count >= 2 across email/phone/address
HIGH_IDENTITY_REUSE_THRESHOLD = 2


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the cleaned dataset from Phase 5.
    
    Prefers parquet format for speed, falls back to CSV if not available.
    
    Returns:
        pd.DataFrame: The cleaned application dataset
    """
    if INPUT_PARQUET.exists():
        print(f"Loading from parquet: {INPUT_PARQUET}")
        df = pd.read_parquet(INPUT_PARQUET)
    elif INPUT_CSV.exists():
        print(f"Loading from CSV: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
    else:
        raise FileNotFoundError(
            f"No cleaned dataset found. Expected:\n"
            f"  - {INPUT_PARQUET}\n"
            f"  - {INPUT_CSV}"
        )
    
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# FEATURE CREATION FUNCTIONS
# =============================================================================

def create_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and validate numeric features from the dataset.
    
    These features are already numeric in the cleaned data.
    This function ensures they are properly typed.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with numeric feature columns
    """
    print("\n--- Creating Numeric Features ---")
    
    numeric_df = pd.DataFrame()
    
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required numeric column: {col}")
        
        # Ensure numeric type
        numeric_df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Report any coercion issues
        null_count = numeric_df[col].isna().sum()
        if null_count > 0:
            print(f"  Warning: {col} has {null_count} null values after conversion")
    
    print(f"  Created {len(NUMERIC_FEATURES)} numeric features")
    return numeric_df


def create_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and validate binary features from the dataset.
    
    Ensures all binary features contain only 0 or 1.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with binary feature columns
    """
    print("\n--- Creating Binary Features ---")
    
    binary_df = pd.DataFrame()
    
    for col in BINARY_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required binary column: {col}")
        
        # Convert to int and validate
        binary_df[col] = df[col].astype(int)
        
        # Check for valid binary values
        unique_vals = binary_df[col].unique()
        invalid_vals = set(unique_vals) - {0, 1}
        if invalid_vals:
            print(f"  Warning: {col} contains non-binary values: {invalid_vals}")
    
    print(f"  Created {len(BINARY_FEATURES)} binary features")
    return binary_df


def create_precomputed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pre-computed features that were created during data generation.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with pre-computed feature columns
    """
    print("\n--- Extracting Pre-computed Features ---")
    
    precomputed_df = pd.DataFrame()
    
    for col in PRECOMPUTED_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required pre-computed column: {col}")
        
        precomputed_df[col] = pd.to_numeric(df[col], errors="coerce")
    
    print(f"  Extracted {len(PRECOMPUTED_FEATURES)} pre-computed features")
    return precomputed_df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new engineered features from existing columns.
    
    Features created:
    - income_age_ratio: annual_income / age
    - tenure_min: min(months_at_address, months_at_employer)
    - night_application_flag: 1 if application_hour in [0,1,2,3,4,5]
    - high_device_velocity_flag: 1 if num_prev_apps_same_device_7d >= 3
    - high_identity_reuse_flag: 1 if any reuse count >= 2
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered feature columns
    """
    print("\n--- Creating Engineered Features ---")
    
    engineered_df = pd.DataFrame()
    
    # 1. Income-to-Age Ratio
    # Measures income relative to age (higher might indicate unrealistic claims)
    # Handle divide-by-zero safely (though age should always be valid)
    engineered_df["income_age_ratio"] = np.where(
        df["age"] > 0,
        df["annual_income"] / df["age"],
        0  # Default to 0 if age is somehow 0 or negative
    )
    print(f"  income_age_ratio: annual_income / age")
    
    # 2. Minimum Tenure
    # Lower tenure at both address and employer may indicate higher risk
    engineered_df["tenure_min"] = df[["months_at_address", "months_at_employer"]].min(axis=1)
    print(f"  tenure_min: min(months_at_address, months_at_employer)")
    
    # 3. Night Application Flag
    # Applications submitted late night / early morning (hours 0-5)
    engineered_df["night_application_flag"] = df["application_hour"].isin(NIGHT_HOURS).astype(int)
    print(f"  night_application_flag: 1 if hour in {NIGHT_HOURS}")
    
    # 4. High Device Velocity Flag
    # Multiple applications from same device in short period
    engineered_df["high_device_velocity_flag"] = (
        df["num_prev_apps_same_device_7d"] >= HIGH_DEVICE_VELOCITY_THRESHOLD
    ).astype(int)
    print(f"  high_device_velocity_flag: 1 if device_7d >= {HIGH_DEVICE_VELOCITY_THRESHOLD}")
    
    # 5. High Identity Reuse Flag
    # Any of email/phone/address has high reuse
    # This captures coordinated attacks and synthetic identity patterns
    reuse_cols = [
        "num_prev_apps_same_email_30d",
        "num_prev_apps_same_phone_30d",
        "num_prev_apps_same_address_30d",
    ]
    max_reuse = df[reuse_cols].max(axis=1)
    engineered_df["high_identity_reuse_flag"] = (
        max_reuse >= HIGH_IDENTITY_REUSE_THRESHOLD
    ).astype(int)
    print(f"  high_identity_reuse_flag: 1 if any reuse >= {HIGH_IDENTITY_REUSE_THRESHOLD}")
    
    print(f"  Created {len(ENGINEERED_FEATURES)} engineered features")
    return engineered_df


def extract_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract categorical features without encoding.
    
    These will remain as string/category columns.
    One-hot encoding can be done later in the modeling pipeline.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with categorical feature columns
    """
    print("\n--- Extracting Categorical Features ---")
    
    categorical_df = pd.DataFrame()
    
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required categorical column: {col}")
        
        categorical_df[col] = df[col].astype(str)
        n_unique = categorical_df[col].nunique()
        print(f"  {col}: {n_unique} unique values")
    
    print(f"  Extracted {len(CATEGORICAL_FEATURES)} categorical features")
    return categorical_df


def extract_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract metadata columns needed for later phases.
    
    These are NOT modeling features, but are preserved for:
    - Joining results back to original data
    - Time-based train/test splits
    - Evaluation by fraud type and difficulty
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with metadata columns
    """
    print("\n--- Extracting Metadata Columns ---")
    
    meta_df = pd.DataFrame()
    
    for col in META_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required metadata column: {col}")
        
        meta_df[col] = df[col]
    
    print(f"  Extracted {len(META_COLUMNS)} metadata columns")
    return meta_df


# =============================================================================
# FEATURE TABLE ASSEMBLY
# =============================================================================

def create_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the complete feature table for Stage 1 modeling.
    
    Combines:
    - Metadata columns (for joining and evaluation)
    - Numeric features
    - Binary features
    - Pre-computed features
    - Engineered features
    - Categorical features
    
    Args:
        df: Cleaned input DataFrame
        
    Returns:
        Complete feature table ready for modeling
    """
    print("\n" + "=" * 60)
    print("CREATING FEATURE TABLE")
    print("=" * 60)
    
    # Create each feature group
    meta_df = extract_meta_columns(df)
    numeric_df = create_numeric_features(df)
    binary_df = create_binary_features(df)
    precomputed_df = create_precomputed_features(df)
    engineered_df = create_engineered_features(df)
    categorical_df = extract_categorical_features(df)
    
    # Combine all features
    feature_table = pd.concat([
        meta_df,
        numeric_df,
        binary_df,
        precomputed_df,
        engineered_df,
        categorical_df,
    ], axis=1)
    
    print(f"\n--- Feature Table Summary ---")
    print(f"  Total rows: {len(feature_table):,}")
    print(f"  Total columns: {len(feature_table.columns)}")
    print(f"    - Metadata: {len(META_COLUMNS)}")
    print(f"    - Numeric: {len(NUMERIC_FEATURES)}")
    print(f"    - Binary: {len(BINARY_FEATURES)}")
    print(f"    - Pre-computed: {len(PRECOMPUTED_FEATURES)}")
    print(f"    - Engineered: {len(ENGINEERED_FEATURES)}")
    print(f"    - Categorical: {len(CATEGORICAL_FEATURES)}")
    
    return feature_table


# =============================================================================
# QUALITY CHECKS
# =============================================================================

def run_feature_quality_checks(df: pd.DataFrame, original_row_count: int) -> bool:
    """
    Run quality checks on the feature table before saving.
    
    Checks:
    1. Row count matches original
    2. application_id is unique
    3. fraud_label exists and is binary
    4. Engineered binary flags are 0/1
    5. No all-null feature columns
    
    Args:
        df: Feature table to check
        original_row_count: Expected number of rows
        
    Returns:
        True if all checks pass, raises error otherwise
    """
    print("\n" + "=" * 60)
    print("RUNNING QUALITY CHECKS")
    print("=" * 60)
    
    all_passed = True
    
    # Check 1: Row count
    if len(df) != original_row_count:
        print(f"  FAIL: Row count mismatch. Expected {original_row_count}, got {len(df)}")
        all_passed = False
    else:
        print(f"  PASS: Row count matches ({len(df):,} rows)")
    
    # Check 2: Unique application_id
    if df["application_id"].nunique() != len(df):
        print(f"  FAIL: application_id is not unique")
        all_passed = False
    else:
        print(f"  PASS: application_id is unique")
    
    # Check 3: fraud_label exists and is binary
    if "fraud_label" not in df.columns:
        print(f"  FAIL: fraud_label column missing")
        all_passed = False
    else:
        fraud_values = set(df["fraud_label"].unique())
        if fraud_values - {0, 1}:
            print(f"  FAIL: fraud_label contains non-binary values: {fraud_values}")
            all_passed = False
        else:
            print(f"  PASS: fraud_label is binary (0/1)")
    
    # Check 4: Engineered binary flags are 0/1
    binary_engineered = [
        "night_application_flag",
        "high_device_velocity_flag",
        "high_identity_reuse_flag",
    ]
    for col in binary_engineered:
        if col in df.columns:
            unique_vals = set(df[col].unique())
            if unique_vals - {0, 1}:
                print(f"  FAIL: {col} contains non-binary values: {unique_vals}")
                all_passed = False
            else:
                print(f"  PASS: {col} is binary (0/1)")
    
    # Check 5: No all-null columns
    all_null_cols = df.columns[df.isna().all()].tolist()
    if all_null_cols:
        print(f"  FAIL: All-null columns found: {all_null_cols}")
        all_passed = False
    else:
        print(f"  PASS: No all-null columns")
    
    # Check 6: Missingness summary
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        print(f"\n  WARNING: Columns with missing values:")
        for col, count in cols_with_missing.items():
            print(f"    - {col}: {count} ({100*count/len(df):.2f}%)")
    else:
        print(f"  PASS: No missing values in any column")
    
    if all_passed:
        print("\n  All quality checks PASSED")
    else:
        raise ValueError("Quality checks failed. See above for details.")
    
    return all_passed


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_feature_table(df: pd.DataFrame) -> None:
    """
    Save the feature table to parquet and CSV formats.
    
    Args:
        df: Feature table to save
    """
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    # Ensure output directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Save parquet (preferred for speed and type preservation)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  Saved: {OUTPUT_PARQUET}")
    
    # Save CSV (for easy inspection)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV}")
    
    # Print file sizes
    parquet_size = OUTPUT_PARQUET.stat().st_size / 1024
    csv_size = OUTPUT_CSV.stat().st_size / 1024
    print(f"\n  File sizes:")
    print(f"    - Parquet: {parquet_size:.1f} KB")
    print(f"    - CSV: {csv_size:.1f} KB")


# =============================================================================
# HELPER FUNCTIONS FOR ANALYSIS
# =============================================================================

def get_feature_columns() -> list:
    """
    Get the list of all modeling feature columns (excludes metadata).
    
    Returns:
        List of feature column names
    """
    return (
        NUMERIC_FEATURES +
        BINARY_FEATURES +
        PRECOMPUTED_FEATURES +
        ENGINEERED_FEATURES +
        CATEGORICAL_FEATURES
    )


def get_numeric_feature_columns() -> list:
    """
    Get numeric features only (for models that need numeric input).
    
    Returns:
        List of numeric feature column names
    """
    return (
        NUMERIC_FEATURES +
        BINARY_FEATURES +
        PRECOMPUTED_FEATURES +
        ENGINEERED_FEATURES
    )


def summarize_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of engineered features.
    
    Args:
        df: Feature table
        
    Returns:
        Summary DataFrame with statistics
    """
    summary_data = []
    
    for col in ENGINEERED_FEATURES:
        if col in df.columns:
            stats = {
                "feature": col,
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "null_count": df[col].isna().sum(),
            }
            
            # For binary features, add rate
            if df[col].isin([0, 1]).all():
                stats["rate_1"] = df[col].mean()
            
            summary_data.append(stats)
    
    return pd.DataFrame(summary_data)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to run the feature engineering pipeline.
    
    Steps:
    1. Load cleaned dataset
    2. Create feature table
    3. Run quality checks
    4. Save outputs
    """
    print("\n" + "=" * 60)
    print("PHASE 6: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Step 1: Load data
    df_cleaned = load_cleaned_dataset()
    original_row_count = len(df_cleaned)
    
    # Step 2: Create feature table
    df_features = create_feature_table(df_cleaned)
    
    # Step 3: Run quality checks
    run_feature_quality_checks(df_features, original_row_count)
    
    # Step 4: Save outputs
    save_feature_table(df_features)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"\nFeature table created with:")
    print(f"  - {len(df_features):,} rows")
    print(f"  - {len(df_features.columns)} columns")
    print(f"  - {len(get_feature_columns())} modeling features")
    print(f"  - {len(META_COLUMNS)} metadata columns")
    
    # Show engineered feature summary
    print("\n--- Engineered Feature Summary ---")
    summary = summarize_engineered_features(df_features)
    print(summary.to_string(index=False))
    
    return df_features


if __name__ == "__main__":
    main()
