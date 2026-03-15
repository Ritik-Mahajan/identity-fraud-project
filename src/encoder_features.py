"""
encoder_features.py

Create text-based encoder features for Stage 2 fraud detection.
Uses a pretrained sentence-transformer model (all-MiniLM-L6-v2) to compute
semantic similarity between application fields and text evidence.

This module does NOT:
- Train or fine-tune any models
- Use heavy deep learning architectures
- Require GPU (runs on CPU)

Features created:
1. application_ocr_similarity - Similarity between claimed identity and OCR text
2. employment_consistency_score - Similarity between employer info and explanation
3. address_consistency_score - Similarity between address info and explanation
4. Text length features
5. Suspicious keyword features

Usage:
    python src/encoder_features.py
    
Or import and use:
    from src.encoder_features import create_text_feature_table
    features_df = create_text_feature_table(borderline_df, cleaned_df)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Input files
BORDERLINE_PATH = DATA_PROCESSED / "borderline_cases.parquet"
CLEANED_PATH = DATA_PROCESSED / "applications_cleaned.parquet"
CLEANED_CSV = DATA_PROCESSED / "applications_cleaned.csv"

# Output files
OUTPUT_PARQUET = DATA_PROCESSED / "text_encoder_features.parquet"
OUTPUT_CSV = DATA_PROCESSED / "text_encoder_features.csv"

# Encoder model
ENCODER_MODEL_NAME = "all-MiniLM-L6-v2"

# Suspicious keywords for fraud detection
SUSPICIOUS_KEYWORDS = [
    "unable",
    "mismatch",
    "inconsistent",
    "low quality",
    "unreadable",
    "multiple applicants",
    "reused",
    "unverifiable",
    "discrepancy",
    "suspicious",
    "differs",
    "incomplete",
    "not match",
    "cannot verify",
    "no record",
]

# Identity fields for building reference texts
IDENTITY_FIELDS = [
    "claimed_first_name",
    "claimed_last_name",
    "address_line",
    "city",
    "state",
    "zip_code",
    "employer_name",
]

ADDRESS_FIELDS = [
    "address_line",
    "city",
    "state",
    "zip_code",
]

EMPLOYER_FIELDS = [
    "employer_name",
    "employer_industry",
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_borderline_dataset() -> pd.DataFrame:
    """
    Load the borderline cases from Phase 8.
    
    Returns:
        DataFrame with borderline cases and text fields
    """
    if BORDERLINE_PATH.exists():
        print(f"Loading borderline cases from: {BORDERLINE_PATH}")
        df = pd.read_parquet(BORDERLINE_PATH)
    else:
        raise FileNotFoundError(f"Borderline cases not found at {BORDERLINE_PATH}")
    
    print(f"Loaded {len(df):,} borderline cases")
    return df


def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the cleaned dataset to get identity fields.
    
    Returns:
        DataFrame with all application fields
    """
    if CLEANED_PATH.exists():
        print(f"Loading cleaned data from: {CLEANED_PATH}")
        df = pd.read_parquet(CLEANED_PATH)
    elif CLEANED_CSV.exists():
        print(f"Loading cleaned data from: {CLEANED_CSV}")
        df = pd.read_csv(CLEANED_CSV)
    else:
        raise FileNotFoundError("Cleaned data not found")
    
    return df


def merge_identity_fields(borderline_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge identity fields from cleaned data into borderline cases.
    
    Args:
        borderline_df: Borderline cases DataFrame
        cleaned_df: Full cleaned dataset
        
    Returns:
        Borderline DataFrame with identity fields added
    """
    # Get identity columns plus application_id for merging
    cols_to_merge = ["application_id"] + IDENTITY_FIELDS + ["employer_industry"]
    cols_available = [c for c in cols_to_merge if c in cleaned_df.columns]
    
    identity_df = cleaned_df[cols_available]
    
    # Merge
    merged = borderline_df.merge(identity_df, on="application_id", how="left")
    
    print(f"Merged identity fields: {len(cols_available) - 1} columns")
    return merged


# =============================================================================
# ENCODER MODEL
# =============================================================================

def load_encoder_model() -> SentenceTransformer:
    """
    Load the pretrained sentence-transformer model.
    
    Uses all-MiniLM-L6-v2 which is:
    - Small (80MB)
    - Fast
    - Good quality embeddings
    - Runs on CPU
    
    Returns:
        SentenceTransformer model
    """
    import os
    
    # Use local cache directory to avoid permission issues
    cache_dir = PROJECT_ROOT / ".model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    
    print(f"\nLoading encoder model: {ENCODER_MODEL_NAME}")
    print(f"Cache directory: {cache_dir}")
    
    model = SentenceTransformer(ENCODER_MODEL_NAME, cache_folder=str(cache_dir))
    print(f"Model loaded successfully")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# =============================================================================
# TEXT BUILDING FUNCTIONS
# =============================================================================

def build_application_identity_text(row: pd.Series) -> str:
    """
    Build a compact text string from application identity fields.
    
    This represents what the applicant claimed on their application.
    
    Args:
        row: DataFrame row with identity fields
        
    Returns:
        Formatted identity text string
    """
    parts = []
    
    # Name
    first_name = str(row.get("claimed_first_name", "")).strip()
    last_name = str(row.get("claimed_last_name", "")).strip()
    if first_name and last_name:
        parts.append(f"Name {first_name} {last_name}")
    
    # Address
    address = str(row.get("address_line", "")).strip()
    city = str(row.get("city", "")).strip()
    state = str(row.get("state", "")).strip()
    zip_code = str(row.get("zip_code", "")).strip()
    if address:
        parts.append(f"Address {address} {city} {state} {zip_code}")
    
    # Employer
    employer = str(row.get("employer_name", "")).strip()
    if employer:
        parts.append(f"Employer {employer}")
    
    return " ".join(parts)


def build_address_reference_text(row: pd.Series) -> str:
    """
    Build a compact address reference text.
    
    Args:
        row: DataFrame row with address fields
        
    Returns:
        Formatted address text string
    """
    address = str(row.get("address_line", "")).strip()
    city = str(row.get("city", "")).strip()
    state = str(row.get("state", "")).strip()
    zip_code = str(row.get("zip_code", "")).strip()
    
    return f"Address {address} {city} {state} {zip_code}"


def build_employer_reference_text(row: pd.Series) -> str:
    """
    Build a compact employer reference text.
    
    Args:
        row: DataFrame row with employer fields
        
    Returns:
        Formatted employer text string
    """
    employer = str(row.get("employer_name", "")).strip()
    industry = str(row.get("employer_industry", "")).strip()
    
    if industry and industry != "nan":
        return f"Employer {employer} in {industry} industry"
    return f"Employer {employer}"


# =============================================================================
# SIMILARITY COMPUTATION
# =============================================================================

def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    # Reshape for sklearn
    e1 = embedding1.reshape(1, -1)
    e2 = embedding2.reshape(1, -1)
    
    similarity = cosine_similarity(e1, e2)[0][0]
    
    # Ensure in [0, 1] range
    return max(0.0, min(1.0, similarity))


def compute_batch_similarities(
    model: SentenceTransformer,
    reference_texts: list,
    comparison_texts: list
) -> np.ndarray:
    """
    Compute similarities between pairs of texts efficiently.
    
    Args:
        model: SentenceTransformer model
        reference_texts: List of reference text strings
        comparison_texts: List of comparison text strings
        
    Returns:
        Array of similarity scores
    """
    print(f"  Encoding {len(reference_texts)} text pairs...")
    
    # Encode all texts in batches
    ref_embeddings = model.encode(reference_texts, show_progress_bar=False)
    comp_embeddings = model.encode(comparison_texts, show_progress_bar=False)
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(reference_texts)):
        sim = compute_cosine_similarity(ref_embeddings[i], comp_embeddings[i])
        similarities.append(sim)
    
    return np.array(similarities)


# =============================================================================
# FEATURE CREATION FUNCTIONS
# =============================================================================

def create_similarity_features(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """
    Create semantic similarity features using the encoder model.
    
    Features:
    - application_ocr_similarity: Identity text vs OCR document text
    - employment_consistency_score: Employer text vs employment explanation
    - address_consistency_score: Address text vs address explanation
    
    Args:
        df: DataFrame with text fields and identity fields
        model: SentenceTransformer model
        
    Returns:
        DataFrame with similarity features added
    """
    print("\n--- Creating Similarity Features ---")
    
    df = df.copy()
    
    # 1. Application-OCR Similarity
    print("Computing application_ocr_similarity...")
    identity_texts = df.apply(build_application_identity_text, axis=1).tolist()
    ocr_texts = df["ocr_document_text"].fillna("").astype(str).tolist()
    df["application_ocr_similarity"] = compute_batch_similarities(model, identity_texts, ocr_texts)
    
    # 2. Employment Consistency Score
    print("Computing employment_consistency_score...")
    employer_texts = df.apply(build_employer_reference_text, axis=1).tolist()
    employment_explanations = df["employment_explanation_text"].fillna("").astype(str).tolist()
    df["employment_consistency_score"] = compute_batch_similarities(model, employer_texts, employment_explanations)
    
    # 3. Address Consistency Score
    print("Computing address_consistency_score...")
    address_texts = df.apply(build_address_reference_text, axis=1).tolist()
    address_explanations = df["address_explanation_text"].fillna("").astype(str).tolist()
    df["address_consistency_score"] = compute_batch_similarities(model, address_texts, address_explanations)
    
    print("Similarity features created")
    return df


def create_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create text length features.
    
    Args:
        df: DataFrame with text fields
        
    Returns:
        DataFrame with length features added
    """
    print("\n--- Creating Text Length Features ---")
    
    df = df.copy()
    
    # Text length features
    df["verification_note_length"] = df["verification_note"].fillna("").astype(str).str.len()
    df["ocr_text_length"] = df["ocr_document_text"].fillna("").astype(str).str.len()
    df["address_explanation_length"] = df["address_explanation_text"].fillna("").astype(str).str.len()
    df["employment_explanation_length"] = df["employment_explanation_text"].fillna("").astype(str).str.len()
    
    print(f"Created 4 text length features")
    return df


def count_suspicious_keywords(text: str, keywords: list = None) -> int:
    """
    Count suspicious keywords in a text string.
    
    Args:
        text: Text to search
        keywords: List of keywords (uses default if None)
        
    Returns:
        Count of keyword occurrences
    """
    if keywords is None:
        keywords = SUSPICIOUS_KEYWORDS
    
    text_lower = str(text).lower()
    count = 0
    
    for keyword in keywords:
        if keyword.lower() in text_lower:
            count += 1
    
    return count


def has_high_risk_keyword(text: str, keywords: list = None) -> int:
    """
    Check if text contains any high-risk keyword.
    
    Args:
        text: Text to search
        keywords: List of keywords
        
    Returns:
        1 if any keyword found, 0 otherwise
    """
    return 1 if count_suspicious_keywords(text, keywords) > 0 else 0


def create_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create suspicious keyword features.
    
    Args:
        df: DataFrame with text fields
        
    Returns:
        DataFrame with keyword features added
    """
    print("\n--- Creating Keyword Features ---")
    print(f"Using {len(SUSPICIOUS_KEYWORDS)} suspicious keywords")
    
    df = df.copy()
    
    # Keyword counts
    df["suspicious_keyword_count_verification"] = df["verification_note"].apply(
        lambda x: count_suspicious_keywords(x)
    )
    df["suspicious_keyword_count_ocr"] = df["ocr_document_text"].apply(
        lambda x: count_suspicious_keywords(x)
    )
    
    # Total keyword count
    df["suspicious_keyword_count_total"] = (
        df["suspicious_keyword_count_verification"] + 
        df["suspicious_keyword_count_ocr"]
    )
    
    # Binary flags
    df["note_has_high_risk_keyword_flag"] = df["verification_note"].apply(
        lambda x: has_high_risk_keyword(x)
    )
    df["ocr_has_high_risk_keyword_flag"] = df["ocr_document_text"].apply(
        lambda x: has_high_risk_keyword(x)
    )
    
    print(f"Created 5 keyword features")
    return df


# =============================================================================
# MAIN FEATURE TABLE CREATION
# =============================================================================

def create_text_feature_table(
    borderline_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    model: SentenceTransformer = None
) -> pd.DataFrame:
    """
    Create the complete text feature table for Stage 2.
    
    Args:
        borderline_df: Borderline cases from Phase 8
        cleaned_df: Full cleaned dataset with identity fields
        model: SentenceTransformer model (loaded if None)
        
    Returns:
        DataFrame with all text features
    """
    print("\n" + "=" * 60)
    print("CREATING TEXT FEATURE TABLE")
    print("=" * 60)
    
    # Load model if not provided
    if model is None:
        model = load_encoder_model()
    
    # Merge identity fields
    df = merge_identity_fields(borderline_df, cleaned_df)
    
    # Create features
    df = create_similarity_features(df, model)
    df = create_text_length_features(df)
    df = create_keyword_features(df)
    
    # Add borderline flag
    df["borderline_flag"] = 1
    
    # Select and order columns for output
    meta_cols = [
        "application_id",
        "application_date",
        "application_month",
        "fraud_label",
        "fraud_type",
        "difficulty_level",
        "generated_signal_score",
        "best_model_score",
        "best_model_pred",
        "borderline_flag",
        "split_label",
    ]
    
    similarity_cols = [
        "application_ocr_similarity",
        "employment_consistency_score",
        "address_consistency_score",
    ]
    
    length_cols = [
        "verification_note_length",
        "ocr_text_length",
        "address_explanation_length",
        "employment_explanation_length",
    ]
    
    keyword_cols = [
        "suspicious_keyword_count_verification",
        "suspicious_keyword_count_ocr",
        "suspicious_keyword_count_total",
        "note_has_high_risk_keyword_flag",
        "ocr_has_high_risk_keyword_flag",
    ]
    
    # Keep text columns for reference
    text_cols = [
        "verification_note",
        "ocr_document_text",
        "address_explanation_text",
        "employment_explanation_text",
    ]
    
    # Build final column order
    all_cols = meta_cols + similarity_cols + length_cols + keyword_cols + text_cols
    available_cols = [c for c in all_cols if c in df.columns]
    
    output_df = df[available_cols]
    
    print(f"\n--- Feature Table Summary ---")
    print(f"  Rows: {len(output_df):,}")
    print(f"  Columns: {len(output_df.columns)}")
    print(f"  Similarity features: {len(similarity_cols)}")
    print(f"  Length features: {len(length_cols)}")
    print(f"  Keyword features: {len(keyword_cols)}")
    
    return output_df


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(df: pd.DataFrame) -> None:
    """
    Save the text feature table to parquet and CSV.
    
    Args:
        df: Text feature DataFrame
    """
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Save parquet
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved: {OUTPUT_PARQUET}")
    
    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    
    # Print file sizes
    parquet_size = OUTPUT_PARQUET.stat().st_size / 1024
    csv_size = OUTPUT_CSV.stat().st_size / 1024
    print(f"\nFile sizes:")
    print(f"  Parquet: {parquet_size:.1f} KB")
    print(f"  CSV: {csv_size:.1f} KB")


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def summarize_features_by_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize feature means by fraud label.
    
    Args:
        df: Text feature DataFrame
        
    Returns:
        Summary DataFrame
    """
    feature_cols = [
        "application_ocr_similarity",
        "employment_consistency_score",
        "address_consistency_score",
        "verification_note_length",
        "ocr_text_length",
        "suspicious_keyword_count_verification",
        "suspicious_keyword_count_ocr",
        "suspicious_keyword_count_total",
        "note_has_high_risk_keyword_flag",
        "ocr_has_high_risk_keyword_flag",
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    
    summary = df.groupby("fraud_label")[available_cols].mean().T
    summary.columns = ["Legitimate (0)", "Fraud (1)"]
    summary["Diff"] = summary["Fraud (1)"] - summary["Legitimate (0)"]
    
    return summary


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to create text encoder features.
    """
    print("\n" + "=" * 60)
    print("PHASE 9: ENCODER/TEXT FEATURES")
    print("=" * 60)
    
    # Step 1: Load data
    borderline_df = load_borderline_dataset()
    cleaned_df = load_cleaned_dataset()
    
    # Step 2: Load encoder model
    model = load_encoder_model()
    
    # Step 3: Create feature table
    features_df = create_text_feature_table(borderline_df, cleaned_df, model)
    
    # Step 4: Save outputs
    save_outputs(features_df)
    
    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY BY FRAUD LABEL")
    print("=" * 60)
    summary = summarize_features_by_label(features_df)
    print(summary.round(3).to_string())
    
    print("\n" + "=" * 60)
    print("ENCODER FEATURES COMPLETE")
    print("=" * 60)
    print(f"\nCreated {len(features_df)} rows with text features")
    print(f"Output: {OUTPUT_PARQUET}")
    
    return features_df


if __name__ == "__main__":
    main()
