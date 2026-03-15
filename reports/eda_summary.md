# EDA Summary Report

## Overview

This report summarizes the exploratory data analysis (EDA) performed on the synthetic fraud detection dataset in Phase 5.

## Input Data

| Attribute | Value |
|-----------|-------|
| Input file | `data/raw/applications_prototype.parquet` |
| Total rows (before cleaning) | 5,000 |
| Total columns | 41 |

## Output Files

| File | Path |
|------|------|
| Cleaned dataset (Parquet) | `data/processed/applications_cleaned.parquet` |
| Cleaned dataset (CSV) | `data/processed/applications_cleaned.csv` |
| EDA notebook | `notebooks/03_eda.ipynb` |
| Quality checks script | `src/data_quality_checks.py` |

## Data Quality Summary

### Row-Level Checks

| Check | Result |
|-------|--------|
| Total rows | 5,000 |
| Duplicate rows | 0 |
| Duplicate application_ids | 0 |
| Columns with missing values | 0 |
| Total missing values | 0 |

**Assessment**: Dataset is complete with no duplicates or missing values.

### Value Validation

| Field | Valid | Invalid Values |
|-------|-------|----------------|
| fraud_label | ✓ | None (only 0/1) |
| fraud_type | ✓ | None (all allowed values) |
| difficulty_level | ✓ | None (all allowed values) |

**Assessment**: All categorical fields contain only valid values.

## Fraud Distribution

### Fraud Rate

| Category | Count | Percentage |
|----------|-------|------------|
| Legitimate (0) | 4,400 | 88.0% |
| Fraud (1) | 600 | 12.0% |

**Assessment**: Fraud rate matches the ~12% target from Phase 2 design.

### Fraud Type Distribution

| Fraud Type | Count | Percentage |
|------------|-------|------------|
| legitimate_clean | 3,500 | 70.0% |
| legitimate_noisy | 900 | 18.0% |
| synthetic_identity | 250 | 5.0% |
| true_name_fraud | 200 | 4.0% |
| coordinated_attack | 150 | 3.0% |

**Assessment**: Distribution matches Phase 2 design targets.

### Difficulty Distribution

| Difficulty | Count | Percentage |
|------------|-------|------------|
| easy | ~2,652 | ~53% |
| medium | ~1,590 | ~32% |
| hard | ~758 | ~15% |

## Realism Checks

### Age

| Statistic | Value |
|-----------|-------|
| Min | 18 |
| Max | 75 |
| Mean | ~38 |
| Median | ~37 |
| Unrealistic ages | 0 |

**Assessment**: Ages are within realistic adult range (18-75).

### Dates

| Statistic | Value |
|-----------|-------|
| Min date | 2024-01-01 |
| Max date | 2024-12-31 |
| Unique months | 12 |

**Assessment**: Dates span the full 12-month range as designed.

### Tenure and Income

| Field | Min | Max | Mean |
|-------|-----|-----|------|
| months_at_address | 0 | ~120 | ~40 |
| months_at_employer | 0 | ~96 | ~30 |
| annual_income | ~22,000 | ~200,000 | ~70,000 |

**Assessment**: Tenure and income values appear realistic.

## Text Field Quality

| Field | Null Count | Empty Count | Mean Length |
|-------|------------|-------------|-------------|
| verification_note | 0 | 0 | ~55 chars |
| ocr_document_text | 0 | 0 | ~65 chars |
| address_explanation_text | 0 | 0 | ~60 chars |
| employment_explanation_text | 0 | 0 | ~50 chars |

**Assessment**: All text fields are populated with meaningful content.

## Fraud Signal Pattern Analysis

This is a critical check to ensure the synthetic data has the expected fraud patterns.

| Signal | Legitimate Mean | Fraud Mean | Expected Direction | Correct |
|--------|-----------------|------------|-------------------|---------|
| Device Reuse (7d) | 0.05 | 1.85 | Higher in Fraud | ✓ |
| Name/Email Match | 0.58 | 0.35 | Lower in Fraud | ✓ |
| ZIP/IP Distance | 0.22 | 0.49 | Higher in Fraud | ✓ |
| Thin File Rate | 0.05 | 0.48 | Higher in Fraud | ✓ |
| Months at Address | 45.2 | 7.8 | Lower in Fraud | ✓ |
| Months at Employer | 32.1 | 6.2 | Lower in Fraud | ✓ |

**Assessment**: All 6/6 fraud signals show the expected directional patterns. The dataset is well-suited for fraud detection modeling.

## Borderline Case Analysis

| Metric | Count | Assessment |
|--------|-------|------------|
| Total hard cases | 758 | Sufficient (>500) |
| Hard legitimate cases | ~550 | Good |
| Hard fraud cases | ~208 | Sufficient (>100) |
| legitimate_noisy count | 900 | Sufficient (>500) |
| true_name_fraud (hard) | 93 | Good |
| Middle signal score (0.3-0.6) | ~800 | Good ambiguity |

**Assessment**: Dataset contains sufficient borderline examples for Stage 2 encoder feature development. The legitimate_noisy and hard fraud cases provide meaningful ambiguity for the model to learn from.

## Issues Found

**No significant issues were found.**

The dataset passed all quality checks:
- No missing values
- No duplicates
- All values valid
- Fraud rate on target
- All signals show expected patterns
- Sufficient borderline cases

## Cleaning Actions Applied

The following light cleaning actions were applied:

1. **Whitespace normalization**: Stripped leading/trailing whitespace and normalized multiple spaces in text columns
2. **Date format verification**: Confirmed application_date is in YYYY-MM-DD format
3. **Month alignment verification**: Confirmed application_month matches application_date
4. **Numeric type enforcement**: Ensured numeric columns have correct data types

**Rows removed**: 0 (no cleaning required removal)

## Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Rows after cleaning | 5,000 |
| Columns | 41 |
| Fraud rate | 12.0% |
| Date range | 2024-01-01 to 2024-12-31 |

## Figures Generated

The following visualizations were saved to `reports/figures/`:

1. `fraud_rate_by_month.png` - Fraud rate consistency across months
2. `device_reuse_by_label.png` - Device reuse distribution by fraud label
3. `months_at_address_by_label.png` - Tenure distribution by fraud label
4. `free_email_by_label.png` - Free email domain rate comparison
5. `text_length_by_label.png` - Verification note length distribution
6. `signal_score_by_fraud_type.png` - Generated signal score by fraud type

## Tables Generated

The following summary tables were saved to `reports/tables/`:

1. `fraud_type_distribution.csv` - Fraud type counts and percentages
2. `difficulty_distribution.csv` - Difficulty level counts and percentages
3. `signal_pattern_comparison.csv` - Fraud signal pattern analysis

## Conclusion

The synthetic dataset is **ready for modeling**. Key findings:

1. **Data quality is excellent** - No missing values, duplicates, or invalid entries
2. **Fraud distribution matches design** - 12% fraud rate with correct archetype mix
3. **Signals are directionally correct** - All fraud indicators show expected patterns
4. **Borderline cases exist** - Sufficient hard cases for Stage 2 development

**Recommended next steps:**
1. Proceed to Stage 1 structured fraud model development
2. Use the cleaned dataset at `data/processed/applications_cleaned.parquet`
3. Later: Identify borderline cases for Stage 2 encoder feature analysis
