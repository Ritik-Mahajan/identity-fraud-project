# Phase 6: Feature Engineering Summary

## Overview

This document summarizes the feature engineering work for Stage 1 structured fraud modeling.

**Date**: Phase 6 Complete  
**Input**: Cleaned dataset from Phase 5  
**Output**: Structured feature table for modeling

---

## Input / Output

| Item | Path |
|------|------|
| Input file | `data/processed/applications_cleaned.parquet` |
| Output parquet | `data/processed/structured_features.parquet` |
| Output CSV | `data/processed/structured_features.csv` |

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total rows | 5,000 |
| Total columns | 30 |
| Modeling features | 23 |
| Metadata columns | 7 |
| Missing values | 0 |

---

## Features Created

### Numeric Features (10)

Direct from cleaned data:

| Feature | Description |
|---------|-------------|
| `age` | Applicant age in years |
| `annual_income` | Annual income in USD |
| `months_at_address` | Tenure at current address |
| `months_at_employer` | Tenure at current employer |
| `num_prev_apps_same_device_7d` | Prior apps from same device (7 days) |
| `num_prev_apps_same_email_30d` | Prior apps from same email (30 days) |
| `num_prev_apps_same_phone_30d` | Prior apps from same phone (30 days) |
| `num_prev_apps_same_address_30d` | Prior apps from same address (30 days) |
| `zip_ip_distance_proxy` | ZIP/IP mismatch severity (0-1) |
| `application_hour` | Hour of application (0-23) |

### Binary Features (3)

Direct from cleaned data:

| Feature | Description |
|---------|-------------|
| `is_free_email_domain` | 1 if email is from free provider |
| `document_uploaded` | 1 if document was uploaded |
| `thin_file_flag` | 1 if applicant has thin credit file |

### Pre-computed Features (1)

| Feature | Description |
|---------|-------------|
| `name_email_match_score` | Score (0-1) for name/email consistency |

### Engineered Features (5)

| Feature | Rule | Purpose |
|---------|------|---------|
| `income_age_ratio` | `annual_income / age` | Detect unrealistic income for age |
| `tenure_min` | `min(months_at_address, months_at_employer)` | Capture minimum stability |
| `night_application_flag` | `1 if hour in [0,1,2,3,4,5]` | Flag late-night applications |
| `high_device_velocity_flag` | `1 if device_7d >= 3` | Flag device abuse |
| `high_identity_reuse_flag` | `1 if any reuse >= 2` | Flag identity element reuse |

### Categorical Features (4)

| Feature | Unique Values |
|---------|---------------|
| `state` | 25 |
| `housing_status` | 4 (rent, own, family, other) |
| `ip_region` | 25 |
| `employer_industry` | 9 |

### Metadata Columns (7)

Preserved for later phases (not modeling features):

- `application_id`
- `application_date`
- `application_month`
- `fraud_label`
- `fraud_type`
- `difficulty_level`
- `generated_signal_score`

---

## Engineered Feature Rules

### income_age_ratio

```
income_age_ratio = annual_income / age
```

- Handles divide-by-zero safely (returns 0 if age <= 0)
- Higher values may indicate unrealistic income claims

### tenure_min

```
tenure_min = min(months_at_address, months_at_employer)
```

- Captures the minimum stability across address and employment
- Lower values indicate higher risk

### night_application_flag

```
night_application_flag = 1 if application_hour in [0, 1, 2, 3, 4, 5] else 0
```

- Flags applications submitted between midnight and 5 AM
- Late-night applications may be suspicious

### high_device_velocity_flag

```
high_device_velocity_flag = 1 if num_prev_apps_same_device_7d >= 3 else 0
```

- Threshold: 3 or more applications from same device in 7 days
- Strong indicator of device abuse or coordinated attacks

### high_identity_reuse_flag

```
high_identity_reuse_flag = 1 if max(email_30d, phone_30d, address_30d) >= 2 else 0
```

- Threshold: 2 or more reuses of any identity element
- Captures synthetic identity and coordinated attack patterns

---

## Feature Discriminative Power

### Binary Flag Rates by Fraud Label

| Feature | Legitimate | Fraud | Ratio |
|---------|------------|-------|-------|
| `is_free_email_domain` | 44.2% | 73.8% | 1.67x |
| `document_uploaded` | 91.1% | 74.7% | 0.82x |
| `thin_file_flag` | 9.1% | 54.8% | 6.02x |
| `night_application_flag` | 6.0% | 42.0% | 7.03x |
| `high_device_velocity_flag` | 0.0% | 32.5% | ∞ |
| `high_identity_reuse_flag` | 1.4% | 67.3% | 47.03x |

### Key Numeric Feature Means by Fraud Label

| Feature | Legitimate | Fraud | Direction |
|---------|------------|-------|-----------|
| `tenure_min` | 33.99 | 6.33 | Lower in fraud ✓ |
| `name_email_match_score` | 0.68 | 0.36 | Lower in fraud ✓ |
| `zip_ip_distance_proxy` | 0.18 | 0.46 | Higher in fraud ✓ |
| `income_age_ratio` | 2142 | 2190 | Slightly higher in fraud |

---

## Missing Value Handling

No missing values were found in the cleaned dataset.

No imputation or filling was required.

---

## Quality Checks Passed

| Check | Result |
|-------|--------|
| Row count matches input | ✓ PASS |
| application_id is unique | ✓ PASS |
| fraud_label is binary (0/1) | ✓ PASS |
| night_application_flag is binary | ✓ PASS |
| high_device_velocity_flag is binary | ✓ PASS |
| high_identity_reuse_flag is binary | ✓ PASS |
| No all-null columns | ✓ PASS |
| No missing values | ✓ PASS |

---

## Output File Sizes

| File | Size |
|------|------|
| `structured_features.parquet` | 199 KB |
| `structured_features.csv` | 765 KB |

---

## Key Observations

1. **Engineered features are highly discriminative:**
   - `high_identity_reuse_flag` has 47x higher rate in fraud
   - `high_device_velocity_flag` appears only in fraud cases
   - `night_application_flag` has 7x higher rate in fraud

2. **Tenure features show expected patterns:**
   - `tenure_min` is 5x lower in fraud (6.3 vs 34 months)
   - This aligns with synthetic identity and new account fraud patterns

3. **Name/email consistency is discriminative:**
   - `name_email_match_score` is nearly 2x lower in fraud (0.36 vs 0.68)

4. **Ready for Stage 1 modeling:**
   - All features validated
   - No missing values
   - Strong signal separation between fraud and legitimate

---

## Next Steps

Phase 7 will use this feature table to:
1. Train baseline Logistic Regression model
2. Train LightGBM/XGBoost models
3. Evaluate using time-based train/test split
4. Generate probability scores for borderline analysis

---

## Files Created

| File | Description |
|------|-------------|
| `src/feature_engineering.py` | Feature engineering module |
| `notebooks/04_feature_engineering.ipynb` | Interactive notebook |
| `data/processed/structured_features.parquet` | Feature table (parquet) |
| `data/processed/structured_features.csv` | Feature table (CSV) |
| `reports/feature_engineering_summary.md` | This summary |
