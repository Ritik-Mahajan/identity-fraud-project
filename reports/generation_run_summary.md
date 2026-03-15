# Data Generation Run Summary

## Overview

This document summarizes the synthetic dataset generation for the Identity Fraud Detection project (Phase 4).

## Generation Details

| Attribute | Value |
|-----------|-------|
| Generator script | `src/data_generation.py` |
| Random seed | 42 |
| Generation date | Phase 4 |
| Python version | 3.11+ |

## Output Files

| File | Path | Description |
|------|------|-------------|
| Main dataset (Parquet) | `data/raw/applications_prototype.parquet` | Primary dataset in Parquet format |
| Main dataset (CSV) | `data/raw/applications_prototype.csv` | Primary dataset in CSV format |
| Generation metadata | `data/interim/generation_metadata.csv` | Metadata with fraud_type, difficulty, signal scores |

## Dataset Statistics

### Size

| Metric | Value |
|--------|-------|
| Total rows | 5,000 |
| Total columns | 40 |
| Date range | 2024-01-01 to 2024-12-31 |
| Unique months | 12 |

### Fraud Rate

| Category | Count | Percentage |
|----------|-------|------------|
| Legitimate (fraud_label=0) | 4,400 | 88.0% |
| Fraud (fraud_label=1) | 600 | 12.0% |

### Fraud Type Distribution

| Fraud Type | Count | Percentage |
|------------|-------|------------|
| legitimate_clean | 3,500 | 70.0% |
| legitimate_noisy | 900 | 18.0% |
| synthetic_identity | 250 | 5.0% |
| true_name_fraud | 200 | 4.0% |
| coordinated_attack | 150 | 3.0% |

### Difficulty Level Distribution

| Difficulty | Count | Percentage |
|------------|-------|------------|
| easy | ~2,650 | ~53% |
| medium | ~1,590 | ~32% |
| hard | ~760 | ~15% |

## Column Summary

### Identifiers / Time (3 columns)
- `application_id` - Unique application identifier
- `application_date` - Application submission date
- `application_month` - YYYY-MM format

### Claimed Identity (5 columns)
- `claimed_first_name`, `claimed_last_name`
- `date_of_birth`, `age`, `ssn_last4`

### Address (4 columns)
- `address_line`, `city`, `state`, `zip_code`

### Contact (4 columns)
- `phone_number`, `email`, `email_domain`, `is_free_email_domain`

### Employment / Financial (7 columns)
- `employer_name`, `employer_industry`, `annual_income`
- `housing_status`, `months_at_address`, `months_at_employer`, `thin_file_flag`

### Digital / Behavioral (8 columns)
- `device_id`, `ip_region`, `application_hour`
- `num_prev_apps_same_device_7d`, `num_prev_apps_same_email_30d`
- `num_prev_apps_same_phone_30d`, `num_prev_apps_same_address_30d`
- `zip_ip_distance_proxy`

### Precomputed Signals (2 columns)
- `name_email_match_score`, `document_uploaded`

### Text Fields (4 columns)
- `verification_note`, `ocr_document_text`
- `address_explanation_text`, `employment_explanation_text`

### Labels / Meta (4 columns)
- `fraud_label`, `fraud_type`, `difficulty_level`, `generated_signal_score`

## Assumptions and Simplifications

### Implemented as Designed
1. **Fraud rate**: Achieved ~12% fraud rate as specified
2. **Archetype distribution**: Matched target percentages
3. **12-month span**: Applications distributed across full year
4. **All 40 columns**: Complete schema implemented

### Simplifications Made
1. **Income generation**: Uses industry-based ranges with age adjustment, not complex economic modeling
2. **Device reuse**: Uses pre-generated fraud device pools rather than true sequential tracking
3. **Coordinated attacks**: Clusters share device IDs but don't simulate true temporal clustering
4. **Name/email matching**: Simple substring matching heuristic
5. **ZIP/IP distance**: Synthetic proxy score, not real geographic calculation

### Design Decisions
1. **Text templates**: Used Phase 3 templates with placeholder substitution
2. **Difficulty assignment**: Probabilistic based on archetype, affects signal intensity
3. **Signal score**: Weighted sum of fraud indicators, for debugging only
4. **Validation**: Basic checks for data integrity, not statistical validation

## Verification Checks Passed

- All required columns present
- `fraud_label` contains only 0/1
- `fraud_type` contains only valid values
- `difficulty_level` contains only valid values
- `age` within 18-75 range
- `application_id` is unique

## Next Steps

1. **Exploratory Data Analysis**: Examine distributions and correlations
2. **Feature Engineering**: Prepare features for Stage 1 model
3. **Stage 1 Model**: Train structured fraud detection model
4. **Borderline Analysis**: Identify cases for Stage 2 encoder

## How to Regenerate

```bash
# Activate virtual environment
source .venv/bin/activate

# Run generator (creates 5000 rows)
python src/data_generation.py

# Or use the notebook
jupyter notebook notebooks/02_data_generation.ipynb
```

To generate larger datasets, modify the `n_rows` parameter:

```python
from src.data_generation import create_dataset, save_outputs

# Generate 20,000 rows
df = create_dataset(n_rows=20000)
save_outputs(df, output_name="applications_20k")
```
