# Phase 7: Baseline Model Summary

## Overview

This document summarizes the Stage 1 baseline fraud models trained using structured features only.

**Date**: Phase 7 Complete  
**Input**: Structured feature table from Phase 6  
**Output**: Trained models and predictions

---

## Data Split

We used a **time-based split** to simulate real deployment:

| Split | Months | Rows | Fraud Rate |
|-------|--------|------|------------|
| Train | 2024-01 to 2024-08 | 3,301 (66.0%) | 11.7% |
| Validation | 2024-09 to 2024-10 | 864 (17.3%) | 13.2% |
| Test | 2024-11 to 2024-12 | 835 (16.7%) | 12.0% |

**Why time-based?** In production, models are trained on historical data and predict on future applications. A random split would leak future information into training.

---

## Features Used

**Total features: 23**

| Category | Count | Features |
|----------|-------|----------|
| Numeric | 13 | age, annual_income, months_at_address, months_at_employer, velocity counts, zip_ip_distance_proxy, application_hour, name_email_match_score, income_age_ratio, tenure_min |
| Binary | 6 | is_free_email_domain, document_uploaded, thin_file_flag, night_application_flag, high_device_velocity_flag, high_identity_reuse_flag |
| Categorical | 4 | state, housing_status, ip_region, employer_industry |

**NOT used** (metadata only):
- application_id, application_date, application_month
- fraud_label (target), fraud_type, difficulty_level, generated_signal_score

---

## Models Trained

### 1. Logistic Regression

- **Purpose**: Interpretable baseline
- **Configuration**: 
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
  - class_weight="balanced" for imbalance handling
  - max_iter=1000, solver="lbfgs"

### 2. LightGBM

- **Purpose**: Gradient boosting (typically best for tabular data)
- **Configuration**:
  - num_leaves=31, learning_rate=0.05
  - is_unbalance=True for class imbalance
  - Early stopping with 50 rounds patience
  - Best iteration: 216

### 3. XGBoost

- **Purpose**: Alternative gradient boosting for comparison
- **Configuration**:
  - max_depth=6, learning_rate=0.05
  - scale_pos_weight for class imbalance
  - Early stopping with 50 rounds patience
  - Best iteration: 29

---

## Test Set Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| Logistic Regression | 0.9883 | 0.9444 | 0.8037 | 0.8600 | 0.8309 |
| **LightGBM** | **0.9950** | **0.9744** | **0.9674** | 0.8900 | **0.9271** |
| XGBoost | 0.9921 | 0.9595 | 0.8980 | 0.8800 | 0.8889 |

### Confusion Matrices (Test Set)

**Logistic Regression:**
```
              Predicted
              0      1
Actual  0   714     21
        1    14     86
```

**LightGBM:**
```
              Predicted
              0      1
Actual  0   732      3
        1    11     89
```

**XGBoost:**
```
              Predicted
              0      1
Actual  0   725     10
        1    12     88
```

---

## Best Model: LightGBM

**LightGBM achieves the best performance** on the test set:

- **ROC-AUC: 0.9950** - Excellent discrimination between fraud and legitimate
- **PR-AUC: 0.9744** - Strong precision-recall tradeoff
- **Precision: 96.7%** - Very few false positives
- **Recall: 89.0%** - Catches most fraud cases
- **F1: 0.9271** - Best overall balance

### Why LightGBM Wins

1. **Handles non-linear relationships** better than Logistic Regression
2. **Feature interactions** are captured automatically
3. **Early stopping** prevents overfitting
4. **Class imbalance handling** via is_unbalance parameter

---

## Key Observations

### 1. All Models Perform Well

Even Logistic Regression achieves 0.988 ROC-AUC, indicating the engineered features from Phase 6 are highly predictive.

### 2. LightGBM Has Best Precision

With only 3 false positives on the test set, LightGBM minimizes unnecessary fraud alerts while maintaining high recall.

### 3. Recall vs Precision Tradeoff

- Logistic Regression: Higher recall (86%) but lower precision (80%)
- LightGBM: Balanced (89% recall, 97% precision)
- XGBoost: Similar to LightGBM but slightly lower precision

### 4. No Significant Overfitting

Validation and test performance are similar, indicating the models generalize well.

---

## Feature Importance (LightGBM)

Top features by gain:

1. `high_identity_reuse_flag` - Reuse of email/phone/address
2. `tenure_min` - Minimum tenure at address/employer
3. `name_email_match_score` - Name-email consistency
4. `thin_file_flag` - Thin credit file indicator
5. `zip_ip_distance_proxy` - Geographic mismatch
6. `num_prev_apps_same_device_7d` - Device velocity
7. `months_at_employer` - Employment tenure
8. `months_at_address` - Address tenure

These align with known fraud patterns:
- Fraudsters reuse identity elements
- Fraudsters have short tenure
- Fraudsters have inconsistent data

---

## Files Created

| File | Description |
|------|-------------|
| `src/train_baseline_models.py` | Training module (~500 lines) |
| `notebooks/05_baseline_models.ipynb` | Interactive training notebook |
| `models/logistic_regression.pkl` | Trained LR model |
| `models/lightgbm_model.pkl` | Trained LightGBM model |
| `models/xgboost_model.pkl` | Trained XGBoost model |
| `reports/tables/baseline_metrics.csv` | All metrics |
| `data/processed/baseline_predictions.parquet` | Predictions for all rows |
| `reports/baseline_model_summary.md` | This summary |

---

## Predictions Table

The predictions table (`baseline_predictions.parquet`) contains:

| Column | Description |
|--------|-------------|
| application_id | Unique identifier |
| application_date | Application date |
| application_month | Month (YYYY-MM) |
| fraud_label | True label (0/1) |
| fraud_type | Fraud archetype |
| difficulty_level | easy/medium/hard |
| generated_signal_score | Latent signal score |
| split_label | train/val/test |
| logistic_regression_score | LR probability |
| logistic_regression_pred | LR prediction (0/1) |
| lightgbm_score | LightGBM probability |
| lightgbm_pred | LightGBM prediction (0/1) |
| xgboost_score | XGBoost probability |
| xgboost_pred | XGBoost prediction (0/1) |

---

## Next Steps (Phase 8)

Use the prediction scores to:

1. **Define the borderline band** - Identify cases where the model is uncertain
2. **Analyze borderline cases** - Understand what makes them difficult
3. **Prepare for Stage 2** - These borderline cases will be routed to the text/encoder model

The key question for Phase 8:
> Can text features improve predictions on cases where the structured model is uncertain?

---

## Beginner-Friendly Interpretation

### What does ROC-AUC of 0.995 mean?

If you randomly pick one fraud case and one legitimate case, the model will correctly rank the fraud case higher 99.5% of the time. This is excellent.

### What does PR-AUC of 0.974 mean?

The model maintains high precision (few false alarms) even at high recall (catching most fraud). This is especially important for imbalanced datasets like fraud detection.

### What does 89% recall mean?

The model catches 89 out of 100 fraud cases. The remaining 11 are "missed" (false negatives).

### What does 97% precision mean?

When the model flags something as fraud, it's correct 97% of the time. Only 3% are false alarms.

### Why is this good for fraud detection?

- High recall = catch most fraud
- High precision = don't annoy legitimate customers with false alerts
- LightGBM achieves both, making it ideal for production use
