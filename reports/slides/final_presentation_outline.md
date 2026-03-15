# Final Presentation Outline

## Selective Encoder-Enhanced Identity Fraud Detection

**Duration**: 10-15 minutes  
**Audience**: Technical interviewers, hiring managers, or project reviewers

---

## Slide 1: Title

**Selective Encoder-Enhanced Identity Fraud Detection for Application Risk**

- Your Name
- Date
- Project Type: End-to-End ML System

---

## Slide 2: The Problem

**Application Fraud is Costly**

- Fraudsters submit fake/stolen identities to open accounts
- Financial losses from approved fraud
- Customer friction from false positives
- Manual review is expensive

**The Challenge**
- Detect fraud with high precision
- Minimize friction for legitimate customers
- Handle ambiguous cases effectively

---

## Slide 3: The Hypothesis

**Can text analysis improve fraud detection on borderline cases?**

Two-Stage Architecture:
```
Application → Stage 1 (Structured) → Confident? → Yes → Use Stage 1
                                        ↓
                                       No
                                        ↓
                              Stage 2 (Text) → Combined Decision
```

**Hypothesis**: Text features (OCR consistency, keyword patterns) will catch fraud missed by structured features on uncertain cases.

---

## Slide 4: Dataset Overview

**Synthetic Dataset Design**

| Attribute | Value |
|-----------|-------|
| Total Applications | 5,000 |
| Fraud Rate | 12% |
| Time Span | 12 months |
| Train/Val/Test | 70%/15%/15% (time-based) |

**5 Fraud Archetypes**
- Legitimate clean (70%)
- Legitimate noisy (18%)
- Synthetic identity (5%)
- True-name fraud (4%)
- Coordinated attack (3%)

---

## Slide 5: Feature Engineering

**23 Structured Features**

| Category | Examples |
|----------|----------|
| Velocity | device_reuse_7d, email_reuse_30d |
| Tenure | months_at_address, months_at_employer |
| Risk Flags | thin_file, night_application |
| Derived | income_age_ratio, name_email_match |

**Most Discriminative**
- `high_identity_reuse_flag`: 67% fraud when true vs 1.4% when false
- `high_device_velocity_flag`: Only appears in fraud

---

## Slide 6: Stage 1 Results

**LightGBM Structured Model**

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.995** |
| PR-AUC | 0.974 |
| Precision | 96.7% |
| Recall | 89.0% |
| F1 | 0.927 |

**Key Insight**: Very strong baseline leaves little room for improvement

---

## Slide 7: Borderline Band Definition

**Identifying Uncertain Cases**

- Borderline band: Stage 1 score in [0.01, 0.99]
- 117 cases (2.3% of data)
- 42% fraud rate (vs 12% overall)
- Mostly `legitimate_noisy` and `true_name_fraud`

**These are the cases where Stage 2 should help**

---

## Slide 8: Text Features

**Encoder-Based Features (MiniLM)**

| Feature | Purpose |
|---------|---------|
| `application_ocr_similarity` | Identity vs document match |
| `employment_consistency_score` | Employer explanation match |
| `suspicious_keyword_count` | Risk indicator words |

**Text-Only Model**: 82% ROC-AUC on borderline cases

---

## Slide 9: Ablation Study

**Four Setups Compared**

| Setup | ROC-AUC | Delta |
|-------|---------|-------|
| 1. Structured Only | **0.995** | — |
| 2. Text Only | 0.818 | -0.177 |
| 3. Combined (All) | 0.957 | **-0.038** |
| 4. Borderline Routed | 0.993 | -0.002 |

**Key Finding**: Adding text to all cases **hurts** performance

---

## Slide 10: Borderline Analysis

**Did Stage 2 Help on Borderline Cases?**

| Setup | Borderline ROC-AUC | Error Reduction |
|-------|-------------------|-----------------|
| Structured | 0.814 | — |
| Combined | 0.790 | 0.0% |
| Routed | 0.790 | 0.0% |

**Answer: No**

Text features provided no error reduction on borderline cases.

---

## Slide 11: Why Didn't Text Help?

**Three Factors**

1. **Strong Baseline**
   - 99.5% ROC-AUC leaves almost no room for improvement
   - Only 14 errors on 835 test cases

2. **Feature Correlation**
   - Text features capture similar patterns as structured features
   - Redundant signal

3. **Small Borderline Set**
   - Only 45 borderline test cases
   - Limited statistical power

---

## Slide 12: Validation Summary

**Comprehensive Validation**

| Analysis | Finding |
|----------|---------|
| Threshold | No change in precision/recall tradeoff |
| Calibration | Stage 1 best calibrated (Brier: 0.015) |
| Stability | Stable across 2 test months |
| Runtime | Stage 2 adds ~6s for borderline cases |

**Conclusion**: Stage 2 adds complexity without benefit

---

## Slide 13: Recommendation

**Use Stage 1 Alone**

```
Application → LightGBM → Decision
```

**Why?**
- Simpler architecture
- No encoder dependency
- Equal or better performance
- Lower latency

**If text must be used**: Borderline routing limits damage

---

## Slide 14: What I Learned

**Technical Lessons**
- Strong feature engineering can eliminate need for complex models
- Ablation studies are essential before adding complexity
- Negative results are valuable when properly validated

**Process Lessons**
- Establish baseline before adding features
- Validate on target subset (borderline), not just overall
- Be willing to recommend simpler solutions

---

## Slide 15: Future Work

**If Continuing This Project**

1. Generate more borderline cases (10x current)
2. Richer text field simulation
3. Alternative encoder models
4. Cost-sensitive evaluation

**But also accept**: Structured model may be sufficient

---

## Slide 16: Q&A

**Key Numbers**
- Stage 1 ROC-AUC: 0.995
- Borderline cases: 2.3%
- Text improvement: None

**Key Takeaway**
> "We tested whether text features help fraud detection. They don't for this dataset. That's a valid finding that saves production complexity."

---

## Appendix Slides (if needed)

### A1: Full Metrics Table

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| Logistic Regression | 0.988 | 0.944 | 82.4% | 84.0% | 0.831 |
| LightGBM | 0.995 | 0.974 | 96.7% | 89.0% | 0.927 |
| XGBoost | 0.992 | 0.960 | 89.8% | 88.0% | 0.889 |

### A2: Feature Importance

Top 5 LightGBM features:
1. `num_prev_apps_same_device_7d`
2. `high_identity_reuse_flag`
3. `months_at_employer`
4. `thin_file_flag`
5. `zip_ip_distance_proxy`

### A3: Code Structure

```
identity-fraud-project/
├── src/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── train_baseline_models.py
│   ├── encoder_features.py
│   └── validate_models.py
├── notebooks/
│   ├── 05_baseline_models.ipynb
│   ├── 08_final_combined_model.ipynb
│   └── 09_model_validation.ipynb
├── models/
│   └── lightgbm_model.pkl
└── reports/
    └── model_validation_summary.md
```
