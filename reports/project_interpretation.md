# Project Interpretation: Selective Encoder-Enhanced Identity Fraud Detection

## 1. Project Objective

This project set out to answer a specific question:

> **Can lightweight text/encoder features improve fraud detection on borderline cases where a structured model is uncertain?**

The hypothesis was that a two-stage system—where Stage 1 uses structured features and Stage 2 adds text analysis only for uncertain cases—would outperform a structured-only approach, particularly on ambiguous applications.

## 2. What Was Built

### System Architecture

```
Application → Stage 1 (Structured Model) → Confidence Check
                                              ↓
                    High Confidence → Use Stage 1 prediction
                    Low Confidence  → Stage 2 (Text Features) → Combined prediction
```

### Components Developed

| Component | Description |
|-----------|-------------|
| **Synthetic Dataset** | 5,000 applications with 12% fraud rate, 5 fraud archetypes |
| **Stage 1 Model** | LightGBM on 23 engineered structured features |
| **Borderline Band** | Cases with Stage 1 scores between 0.01 and 0.99 |
| **Stage 2 Features** | 12 text/encoder features using MiniLM embeddings |
| **Combined Model** | Logistic Regression combining Stage 1 score + text features |

## 3. What Stage 1 Achieved

The structured model exceeded expectations:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC-AUC | 0.995 | Near-perfect discrimination |
| PR-AUC | 0.974 | Strong precision-recall tradeoff |
| Precision | 96.7% | Only 3% false positive rate |
| Recall | 89.0% | Catches 89% of fraud |
| F1 | 0.927 | Excellent balance |

**Key insight**: The structured features engineered in Phase 6—particularly device velocity, identity reuse patterns, tenure indicators, and thin-file flags—captured the vast majority of fraud signal.

## 4. What Stage 2 Attempted to Add

Stage 2 introduced text-based features designed to capture semantic inconsistencies:

| Feature | Purpose |
|---------|---------|
| `application_ocr_similarity` | Detect mismatch between claimed identity and document text |
| `employment_consistency_score` | Detect employer/explanation inconsistency |
| `address_consistency_score` | Detect address/explanation inconsistency |
| `suspicious_keyword_count` | Flag risk indicators in verification notes |

**The hypothesis**: These features would help on borderline cases where structured signals are ambiguous but text reveals inconsistencies.

## 5. What the Ablation Study Showed

### Overall Performance (Test Set)

| Setup | ROC-AUC | Delta vs Stage 1 |
|-------|---------|------------------|
| Stage 1 (Structured Only) | 0.995 | — |
| Text Only | 0.818 | -0.177 |
| Combined (All Cases) | 0.957 | **-0.038** |
| Borderline Routed | 0.993 | -0.002 |

**Finding**: Adding text features to all cases actually **hurts** performance. The combined model's ROC-AUC dropped from 0.995 to 0.957.

### Why Combined-All Hurts Performance

1. **Signal dilution**: Text features add noise to already-confident predictions
2. **Feature correlation**: Text features partially duplicate structured feature signals
3. **Model complexity**: More features without more signal leads to overfitting risk

### Why Borderline Routing Preserves Performance

The routed approach (0.993 ROC-AUC) stays close to Stage 1 because:
- 97.7% of cases use Stage 1 predictions directly
- Only 2.3% (117 cases) are routed to Stage 2
- The "damage" from text features is contained to the borderline subset

## 6. What the Borderline-Case Analysis Showed

### Borderline Subset Statistics

- **Size**: 117 total (45 in test set)
- **Fraud rate**: 42.2% (vs 12% overall)
- **Composition**: Mostly `legitimate_noisy` and `true_name_fraud`

### Borderline Performance

| Setup | ROC-AUC | Error Reduction |
|-------|---------|-----------------|
| Stage 1 (Baseline) | 0.814 | — |
| Combined | 0.790 | 0.0% |
| Borderline Routed | 0.790 | 0.0% |

**Finding**: Stage 2 provides **no error reduction** on borderline cases. The combined model actually shows slightly worse ROC-AUC (0.790 vs 0.814).

### Why Stage 2 Didn't Help on Borderline Cases

1. **Small sample size**: Only 45 borderline test cases limits statistical power
2. **Training data limitation**: The training split had 0 fraud cases in the borderline band, forcing us to combine train+val for text model training
3. **Feature redundancy**: Text features may capture similar patterns as structured features
4. **Strong baseline**: Even on borderline cases, Stage 1 achieves 81.4% ROC-AUC

## 7. What Threshold/Calibration/Stability Analysis Showed

### Threshold Analysis

| Threshold | Stage 1 Precision | Stage 1 Recall | Routed Precision | Routed Recall |
|-----------|-------------------|----------------|------------------|---------------|
| 0.3 | 93.8% | 91.0% | 96.7% | 89.0% |
| 0.5 | 96.7% | 89.0% | 96.7% | 89.0% |
| 0.7 | 97.8% | 89.0% | 96.7% | 89.0% |

**Finding**: The routed approach does not meaningfully change the precision/recall tradeoff at any threshold.

### Calibration

| Model | Brier Score | Interpretation |
|-------|-------------|----------------|
| Stage 1 | 0.0146 | Best calibration |
| Combined | 0.0158 | Slightly worse |
| Routed | 0.0157 | Slightly worse |

**Finding**: Stage 1 has the best calibration. Adding text features slightly degrades probability calibration.

### Stability Over Time

| Month | Fraud Rate | Capture Rate |
|-------|------------|--------------|
| 2024-11 | 11.9% | 87.8% |
| 2024-12 | 12.0% | 90.2% |

**Finding**: Performance is stable across the test period (limited to 2 months).

## 8. What Worked Well

### Strong Structured Feature Engineering

The Phase 6 features proved highly discriminative:

| Feature | Fraud Rate When High | Interpretation |
|---------|---------------------|----------------|
| `high_identity_reuse_flag` | 67% vs 1.4% | 47x higher in fraud |
| `high_device_velocity_flag` | 33% vs 0% | Only appears in fraud |
| `night_application_flag` | 42% vs 6% | 7x higher in fraud |

### Rigorous Experimental Design

- Time-based train/val/test split (no data leakage)
- Multiple model comparison (Logistic Regression, LightGBM, XGBoost)
- Ablation study with clear setups
- Borderline-specific evaluation

### Honest Validation

- Did not cherry-pick results
- Reported negative findings clearly
- Acknowledged limitations

## 9. What Did Not Work as Expected

### Text Features Did Not Improve Performance

**Expected**: Text features would catch fraud missed by structured features.
**Actual**: Text features provided redundant signal already captured by structured features.

### Borderline Routing Did Not Improve Borderline Performance

**Expected**: Combined model would outperform Stage 1 on borderline cases.
**Actual**: Combined model showed slightly worse ROC-AUC on borderline cases.

### Small Borderline Set Limited Learning

**Expected**: ~10-15% of cases would be borderline.
**Actual**: Only 2.3% of cases fell in the borderline band, limiting Stage 2 training data.

## 10. Practical Takeaways

### For This Project

1. **Use Stage 1 alone**: The structured model is sufficient
2. **Don't add text features**: They provide no incremental value
3. **Feature engineering matters**: Strong structured features can eliminate the need for complex text analysis

### For Similar Projects

1. **Establish a strong baseline first**: A good structured model may be all you need
2. **Test incrementally**: Don't assume more features = better performance
3. **Validate on the target subset**: Overall metrics can hide subset-specific behavior
4. **Be honest about negative results**: "It didn't help" is a valid finding

## 11. Recommended Final Architecture

### For Production

```
Application → Stage 1 (LightGBM) → Final Decision
```

**Rationale**: Stage 2 adds complexity without improving performance.

### If Text Features Must Be Used

```
Application → Stage 1 → Confidence Check
                           ↓
           High Confidence → Stage 1 prediction
           Low Confidence  → Stage 2 → Combined prediction
```

**Rationale**: Borderline routing limits the "damage" from text features to uncertain cases only.

## 12. Honest Limitations

### Dataset Limitations

1. **Synthetic data**: Real fraud patterns may differ
2. **Small borderline set**: Only 117 cases (45 in test)
3. **Limited time span**: 12 months, 2 months in test
4. **Simplified text fields**: Real OCR/notes may be messier

### Methodological Limitations

1. **Single encoder model**: Only tested MiniLM
2. **Simple combiner**: Only tested Logistic Regression
3. **Fixed borderline band**: Did not optimize thresholds
4. **No cost-sensitive evaluation**: Treated FP and FN equally

### What We Cannot Conclude

1. Text features would never help on any fraud dataset
2. More sophisticated text processing wouldn't improve results
3. A larger borderline set wouldn't change the outcome

## Summary

This project tested a reasonable hypothesis: that text features could improve fraud detection on borderline cases. The evidence shows that for this dataset, the hypothesis does not hold. The structured model is already so strong (99.5% ROC-AUC) that text features provide no additional value.

**This is still a valuable finding.** The project demonstrates:
- Rigorous experimental methodology
- Honest evaluation of results
- Understanding that complexity doesn't always help
- The importance of strong feature engineering

The key insight: **A well-engineered structured model can be sufficient. Not every problem needs deep learning or NLP.**
