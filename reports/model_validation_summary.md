# Phase 11: Model Validation Summary

## Validation Goals

This phase performed comprehensive validation of the fraud detection system to answer four key questions:

1. **Does the encoder help on borderline cases?**
2. **Does the combined system improve precision/recall?**
3. **Does performance remain stable over time?**
4. **Is the added complexity worth it?**

## Candidate Setups Compared

| Setup | Description |
|-------|-------------|
| 1. Logistic Regression | Phase 7 baseline structured model |
| 2. LightGBM (Stage 1) | Best Phase 7 structured model |
| 3. Text Only | Text features only (borderline cases) |
| 4. Combined (All) | Stage 1 + text features on all cases |
| 5. Borderline Routed | Stage 1 for confident, combined for borderline |

## Key Overall Findings

### Test Set Performance (Overall)

| Setup | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| Logistic Regression | 0.988 | 0.944 | 82.4% | 84.0% | 0.831 |
| **LightGBM (Stage 1)** | **0.995** | **0.974** | **96.7%** | **89.0%** | **0.927** |
| Text Only | 0.818 | 0.755 | 72.2% | 68.4% | 0.703 |
| Combined (All) | 0.957 | 0.948 | 96.7% | 89.0% | 0.927 |
| Borderline Routed | 0.993 | 0.966 | 96.7% | 89.0% | 0.927 |

**Finding**: LightGBM (Stage 1) achieves the best overall performance with 99.5% ROC-AUC.

### Key Comparisons

| Comparison | ROC-AUC Delta | Interpretation |
|------------|---------------|----------------|
| LightGBM vs Combined (All) | -0.038 | Combined HURTS performance |
| LightGBM vs Borderline Routed | -0.003 | Routed is slightly worse |
| LightGBM vs Text Only | -0.177 | Text alone is much weaker |

## Borderline-Case Findings

### Borderline Subset Statistics

- **Total borderline test cases**: 45
- **Fraud rate in borderline**: 42.2%
- **Fraud types**: legitimate_noisy, true_name_fraud

### Borderline Subset Performance

| Setup | ROC-AUC | PR-AUC | Error Reduction |
|-------|---------|--------|-----------------|
| LightGBM (Baseline) | 0.814 | 0.774 | — |
| Combined (All) | 0.790 | 0.741 | 0.0% |
| Borderline Routed | 0.790 | 0.741 | 0.0% |

**Finding**: Stage 2 provides **NO error reduction** on borderline cases. The combined model actually shows slightly worse ROC-AUC (0.790 vs 0.814).

### Why Stage 2 Doesn't Help on Borderline Cases

1. **Small sample size**: Only 45 borderline test cases limits statistical power
2. **Training data limitation**: Training split had 0 fraud cases in borderline band
3. **Feature redundancy**: Text features may be correlated with structured features
4. **Strong baseline**: LightGBM already captures most fraud signal

## Threshold Findings

### Threshold Analysis (Test Set)

| Threshold | LightGBM Precision | LightGBM Recall | Routed Precision | Routed Recall |
|-----------|-------------------|-----------------|------------------|---------------|
| 0.3 | 93.8% | 91.0% | 96.7% | 89.0% |
| 0.4 | 94.7% | 90.0% | 96.7% | 89.0% |
| 0.5 | 96.7% | 89.0% | 96.7% | 89.0% |
| 0.6 | 96.7% | 89.0% | 96.7% | 89.0% |
| 0.7 | 97.8% | 89.0% | 96.7% | 89.0% |

**Finding**: The routed approach does not meaningfully change the precision/recall tradeoff. Both systems achieve similar performance at all thresholds.

## Calibration Findings

### Calibration Metrics (Test Set)

| Setup | Brier Score | Mean Abs Calibration Error |
|-------|-------------|---------------------------|
| LightGBM (Stage 1) | 0.0146 | 0.182 |
| Combined (All) | 0.0158 | 0.210 |
| Borderline Routed | 0.0157 | 0.210 |

**Finding**: 
- All models have low Brier scores (good probabilistic predictions)
- LightGBM has the best calibration
- Scores are calibrated enough for decisioning, primarily useful for ranking

## Stability Findings

### Monthly Stability (Test Set)

| Month | Fraud Rate | Avg Score | Capture Rate |
|-------|------------|-----------|--------------|
| 2024-11 | 11.9% | 0.111 | 87.8% |
| 2024-12 | 12.0% | 0.114 | 90.2% |

**Finding**: Performance is stable across the 2 months in the test set.

**Limitation**: Only 2 months of test data available; more data would strengthen this conclusion.

## Runtime/Practicality Findings

### Runtime Summary

| Stage | Time | Notes |
|-------|------|-------|
| Stage 1 Model Load | 0.04s | LightGBM model |
| Stage 2 Combined Model Load | 0.0002s | Logistic Regression |
| Encoder Features (estimate) | 5.85s | 117 borderline cases @ 50ms each |
| **Total Stage 2 Overhead** | **~6s** | For borderline cases only |

### Practicality Assessment

| Factor | Assessment |
|--------|------------|
| Performance gain | **None** (Stage 2 doesn't improve ROC-AUC) |
| Additional complexity | Encoder model + combiner model |
| Additional runtime | ~6 seconds for borderline cases |
| Additional dependencies | sentence-transformers (MiniLM) |

**Finding**: The added Stage 2 complexity is **NOT justified** for this dataset.

## Final Answer: Was Stage 2 Worth It?

### Question 1: Does the encoder help on borderline cases?

**Answer: NO**

- On borderline cases, the combined model shows similar or worse ROC-AUC (0.790 vs 0.814)
- No meaningful error reduction was observed
- The borderline subset is small (45 test cases), limiting Stage 2 learning

### Question 2: Does the combined system improve precision/recall?

**Answer: NO**

- Combined-all actually DECREASES overall ROC-AUC (0.995 → 0.957)
- Borderline routing preserves most performance (0.993) but doesn't improve it
- Threshold analysis shows no meaningful change in precision/recall tradeoff

### Question 3: Does performance remain stable over time?

**Answer: YES (with caveats)**

- Fraud rate and capture rates are stable within the test period
- Limited to 2 months of test data

### Question 4: Is the added complexity worth it?

**Answer: NO**

- Stage 1 (LightGBM) achieves 99.5% ROC-AUC
- Stage 2 adds complexity without improving performance
- The encoder adds ~6 seconds of runtime for borderline cases
- **Recommendation: Use Stage 1 alone**

## Key Insight

The structured features engineered in Phase 6 (device velocity, identity reuse, tenure, thin_file_flag, etc.) capture most of the fraud signal. The text features from Phase 9 (OCR similarity, keyword counts) provide **redundant information** that doesn't help the already-strong model.

## Honest Conclusion

**Stage 2 does NOT improve the system.**

The two-stage architecture was a valid hypothesis to test, but the evidence shows it's not beneficial for this dataset. The structured model is already so strong (99.5% ROC-AUC) that text features provide no additional value.

**For interviews**: This is actually a valuable finding. The project demonstrates:
1. Rigorous hypothesis testing
2. Honest evaluation of results
3. Understanding that "more features" doesn't always help
4. The importance of strong feature engineering

## Output Files

### Tables

| File | Description |
|------|-------------|
| `reports/tables/validation_metrics_summary.csv` | All validation metrics |
| `reports/tables/threshold_analysis.csv` | Threshold analysis results |
| `reports/tables/calibration_summary.csv` | Calibration metrics |
| `reports/tables/runtime_summary.csv` | Runtime measurements |
| `reports/tables/monthly_stability.csv` | Monthly stability metrics |

### Figures

| File | Description |
|------|-------------|
| `reports/figures/score_distributions.png` | Score distributions by label |
| `reports/figures/threshold_analysis.png` | Precision/recall vs threshold |
| `reports/figures/calibration_curve.png` | Calibration curves |
| `reports/figures/monthly_stability.png` | Monthly stability charts |
| `reports/figures/borderline_comparison.png` | Borderline subset comparison |
| `reports/figures/roc_curves.png` | ROC curve comparison |

## Recommendation

**Use Stage 1 (LightGBM structured model) alone.**

The two-stage architecture adds complexity without improving performance. For production deployment, the simpler Stage 1 model is:
- Faster (no encoder overhead)
- Simpler to maintain
- Equally or more effective
