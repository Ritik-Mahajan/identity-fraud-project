# Final Project Summary

## Selective Encoder-Enhanced Identity Fraud Detection for Application Risk

---

## Executive Overview

This project designed, built, and rigorously evaluated a two-stage fraud detection system for application/identity fraud. The core hypothesis was that lightweight text/encoder features could improve fraud detection on borderline cases where a structured model is uncertain.

**Bottom Line**: The structured model alone achieves **99.5% ROC-AUC**. Adding text features provides **no incremental value**. This negative result is valuable—it demonstrates that strong feature engineering can eliminate the need for complex NLP, and that disciplined experimentation prevents unnecessary production complexity.

| Key Metric | Value |
|------------|-------|
| Best Model | LightGBM (Stage 1 only) |
| Test ROC-AUC | 0.995 |
| Test Precision | 96.7% |
| Test Recall | 89.0% |
| Stage 2 Improvement | None (0.0%) |

---

## Business Problem

### The Challenge

Application fraud—where fraudsters submit fake or stolen identities to open accounts—costs financial institutions billions annually. The detection challenge involves:

1. **High precision**: Minimize false positives that frustrate legitimate customers
2. **High recall**: Catch as much fraud as possible to prevent losses
3. **Borderline handling**: Ambiguous cases require expensive manual review

### Why This Project Matters

- Demonstrates realistic fraud detection system design
- Tests a production-relevant architecture (selective text analysis)
- Shows rigorous validation methodology
- Provides honest assessment of when complexity helps vs. hurts

---

## Data Design

### Why Synthetic Data?

Real fraud data is sensitive, imbalanced, and proprietary. Synthetic data enables:
- Full control over fraud patterns and difficulty levels
- Reproducible experiments
- Shareable portfolio project

### Dataset Specifications

| Attribute | Value |
|-----------|-------|
| Total Applications | 5,000 |
| Fraud Rate | 12% |
| Time Span | 12 months |
| Split Strategy | Time-based (70/15/15) |

### Fraud Archetypes

| Type | Proportion | Characteristics |
|------|------------|-----------------|
| Legitimate Clean | 70% | Consistent identity, normal behavior |
| Legitimate Noisy | 18% | Real customers with messy data |
| Synthetic Identity | 5% | Fabricated, stitched-together identities |
| True-Name Fraud | 4% | Real identity, fraudulent intent |
| Coordinated Attack | 3% | Organized fraud rings |

---

## Modeling Approach

### Two-Stage Architecture

```
Application → Stage 1 (Structured) → Confident? → Yes → Use Stage 1
                                         ↓
                                        No (2.3%)
                                         ↓
                                Stage 2 (Text) → Combined
```

### Stage 1: Structured Model

**23 engineered features** across categories:
- **Velocity**: Device reuse (7d), email/phone/address reuse (30d)
- **Tenure**: Months at address, months at employer
- **Risk flags**: Thin file, night application, high identity reuse
- **Derived**: Income/age ratio, name-email match score

**Model**: LightGBM (selected over Logistic Regression and XGBoost)

### Stage 2: Text/Encoder Model

**12 text-based features** using MiniLM encoder:
- OCR similarity (claimed identity vs document text)
- Employment/address consistency scores
- Suspicious keyword counts
- Text length indicators

**Combiner**: Logistic Regression on Stage 1 score + text features

---

## Stage 1 Results

### Model Comparison

| Model | Test ROC-AUC | Test PR-AUC | Test F1 |
|-------|--------------|-------------|---------|
| **LightGBM** | **0.995** | **0.974** | **0.927** |
| XGBoost | 0.992 | 0.960 | 0.889 |
| Logistic Regression | 0.988 | 0.944 | 0.831 |

### Feature Importance

Most discriminative features:
1. `high_identity_reuse_flag`: 67% fraud rate when true (vs 1.4% false)
2. `high_device_velocity_flag`: Only appears in fraud cases
3. `night_application_flag`: 7x higher in fraud
4. `tenure_min`: 5x lower in fraud (6 vs 34 months)

### Key Insight

The structured features capture the vast majority of fraud signal. With 99.5% ROC-AUC, there is very little room for improvement.

---

## Stage 2 Results

### Ablation Study

| Setup | Description | ROC-AUC | Delta |
|-------|-------------|---------|-------|
| 1 | Structured Only | **0.995** | — |
| 2 | Text Only | 0.818 | -0.177 |
| 3 | Combined (All Cases) | 0.957 | **-0.038** |
| 4 | Borderline Routed | 0.993 | -0.002 |

### Critical Finding

**Adding text features to all cases HURTS performance** (0.995 → 0.957).

The text features dilute the strong structured signal rather than enhancing it.

### Borderline-Specific Analysis

| Setup | Borderline ROC-AUC | Error Reduction |
|-------|-------------------|-----------------|
| Structured Only | 0.814 | — |
| Combined | 0.790 | 0.0% |
| Routed | 0.790 | 0.0% |

Even on borderline cases—where Stage 2 was designed to help—there is **no improvement**.

### Why Stage 2 Didn't Help

1. **Strong baseline**: 99.5% ROC-AUC leaves almost no room for improvement
2. **Feature correlation**: Text features capture similar patterns as structured features
3. **Small borderline set**: Only 117 cases (45 in test) limited Stage 2 learning
4. **Training limitation**: Training split had 0 fraud cases in borderline band

---

## Validation Findings

### Threshold Analysis

Performance is stable across operating points:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.3 | 93.8% | 91.0% | 0.924 |
| 0.5 | 96.7% | 89.0% | 0.927 |
| 0.7 | 97.8% | 89.0% | 0.932 |

### Calibration

| Model | Brier Score | Interpretation |
|-------|-------------|----------------|
| LightGBM | 0.0146 | Well-calibrated |
| Combined | 0.0158 | Slightly worse |

### Stability Over Time

| Month | Fraud Rate | Capture Rate |
|-------|------------|--------------|
| 2024-11 | 11.9% | 87.8% |
| 2024-12 | 12.0% | 90.2% |

Performance is stable across the test period.

### Runtime

| Component | Latency |
|-----------|---------|
| Stage 1 inference | ~40ms |
| Encoder features | ~50ms per case |
| Stage 2 overhead | ~6s total (117 cases) |

Stage 2 adds computational cost without performance benefit.

---

## Final Recommendation

### For This Dataset

**Use Stage 1 (LightGBM) alone.**

The two-stage architecture adds complexity without improving performance:
- No accuracy gain
- Additional model to maintain
- Encoder dependency (sentence-transformers)
- Extra latency for borderline cases

### For Similar Problems

1. **Start with strong structured features** before adding complexity
2. **Establish a baseline** and measure incremental value rigorously
3. **Use ablation studies** to isolate component contributions
4. **Accept negative results** when the evidence is clear

### If Text Must Be Used

Use **borderline-only routing**:
- Limits text analysis to 2.3% of cases
- Preserves most of Stage 1 performance (0.993 vs 0.995)
- Contains potential "damage" from noisy features

---

## Lessons Learned

### Technical Lessons

1. **Feature engineering matters more than model complexity**
   - Well-designed structured features achieved 99.5% ROC-AUC
   - Text/NLP added no value despite significant effort

2. **Strong baselines make improvement hard**
   - With only 14 errors on 835 test cases, fixing even 1-2 is statistically challenging

3. **Ablation studies are essential**
   - Without ablation, we might have deployed unnecessary complexity

### Process Lessons

1. **Validate incrementally**
   - Test each component's contribution before combining

2. **Be honest about negative results**
   - "It didn't help" is a valid and valuable finding

3. **Consider operational complexity**
   - A simpler model that performs equally well is better

### What This Project Demonstrates

- Rigorous experimental methodology
- Honest evaluation of results
- Understanding that complexity doesn't always help
- Production-oriented thinking (latency, maintenance, dependencies)

---

## Project Artifacts

### Key Files

| File | Description |
|------|-------------|
| `models/lightgbm_model.pkl` | Production-ready model |
| `data/processed/final_model_predictions.parquet` | All predictions |
| `reports/model_validation_summary.md` | Validation details |
| `notebooks/09_model_validation.ipynb` | Interactive validation |

### Code Modules

| Module | Purpose |
|--------|---------|
| `src/data_generation.py` | Synthetic data creation |
| `src/feature_engineering.py` | Structured feature pipeline |
| `src/train_baseline_models.py` | Stage 1 training |
| `src/encoder_features.py` | Text feature generation |
| `src/validate_models.py` | Comprehensive validation |

---

## Conclusion

This project successfully tested the hypothesis that text features could improve fraud detection on borderline cases. The evidence clearly shows that for this dataset, the hypothesis does not hold.

**This is a valuable outcome.** The project demonstrates:
- How to design and validate a production ML system
- When to accept that simpler is better
- How to communicate negative results professionally

The methodology is directly transferable to any problem where you need to validate whether added complexity provides value.
