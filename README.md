# Selective Encoder-Enhanced Identity Fraud Detection

## Project Overview

A production-oriented fraud detection system that tests whether lightweight text/encoder features can improve fraud detection on borderline cases where structured models are uncertain.

**Key Result**: The structured model achieves **99.5% ROC-AUC**. Text features provide **no incremental value**—a valid negative result demonstrating rigorous experimental methodology.

---

## Business Problem

**Application fraud** costs financial institutions billions annually. Fraudsters submit fake or stolen identities to open accounts, obtain credit, or commit other financial crimes.

**The Challenge**:
- Detect fraud with high precision (minimize false positives that frustrate legitimate customers)
- Maintain high recall (catch as much fraud as possible)
- Handle ambiguous cases where signals are mixed

**Why This Matters**:
- A 1% improvement in fraud detection can save millions at scale
- False positives create customer friction and operational costs
- Borderline cases require the most manual review—automating these decisions has high ROI

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      NEW APPLICATION                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 STAGE 1: STRUCTURED MODEL                    │
│                                                              │
│  • 23 engineered features (velocity, tenure, risk flags)    │
│  • LightGBM classifier                                       │
│  • 99.5% ROC-AUC, 96.7% precision, 89% recall               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  CONFIDENT?     │
                    │  Score < 0.01   │
                    │  or > 0.99?     │
                    └─────────────────┘
                     /              \
                   YES              NO
                   /                  \
                  ▼                    ▼
    ┌──────────────────┐    ┌──────────────────────────┐
    │ USE STAGE 1      │    │ STAGE 2: TEXT ANALYSIS   │
    │ (97.7% of cases) │    │ (2.3% of cases)          │
    └──────────────────┘    │                          │
                            │ • MiniLM encoder         │
                            │ • OCR similarity         │
                            │ • Keyword detection      │
                            │ • Combined decision      │
                            └──────────────────────────┘
```

**Why Two Stages?**
- Stage 1 handles clear-cut cases efficiently
- Stage 2 applies expensive text analysis only where needed
- Borderline routing limits computational cost and potential noise

---

## Dataset Design

### Why Synthetic Data?

Real fraud data is:
- Highly sensitive (PII, regulatory constraints)
- Imbalanced (often <1% fraud)
- Proprietary (not shareable for portfolio projects)

Synthetic data allows:
- Full control over fraud patterns and difficulty
- Reproducible experiments

### Dataset Specifications

| Attribute | Value |
|-----------|-------|
| Total Applications | 5,000 |
| Fraud Rate | 12% |
| Time Span | 12 months |
| Train/Val/Test Split | 70%/15%/15% (time-based) |

### Fraud Archetypes

| Type | % of Data | Description |
|------|-----------|-------------|
| Legitimate Clean | 70% | Normal applications, consistent data |
| Legitimate Noisy | 18% | Real customers with messy data (recent moves, typos) |
| Synthetic Identity | 5% | Fabricated identities, stitched-together data |
| True-Name Fraud | 4% | Real identity, fraudulent intent |
| Coordinated Attack | 3% | Organized fraud rings, shared infrastructure |

---

## Phase Summary

| Phase | Description | Key Output |
|-------|-------------|------------|
| 1-3 | Project setup, dataset design, source dictionaries | Design docs, lookup files |
| 4 | Synthetic data generator | 5,000 applications |
| 5 | Data quality / EDA | Cleaned dataset |
| 6 | Feature engineering | 23 structured features |
| 7 | Baseline models | LightGBM (0.995 ROC-AUC) |
| 8 | Borderline band definition | 117 uncertain cases identified |
| 9 | Encoder/text features | 12 text-based features |
| 10 | Combined model | Ablation study |
| 11 | Validation | Comprehensive evaluation |
| 12 | Interpretation | Final conclusions |

---

## Models Used

### Stage 1: Structured Models

| Model | Test ROC-AUC | Test PR-AUC | Notes |
|-------|--------------|-------------|-------|
| **LightGBM** | **0.995** | **0.974** | **Selected for production** |
| XGBoost | 0.992 | 0.960 | Close second |
| Logistic Regression | 0.988 | 0.944 | Interpretable baseline |

### Stage 2: Text/Encoder Model

| Component | Description |
|-----------|-------------|
| Encoder | all-MiniLM-L6-v2 (sentence-transformers) |
| Features | OCR similarity, consistency scores, keyword counts |
| Combiner | Logistic Regression on Stage 1 score + text features |

---

## Validation Approach

### Ablation Study

Tested four configurations to isolate the value of each component:

| Setup | ROC-AUC | Verdict |
|-------|---------|---------|
| Structured Only | 0.995 | **Best** |
| Text Only | 0.818 | Weak alone |
| Combined (All Cases) | 0.957 | **Hurts performance** |
| Borderline Routed | 0.993 | Preserves performance |

### Additional Validation

- **Threshold analysis**: Precision/recall at multiple operating points
- **Calibration**: Brier score and calibration curves
- **Stability**: Performance across time periods
- **Runtime**: Latency measurements for production feasibility

---

## Key Findings

### 1. Structured Features Capture Most Signal

The engineered features are highly discriminative:
- `high_identity_reuse_flag`: 67% fraud rate when true vs 1.4% when false
- `high_device_velocity_flag`: Appears only in fraud cases
- `night_application_flag`: 7x more common in fraud

### 2. Text Features Add No Incremental Value

- Adding text to all cases **decreases** ROC-AUC from 0.995 to 0.957
- Even on borderline cases, combined model shows no improvement
- Text features are correlated with structured features (redundant signal)

### 3. Borderline Routing Is the Safest Architecture

If text must be used:
- Apply only to uncertain cases (2.3% of traffic)
- Limits potential "damage" from noisy features
- Preserves most of Stage 1 performance (0.993 vs 0.995)

### 4. Strong Baselines Make Improvement Hard

With 99.5% ROC-AUC:
- Only 14 errors on 835 test cases
- Very little room for improvement
- Small borderline set (45 test cases) limits statistical power

---

## Final Conclusion

**Recommendation: Use Stage 1 (LightGBM) alone.**

The two-stage architecture adds complexity without improving performance. This is a valuable finding:
- Demonstrates that strong feature engineering can eliminate the need for complex NLP
- Shows disciplined experimental methodology
- Proves that "more features" doesn't always mean "better model"

**The methodology is transferable** to any problem where you need to validate whether added complexity provides value.

---

## Project Structure

```
identity-fraud-project/
├── data/
│   ├── external/           # Source dictionaries
│   ├── raw/                # Generated raw data
│   └── processed/          # Final datasets
├── notebooks/
│   ├── 02_data_generation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_borderline_band.ipynb
│   ├── 07_encoder_features.ipynb
│   ├── 08_final_combined_model.ipynb
│   └── 09_model_validation.ipynb
├── src/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── train_baseline_models.py
│   ├── define_borderline_band.py
│   ├── encoder_features.py
│   ├── train_final_combined_model.py
│   └── validate_models.py
├── models/
│   └── lightgbm_model.pkl  # Production model
├── reports/
│   ├── final_project_summary.md
│   ├── project_interpretation.md
│   └── slides/
└── requirements.txt
```

---

## How to Run

```bash
# 1. Clone and setup
git clone <repo-url>
cd identity-fraud-project
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data
python src/data_generation.py

# 4. Run feature engineering
python src/feature_engineering.py

# 5. Train baseline models
python src/train_baseline_models.py

# 6. Generate text features (requires ~80MB model download)
python src/encoder_features.py

# 7. Train combined model and run ablation
python src/train_final_combined_model.py

# 8. Run comprehensive validation
python src/validate_models.py
```

---

## Limitations

### Dataset Limitations
- **Synthetic data**: Real fraud patterns may differ
- **Small scale**: 5,000 rows; production systems see millions
- **Limited text variation**: Template-based generation

### Methodological Limitations
- **Single encoder**: Only tested MiniLM
- **Simple combiner**: Only Logistic Regression for Stage 2
- **Fixed thresholds**: Did not optimize borderline band

### What We Cannot Conclude
- Text features would never help on any fraud dataset
- More sophisticated NLP wouldn't improve results
- Results would hold at production scale

---

## Future Improvements

1. **Larger dataset** (50K-100K rows) with more borderline cases
2. **Richer text simulation** with realistic OCR errors
3. **Alternative encoders** (E5, BGE, domain-specific)
4. **Training the Encoders**
5. **Cost-sensitive evaluation** (different costs for FP vs FN)
6. **Real data validation** if available

See `reports/future_improvements.md` for detailed roadmap.

---

## Author

Built as a portfolio project demonstrating end-to-end ML system design, rigorous experimentation, and honest evaluation of results.

**Key Skills Demonstrated**:
- Feature engineering for fraud detection
- Two-stage ML system architecture
- Ablation studies and model validation
- Handling negative experimental results professionally
