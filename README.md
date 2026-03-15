# Selective Encoder-Enhanced Identity Fraud Detection for Application Risk

## Project Overview

This project builds and evaluates a **two-stage fraud detection system** for application/identity fraud. The hypothesis was that lightweight text/encoder features could improve fraud detection on borderline cases where a structured model is uncertain.

**Key Finding**: The structured model alone achieves **99.5% ROC-AUC**, and adding text features provides no incremental value. This is a valid negative result that demonstrates rigorous experimental methodology.

## Quick Results

| Model | Test ROC-AUC | Test PR-AUC | Precision | Recall |
|-------|--------------|-------------|-----------|--------|
| **LightGBM (Stage 1)** | **0.995** | **0.974** | **96.7%** | **89.0%** |
| Borderline Routed | 0.993 | 0.966 | 96.7% | 89.0% |
| Combined (All Cases) | 0.957 | 0.948 | 96.7% | 89.0% |

**Recommendation**: Use Stage 1 (LightGBM) alone. The two-stage architecture adds complexity without improving performance.

## System Architecture

```
Application → Stage 1 (LightGBM) → Confidence Check
                                       ↓
                   High Confidence → Use Stage 1 prediction (97.7% of cases)
                   Low Confidence  → Stage 2 (Text) → Combined prediction (2.3%)
```

## Project Structure

```
identity-fraud-project/
├── data/
│   ├── external/          # Source dictionaries for data generation
│   ├── raw/               # Generated raw dataset
│   ├── interim/           # Intermediate data
│   └── processed/         # Final processed datasets
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
│   ├── data_quality_checks.py
│   ├── feature_engineering.py
│   ├── train_baseline_models.py
│   ├── define_borderline_band.py
│   ├── encoder_features.py
│   ├── train_final_combined_model.py
│   └── validate_models.py
├── models/
│   ├── lightgbm_model.pkl
│   ├── logistic_regression.pkl
│   ├── xgboost_model.pkl
│   └── final_combined_logistic_regression.pkl
├── reports/
│   ├── project_interpretation.md      # Detailed interpretation
│   ├── final_project_summary.md       # Executive summary
│   ├── interview_talking_points.md    # Interview prep
│   ├── model_validation_summary.md    # Validation findings
│   ├── future_improvements.md         # Next steps
│   ├── tables/                        # CSV metrics
│   ├── figures/                       # Visualizations
│   └── slides/                        # Presentation outline
├── requirements.txt
└── README.md
```

## Phase Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Project setup | Complete |
| Phase 2 | Dataset design | Complete |
| Phase 3 | Source dictionaries | Complete |
| Phase 4 | Synthetic data generator | Complete |
| Phase 5 | Data quality / EDA | Complete |
| Phase 6 | Feature engineering | Complete |
| Phase 7 | Baseline structured models | Complete |
| Phase 8 | Borderline band definition | Complete |
| Phase 9 | Encoder/text features | Complete |
| Phase 10 | Final combined model | Complete |
| Phase 11 | Model validation | Complete |
| Phase 12 | Interpretation / refinement | Complete |
| Phase 13 | Demo and packaging | Pending |

## Key Findings

### 1. Structured Features Are Sufficient

The engineered features capture most fraud signal:
- `high_identity_reuse_flag`: 67% fraud rate when true (vs 1.4% when false)
- `high_device_velocity_flag`: Only appears in fraud cases
- `night_application_flag`: 7x higher in fraud

### 2. Text Features Provide No Incremental Value

- Adding text to all cases **hurts** performance (0.995 → 0.957 ROC-AUC)
- Borderline routing preserves performance but doesn't improve it
- Text features are correlated with structured features

### 3. Ablation Study Results

| Setup | ROC-AUC | Verdict |
|-------|---------|---------|
| Structured Only | 0.995 | **Best** |
| Text Only | 0.818 | Weak |
| Combined (All) | 0.957 | Hurts |
| Borderline Routed | 0.993 | Preserves |

## Dataset

- **5,000 applications** with 12% fraud rate
- **5 fraud archetypes**: legitimate_clean, legitimate_noisy, synthetic_identity, true_name_fraud, coordinated_attack
- **41 columns** covering identity, address, contact, employment, digital behavior, and text fields
- **Time-based split**: Train (70%), Validation (15%), Test (15%)

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd identity-fraud-project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the data generator
python src/data_generation.py

# Train baseline models
python src/train_baseline_models.py

# Run validation
python src/validate_models.py
```

## Key Files

| File | Description |
|------|-------------|
| `models/lightgbm_model.pkl` | Best Stage 1 model (recommended for production) |
| `data/processed/final_model_predictions.parquet` | All predictions with ablation study |
| `reports/project_interpretation.md` | Detailed project interpretation |
| `reports/interview_talking_points.md` | Interview preparation guide |
| `notebooks/09_model_validation.ipynb` | Comprehensive validation notebook |

## Constraints

- Runs on a **16GB MacBook Air M4**
- No LLM training from scratch
- Lightweight tools only (pandas, numpy, scikit-learn, sentence-transformers)
- Beginner-friendly, modular, well-commented code

## What I Learned

1. **Strong feature engineering can eliminate the need for complex models**
2. **Ablation studies are essential before adding complexity**
3. **Negative results are valuable when properly validated**
4. **Not every problem needs deep learning or NLP**

## Future Improvements

See `reports/future_improvements.md` for detailed next steps:
1. Generate larger dataset with more borderline cases
2. Richer text field simulation
3. Alternative encoder models
4. Cost-sensitive evaluation

## License

For educational and research purposes.
