# Phase 8: Borderline Band Definition

## Overview

This document defines the borderline band for Stage 2 text/encoder analysis. Cases in this band are where the Stage 1 structured model is uncertain, and text features might provide additional signal.

**Date**: Phase 8 Complete  
**Model Used**: LightGBM (best from Phase 7)  
**Final Band**: 0.01 to 0.99

---

## Why Define a Borderline Band?

The Stage 1 LightGBM model is **very confident** on most cases:
- 87.9% of cases have scores < 0.10 (confident legitimate)
- 11.4% of cases have scores > 0.90 (confident fraud)
- Only 2.3% of cases fall in the uncertain middle region

The borderline band captures the truly uncertain cases where:
1. The structured model can't make a confident decision
2. Text/semantic features might provide additional signal
3. The model has higher error rates

---

## Candidate Bands Evaluated

| Band | Rows | % Total | Fraud Rate | Error Rate | Hard | Medium | Easy |
|------|------|---------|------------|------------|------|--------|------|
| **0.01-0.99** | **117** | **2.34%** | **39.3%** | **20.5%** | **62** | **41** | **14** |
| 0.05-0.95 | 46 | 0.92% | 52.2% | 30.4% | 29 | 13 | 4 |
| 0.10-0.90 | 31 | 0.62% | 61.3% | 38.7% | 21 | 10 | 0 |
| 0.15-0.85 | 28 | 0.56% | 57.1% | 42.9% | 19 | 9 | 0 |
| 0.20-0.80 | 23 | 0.46% | 60.9% | 47.8% | 16 | 7 | 0 |

---

## Final Selected Band: 0.01 to 0.99

### Band Statistics

| Metric | Value |
|--------|-------|
| Score range | 0.01 to 0.99 |
| Cases in band | 117 |
| Percent of total | 2.34% |
| Fraud rate | 39.3% |
| Legitimate rate | 60.7% |
| Error rate | 20.5% (24 errors) |
| Score mean | 0.3376 |
| Score std | 0.3910 |

### Difficulty Distribution

| Difficulty | Count | Percent |
|------------|-------|---------|
| Hard | 62 | 53.0% |
| Medium | 41 | 35.0% |
| Easy | 14 | 12.0% |

### Fraud Type Distribution

| Fraud Type | Count | Percent |
|------------|-------|---------|
| legitimate_noisy | 68 | 58.1% |
| true_name_fraud | 32 | 27.4% |
| synthetic_identity | 14 | 12.0% |
| legitimate_clean | 3 | 2.6% |

---

## Justification for Band Selection

### Why 0.01-0.99?

1. **Contains enough cases (117)** for meaningful Stage 2 analysis
   - Narrower bands have too few cases (23-46) for reliable evaluation

2. **High error rate (20.5%)** - These are truly uncertain cases
   - Overall model error rate is ~1%
   - Borderline band has 20x higher error rate

3. **Appropriate difficulty mix** - 88% non-easy cases
   - 62 hard cases + 41 medium cases
   - Only 14 easy cases (likely edge cases)

4. **Right fraud types present** - The ambiguous types
   - `legitimate_noisy` (68): Legitimate but with suspicious signals - text might confirm legitimacy
   - `true_name_fraud` (32): Fraud with clean signals - text inconsistencies might reveal fraud

5. **Balanced fraud rate (~40%)** - Good for analysis
   - Not too skewed toward fraud or legitimate
   - Allows evaluation of both false positives and false negatives

### Why Not Narrower Bands?

- **0.05-0.95**: Only 46 cases - too few for meaningful analysis
- **0.10-0.90**: Only 31 cases - too few
- **0.20-0.80**: Only 23 cases - too few

The 0.01-0.99 band captures all cases where the model has any meaningful uncertainty.

---

## Can Text Features Help?

### Analysis of Borderline Cases

The borderline band contains exactly the types of cases where text/semantic features should help:

1. **legitimate_noisy (68 cases)**
   - These have suspicious structured signals but are actually legitimate
   - Text might confirm legitimacy (e.g., "Employer verified through callback")
   - Semantic consistency between text fields might indicate authenticity

2. **true_name_fraud (32 cases)**
   - These have relatively clean structured signals but are actually fraud
   - Text inconsistencies might reveal fraud (e.g., "Address on document differs from application")
   - OCR text mismatches might indicate document manipulation

3. **synthetic_identity (14 cases)**
   - Text fields might show templated/generic patterns
   - Semantic inconsistencies between claimed identity and document text

### Expected Text Signal Patterns

| Text Field | Legitimate Signal | Fraud Signal |
|------------|-------------------|--------------|
| verification_note | "Verified", "Confirmed", "Consistent" | "Unable to verify", "Inconsistent", "Mismatch" |
| ocr_document_text | Matches claimed name/address | Differs from application, incomplete |
| address_explanation_text | Clear, specific explanation | Vague, inconsistent |
| employment_explanation_text | Verifiable details | Unverifiable, generic |

---

## Borderline Cases Output

### File Location

`data/processed/borderline_cases.parquet`

### Columns Included

| Column | Description |
|--------|-------------|
| application_id | Unique identifier |
| application_date | Application date |
| application_month | Month (YYYY-MM) |
| fraud_label | True label (0/1) |
| fraud_type | Fraud archetype |
| difficulty_level | easy/medium/hard |
| generated_signal_score | Latent signal score |
| best_model_score | LightGBM probability |
| best_model_pred | LightGBM prediction (0/1) |
| borderline_band_low | 0.01 |
| borderline_band_high | 0.99 |
| verification_note | Verification note text |
| ocr_document_text | OCR document text |
| address_explanation_text | Address explanation |
| employment_explanation_text | Employment explanation |
| split_label | train/val/test |
| + other prediction columns | LR, XGBoost scores |

---

## Summary

The 0.01-0.99 borderline band:

- Contains **117 uncertain cases** (2.3% of data)
- Has **20.5% error rate** (vs 1% overall)
- Is **88% hard/medium difficulty** cases
- Contains **legitimate_noisy and true_name_fraud** - the ambiguous types
- Is **suitable for Stage 2 text/encoder analysis**

---

## Next Steps (Phase 9)

1. Create text/encoder features for borderline cases
2. Use lightweight encoder (e.g., MiniLM) for semantic embeddings
3. Create text consistency features:
   - Name consistency between claimed name and OCR text
   - Address consistency between claimed address and document text
   - Semantic similarity between explanation fields
4. Evaluate whether text features improve predictions on borderline cases

---

## Files Created

| File | Description |
|------|-------------|
| `src/define_borderline_band.py` | Borderline band definition module |
| `notebooks/06_borderline_band.ipynb` | Interactive analysis notebook |
| `data/processed/borderline_cases.parquet` | Borderline cases with text fields |
| `reports/borderline_band_definition.md` | This summary |
