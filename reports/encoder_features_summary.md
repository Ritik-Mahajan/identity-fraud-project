# Phase 9: Encoder Features Summary

## Overview

This document summarizes the text-based encoder features created for Stage 2 fraud detection.

**Date**: Phase 9 Complete  
**Model**: all-MiniLM-L6-v2 (sentence-transformers)  
**Input**: Borderline cases from Phase 8 (117 rows)

---

## Encoder Model

| Property | Value |
|----------|-------|
| Model name | `all-MiniLM-L6-v2` |
| Framework | sentence-transformers |
| Embedding dimension | 384 |
| Model size | ~80 MB |
| Inference | CPU (no GPU required) |

**Why this model?**
- Lightweight and fast
- High-quality sentence embeddings
- Runs efficiently on a 16GB MacBook Air
- No fine-tuning required

---

## Input Data

| Item | Value |
|------|-------|
| Input file | `data/processed/borderline_cases.parquet` |
| Rows | 117 (borderline cases from Phase 8) |
| Fraud rate | 39.3% |
| Text fields used | verification_note, ocr_document_text, address_explanation_text, employment_explanation_text |

---

## Features Created

### 1. Similarity Features (3)

These use the encoder to compute semantic similarity between application fields and text evidence.

| Feature | Description | Logic |
|---------|-------------|-------|
| `application_ocr_similarity` | Similarity between claimed identity and OCR text | Cosine similarity of embeddings |
| `employment_consistency_score` | Similarity between employer info and employment explanation | Cosine similarity of embeddings |
| `address_consistency_score` | Similarity between address info and address explanation | Cosine similarity of embeddings |

**Reference text construction:**

- **Application identity text**: `"Name {first_name} {last_name} Address {address_line} {city} {state} {zip_code} Employer {employer_name}"`
- **Address reference text**: `"Address {address_line} {city} {state} {zip_code}"`
- **Employer reference text**: `"Employer {employer_name} in {industry} industry"`

### 2. Text Length Features (4)

| Feature | Description |
|---------|-------------|
| `verification_note_length` | Character count of verification note |
| `ocr_text_length` | Character count of OCR document text |
| `address_explanation_length` | Character count of address explanation |
| `employment_explanation_length` | Character count of employment explanation |

### 3. Keyword Features (5)

| Feature | Description |
|---------|-------------|
| `suspicious_keyword_count_verification` | Count of suspicious keywords in verification note |
| `suspicious_keyword_count_ocr` | Count of suspicious keywords in OCR text |
| `suspicious_keyword_count_total` | Total suspicious keyword count |
| `note_has_high_risk_keyword_flag` | 1 if verification note has any suspicious keyword |
| `ocr_has_high_risk_keyword_flag` | 1 if OCR text has any suspicious keyword |

---

## Suspicious Keyword List

The following keywords are used to detect suspicious patterns:

1. unable
2. mismatch
3. inconsistent
4. low quality
5. unreadable
6. multiple applicants
7. reused
8. unverifiable
9. discrepancy
10. suspicious
11. differs
12. incomplete
13. not match
14. cannot verify
15. no record

---

## Feature Analysis by Fraud Label

| Feature | Legitimate (0) | Fraud (1) | Difference |
|---------|----------------|-----------|------------|
| `application_ocr_similarity` | 0.565 | 0.370 | **-0.195** |
| `employment_consistency_score` | 0.335 | 0.336 | +0.001 |
| `address_consistency_score` | 0.235 | 0.269 | +0.034 |
| `verification_note_length` | 48.4 | 50.8 | +2.4 |
| `ocr_text_length` | 64.8 | 56.3 | -8.5 |
| `suspicious_keyword_count_verification` | 0.11 | 0.24 | **+0.13** |
| `suspicious_keyword_count_ocr` | 0.16 | 0.57 | **+0.41** |
| `suspicious_keyword_count_total` | 0.27 | 0.80 | **+0.54** |
| `note_has_high_risk_keyword_flag` | 11.3% | 23.9% | **+12.6%** |
| `ocr_has_high_risk_keyword_flag` | 12.7% | 54.3% | **+41.7%** |

### Key Findings

1. **Application-OCR similarity is discriminative:**
   - Fraud cases have 0.370 similarity vs 0.565 for legitimate
   - This confirms that fraudulent OCR documents don't match claimed identity

2. **Suspicious keywords are highly discriminative:**
   - Fraud cases have 3x more suspicious keywords in OCR text (0.57 vs 0.16)
   - 54% of fraud cases have high-risk keywords in OCR vs only 13% of legitimate

3. **Employment and address consistency are less discriminative:**
   - These features show minimal difference between fraud and legitimate
   - This makes sense - explanations may be similar regardless of fraud status

---

## Output Files

| File | Size | Description |
|------|------|-------------|
| `data/processed/text_encoder_features.parquet` | 32.8 KB | Feature table (Parquet) |
| `data/processed/text_encoder_features.csv` | 45.1 KB | Feature table (CSV) |

### Output Columns

**Metadata (11):**
- application_id, application_date, application_month
- fraud_label, fraud_type, difficulty_level
- generated_signal_score, best_model_score, best_model_pred
- borderline_flag, split_label

**Similarity features (3):**
- application_ocr_similarity
- employment_consistency_score
- address_consistency_score

**Length features (4):**
- verification_note_length
- ocr_text_length
- address_explanation_length
- employment_explanation_length

**Keyword features (5):**
- suspicious_keyword_count_verification
- suspicious_keyword_count_ocr
- suspicious_keyword_count_total
- note_has_high_risk_keyword_flag
- ocr_has_high_risk_keyword_flag

**Text fields (4):**
- verification_note
- ocr_document_text
- address_explanation_text
- employment_explanation_text

---

## Implementation Notes

1. **Batch encoding**: All texts are encoded in batches for efficiency
2. **Local cache**: Model is cached in `.model_cache/` directory
3. **No fine-tuning**: Model is used as-is for inference only
4. **CPU inference**: No GPU required

---

## Next Steps (Phase 10)

1. Combine Stage 1 structured scores with Stage 2 text features
2. Train a combined model on borderline cases
3. Evaluate whether text features improve predictions
4. Compare structured-only vs combined approaches

---

## Files Created

| File | Description |
|------|-------------|
| `src/encoder_features.py` | Encoder features module (~500 lines) |
| `notebooks/07_encoder_features.ipynb` | Interactive notebook |
| `data/processed/text_encoder_features.parquet` | Feature table |
| `data/processed/text_encoder_features.csv` | Feature table (CSV) |
| `reports/encoder_features_summary.md` | This summary |
