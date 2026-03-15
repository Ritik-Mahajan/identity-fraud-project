# Retrofit Alignment Audit

## Overview

This document audits the existing implementation against the intended design philosophy and generation logistics that were not explicitly documented during initial development.

**Audit Date**: Phase 5 Retrofit  
**Auditor**: Automated review of existing code and documentation

---

## Audit Summary

| Category | Status |
|----------|--------|
| Overall Alignment | **Strong** |
| Regeneration Required | **No** |
| Minor Improvements Made | Documentation only |

---

## What Already Aligns Well

### 1. Believable Row Structure ✓

**Intended**: Each row should represent a believable application, not random values.

**Implementation**: The `generate_base_identity()` function creates coherent profiles by:
- Sampling from realistic name dictionaries (94 first names, 100 last names)
- Using consistent city/state/ZIP combinations (44 locations)
- Generating age-appropriate income based on industry
- Creating plausible street addresses

**Evidence**: EDA confirmed realistic distributions:
- Age range: 18-75 (mean ~38)
- Income range: $22K-$200K (industry-appropriate)
- Tenure: 0-120 months (fraud has lower values)

### 2. All Major Field Groups ✓

**Intended**: Dataset should include identity, behavioral, digital footprint, text evidence, and labels.

**Implementation**: All 40+ columns are present across all field groups:
- Identity: 5 columns (name, DOB, age, SSN)
- Address: 4 columns
- Contact: 4 columns
- Employment/Financial: 7 columns
- Digital/Behavioral: 8 columns
- Text: 4 columns
- Labels/Meta: 4 columns

**Evidence**: Schema matches `reports/dataset_schema.md` exactly.

### 3. Fraud Archetypes ✓

**Intended**: Five distinct archetypes with different signal patterns.

**Implementation**: Separate generator functions for each:
- `generate_legitimate_clean()` - Low risk, consistent
- `generate_legitimate_noisy()` - Legitimate with ambiguity
- `generate_synthetic_identity()` - Fabricated identity
- `generate_true_name_fraud()` - Stolen identity
- `generate_coordinated_attack()` - Cluster-based

**Evidence**: Distribution matches design:
- legitimate_clean: 70% (target: 70%)
- legitimate_noisy: 18% (target: 18%)
- synthetic_identity: 5% (target: 5%)
- true_name_fraud: 4% (target: 4%)
- coordinated_attack: 3% (target: 3%)

### 4. Ambiguity / Hard Cases ✓

**Intended**: Dataset should include borderline cases by design.

**Implementation**: 
- Difficulty levels assigned per archetype
- Signal intensity varies by difficulty
- Hard fraud rows have cleaner signals
- Legitimate noisy rows have suspicious signals

**Evidence**: EDA confirmed:
- 758 hard cases (15.2%)
- 900 legitimate_noisy rows
- 93 true_name_fraud hard cases
- ~800 rows with middle signal scores (0.3-0.6)

### 5. Reusable Attack Pools ✓

**Intended**: Maintain suspicious device/phone/address pools for fraud.

**Implementation**: `shared_pools` dictionary contains:
```python
shared_pools = {
    "fraud_devices": [f"DEV-FRAUD-{i:04d}" for i in range(50)],
    "cluster_devices": {...},  # Per-cluster devices
}
```

**Evidence**: Code at lines 1206-1215 in `data_generation.py` implements this exactly.

### 6. Text Generation Logic ✓

**Intended**: Template-based generation with placeholder substitution.

**Implementation**:
- 12 template CSV files in `data/external/`
- `fill_template()` function handles placeholder substitution
- Template selection varies by fraud type and difficulty

**Evidence**: Text fields are populated (no nulls, mean length 50-65 chars).

### 7. OCR Mismatch Logic ✓

**Intended**: Fraud OCR should contain mismatches; legitimate should mostly align.

**Implementation**:
- Separate `legit_ocr_templates.csv` and `fraud_ocr_templates.csv`
- `generate_ocr_text()` selects templates based on fraud type
- Hard fraud uses mixed templates (50% fraud)
- Legitimate noisy occasionally uses fraud templates (15%)

**Evidence**: Template files contain appropriate patterns:
- Legit: "Name {first_name} {last_name} Address {address_line}..."
- Fraud: "Name partially unreadable address mismatch"

### 8. Latent Signal Score Logic ✓

**Intended**: Fraud should arise from weighted combinations of signals.

**Implementation**: `compute_generated_signal_score()` uses weighted logic:
- Device reuse: +0.15 (≥3) or +0.08 (≥1)
- Phone reuse: +0.10 (≥2)
- Address reuse: +0.12 (≥3)
- Free email: +0.05
- Low name match: +0.10 (<0.3)
- Low tenure: +0.08 (<6 months)
- Thin file: +0.10
- ZIP/IP mismatch: +0.12 (>0.7)

**Evidence**: Signal score correlates with fraud (fraud mean higher than legit mean).

### 9. Time Structure Across Months ✓

**Intended**: Dataset should span 12 months for time-based validation.

**Implementation**:
- `START_DATE = datetime(2024, 1, 1)`
- `END_DATE = datetime(2024, 12, 31)`
- `generate_application_date()` samples uniformly

**Evidence**: EDA confirmed dates span 2024-01-01 to 2024-12-31 with 12 unique months.

### 10. Hidden Metadata Preservation ✓

**Intended**: Preserve fraud_type, difficulty_level, generated_signal_score.

**Implementation**:
- All three fields included in main dataset
- Separate metadata file saved to `data/interim/generation_metadata.csv`

**Evidence**: Files exist and contain correct columns.

---

## What Partially Aligns

### 1. Phone/Address Pool Reuse (Partial)

**Intended**: Maintain explicit suspicious phone/address pools.

**Implementation**: 
- Device pools are explicit (✓)
- Phone/address reuse is simulated via counts, not explicit pools

**Assessment**: This is acceptable. The reuse counts achieve the same effect without requiring complex pool management. The EDA confirmed fraud has higher reuse counts.

**Recommendation**: No change needed. Document this as a simplification.

### 2. Coordinated Attack Clustering (Partial)

**Intended**: Clusters of 3-10 applications with shared infrastructure.

**Implementation**:
- Cluster device IDs are shared
- Cluster counter assigns ~5 apps per cluster
- Phone/address sharing is simulated via high reuse counts

**Assessment**: The implementation captures the essence but doesn't create true sequential clusters with shared phones/addresses.

**Recommendation**: Acceptable for prototype. Could enhance in future versions.

### 3. Time Evolution of Fraud Patterns (Partial)

**Intended**: Fraud patterns can shift over time slightly.

**Implementation**: 
- Dates are uniformly distributed
- No explicit monthly variation in fraud patterns

**Assessment**: The 12-month span supports time-based splits, but fraud patterns don't evolve over time.

**Recommendation**: Acceptable for prototype. Could add monthly variation in future.

---

## What is Missing

### 1. Explicit Nickname Handling (Minor Gap)

**Intended**: legitimate_noisy should include nickname vs formal name scenarios.

**Implementation**: 
- Email mismatch is implemented via `match_name=False`
- No explicit nickname mapping (e.g., "Michael" → "Mike")

**Assessment**: The effect is achieved through email mismatch, but explicit nickname logic would be more realistic.

**Recommendation**: Document as future enhancement. Not critical for prototype.

### 2. Income Unrealism Signal (Minor Gap)

**Intended**: Fraud might have unrealistic income for age/employer.

**Implementation**:
- Income is generated realistically for all rows
- No explicit unrealistic income for fraud

**Assessment**: This is a minor signal that wasn't implemented. Other signals compensate.

**Recommendation**: Could add in future. Not critical.

---

## Minimal Corrections Made

### Documentation Updates Only

No code changes were required. The following documentation was added:

1. **`reports/dataset_design_philosophy.md`** - Captures design principles
2. **`reports/synthetic_data_logistics.md`** - Documents generation pipeline
3. **`reports/retrofit_alignment_audit.md`** - This audit document

---

## Future Improvements (Optional)

These are not required for the current project but could enhance future versions:

| Improvement | Priority | Complexity |
|-------------|----------|------------|
| Explicit nickname mapping | Low | Low |
| Monthly fraud rate variation | Low | Medium |
| True phone/address pool sharing | Low | Medium |
| Income unrealism for fraud | Low | Low |
| More OCR messiness variations | Low | Low |

---

## Regeneration Decision

### Is the current generated dataset acceptable?

**YES** - The dataset is acceptable for continuing the project.

### Reasons:

1. **All core design elements are implemented**
   - 5 fraud archetypes with correct distributions
   - All 40+ columns present
   - Difficulty levels working correctly

2. **All fraud signals show expected patterns**
   - 6/6 directional checks passed in EDA
   - Device reuse higher in fraud
   - Tenure lower in fraud
   - Name/email match lower in fraud

3. **Sufficient borderline cases exist**
   - 758 hard cases (15.2%)
   - 900 legitimate_noisy rows
   - Adequate ambiguity for Stage 2 testing

4. **Data quality is excellent**
   - No missing values
   - No duplicates
   - All values valid

5. **Partial alignments are acceptable simplifications**
   - Simulated reuse counts achieve same effect as explicit pools
   - Uniform time distribution supports time-based splits

### Recommendation:

**Continue with the existing dataset.** No regeneration is needed.

---

## Audit Conclusion

The existing implementation is **well-aligned** with the intended design philosophy and generation logistics. The development team implemented the core concepts correctly without having the explicit design documents.

Key strengths:
- Modular, well-commented code
- Correct fraud archetype distributions
- Working difficulty-based signal intensity
- Proper attack pool implementation
- Comprehensive text generation

The minor gaps identified are acceptable simplifications for a prototype dataset and do not impact the project's ability to proceed with modeling.
