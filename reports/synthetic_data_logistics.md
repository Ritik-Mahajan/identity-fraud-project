# Synthetic Data Generation Logistics

## Overview

This document describes the detailed logistics for creating the synthetic-but-realistic dataset. It serves as a reference for understanding how the data generator works.

---

## Generation Pipeline

The data generation follows a 10-step conceptual pipeline:

```
1. Build canonical applicant profile
2. Add digital/application behavior
3. Generate text fields
4. Create OCR text with mild messiness
5. Apply fraud-generation transformations
6. Use attack pools for reuse patterns
7. Derive labels from latent signals
8. Inject realistic noise
9. Include time evolution
10. Save row-level data and hidden truth
```

---

## Step 1: Build Canonical Applicant Profile

Generate a plausible clean identity with:

| Field | Source |
|-------|--------|
| `claimed_first_name` | `first_names.csv` dictionary |
| `claimed_last_name` | `last_names.csv` dictionary |
| `date_of_birth` / `age` | Generated (18-75 range, weighted toward working age) |
| `address_line` | Street number + `street_names.csv` |
| `city` / `state` / `zip_code` | `cities_states_zips.csv` dictionary |
| `phone_number` | Generated US format (XXX-XXX-XXXX) |
| `email` | Derived from name + domain |
| `employer_name` / `employer_industry` | `employers.csv` dictionary |
| `annual_income` | Generated based on industry and age |
| `housing_status` | Sampled (rent/own/family/other) |
| `months_at_address` | Generated based on fraud type |
| `months_at_employer` | Generated based on fraud type |

---

## Step 2: Add Digital/Application Behavior

Generate behavioral fields:

| Field | Logic |
|-------|-------|
| `device_id` | Unique or from shared pool (fraud) |
| `ip_region` | Usually matches state, sometimes different |
| `application_hour` | 0-23, weighted by fraud type |
| `num_prev_apps_same_device_7d` | Low for legit, high for fraud |
| `num_prev_apps_same_email_30d` | Low for legit, higher for fraud |
| `num_prev_apps_same_phone_30d` | Low for legit, higher for fraud |
| `num_prev_apps_same_address_30d` | Low for legit, higher for fraud |

---

## Step 3: Generate Text Fields

Use template-based generation with placeholders:

### Templates Used

| Field | Template Source |
|-------|-----------------|
| `verification_note` | `legit_verification_note_templates.csv` or `fraud_verification_note_templates.csv` |
| `ocr_document_text` | `legit_ocr_templates.csv` or `fraud_ocr_templates.csv` |
| `address_explanation_text` | `address_explanation_templates.csv` |
| `employment_explanation_text` | `employment_explanation_templates.csv` |

### Placeholder Substitution

Templates contain placeholders like `{first_name}`, `{city}`, `{employer_name}` that are filled with actual row values.

### Template Selection Logic

- **Legitimate clean**: Almost always uses legit templates
- **Legitimate noisy**: Mostly legit, occasionally fraud-like (15-20%)
- **Fraud (easy)**: Mostly fraud templates (80-85%)
- **Fraud (hard)**: Mixed templates (50-60% fraud)

---

## Step 4: Create OCR Text with Mild Messiness

OCR text simulates document scanning output:

### Legitimate OCR Patterns
- Mostly aligns with claimed fields
- May have minor formatting variations
- Examples: "Name Michael Carter Address 411 West Pine St Springfield IL"

### Fraud OCR Patterns
- Contains mismatches with application data
- Missing details or low quality cues
- Examples: "Name partially unreadable address mismatch"

---

## Step 5: Apply Fraud-Generation Transformations

Instead of generating fraud from scratch, conceptually transform base identities:

### Synthetic Identity Transformation
- Keep identity elements but make them inconsistent
- Increase reuse counts
- Lower tenure values
- Set thin_file_flag more often
- Use fraud OCR templates

### True Name Fraud Transformation
- Keep identity mostly intact (stolen identity)
- Add behavioral anomalies (velocity, IP mismatch)
- Use subtle text inconsistencies
- Maintain realistic tenure (harder to detect)

### Coordinated Attack Transformation
- Share device_id across cluster
- Use similar addresses/phones
- Apply template-like text patterns
- Coordinate timing (similar hours)

---

## Step 6: Create Attack Pools

Maintain reusable suspicious pools:

### Fraud Device Pool
```python
fraud_devices = [f"DEV-FRAUD-{i:04d}" for i in range(50)]
```
- 50 pre-generated suspicious device IDs
- Fraud rows sample from this pool with 70% probability

### Cluster Device Pool
```python
cluster_devices = {
    "CLUSTER-000": "DEV-CLUSTER-0000",
    "CLUSTER-001": "DEV-CLUSTER-0001",
    ...
}
```
- Coordinated attack rows share cluster-specific devices
- Each cluster has ~5 applications

### Implicit Pools
- Phone numbers and addresses are not explicitly pooled
- Reuse counts are generated directly (simulating the effect)

---

## Step 7: Derive Labels from Latent Signals

The `generated_signal_score` is computed from weighted fraud indicators:

| Signal | Weight |
|--------|--------|
| Device reuse ≥ 3 | +0.15 |
| Device reuse ≥ 1 | +0.08 |
| Phone reuse ≥ 2 | +0.10 |
| Address reuse ≥ 3 | +0.12 |
| Email reuse ≥ 2 | +0.08 |
| Free email domain | +0.05 |
| Low name/email match (< 0.3) | +0.10 |
| Low tenure at address (< 6 mo) | +0.08 |
| Low tenure at employer (< 3 mo) | +0.08 |
| Thin file flag | +0.10 |
| High ZIP/IP distance (> 0.7) | +0.12 |
| Unusual hour (0-5) | +0.05 |
| No document uploaded | +0.05 |

Final score is clamped to [0, 1] with small random noise.

---

## Step 8: Inject Realistic Noise

To avoid an overly easy dataset:

### Legitimate Noise
- Some legitimate rows use free email domains
- Some legitimate OCR has formatting imperfections
- Recent movers have IP/address mismatch
- Short tenure for recent graduates

### Fraud Noise
- Some fraud rows have cleaner signals (hard difficulty)
- True name fraud looks more legitimate
- Hard coordinated attacks have moderate reuse

### Difficulty-Based Signal Intensity

| Difficulty | Fraud Signal Intensity |
|------------|------------------------|
| Easy | High reuse, obvious patterns |
| Medium | Moderate signals |
| Hard | Low signals, looks cleaner |

---

## Step 9: Include Time Evolution

### Date Distribution
- Applications span 12 months (2024-01-01 to 2024-12-31)
- Dates are uniformly distributed across the range
- `application_month` derived from `application_date`

### Potential for Temporal Patterns
- Current implementation: uniform distribution
- Future enhancement: could add monthly fraud rate variation

---

## Step 10: Save Row-Level Data and Hidden Truth

### Main Dataset Output
- All 40+ columns including features and labels
- Saved as both Parquet and CSV

### Hidden Metadata Output
- `application_id`
- `fraud_type` (ground truth archetype)
- `difficulty_level` (easy/medium/hard)
- `generated_signal_score` (latent fraud score)

---

## Implementation Summary

The current implementation in `src/data_generation.py` follows this pipeline:

| Step | Implementation |
|------|----------------|
| 1. Profile generation | `generate_base_identity()` |
| 2. Behavioral fields | Per-archetype generators |
| 3. Text generation | `generate_verification_note()`, `generate_ocr_text()`, etc. |
| 4. OCR messiness | Template selection based on fraud type |
| 5. Fraud transformations | `generate_synthetic_identity()`, `generate_true_name_fraud()`, `generate_coordinated_attack()` |
| 6. Attack pools | `shared_pools` dict with `fraud_devices` and `cluster_devices` |
| 7. Signal score | `compute_generated_signal_score()` |
| 8. Noise injection | Probabilistic logic in each generator |
| 9. Time evolution | `generate_application_date()` |
| 10. Save outputs | `save_outputs()` |

---

## Key Configuration

```python
FRAUD_TYPE_DISTRIBUTION = {
    "legitimate_clean": 0.70,
    "legitimate_noisy": 0.18,
    "synthetic_identity": 0.05,
    "true_name_fraud": 0.04,
    "coordinated_attack": 0.03,
}

DIFFICULTY_DISTRIBUTION = {
    "legitimate_clean": {"easy": 0.70, "medium": 0.25, "hard": 0.05},
    "legitimate_noisy": {"easy": 0.10, "medium": 0.50, "hard": 0.40},
    "synthetic_identity": {"easy": 0.20, "medium": 0.50, "hard": 0.30},
    "true_name_fraud": {"easy": 0.10, "medium": 0.45, "hard": 0.45},
    "coordinated_attack": {"easy": 0.15, "medium": 0.55, "hard": 0.30},
}

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
RANDOM_SEED = 42
```

---

## Reproducibility

The generator uses:
- `random.seed(42)` for Python random
- `np.random.seed(42)` for NumPy random
- Consistent dictionary loading order

This ensures identical output for the same seed.
