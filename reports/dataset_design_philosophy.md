# Dataset Design Philosophy

## Overview

This document captures the design philosophy behind the synthetic dataset for the Identity Fraud Detection project. The goal is not perfect simulation, but realistic-enough structure for fraud modeling and borderline-case analysis.

---

## Core Principle: Believable Applications

Each row in the dataset should represent a **believable application**, not random values. This means:

1. **Identity fields** should form coherent profiles (name, DOB, address, employer)
2. **Behavioral fields** should reflect realistic patterns (tenure, reuse counts)
3. **Digital footprint fields** should capture application context (device, IP, hour)
4. **Text evidence** should read like real verification notes and OCR output
5. **Fraud labels** should arise from realistic fraud mechanisms, not random noise

---

## Four Pillars of Dataset Design

### 1. Realism

The data should feel like it came from a real application-risk workflow:

- Names, addresses, and employers should be plausible
- Income should align with industry and age
- Tenure at address/employer should follow realistic distributions
- Text fields should use natural language patterns

### 2. Pattern Diversity

The dataset should include multiple fraud archetypes:

| Archetype | Description |
|-----------|-------------|
| `legitimate_clean` | Standard clean applications |
| `legitimate_noisy` | Legitimate but with ambiguity |
| `synthetic_identity` | Fabricated/stitched identities |
| `true_name_fraud` | Real identity misused |
| `coordinated_attack` | Cluster-based attacks |

Each archetype has distinct signal patterns that the model should learn.

### 3. Ambiguity

The dataset must include hard/borderline cases:

- Some legitimate rows should look slightly suspicious
- Some fraud rows should look relatively clean
- Difficulty levels (easy/medium/hard) control signal intensity
- This ambiguity is essential for testing Stage 2 encoder features

### 4. Controlled Randomness

Randomness should be purposeful, not arbitrary:

- Random sampling from realistic dictionaries (names, cities, employers)
- Probabilistic signal generation based on fraud type
- Noise injection to avoid overly clean patterns
- Reproducibility via random seeds

---

## What the Dataset Must Support

### Stage 1: Structured Fraud Modeling

The structured features should enable a baseline fraud model:

- Device/phone/email/address reuse counts
- Tenure at address and employer
- Thin file flag
- ZIP/IP distance proxy
- Name/email match score
- Free email domain indicator

### Stage 2: Encoder/Text Consistency Features

The text fields should enable semantic analysis:

- `verification_note` - Internal review notes
- `ocr_document_text` - Simulated document OCR
- `address_explanation_text` - Address discrepancy explanations
- `employment_explanation_text` - Employment discrepancy explanations

Text should show consistency patterns:
- Legitimate: Text aligns with application data
- Fraud: Text contains mismatches, suspicious language, or inconsistencies

### Stage 3: Combined Model

The final model should combine:
- Stage 1 structured predictions
- Stage 2 encoder-derived features
- Focus on improving borderline case predictions

### Validation on Borderline Cases

The dataset should include enough borderline cases to:
- Test whether Stage 2 features add value
- Evaluate model performance on hard cases
- Support out-of-time validation via 12-month span

---

## Fraud Generation Philosophy

**Fraud should not be generated as random noise.**

Instead, fraud arises from realistic mechanisms:

### Synthetic Identity Fraud
- Stitched identity elements that don't fully cohere
- Higher reuse of shared fraud infrastructure
- Low tenure (new/fabricated history)
- Thin file behavior
- OCR mismatches with application data

### True Name Fraud
- Identity appears realistic (stolen from real person)
- Fraud signals come from behavior, not identity
- Geographic inconsistencies (IP vs address)
- Velocity anomalies
- Subtle text inconsistencies

### Coordinated Attacks
- Shared device/phone/address across cluster
- Template-like text patterns
- Coordinated timing
- Slight identity variations to avoid exact duplicates

---

## Ambiguity by Design

### Legitimate Noisy Cases

Some legitimate applications should trigger mild fraud signals:

- Recent college graduate: short tenure, new address
- Nickname usage: "Mike" vs "Michael" causing email mismatch
- Recent relocation: IP/address mismatch
- Free email usage: common among legitimate users too

These cases test whether the model avoids false positives.

### Hard Fraud Cases

Some fraud applications should look relatively clean:

- True name fraud with realistic tenure
- Synthetic identity with moderate (not extreme) reuse
- Coordinated attacks with subtle patterns

These cases test whether Stage 2 features improve detection.

---

## Time Structure

The dataset spans 12 months to support:

- Time-based train/validation/test splits
- Monthly fraud rate analysis
- Out-of-time validation
- Potential for temporal pattern detection

---

## Hidden Metadata

The dataset preserves hidden metadata for analysis:

| Field | Purpose |
|-------|---------|
| `fraud_type` | Ground truth archetype |
| `difficulty_level` | How obvious the case is |
| `generated_signal_score` | Latent fraud score for debugging |

This metadata is not used in modeling but supports:
- Error analysis by fraud type
- Performance evaluation by difficulty
- Debugging and dataset validation

---

## Summary

The dataset design philosophy prioritizes:

1. **Believability** over perfect simulation
2. **Pattern diversity** across fraud archetypes
3. **Ambiguity** to test model boundaries
4. **Controlled randomness** for reproducibility
5. **Support for multi-stage modeling** (structured → encoder → combined)

This philosophy guides all data generation decisions.
