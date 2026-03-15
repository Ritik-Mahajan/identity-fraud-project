# Fraud and Legitimate Archetypes

## Overview

This document defines the five application archetypes used in the synthetic dataset. Each archetype has distinct characteristics that the generator must implement.

---

## Legitimate Archetypes

### 1. legitimate_clean

**Description**: Standard legitimate applications with consistent, low-risk profiles.

**Characteristics**:

| Attribute | Pattern |
|-----------|---------|
| Identity fields | Consistent across all fields |
| Phone/email/device/address reuse | Low (0-1 prior apps) |
| Income and tenure | Realistic for age and employer |
| OCR text | Mostly matches application data |
| Verification notes | Benign, confirmatory |
| Name/email consistency | Moderate to high match score |
| Free email domain | Less common |
| Thin file flag | Usually 0 |

**Difficulty**: Mostly easy (clear non-fraud signal)

---

### 2. legitimate_noisy

**Description**: Legitimate applications with noise or ambiguity that could be mistaken for fraud signals.

**Characteristics**:

| Attribute | Pattern |
|-----------|---------|
| Name fields | May use nickname vs formal name |
| Address tenure | Short (recent move) |
| Employer tenure | Short (recent job change) |
| OCR text | Formatting imperfections, typos |
| Free email domain | May appear |
| Text fields | Some inconsistencies but explainable |
| Name/email match | Lower due to nickname usage |

**Purpose**: Creates realistic borderline non-fraud examples that challenge the model.

**Difficulty**: Medium to hard

**Example scenarios**:
- Recent college graduate with new job and new apartment
- Person who goes by "Mike" but documents say "Michael"
- Recent relocation for work with temporary address mismatch

---

## Fraud Archetypes

### 3. synthetic_identity

**Description**: Fabricated identities created by stitching together elements from multiple sources.

**Characteristics**:

| Attribute | Pattern |
|-----------|---------|
| Identity elements | Stitched/fabricated combinations |
| Phone/device/address reuse | Higher than legitimate |
| Tenure (address and employer) | Low |
| Thin file flag | Often 1 |
| Free email domain | More common |
| OCR text | Mismatched with application |
| Verification notes | Suspicious language |
| Text consistency | Moderate to high inconsistency |

**Fraud signals**:
- SSN may not match typical patterns for claimed age
- Address history appears shallow
- Employment difficult to verify
- Document text doesn't align with entered data

**Difficulty**: Mix of easy, medium, and hard

---

### 4. true_name_fraud

**Description**: Real identity used fraudulently (stolen or misused identity).

**Characteristics**:

| Attribute | Pattern |
|-----------|---------|
| Identity appearance | More realistic than synthetic |
| Primary fraud signals | Velocity, reuse, regional mismatch |
| Text inconsistency | Subtle |
| Overall appearance | Cleaner than synthetic fraud |

**Fraud signals**:
- Behavioral anomalies (unusual application patterns)
- Geographic inconsistencies (IP vs address)
- Velocity spikes
- Subtle text field inconsistencies

**Purpose**: Creates harder fraud cases that look more legitimate on surface.

**Difficulty**: Mostly medium and hard

---

### 5. coordinated_attack

**Description**: Cluster-based fraud where multiple applications share attack infrastructure.

**Characteristics**:

| Attribute | Pattern |
|-----------|---------|
| Device ID | Shared across cluster |
| Phone number | Shared or sequential |
| Address | Shared or nearby |
| Verification note templates | Reused with slight variation |
| Identity fields | Slight variations across cluster |

**Fraud signals**:
- Multiple applications from same device
- Repeated suspicious patterns
- Template-like text fields
- Coordinated timing

**Cluster behavior**:
- 3-10 applications per cluster
- Submitted within short time window
- Shared infrastructure elements
- Slight identity variations to avoid exact duplicates

**Difficulty**: Mix (some clusters obvious, some subtle)

---

## Archetype Distribution Summary

| Archetype | Category | Approx % of Total |
|-----------|----------|-------------------|
| legitimate_clean | Legitimate | ~60% |
| legitimate_noisy | Legitimate | ~28% |
| synthetic_identity | Fraud | ~5% |
| true_name_fraud | Fraud | ~4% |
| coordinated_attack | Fraud | ~3% |

*Note: Percentages are approximate targets. Exact distribution may vary slightly during generation.*

---

## Difficulty by Archetype

| Archetype | Easy | Medium | Hard |
|-----------|------|--------|------|
| legitimate_clean | 70% | 25% | 5% |
| legitimate_noisy | 10% | 50% | 40% |
| synthetic_identity | 20% | 50% | 30% |
| true_name_fraud | 10% | 45% | 45% |
| coordinated_attack | 15% | 55% | 30% |

---

## Key Distinctions for Modeling

### Structured Features Alone Should Catch:
- Most `synthetic_identity` (high reuse, thin file, low tenure)
- Most `coordinated_attack` (velocity, device reuse)
- Easy `true_name_fraud` (obvious behavioral anomalies)

### Text/Encoder Features Should Help With:
- Hard `true_name_fraud` (subtle text inconsistencies)
- Hard `synthetic_identity` (document mismatches)
- Distinguishing `legitimate_noisy` from fraud (explainable vs suspicious text)

### Borderline Cases (Stage 2 Targets):
- `legitimate_noisy` with low tenure and nickname mismatch
- `true_name_fraud` with clean structured features
- `synthetic_identity` with moderate (not extreme) reuse patterns
