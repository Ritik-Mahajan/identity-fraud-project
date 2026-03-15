# Data Generation Design

## Overview

This document specifies how the synthetic dataset will be generated. The generator must produce realistic fraud and legitimate patterns that support both structured modeling and text-based consistency analysis.

---

## Population Distribution

### Overall Split

| Category | Percentage |
|----------|------------|
| Legitimate | ~88% |
| Fraud | ~12% |

### Legitimate Subgroups

| Subgroup | Description |
|----------|-------------|
| `legitimate_clean` | Consistent, low-risk profiles |
| `legitimate_noisy` | Legitimate but with ambiguity |

### Fraud Subgroups

| Subgroup | Description |
|----------|-------------|
| `synthetic_identity` | Fabricated/stitched identities |
| `true_name_fraud` | Real identity misused |
| `coordinated_attack` | Cluster-based attack patterns |

---

## Difficulty Distribution

### Fraud Rows

| Difficulty | Percentage |
|------------|------------|
| easy | 15% |
| medium | 50% |
| hard | 35% |

### Legitimate Rows

| Difficulty | Percentage |
|------------|------------|
| easy (clean) | 60% |
| medium (noisy) | 30% |
| hard (borderline) | 10% |

---

## Signal Design

### Strong Fraud Signals

These features should correlate strongly with fraud:

| Signal | Fraud Pattern |
|--------|---------------|
| `num_prev_apps_same_device_7d` | High device reuse |
| `num_prev_apps_same_phone_30d` | High phone reuse |
| `num_prev_apps_same_address_30d` | High address reuse |
| `num_prev_apps_same_email_30d` | High email reuse |
| `months_at_address` | Low tenure |
| `months_at_employer` | Low tenure |
| `thin_file_flag` | = 1 more often |
| `is_free_email_domain` | = 1 more often |
| `zip_ip_distance_proxy` | High mismatch |
| `name_email_match_score` | Low score |
| `verification_note` | Suspicious language |
| `ocr_document_text` | Inconsistent with application |

### Moderate Fraud Signals

| Signal | Fraud Pattern |
|--------|---------------|
| `application_hour` | Unusual hours (late night) |
| `annual_income` | Unrealistic for age/employer |
| `document_uploaded` | Missing or messy |
| `address_explanation_text` | Vague or suspicious |
| `employment_explanation_text` | Vague or suspicious |

### Legitimate Noisy Signals (Not Fraud)

These patterns appear in legitimate_noisy rows to create realistic borderline cases:

| Signal | Pattern |
|--------|---------|
| `name_email_match_score` | Low due to nickname |
| `months_at_address` | Low due to recent move |
| `months_at_employer` | Low due to recent job change |
| `ocr_document_text` | Typos or formatting issues |
| `is_free_email_domain` | = 1 (common for legitimate users too) |

---

## Text Field Generation Design

### verification_note

**Purpose**: Internal notes from verification process.

**Legitimate examples**:
- "Applicant confirmed recent move and provided updated utility bill."
- "Employer verified through callback; start date consistent."
- "Name variation due to nickname, documents otherwise consistent."

**Fraud-like examples**:
- "Unable to verify employer through listed contact."
- "Address appears associated with multiple prior applicants."
- "Document text inconsistent with entered application details."
- "High device reuse observed across unrelated identities."

### ocr_document_text

**Purpose**: Simulated OCR output from identity/address/employment documents.

**Format**: Semi-structured, slightly messy text.

**Legitimate examples**:
- "Name Michael Carter Address 411 West Pine St Springfield IL"
- "Utility statement confirms service address and applicant name"
- "Employer letter confirms salary and start date"

**Fraud-like examples**:
- "Name partially unreadable address mismatch"
- "Document image low quality text fields incomplete"
- "Address line inconsistent with entered ZIP"
- "Name on document differs from application spelling"

### address_explanation_text

**Purpose**: Explanation for address inconsistency or recent move.

**Legitimate example**:
- "Recently relocated for work, mailing and residential address differ temporarily."

**Fraud-like example**:
- "Applicant unable to clearly explain address discrepancy."

### employment_explanation_text

**Purpose**: Explanation for employment inconsistency or short tenure.

**Legitimate example**:
- "Recently joined employer after graduation."

**Fraud-like example**:
- "Employer name provided but no verifiable tenure details."

---

## Time Distribution

- Dataset spans 12 months
- Applications distributed across months to enable:
  - Time-based train/validation/test splits
  - Monthly fraud rate analysis
  - Trend detection

---

## Reuse and Velocity Logic

### Device Reuse

- Legitimate: Mostly unique device IDs
- Fraud: Higher probability of shared device IDs across applications

### Phone/Email/Address Reuse

- Legitimate: Low reuse counts (0-1)
- Fraud: Higher reuse counts, especially for coordinated attacks

### Coordinated Attack Clusters

- Share device_id, phone_number, or address across multiple applications
- Use similar verification_note templates
- Slight variations in identity fields

---

## Generated Signal Score

`generated_signal_score` is a hidden latent variable:
- Used during generation to control fraud likelihood
- NOT a real production feature
- Useful for debugging and analysis
- Should correlate with fraud_label but with noise
