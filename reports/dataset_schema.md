# Dataset Schema Specification

## Purpose

This dataset simulates application/identity fraud detection at application time. Each row represents one application.

**Target variable**: `fraud_label` (binary)
- `fraud_label = 1` → Fraudulent application
- `fraud_label = 0` → Legitimate application

## Dataset Requirements

The dataset must support:
- Stage 1 structured fraud modeling
- Stage 2 encoder/text consistency features
- Borderline-case analysis
- Out-of-time validation

## Target Dataset Sizes

| Version | Rows |
|---------|------|
| Prototype | 5,000 |
| First serious | 20,000 |
| Full local | 80,000 - 100,000 |

## Time Structure

- Span: 12 months
- Enables time-based train/validation/test splits
- Enables monthly trend analysis

## Fraud Rate

| Category | Percentage |
|----------|------------|
| Legitimate | ~88% |
| Fraud | ~12% |

---

## Column Schema

### Identifiers / Time

| Column | Type | Description |
|--------|------|-------------|
| `application_id` | string | Unique ID for each application |
| `application_date` | date | Date application was submitted |
| `application_month` | string | YYYY-MM derived from application_date |

### Claimed Identity Fields

| Column | Type | Description |
|--------|------|-------------|
| `claimed_first_name` | string | Applicant-provided first name |
| `claimed_last_name` | string | Applicant-provided last name |
| `date_of_birth` | date | Applicant date of birth |
| `age` | integer | Derived from DOB and application date |
| `ssn_last4` | string | Four-digit string (simulated only) |

### Address Fields

| Column | Type | Description |
|--------|------|-------------|
| `address_line` | string | Street address |
| `city` | string | City name |
| `state` | string | State abbreviation |
| `zip_code` | string | ZIP code |

### Contact Fields

| Column | Type | Description |
|--------|------|-------------|
| `phone_number` | string | Claimed phone number |
| `email` | string | Claimed email address |
| `email_domain` | string | Domain extracted from email |
| `is_free_email_domain` | integer | 1 if free provider (gmail, yahoo, etc.), else 0 |

### Employment / Financial Fields

| Column | Type | Description |
|--------|------|-------------|
| `employer_name` | string | Claimed employer |
| `employer_industry` | string | Industry of employer |
| `annual_income` | integer | Annual income in USD |
| `housing_status` | string | One of: rent, own, family, other |
| `months_at_address` | integer | Tenure at current address |
| `months_at_employer` | integer | Tenure at current employer |
| `thin_file_flag` | integer | 1 if low-history/thin-file profile |

### Digital / Behavioral Fields

| Column | Type | Description |
|--------|------|-------------|
| `device_id` | string | Simulated device fingerprint ID |
| `ip_region` | string | Simulated region from IP behavior |
| `application_hour` | integer | Hour of day (0-23) |
| `num_prev_apps_same_device_7d` | integer | Prior apps from same device in 7 days |
| `num_prev_apps_same_email_30d` | integer | Prior apps from same email in 30 days |
| `num_prev_apps_same_phone_30d` | integer | Prior apps from same phone in 30 days |
| `num_prev_apps_same_address_30d` | integer | Prior apps from same address in 30 days |
| `zip_ip_distance_proxy` | float | Proxy for address/IP mismatch severity |

### Precomputed / Engineered-Style Raw Signals

| Column | Type | Description |
|--------|------|-------------|
| `name_email_match_score` | float | Score 0-1: email handle matches claimed name |
| `document_uploaded` | integer | 1 if document available, else 0 |

### Text / Semi-Unstructured Fields

| Column | Type | Description |
|--------|------|-------------|
| `verification_note` | string | Internal verification note text |
| `ocr_document_text` | string | OCR-like extracted text from documents |
| `address_explanation_text` | string | Explanation for address inconsistency |
| `employment_explanation_text` | string | Explanation for employment inconsistency |

### Label / Meta Fields

| Column | Type | Description |
|--------|------|-------------|
| `fraud_label` | integer | 0 = legitimate, 1 = fraudulent |
| `fraud_type` | string | Subgroup category (see below) |
| `difficulty_level` | string | easy, medium, or hard |
| `generated_signal_score` | float | Hidden latent score used during generation |

---

## Fraud Type Values

| Value | Description |
|-------|-------------|
| `legitimate_clean` | Clean legitimate application |
| `legitimate_noisy` | Legitimate with noise/ambiguity |
| `synthetic_identity` | Stitched/fabricated identity fraud |
| `true_name_fraud` | Real identity used fraudulently |
| `coordinated_attack` | Cluster-based attack fraud |

---

## Difficulty Level Values

| Value | Description |
|-------|-------------|
| `easy` | Very clear pattern (obvious fraud or clean legitimate) |
| `medium` | Moderate ambiguity |
| `hard` | Borderline case |
