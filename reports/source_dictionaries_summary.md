# Source Dictionaries Summary

## Overview

This document summarizes the lookup files created in Phase 3 for synthetic data generation. These files are stored in `data/external/` and provide realistic values for generating application fraud detection data.

## Files Created

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `first_names.csv` | 94 | first_name | Pool of realistic first names for applicant identity |
| `last_names.csv` | 100 | last_name | Pool of realistic last names for applicant identity |
| `cities_states_zips.csv` | 44 | city, state, zip_code | US city/state/ZIP combinations for address generation |
| `street_names.csv` | 80 | street_name | Street name pool for address_line generation |
| `employers.csv` | 61 | employer_name, employer_industry | Employer names with industry classification |
| `email_domains.csv` | 25 | domain, is_free | Email domains with free/paid classification |
| `legit_verification_note_templates.csv` | 25 | template | Templates for legitimate verification notes |
| `fraud_verification_note_templates.csv` | 25 | template | Templates for fraud-like verification notes |
| `legit_ocr_templates.csv` | 20 | template | Templates for legitimate OCR document text |
| `fraud_ocr_templates.csv` | 20 | template | Templates for fraud-like OCR document text |
| `address_explanation_templates.csv` | 13 | label, template | Address explanation templates (legit + fraud_like) |
| `employment_explanation_templates.csv` | 13 | label, template | Employment explanation templates (legit + fraud_like) |

## File Details

### Identity Files

**first_names.csv** (94 rows)
- Common US first names
- Mix of male and female names
- Used for `claimed_first_name` field

**last_names.csv** (100 rows)
- Common US last names
- Diverse ethnic representation
- Used for `claimed_last_name` field

### Location Files

**cities_states_zips.csv** (44 rows)
- Major US cities with valid state and ZIP combinations
- Covers multiple regions (East, West, South, Midwest)
- Used for `city`, `state`, `zip_code` fields

**street_names.csv** (80 rows)
- Common US street name patterns
- Mix of numbered and named streets
- Used to construct `address_line` field

### Employment Files

**employers.csv** (61 rows)
- Mix of well-known companies and generic employers
- Industries covered:
  - retail (7)
  - technology (12)
  - healthcare (8)
  - finance (8)
  - insurance (5)
  - logistics (5)
  - education (5)
  - hospitality (6)
  - manufacturing (6)
- Used for `employer_name` and `employer_industry` fields

### Email Files

**email_domains.csv** (25 rows)
- 11 free email domains (gmail, yahoo, outlook, etc.)
- 14 non-free domains (ISP and corporate-style)
- Used for `email` and `is_free_email_domain` fields

### Text Template Files

**legit_verification_note_templates.csv** (25 rows)
- Benign, confirmatory verification notes
- Examples: "All documents match application details"
- Used for legitimate and legitimate_noisy archetypes

**fraud_verification_note_templates.csv** (25 rows)
- Suspicious verification notes
- Examples: "Unable to verify employer through listed contact"
- Used for fraud archetypes

**legit_ocr_templates.csv** (20 rows)
- Clean OCR-style document text
- Includes placeholders for name, address, employer
- Used for legitimate archetypes

**fraud_ocr_templates.csv** (20 rows)
- Inconsistent or problematic OCR text
- Shows mismatches and quality issues
- Used for fraud archetypes

**address_explanation_templates.csv** (13 rows)
- 7 legitimate explanations (recent move, relocation, etc.)
- 6 fraud-like explanations (unable to explain, inconsistent, etc.)
- Used for `address_explanation_text` field

**employment_explanation_templates.csv** (13 rows)
- 7 legitimate explanations (new job, graduation, career change)
- 6 fraud-like explanations (unverifiable, inconsistent)
- Used for `employment_explanation_text` field

## Template Placeholders

Templates may include these placeholders for dynamic substitution:

| Placeholder | Source Field |
|-------------|--------------|
| `{first_name}` | claimed_first_name |
| `{last_name}` | claimed_last_name |
| `{city}` | city |
| `{state}` | state |
| `{zip_code}` | zip_code |
| `{address_line}` | address_line |
| `{employer_name}` | employer_name |
| `{employer_industry}` | employer_industry |
| `{annual_income}` | annual_income |
| `{months_at_employer}` | months_at_employer |

## Usage in Data Generator

The data generator (Phase 4) will:

1. Load all CSV files at startup
2. Sample from identity/location/employer files to create base application data
3. Select appropriate text templates based on fraud_type and difficulty_level
4. Substitute placeholders with actual generated values
5. Apply noise and variation based on archetype rules

## Validation

Run `src/inspect_dictionaries.py` to verify all files are present and correctly formatted.
