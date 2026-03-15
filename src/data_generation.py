"""
data_generation.py

Synthetic data generator for the Identity Fraud Detection project.
Creates realistic application data with fraud and legitimate patterns.

This module generates:
- legitimate_clean: Standard clean applications
- legitimate_noisy: Legitimate but with ambiguity
- synthetic_identity: Fabricated identity fraud
- true_name_fraud: Stolen/misused identity fraud
- coordinated_attack: Cluster-based fraud attacks

Usage:
    python src/data_generation.py
    
Or import and use:
    from src.data_generation import create_dataset
    df = create_dataset(n_rows=5000)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import random
import string


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

# Fraud type distribution (must sum to 1.0)
FRAUD_TYPE_DISTRIBUTION = {
    "legitimate_clean": 0.70,
    "legitimate_noisy": 0.18,
    "synthetic_identity": 0.05,
    "true_name_fraud": 0.04,
    "coordinated_attack": 0.03,
}

# Difficulty distribution by fraud type
DIFFICULTY_DISTRIBUTION = {
    "legitimate_clean": {"easy": 0.70, "medium": 0.25, "hard": 0.05},
    "legitimate_noisy": {"easy": 0.10, "medium": 0.50, "hard": 0.40},
    "synthetic_identity": {"easy": 0.20, "medium": 0.50, "hard": 0.30},
    "true_name_fraud": {"easy": 0.10, "medium": 0.45, "hard": 0.45},
    "coordinated_attack": {"easy": 0.15, "medium": 0.55, "hard": 0.30},
}

# Valid values for validation
VALID_FRAUD_TYPES = list(FRAUD_TYPE_DISTRIBUTION.keys())
VALID_DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
VALID_HOUSING_STATUS = ["rent", "own", "family", "other"]

# Date range for applications (12 months)
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# DICTIONARY LOADING
# =============================================================================

def load_dictionaries() -> dict:
    """
    Load all source dictionary CSV files from data/external.
    
    Returns:
        dict: Dictionary containing all loaded DataFrames
    """
    dictionaries = {}
    
    # List of required files
    files_to_load = [
        ("first_names", "first_names.csv"),
        ("last_names", "last_names.csv"),
        ("locations", "cities_states_zips.csv"),
        ("street_names", "street_names.csv"),
        ("employers", "employers.csv"),
        ("email_domains", "email_domains.csv"),
        ("legit_verification_notes", "legit_verification_note_templates.csv"),
        ("fraud_verification_notes", "fraud_verification_note_templates.csv"),
        ("legit_ocr", "legit_ocr_templates.csv"),
        ("fraud_ocr", "fraud_ocr_templates.csv"),
        ("address_explanations", "address_explanation_templates.csv"),
        ("employment_explanations", "employment_explanation_templates.csv"),
    ]
    
    for key, filename in files_to_load:
        filepath = DATA_EXTERNAL / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required dictionary file not found: {filepath}")
        dictionaries[key] = pd.read_csv(filepath)
    
    return dictionaries


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_application_id() -> str:
    """Generate a unique application ID."""
    return f"APP-{uuid.uuid4().hex[:12].upper()}"


def generate_application_date(start: datetime, end: datetime) -> datetime:
    """Generate a random date between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def generate_ssn_last4() -> str:
    """Generate a random 4-digit SSN suffix."""
    return f"{random.randint(0, 9999):04d}"


def generate_phone_number() -> str:
    """Generate a realistic US phone number."""
    area_code = random.randint(200, 999)
    exchange = random.randint(200, 999)
    subscriber = random.randint(1000, 9999)
    return f"{area_code}-{exchange}-{subscriber}"


def generate_device_id(pool: list = None) -> str:
    """
    Generate a device ID.
    
    Args:
        pool: Optional list of device IDs to sample from (for reuse scenarios)
    
    Returns:
        str: Device ID
    """
    if pool and random.random() < 0.7:  # 70% chance to reuse from pool
        return random.choice(pool)
    return f"DEV-{uuid.uuid4().hex[:16].upper()}"


def generate_email(first_name: str, last_name: str, domain: str, 
                   match_name: bool = True) -> str:
    """
    Generate an email address.
    
    Args:
        first_name: First name
        last_name: Last name
        domain: Email domain
        match_name: If True, email will contain name elements
    
    Returns:
        str: Email address
    """
    first_lower = first_name.lower()
    last_lower = last_name.lower()
    
    if match_name:
        # Name-based email patterns
        patterns = [
            f"{first_lower}.{last_lower}",
            f"{first_lower}_{last_lower}",
            f"{first_lower}{last_lower}",
            f"{first_lower[0]}{last_lower}",
            f"{first_lower}.{last_lower[0]}",
            f"{first_lower}{random.randint(1, 99)}",
            f"{first_lower}.{last_lower}{random.randint(1, 99)}",
        ]
        handle = random.choice(patterns)
    else:
        # Generic/random email patterns (less name alignment)
        patterns = [
            f"user{random.randint(1000, 9999)}",
            f"account{random.randint(100, 999)}",
            f"{first_lower[0]}{random.randint(1000, 9999)}",
            f"{''.join(random.choices(string.ascii_lowercase, k=6))}{random.randint(10, 99)}",
        ]
        handle = random.choice(patterns)
    
    return f"{handle}@{domain}"


def compute_name_email_match_score(first_name: str, last_name: str, email: str) -> float:
    """
    Compute how well the email matches the claimed name.
    
    Args:
        first_name: First name
        last_name: Last name
        email: Email address
    
    Returns:
        float: Score from 0 to 1
    """
    handle = email.split("@")[0].lower()
    first_lower = first_name.lower()
    last_lower = last_name.lower()
    
    score = 0.0
    
    # Check for first name presence
    if first_lower in handle:
        score += 0.4
    elif first_lower[:3] in handle:  # First 3 chars
        score += 0.2
    
    # Check for last name presence
    if last_lower in handle:
        score += 0.4
    elif last_lower[:3] in handle:  # First 3 chars
        score += 0.2
    
    # Check for initials
    if first_lower[0] in handle and last_lower[0] in handle:
        score += 0.1
    
    # Add small random noise
    score += random.uniform(-0.05, 0.05)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def generate_address_line(street_names: list) -> str:
    """Generate a street address."""
    number = random.randint(100, 9999)
    street = random.choice(street_names)
    return f"{number} {street}"


def generate_age() -> int:
    """Generate a realistic adult age (18-75)."""
    # Use a distribution that favors working-age adults
    if random.random() < 0.7:
        return random.randint(22, 55)  # Most common
    elif random.random() < 0.5:
        return random.randint(18, 21)  # Young adults
    else:
        return random.randint(56, 75)  # Older adults


def generate_dob(age: int, application_date: datetime) -> datetime:
    """Generate a date of birth consistent with age."""
    birth_year = application_date.year - age
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)  # Safe for all months
    return datetime(birth_year, birth_month, birth_day)


def generate_income(employer_industry: str, age: int) -> int:
    """
    Generate a plausible annual income based on industry and age.
    
    Args:
        employer_industry: Industry of employer
        age: Applicant age
    
    Returns:
        int: Annual income in USD
    """
    # Base income ranges by industry
    industry_ranges = {
        "technology": (60000, 180000),
        "finance": (55000, 200000),
        "healthcare": (45000, 150000),
        "education": (35000, 90000),
        "retail": (25000, 55000),
        "hospitality": (22000, 50000),
        "manufacturing": (35000, 85000),
        "logistics": (30000, 70000),
        "insurance": (45000, 120000),
    }
    
    base_min, base_max = industry_ranges.get(employer_industry, (30000, 80000))
    
    # Adjust for age (experience)
    if age < 25:
        multiplier = 0.7
    elif age < 35:
        multiplier = 0.9
    elif age < 50:
        multiplier = 1.1
    else:
        multiplier = 1.0
    
    income = random.randint(int(base_min * multiplier), int(base_max * multiplier))
    
    # Round to nearest 1000
    return round(income / 1000) * 1000


def fill_template(template: str, data: dict) -> str:
    """
    Fill a template string with data values.
    
    Args:
        template: Template string with {placeholders}
        data: Dictionary of values to substitute
    
    Returns:
        str: Filled template
    """
    try:
        return template.format(**data)
    except KeyError:
        # If some placeholders are missing, return template as-is
        return template


# =============================================================================
# TEXT GENERATION FUNCTIONS
# =============================================================================

def generate_verification_note(fraud_type: str, difficulty: str, 
                                dictionaries: dict, data: dict) -> str:
    """
    Generate a verification note based on fraud type and difficulty.
    
    Args:
        fraud_type: Type of fraud/legitimate
        difficulty: Difficulty level
        dictionaries: Loaded dictionaries
        data: Row data for template filling
    
    Returns:
        str: Verification note text
    """
    is_fraud = fraud_type in ["synthetic_identity", "true_name_fraud", "coordinated_attack"]
    
    if is_fraud:
        # Fraud rows use fraud templates more often
        if difficulty == "hard":
            # Hard fraud might have mixed signals
            use_fraud = random.random() < 0.6
        else:
            use_fraud = random.random() < 0.85
        
        if use_fraud:
            templates = dictionaries["fraud_verification_notes"]["template"].tolist()
        else:
            templates = dictionaries["legit_verification_notes"]["template"].tolist()
    else:
        # Legitimate rows
        if fraud_type == "legitimate_noisy" and difficulty in ["medium", "hard"]:
            # Noisy legitimate might have some concerning notes
            use_fraud = random.random() < 0.2
        else:
            use_fraud = False
        
        if use_fraud:
            templates = dictionaries["fraud_verification_notes"]["template"].tolist()
        else:
            templates = dictionaries["legit_verification_notes"]["template"].tolist()
    
    template = random.choice(templates)
    return fill_template(template, data)


def generate_ocr_text(fraud_type: str, difficulty: str,
                      dictionaries: dict, data: dict) -> str:
    """
    Generate OCR document text based on fraud type and difficulty.
    
    Args:
        fraud_type: Type of fraud/legitimate
        difficulty: Difficulty level
        dictionaries: Loaded dictionaries
        data: Row data for template filling
    
    Returns:
        str: OCR text
    """
    is_fraud = fraud_type in ["synthetic_identity", "true_name_fraud", "coordinated_attack"]
    
    if is_fraud:
        if difficulty == "hard":
            use_fraud = random.random() < 0.5
        else:
            use_fraud = random.random() < 0.8
        
        if use_fraud:
            templates = dictionaries["fraud_ocr"]["template"].tolist()
        else:
            templates = dictionaries["legit_ocr"]["template"].tolist()
    else:
        if fraud_type == "legitimate_noisy":
            use_fraud = random.random() < 0.15
        else:
            use_fraud = False
        
        if use_fraud:
            templates = dictionaries["fraud_ocr"]["template"].tolist()
        else:
            templates = dictionaries["legit_ocr"]["template"].tolist()
    
    template = random.choice(templates)
    return fill_template(template, data)


def generate_address_explanation(fraud_type: str, difficulty: str,
                                  dictionaries: dict, data: dict) -> str:
    """
    Generate address explanation text.
    
    Args:
        fraud_type: Type of fraud/legitimate
        difficulty: Difficulty level
        dictionaries: Loaded dictionaries
        data: Row data for template filling
    
    Returns:
        str: Address explanation text
    """
    df = dictionaries["address_explanations"]
    is_fraud = fraud_type in ["synthetic_identity", "true_name_fraud", "coordinated_attack"]
    
    if is_fraud:
        if difficulty == "hard":
            use_fraud = random.random() < 0.5
        else:
            use_fraud = random.random() < 0.75
    else:
        use_fraud = False
    
    if use_fraud:
        templates = df[df["label"] == "fraud_like"]["template"].tolist()
    else:
        templates = df[df["label"] == "legit"]["template"].tolist()
    
    template = random.choice(templates)
    return fill_template(template, data)


def generate_employment_explanation(fraud_type: str, difficulty: str,
                                     dictionaries: dict, data: dict) -> str:
    """
    Generate employment explanation text.
    
    Args:
        fraud_type: Type of fraud/legitimate
        difficulty: Difficulty level
        dictionaries: Loaded dictionaries
        data: Row data for template filling
    
    Returns:
        str: Employment explanation text
    """
    df = dictionaries["employment_explanations"]
    is_fraud = fraud_type in ["synthetic_identity", "true_name_fraud", "coordinated_attack"]
    
    if is_fraud:
        if difficulty == "hard":
            use_fraud = random.random() < 0.5
        else:
            use_fraud = random.random() < 0.75
    else:
        use_fraud = False
    
    if use_fraud:
        templates = df[df["label"] == "fraud_like"]["template"].tolist()
    else:
        templates = df[df["label"] == "legit"]["template"].tolist()
    
    template = random.choice(templates)
    return fill_template(template, data)


# =============================================================================
# BASE IDENTITY GENERATION
# =============================================================================

def generate_base_identity(dictionaries: dict, application_date: datetime) -> dict:
    """
    Generate a base identity profile with all core fields.
    
    Args:
        dictionaries: Loaded dictionaries
        application_date: Date of application
    
    Returns:
        dict: Base identity data
    """
    # Sample from dictionaries
    first_name = random.choice(dictionaries["first_names"]["first_name"].tolist())
    last_name = random.choice(dictionaries["last_names"]["last_name"].tolist())
    
    location = dictionaries["locations"].sample(1).iloc[0]
    city = location["city"]
    state = location["state"]
    zip_code = str(location["zip_code"])
    
    street_names = dictionaries["street_names"]["street_name"].tolist()
    address_line = generate_address_line(street_names)
    
    employer_row = dictionaries["employers"].sample(1).iloc[0]
    employer_name = employer_row["employer_name"]
    employer_industry = employer_row["employer_industry"]
    
    age = generate_age()
    dob = generate_dob(age, application_date)
    
    return {
        "claimed_first_name": first_name,
        "claimed_last_name": last_name,
        "date_of_birth": dob.strftime("%Y-%m-%d"),
        "age": age,
        "ssn_last4": generate_ssn_last4(),
        "address_line": address_line,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "employer_name": employer_name,
        "employer_industry": employer_industry,
        "phone_number": generate_phone_number(),
    }


# =============================================================================
# ARCHETYPE GENERATORS
# =============================================================================

def generate_legitimate_clean(dictionaries: dict, application_date: datetime,
                               difficulty: str) -> dict:
    """
    Generate a legitimate_clean application.
    
    Characteristics:
    - Consistent identity fields
    - Low reuse counts
    - Realistic income and tenure
    - Matching OCR text
    - Benign verification notes
    """
    base = generate_base_identity(dictionaries, application_date)
    
    # Email - usually matches name, less often free domain
    email_domains = dictionaries["email_domains"]
    if random.random() < 0.6:  # 60% non-free
        domain_row = email_domains[email_domains["is_free"] == 0].sample(1).iloc[0]
    else:
        domain_row = email_domains[email_domains["is_free"] == 1].sample(1).iloc[0]
    
    domain = domain_row["domain"]
    is_free = int(domain_row["is_free"])
    email = generate_email(base["claimed_first_name"], base["claimed_last_name"], 
                          domain, match_name=True)
    
    # Behavioral fields - low risk
    months_at_address = random.randint(12, 120)
    months_at_employer = random.randint(6, 96)
    thin_file_flag = 0 if random.random() < 0.95 else 1
    
    # Reuse counts - very low
    num_prev_apps_same_device_7d = 0 if random.random() < 0.95 else 1
    num_prev_apps_same_email_30d = 0 if random.random() < 0.98 else 1
    num_prev_apps_same_phone_30d = 0 if random.random() < 0.98 else 1
    num_prev_apps_same_address_30d = 0 if random.random() < 0.95 else 1
    
    # IP/ZIP mismatch - low
    zip_ip_distance_proxy = random.uniform(0, 0.3)
    
    # Application hour - normal business hours more likely
    if random.random() < 0.8:
        application_hour = random.randint(8, 20)
    else:
        application_hour = random.randint(0, 23)
    
    # Document uploaded - usually yes
    document_uploaded = 1 if random.random() < 0.92 else 0
    
    # Housing status
    housing_status = random.choices(
        ["own", "rent", "family", "other"],
        weights=[0.35, 0.45, 0.15, 0.05]
    )[0]
    
    # Income
    annual_income = generate_income(base["employer_industry"], base["age"])
    
    # Name/email match score
    name_email_match_score = compute_name_email_match_score(
        base["claimed_first_name"], base["claimed_last_name"], email
    )
    
    # IP region - usually matches state
    ip_region = base["state"] if random.random() < 0.9 else random.choice(
        dictionaries["locations"]["state"].tolist()
    )
    
    row = {
        **base,
        "email": email,
        "email_domain": domain,
        "is_free_email_domain": is_free,
        "annual_income": annual_income,
        "housing_status": housing_status,
        "months_at_address": months_at_address,
        "months_at_employer": months_at_employer,
        "thin_file_flag": thin_file_flag,
        "device_id": generate_device_id(),
        "ip_region": ip_region,
        "application_hour": application_hour,
        "num_prev_apps_same_device_7d": num_prev_apps_same_device_7d,
        "num_prev_apps_same_email_30d": num_prev_apps_same_email_30d,
        "num_prev_apps_same_phone_30d": num_prev_apps_same_phone_30d,
        "num_prev_apps_same_address_30d": num_prev_apps_same_address_30d,
        "zip_ip_distance_proxy": round(zip_ip_distance_proxy, 3),
        "name_email_match_score": round(name_email_match_score, 3),
        "document_uploaded": document_uploaded,
        "fraud_label": 0,
        "fraud_type": "legitimate_clean",
        "difficulty_level": difficulty,
    }
    
    return row


def generate_legitimate_noisy(dictionaries: dict, application_date: datetime,
                               difficulty: str) -> dict:
    """
    Generate a legitimate_noisy application.
    
    Characteristics:
    - May have nickname vs formal name issues
    - Short tenure (recent move/job change)
    - OCR formatting imperfections
    - Some text inconsistencies but explainable
    """
    base = generate_base_identity(dictionaries, application_date)
    
    # Email - more often free domain, sometimes lower name match
    email_domains = dictionaries["email_domains"]
    if random.random() < 0.55:  # 55% free
        domain_row = email_domains[email_domains["is_free"] == 1].sample(1).iloc[0]
    else:
        domain_row = email_domains[email_domains["is_free"] == 0].sample(1).iloc[0]
    
    domain = domain_row["domain"]
    is_free = int(domain_row["is_free"])
    
    # Sometimes use non-matching email (nickname scenario)
    match_name = random.random() < 0.7
    email = generate_email(base["claimed_first_name"], base["claimed_last_name"],
                          domain, match_name=match_name)
    
    # Behavioral fields - shorter tenure (recent changes)
    months_at_address = random.randint(1, 24)
    months_at_employer = random.randint(1, 18)
    thin_file_flag = 1 if random.random() < 0.25 else 0
    
    # Reuse counts - still low but slightly higher
    num_prev_apps_same_device_7d = 0 if random.random() < 0.90 else random.randint(1, 2)
    num_prev_apps_same_email_30d = 0 if random.random() < 0.95 else 1
    num_prev_apps_same_phone_30d = 0 if random.random() < 0.95 else 1
    num_prev_apps_same_address_30d = 0 if random.random() < 0.85 else random.randint(1, 2)
    
    # IP/ZIP mismatch - slightly higher (recent move)
    zip_ip_distance_proxy = random.uniform(0.1, 0.5)
    
    # Application hour - slightly more varied
    if random.random() < 0.7:
        application_hour = random.randint(8, 21)
    else:
        application_hour = random.randint(0, 23)
    
    document_uploaded = 1 if random.random() < 0.85 else 0
    
    housing_status = random.choices(
        ["rent", "family", "own", "other"],
        weights=[0.50, 0.25, 0.20, 0.05]
    )[0]
    
    annual_income = generate_income(base["employer_industry"], base["age"])
    
    name_email_match_score = compute_name_email_match_score(
        base["claimed_first_name"], base["claimed_last_name"], email
    )
    
    # IP region - sometimes different (recent move)
    ip_region = base["state"] if random.random() < 0.7 else random.choice(
        dictionaries["locations"]["state"].tolist()
    )
    
    row = {
        **base,
        "email": email,
        "email_domain": domain,
        "is_free_email_domain": is_free,
        "annual_income": annual_income,
        "housing_status": housing_status,
        "months_at_address": months_at_address,
        "months_at_employer": months_at_employer,
        "thin_file_flag": thin_file_flag,
        "device_id": generate_device_id(),
        "ip_region": ip_region,
        "application_hour": application_hour,
        "num_prev_apps_same_device_7d": num_prev_apps_same_device_7d,
        "num_prev_apps_same_email_30d": num_prev_apps_same_email_30d,
        "num_prev_apps_same_phone_30d": num_prev_apps_same_phone_30d,
        "num_prev_apps_same_address_30d": num_prev_apps_same_address_30d,
        "zip_ip_distance_proxy": round(zip_ip_distance_proxy, 3),
        "name_email_match_score": round(name_email_match_score, 3),
        "document_uploaded": document_uploaded,
        "fraud_label": 0,
        "fraud_type": "legitimate_noisy",
        "difficulty_level": difficulty,
    }
    
    return row


def generate_synthetic_identity(dictionaries: dict, application_date: datetime,
                                 difficulty: str, shared_pools: dict) -> dict:
    """
    Generate a synthetic_identity fraud application.
    
    Characteristics:
    - Stitched/fabricated identity elements
    - Higher reuse across phone/device/address
    - Low tenure
    - Thin file behavior
    - Mismatched OCR text
    """
    base = generate_base_identity(dictionaries, application_date)
    
    # Email - more free domains, often lower name match
    email_domains = dictionaries["email_domains"]
    if random.random() < 0.75:  # 75% free
        domain_row = email_domains[email_domains["is_free"] == 1].sample(1).iloc[0]
    else:
        domain_row = email_domains[email_domains["is_free"] == 0].sample(1).iloc[0]
    
    domain = domain_row["domain"]
    is_free = int(domain_row["is_free"])
    
    # Lower name match for synthetic identities
    match_name = random.random() < 0.4
    email = generate_email(base["claimed_first_name"], base["claimed_last_name"],
                          domain, match_name=match_name)
    
    # Behavioral fields - low tenure, thin file
    months_at_address = random.randint(0, 12)
    months_at_employer = random.randint(0, 8)
    thin_file_flag = 1 if random.random() < 0.65 else 0
    
    # Reuse counts - higher (shared fraud infrastructure)
    # Adjust based on difficulty
    if difficulty == "easy":
        num_prev_apps_same_device_7d = random.randint(2, 5)
        num_prev_apps_same_email_30d = random.randint(1, 3)
        num_prev_apps_same_phone_30d = random.randint(1, 4)
        num_prev_apps_same_address_30d = random.randint(2, 5)
    elif difficulty == "medium":
        num_prev_apps_same_device_7d = random.randint(1, 3)
        num_prev_apps_same_email_30d = random.randint(0, 2)
        num_prev_apps_same_phone_30d = random.randint(0, 2)
        num_prev_apps_same_address_30d = random.randint(1, 3)
    else:  # hard
        num_prev_apps_same_device_7d = random.randint(0, 2)
        num_prev_apps_same_email_30d = random.randint(0, 1)
        num_prev_apps_same_phone_30d = random.randint(0, 1)
        num_prev_apps_same_address_30d = random.randint(0, 2)
    
    # IP/ZIP mismatch - higher
    if difficulty == "easy":
        zip_ip_distance_proxy = random.uniform(0.6, 1.0)
    elif difficulty == "medium":
        zip_ip_distance_proxy = random.uniform(0.3, 0.7)
    else:
        zip_ip_distance_proxy = random.uniform(0.1, 0.5)
    
    # Application hour - more unusual hours
    if random.random() < 0.4:
        application_hour = random.choice([0, 1, 2, 3, 4, 5, 22, 23])
    else:
        application_hour = random.randint(0, 23)
    
    document_uploaded = 1 if random.random() < 0.7 else 0
    
    housing_status = random.choices(
        ["rent", "family", "other", "own"],
        weights=[0.50, 0.30, 0.15, 0.05]
    )[0]
    
    annual_income = generate_income(base["employer_industry"], base["age"])
    
    name_email_match_score = compute_name_email_match_score(
        base["claimed_first_name"], base["claimed_last_name"], email
    )
    
    # IP region - often different from address
    ip_region = base["state"] if random.random() < 0.4 else random.choice(
        dictionaries["locations"]["state"].tolist()
    )
    
    # Use shared device pool for reuse
    device_id = generate_device_id(shared_pools.get("fraud_devices", []))
    
    row = {
        **base,
        "email": email,
        "email_domain": domain,
        "is_free_email_domain": is_free,
        "annual_income": annual_income,
        "housing_status": housing_status,
        "months_at_address": months_at_address,
        "months_at_employer": months_at_employer,
        "thin_file_flag": thin_file_flag,
        "device_id": device_id,
        "ip_region": ip_region,
        "application_hour": application_hour,
        "num_prev_apps_same_device_7d": num_prev_apps_same_device_7d,
        "num_prev_apps_same_email_30d": num_prev_apps_same_email_30d,
        "num_prev_apps_same_phone_30d": num_prev_apps_same_phone_30d,
        "num_prev_apps_same_address_30d": num_prev_apps_same_address_30d,
        "zip_ip_distance_proxy": round(zip_ip_distance_proxy, 3),
        "name_email_match_score": round(name_email_match_score, 3),
        "document_uploaded": document_uploaded,
        "fraud_label": 1,
        "fraud_type": "synthetic_identity",
        "difficulty_level": difficulty,
    }
    
    return row


def generate_true_name_fraud(dictionaries: dict, application_date: datetime,
                              difficulty: str, shared_pools: dict) -> dict:
    """
    Generate a true_name_fraud application.
    
    Characteristics:
    - More realistic identity than synthetic
    - Fraud signals come from velocity, reuse, regional mismatch
    - Subtle text inconsistencies
    - Cleaner overall appearance
    """
    base = generate_base_identity(dictionaries, application_date)
    
    # Email - more balanced, often matches name (stolen identity)
    email_domains = dictionaries["email_domains"]
    if random.random() < 0.55:
        domain_row = email_domains[email_domains["is_free"] == 1].sample(1).iloc[0]
    else:
        domain_row = email_domains[email_domains["is_free"] == 0].sample(1).iloc[0]
    
    domain = domain_row["domain"]
    is_free = int(domain_row["is_free"])
    
    # Higher name match (using real person's identity)
    match_name = random.random() < 0.7
    email = generate_email(base["claimed_first_name"], base["claimed_last_name"],
                          domain, match_name=match_name)
    
    # Behavioral fields - more realistic tenure
    months_at_address = random.randint(3, 48)
    months_at_employer = random.randint(2, 36)
    thin_file_flag = 1 if random.random() < 0.3 else 0
    
    # Reuse counts - moderate (behavioral anomalies)
    if difficulty == "easy":
        num_prev_apps_same_device_7d = random.randint(2, 4)
        num_prev_apps_same_email_30d = random.randint(1, 3)
        num_prev_apps_same_phone_30d = random.randint(1, 3)
        num_prev_apps_same_address_30d = random.randint(1, 3)
    elif difficulty == "medium":
        num_prev_apps_same_device_7d = random.randint(1, 2)
        num_prev_apps_same_email_30d = random.randint(0, 2)
        num_prev_apps_same_phone_30d = random.randint(0, 2)
        num_prev_apps_same_address_30d = random.randint(0, 2)
    else:  # hard - looks very clean
        num_prev_apps_same_device_7d = random.randint(0, 1)
        num_prev_apps_same_email_30d = random.randint(0, 1)
        num_prev_apps_same_phone_30d = random.randint(0, 1)
        num_prev_apps_same_address_30d = random.randint(0, 1)
    
    # IP/ZIP mismatch - key signal for true name fraud
    if difficulty == "easy":
        zip_ip_distance_proxy = random.uniform(0.5, 0.9)
    elif difficulty == "medium":
        zip_ip_distance_proxy = random.uniform(0.3, 0.6)
    else:
        zip_ip_distance_proxy = random.uniform(0.1, 0.4)
    
    # Application hour - slightly unusual
    if random.random() < 0.3:
        application_hour = random.choice([0, 1, 2, 3, 4, 5, 22, 23])
    else:
        application_hour = random.randint(6, 22)
    
    document_uploaded = 1 if random.random() < 0.8 else 0
    
    housing_status = random.choices(
        ["rent", "own", "family", "other"],
        weights=[0.40, 0.35, 0.20, 0.05]
    )[0]
    
    annual_income = generate_income(base["employer_industry"], base["age"])
    
    name_email_match_score = compute_name_email_match_score(
        base["claimed_first_name"], base["claimed_last_name"], email
    )
    
    # IP region - often different (key fraud signal)
    ip_region = base["state"] if random.random() < 0.5 else random.choice(
        dictionaries["locations"]["state"].tolist()
    )
    
    device_id = generate_device_id(shared_pools.get("fraud_devices", []))
    
    row = {
        **base,
        "email": email,
        "email_domain": domain,
        "is_free_email_domain": is_free,
        "annual_income": annual_income,
        "housing_status": housing_status,
        "months_at_address": months_at_address,
        "months_at_employer": months_at_employer,
        "thin_file_flag": thin_file_flag,
        "device_id": device_id,
        "ip_region": ip_region,
        "application_hour": application_hour,
        "num_prev_apps_same_device_7d": num_prev_apps_same_device_7d,
        "num_prev_apps_same_email_30d": num_prev_apps_same_email_30d,
        "num_prev_apps_same_phone_30d": num_prev_apps_same_phone_30d,
        "num_prev_apps_same_address_30d": num_prev_apps_same_address_30d,
        "zip_ip_distance_proxy": round(zip_ip_distance_proxy, 3),
        "name_email_match_score": round(name_email_match_score, 3),
        "document_uploaded": document_uploaded,
        "fraud_label": 1,
        "fraud_type": "true_name_fraud",
        "difficulty_level": difficulty,
    }
    
    return row


def generate_coordinated_attack(dictionaries: dict, application_date: datetime,
                                 difficulty: str, shared_pools: dict,
                                 cluster_id: str = None) -> dict:
    """
    Generate a coordinated_attack fraud application.
    
    Characteristics:
    - Shared device/phone/address across cluster
    - Template-like patterns
    - Multiple applications with slight identity variation
    """
    base = generate_base_identity(dictionaries, application_date)
    
    # Email - mostly free domains
    email_domains = dictionaries["email_domains"]
    if random.random() < 0.8:
        domain_row = email_domains[email_domains["is_free"] == 1].sample(1).iloc[0]
    else:
        domain_row = email_domains[email_domains["is_free"] == 0].sample(1).iloc[0]
    
    domain = domain_row["domain"]
    is_free = int(domain_row["is_free"])
    
    match_name = random.random() < 0.3
    email = generate_email(base["claimed_first_name"], base["claimed_last_name"],
                          domain, match_name=match_name)
    
    # Behavioral fields - low tenure
    months_at_address = random.randint(0, 6)
    months_at_employer = random.randint(0, 6)
    thin_file_flag = 1 if random.random() < 0.7 else 0
    
    # Reuse counts - HIGH (coordinated attack signature)
    if difficulty == "easy":
        num_prev_apps_same_device_7d = random.randint(4, 8)
        num_prev_apps_same_email_30d = random.randint(0, 2)
        num_prev_apps_same_phone_30d = random.randint(2, 5)
        num_prev_apps_same_address_30d = random.randint(3, 7)
    elif difficulty == "medium":
        num_prev_apps_same_device_7d = random.randint(2, 5)
        num_prev_apps_same_email_30d = random.randint(0, 1)
        num_prev_apps_same_phone_30d = random.randint(1, 3)
        num_prev_apps_same_address_30d = random.randint(2, 4)
    else:  # hard
        num_prev_apps_same_device_7d = random.randint(1, 3)
        num_prev_apps_same_email_30d = random.randint(0, 1)
        num_prev_apps_same_phone_30d = random.randint(0, 2)
        num_prev_apps_same_address_30d = random.randint(1, 3)
    
    # IP/ZIP mismatch
    if difficulty == "easy":
        zip_ip_distance_proxy = random.uniform(0.5, 1.0)
    elif difficulty == "medium":
        zip_ip_distance_proxy = random.uniform(0.3, 0.7)
    else:
        zip_ip_distance_proxy = random.uniform(0.2, 0.5)
    
    # Application hour - coordinated timing
    if random.random() < 0.5:
        application_hour = random.choice([1, 2, 3, 4, 5, 23])
    else:
        application_hour = random.randint(0, 23)
    
    document_uploaded = 1 if random.random() < 0.6 else 0
    
    housing_status = random.choices(
        ["rent", "family", "other", "own"],
        weights=[0.45, 0.35, 0.15, 0.05]
    )[0]
    
    annual_income = generate_income(base["employer_industry"], base["age"])
    
    name_email_match_score = compute_name_email_match_score(
        base["claimed_first_name"], base["claimed_last_name"], email
    )
    
    # IP region - shared across cluster
    ip_region = random.choice(dictionaries["locations"]["state"].tolist())
    
    # Use shared cluster device
    if "cluster_devices" in shared_pools and cluster_id in shared_pools["cluster_devices"]:
        device_id = shared_pools["cluster_devices"][cluster_id]
    else:
        device_id = generate_device_id(shared_pools.get("fraud_devices", []))
    
    row = {
        **base,
        "email": email,
        "email_domain": domain,
        "is_free_email_domain": is_free,
        "annual_income": annual_income,
        "housing_status": housing_status,
        "months_at_address": months_at_address,
        "months_at_employer": months_at_employer,
        "thin_file_flag": thin_file_flag,
        "device_id": device_id,
        "ip_region": ip_region,
        "application_hour": application_hour,
        "num_prev_apps_same_device_7d": num_prev_apps_same_device_7d,
        "num_prev_apps_same_email_30d": num_prev_apps_same_email_30d,
        "num_prev_apps_same_phone_30d": num_prev_apps_same_phone_30d,
        "num_prev_apps_same_address_30d": num_prev_apps_same_address_30d,
        "zip_ip_distance_proxy": round(zip_ip_distance_proxy, 3),
        "name_email_match_score": round(name_email_match_score, 3),
        "document_uploaded": document_uploaded,
        "fraud_label": 1,
        "fraud_type": "coordinated_attack",
        "difficulty_level": difficulty,
    }
    
    return row


# =============================================================================
# SIGNAL SCORE AND DIFFICULTY
# =============================================================================

def compute_generated_signal_score(row: dict) -> float:
    """
    Compute a hidden latent fraud signal score based on fraud indicators.
    
    This score is for metadata/debugging only, not a real production feature.
    
    Args:
        row: Application row data
    
    Returns:
        float: Score from 0 to 1 (higher = more fraud-like)
    """
    score = 0.0
    
    # Device reuse (strong signal)
    if row["num_prev_apps_same_device_7d"] >= 3:
        score += 0.15
    elif row["num_prev_apps_same_device_7d"] >= 1:
        score += 0.08
    
    # Phone reuse
    if row["num_prev_apps_same_phone_30d"] >= 2:
        score += 0.10
    elif row["num_prev_apps_same_phone_30d"] >= 1:
        score += 0.05
    
    # Address reuse
    if row["num_prev_apps_same_address_30d"] >= 3:
        score += 0.12
    elif row["num_prev_apps_same_address_30d"] >= 1:
        score += 0.06
    
    # Email reuse
    if row["num_prev_apps_same_email_30d"] >= 2:
        score += 0.08
    elif row["num_prev_apps_same_email_30d"] >= 1:
        score += 0.04
    
    # Free email domain
    if row["is_free_email_domain"] == 1:
        score += 0.05
    
    # Low name/email match
    if row["name_email_match_score"] < 0.3:
        score += 0.10
    elif row["name_email_match_score"] < 0.5:
        score += 0.05
    
    # Low tenure
    if row["months_at_address"] < 6:
        score += 0.08
    elif row["months_at_address"] < 12:
        score += 0.04
    
    if row["months_at_employer"] < 3:
        score += 0.08
    elif row["months_at_employer"] < 6:
        score += 0.04
    
    # Thin file
    if row["thin_file_flag"] == 1:
        score += 0.10
    
    # ZIP/IP mismatch
    if row["zip_ip_distance_proxy"] > 0.7:
        score += 0.12
    elif row["zip_ip_distance_proxy"] > 0.4:
        score += 0.06
    
    # Unusual application hour
    if row["application_hour"] in [0, 1, 2, 3, 4, 5]:
        score += 0.05
    
    # No document
    if row["document_uploaded"] == 0:
        score += 0.05
    
    # Add noise
    score += random.uniform(-0.05, 0.05)
    
    # Clamp to [0, 1]
    return round(max(0.0, min(1.0, score)), 3)


def assign_difficulty_level(fraud_type: str) -> str:
    """
    Assign a difficulty level based on fraud type distribution.
    
    Args:
        fraud_type: Type of fraud/legitimate
    
    Returns:
        str: Difficulty level (easy, medium, hard)
    """
    dist = DIFFICULTY_DISTRIBUTION[fraud_type]
    return random.choices(
        list(dist.keys()),
        weights=list(dist.values())
    )[0]


# =============================================================================
# DATASET CREATION
# =============================================================================

def create_dataset(n_rows: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Create the synthetic fraud detection dataset.
    
    Args:
        n_rows: Number of rows to generate
        seed: Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Loading dictionaries...")
    dictionaries = load_dictionaries()
    
    print(f"Generating {n_rows} applications...")
    
    # Calculate counts per fraud type
    fraud_type_counts = {
        ft: int(n_rows * pct) 
        for ft, pct in FRAUD_TYPE_DISTRIBUTION.items()
    }
    
    # Adjust to ensure we hit exactly n_rows
    total = sum(fraud_type_counts.values())
    if total < n_rows:
        fraud_type_counts["legitimate_clean"] += (n_rows - total)
    
    # Create shared pools for fraud device reuse
    shared_pools = {
        "fraud_devices": [f"DEV-FRAUD-{i:04d}" for i in range(50)],
        "cluster_devices": {},
    }
    
    # Pre-create cluster device IDs
    n_clusters = fraud_type_counts["coordinated_attack"] // 5 + 1
    for i in range(n_clusters):
        cluster_id = f"CLUSTER-{i:03d}"
        shared_pools["cluster_devices"][cluster_id] = f"DEV-CLUSTER-{i:04d}"
    
    rows = []
    cluster_counter = 0
    
    for fraud_type, count in fraud_type_counts.items():
        print(f"  Generating {count} {fraud_type} rows...")
        
        for i in range(count):
            # Generate application date
            app_date = generate_application_date(START_DATE, END_DATE)
            
            # Assign difficulty
            difficulty = assign_difficulty_level(fraud_type)
            
            # Generate row based on fraud type
            if fraud_type == "legitimate_clean":
                row = generate_legitimate_clean(dictionaries, app_date, difficulty)
            elif fraud_type == "legitimate_noisy":
                row = generate_legitimate_noisy(dictionaries, app_date, difficulty)
            elif fraud_type == "synthetic_identity":
                row = generate_synthetic_identity(dictionaries, app_date, difficulty, shared_pools)
            elif fraud_type == "true_name_fraud":
                row = generate_true_name_fraud(dictionaries, app_date, difficulty, shared_pools)
            elif fraud_type == "coordinated_attack":
                cluster_id = f"CLUSTER-{cluster_counter // 5:03d}"
                row = generate_coordinated_attack(dictionaries, app_date, difficulty, 
                                                   shared_pools, cluster_id)
                cluster_counter += 1
            
            # Add identifiers and time fields
            row["application_id"] = generate_application_id()
            row["application_date"] = app_date.strftime("%Y-%m-%d")
            row["application_month"] = app_date.strftime("%Y-%m")
            
            # Generate text fields
            template_data = {
                "first_name": row["claimed_first_name"],
                "last_name": row["claimed_last_name"],
                "city": row["city"],
                "state": row["state"],
                "zip_code": row["zip_code"],
                "address_line": row["address_line"],
                "employer_name": row["employer_name"],
                "employer_industry": row["employer_industry"],
                "annual_income": row["annual_income"],
                "months_at_employer": row["months_at_employer"],
            }
            
            row["verification_note"] = generate_verification_note(
                fraud_type, difficulty, dictionaries, template_data
            )
            row["ocr_document_text"] = generate_ocr_text(
                fraud_type, difficulty, dictionaries, template_data
            )
            row["address_explanation_text"] = generate_address_explanation(
                fraud_type, difficulty, dictionaries, template_data
            )
            row["employment_explanation_text"] = generate_employment_explanation(
                fraud_type, difficulty, dictionaries, template_data
            )
            
            # Compute signal score
            row["generated_signal_score"] = compute_generated_signal_score(row)
            
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns to match schema
    column_order = [
        # Identifiers / Time
        "application_id", "application_date", "application_month",
        # Claimed Identity
        "claimed_first_name", "claimed_last_name", "date_of_birth", "age", "ssn_last4",
        # Address
        "address_line", "city", "state", "zip_code",
        # Contact
        "phone_number", "email", "email_domain", "is_free_email_domain",
        # Employment / Financial
        "employer_name", "employer_industry", "annual_income", "housing_status",
        "months_at_address", "months_at_employer", "thin_file_flag",
        # Digital / Behavioral
        "device_id", "ip_region", "application_hour",
        "num_prev_apps_same_device_7d", "num_prev_apps_same_email_30d",
        "num_prev_apps_same_phone_30d", "num_prev_apps_same_address_30d",
        "zip_ip_distance_proxy",
        # Precomputed signals
        "name_email_match_score", "document_uploaded",
        # Text fields
        "verification_note", "ocr_document_text", 
        "address_explanation_text", "employment_explanation_text",
        # Labels / Meta
        "fraud_label", "fraud_type", "difficulty_level", "generated_signal_score",
    ]
    
    df = df[column_order]
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"Dataset created with {len(df)} rows.")
    
    return df


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the generated dataset for correctness.
    
    Args:
        df: Generated DataFrame
    
    Raises:
        ValueError: If validation fails
    """
    print("Validating dataset...")
    
    # Check for required columns
    required_columns = [
        "application_id", "application_date", "application_month",
        "claimed_first_name", "claimed_last_name", "date_of_birth", "age", "ssn_last4",
        "address_line", "city", "state", "zip_code",
        "phone_number", "email", "email_domain", "is_free_email_domain",
        "employer_name", "employer_industry", "annual_income", "housing_status",
        "months_at_address", "months_at_employer", "thin_file_flag",
        "device_id", "ip_region", "application_hour",
        "num_prev_apps_same_device_7d", "num_prev_apps_same_email_30d",
        "num_prev_apps_same_phone_30d", "num_prev_apps_same_address_30d",
        "zip_ip_distance_proxy", "name_email_match_score", "document_uploaded",
        "verification_note", "ocr_document_text",
        "address_explanation_text", "employment_explanation_text",
        "fraud_label", "fraud_type", "difficulty_level", "generated_signal_score",
    ]
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check fraud_label values
    invalid_labels = set(df["fraud_label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Invalid fraud_label values: {invalid_labels}")
    
    # Check fraud_type values
    invalid_types = set(df["fraud_type"].unique()) - set(VALID_FRAUD_TYPES)
    if invalid_types:
        raise ValueError(f"Invalid fraud_type values: {invalid_types}")
    
    # Check difficulty_level values
    invalid_difficulty = set(df["difficulty_level"].unique()) - set(VALID_DIFFICULTY_LEVELS)
    if invalid_difficulty:
        raise ValueError(f"Invalid difficulty_level values: {invalid_difficulty}")
    
    # Check age range
    if df["age"].min() < 18 or df["age"].max() > 100:
        raise ValueError(f"Age out of range: min={df['age'].min()}, max={df['age'].max()}")
    
    # Check application_id uniqueness
    if df["application_id"].duplicated().any():
        raise ValueError("Duplicate application_id values found")
    
    print("Validation passed!")


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_outputs(df: pd.DataFrame, output_name: str = "applications_prototype") -> dict:
    """
    Save the generated dataset and metadata.
    
    Args:
        df: Generated DataFrame
        output_name: Base name for output files
    
    Returns:
        dict: Paths to saved files
    """
    # Ensure output directories exist
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    parquet_path = DATA_RAW / f"{output_name}.parquet"
    csv_path = DATA_RAW / f"{output_name}.csv"
    
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    
    print(f"Saved: {parquet_path}")
    print(f"Saved: {csv_path}")
    
    # Save metadata
    metadata_cols = ["application_id", "fraud_type", "difficulty_level", "generated_signal_score"]
    metadata_df = df[metadata_cols]
    metadata_path = DATA_INTERIM / "generation_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Saved: {metadata_path}")
    
    return {
        "parquet": parquet_path,
        "csv": csv_path,
        "metadata": metadata_path,
    }


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the generated dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal rows: {len(df)}")
    
    # Fraud rate
    fraud_rate = df["fraud_label"].mean() * 100
    print(f"Fraud rate: {fraud_rate:.1f}%")
    
    # Fraud type distribution
    print("\nFraud type distribution:")
    fraud_type_counts = df["fraud_type"].value_counts()
    for ft, count in fraud_type_counts.items():
        pct = count / len(df) * 100
        print(f"  {ft}: {count} ({pct:.1f}%)")
    
    # Difficulty distribution
    print("\nDifficulty distribution:")
    difficulty_counts = df["difficulty_level"].value_counts()
    for diff, count in difficulty_counts.items():
        pct = count / len(df) * 100
        print(f"  {diff}: {count} ({pct:.1f}%)")
    
    # Date range
    print(f"\nDate range: {df['application_date'].min()} to {df['application_date'].max()}")
    
    print("=" * 60)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to generate the prototype dataset."""
    print("=" * 60)
    print("IDENTITY FRAUD DETECTION - DATA GENERATOR")
    print("=" * 60)
    
    # Generate dataset
    df = create_dataset(n_rows=5000)
    
    # Validate
    validate_dataset(df)
    
    # Save outputs
    save_outputs(df)
    
    # Print summary
    print_summary(df)
    
    print("\nData generation complete!")
    
    return df


if __name__ == "__main__":
    main()
