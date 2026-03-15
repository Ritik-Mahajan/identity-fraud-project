"""
inspect_dictionaries.py

Utility script to inspect all source dictionary CSV files in data/external/.
Prints file name, row count, columns, and first 3 rows for each file.
Fails clearly if any required file is missing.

Usage:
    python src/inspect_dictionaries.py
"""

import pandas as pd
from pathlib import Path
import sys


# Define the project root and data directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "external"

# List of required dictionary files
REQUIRED_FILES = [
    "first_names.csv",
    "last_names.csv",
    "cities_states_zips.csv",
    "street_names.csv",
    "employers.csv",
    "email_domains.csv",
    "legit_verification_note_templates.csv",
    "fraud_verification_note_templates.csv",
    "legit_ocr_templates.csv",
    "fraud_ocr_templates.csv",
    "address_explanation_templates.csv",
    "employment_explanation_templates.csv",
]


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists and return True/False."""
    return filepath.exists()


def load_and_inspect(filepath: Path) -> None:
    """Load a CSV file and print its summary information."""
    # Load the CSV
    df = pd.read_csv(filepath)
    
    # Print file information
    print(f"\n{'='*60}")
    print(f"File: {filepath.name}")
    print(f"{'='*60}")
    print(f"Row count: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


def main():
    """Main function to inspect all dictionary files."""
    print("=" * 60)
    print("SOURCE DICTIONARY INSPECTION")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory does not exist: {DATA_DIR}")
        sys.exit(1)
    
    # Check for missing files first
    missing_files = []
    for filename in REQUIRED_FILES:
        filepath = DATA_DIR / filename
        if not check_file_exists(filepath):
            missing_files.append(filename)
    
    # If any files are missing, report and exit
    if missing_files:
        print(f"\nERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    
    print(f"\nAll {len(REQUIRED_FILES)} required files found.")
    
    # Inspect each file
    for filename in REQUIRED_FILES:
        filepath = DATA_DIR / filename
        try:
            load_and_inspect(filepath)
        except Exception as e:
            print(f"\nERROR reading {filename}: {e}")
            sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files inspected: {len(REQUIRED_FILES)}")
    print("All files loaded successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
