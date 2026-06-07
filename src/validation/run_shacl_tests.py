"""
src/validation/run_shacl_tests.py

This script runs controlled SHACL validation tests.

Why this is needed:
- A valid TTL file should PASS.
- Invalid TTL files should FAIL.
- This proves our SHACL rules can detect EEG metadata errors.
"""

import subprocess
import sys
from pathlib import Path


# Each test has:
# 1. TTL file path
# 2. Expected result
#
# True  = should pass SHACL
# False = should fail SHACL
TEST_CASES = [
    ("tests/shacl/valid_patient.ttl", True),
    ("tests/shacl/invalid_missing_sampling_rate.ttl", False),
    ("tests/shacl/invalid_missing_channels.ttl", False),
]


def run_one_test(ttl_file, expected_pass):
    """
    Run one SHACL test case.
    """

    print()
    print("=================================================")
    print(f"Testing file     : {ttl_file}")
    print(f"Expected result  : {'PASS' if expected_pass else 'FAIL'}")
    print("=================================================")

    # Run the validator using the same Python environment.
    result = subprocess.run(
        [
            sys.executable,
            "src/validation/shacl_validator.py",
            ttl_file,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Print validator output
    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    # Validator return codes:
    # 0 = SHACL passed
    # 1 = SHACL failed
    # 2 = unexpected error
    actual_pass = result.returncode == 0

    if actual_pass == expected_pass:
        print("[OK] Test behaved as expected.")
        return True

    print("[WRONG] Test did not behave as expected.")
    print(f"Expected pass: {expected_pass}")
    print(f"Actual pass  : {actual_pass}")
    print(f"Return code  : {result.returncode}")
    return False


def main():
    """
    Run all SHACL test cases.
    """

    all_ok = True

    for ttl_file, expected_pass in TEST_CASES:
        if not Path(ttl_file).exists():
            print(f"[MISSING] Test file not found: {ttl_file}")
            all_ok = False
            continue

        test_ok = run_one_test(ttl_file, expected_pass)

        if not test_ok:
            all_ok = False

    print()
    print("=================================================")

    if all_ok:
        print("[SUCCESS] All SHACL tests behaved as expected.")
        print("=================================================")
        return 0

    print("[FAILED] Some SHACL tests did not behave as expected.")
    print("=================================================")
    return 1


if __name__ == "__main__":
    sys.exit(main())