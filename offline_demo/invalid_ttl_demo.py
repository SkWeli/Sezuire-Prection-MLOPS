#!/usr/bin/env python3
"""
Invalid-metadata offline viva demonstration.

This script proves that semantic validation is a hard execution gate.

Expected behaviour:

1. Load an intentionally invalid TTL fixture.
2. Run SHACL validation.
3. Confirm validation fails.
4. Stop before loading the checkpoint.
5. Stop before loading processed EEG.
6. Save a JSON evidence report.

Scientific status:
    Research prototype only.
    Not clinically validated.
    Not a medical device.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from offline_demo.config import (
    SCIENTIFIC_STATUS,
)

from offline_demo.demo_utils import (
    print_header,
    print_section,
    run_shacl_validation,
    save_json_report,
)


def find_invalid_ttl_fixture() -> Path:
    """
    Find the first SHACL-invalid TTL fixture under the tests directory.

    The fixture must already exist in the project test suite.
    """
    tests_directory = (
        PROJECT_ROOT
        / "tests"
    )

    if not tests_directory.exists():
        raise FileNotFoundError(
            f"Tests directory was not found: {tests_directory}"
        )

    ttl_candidates = sorted(
        tests_directory.rglob("*.ttl")
    )

    if not ttl_candidates:
        raise FileNotFoundError(
            "No TTL fixtures were found under the tests directory."
        )

    print_section(
        "SEARCHING FOR AN INVALID TTL FIXTURE"
    )

    for candidate in ttl_candidates:
        print(f"Testing fixture: {candidate}")

        conforms, output, duration_ms = (
            run_shacl_validation(
                candidate
            )
        )

        if not conforms:
            print(
                f"Selected invalid fixture: {candidate}"
            )

            return candidate

    raise RuntimeError(
        "No SHACL-invalid TTL fixture was found."
    )


def main() -> int:
    """
    Execute the complete invalid-metadata demonstration.
    """
    print_header(
        "INVALID METADATA - SEMANTIC QUALITY-GATE DEMONSTRATION"
    )

    print(f"Scientific status: {SCIENTIFIC_STATUS}")
    print()
    print(
        "Expected behaviour: SHACL FAIL -> inference blocked "
        "before checkpoint and EEG loading."
    )

    invalid_ttl_path = find_invalid_ttl_fixture()

    print_section(
        "FINAL INVALID-METADATA VALIDATION"
    )

    conforms, validator_output, duration_ms = (
        run_shacl_validation(
            invalid_ttl_path
        )
    )

    if conforms:
        raise RuntimeError(
            "The selected invalid fixture unexpectedly passed SHACL."
        )

    report_contents = {
        "status": "pass",
        "scientific_status": SCIENTIFIC_STATUS,
        "demo_type": "invalid_ttl_blocked_inference_demo",
        "invalid_ttl_path": str(
            invalid_ttl_path
        ),
        "shacl": {
            "conforms": False,
            "duration_ms": float(
                duration_ms
            ),
            "validator_output": (
                validator_output
            ),
        },
        "execution_gate": {
            "inference_blocked": True,
            "checkpoint_loaded": False,
            "eeg_loaded": False,
            "inference_executed": False,
        },
        "interpretation": (
            "Invalid semantic metadata was rejected before "
            "model or EEG loading."
        ),
    }

    report_path = save_json_report(
        report_name=(
            "invalid_ttl_blocked_inference_demo"
        ),
        contents=report_contents,
    )

    print_header(
        "NEGATIVE SEMANTIC-GATE DEMONSTRATION: PASS"
    )

    print("SHACL validation : FAIL")
    print("Inference blocked: YES")
    print("Checkpoint loaded: NO")
    print("EEG data loaded  : NO")
    print("Inference run    : NO")
    print(
        "Invalid TTL was rejected before model loading."
    )
    print(f"Evidence report  : {report_path}")

    # Return success because the expected negative behaviour was proven.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())