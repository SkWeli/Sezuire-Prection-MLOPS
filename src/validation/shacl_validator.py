"""
src/validation/shacl_validator.py

This script validates an RDF/Turtle (.ttl) data file using SHACL rules.

Why this is needed:
- Our EEG pipeline generates RDF metadata files.
- SHACL checks whether those metadata files follow our semantic rules.
- If validation fails, the pipeline should stop before model training.
"""

import argparse
import sys
from pathlib import Path

from pyshacl import validate


# Default project paths
DEFAULT_SHAPES_FILE = Path("ontology/shacl_shapes.ttl")
DEFAULT_ONTOLOGY_FILE = Path("ontology/eeg_epilepsy.ttl")


def run_shacl_validation(data_file, shapes_file=DEFAULT_SHAPES_FILE, ontology_file=DEFAULT_ONTOLOGY_FILE):
    """
    Run SHACL validation.

    Parameters
    ----------
    data_file : str or Path
        RDF/Turtle file to validate.
        Example: data/processed/tusz/aaaaaajy/aaaaaajy.ttl

    shapes_file : str or Path
        SHACL shapes file.
        Example: ontology/shacl_shapes.ttl

    ontology_file : str or Path
        Ontology file used for inference.
        Example: ontology/eeg_epilepsy.ttl

    Returns
    -------
    bool
        True if data conforms to SHACL.
        False if data violates SHACL.
    str
        SHACL validation report text.
    """

    data_file = Path(data_file)
    shapes_file = Path(shapes_file)
    ontology_file = Path(ontology_file)

    # Basic file checks
    if not data_file.exists():
        raise FileNotFoundError(f"Data graph not found: {data_file}")

    if not shapes_file.exists():
        raise FileNotFoundError(f"SHACL shapes file not found: {shapes_file}")

    if not ontology_file.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_file}")

    print(f"[INFO] Data graph : {data_file}")
    print(f"[INFO] Shapes     : {shapes_file.resolve()}")
    print(f"[INFO] Ontology   : {ontology_file.resolve()}")
    print("[INFO] Inference  : rdfs")
    print()

    # Run SHACL validation
    conforms, results_graph, results_text = validate(
        data_graph=str(data_file),
        shacl_graph=str(shapes_file),
        ont_graph=str(ontology_file),
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
        meta_shacl=False,
        advanced=True,
        js=False,
    )

    return conforms, results_text

def run_validation(
    data_graph_path,
    shapes_path=DEFAULT_SHAPES_FILE,
    ontology_path=DEFAULT_ONTOLOGY_FILE,
    inference="rdfs",
):
    """
    Compatibility entry point used by the pytest test suite.

    Parameters use the names expected by tests/test_shacl.py and are
    translated to the existing run_shacl_validation() implementation.
    """
    result = run_shacl_validation(
        data_file=data_graph_path,
        shapes_file=shapes_path,
        ontology_file=ontology_path,
    )

    # pySHACL commonly returns:
    # (conforms, validation_report_graph, validation_report_text)
    if isinstance(result, tuple) and len(result) == 3:
        conforms, _report_graph, report_text = result
        return bool(conforms), str(report_text)

    # Preserve an existing two-value result.
    if isinstance(result, tuple) and len(result) == 2:
        conforms, report = result
        return bool(conforms), str(report)

    raise TypeError(
        "run_shacl_validation() returned an unexpected result: "
        f"{type(result).__name__}: {result!r}"
    )

def main():
    """
    Command-line entry point.

    Example:
        python src/validation/shacl_validator.py data/processed/tusz/aaaaaajy/aaaaaajy.ttl
    """

    parser = argparse.ArgumentParser(
        description="Validate RDF/Turtle EEG metadata using SHACL."
    )

    parser.add_argument(
        "data_file",
        help="Path to RDF/Turtle data file to validate."
    )

    parser.add_argument(
        "--shapes",
        default=str(DEFAULT_SHAPES_FILE),
        help="Path to SHACL shapes file."
    )

    parser.add_argument(
        "--ontology",
        default=str(DEFAULT_ONTOLOGY_FILE),
        help="Path to ontology file."
    )

    args = parser.parse_args()

    try:
        conforms, results_text = run_shacl_validation(
            data_file=args.data_file,
            shapes_file=args.shapes,
            ontology_file=args.ontology,
        )

        if conforms:
            print("[PASS] Data graph conforms to all SHACL constraints.")
            print()
            print(results_text)
            return 0

        print("[FAIL] Data graph violates one or more SHACL constraints.")
        print()
        print("---- Violation Report ----------------------------------------")
        print(results_text)
        print("--------------------------------------------------------------")
        return 1

    except Exception as error:
        print("[ERROR] SHACL validation failed because of an unexpected error.")
        print(str(error))
        return 2


if __name__ == "__main__":
    sys.exit(main())