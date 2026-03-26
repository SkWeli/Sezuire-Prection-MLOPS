"""
src/validation/shacl_validator.py
──────────────────────────────────
CLI tool that validates an RDF data graph against the EEG epilepsy
SHACL shapes.  Designed to be invoked from GitHub Actions CI or DVC
pipeline stages.

Usage
-----
    python -m src.validation.shacl_validator path/to/data_graph.ttl
    python -m src.validation.shacl_validator path/to/data_graph.ttl 
        --shapes ontology/shacl_shapes.ttl 
        --ontology ontology/eeg_epilepsy.ttl

Exit codes
----------
    0 — validation passed (confirms=True)
    1 — validation failed (constraint violations found)
    2 — usage / file-not-found error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyshacl import validate
from rdflib import Graph

# Default file locations (relative to repo root)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SHAPES    = _REPO_ROOT / "ontology" / "shacl_shapes.ttl"
_DEFAULT_ONTOLOGY  = _REPO_ROOT / "ontology" / "eeg_epilepsy.ttl"


def _load_graph(path: Path, label: str) -> Graph:
    """Parse a Turtle file into an rdflib Graph, with a clear error on failure."""
    if not path.is_file():
        print(f"[ERROR] {label} file not found: {path}", file=sys.stderr)
        sys.exit(2)
    g = Graph()
    try:
        g.parse(str(path), format="turtle")
    except Exception as exc:
        print(f"[ERROR] Failed to parse {label} '{path}': {exc}", file=sys.stderr)
        sys.exit(2)
    return g


def run_validation(
    data_graph_path: Path,
    shapes_path: Path = _DEFAULT_SHAPES,
    ontology_path: Path | None = _DEFAULT_ONTOLOGY,
    *,
    inference: str = "rdfs",
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Load the data graph and shapes graph, run pyshacl validation, and
    return (conforms, report_text).

    Parameters
    ----------
    data_graph_path : Path
        RDF Turtle file containing the instance data to validate.
    shapes_path : Path
        SHACL shapes graph Turtle file.
    ontology_path : Path | None
        Optional OWL ontology for inference-based expansion.
    inference : str
        pyshacl inference mode: 'rdfs', 'owlrl', 'both', or 'none'.
    verbose : bool
        Pass through to pyshacl for extra debug output.

    Returns
    -------
    (conforms, report_text)
    """
    data_graph  = _load_graph(data_graph_path, "data graph")
    shapes_graph = _load_graph(shapes_path, "shapes graph")

    ont_graph: Graph | None = None
    if ontology_path is not None and ontology_path.is_file():
        ont_graph = _load_graph(ontology_path, "ontology graph")

    conforms, results_graph, results_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        ont_graph=ont_graph,
        inference=inference,
        abort_on_first=False,   # collect ALL violations, not just the first
        allow_infos=True,
        allow_warnings=True,
        meta_shacl=False,
        debug=verbose,
    )

    return conforms, results_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shacl_validator",
        description="Validate an EEG pipeline RDF graph against SHACL constraints.",
    )
    parser.add_argument(
        "data_graph",
        type=Path,
        help="Path to the RDF data graph (Turtle .ttl file) to validate.",
    )
    parser.add_argument(
        "--shapes",
        type=Path,
        default=_DEFAULT_SHAPES,
        metavar="PATH",
        help=f"Path to SHACL shapes file (default: {_DEFAULT_SHAPES}).",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=_DEFAULT_ONTOLOGY,
        metavar="PATH",
        help=(
            f"Path to OWL ontology file for RDFS inference expansion "
            f"(default: {_DEFAULT_ONTOLOGY}).  Pass 'none' to disable."
        ),
    )
    parser.add_argument(
        "--inference",
        choices=["rdfs", "owlrl", "both", "none"],
        default="rdfs",
        help="Inference mode passed to pyshacl (default: rdfs).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable pyshacl debug output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns exit code (0 = pass, 1 = fail, 2 = error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    ontology_path: Path | None = args.ontology
    if str(ontology_path).lower() == "none":
        ontology_path = None

    print(f"[INFO] Data graph : {args.data_graph}")
    print(f"[INFO] Shapes     : {args.shapes}")
    print(f"[INFO] Ontology   : {ontology_path or '(disabled)'}")
    print(f"[INFO] Inference  : {args.inference}")
    print()

    conforms, report_text = run_validation(
        data_graph_path=args.data_graph,
        shapes_path=args.shapes,
        ontology_path=ontology_path,
        inference=args.inference,
        verbose=args.verbose,
    )

    if conforms:
        print("✅  PASS — data graph conforms to all SHACL constraints.")
        print()
        print(report_text)
        return 0
    else:
        print("❌  FAIL — data graph violates one or more SHACL constraints.")
        print()
        print("── Violation Report ──────────────────────────────────────────")
        print(report_text)
        print("──────────────────────────────────────────────────────────────")
        return 1


if __name__ == "__main__":
    sys.exit(main())
