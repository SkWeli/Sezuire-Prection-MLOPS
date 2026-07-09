"""
src/validation/rdf_generator.py

This file creates a simple RDF/Turtle (.ttl) metadata file
for each processed EEG patient/subject.

Why we need this:
- The .npz file stores EEG arrays for machine learning.
- SHACL cannot validate .npz directly.
- So we create a .ttl file that describes the .npz file semantically.
- Then SHACL can check whether required metadata is present.
"""

from pathlib import Path
import argparse
import numpy as np


def clean_name(value):
    """
    Make a text value safe for RDF resource names.

    Example:
        "FP1-F7" -> "FP1_F7"

    Why:
    RDF resource names should not contain messy characters like spaces,
    slashes, or hyphens in the local identifier.
    """
    return (
        str(value)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def generate_tusz_ttl(npz_path, ttl_path=None):
    """
    Generate a Turtle (.ttl) metadata file for one processed TUSZ .npz file.

    Parameters
    ----------
    npz_path : str or Path
        Path to the processed .npz file.
        Example: data/processed/tusz/aaaaaajy/aaaaaajy.npz

    ttl_path : str or Path or None
        Output path for the .ttl file.
        If None, it saves beside the .npz file.
        Example: data/processed/tusz/aaaaaajy/aaaaaajy.ttl
    """

    # Convert input path to a Path object
    npz_path = Path(npz_path)

    # If user did not provide TTL path, create it beside the .npz file
    if ttl_path is None:
        ttl_path = npz_path.with_suffix(".ttl")
    else:
        ttl_path = Path(ttl_path)

    # Check whether the .npz file exists
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # Load the processed EEG metadata from .npz
    data = np.load(npz_path, allow_pickle=True)

    # Read values saved by your preprocessing code
    patient_id = str(data["patient_id"])
    ch_names = list(data["ch_names"])
    sfreq = float(data["sfreq"])
    n_windows = int(data["n_windows"])
    n_interictal = int(data.get("n_interictal", 0))
    n_pre_ictal  = int(data.get("n_pre_ictal", 0))
    n_ictal      = int(data.get("n_ictal", 0))

    # Create simple IDs for RDF resources
    patient_res = f"patient_{clean_name(patient_id)}"
    session_res = f"session_{clean_name(patient_id)}"
    sampling_res = f"sampling_{clean_name(patient_id)}"

    #Calculate total seizures or map them directly into your RDF triples
    n_seizures = n_ictal

    # These are the preprocessing steps your TUSZ loader applies
    preprocessing_steps = [
        ("bandpass_filter", 1),
        ("notch_filter", 2),
        ("average_reference", 3),
        ("resampling", 4),
        ("sliding_window_epoching", 5),
        ("padding_or_truncation", 6),
    ]

    # Start writing the TTL content as plain text
    # This keeps the code very simple and easy to understand.
    ttl_lines = []

    # Prefixes tell RDF what short names mean
    ttl_lines.append("@prefix eeg: <http://example.org/eeg-epilepsy#> .")
    ttl_lines.append("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
    ttl_lines.append("")

    # Patient individual
    ttl_lines.append(f"eeg:{patient_res} a eeg:Patient ;")
    ttl_lines.append(f'    eeg:subjectID "{patient_id}" ;')
    ttl_lines.append(f"    eeg:hasSession eeg:{session_res} .")
    ttl_lines.append("")

    # Recording session individual
    # Important: SHACL requires sessionID, at least 19 channels,
    # one sampling rate, and at least one preprocessing step.
    ttl_lines.append(f"eeg:{session_res} a eeg:RecordingSession ;")
    ttl_lines.append(f'    eeg:sessionID "{patient_id}_session" ;')
    ttl_lines.append(f"    eeg:hasSamplingRate eeg:{sampling_res} ;")

    # Add channel links
    for index, channel_name in enumerate(ch_names):
        channel_res = f"channel_{clean_name(patient_id)}_{index + 1}"
        ttl_lines.append(f"    eeg:hasChannel eeg:{channel_res} ;")

    # Add preprocessing step links
    for step_name, step_order in preprocessing_steps:
        step_res = f"step_{clean_name(patient_id)}_{step_order}"
        if step_order == preprocessing_steps[-1][1]:
            # Last property ends with a dot
            ttl_lines.append(f"    eeg:hasPreprocessingStep eeg:{step_res} .")
        else:
            ttl_lines.append(f"    eeg:hasPreprocessingStep eeg:{step_res} ;")

    ttl_lines.append("")

    # Sampling rate individual
    ttl_lines.append(f"eeg:{sampling_res} a eeg:SamplingRate ;")
    ttl_lines.append(f'    eeg:frequencyHz "{sfreq}"^^xsd:float .')
    ttl_lines.append("")

    # Channel individuals
    for index, channel_name in enumerate(ch_names):
        channel_res = f"channel_{clean_name(patient_id)}_{index + 1}"

        ttl_lines.append(f"eeg:{channel_res} a eeg:EEGChannel ;")
        ttl_lines.append(f'    eeg:channelLabel "{channel_name}" .')
        ttl_lines.append("")

    # Preprocessing step individuals
    for step_name, step_order in preprocessing_steps:
        step_res = f"step_{clean_name(patient_id)}_{step_order}"

        ttl_lines.append(f"eeg:{step_res} a eeg:PreprocessingStep ;")
        ttl_lines.append(f'    eeg:filterType "{step_name}" ;')
        ttl_lines.append(f'    eeg:stepOrder "{step_order}"^^xsd:integer .')
        ttl_lines.append("")

    # Extra metadata
    # Your current SHACL file may not check these yet,
    # but they are useful for thesis/demo evidence.
    ttl_lines.append(f"eeg:{session_res}")
    ttl_lines.append(f'    eeg:processedFile "{npz_path.as_posix()}" ;')
    ttl_lines.append(f'    eeg:numberOfWindows "{n_windows}"^^xsd:integer ;')
    ttl_lines.append(f'    eeg:numberOfSeizureWindows "{n_seizures}"^^xsd:integer .')
    ttl_lines.append("")

    # Save the TTL file
    ttl_path.write_text("\n".join(ttl_lines), encoding="utf-8")

    print(f"✅ TTL metadata generated: {ttl_path}")
    return ttl_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RDF/Turtle metadata for a processed EEG .npz file"
    )

    parser.add_argument(
        "--npz",
        required=True,
        help="Path to processed .npz file"
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Optional output .ttl path"
    )

    args = parser.parse_args()

    generate_tusz_ttl(args.npz, args.out)