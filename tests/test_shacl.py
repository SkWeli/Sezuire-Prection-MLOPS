"""
tests/test_shacl.py
────────────────────
pytest unit tests for the SHACL validation layer.

Each test builds a minimal RDF data graph as a Turtle string, writes it
to a temporary file, and runs it through run_validation() from
src/validation/shacl_validator.py.

Tested scenarios
----------------
  test_valid_graph_passes            – well-formed session passes all shapes
  test_missing_channels_fails        – session with < 19 channels is rejected
  test_invalid_sampling_rate_fails   – 100 Hz SamplingRate is rejected
  test_onset_after_offset_fails      – seizure where onset >= offset is rejected
  test_empty_subject_id_fails        – Patient with empty subjectID is rejected
  test_short_preictal_window_fails   – PreIctalWindow with 20 s duration rejected
  test_long_preictal_window_fails    – PreIctalWindow with 130 s duration rejected
  test_missing_filter_type_fails     – PreprocessingStep without filterType rejected
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from rdflib import Graph

# import the function under test 
from src.validation.shacl_validator import run_validation

# resolve paths relative to repo root 
_REPO_ROOT   = Path(__file__).resolve().parents[1]
_SHAPES_PATH = _REPO_ROOT / "ontology" / "shacl_shapes.ttl"
_ONTO_PATH   = _REPO_ROOT / "ontology" / "eeg_epilepsy.ttl"

# HELPERS

def _channel_block(n: int, start: int = 1) -> str:
    """Return Turtle triples declaring *n* EEGChannel individuals."""
    lines = []
    for i in range(start, start + n):
        lines.append(
            f"eeg:ch{i:02d} a eeg:EEGChannel ; eeg:channelLabel \"CH{i:02d}\" ."
        )
    return "\n".join(lines)


def _session_channel_links(n: int, start: int = 1) -> str:
    """Return eeg:hasChannel triples for *n* channels."""
    return " ; ".join(f"eeg:hasChannel eeg:ch{i:02d}" for i in range(start, start + n))


# Turtle prefix header reused across all test graphs
_PREFIXES = textwrap.dedent("""\
    @prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
    @prefix eeg:  <http://example.org/eeg-epilepsy#> .
""")


def _write_ttl(tmp_path: Path, content: str) -> Path:
    """Write Turtle content as UTF-8 and return its path."""
    p = tmp_path / "test_data.ttl"
    p.write_text(
        _PREFIXES + "\n" + textwrap.dedent(content),
        encoding="utf-8",
    )
    return p


def _validate(tmp_path: Path, content: str) -> tuple[bool, str]:
    """Build a temp graph, run SHACL validation, return (conforms, report)."""
    path = _write_ttl(tmp_path, content)
    return run_validation(
        data_graph_path=path,
        shapes_path=_SHAPES_PATH,
        ontology_path=_ONTO_PATH,
        inference="rdfs",
    )

# TESTS
class TestValidGraph:
    """A fully compliant graph must pass all SHACL constraints."""

    def test_valid_graph_passes(self, tmp_path: Path) -> None:
        channels_decl  = _channel_block(19)
        channels_links = _session_channel_links(19)

        ttl = f"""
        # ── Patient ─────────────────────────────────
        eeg:patient01 a eeg:Patient ;
            eeg:subjectID   "chb01" ;
            eeg:hasSession  eeg:session01 .

        # ── SamplingRate ─────────────────────────────
        eeg:sr256 a eeg:SamplingRate ;
            eeg:frequencyHz "256.0"^^xsd:float .

        # ── 19 EEG Channels ──────────────────────────
        {channels_decl}

        # ── PreprocessingStep ────────────────────────
        eeg:step01 a eeg:PreprocessingStep ;
            eeg:stepOrder  "1"^^xsd:integer ;
            eeg:filterType "notch" .

        eeg:step02 a eeg:PreprocessingStep ;
            eeg:stepOrder  "2"^^xsd:integer ;
            eeg:filterType "bandpass" .

        # ── PreIctalWindow ───────────────────────────
        eeg:piw01 a eeg:PreIctalWindow ;
            eeg:windowDuration "60.0"^^xsd:float ;
            eeg:windowStart    "40.0"^^xsd:float ;
            eeg:windowEnd      "100.0"^^xsd:float .

        # ── SeizureEvent ─────────────────────────────
        eeg:sz01 a eeg:SeizureEvent ;
            eeg:hasOnset          "100.0"^^xsd:float ;
            eeg:hasOffset         "130.0"^^xsd:float ;
            eeg:hasPreIctalWindow eeg:piw01 .

        # ── RecordingSession ─────────────────────────
        eeg:session01 a eeg:RecordingSession ;
            eeg:sessionID             "chb01_01" ;
            eeg:hasSamplingRate       eeg:sr256 ;
            eeg:hasSeizureEvent       eeg:sz01 ;
            eeg:hasPreprocessingStep  eeg:step01 ;
            eeg:hasPreprocessingStep  eeg:step02 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is True, f"Expected PASS but got violations:\n{report}"


class TestMissingChannels:
    """A session with fewer than 19 channels must fail."""

    def test_missing_channels_fails(self, tmp_path: Path) -> None:
        # Only 5 channels — well below the minimum of 19
        channels_decl  = _channel_block(5)
        channels_links = _session_channel_links(5)

        ttl = f"""
        eeg:patient02 a eeg:Patient ;
            eeg:subjectID  "chb02" ;
            eeg:hasSession eeg:session02 .

        eeg:sr256 a eeg:SamplingRate ;
            eeg:frequencyHz "256.0"^^xsd:float .

        {channels_decl}

        eeg:step01 a eeg:PreprocessingStep ;
            eeg:stepOrder  "1"^^xsd:integer ;
            eeg:filterType "notch" .

        eeg:session02 a eeg:RecordingSession ;
            eeg:sessionID            "chb02_01" ;
            eeg:hasSamplingRate      eeg:sr256 ;
            eeg:hasPreprocessingStep eeg:step01 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is False, "Expected FAIL (< 19 channels) but got PASS."
        assert "hasChannel" in report or "19" in report, (
            f"Report does not mention the channel constraint:\n{report}"
        )


class TestInvalidSamplingRate:
    """A SamplingRate of 100 Hz is below the allowed minimum of 200 Hz."""

    def test_invalid_sampling_rate_fails(self, tmp_path: Path) -> None:
        channels_decl  = _channel_block(19)
        channels_links = _session_channel_links(19)

        ttl = f"""
        eeg:patient03 a eeg:Patient ;
            eeg:subjectID  "chb03" ;
            eeg:hasSession eeg:session03 .

        # 100 Hz — below the allowed [200, 600] Hz range
        eeg:sr100 a eeg:SamplingRate ;
            eeg:frequencyHz "100.0"^^xsd:float .

        {channels_decl}

        eeg:step01 a eeg:PreprocessingStep ;
            eeg:stepOrder  "1"^^xsd:integer ;
            eeg:filterType "notch" .

        eeg:session03 a eeg:RecordingSession ;
            eeg:sessionID            "chb03_01" ;
            eeg:hasSamplingRate      eeg:sr100 ;
            eeg:hasPreprocessingStep eeg:step01 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is False, "Expected FAIL (100 Hz) but got PASS."
        assert "frequencyHz" in report or "200" in report or "600" in report, (
            f"Report does not mention the sampling-rate constraint:\n{report}"
        )


class TestSeizureAnnotation:
    """SeizureEvent onset must be strictly less than offset."""

    def test_onset_after_offset_fails(self, tmp_path: Path) -> None:
        channels_decl  = _channel_block(19)
        channels_links = _session_channel_links(19)

        ttl = f"""
        eeg:patient04 a eeg:Patient ;
            eeg:subjectID  "chb04" ;
            eeg:hasSession eeg:session04 .

        eeg:sr256 a eeg:SamplingRate ;
            eeg:frequencyHz "256.0"^^xsd:float .

        {channels_decl}

        eeg:step01 a eeg:PreprocessingStep ;
            eeg:stepOrder  "1"^^xsd:integer ;
            eeg:filterType "notch" .

        # Invalid: onset (200 s) is AFTER offset (100 s)
        eeg:sz04 a eeg:SeizureEvent ;
            eeg:hasOnset  "200.0"^^xsd:float ;
            eeg:hasOffset "100.0"^^xsd:float .

        eeg:session04 a eeg:RecordingSession ;
            eeg:sessionID            "chb04_01" ;
            eeg:hasSamplingRate      eeg:sr256 ;
            eeg:hasSeizureEvent      eeg:sz04 ;
            eeg:hasPreprocessingStep eeg:step01 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is False, "Expected FAIL (onset > offset) but got PASS."
        assert "onset" in report.lower() or "offset" in report.lower(), (
            f"Report does not mention the onset/offset constraint:\n{report}"
        )

    def test_onset_equal_offset_fails(self, tmp_path: Path) -> None:
        """onset == offset is also invalid (must be *strictly* less than)."""
        channels_decl  = _channel_block(19)
        channels_links = _session_channel_links(19)

        ttl = f"""
        eeg:patient05 a eeg:Patient ;
            eeg:subjectID  "chb05" ;
            eeg:hasSession eeg:session05 .

        eeg:sr256 a eeg:SamplingRate ;
            eeg:frequencyHz "256.0"^^xsd:float .

        {channels_decl}

        eeg:step01 a eeg:PreprocessingStep ;
            eeg:stepOrder  "1"^^xsd:integer ;
            eeg:filterType "notch" .

        # Invalid: onset == offset
        eeg:sz05 a eeg:SeizureEvent ;
            eeg:hasOnset  "100.0"^^xsd:float ;
            eeg:hasOffset "100.0"^^xsd:float .

        eeg:session05 a eeg:RecordingSession ;
            eeg:sessionID            "chb05_01" ;
            eeg:hasSamplingRate      eeg:sr256 ;
            eeg:hasSeizureEvent      eeg:sz05 ;
            eeg:hasPreprocessingStep eeg:step01 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is False, "Expected FAIL (onset == offset) but got PASS."


class TestPatientSubjectID:
    """Patient must have a non-empty subjectID."""

    def test_empty_subject_id_fails(self, tmp_path: Path) -> None:
        channels_decl  = _channel_block(19)
        channels_links = _session_channel_links(19)

        ttl = f"""
        # Empty subjectID — should fail
        eeg:patient06 a eeg:Patient ;
            eeg:subjectID  "" ;
            eeg:hasSession eeg:session06 .

        eeg:sr256 a eeg:SamplingRate ;
            eeg:frequencyHz "256.0"^^xsd:float .

        {channels_decl}

        eeg:step01 a eeg:PreprocessingStep ;
            eeg:stepOrder  "1"^^xsd:integer ;
            eeg:filterType "notch" .

        eeg:session06 a eeg:RecordingSession ;
            eeg:sessionID            "chb06_01" ;
            eeg:hasSamplingRate      eeg:sr256 ;
            eeg:hasPreprocessingStep eeg:step01 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is False, "Expected FAIL (empty subjectID) but got PASS."
        assert "subjectID" in report or "minLength" in report, (
            f"Report does not mention the subjectID constraint:\n{report}"
        )


class TestPreIctalWindow:
    """PreIctalWindow duration must be in [30, 120] seconds."""

    def test_short_preictal_window_fails(self, tmp_path: Path) -> None:
        channels_decl  = _channel_block(19)
        channels_links = _session_channel_links(19)

        ttl = f"""
        eeg:patient07 a eeg:Patient ;
            eeg:subjectID  "chb07" ;
            eeg:hasSession eeg:session07 .

        eeg:sr256 a eeg:SamplingRate ;
            eeg:frequencyHz "256.0"^^xsd:float .

        {channels_decl}

        eeg:step01 a eeg:PreprocessingStep ;
            eeg:stepOrder  "1"^^xsd:integer ;
            eeg:filterType "notch" .

        # 20 s — below the 30 s minimum
        eeg:piw07 a eeg:PreIctalWindow ;
            eeg:windowDuration "20.0"^^xsd:float ;
            eeg:windowStart    "80.0"^^xsd:float ;
            eeg:windowEnd      "100.0"^^xsd:float .

        eeg:sz07 a eeg:SeizureEvent ;
            eeg:hasOnset          "100.0"^^xsd:float ;
            eeg:hasOffset         "130.0"^^xsd:float ;
            eeg:hasPreIctalWindow eeg:piw07 .

        eeg:session07 a eeg:RecordingSession ;
            eeg:sessionID            "chb07_01" ;
            eeg:hasSamplingRate      eeg:sr256 ;
            eeg:hasSeizureEvent      eeg:sz07 ;
            eeg:hasPreprocessingStep eeg:step01 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is False, "Expected FAIL (20 s window < 30 s min) but got PASS."

    def test_long_preictal_window_fails(self, tmp_path: Path) -> None:
        channels_decl  = _channel_block(19)
        channels_links = _session_channel_links(19)

        ttl = f"""
        eeg:patient08 a eeg:Patient ;
            eeg:subjectID  "chb08" ;
            eeg:hasSession eeg:session08 .

        eeg:sr256 a eeg:SamplingRate ;
            eeg:frequencyHz "256.0"^^xsd:float .

        {channels_decl}

        eeg:step01 a eeg:PreprocessingStep ;
            eeg:stepOrder  "1"^^xsd:integer ;
            eeg:filterType "notch" .

        # 130 s — above the 120 s maximum
        eeg:piw08 a eeg:PreIctalWindow ;
            eeg:windowDuration "130.0"^^xsd:float ;
            eeg:windowStart    "0.0"^^xsd:float ;
            eeg:windowEnd      "130.0"^^xsd:float .

        eeg:sz08 a eeg:SeizureEvent ;
            eeg:hasOnset          "130.0"^^xsd:float ;
            eeg:hasOffset         "175.0"^^xsd:float ;
            eeg:hasPreIctalWindow eeg:piw08 .

        eeg:session08 a eeg:RecordingSession ;
            eeg:sessionID            "chb08_01" ;
            eeg:hasSamplingRate      eeg:sr256 ;
            eeg:hasSeizureEvent      eeg:sz08 ;
            eeg:hasPreprocessingStep eeg:step01 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is False, "Expected FAIL (130 s window > 120 s max) but got PASS."


class TestPreprocessingStep:
    """PreprocessingStep must declare a filterType."""

    def test_missing_filter_type_fails(self, tmp_path: Path) -> None:
        channels_decl  = _channel_block(19)
        channels_links = _session_channel_links(19)

        ttl = f"""
        eeg:patient09 a eeg:Patient ;
            eeg:subjectID  "chb09" ;
            eeg:hasSession eeg:session09 .

        eeg:sr256 a eeg:SamplingRate ;
            eeg:frequencyHz "256.0"^^xsd:float .

        {channels_decl}

        # No eeg:filterType declared — should fail
        eeg:step09 a eeg:PreprocessingStep ;
            eeg:stepOrder "1"^^xsd:integer .

        eeg:session09 a eeg:RecordingSession ;
            eeg:sessionID            "chb09_01" ;
            eeg:hasSamplingRate      eeg:sr256 ;
            eeg:hasPreprocessingStep eeg:step09 ;
            {channels_links} .
        """

        conforms, report = _validate(tmp_path, ttl)
        assert conforms is False, "Expected FAIL (missing filterType) but got PASS."
        assert "filterType" in report, (
            f"Report does not mention the filterType constraint:\n{report}"
        )
