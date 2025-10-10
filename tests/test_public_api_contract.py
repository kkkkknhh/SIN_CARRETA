"""Public API contract enforcement tests."""

import inspect

import causal_pattern_detector as cpd
import dag_validation as dv
import document_segmenter as ds


def test_causal_detector_init_signature_exact() -> None:
    sig = str(inspect.signature(cpd.CausalPatternDetector.__init__))
    assert sig == "(self)", f"Extra params detected: {sig}"


def test_dag_validator_init_signature_exact() -> None:
    sig = str(inspect.signature(dv.DAGValidator.__init__))
    assert sig == "(self)", f"Constructor must be empty: {sig}"


def test_document_segmenter_segment_signature() -> None:
    sig = str(inspect.signature(ds.DocumentSegmenter.segment))
    assert sig.startswith("(self, text"), f"segment() must take `text` first: {sig}"
