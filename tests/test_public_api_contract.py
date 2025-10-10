"""Public API contract enforcement tests."""

import inspect

import document_segmenter as ds
import plan_sanitizer as ps


def test_plan_sanitizer_init_is_zero_arg() -> None:
    """PlanSanitizer.__init__ must remain an exact zero-argument signature."""

    assert str(inspect.signature(ps.PlanSanitizer.__init__)) == "(self)"


def test_document_segmenter_init_is_zero_arg() -> None:
    """DocumentSegmenter.__init__ must remain an exact zero-argument signature."""

    assert str(inspect.signature(ds.DocumentSegmenter.__init__)) == "(self)"
