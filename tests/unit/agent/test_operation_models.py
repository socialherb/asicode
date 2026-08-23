"""
Tests for operation_models (live subset).

The planner-lane models (Operation / OperationPlan / ExecutorState / ...)
were removed as dead code (P-PCLa) — only the production-consumed enums
remain, and only their tests survive here.
"""

from external_llm.agent.operation_models import FailureClass, normalize_failure_class


def test_normalize_failure_class_none():

    assert normalize_failure_class(None) == FailureClass.UNKNOWN


def test_normalize_failure_class_empty():

    assert normalize_failure_class("") == FailureClass.UNKNOWN


def test_normalize_failure_class_exact():

    assert normalize_failure_class("syntax_error") == FailureClass.SYNTAX_ERROR


def test_normalize_failure_class_unknown():

    assert normalize_failure_class("bogus") == FailureClass.UNKNOWN
