"""Regression tests for duplicate_definition_scanner + dead-block test-file
detection — covers the Go false-positive fixes:
  1. Go method-receiver disambiguation (interface impls no longer collide).
  2. Language-agnostic test-file conventions (_is_dynamic_invocation_file).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from external_llm.analysis._dead_block_shared import _is_dynamic_invocation_file
from external_llm.analysis.duplicate_definition_scanner import (
    _HAS_TS,
    scan_duplicate_definitions,
)

# ── fixtures ───────────────────────────────────────────────────────────────

GO_INTERFACE_IMPLS = """\
package main

type Model interface {
    Render() string
}

type A struct{}
func (a *A) Render() string { return "a" }
func (a *A) Len() int { return 1 }

type B struct{}
func (b *B) Render() string { return "b" }
func (b *B) Len() int { return 2 }

type C struct{}
func (c *C) Render() string { return "c" }
"""

GO_REAL_DUP = """\
package main
type A struct{}
func (a *A) Render() string { return "a" }
func (a *A) Render() string { return "dup" }
"""

PY_REAL_DUP = """\
def foo():
    pass

def foo():
    pass
"""

PY_NO_DUP = """\
def helper():
    pass

def other():
    pass
"""


def _scan(content: str, fname: str):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, fname)
        with open(p, "w") as f:
            f.write(content)
        return scan_duplicate_definitions(repo_root=d, file_paths=[p])


# ── duplicate_definition_scanner: Go receiver disambiguation ───────────────

@pytest.mark.skipif(not _HAS_TS, reason="tree-sitter not installed")
class TestGoReceiverDisambiguation:
    """``func (a *A) Render`` and ``func (b *B) Render`` are distinct symbols —
    they must NOT be flagged as duplicate definitions (was 50 false positives
    before the receiver was added to the dedup key)."""

    def test_interface_impls_not_flagged(self):
        cands = _scan(GO_INTERFACE_IMPLS, "impls.go")
        names = sorted(c.name for c in cands)
        assert names == [], f"interface impl methods falsely flagged: {names}"

    def test_real_duplicate_same_receiver_caught(self):
        cands = _scan(GO_REAL_DUP, "dup.go")
        assert len(cands) == 1
        assert cands[0].name == "Render"
        assert len(cands[0].occurrences) == 2

    def test_value_and_pointer_receiver_same_type_normalized(self):
        # ``(a *A)`` and ``(b A)`` target the same type → same dedup key, so a
        # genuine cross-receiver duplicate collapses into one group.
        src = (
            "package main\n"
            "type A struct{}\n"
            "func (a *A) Render() string { return \"ptr\" }\n"
            "func (b A) Render() string { return \"val\" }\n"
        )
        cands = _scan(src, "amb.go")
        assert len(cands) == 1
        assert cands[0].name == "Render"


# ── duplicate_definition_scanner: Python regression (no receiver) ──────────

class TestPythonDuplicateRegression:
    """Python path (receiver always None) must keep working after the
    tuple-shape change from 4- to 5-tuple."""

    def test_python_real_duplicate_caught(self):
        cands = _scan(PY_REAL_DUP, "mod.py")
        assert len(cands) == 1
        assert cands[0].name == "foo"
        assert len(cands[0].occurrences) == 2

    def test_python_no_duplicate(self):
        cands = _scan(PY_NO_DUP, "mod.py")
        assert cands == []


# ── _is_dynamic_invocation_file: language-agnostic test-file detection ─────

@pytest.mark.parametrize("path, expected", [
    # Python
    ("tests/test_foo.py", True),
    ("pkg/test_bar.py", True),
    ("pkg/bar_test.py", True),
    ("conftest.py", True),
    ("pkg/utils.py", False),
    # Go (the core fix — was False before, flooding 642 public "dead")
    ("cmd/main_test.go", True),
    ("cmd/main.go", False),
    # Java (JUnit)
    ("src/FooTest.java", True),
    ("src/FooTests.java", True),
    ("src/TestFoo.java", True),
    ("src/Foo.java", False),
    # Kotlin
    ("src/FooTest.kt", True),
    ("src/Foo.kt", False),
    # TS/JS (Jest/Mocha/Vitest)
    ("src/foo.test.ts", True),
    ("src/foo.spec.ts", True),
    ("src/foo.test.js", True),
    ("src/foo.spec.jsx", True),
    ("src/foo.ts", False),
    # Rust
    ("src/foo_test.rs", True),
    ("src/main.rs", False),
    # directory-based (already language-agnostic)
    ("__tests__/helper.go", True),
    ("tests/sub/main.go", True),
])
def test_is_dynamic_invocation_file_multilang(path, expected):
    assert _is_dynamic_invocation_file(path) is expected
