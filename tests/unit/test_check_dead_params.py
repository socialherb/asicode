"""Tests for the net-new dead-parameter baseline-diff gate.

The live invariant (no new unused parameters beyond baseline) is inherently
vacuously-passable if the scanner is weakened, so the precision parametrize
tests (dangerous vs safe shapes) are what actually guard detection
capability — same pattern as test_check_discarded_signal.py.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_dead_params.py"
_spec = importlib.util.spec_from_file_location("check_dead_params", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


# ── precision: unused params ARE flagged ────────────────────────────────────
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # plain unused params (both x and y unused)
        ("    a = 1\n    b = 2\n    c = 3\n    return a + b + c", ["m.py::f::x", "m.py::f::y"]),
        # unused param next to a used one
        ("    a = 1\n    b = 2\n    c = 3\n    return x + a + b + c", ["m.py::f::y"]),
        # param used only in a nested closure (closure use counts)
        ("    a = 1\n    b = 2\n    def inner():\n        return x\n    return inner", ["m.py::f::y"]),
        # shadowed nested param: inner(x) shadows — outer x genuinely unused
        ("    a = 1\n    b = 2\n    def inner(x):\n        return x\n    return inner", ["m.py::f::x", "m.py::f::y"]),
    ],
    ids=["plain", "mixed", "closure-use", "shadowed"],
)
def test_scan_flags_unused_params(body, expected):
    src = "def f(x, *, y):\n" + body + "\n"
    keys = g._scan_source(src, "m.py")
    assert keys == expected, keys


def test_scan_flags_method_of_base_less_class():
    src = "class C:\n    def m(self, x):\n        a = 1\n        b = 2\n        c = 3\n        return a + b + c\n"
    assert g._scan_source(src, "m.py") == ["m.py::C.m::x"]


def test_async_function_flagged():
    src = "async def f(x):\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n"
    assert g._scan_source(src, "m.py") == ["m.py::f::x"]


# ── precision: safe shapes are NOT flagged (0 keys) ─────────────────────────
@pytest.mark.parametrize(
    ("src", "why"),
    [
        # used param
        ("def f(x):\n    a = 1\n    b = 2\n    c = 3\n    return x + a + b + c\n", "used"),
        # underscore prefix = documented "ignored" convention
        ("def f(_x):\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n", "underscore"),
        # vararg functions are open-ended interfaces
        ("def f(x, **kwargs):\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n", "vararg"),
        # decorated functions (routes/callbacks) may be signature-inspected
        ("@router.get('/x')\ndef f(x):\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n", "decorated"),
        # classes with bases are override surfaces
        (
            "class C(Base):\n    def m(self, x):\n        a = 1\n        b = 2\n        c = 3\n        return a + b + c\n",
            "class-with-base",
        ),
        # tiny wrapper bodies (< 4 lines) are callback shims
        ("def f(x):\n    return 1\n", "tiny-body"),
        # self is conventionally unlisted
        ("class C:\n    def m(self):\n        a = 1\n        b = 2\n        c = 3\n        return a + b + c\n", "self"),
    ],
    ids=["used", "underscore", "vararg", "decorated", "class-base", "tiny", "self"],
)
def test_scan_ignores_safe_shapes(src, why):
    keys = g._scan_source(src, "m.py")
    assert keys == [], (keys, why)


def test_key_format_module_function():
    src = "def f(x):\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n"
    assert g._scan_source(src, "m.py") == ["m.py::f::x"]


def test_key_format_class_method():
    src = "class C:\n    def m(self, x):\n        a = 1\n        b = 2\n        c = 3\n        return a + b + c\n"
    assert g._scan_source(src, "m.py") == ["m.py::C.m::x"]


def test_nested_function_own_unused_param_found():
    # outer f's x IS used (via closure) — the INNER g's own unused param must
    # be found separately, and the outer scan must not double-report.
    src = (
        "def f(x):\n"
        "    a = 1\n"
        "    b = 2\n"
        "    def g(y):\n"
        "        c = 1\n"
        "        d = 2\n"
        "        e = 3\n"
        "        return x + c + d + e\n"
        "    return g\n"
    )
    keys = g._scan_source(src, "m.py")
    assert keys == ["m.py::g::y"], keys


def test_syntax_error_returns_empty():
    assert g._scan_source("def f(:\n", "m.py") == []
