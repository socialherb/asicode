"""A syntax validator must not get *weaker* when its toolchain misbehaves.

``_validate_syntax_impl`` runs an external checker (node/tsc/go/javac/kotlinc)
and handles three failure shapes: the binary is missing, it times out, or it
raises something else. Every provider already fell back to tree-sitter for
``FileNotFoundError`` — but five of them answered ``ok=True`` for the other
two, so a checker that CRASHED declared broken code valid while a checker that
was simply ABSENT caught it.

That inversion matters because this validator is the post-write rollback gate:
``ok=True`` on corrupted output means no rollback. And a timeout is most likely
exactly when the machine is already struggling.

These tests assert the property (absence == failure, and neither invents a
syntax error for valid code) over the LIVE provider registry rather than a
hardcoded list, so a newly added provider that reintroduces the pattern fails
here instead of shipping.
"""
from __future__ import annotations

import subprocess

import pytest

from external_llm.languages import LanguageRegistry
from external_llm.languages.base import LanguageId

# (LanguageId, filename, syntactically-broken source, valid source)
CASES = [
    (LanguageId.JAVASCRIPT, "x.js",
     "function a( {\n  return 1;\n}\n",
     "function a() {\n  return 1;\n}\n"),
    (LanguageId.TYPESCRIPT, "x.ts",
     "function a( {\n  return 1;\n}\n",
     "function a(): number {\n  return 1;\n}\n"),
    (LanguageId.GO, "x.go",
     "package main\nfunc a( {\n\treturn 1\n}\n",
     "package main\n\nfunc a() int {\n\treturn 1\n}\n"),
    (LanguageId.JAVA, "X.java",
     "class A { void a( { } }\n",
     "class A { void a() { } }\n"),
    (LanguageId.KOTLIN, "X.kt",
     "fun a( { return 1 }\n",
     "fun a(): Int { return 1 }\n"),
]

FAILURES = [
    pytest.param(subprocess.TimeoutExpired("tool", 10), id="timeout"),
    pytest.param(OSError("boom"), id="oserror"),
]


def _provider_and_module(lang: LanguageId):
    """Return (provider_class, defining_module) for *lang* from the registry.

    Fresh instances matter: ``SyntaxProvider.validate_syntax`` memoises per
    instance keyed by (path, content-hash), so reusing one would serve a verdict
    computed BEFORE the subprocess was patched and the test would pass
    vacuously. (Found the hard way — the first version of this check reported
    every provider healthy for exactly that reason.)
    """
    import sys

    providers = {p.language_id(): p for p in set(LanguageRegistry.instance()._providers.values())}
    prov = providers.get(lang)
    if prov is None:
        pytest.skip(f"no provider registered for {lang}")
    cls = type(prov)
    return cls, sys.modules[cls.__module__]


def _validate_with_failing_subprocess(lang, file_path, content, exc):
    cls, mod = _provider_and_module(lang)
    if not hasattr(mod, "subprocess"):
        pytest.skip(f"{mod.__name__} does not shell out")
    real = mod.subprocess.run

    def boom(*a, **k):
        raise exc

    mod.subprocess.run = boom
    try:
        return cls().validate_syntax(file_path, content).ok  # fresh -> no memo
    finally:
        mod.subprocess.run = real


@pytest.mark.parametrize("lang,path,broken,_valid", [(c[0], c[1], c[2], c[3]) for c in CASES])
@pytest.mark.parametrize("exc", FAILURES)
def test_toolchain_failure_still_rejects_broken_source(lang, path, broken, _valid, exc):
    """Regression: these returned ok=True, so a crashed checker skipped rollback."""
    assert _validate_with_failing_subprocess(lang, path, broken, exc) is False, (
        f"{lang.value}: broken source declared valid when the checker failed"
    )


@pytest.mark.parametrize("lang,path,_broken,valid", [(c[0], c[1], c[2], c[3]) for c in CASES])
@pytest.mark.parametrize("exc", FAILURES)
def test_toolchain_failure_does_not_reject_valid_source(lang, path, _broken, valid, exc):
    """The fallback must not invent syntax errors — that would block real edits."""
    assert _validate_with_failing_subprocess(lang, path, valid, exc) is True, (
        f"{lang.value}: valid source rejected when the checker failed"
    )


@pytest.mark.parametrize("lang,path,broken,valid", CASES)
def test_absence_and_failure_agree(lang, path, broken, valid):
    """Absence was always handled correctly; failure must match it exactly."""
    absent = [
        _validate_with_failing_subprocess(lang, path, src, FileNotFoundError("gone"))
        for src in (broken, valid)
    ]
    failed = [
        _validate_with_failing_subprocess(lang, path, src, subprocess.TimeoutExpired("t", 10))
        for src in (broken, valid)
    ]
    assert absent == failed, (
        f"{lang.value}: tool-absent {absent} diverges from tool-failed {failed}"
    )
