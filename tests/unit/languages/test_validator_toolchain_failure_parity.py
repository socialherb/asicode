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
    (LanguageId.JAVASCRIPT, "x.js", "function a( {\n  return 1;\n}\n", "function a() {\n  return 1;\n}\n"),
    (LanguageId.TYPESCRIPT, "x.ts", "function a( {\n  return 1;\n}\n", "function a(): number {\n  return 1;\n}\n"),
    (
        LanguageId.GO,
        "x.go",
        "package main\nfunc a( {\n\treturn 1\n}\n",
        "package main\n\nfunc a() int {\n\treturn 1\n}\n",
    ),
    (LanguageId.JAVA, "X.java", "class A { void a( { } }\n", "class A { void a() { } }\n"),
    (LanguageId.KOTLIN, "X.kt", "fun a( { return 1 }\n", "fun a(): Int { return 1 }\n"),
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
    absent = [_validate_with_failing_subprocess(lang, path, src, FileNotFoundError("gone")) for src in (broken, valid)]
    failed = [
        _validate_with_failing_subprocess(lang, path, src, subprocess.TimeoutExpired("t", 10))
        for src in (broken, valid)
    ]
    assert absent == failed, f"{lang.value}: tool-absent {absent} diverges from tool-failed {failed}"


# ── the semantic half: a check that did not run must not answer "clean" ────
# validate_syntax always genuinely runs (tree-sitter is always there), so the
# tests above are about a WEAKER verdict. validate_semantics has the opposite
# failure: it depends on a toolchain the user may simply not have installed —
# pyright, tsc, go, javac, kotlinc — and every skip path answered ok=True with
# an empty error list, which downstream is exactly what a clean check produces.
# A `pip install asicode` with no node on the machine therefore had every Python
# edit reported to the model as semantically verified.


SEMANTIC_CASES = [
    (LanguageId.PYTHON, "x.py"),
    (LanguageId.TYPESCRIPT, "x.ts"),
    (LanguageId.GO, "x.go"),
    (LanguageId.JAVA, "X.java"),
    (LanguageId.KOTLIN, "X.kt"),
]

SEMANTIC_FAILURES = [
    pytest.param(FileNotFoundError("gone"), id="not-installed"),
    pytest.param(subprocess.TimeoutExpired("tool", 10), id="timeout"),
    pytest.param(OSError("boom"), id="oserror"),
]


@pytest.mark.parametrize("lang,name", SEMANTIC_CASES)
@pytest.mark.parametrize("exc", SEMANTIC_FAILURES)
def test_semantic_skip_is_not_reported_as_checked(lang, name, exc, tmp_path):
    """A toolchain that never ran must say so, not return a clean verdict."""
    cls, mod = _provider_and_module(lang)
    # python_provider imports subprocess INSIDE _run_pyright, so it has no
    # module-level attribute to patch — patch the stdlib module it will import,
    # which is the same object. Skipping instead (as the syntax helper above
    # does) would silently drop Python, the language most users run this on.
    holder = mod.subprocess if hasattr(mod, "subprocess") else subprocess
    # A real file, and the project markers each provider requires, so the run
    # reaches the subprocess call rather than skipping earlier for its own
    # reasons — those paths are unchecked too, but this test is about the
    # toolchain itself failing.
    src = tmp_path / name
    src.write_text("x = 1\n", encoding="utf-8")
    for marker in ("go.mod", "tsconfig.json", "pom.xml"):
        (tmp_path / marker).write_text("{}\n", encoding="utf-8")

    real = holder.run

    def boom(*a, **k):
        raise exc

    holder.run = boom
    try:
        result = cls().validate_semantics(str(src))
    finally:
        holder.run = real

    assert result.checked is False, (
        f"{lang.value}: a semantic checker that never ran reported checked=True, "
        "which is indistinguishable from a clean verdict"
    )
    assert result.skip_reason, f"{lang.value}: skip carries no reason for the model"
    assert result.ok is True, f"{lang.value}: an unavailable semantic checker must stay non-blocking"
    assert result.errors == []


def test_a_real_clean_semantic_verdict_is_still_marked_checked(tmp_path):
    """The opposite direction: a tool that RAN and found nothing is `checked`.

    Reusing the skip constructor for a successful run would make every healthy
    check look unavailable — the same conflation, mirrored.
    """
    pytest.importorskip("shutil")
    import shutil

    if shutil.which("pyright") is None:
        pytest.skip("pyright not installed")
    src = tmp_path / "clean.py"
    src.write_text("x: int = 1\n", encoding="utf-8")

    cls, _ = _provider_and_module(LanguageId.PYTHON)
    result = cls().validate_semantics(str(src))

    assert result.checked is True, "a completed pyright run reported as unchecked"
    assert result.skip_reason == ""
    assert result.ok is True
