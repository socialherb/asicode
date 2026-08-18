"""Gate: the coupled-test exclusion pattern must match PATH JOINS, not prose.

0.2.24 release incident: the quoted-token alternatives of _COUPLED_TEST_PAT
matched any quoted "tools"/"webapp" followed by a slash — including comment
prose like: omit "tools"/"tool_choice" keys — written in two provider
regression tests. Both files were silently dropped from the public snapshot
(caught only as unexpected deletions in the pre-push release-delta review),
so the P1/P3 fixes would have shipped without their regression tests.

The corrected rule requires a quoted path join that ends in a FILENAME
(dot + extension) at any depth of quoted directory components — the shapes
that actually load excluded files at runtime. Probes below are built with
chr(34) so this file's own source cannot self-exclude via the pattern it
tests (a literal winning probe here would drop this gate from the snapshot).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "export_public.py"

_Q = chr(34)  # double quote, kept out of string literals on purpose


def _load():
    """Load export_public.py as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("_export_public_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prose_quoted_words_are_not_coupled():
    ep = _load()
    prose = f"omit {_Q}tools{_Q}/{_Q}tool_choice{_Q} keys entirely"
    assert ep._COUPLED_TEST_PAT.search(prose) is None


def test_path_join_to_py_file_is_coupled():
    ep = _load()
    join = f"REPO / {_Q}tools{_Q} / {_Q}ast_diff_verifier.py{_Q}"
    assert ep._COUPLED_TEST_PAT.search(join)


def test_path_join_to_non_py_files_is_coupled():
    """webapp reads are .js/.html just as often as .py — all count."""
    ep = _load()
    join_html = f"_R / {_Q}webapp{_Q} / {_Q}ui{_Q} / {_Q}templates{_Q} / {_Q}ui.html{_Q}"
    assert ep._COUPLED_TEST_PAT.search(join_html)
    join_js = f"REPO / {_Q}webapp{_Q} / {_Q}ui{_Q} / {_Q}static{_Q} / {_Q}agent-panel.js{_Q}"
    assert ep._COUPLED_TEST_PAT.search(join_js)


def test_contiguous_path_string_is_coupled():
    ep = _load()
    contig = "tools" + "/" + "x.py"
    assert ep._COUPLED_TEST_PAT.search(contig)


def test_real_repo_files_ship_and_exclude_correctly():
    """End-to-end on the actual tree — non-vacuous in BOTH trees.

    The prose-incident files exist in the private tree and the public
    snapshot alike, so the not-excluded assertion runs everywhere. The
    webapp-reading gates exist only privately; in the public snapshot their
    absence is itself the exclusion having been applied, so each tree
    asserts the invariant it can actually observe.
    """
    ep = _load()
    for rel in ("tests/unit/test_client_red_green_openai.py",
                "tests/unit/test_providers_red_green_deepseek.py"):
        assert ep.is_excluded(rel) is None
    for rel in ("tests/unit/test_sse_emit_consume_gate.py",
                "tests/unit/test_version_drift_gate.py"):
        if (REPO / rel).exists():  # private tree: excluded by the pattern
            assert ep.is_excluded(rel) == "coupled-test"
        else:  # public snapshot: the exclusion already removed the file
            assert not (REPO / rel).exists()
