"""Tests for external_llm/analysis/vulture_scanner.py.

Focus: the ``exclude_kinds`` overlap filter (Option B). vulture's per-file view
of module-level functions/classes is redundant with ``public_dead_code_scanner``
(which resolves cross-file references), so those kinds are excluded by default.
vulture's UNIQUE value — class-level ``method``/``variable``/``property`` — must
survive the default filter. (vulture distinguishes ``function`` from ``method``.)
"""

from __future__ import annotations

import os

import pytest

vulture = pytest.importorskip("vulture.core")  # skip entire module if optional dep missing

from external_llm.analysis.vulture_scanner import (
    _PUBLIC_DEAD_CODE_OVERLAP_KINDS,
    _VULTURE_CODE_BY_KIND,
    _VULTURE_KIND_MAP,
    scan_vulture_dead_code,
)

# A source exercising every vulture ``typ`` that maps to a non-excluded kind.
_SAMPLE = """\
def module_level_func():
    pass


class SomeClass:
    def method_inside_class(self):
        pass

    class_var = 123

    @property
    def some_prop(self):
        return 1
"""


def _scan(tmp_path, **kwargs):
    """Write _SAMPLE into tmp_path and run the scanner with min_confidence=0."""
    repo_root = str(tmp_path)
    (tmp_path / "probe.py").write_text(_SAMPLE)
    return scan_vulture_dead_code(
        repo_root=repo_root,
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
        **kwargs,
    )


# ── Constants / invariants ──────────────────────────────────────────────


def test_overlap_constant_is_exactly_function_and_class():
    assert frozenset({"function", "class"}) == _PUBLIC_DEAD_CODE_OVERLAP_KINDS


def test_kind_map_distinguishes_function_from_method():
    """Critical: vulture's ``method`` typ must map distinctly from ``function``.
        If they collapsed to one kind, the overlap filter would wrongly drop
    class-level methods (which public_dead_code_scanner does NOT cover)."""
    assert _VULTURE_KIND_MAP["function"] == "function"
    assert _VULTURE_KIND_MAP["method"] == "method"
    assert "method" not in _PUBLIC_DEAD_CODE_OVERLAP_KINDS


def test_vulture_code_by_kind_matches_cli_codes():
    """The scanner's noqa contract must mirror vulture CLI codes (core.py:32-38)
    so a "# noqa: V1xx" comment suppresses the same candidate in both paths."""
    assert _VULTURE_CODE_BY_KIND == {
        "attribute": "V101",
        "class": "V102",
        "function": "V103",
        "import": "V104",
        "method": "V105",
        "property": "V106",
        "variable": "V107",
    }


# ── V-code noqa suppression (flake8-style contract, implemented in-scanner) ─


def test_v_code_noqa_suppresses_matching_candidate(tmp_path):
    """A "# noqa: V105" on an unused method suppresses it — vulture's own
    get_unused_code() applies no noqa filtering (that lives in its CLI), so the
    scanner must honor the flake8-style code itself."""
    src = """\
class C:
    def dead_method(self):  # noqa: V105 — documented intentional
        pass
"""
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "dead_method" not in names


def test_v_code_noqa_does_not_suppress_other_kind(tmp_path):
    """Code matching is kind-precise: a V101 (attribute) noqa must NOT suppress
    an unused method (V105)."""
    src = """\
class C:
    def dead_method(self):  # noqa: V101 — wrong kind, must not match
        pass
"""
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "dead_method" in names


# ── Default behavior: overlap excluded, unique kinds kept ────────────────


def test_default_excludes_module_level_function_and_class(tmp_path):
    cands = _scan(tmp_path)
    kinds = {c.kind for c in cands}
    assert "function" not in kinds, "module-level fn must defer to public_dead_code_scanner"
    assert "class" not in kinds, "module-level class must defer to public_dead_code_scanner"


def test_method_kind_survives_default_filter(tmp_path):
    """The pivotal regression guard: class methods are vulture-only signal and
    MUST survive the default overlap filter."""
    cands = _scan(tmp_path)
    names = {c.name for c in cands}
    assert "method_inside_class" in names
    assert any(c.kind == "method" for c in cands)


def test_default_keeps_unique_kinds(tmp_path):
    cands = _scan(tmp_path)
    kinds = {c.kind for c in cands}
    # method + property + variable are all class-level/private-scope -> kept
    assert {"method", "property", "variable"} <= kinds


# ── Override semantics ──────────────────────────────────────────────────


def test_exclude_kinds_empty_keeps_everything(tmp_path):
    cands = _scan(tmp_path, exclude_kinds=())
    kinds = {c.kind for c in cands}
    assert {"function", "class", "method", "property", "variable"} <= kinds


def test_exclude_kinds_custom_replaces_default(tmp_path):
    """Passing exclude_kinds replaces the default set (does not augment it)."""
    cands = _scan(tmp_path, exclude_kinds={"method"})
    kinds = {c.kind for c in cands}
    assert "method" not in kinds
    # function/class are NOT in the custom set -> they reappear (default overridden)
    assert "function" in kinds
    assert "class" in kinds


# ── Always-live dunder still filtered (regression for reorder) ───────────


def test_always_live_dunder_still_filtered(tmp_path):
    """The kind-filter reorder must not break the _ALWAYS_LIVE name filter."""
    src = """\
class C:
    def __init__(self):
        pass
"""
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "__init__" not in names


# ── Non-dunder framework protocols are always-live ──────────────────────


def test_always_live_includes_framework_protocols():
    """Non-dunder protocol methods invoked by a framework with no static caller
    (Enum._missing_, HTMLParser.handle_*) must be filtered like dunders."""
    from external_llm.analysis.vulture_scanner import _ALWAYS_LIVE

    assert "_missing_" in _ALWAYS_LIVE
    assert "handle_starttag" in _ALWAYS_LIVE


# ── Test files: parsed for reachability, candidates suppressed ───────────


def test_is_test_path_classifies_correctly():
    from external_llm.analysis.vulture_scanner import _is_test_path

    # test files
    assert _is_test_path("tests/unit/test_foo.py")
    assert _is_test_path("tests/conftest.py")
    assert _is_test_path("pkg/testing/test_bar.py")
    assert _is_test_path("something_test.py")
    # production files (must NOT be suppressed)
    assert not _is_test_path("external_llm/testing/symbol_aware_test_finder.py")
    assert not _is_test_path("external_llm/agent/foo.py")


def test_test_file_candidates_suppressed(tmp_path):
    """Test files are parsed for cross-file reachability but their own
    candidates (fixtures/parametrize) are dropped; a production sibling's
    candidates survive."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_noise.py").write_text("unused_fixture_in_test = 123\n")
    (tmp_path / "probe.py").write_text("class C:\n    def dead_method(self):\n        pass\n")
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py", "tests/test_noise.py"],
        repo_graph=None,
        min_confidence=0,
    )
    files = {c.file for c in cands}
    assert not any("test_noise" in f for f in files)
    assert any("probe.py" in f for f in files)


# ── String-dispatched callables are suppressed (handler maps / getattr) ──


def test_string_dispatch_live_method_suppressed(tmp_path):
    """A method referenced as a quoted identifier (handler-map value later
    resolved via getattr) is plausibly dispatched → suppressed. A truly
    unreferenced sibling method must STILL be reported (no over-suppression)."""
    src = (
        "class Tools:\n"
        '    HANDLER_MAP = {"a": "_dispatched_handler"}\n\n'
        "    def _dispatched_handler(self):\n"
        "        pass\n\n"
        "    def _truly_dead(self):\n"
        "        pass\n"
    )
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "_dispatched_handler" not in names
    assert "_truly_dead" in names


def test_string_dispatch_cross_file_suppressed_in_file_paths_scope(tmp_path):
    """Handler-map strings in a registry file OUTSIDE the scanned targets must
    still suppress the dispatched methods. Regression: in ``file_paths_only``
    scope the dispatch-literal pass was limited to the target files, so every
    dynamically dispatched method in a leaf-scanned file came back as a false
    positive (e.g. tool_registry.py's ``{"search_web": "_tool_search_web"}``
    referencing a method in web_search_tools.py)."""
    (tmp_path / "registry.py").write_text('HANDLER_MAP = {"web": "_dispatched_handler"}\n')
    (tmp_path / "probe.py").write_text(
        "class Tools:\n    def _dispatched_handler(self):\n        pass\n\n    def _truly_dead(self):\n        pass\n"
    )

    class _LeafGraph:  # get_importers → [] ⇒ file_paths_only scope
        @staticmethod
        def get_importers(file_path):
            return []

    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=_LeafGraph(),
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "_dispatched_handler" not in names
    assert "_truly_dead" in names


def test_cross_file_usage_suppressed_in_file_paths_scope(tmp_path):
    """Names used from OTHER modules stay live in leaf scope. Regression:
    vulture's unused check uses a GLOBAL used-names basis that only fills in
    during full-project scans, so ``file_paths_only`` mode falsely reported
    (a) methods called from another module (e.g. patch_mixin →
    ``_patch_failure_snippet``) and (b) TYPE_CHECKING imports consumed only
    via string annotations (e.g. ``-> "ToolResult"``). The scanner seeds a
    live-name set from repo-graph caller edges to reproduce the full-project
    basis; a truly dead sibling must STILL be reported."""
    import types

    (tmp_path / "probe.py").write_text(
        "from __future__ import annotations\n\n"
        "if TYPE_CHECKING:\n"
        "    from registry import AnnotOnly\n\n"
        "class Worker:\n"
        "    def _cross_file_method(self):\n"
        "        pass\n\n"
        "    def _truly_dead(self):\n"
        "        pass\n\n"
        'def _make() -> "AnnotOnly":\n'
        "    ...\n"
    )

    class _GraphStub:  # models RepositoryGraph: leaf scope + caller edges
        @staticmethod
        def get_importers(file_path):
            return []  # leaf ⇒ file_paths_only scope

        @staticmethod
        def get_symbols_in_file(rel):
            if rel == "probe.py":
                return [
                    types.SimpleNamespace(name="Worker"),
                    types.SimpleNamespace(name="_cross_file_method"),
                    types.SimpleNamespace(name="_truly_dead"),
                ]
            return []

        @staticmethod
        def get_callers(name):
            # _cross_file_method is invoked from caller.py; AnnotOnly is
            # constructed elsewhere (its import here is annotation-only).
            return [object()] if name in {"_cross_file_method", "AnnotOnly"} else []

    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=_GraphStub(),
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "_cross_file_method" not in names
    assert "AnnotOnly" not in names
    assert "_truly_dead" in names


def test_cross_file_value_ref_suppressed_in_file_paths_scope(tmp_path):
    """Module-level variables consumed by VALUE from another file stay live in
    leaf scope. Regression: the structural-scan gate (pre-commit per-file
    mode) reports ``SCAN_LANGUAGES`` in scan_walk.py as dead — it is imported
    by scanner_registry.py as the identity alias ``_TS_LANGUAGES =
    SCAN_LANGUAGES``, a value reference with NO call edge, so the caller-edge
    discovery in ``_caller_live`` cannot see it and vulture's per-file scan
    neither. The injected ``cross_file_referenced_names`` (whole-repo set, the
    same contract public_dead_code_scanner uses) must suppress it; a name not
    in that set must STILL be reported."""
    (tmp_path / "probe.py").write_text("SCAN_LANGUAGES = frozenset({'python'})\n\nTRULY_DEAD = frozenset({'gone'})\n")

    class _LeafGraph:  # get_importers → [] ⇒ file_paths_only scope
        @staticmethod
        def get_importers(file_path):
            return []

    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=_LeafGraph(),
        min_confidence=0,
        cross_file_referenced_names={"SCAN_LANGUAGES"},
    )
    names = {c.name for c in cands}
    assert "SCAN_LANGUAGES" not in names
    assert "TRULY_DEAD" in names


def test_cross_file_value_ref_ignored_in_full_project_scope(tmp_path):
    """In full_project scope vulture scans every project file itself, so the
    injected cross-file set must NOT suppress anything — a name in the set but
    unused in the scanned corpus is genuinely dead (no double-suppression)."""
    (tmp_path / "probe.py").write_text("SCAN_LANGUAGES = frozenset({'python'})\n\nOTHER_DEAD = frozenset({'gone'})\n")
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,  # no graph ⇒ full_project scope
        min_confidence=0,
        cross_file_referenced_names={"SCAN_LANGUAGES", "OTHER_DEAD"},
    )
    names = {c.name for c in cands}
    assert "SCAN_LANGUAGES" in names
    assert "OTHER_DEAD" in names


# ── visitor-protocol suppression (libcst/ast dispatch hooks) ────────────────
# libcst (CSTVisitor/CSTTransformer) and ast (NodeVisitor/NodeTransformer)
# dispatch per-node-type hooks (visit_<Node>, leave_<Node>) and lifecycle
# methods (on_visit/on_leave/generic_visit) via getattr — no static caller, so
# vulture reports them as dead. Detection is STRUCTURAL (the enclosing class
# inherits from a known visitor base), not name-based, so a coincidentally
# named method in a non-visitor class is still reported.


def test_visitor_protocol_methods_suppressed(tmp_path):
    """Framework-dispatched visitor hooks in a visitor subclass are suppressed;
    a truly dead sibling method must STILL be reported (no over-suppression)."""
    src = (
        "class _Probe(CSTTransformer):\n"
        "    def visit_FunctionDef(self, node):\n"
        "        x = 1\n\n"
        "    def leave_ClassDef(self, node):\n"
        "        x = 1\n\n"
        "    def on_visit(self, node):\n"
        "        x = 1\n\n"
        "    def generic_visit(self, node):\n"
        "        x = 1\n\n"
        "    def _truly_dead(self):\n"
        "        x = 1\n"
    )
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "visit_FunctionDef" not in names
    assert "leave_ClassDef" not in names
    assert "on_visit" not in names
    assert "generic_visit" not in names
    assert "_truly_dead" in names


def test_visitor_hook_not_suppressed_in_non_visitor_class(tmp_path):
    """A method named visit_* in a NON-visitor class is real dead code — the
    structural base-class check must NOT suppress it (over-suppression guard
    against a naive name-prefix rule)."""
    src = (
        "class HttpClient:\n"  # NOT a visitor subclass
        "    def visit_url(self, url):\n"  # coincidentally named, truly dead
        "        x = 1\n"
    )
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "visit_url" in names


def test_visitor_subclass_via_same_file_ancestor(tmp_path):
    """A class that inherits a visitor base through a same-file ancestor (not
    directly) is still recognized — the transitive base-name resolution fires."""
    src = (
        "class _Base(NodeVisitor):\n"
        "    pass\n\n"
        "class _Derived(_Base):\n"
        "    def visit_Module(self, node):\n"
        "        x = 1\n\n"
        "    def _truly_dead(self):\n"
        "        x = 1\n"
    )
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "visit_Module" not in names
    assert "_truly_dead" in names


# ── full_project must not parse vendored dirs (.venv/node_modules) ─────────
# Regression for the fix replacing ``scan_paths=[repo_root]`` with an explicit
# project file list. ``vulture.scavenge([repo_root])`` walks the tree with
# vulture's own (looser) exclude rules and parsed .venv/node_modules — 16658
# files vs 956 here, ~91% of run_structural_scan wall time, plus ~20k vendored
# false positives. With repo_graph=None the scope decision returns
# "full_project", so this is the path the fix targets.


def test_full_project_skips_vendored_dirs(tmp_path):
    """full_project mode must enumerate the project .py set explicitly, never
    walking into .venv. We place a dead-code file under .venv AND list it in
    file_paths: if vulture scanned it it would be reported; the skip keeps it
    absent from results."""
    (tmp_path / "real.py").write_text("class C:\n    def unused_method(self):\n        pass\n")
    vendored_dir = tmp_path / ".venv" / "site-packages" / "somepkg"
    vendored_dir.mkdir(parents=True)
    (vendored_dir / "vendored.py").write_text("def totally_dead_vendored():\n    pass\n")
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["real.py", ".venv/site-packages/somepkg/vendored.py"],
        repo_graph=None,
        min_confidence=0,
    )
    files = {c.file for c in cands}
    assert not any(".venv" in f for f in files), f"vendored file was scanned: {files}"
    assert any(f == "real.py" for f in files), "project file not scanned"


def test_full_project_scans_entire_project_when_no_targets(tmp_path):
    """With no file_paths targets, full_project must walk the whole project
    (via _collect_project_py_files) and report dead code from any project file
    — not scan nothing. This locks in that the fix enumerates the project set
    instead of relying on the caller-supplied file_paths."""
    (tmp_path / "dead.py").write_text("class C:\n    def unused_method(self):\n        pass\n")
    (tmp_path / "alive.py").write_text("x = 1\nprint(x)\n")
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=[],  # no targets → full_project walks everything
        repo_graph=None,  # → full_project
        min_confidence=0,
    )
    files = {c.file for c in cands}
    # dead.py was discovered by walking the project, not from file_paths (empty).
    assert "dead.py" in files


# ── Cooperative cancellation ──────────────────────────────────────────────


def test_is_cancelled_none_safe():
    """``_is_cancelled(None)`` must be False — the helper is called from every
    checkpoint and None (no cancel_event wired) is the common case for direct
    API callers and tests."""
    import threading

    from external_llm.analysis.vulture_scanner import _is_cancelled

    assert _is_cancelled(None) is False
    ev = threading.Event()
    assert _is_cancelled(ev) is False
    ev.set()
    assert _is_cancelled(ev) is True


def test_cancelled_scan_returns_empty_without_scavenge(tmp_path, monkeypatch):
    """A pre-set cancel_event short-circuits the scan to [] before the expensive
    scavenge runs. Verified by spying on ``vulture.core.Vulture`` — its
    ``scavenge`` must never be invoked."""
    import threading

    import vulture.core as vcore

    import external_llm.analysis.vulture_scanner as vs

    scavenge_calls: list = []

    class _SpyVulture:
        def scavenge(self, *a, **k):
            scavenge_calls.append(1)

        def get_unused_code(self, *a, **k):
            return []

    monkeypatch.setattr(vcore, "Vulture", _SpyVulture)

    ev = threading.Event()
    ev.set()
    result = vs.scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
        cancel_event=ev,
    )
    assert result == []
    assert scavenge_calls == [], f"scavenge ran {len(scavenge_calls)} time(s) despite cancel_event being set"


def test_scavenge_with_cancel_aborts_promptly():
    """The core Step-C guarantee: when scavenge is an opaque long call, setting
    cancel_event mid-call makes ``_scavenge_with_cancel`` return False well
    before the call would naturally complete (the daemon thread is abandoned).
    This is what makes ESC responsive DURING the dominant-cost scavenge."""
    import threading
    import time

    from external_llm.analysis.vulture_scanner import _scavenge_with_cancel

    class _SlowVulture:
        def scavenge(self, paths, exclude=None):
            time.sleep(1.0)  # simulate a long opaque parse

    ev = threading.Event()

    def _cancel_soon():
        time.sleep(0.05)
        ev.set()

    threading.Thread(target=_cancel_soon, daemon=True).start()
    t0 = time.time()
    ok = _scavenge_with_cancel(_SlowVulture(), ["x.py"], [], cancel_event=ev)
    elapsed = time.time() - t0
    assert ok is False, "expected cancel → False"
    assert elapsed < 0.4, (
        f"cancel took {elapsed:.2f}s — scavenge did not abort promptly "
        "(should return within the poll interval, not wait for the 1s sleep)"
    )


def test_scavenge_with_cancel_runs_inline_when_no_event():
    """cancel_event=None (non-interactive path) runs scavenge inline with no
    thread overhead and returns True on completion."""
    from external_llm.analysis.vulture_scanner import _scavenge_with_cancel

    calls: list = []

    class _V:
        def scavenge(self, paths, exclude=None):
            calls.append((paths, exclude))

    ok = _scavenge_with_cancel(_V(), ["a.py"], ["ex"], cancel_event=None)
    assert ok is True
    assert calls == [(["a.py"], ["ex"])]


def test_scavenge_with_cancel_pre_set_returns_false():
    """A pre-set cancel_event returns False without invoking scavenge at all."""
    import threading

    from external_llm.analysis.vulture_scanner import _scavenge_with_cancel

    calls: list = []

    class _V:
        def scavenge(self, paths, exclude=None):
            calls.append(1)

    ev = threading.Event()
    ev.set()
    ok = _scavenge_with_cancel(_V(), ["a.py"], [], cancel_event=ev)
    assert ok is False
    assert calls == [], "scavenge should not run when cancel_event is pre-set"


# ── Pre-processing fingerprint caches (aux-pass reuse) ────────────────────────
# The two pre-processing passes (_dispatch_names_for_file / _visitor_hooks_for_file)
# are pure functions of file content, cached per (st_mtime_ns, st_size)
# fingerprint: repeated scans in one process skip re-tokenizing/re-parsing
# unchanged files, and an edit invalidates exactly the edited file.


def test_fingerprint_cache_hit_stale_miss_and_lru_eviction():
    """Cache serves a hit on the same fingerprint, drops stale entries, and
    evicts LRU when bounded."""
    from external_llm.analysis.vulture_scanner import _FingerprintCache

    cache = _FingerprintCache(maxsize=2)
    f1, f2, f3 = (1, 100), (2, 200), (3, 300)
    cache.put("a.py", f1, "r1")
    assert cache.get("a.py", f1) == "r1"
    # Different fingerprint (file edited) → miss, stale entry dropped.
    assert cache.get("a.py", (9, 100)) is None
    assert cache.get("a.py", f1) is None
    # LRU eviction: a.py is least-recently-used → evicted at maxsize.
    cache.put("a.py", f1, "r1")
    cache.put("b.py", f2, "r2")
    cache.put("c.py", f3, "r3")
    assert cache.get("a.py", f1) is None
    assert cache.get("b.py", f2) == "r2"
    assert cache.get("c.py", f3) == "r3"


def test_source_lines_cache_fifo_evicts_oldest_and_stays_correct(tmp_path, monkeypatch):
    """_source_lines_cache must stay bounded: past the cap the oldest entry is
    evicted FIFO, and an evicted path is transparently re-read on the next
    probe — noqa answers stay correct before and after eviction."""
    from external_llm.analysis.vulture_scanner import (
        _SOURCE_LINES_CACHE_MAX_ENTRIES,
        _source_line_has_noqa,
        _source_lines_cache,
    )

    assert _SOURCE_LINES_CACHE_MAX_ENTRIES > 0
    _source_lines_cache.clear()
    monkeypatch.setattr("external_llm.analysis.vulture_scanner._SOURCE_LINES_CACHE_MAX_ENTRIES", 2)
    try:
        files = {}
        for i, tagged in enumerate((False, False, True)):
            p = tmp_path / f"mod{i}.py"
            p.write_text("import os  # noqa: F401\n" if tagged else "import os\n")
            files[i] = p
        # Three distinct paths with cap=2 → the FIRST entry is evicted.
        assert _source_line_has_noqa(str(files[0]), 1, {"F401"}) is False
        assert _source_line_has_noqa(str(files[1]), 1, {"F401"}) is False
        assert _source_line_has_noqa(str(files[2]), 1, {"F401"}) is True
        assert len(_source_lines_cache) == 2
        assert str(files[0]) not in _source_lines_cache, "oldest entry must be evicted"
        # Evicted path is re-read on the next probe; re-insert refreshes it to
        # the back so the NEXT-oldest (files[1]) becomes the eviction victim.
        assert _source_line_has_noqa(str(files[0]), 1, {"F401"}) is False
        assert str(files[1]) not in _source_lines_cache
        # Cached (hit) entries keep answering correctly after eviction.
        assert _source_line_has_noqa(str(files[2]), 1, {"F401"}) is True
    finally:
        _source_lines_cache.clear()


def test_stat_fingerprint_none_for_missing_file(tmp_path):
    from external_llm.analysis.vulture_scanner import _stat_fingerprint

    assert _stat_fingerprint(str(tmp_path / "nope.py")) is None


def test_dispatch_collector_reuses_cache_and_invalidates_on_edit(tmp_path):
    """_collect_dispatch_live_names serves a stable result from the fingerprint
    cache while the file is unchanged, and recomputes after an edit."""
    import external_llm.analysis.vulture_scanner as vs

    probe = tmp_path / "probe.py"
    probe.write_text('DISPATCH = {"a": "_tool_grep"}\n')
    first = vs._collect_dispatch_live_names([str(probe)])
    assert "_tool_grep" in first
    assert str(probe) in vs._dispatch_names_cache._data, "unchanged file must be cached"
    second = vs._collect_dispatch_live_names([str(probe)])
    assert second == first
    # Edit (different size → different fingerprint) → recompute reflects it.
    probe.write_text('DISPATCH = {"b": "_other_longer_handler_name"}\n')
    edited = vs._collect_dispatch_live_names([str(probe)])
    assert "_other_longer_handler_name" in edited
    assert "_tool_grep" not in edited


def test_visitor_hooks_collector_reuses_cache_and_invalidates_on_edit(tmp_path):
    import os

    import external_llm.analysis.vulture_scanner as vs

    probe = tmp_path / "probe.py"
    probe.write_text("class V(NodeVisitor):\n    def visit_Module(self, node):\n        x = 1\n")
    expected = {(os.path.abspath(str(probe)), 2)}
    first = vs._collect_visitor_hook_linenos([str(probe)])
    assert first == expected
    assert str(probe) in vs._visitor_hooks_cache._data, "unchanged file must be cached"
    assert vs._collect_visitor_hook_linenos([str(probe)]) == first
    # Remove the hook → recompute reflects the edit.
    probe.write_text("class V(NodeVisitor):\n    pass\n")
    assert vs._collect_visitor_hook_linenos([str(probe)]) == set()


def test_syntax_error_file_cached_empty_then_recomputed_after_fix(tmp_path):
    """A broken file is cached as empty (no names) and recomputed once fixed —
    the fingerprint changes, so the stale empty result is never served."""
    import external_llm.analysis.vulture_scanner as vs

    probe = tmp_path / "broken.py"
    probe.write_text("def broken(:\n")
    assert vs._collect_dispatch_live_names([str(probe)]) == frozenset()
    assert str(probe) in vs._dispatch_names_cache._data
    probe.write_text('"ok"\n')
    assert vs._collect_dispatch_live_names([str(probe)]) == frozenset({"ok"})


# ── Framework-live suppression (_framework_live_for_file) ──────────────────
# Vulture's per-file view cannot see framework dispatch: enum members,
# pydantic/dataclass fields, http.server protocol methods, and foreign-object
# attribute assignments are live by contract.  The filter is STRUCTURAL
# (inheritance / decorator / AST target shape) — a coincidentally named
# business attribute outside a framework class must still be reported.


def _framework_scan(tmp_path, src: str) -> list:
    """Run the scanner over a single probe file, min_confidence=0."""
    (tmp_path / "probe.py").write_text(src)
    return scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )


def test_enum_members_suppressed(tmp_path):
    """Enum members are consumed by the Enum machinery — never dead code."""
    src = """\
import enum


class Status(enum.Enum):
    OK = "ok"
    FAILED = "failed"
"""
    cands = _framework_scan(tmp_path, src)
    assert {c.name for c in cands} <= {"Status"}


def test_pydantic_fields_validator_suppressed(tmp_path):
    """BaseModel fields / model_config / @model_validator are pydantic contracts."""
    src = """\
from pydantic import BaseModel, ConfigDict, model_validator


class Req(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    count: int = 0

    @model_validator(mode="after")
    def _check(self):
        return self
"""
    cands = _framework_scan(tmp_path, src)
    names = {c.name for c in cands}
    assert not (names & {"name", "count", "model_config", "_check"})


def test_pydantic_through_local_base_chain(tmp_path):
    """Inheritance through a same-file base (``_BaseModel`` → ``BaseModel``)."""
    src = """\
from pydantic import BaseModel


class _BaseModel(BaseModel):
    pass


class Concrete(_BaseModel):
    llm_context_attached: bool = False
"""
    cands = _framework_scan(tmp_path, src)
    assert not any(c.name == "llm_context_attached" for c in cands)


def test_dataclass_fields_suppressed(tmp_path):
    """@dataclass annotated assignments are instance fields, not dead vars."""
    src = """\
from dataclasses import dataclass


@dataclass
class Spec:
    task_id: str
    priority: int = 0
"""
    cands = _framework_scan(tmp_path, src)
    assert not any(c.name in {"task_id", "priority"} for c in cands)


def test_http_handler_protocol_suppressed(tmp_path):
    """BaseHTTPRequestHandler verbs/protocol attrs are dispatched by http.server."""
    src = """\
from http.server import BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    server_version = "probe/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self.close_connection = True

    def do_POST(self):
        pass
"""
    cands = _framework_scan(tmp_path, src)
    names = {c.name for c in cands}
    assert not (names & {"server_version", "protocol_version", "log_message", "do_GET", "do_POST", "close_connection"})


def test_http_handler_wrapper_across_files_suppressed(tmp_path):
    """A local wrapper around BaseHTTPRequestHandler keeps the protocol surface
    framework-live even when the inheritance chain crosses a file boundary.

    Regression: mcp/_session_queue.QuietHttpHandler wraps BaseHTTPRequestHandler
    (quiet disconnect handling); its subclasses in sse_server.py /
    streamable_server.py must still get do_VERB / server_version /
    close_connection suppressed.  _inherits_from is same-file only, so the
    wrapper's name must itself be registered in _HTTP_BASE_NAMES.
    """
    (tmp_path / "_base.py").write_text(
        "from http.server import BaseHTTPRequestHandler\n"
        "\n"
        "\n"
        "class QuietHttpHandler(BaseHTTPRequestHandler):\n"
        "    def handle_one_request(self):\n"
        "        try:\n"
        "            super().handle_one_request()\n"
        "        except (ConnectionResetError, BrokenPipeError):\n"
        "            self.close_connection = True\n",
        encoding="utf-8",
    )
    src = """\
from _base import QuietHttpHandler


class Handler(QuietHttpHandler):
    server_version = "probe/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self.close_connection = True

    def do_POST(self):
        pass
"""
    cands = _framework_scan(tmp_path, src)
    names = {c.name for c in cands}
    assert not (names & {"server_version", "protocol_version", "log_message", "do_GET", "do_POST", "close_connection"})


def test_foreign_object_attribute_assignment_suppressed(tmp_path):
    """``obj.attr = ...`` (obj not bare ``self``) is library-object config."""
    src = """\
import sqlite3


def open_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn
"""
    cands = _framework_scan(tmp_path, src)
    assert not any(c.name == "row_factory" for c in cands)


def test_self_attribute_assign_still_reported(tmp_path):
    """Bare ``self.x = ...`` stays vulture-checkable — not framework-live."""
    src = """\
class C:
    def __init__(self):
        self.orphan_flag = True
"""
    cands = _framework_scan(tmp_path, src)
    assert any(c.name == "orphan_flag" for c in cands)


def test_framework_filter_not_name_based(tmp_path):
    """A ``do_``-named method OUTSIDE an HTTP handler is still reported."""
    src = """\
class Business:
    def do_something(self):
        pass
"""
    cands = _framework_scan(tmp_path, src)
    assert any(c.name == "do_something" for c in cands)


def test_framework_live_cache_invalidates_on_edit(tmp_path):
    """_framework_live_for_file must drop stale entries when the file changes."""
    from external_llm.analysis.vulture_scanner import _framework_live_for_file

    p = tmp_path / "probe.py"
    p.write_text("class S:\n    x = 1\n")
    first = _framework_live_for_file(str(p))
    assert (2, "x") not in first  # plain class → not framework-live

    p.write_text("import enum\n\nclass S(enum.Enum):\n    x = 1\n")
    second = _framework_live_for_file(str(p))
    assert (4, "x") in second  # now an enum member → live
    assert first != second  # cache was invalidated by the edit


def test_scavenge_tolerant_retries_without_vanished_path(tmp_path):
    """A file that vanishes between enumeration and read makes vulture
    sys.exit via SystemExit (get_modules — BaseException, invisible to
    ``except Exception``).  ``_scavenge_tolerant`` must drop the vanished
    path and retry instead of killing the whole scan (the zero-tolerance
    structural gate must not die because a sibling worker/editor removed a
    file mid-run — observed 2026-08-08)."""
    from external_llm.analysis.vulture_scanner import _scavenge_tolerant

    keep = tmp_path / "keep.py"
    keep.write_text("def _k():\n    return 1\n")
    gone = tmp_path / "gone.py"
    gone.write_text("def _g():\n    return 2\n")
    gone.unlink()  # vanished BEFORE the call — the gate's real TOCTOU shape

    calls: list[list] = []

    class _VanishingVulture:
        def scavenge(self, paths, exclude=None):
            calls.append(list(paths))
            for p in paths:
                if not os.path.exists(p):
                    raise SystemExit(f"Error: {p} could not be found.")

    v = _VanishingVulture()
    _scavenge_tolerant(v, [str(keep), str(gone)], [])
    assert len(calls) == 2, f"expected retry after drop, got {len(calls)} call(s)"
    assert calls[0] == [str(keep), str(gone)]
    assert calls[1] == [str(keep)], "vanished path must be dropped for retry"


def test_scavenge_tolerant_reraises_unexpected_system_exit(tmp_path):
    """SystemExit with NO vanished listed path must propagate — the helper
    must never mask unexpected failures."""
    from external_llm.analysis.vulture_scanner import _scavenge_tolerant

    class _BoomVulture:
        def scavenge(self, paths, exclude=None):
            raise SystemExit("something else entirely")

    with pytest.raises(SystemExit):
        _scavenge_tolerant(_BoomVulture(), ["x.py"], [])


# ── Per-file scan disk cache (round 32-P-F) ──────────────────────────────────


class _CountingVulture:
    """Real vulture.core.Vulture that records per-file scan() calls."""

    def __init__(self):
        self.scans: list[str] = []
        self._inner = vulture.Vulture()

    def scan(self, code, filename=""):
        self.scans.append(filename)
        return self._inner.scan(code, filename)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _vulture_files(tmp_path):
    """Minimal 3-file repo: cross-file live ref + dead code + enum member."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "mod.py").write_text(
        "def live_fn():\n    return 1\n\n"
        "def dead_fn():\n    return 2\n\n"
        "class C:\n"
        "    def dead_m(self):\n        return 4\n"
    )
    (tmp_path / "main.py").write_text("from pkg.mod import live_fn\nlive_fn()\n")
    (tmp_path / "enumfile.py").write_text("import enum\n\nclass E(enum.Enum):\n    A = 1\n")
    return [
        str(tmp_path / "pkg" / "mod.py"),
        str(tmp_path / "main.py"),
        str(tmp_path / "enumfile.py"),
    ]


def _scan_unused(tmp_path, paths, **kw):
    from external_llm.analysis.vulture_scanner import (
        _scan_vulture_files_with_cache,
    )

    v = _CountingVulture()
    ok = _scan_vulture_files_with_cache(v, paths, [], str(tmp_path), **kw)
    assert ok
    return v


def _unused_names(v):
    return sorted((i.name, i.typ, i.first_lineno) for i in v.get_unused_code(min_confidence=0))


def test_scan_cache_hot_run_matches_cold_and_reuses_all_entries(tmp_path):
    """A cache-hot run must re-parse NOTHING while producing the same
    get_unused_code() output as the cold full scan — cross-file reachability
    is preserved because used names are rehydrated into the global set."""
    from external_llm.analysis.vulture_scanner import _vulture_cache_path

    paths = _vulture_files(tmp_path)

    cold = _scan_unused(tmp_path, paths)
    cold_results = _unused_names(cold)
    assert len(cold.scans) == len(paths), "cold run must scan every file"
    assert os.path.exists(_vulture_cache_path(str(tmp_path)))

    hot = _scan_unused(tmp_path, paths)
    assert hot.scans == [], "hot run must re-parse nothing"
    assert _unused_names(hot) == cold_results, "cache-rehydrated run must match the full scan"


def test_scan_cache_restore_str_filename_matches_cold_path_filename(tmp_path):
    """A cache-hot (restored) run's get_unused_code() must be identical to the
    cold (scanned) run in ORDER as well as content — the restore path stores
    Item filenames as ``str`` while a real ``v.scan`` uses ``pathlib.Path``.
    ``get_unused_code()`` sorts by ``str(filename).lower()`` and hashes
    ``(filename, lineno, name)``, so the two surfaces must agree bit-for-bit
    on every output tuple (name, typ, lineno) in raw sorted order."""
    from pathlib import Path

    import vulture.core

    from external_llm.analysis.vulture_scanner import (
        _load_vulture_scan_cache,
        _restore_vulture_entry,
    )

    paths = _vulture_files(tmp_path)

    cold = _scan_unused(tmp_path, paths)
    cold_raw = [(i.name, i.typ, i.first_lineno) for i in cold.get_unused_code(min_confidence=0)]

    # Rehydrate the same cache into a fresh Vulture with the str-filename path.
    files = _load_vulture_scan_cache(str(tmp_path), vulture.__version__)
    v = vulture.core.Vulture(verbose=False)
    for _rel, entry in files.items():
        _restore_vulture_entry(v, entry)
    hot_raw = [(i.name, i.typ, i.first_lineno) for i in v.get_unused_code(min_confidence=0)]
    assert hot_raw == cold_raw, "restored (str filename) get_unused_code order must match cold (Path filename)"
    # The restored items must actually carry a plain str filename.
    assert all(
        isinstance(i.filename, str) and not isinstance(i.filename, Path) for i in v.get_unused_code(min_confidence=0)
    )


def test_scan_cache_reparses_only_edited_file(tmp_path):
    """Editing one file invalidates exactly that file's entry — the other
    entries are rehydrated and the edit's new dead code is detected."""
    paths = _vulture_files(tmp_path)
    _scan_unused(tmp_path, paths)

    (tmp_path / "pkg" / "mod.py").write_text(
        "def live_fn():\n    return 1\n\ndef dead_fn():\n    return 2\n\ndef newly_dead():\n    return 9\n"
    )
    v = _scan_unused(tmp_path, paths)
    assert v.scans == [str(tmp_path / "pkg" / "mod.py")], "only the edited file must be re-scanned"
    names = {i.name for i in v.get_unused_code(min_confidence=0)}
    assert "newly_dead" in names


def test_scan_cache_corrupt_file_falls_back_to_full_scan(tmp_path):
    """A corrupt cache file must never change results — fail-open to a full
    re-scan (same contract as the structural graph cache)."""
    from pathlib import Path

    from external_llm.analysis.vulture_scanner import _vulture_cache_path

    paths = _vulture_files(tmp_path)
    cold = _scan_unused(tmp_path, paths)
    cold_results = _unused_names(cold)

    Path(_vulture_cache_path(str(tmp_path))).write_text("{not json")
    v = _scan_unused(tmp_path, paths)
    assert len(v.scans) == len(paths), "corrupt cache must trigger a full scan"
    assert _unused_names(v) == cold_results


def test_scan_cache_vulture_version_mismatch_discarded(tmp_path):
    """A cache written by a different vulture version must be discarded —
    its item shape is not guaranteed compatible."""
    import json
    from pathlib import Path

    from external_llm.analysis.vulture_scanner import _vulture_cache_path

    paths = _vulture_files(tmp_path)
    _scan_unused(tmp_path, paths)

    p = Path(_vulture_cache_path(str(tmp_path)))
    payload = json.loads(p.read_text())
    payload["vulture"] = "0.0.0-other"
    p.write_text(json.dumps(payload))

    v = _scan_unused(tmp_path, paths)
    assert len(v.scans) == len(paths), "a vulture-version mismatch must discard the cache"


def test_scan_cache_skips_vanished_file(tmp_path):
    """A path that vanishes between enumeration and read must be skipped —
    per-file granularity makes one missing file harmless (the whole-set
    SystemExit retry loop only existed for scavenge())."""
    paths = _vulture_files(tmp_path)
    v = _scan_unused(tmp_path, paths)
    results_before = _unused_names(v)

    # Second run with an extra path that does not exist — the vanished file
    # must be skipped while every cached file is rehydrated and results stay
    # identical (the whole-set SystemExit abort cannot happen per-file).
    v2 = _scan_unused(tmp_path, [*paths, str(tmp_path / "gone.py")])
    assert "gone.py" not in [os.path.basename(s) for s in v2.scans]
    assert _unused_names(v2) == results_before, "a vanished file must be skipped, not abort the scan"


def test_scan_cache_used_set_is_order_independent(tmp_path):
    """The cached per-file "used" set must be ABSOLUTE (pure function of file
    content), not a scan-order delta: an entry written while other files were
    scanned first, restored ALONE in per-file mode, must not flag names that
    were pre-used by those other files.  Regression 2026-08-16: full-repo
    entries restored in per-file mode falsely flagged ``logger``/``ast``
    as unused imports (format bumped 1 → 2).
    """
    from external_llm.analysis.vulture_scanner import (
        _load_vulture_scan_cache,
        _scan_vulture_files_with_cache,
    )

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("import logging\nlogging.getLogger(__name__)\n")
    (tmp_path / "pkg" / "b.py").write_text(
        "import ast\nimport logging\n\ndef f():\n    return ast.parse('x'), logging.getLogger(__name__)\n"
    )
    a = str(tmp_path / "pkg" / "a.py")
    b = str(tmp_path / "pkg" / "b.py")

    # Cold scan in [a, b] order — with the old delta format, b's "used" would
    # miss logging/ast because a first-used them.
    v = _CountingVulture()
    assert _scan_vulture_files_with_cache(v, [a, b], [], str(tmp_path))
    rel_b = os.path.relpath(b, str(tmp_path))
    import vulture as _vulture_pkg

    b_used = set(_load_vulture_scan_cache(str(tmp_path), _vulture_pkg.__version__)[rel_b]["used"])
    assert {"logging", "ast"} <= b_used, "b's cached used set must be absolute, not a scan-order delta"

    # Restore ONLY b (per-file mode) — no false unused-import verdicts.
    v2 = _CountingVulture()
    assert _scan_vulture_files_with_cache(v2, [b], [], str(tmp_path))
    assert v2.scans == [], "b must be served from the cache"
    unused = {(i.name, i.typ) for i in v2.get_unused_code(min_confidence=0)}
    assert not any(typ == "import" for _name, typ in unused), "restoring b alone must not flag its imports as unused"


def test_scan_cache_exclude_pattern_skips_matching_files(tmp_path):
    """Exclude patterns must keep scavenge() semantics: bare patterns are
    wrapped as ``*pat*`` and matched case-insensitively."""
    from external_llm.analysis.vulture_scanner import (
        _scan_vulture_files_with_cache,
    )

    paths = _vulture_files(tmp_path)
    v = _CountingVulture()
    ok = _scan_vulture_files_with_cache(v, paths, ["main"], str(tmp_path))
    assert ok
    assert v.scans == [
        str(tmp_path / "pkg" / "mod.py"),
        str(tmp_path / "enumfile.py"),
    ], "exclude patterns must use scavenge semantics (*pat* wrapping)"


def test_scan_cache_pre_set_cancel_returns_false(tmp_path):
    """A pre-set cancel_event returns False without scanning any file."""
    import threading

    from external_llm.analysis.vulture_scanner import (
        _scan_vulture_files_with_cache,
    )

    paths = _vulture_files(tmp_path)
    ev = threading.Event()
    ev.set()
    v = _CountingVulture()
    ok = _scan_vulture_files_with_cache(v, paths, [], str(tmp_path), cancel_event=ev)
    assert ok is False
    assert v.scans == []


# ── Pre-processing disk cache (dispatch / visitor-hook / framework-live) ──


def test_preprocess_cache_skips_recomputation_on_second_scan(tmp_path, monkeypatch):
    """The dispatch / visitor-hook / framework-live passes are pure per-file
    functions whose in-memory caches die with the process; the v.scan disk
    cache must rehydrate them so a second scan in a fresh call recomputes
    NOTHING (measured ~6s saved in the structural gate)."""
    from external_llm.analysis.vulture_scanner import (
        _dispatch_names_cache,
        _dispatch_names_for_file,
        _framework_live_cache,
        _framework_live_for_file,
        _stat_fingerprint,
        _visitor_hooks_cache,
        _visitor_hooks_for_file,
        scan_vulture_dead_code,
    )

    paths = _vulture_files(tmp_path)
    # Count CACHE MISSES (the collection loop always CALLS the per-file
    # function; only a miss reaches the tokenize/parse body).
    misses = {"dispatch": 0, "vhooks": 0, "framework": 0}

    def counting(cache, key, fn):
        def wrapper(path):
            fp = _stat_fingerprint(path)
            if fp is None or cache.get(path, fp) is None:
                misses[key] = misses.get(key, 0) + 1
            return fn(path)

        return wrapper

    monkeypatch.setattr(
        "external_llm.analysis.vulture_scanner._dispatch_names_for_file",
        counting(_dispatch_names_cache, "dispatch", _dispatch_names_for_file),
    )
    monkeypatch.setattr(
        "external_llm.analysis.vulture_scanner._visitor_hooks_for_file",
        counting(_visitor_hooks_cache, "vhooks", _visitor_hooks_for_file),
    )
    monkeypatch.setattr(
        "external_llm.analysis.vulture_scanner._framework_live_for_file",
        counting(_framework_live_cache, "framework", _framework_live_for_file),
    )

    first = scan_vulture_dead_code(repo_root=str(tmp_path), file_paths=paths)
    assert misses["dispatch"] > 0, "cold run must compute dispatch names"
    assert misses["vhooks"] > 0, "cold run must compute visitor hooks"
    assert misses["framework"] > 0, "cold run must compute framework-live"

    misses.clear()
    second = scan_vulture_dead_code(repo_root=str(tmp_path), file_paths=paths)
    assert misses.get("dispatch", 0) == 0, f"dispatch recomputed on warm scan ({misses.get('dispatch', 0)} misses)"
    assert misses.get("vhooks", 0) == 0, f"visitor hooks recomputed on warm scan ({misses.get('vhooks', 0)} misses)"
    assert misses.get("framework", 0) == 0, (
        f"framework-live recomputed on warm scan ({misses.get('framework', 0)} misses)"
    )
    assert [c.name for c in first] == [c.name for c in second], "warm scan must reproduce the cold candidate set"


def test_preprocess_cache_invalidates_only_edited_file(tmp_path, monkeypatch):
    """Editing one file invalidates ITS pre-processing results; the others
    stay cache-served (fingerprint-keyed invalidation, same contract as the
    v.scan cache)."""
    import time

    from external_llm.analysis.vulture_scanner import (
        _dispatch_names_cache,
        _dispatch_names_for_file,
        _stat_fingerprint,
        scan_vulture_dead_code,
    )

    paths = _vulture_files(tmp_path)
    scan_vulture_dead_code(repo_root=str(tmp_path), file_paths=paths)

    # Edit ONE file (new string literal) — ensure mtime/size both change.
    target = tmp_path / "main.py"
    before = target.read_text()
    target.write_text(before + "\n# touch\n")
    time.sleep(0.01)

    misses = {"dispatch": 0}
    real = _dispatch_names_for_file

    def counting(path):
        fp = _stat_fingerprint(path)
        if fp is None or _dispatch_names_cache.get(path, fp) is None:
            misses["dispatch"] = misses.get("dispatch", 0) + 1
        return real(path)

    monkeypatch.setattr("external_llm.analysis.vulture_scanner._dispatch_names_for_file", counting)
    scan_vulture_dead_code(repo_root=str(tmp_path), file_paths=paths)
    assert misses["dispatch"] == 1, f"expected exactly the edited file to be recomputed, got {misses['dispatch']}"


def test_vulture_cache_path_goes_through_path_guard(tmp_path, monkeypatch):
    """P-1 regression: the cache path must route through the fail-closed guard."""
    from external_llm.analysis import parse_cache
    from external_llm.analysis.vulture_scanner import (
        _VULTURE_CACHE_VERSION,
        _vulture_cache_path,
    )

    calls = []
    monkeypatch.setattr(
        parse_cache,
        "cache_file_path",
        lambda root, filename: calls.append((root, filename)) or "/guarded",
    )
    assert _vulture_cache_path(str(tmp_path)) == "/guarded"
    assert calls == [(str(tmp_path), f"vulture_scan_v{_VULTURE_CACHE_VERSION}.json")]


# ── Save-skip policy + v3 entry format (round 32-F2) ───────────────────────


def test_scan_cache_v3_entry_hoists_filename(tmp_path):
    """v3 entries store the file path ONCE ("fn"); items never repeat it —
    the repeated key was ~19% of the payload (109K absolute paths)."""
    import json
    from pathlib import Path

    from external_llm.analysis.vulture_scanner import _vulture_cache_path

    paths = _vulture_files(tmp_path)
    _scan_unused(tmp_path, paths)

    payload = json.loads(Path(_vulture_cache_path(str(tmp_path))).read_text())
    assert payload["format"] == 3
    entries = payload["files"]
    assert entries, "cold run must persist (small corpus always saves)"
    for rel, entry in entries.items():
        assert entry.get("fn"), f"{rel}: entry-level fn missing"
        for item in entry.get("items", []):
            assert "filename" not in item, f"{rel}: items must not repeat filename"


def test_scan_cache_partial_update_of_large_corpus_skips_save(tmp_path, monkeypatch):
    """A 1-file edit of a large corpus must NOT pay a full-payload
    serialisation: the save is skipped and the next run rescans that file
    (recompute ≈ 0.1s ≪ serialise ≈ 4.6s on asicode)."""
    import os
    from pathlib import Path

    from external_llm.analysis import parse_cache as pc
    from external_llm.analysis.vulture_scanner import _vulture_cache_path

    # Shrink the policy so a 3-file repo counts as "large": 1/3 ≤ 50% skips.
    monkeypatch.setattr(pc, "SAVE_SKIP_MIN_ENTRIES", 2)
    monkeypatch.setattr(pc, "SAVE_SKIP_MAX_FRACTION", 0.5)

    paths = _vulture_files(tmp_path)
    _scan_unused(tmp_path, paths)
    cache_file = Path(_vulture_cache_path(str(tmp_path)))
    saved_mtime = os.stat(cache_file).st_mtime_ns

    (tmp_path / "pkg" / "mod.py").write_text(
        "def live_fn():\n    return 1\n\ndef dead_fn():\n    return 2\n\ndef newly_dead():\n    return 9\n"
    )
    v = _scan_unused(tmp_path, paths)
    assert os.stat(cache_file).st_mtime_ns == saved_mtime, "a partial update of a large corpus must skip the save"
    assert v.scans == [str(tmp_path / "pkg" / "mod.py")], "the edited file was still rescanned in-memory"

    # The skip costs one bounded rescan on the NEXT run — fail-open.
    v2 = _scan_unused(tmp_path, paths)
    assert v2.scans == [str(tmp_path / "pkg" / "mod.py")], "next process must recompute the skipped entry"


def test_cold_full_scan_writes_payload_exactly_once(tmp_path, monkeypatch):
    """Cold full flow (scan + preprocess sync) must serialise the payload
    exactly ONCE — the scan defers its persistence to the caller's single
    decision point instead of saving at both the scan site and the finally."""
    from external_llm.analysis import vulture_scanner as vs

    saves: list[int] = []
    real = vs._save_vulture_scan_cache

    def spy(root, files, version):
        saves.append(len(files))
        return real(root, files, version)

    monkeypatch.setattr(vs, "_save_vulture_scan_cache", spy)

    _vulture_files(tmp_path)
    vs.scan_vulture_dead_code(repo_root=str(tmp_path))
    assert len(saves) == 1, "cold run must write the payload exactly once"


def test_preprocess_warm_rehydrate_keeps_visitor_hooks_with_relative_repo_root(tmp_path, monkeypatch):
    """Regression (round 32-F2): ``_warm_preprocess_caches`` built vhook
    tuples from ``normpath(join(repo_root, rel))`` — RELATIVE when repo_root
    is relative — while the fresh path (``_visitor_hooks_for_file``) and the
    candidate check key by ``abspath``.  Visitor hooks rehydrated from disk
    silently stopped matching, leaking visit_* methods as false dead code.
    The gate always passes an absolute repo_root (which is why CI stayed
    green); any relative-root caller hit the leak on its second scan."""
    from external_llm.analysis import vulture_scanner as vs

    (tmp_path / "visitor_mod.py").write_text(
        "import ast\n\n"
        "class V(ast.NodeVisitor):\n"
        "    def visit_ClassDef(self, node):\n"
        "        return self.generic_visit(node)\n"
    )
    monkeypatch.chdir(tmp_path)

    cold = vs.scan_vulture_dead_code(repo_root=".")
    assert all(c.name != "visit_ClassDef" for c in cold), "cold run must suppress the visitor hook"

    # Emulate a fresh process: the module-level in-memory fingerprint caches
    # are what a new interpreter would NOT have — only the DISK cache remains.
    vs._dispatch_names_cache._data.clear()
    vs._visitor_hooks_cache._data.clear()
    vs._framework_live_cache._data.clear()

    warm = vs.scan_vulture_dead_code(repo_root=".")
    leaked = [c for c in warm if c.name == "visit_ClassDef"]
    assert not leaked, "rehydrated visitor hooks must still suppress visit_*"
