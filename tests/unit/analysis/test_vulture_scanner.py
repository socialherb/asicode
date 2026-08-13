'''Tests for external_llm/analysis/vulture_scanner.py.

Focus: the ``exclude_kinds`` overlap filter (Option B). vulture's per-file view
of module-level functions/classes is redundant with ``public_dead_code_scanner``
(which resolves cross-file references), so those kinds are excluded by default.
vulture's UNIQUE value — class-level ``method``/``variable``/``property`` — must
survive the default filter. (vulture distinguishes ``function`` from ``method``.)
'''
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
    '''Write _SAMPLE into tmp_path and run the scanner with min_confidence=0.'''
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
    '''Critical: vulture's ``method`` typ must map distinctly from ``function``.
    If they collapsed to one kind, the overlap filter would wrongly drop
class-level methods (which public_dead_code_scanner does NOT cover).'''
    assert _VULTURE_KIND_MAP["function"] == "function"
    assert _VULTURE_KIND_MAP["method"] == "method"
    assert "method" not in _PUBLIC_DEAD_CODE_OVERLAP_KINDS


def test_vulture_code_by_kind_matches_cli_codes():
    '''The scanner's noqa contract must mirror vulture CLI codes (core.py:32-38)
    so a "# noqa: V1xx" comment suppresses the same candidate in both paths.'''
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
    '''A "# noqa: V105" on an unused method suppresses it — vulture's own
    get_unused_code() applies no noqa filtering (that lives in its CLI), so the
    scanner must honor the flake8-style code itself.'''
    src = """\
class C:
    def dead_method(self):  # noqa: V105 — documented intentional
        pass
"""
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path), file_paths=["probe.py"],
        repo_graph=None, min_confidence=0,
    )
    names = {c.name for c in cands}
    assert "dead_method" not in names


def test_v_code_noqa_does_not_suppress_other_kind(tmp_path):
    '''Code matching is kind-precise: a V101 (attribute) noqa must NOT suppress
    an unused method (V105).'''
    src = """\
class C:
    def dead_method(self):  # noqa: V101 — wrong kind, must not match
        pass
"""
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path), file_paths=["probe.py"],
        repo_graph=None, min_confidence=0,
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
    '''The pivotal regression guard: class methods are vulture-only signal and
    MUST survive the default overlap filter.'''
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
    '''Passing exclude_kinds replaces the default set (does not augment it).'''
    cands = _scan(tmp_path, exclude_kinds={"method"})
    kinds = {c.kind for c in cands}
    assert "method" not in kinds
    # function/class are NOT in the custom set -> they reappear (default overridden)
    assert "function" in kinds
    assert "class" in kinds


# ── Always-live dunder still filtered (regression for reorder) ───────────


def test_always_live_dunder_still_filtered(tmp_path):
    '''The kind-filter reorder must not break the _ALWAYS_LIVE name filter.'''
    src = """\
class C:
    def __init__(self):
        pass
"""
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path), file_paths=["probe.py"],
        repo_graph=None, min_confidence=0,
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
    (tmp_path / "tests" / "test_noise.py").write_text(
        "unused_fixture_in_test = 123\n"
    )
    (tmp_path / "probe.py").write_text(
        "class C:\n"
        "    def dead_method(self):\n"
        "        pass\n"
    )
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py", "tests/test_noise.py"],
        repo_graph=None, min_confidence=0,
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
        repo_root=str(tmp_path), file_paths=["probe.py"],
        repo_graph=None, min_confidence=0,
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
    (tmp_path / "registry.py").write_text(
        'HANDLER_MAP = {"web": "_dispatched_handler"}\n'
    )
    (tmp_path / "probe.py").write_text(
        "class Tools:\n"
        "    def _dispatched_handler(self):\n"
        "        pass\n\n"
        "    def _truly_dead(self):\n"
        "        pass\n"
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
        "def _make() -> \"AnnotOnly\":\n"
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
        repo_root=str(tmp_path), file_paths=["probe.py"],
        repo_graph=None, min_confidence=0,
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
        "class HttpClient:\n"            # NOT a visitor subclass
        "    def visit_url(self, url):\n"  # coincidentally named, truly dead
        "        x = 1\n"
    )
    (tmp_path / "probe.py").write_text(src)
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path), file_paths=["probe.py"],
        repo_graph=None, min_confidence=0,
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
        repo_root=str(tmp_path), file_paths=["probe.py"],
        repo_graph=None, min_confidence=0,
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
    '''full_project mode must enumerate the project .py set explicitly, never
    walking into .venv. We place a dead-code file under .venv AND list it in
    file_paths: if vulture scanned it it would be reported; the skip keeps it
    absent from results.'''
    (tmp_path / "real.py").write_text(
        "class C:\n    def unused_method(self):\n        pass\n"
    )
    vendored_dir = tmp_path / ".venv" / "site-packages" / "somepkg"
    vendored_dir.mkdir(parents=True)
    (vendored_dir / "vendored.py").write_text(
        "def totally_dead_vendored():\n    pass\n"
    )
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
    '''With no file_paths targets, full_project must walk the whole project
    (via _collect_project_py_files) and report dead code from any project file
    — not scan nothing. This locks in that the fix enumerates the project set
    instead of relying on the caller-supplied file_paths.'''
    (tmp_path / "dead.py").write_text(
        "class C:\n    def unused_method(self):\n        pass\n"
    )
    (tmp_path / "alive.py").write_text("x = 1\nprint(x)\n")
    cands = scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=[],          # no targets → full_project walks everything
        repo_graph=None,        # → full_project
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
        repo_root=str(tmp_path), file_paths=["probe.py"], repo_graph=None,
        min_confidence=0, cancel_event=ev,
    )
    assert result == []
    assert scavenge_calls == [], (
        f"scavenge ran {len(scavenge_calls)} time(s) despite cancel_event being set"
    )


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
    monkeypatch.setattr(
        "external_llm.analysis.vulture_scanner._SOURCE_LINES_CACHE_MAX_ENTRIES", 2
    )
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
    probe.write_text(
        "class V(NodeVisitor):\n"
        "    def visit_Module(self, node):\n"
        "        x = 1\n"
    )
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
    '''Run the scanner over a single probe file, min_confidence=0.'''
    (tmp_path / "probe.py").write_text(src)
    return scan_vulture_dead_code(
        repo_root=str(tmp_path),
        file_paths=["probe.py"],
        repo_graph=None,
        min_confidence=0,
    )


def test_enum_members_suppressed(tmp_path):
    '''Enum members are consumed by the Enum machinery — never dead code.'''
    src = """\
import enum


class Status(enum.Enum):
    OK = "ok"
    FAILED = "failed"
"""
    cands = _framework_scan(tmp_path, src)
    assert {c.name for c in cands} <= {"Status"}


def test_pydantic_fields_validator_suppressed(tmp_path):
    '''BaseModel fields / model_config / @model_validator are pydantic contracts.'''
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
    '''Inheritance through a same-file base (``_BaseModel`` → ``BaseModel``).'''
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
    '''@dataclass annotated assignments are instance fields, not dead vars.'''
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
    '''BaseHTTPRequestHandler verbs/protocol attrs are dispatched by http.server.'''
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
    assert not (names & {"server_version", "protocol_version", "log_message",
                         "do_GET", "do_POST", "close_connection"})


def test_foreign_object_attribute_assignment_suppressed(tmp_path):
    '''``obj.attr = ...`` (obj not bare ``self``) is library-object config.'''
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
    '''Bare ``self.x = ...`` stays vulture-checkable — not framework-live.'''
    src = """\
class C:
    def __init__(self):
        self.orphan_flag = True
"""
    cands = _framework_scan(tmp_path, src)
    assert any(c.name == "orphan_flag" for c in cands)


def test_framework_filter_not_name_based(tmp_path):
    '''A ``do_``-named method OUTSIDE an HTTP handler is still reported.'''
    src = """\
class Business:
    def do_something(self):
        pass
"""
    cands = _framework_scan(tmp_path, src)
    assert any(c.name == "do_something" for c in cands)


def test_framework_live_cache_invalidates_on_edit(tmp_path):
    '''_framework_live_for_file must drop stale entries when the file changes.'''
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
