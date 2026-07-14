'''Tests for external_llm/analysis/vulture_scanner.py.

Focus: the ``exclude_kinds`` overlap filter (Option B). vulture's per-file view
of module-level functions/classes is redundant with ``public_dead_code_scanner``
(which resolves cross-file references), so those kinds are excluded by default.
vulture's UNIQUE value — class-level ``method``/``variable``/``property`` — must
survive the default filter. (vulture distinguishes ``function`` from ``method``.)
'''
from __future__ import annotations

import pytest

vulture = pytest.importorskip("vulture.core")  # skip entire module if optional dep missing

from external_llm.analysis.vulture_scanner import (  # noqa: E402
    _PUBLIC_DEAD_CODE_OVERLAP_KINDS,
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
    assert _PUBLIC_DEAD_CODE_OVERLAP_KINDS == frozenset({"function", "class"})


def test_kind_map_distinguishes_function_from_method():
    '''Critical: vulture's ``method`` typ must map distinctly from ``function``.
    If they collapsed to one kind, the overlap filter would wrongly drop
class-level methods (which public_dead_code_scanner does NOT cover).'''
    assert _VULTURE_KIND_MAP["function"] == "function"
    assert _VULTURE_KIND_MAP["method"] == "method"
    assert "method" not in _PUBLIC_DEAD_CODE_OVERLAP_KINDS


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
