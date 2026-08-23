"""PARITY guard: every write tool invalidates the caches it invalidates for.

``_invalidate_cache_after_write`` clears the file cache, the call-graph index,
the RAG index and the per-root file-walk caches. It was reachable from exactly
TWO handler-internal call sites (``write_plan``, and one ``apply_patch`` branch
guarded by ``if touched:``) while ``_WRITE_TOOLS`` has seven members.

Measured before the fix: a *successful* ``apply_patch`` — new file and existing
file alike — invoked it ZERO times. Every cache kept serving pre-write state
until its TTL expired, so ``find_symbol`` answered "No definitions found" for a
function the agent had just written to disk. ``edit_text`` / ``edit_ast`` /
``anchor_edit`` / ``modify_symbol`` / ``edit_file`` never had a call site at all.

The invalidation now happens in ``dispatch``, at the same central post-success
point the semantic auto-repair uses, so a write tool cannot skip it. These tests
drive each tool through the real ``dispatch`` path — invoking the handler
directly would bypass exactly the layer under test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import external_llm.agent._shared_utils as su
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


@pytest.fixture
def cfg():
    return AgentConfig(
        max_turns=5,
        run_tests=False,
        run_lint=False,
        auto_test_on_patch=False,
        self_review_enabled=False,
        rag_enabled=False,
        vector_cache_enabled=False,
        parallel_tool_execution_enabled=False,
    )


@pytest.fixture
def repo(cfg):
    """A registry over a temp repo with the walk cache already warmed.

    Warming matters: the bug is invisible on a cold cache, because the first
    walk after the write sees the new file anyway. The stale-read only appears
    when a walk was cached BEFORE the write.
    """
    d = Path(tempfile.mkdtemp(prefix="asr-invalidation-"))
    (d / "a.py").write_text("x = 1\n", encoding="utf-8")
    reg = ToolRegistry(str(d), cfg)
    su._walk_py_files(Path(reg.repo_root), 100)
    return reg, d


def _visible(reg) -> set[str]:
    return {p.name for p in su._walk_py_files(Path(reg.repo_root), 100)}


def _find(reg, name: str) -> str:
    return reg.dispatch("find_symbol", {"name": name}).content or ""


# ── one case per write tool ───────────────────────────────────────────────
# Each returns the dispatch args; the file it targets is pre-created where the
# tool requires an existing target, so only the *symbol* is new.


def _case_apply_patch_new(d: Path) -> tuple[str, dict]:
    return "apply_patch", {"patch": "--- /dev/null\n+++ b/brand_new.py\n@@ -0,0 +1 @@\n+def brand_new(): pass\n"}


def _case_apply_patch_existing(d: Path) -> tuple[str, dict]:
    (d / "brand_new.py").write_text("def placeholder():\n    pass\n", encoding="utf-8")
    return "apply_patch", {
        "patch": (
            "--- a/brand_new.py\n+++ b/brand_new.py\n@@ -1,2 +1,2 @@\n"
            "-def placeholder():\n+def brand_new():\n     pass\n"
        )
    }


def _case_edit_text(d: Path) -> tuple[str, dict]:
    (d / "brand_new.py").write_text("def placeholder(): pass\n", encoding="utf-8")
    return "edit_text", {
        "path": "brand_new.py",
        "old_string": "placeholder",
        "new_string": "brand_new",
    }


def _case_anchor_edit(d: Path) -> tuple[str, dict]:
    (d / "brand_new.py").write_text("def placeholder():\n    pass\n", encoding="utf-8")
    return "anchor_edit", {
        "file_path": "brand_new.py",
        "anchor_pattern": "def placeholder():",
        "edit_mode": "replace_line",
        "code_snippet": "def brand_new():",
    }


def _case_modify_symbol(d: Path) -> tuple[str, dict]:
    (d / "brand_new.py").write_text("def placeholder():\n    pass\n", encoding="utf-8")
    return "modify_symbol", {
        "file_path": "brand_new.py",
        "symbol": "placeholder",
        "code": "def brand_new():\n    return 1\n",
    }


_CASES = {
    "apply_patch:new-file": _case_apply_patch_new,
    "apply_patch:existing": _case_apply_patch_existing,
    "edit_text": _case_edit_text,
    "anchor_edit": _case_anchor_edit,
    "modify_symbol": _case_modify_symbol,
}


@pytest.mark.parametrize("label", sorted(_CASES))
def test_write_tool_invalidates_the_walk_cache(repo, label):
    """After a successful write, a file the agent just created/renamed into
    existence must be findable immediately — not after the 30 s cache TTL."""
    reg, d = repo
    tool, args = _CASES[label](d)
    # Precondition: the warmed cache predates this file, so a stale read is
    # possible. Without this the test could pass on a cache that was never warm.
    assert "brand_new.py" not in _visible(reg)

    result = reg.dispatch(tool, args)
    assert result.ok, f"{label} failed to write: {result.error}"

    assert "brand_new.py" in _visible(reg), f"{label}: walk cache still stale after a successful write"
    assert "Found" in _find(reg, "brand_new"), f"{label}: find_symbol cannot see a symbol that is on disk"


@pytest.mark.parametrize("label", sorted(_CASES))
def test_write_tool_reaches_the_central_invalidation_hook(repo, label, monkeypatch):
    """Structural companion to the behavioural test: the hook must actually be
    invoked. Before the fix this counter was 0 for every case below."""
    reg, d = repo
    calls: list = []
    orig = type(reg)._invalidate_cache_after_write
    monkeypatch.setattr(
        type(reg),
        "_invalidate_cache_after_write",
        lambda self, paths: (calls.append(list(paths)), orig(self, paths))[1],
    )
    tool, args = _CASES[label](d)
    result = reg.dispatch(tool, args)
    assert result.ok, f"{label} failed to write: {result.error}"
    assert calls, f"{label}: _invalidate_cache_after_write was never called"


def test_failed_write_does_not_invalidate(repo, monkeypatch):
    """Invalidation is gated on result.ok — a rejected edit changed nothing, and
    paying the RAG/call-graph re-index for it would be pure waste."""
    reg, _d = repo
    calls: list = []
    orig = type(reg)._invalidate_cache_after_write
    monkeypatch.setattr(
        type(reg),
        "_invalidate_cache_after_write",
        lambda self, paths: (calls.append(list(paths)), orig(self, paths))[1],
    )
    result = reg.dispatch(
        "edit_text",
        {
            "path": "a.py",
            "old_string": "NO_SUCH_TEXT_ANYWHERE",
            "new_string": "x",
        },
    )
    assert not result.ok
    assert calls == [], "invalidation ran for a failed write"


def test_read_only_tool_does_not_invalidate(repo, monkeypatch):
    reg, _d = repo
    calls: list = []
    orig = type(reg)._invalidate_cache_after_write
    monkeypatch.setattr(
        type(reg),
        "_invalidate_cache_after_write",
        lambda self, paths: (calls.append(list(paths)), orig(self, paths))[1],
    )
    reg.dispatch("find_symbol", {"name": "anything"})
    assert calls == [], "invalidation ran for a read-only tool"


# ── The same guarantee, for the two paths the original fix did not cover ─────
# 1. NON-Python languages. _invalidate_cache_after_write cleared six caches but
#    not the non-Python symbol caches, so a Go/Rust/Java symbol the agent had
#    just written stayed invisible for the full 30 s TTL while an equivalent
#    Python edit was visible at once.
# 2. bash. That hook needs target paths, so it is only reachable from the write
#    TOOLS — yet the agent's own no-tool nudge instructs it to create files with
#    `bash('cat > path << EOF ...')`. A mutating bash cleared the tool-result
#    cache and nothing else, so bash-authored files (Python ones too) were
#    invisible to find_symbol.


@pytest.fixture
def multilang_repo(cfg):
    """Temp repo with BOTH caches warmed, Python and non-Python."""
    d = Path(tempfile.mkdtemp(prefix="asr-invalidation-ml-"))
    (d / "a.py").write_text("def py_alpha():\n    return 1\n", encoding="utf-8")
    (d / "a.go").write_text("package main\n\nfunc GoAlpha() int { return 1 }\n", encoding="utf-8")
    reg = ToolRegistry(str(d), cfg)
    # Warm every layer: walk cache, non-Python index, and the probe's file list.
    su._walk_py_files(Path(reg.repo_root), 100)
    _find(reg, "py_alpha")
    _find(reg, "GoAlpha")
    return reg, d


def test_nonpython_symbol_visible_right_after_edit(multilang_repo):
    """Editing a .go file must expose its new symbol immediately."""
    reg, d = multilang_repo
    res = reg.dispatch(
        "edit_text",
        {
            "file_path": "a.go",
            "old_string": "func GoAlpha()",
            "new_string": "func GoBeta() int { return 9 }\n\nfunc GoAlpha()",
        },
    )
    assert res.ok, res.error
    assert "GoBeta" in (d / "a.go").read_text(encoding="utf-8")
    assert "No definitions found" not in _find(reg, "GoBeta"), "non-Python symbol invisible after the agent's own edit"


def test_python_and_nonpython_are_symmetric(multilang_repo):
    """Neither language may be the privileged one."""
    reg, _d = multilang_repo
    reg.dispatch(
        "edit_text",
        {
            "file_path": "a.py",
            "old_string": "def py_alpha():",
            "new_string": "def PyBeta():\n    return 9\n\n\ndef py_alpha():",
        },
    )
    reg.dispatch(
        "edit_text",
        {
            "file_path": "a.go",
            "old_string": "func GoAlpha()",
            "new_string": "func GoBeta() int { return 9 }\n\nfunc GoAlpha()",
        },
    )
    py_ok = "No definitions found" not in _find(reg, "PyBeta")
    go_ok = "No definitions found" not in _find(reg, "GoBeta")
    assert py_ok == go_ok is True, f"asymmetry: python={py_ok} nonpython={go_ok}"


@pytest.mark.parametrize(
    "fname,body,symbol",
    [
        ("bash_made.py", "def BashPySym():\\n    return 1\\n", "BashPySym"),
        ("bash_made.go", "package main\\n\\nfunc BashGoSym() int { return 1 }\\n", "BashGoSym"),
    ],
)
def test_bash_created_file_is_visible_to_find_symbol(multilang_repo, fname, body, symbol):
    """bash is a first-class write path and must invalidate like the tools do."""
    reg, d = multilang_repo
    res = reg.dispatch("bash", {"command": f"printf '{body}' > {fname}"})
    assert res.ok, res.error
    assert (d / fname).exists()
    assert "No definitions found" not in _find(reg, symbol), f"{fname} created by bash is invisible to find_symbol"


def test_readonly_bash_does_not_invalidate(multilang_repo):
    """Guard the other direction: over-invalidating would gut the hit rate.

    The dispatch comment is explicit that read-only bash must leave the cache
    intact "because the model interleaves such commands between edits", so a
    fix that simply cleared on every bash would be a regression wearing a
    correctness costume.
    """
    reg, _ = multilang_repo
    reg.dispatch("read_file", {"path": "a.py"})
    reg.dispatch("bash", {"command": "ls -la"})
    again = reg.dispatch("read_file", {"path": "a.py"})
    assert bool((again.metadata or {}).get("cache_hit")) is True, "read-only bash cleared the read cache"


def test_mutating_bash_drops_the_facade_rg_graph(multilang_repo):
    """B1: unknown-scope (bash) invalidation must drop the facade's RG graph.

    ``_invalidate_caches_unknown_scope`` cleared the CallGraphIndexer
    (``cgi.invalidate()``) but never ``self._call_graph.invalidate()`` — the
    RepositoryGraph held inside the facade is a SEPARATE build serving
    get_symbol / get_importers / get_file_dependencies / get_symbols_in_file.
    A warm RG graph therefore kept serving pre-bash state, i.e. the exact
    "cannot find a symbol in code it just wrote" class the wholesale drop
    exists for. The write-tool path has always been symmetric
    (``_call_graph.invalidate_files``); this restores the unknown-scope mirror.
    """
    reg, d = multilang_repo
    # Warm the facade RG graph so a stale read is possible (the bug is
    # invisible on a cold graph, which rebuilds lazily on the next query).
    assert reg._call_graph.get_symbol("py_alpha") is not None, "precondition: RG graph should serve existing symbols"
    assert reg._call_graph._graph is not None, "precondition: RG graph should be warm"

    res = reg.dispatch("bash", {"command": "printf 'def BashGsgSym():\\n    return 1\\n' > bash_gsg.py"})
    assert res.ok, res.error
    assert (d / "bash_gsg.py").exists()
    assert reg._call_graph._graph is None, (
        "mutating bash left the facade RG graph warm — RG-backed queries served pre-bash state (B1)"
    )
    # The next RG-backed query rebuilds lazily and must see the new symbol.
    node = reg._call_graph.get_symbol("BashGsgSym")
    assert node is not None and node.name == "BashGsgSym", "RG get_symbol cannot see a symbol created by mutating bash"
