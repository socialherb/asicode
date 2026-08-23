"""A hunk with no context lines is placed on trust — say so.

``_verify_c0_placement`` skips such hunks by design ("nothing to verify
against"), but the exposure is wider than that method's name implies: a hunk
carrying no context has nothing to match on ANY strategy, so it is placed by
line number by the FIRST ``git apply`` in the ladder and never reaches the
``-C0`` path the verifier guards.

Nothing downstream catches a stale line number either. Measured before this
change: ``@@ -8,0 +9,2 @@`` against

    def beta():
        c = 3
        return c

landed the two added statements after the ``return`` — unreachable code that
compiles cleanly, so the post-write syntax gate passed it and apply_patch
returned ok=True with no signal at all.

The behaviour is to REPORT, not refuse: context-free hunks are legitimate
``diff -U0`` output, and refusing would break working callers to catch a
malformed minority. The agent is the last thing that can notice a misplaced
insert, so it is told which hunks were unverifiable.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.patch_engine import PatchEngine

SRC = "def alpha():\n    a = 1\n    b = 2\n    return a + b\n\n\ndef beta():\n    c = 3\n    return c\n"

CTX_FREE_INSERT = "--- a/m.py\n+++ b/m.py\n@@ -8,0 +9,2 @@\n+    x = 99\n+    y = 100\n"
NORMAL_HUNK = (
    "--- a/m.py\n+++ b/m.py\n@@ -1,4 +1,4 @@\n def alpha():\n     a = 1\n-    b = 2\n+    b = 22\n     return a + b\n"
)


# ── detector ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name,patch,expected",
    [
        ("pure insert", CTX_FREE_INSERT, ["m.py @@ -8,0 +9,2 @@"]),
        ("with context", NORMAL_HUNK, []),
        (
            "delete-only, no context",
            "--- a/m.py\n+++ b/m.py\n@@ -3,2 +2,0 @@\n-    x = 1\n-    y = 2\n",
            ["m.py @@ -3,2 +2,0 @@"],
        ),
        # A brand-new file has no prior content to be misplaced against.
        ("new file excluded", "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,2 @@\n+a = 1\n+b = 2\n", []),
        (
            "mixed: only the bare one is flagged",
            NORMAL_HUNK + "@@ -8,0 +9,1 @@\n+    z = 3\n",
            ["m.py @@ -8,0 +9,1 @@"],
        ),
        ("empty", "", []),
    ],
)
def test_context_free_hunk_detection(name, patch, expected):
    assert PatchEngine.context_free_hunks(patch) == expected, name


# ── end to end through dispatch ─────────────────────────────────────────────
@pytest.fixture
def repo():
    d = Path(tempfile.mkdtemp(prefix="asr-unverifiable-"))
    subprocess.run(["git", "init", "-q", "."], cwd=d, check=True)
    (d / "m.py").write_text(SRC, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=d, check=True)
    return d


def _apply(repo: Path, patch: str):
    (repo / "m.py").write_text(SRC, encoding="utf-8")
    reg = ToolRegistry(str(repo), AgentConfig())
    return reg.dispatch("apply_patch", {"patch": patch})


def test_context_free_hunk_is_reported_on_success(repo):
    res = _apply(repo, CTX_FREE_INSERT)
    assert res.ok, res.error
    assert (res.metadata or {}).get("unverifiable_hunks") == ["m.py @@ -8,0 +9,2 @@"]
    assert "NOTE:" in (res.content or "")
    assert "could NOT be verified" in (res.content or "")


def test_context_free_hunk_still_applies(repo):
    """Reporting must not turn into refusing — the patch is still applied."""
    res = _apply(repo, CTX_FREE_INSERT)
    assert res.ok
    assert "x = 99" in (repo / "m.py").read_text(encoding="utf-8")


def test_normal_hunk_gets_no_notice(repo):
    """No noise on the common path, or the notice stops meaning anything."""
    res = _apply(repo, NORMAL_HUNK)
    assert res.ok, res.error
    assert (res.metadata or {}).get("unverifiable_hunks") is None
    assert "NOTE:" not in (res.content or "")
    assert "b = 22" in (repo / "m.py").read_text(encoding="utf-8")


def test_failed_apply_gets_no_notice(repo):
    """A refusal already carries its own error; the notice would only confuse."""
    bogus = "--- a/m.py\n+++ b/m.py\n@@ -1,3 +1,3 @@\n def nonexistent():\n-    zzz = 1\n+    zzz = 2\n     qqq = 3\n"
    res = _apply(repo, bogus)
    if res.ok:
        pytest.skip("tolerant ladder applied the bogus patch; not this test's subject")
    assert "unverifiable_hunks" not in (res.metadata or {})


def test_notice_is_attached_regardless_of_which_success_path_ran(repo):
    """The wrapper covers every ok=True return, not a hand-picked few.

    _tool_apply_patch has several success returns; this repo has twice shipped
    a guard wired into only some of a family's call sites, so the notice is
    attached once after the fact rather than at each return.
    """
    import external_llm.agent.tool_handlers.write_tools as wt
    from external_llm.agent.tool_registry import ToolResult

    reg = ToolRegistry(str(repo), AgentConfig())
    # Force an arbitrary alternative success shape out of the implementation.
    reg._tool_apply_patch_impl = lambda args: ToolResult(  # type: ignore[method-assign]
        ok=True,
        content="applied via some other branch",
        error=None,
    )
    assert hasattr(wt.WriteToolsMixin, "_tool_apply_patch_impl")
    res = reg._tool_apply_patch({"patch": CTX_FREE_INSERT})
    assert (res.metadata or {}).get("unverifiable_hunks") == ["m.py @@ -8,0 +9,2 @@"]
