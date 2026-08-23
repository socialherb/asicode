"""No write tool may write outside repo_root.

``_secure_path``'s docstring states the contract directly: ``confine=True``
"forces the repo-boundary check to run REGARDLESS of ``unrestricted_read``…
so writes can never escape repo_root even on a trusted CLI — the unrestricted
flag is a READ capability only, never a write capability."

Three of the seven write tools did not honour it, because they never called
``_secure_path`` at all: ``edit_text``, ``anchor_edit`` and ``edit_file``
resolved their target with a bare ``Path(repo_root) / file_path``, which walks
straight out of the repo on a leading ``../`` and ignores repo_root entirely on
an absolute path. ``modify_symbol``, ``edit_ast``, ``write_plan`` and
``apply_patch`` were already guarded.

Three things made this worse than a stray-file bug:

* it did NOT depend on the trust flag — ``unrestricted_read`` defaults to
  False and the escape happened anyway, so "the CLI trusts its user" does not
  explain it;
* that default is the webapp's configuration, which the same docstring flags as
  having an attacker-controlled repo_root;
* ``edit_text`` and ``anchor_edit`` are exposed to the model in
  ``tool_schemas``, so a ``../`` in a model-emitted path was enough to reach it.

The logs actively misled: a run would print "Path traversal attempt blocked"
from a *different*, guarded tool while the unguarded ones wrote through without
reaching the check at all.

Both directions are asserted. Over-blocking would be its own outage, so the
in-repo cases (absolute-inside-repo, nested, ``./x``, and ``sub/../x`` which
normalises back inside) must keep working.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

GOOD_SRC = "def alpha():\n    return 1\n"
OUTSIDE_MARK = "ORIGINAL OUTSIDE\n"
PWNED = "PWNED"


@pytest.fixture
def sandbox():
    """repo/ plus a sibling outside/ holding the file an escape would clobber."""
    base = Path(tempfile.mkdtemp(prefix="asr-boundary-"))
    repo = base / "repo"
    (repo / "sub" / "deep").mkdir(parents=True)
    (repo / "mod.py").write_text(GOOD_SRC, encoding="utf-8")
    (repo / "sub" / "deep" / "x.py").write_text(GOOD_SRC, encoding="utf-8")
    outside = base / "outside"
    outside.mkdir()
    victim = outside / "victim.txt"
    victim.write_text(OUTSIDE_MARK, encoding="utf-8")
    cfg = AgentConfig()
    # Pin the premise: the escape is NOT explained by the trust flag.
    assert cfg.unrestricted_read is False
    return ToolRegistry(str(repo), cfg), repo, victim


def _escape_args(target: str):
    """One entry per write tool, each aimed at *target* (outside the repo)."""
    return [
        ("edit_text", {"file_path": target, "old_string": "ORIGINAL", "new_string": PWNED}),
        (
            "anchor_edit",
            {"file_path": target, "anchor_pattern": "ORIGINAL", "edit_mode": "replace_line", "code_snippet": PWNED},
        ),
        ("edit_file", {"path": target, "operations": [{"type": "replace", "anchor": "ORIGINAL", "content": PWNED}]}),
        # `file_path` too: raw-argument recovery can populate it instead of `path`.
        (
            "edit_file",
            {"file_path": target, "operations": [{"type": "replace", "anchor": "ORIGINAL", "content": PWNED}]},
        ),
        ("edit_ast", {"file_path": target, "operation": "replace", "target": "x", "code": PWNED}),
        ("modify_symbol", {"file_path": target, "symbol": "x", "code": PWNED}),
        ("write_plan", {"plan": {"ops": [{"type": "create_file", "path": target, "content": PWNED + "\n"}]}}),
        ("apply_patch", {"patch": f"--- a/{target}\n+++ b/{target}\n@@ -1 +1 @@\n-ORIGINAL OUTSIDE\n+{PWNED}\n"}),
    ]


@pytest.mark.parametrize("traversal", [True, False], ids=["relative-..", "absolute"])
def test_no_write_tool_escapes_repo_root(sandbox, traversal):
    reg, _repo, victim = sandbox
    target = "../outside/victim.txt" if traversal else str(victim)
    escaped = []
    for tool, args in _escape_args(target):
        victim.write_text(OUTSIDE_MARK, encoding="utf-8")
        # a raise is an acceptable refusal; only the file state matters
        with contextlib.suppress(Exception):
            reg.dispatch(tool, dict(args))
        if PWNED in victim.read_text(encoding="utf-8"):
            escaped.append(tool)
    assert not escaped, f"wrote outside repo_root via: {sorted(set(escaped))}"


def test_escape_does_not_depend_on_the_trust_flag(sandbox):
    """Even with unrestricted_read=True the write must stay confined.

    That flag is documented as a READ capability. If a future change routes the
    write guard through the flag-respecting default mode, this fails.
    """
    _, repo, victim = sandbox
    reg = ToolRegistry(str(repo), AgentConfig(unrestricted_read=True))
    victim.write_text(OUTSIDE_MARK, encoding="utf-8")
    reg.dispatch("edit_text", {"file_path": "../outside/victim.txt", "old_string": "ORIGINAL", "new_string": PWNED})
    assert PWNED not in victim.read_text(encoding="utf-8"), (
        "unrestricted_read must not grant WRITE access outside repo_root"
    )


@pytest.mark.parametrize(
    "rel,target",
    [
        ("mod.py", "mod.py"),
        ("./mod.py", "mod.py"),
        ("sub/../mod.py", "mod.py"),  # normalises back inside the repo
        ("sub/deep/x.py", "sub/deep/x.py"),
    ],
)
def test_in_repo_paths_still_write(sandbox, rel, target):
    """Guard the other direction — over-blocking would break ordinary edits."""
    reg, repo, _ = sandbox
    (repo / target).write_text(GOOD_SRC, encoding="utf-8")
    res = reg.dispatch("edit_text", {"file_path": rel, "old_string": "return 1", "new_string": "return 42"})
    assert res.ok, res.error
    assert "return 42" in (repo / target).read_text(encoding="utf-8")


def test_absolute_path_inside_repo_still_writes(sandbox):
    reg, repo, _ = sandbox
    (repo / "mod.py").write_text(GOOD_SRC, encoding="utf-8")
    res = reg.dispatch(
        "edit_text", {"file_path": str(repo / "mod.py"), "old_string": "return 1", "new_string": "return 42"}
    )
    assert res.ok, res.error
    assert "return 42" in (repo / "mod.py").read_text(encoding="utf-8")


def test_every_write_tool_is_covered(sandbox):
    """The escape list must not silently drift from _WRITE_TOOLS.

    A tool added later without a boundary check would otherwise be untested.
    """
    reg, _, _ = sandbox
    covered = {tool for tool, _ in _escape_args("x")}
    missing = set(reg._WRITE_TOOLS) - covered
    assert not missing, f"write tools with no boundary test: {sorted(missing)}"
