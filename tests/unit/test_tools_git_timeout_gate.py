"""Gate: every git invocation in tools/ scripts must be timeout-bounded.

27th audit round found git subprocess calls without ``timeout=`` (including
``git checkout -- .`` / ``git clean -fd``) plus os.popen/os.system git calls
in tools/bench_strategy_ladder.py that cannot be timeout-bounded at all. A
hung git (index.lock contention, network FS) would block a bench run forever
and leave the repo mid-restore.

Coverage widened after the auto_learning_loop hang: the git-only rule let a
bare ``subprocess.run`` to the monitor (an agent-spawning, 10-minute-class
child) through ungated — the loop then blocked forever when the monitor hung.
Every blocking run-family call in tools/ is now bounded, not just the git ones.

This gate scans tools/*.py statically:
  1. every ``subprocess.run/check_output/check_call/call`` whose first arg
     is ``git`` must pass ``timeout=``;
  2. ``os.popen``/``os.system`` must never be used for git commands (they
     have no timeout mechanism).
  3. every ``subprocess.run/check_output/check_call/call`` (any command)
     must pass ``timeout=`` — long-running launches may satisfy this via the
     shared ``run_bounded_subprocess`` helper instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.subprocess_gate_detectors import run_family_without_timeout

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"

_SUBPROCESS_FUNCS = {"run", "check_output", "check_call", "call"}


def _tools_py_files() -> list[Path]:
    return sorted(TOOLS_DIR.glob("*.py"))


def _subprocess_aliases(tree: ast.AST) -> set[str]:
    """Names bound to the subprocess module (incl. ``import subprocess as _sp``)."""
    aliases = {"subprocess"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "subprocess":
                    aliases.add(a.asname or "subprocess")
    return aliases


def _first_arg_is_git(call: ast.Call) -> bool:
    """True if the first positional arg is ``git`` or a list starting with ``git``."""
    if not call.args:
        return False
    first = call.args[0]
    if isinstance(first, ast.Constant) and first.value == "git":
        return True
    if isinstance(first, ast.List) and first.elts:
        first_el = first.elts[0]
        return isinstance(first_el, ast.Constant) and first_el.value == "git"
    return False


def _has_timeout_kw(call: ast.Call) -> bool:
    return any(kw.arg == "timeout" for kw in call.keywords)


def _git_subprocess_calls(tree: ast.AST) -> list[tuple[ast.Call, int]]:
    """(call, lineno) for subprocess.* git calls lacking timeout=."""
    aliases = _subprocess_aliases(tree)
    hits: list[tuple[ast.Call, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id in aliases):
            continue
        if func.attr not in _SUBPROCESS_FUNCS:
            continue
        if not _first_arg_is_git(node):
            continue
        if not _has_timeout_kw(node):
            hits.append((node, node.lineno))
    return hits


def _git_popen_system_calls(tree: ast.AST) -> list[tuple[ast.Call, int]]:
    """(call, lineno) for os.popen/os.system invocations of git."""
    hits: list[tuple[ast.Call, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "os"):
            continue
        if func.attr not in {"popen", "system"}:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and "git" in first.value:
            hits.append((node, node.lineno))
        elif isinstance(first, ast.JoinedStr) and "git" in ast.unparse(first):
            # f'cd "..." && git ...' — f-strings are JoinedStr, not Constant
            hits.append((node, node.lineno))
    return hits


@pytest.mark.parametrize("path", [str(p) for p in _tools_py_files()], ids=[p.name for p in _tools_py_files()])
def test_all_subprocess_calls_have_timeout(path: str):
    """Every blocking subprocess call in tools/ must pass timeout= — not just git ones."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    missing = run_family_without_timeout(tree)
    assert not missing, (
        f"{Path(path).name}: {len(missing)} subprocess call(s) without timeout= at "
        f"lines {[ln for _, ln in missing]} — pass timeout= or use run_bounded_subprocess"
    )


@pytest.mark.parametrize("path", [str(p) for p in _tools_py_files()], ids=[p.name for p in _tools_py_files()])
def test_git_subprocess_calls_have_timeout(path: str):
    """subprocess git calls must pass timeout= (hung git must not block a bench)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    missing = _git_subprocess_calls(tree)
    assert not missing, (
        f"{Path(path).name}: {len(missing)} git subprocess call(s) without timeout= at "
        f"lines {[ln for _, ln in missing]}"
    )


@pytest.mark.parametrize("path", [str(p) for p in _tools_py_files()], ids=[p.name for p in _tools_py_files()])
def test_no_git_via_os_popen_system(path: str):
    """os.popen/os.system have no timeout mechanism — git must not use them."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    hits = _git_popen_system_calls(tree)
    assert not hits, (
        f"{Path(path).name}: {len(hits)} git call(s) via os.popen/os.system at "
        f"lines {[ln for _, ln in hits]} — convert to subprocess.run(timeout=...)"
    )
