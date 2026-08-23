"""Shared AST detectors for the subprocess timeout gates.

Both ``test_external_llm_subprocess_gate.py`` and
``test_webapp_subprocess_gate.py`` enforce the same three rules over
their respective production trees:

   1. every ``subprocess.run/check_output/check_call/call`` must pass ``timeout=``;
   2. ``os.popen``/``os.system`` are banned outright (no timeout mechanism);
   3. ``subprocess.Popen`` with a visible ``stdout=``/``stderr=subprocess.PIPE``
      must be bounded in the same function: ``proc.wait(timeout=...)`` /
      ``proc.communicate(timeout=...)``, or the proc handed to another call
      (delegated bound, e.g. ``_capture_bounded(proc, timeout, ...)``).
      Popen without PIPE (DEVNULL / log file) is fire-and-forget, and a Popen
      with a ``**kwargs`` splat cannot be inspected statically — both exempt.

``_subprocess_aliases`` tracks ``import subprocess [as x]`` AND plain
assignment aliases (``_sp = subprocess``) to fixed point, so call sites using
either spelling are inspected.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SUBPROCESS_FUNCS = {"run", "check_output", "check_call", "call"}
_DEFAULT_SKIP_PARTS = {"tests", "__pycache__"}


def prod_py_files(root: Path, skip_parts: set[str] | None = None) -> list[Path]:
    """All ``*.py`` files under ``root`` excluding test/cache directories."""
    skip = _DEFAULT_SKIP_PARTS if skip_parts is None else skip_parts
    return sorted(p for p in root.rglob("*.py") if not any(part in skip for part in p.parts))


def _subprocess_aliases(tree: ast.AST) -> set[str]:
    """Names bound to the subprocess module, to fixed point.

    Covers ``import subprocess``, ``import subprocess as _sp``, and assignment
    aliases such as ``_sp = subprocess`` (seen in webapp/routes/edit_run.py) —
    including chains (``a = subprocess; b = a``).
    """
    aliases: set[str] = {"subprocess"}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "subprocess":
                        alias = a.asname or "subprocess"
                        if alias not in aliases:
                            aliases.add(alias)
                            changed = True
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                # _sp = subprocess  (value may itself be an alias, e.g. b = a)
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id not in aliases:
                        aliases.add(t.id)
                        changed = True
    return aliases


def _has_timeout_kw(call: ast.Call) -> bool:
    return any(kw.arg == "timeout" for kw in call.keywords)


def run_family_without_timeout(tree: ast.AST) -> list[tuple[ast.Call, int]]:
    """(call, lineno) for subprocess.{run,check_output,check_call,call} lacking timeout=."""
    aliases = _subprocess_aliases(tree)
    hits: list[tuple[ast.Call, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id in aliases):
            continue
        if node.func.attr in _SUBPROCESS_FUNCS and not _has_timeout_kw(node):
            hits.append((node, node.lineno))
    return hits


def os_popen_system_calls(tree: ast.AST) -> list[tuple[ast.Call, int]]:
    """(call, lineno) for any os.popen/os.system call."""
    hits: list[tuple[ast.Call, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "os"):
            continue
        if node.func.attr in {"popen", "system"}:
            hits.append((node, node.lineno))
    return hits


def _is_popen_call(node: ast.Call, aliases: set[str]) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "Popen"
        and isinstance(func.value, ast.Name)
        and func.value.id in aliases
    )


def _is_pipe_expr(val: ast.AST) -> bool:
    if isinstance(val, ast.Attribute) and val.attr == "PIPE":
        return True
    return isinstance(val, ast.Name) and val.id == "PIPE"


def _has_pipe(node: ast.Call) -> bool:
    """True if a visible stdout=/stderr= (kwarg or positional) is subprocess.PIPE."""
    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
    args = node.args
    for idx, kw_name in enumerate(("stdout", "stderr"), start=2):
        val = kwargs.get(kw_name)
        if val is None and len(args) > idx:
            val = args[idx]  # Popen positional: (args, bufsize, executable, stdin, stdout, stderr)
        if val is not None and _is_pipe_expr(val):
            return True
    return False


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    cur = node
    while cur is not None:
        parent = parents.get(cur)
        if parent is None:
            return None
        # Module is the top-level scope: module-level Popen (self-test snippets,
        # unusual production shapes) is still bounded-checkable within it.
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            return parent
        cur = parent
    return None


def _assigned_var(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> str | None:
    """Name the Popen call is assigned to (``proc = subprocess.Popen(...)``), if any.

    Accepts both plain assignment (``proc = Popen(...)``) and annotated
    assignment (``proc: Popen[str] = Popen(...)`` — the pyright-driven typing
    pattern used in tool_handlers/read_tools.py). The annotation form parses
    as an :class:`ast.AnnAssign`, not :class:`ast.Assign`; ignoring it would
    false-positive on a call that IS bound.
    """
    parent = parents.get(node)
    if (
        isinstance(parent, ast.Assign)
        and parent.value is node
        and len(parent.targets) == 1
        and isinstance(parent.targets[0], ast.Name)
    ):
        return parent.targets[0].id
    if isinstance(parent, ast.AnnAssign) and parent.value is node and isinstance(parent.target, ast.Name):
        return parent.target.id
    return None


def _bounded_in_scope(fn: ast.AST, var: str) -> bool:
    """True if fn contains ``var.wait/communicate(timeout=...)`` or hands ``var`` to a call."""
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == var
            and func.attr in {"wait", "communicate"}
            and _has_timeout_kw(sub)
        ):
            return True
        # Handoff to another call (delegated bound, e.g. _capture_bounded(proc, ...)).
        for arg in sub.args:
            if isinstance(arg, ast.Name) and arg.id == var:
                return True
        for kw in sub.keywords:
            if kw.arg and isinstance(kw.value, ast.Name) and kw.value.id == var:
                return True
    return False


def popen_unbounded(tree: ast.AST) -> list[tuple[ast.Call, int, str]]:
    """(call, lineno, reason) for Popen-with-PIPE lacking a same-scope bound."""
    aliases = _subprocess_aliases(tree)
    parents = _parent_map(tree)
    hits: list[tuple[ast.Call, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_popen_call(node, aliases) or not _has_pipe(node):
            continue
        if any(kw.arg is None for kw in node.keywords):
            continue  # **kwargs splat — pipes not statically visible (fire-and-forget by design)
        fn = _enclosing_function(node, parents)
        var = _assigned_var(node, parents)
        if fn is None or var is None:
            hits.append((node, node.lineno, "Popen with PIPE not assigned to a name (and no enclosing scope)"))
            continue
        if not _bounded_in_scope(fn, var):
            hits.append((node, node.lineno, f"proc '{var}' has no wait/communicate(timeout=) nor handoff in scope"))
    return hits


def all_violations(tree: ast.AST) -> dict[str, list]:
    """Convenience for tests: every detector keyed by rule name."""
    return {
        "run": run_family_without_timeout(tree),
        "os": os_popen_system_calls(tree),
        "popen": popen_unbounded(tree),
    }


def detector_self_tests() -> None:
    """Mutation-style sensitivity checks; raises AssertionError when a detector is broken.

    Called by each gate file so the gates cannot pass vacuously: if a detector
    silently stopped firing, the compliance parametrizations would still pass
    on the (clean) production trees, so this pins each detector's behaviour on
    synthetic violations AND safe forms.
    """

    def hits(src: str) -> dict[str, list]:
        return all_violations(ast.parse(src))

    # run-family: violation fires, timeout form quiet
    assert len(hits("import subprocess\nsubprocess.run(['ls'])\n")["run"]) == 1
    assert hits("import subprocess\nsubprocess.run(['ls'], timeout=30)\n")["run"] == []
    # alias via import-as
    assert len(hits("import subprocess as _sp\n_sp.check_output(['x'])\n")["run"]) == 1
    # alias via plain assignment (webapp/routes/edit_run.py pattern), incl. chain
    assert len(hits("import subprocess\n_sp = subprocess\n_sp.check_output(['x'])\n")["run"]) == 1
    assert len(hits("import subprocess\n_sp = subprocess\n_sp.check_output(['x'], timeout=5)\n")["run"]) == 0
    assert len(hits("import subprocess\na = subprocess\nb = a\nb.run(['x'])\n")["run"]) == 1
    # Popen: plain wait() fires, wait(timeout=) quiet, handoff quiet, DEVNULL/splat exempt
    assert (
        len(hits("import subprocess\nproc = subprocess.Popen(['x'], stdout=subprocess.PIPE)\nproc.wait()\n")["popen"])
        == 1
    )
    assert (
        hits("import subprocess\nproc = subprocess.Popen(['x'], stdout=subprocess.PIPE)\nproc.wait(timeout=5)\n")[
            "popen"
        ]
        == []
    )
    assert (
        hits("import subprocess\nproc = subprocess.Popen(['x'], stdout=subprocess.PIPE)\ncapture(proc, timeout=5)\n")[
            "popen"
        ]
        == []
    )
    assert hits("import subprocess\nsubprocess.Popen(['x'], stdout=subprocess.DEVNULL)\n")["popen"] == []
    assert hits("import subprocess\nproc = subprocess.Popen(['x'], **_kw)\n")["popen"] == []
    # Annotated assignment (AnnAssign): the pyright typing pattern must be
    # recognised as a binding — unbound annotated form still fires.
    assert (
        hits(
            "import subprocess\nproc: subprocess.Popen[str] = subprocess.Popen(['x'], stdout=subprocess.PIPE)\nproc.wait(timeout=5)\n"
        )["popen"]
        == []
    )
    assert (
        len(
            hits("import subprocess\nproc: subprocess.Popen[str] = subprocess.Popen(['x'], stdout=subprocess.PIPE)\n")[
                "popen"
            ]
        )
        == 1
    )
    # os.popen/os.system banned
    assert len(hits("import os\nos.system('git status')\n")["os"]) == 1
    assert len(hits("import os\nos.popen('ls')\n")["os"]) == 1
