#!/usr/bin/env python3
"""Check no NEW ``subprocess`` call is left unguarded against a missing binary.

A subprocess call is *unguarded* when no enclosing ``try`` whose ``except``
clause catches ``OSError`` (or a supertype: ``Exception`` / ``BaseException`` /
``IOError`` / ``EnvironmentError`` / bare ``except``) covers it.

Why OSError specifically
-----------------------
A missing external binary raises ``FileNotFoundError`` — a subclass of
``OSError``.  Catching ``subprocess.SubprocessError`` does **not** help: that
hierarchy covers ``CalledProcessError`` (non-zero exit) and ``TimeoutExpired``,
*not* the binary being absent.  The motivating incident was exactly this gap:
``symbol_search.py`` invoked ``ripgrep`` inside a ``try`` that caught only
``(AttributeError, TypeError)`` / ``subprocess.SubprocessError``, so on a plain
``pip install asicode`` (where ``rg`` is an *optional* ``[search]`` extra per
``pyproject.toml``) every default ``find_symbol``/``read_symbol`` call crashed
with ``FileNotFoundError``.  Three call sites had the defect; two correctly
neighbouring sites used a ``shutil.which("rg")`` guard and were unaffected.

This gate makes the class of bug un-shippable: a new ``subprocess.run``/
``Popen``/``call``/``check_call``/``check_output`` that is not OSError-guarded
fails the build.  It does NOT touch the 2 known pre-existing sites (frozen as a
baseline): ``run_bounded_subprocess`` is a thin wrapper whose *caller* owns the
error contract, and ``TestRunner._run_cmd`` spawns pytest/npm whose absence is
the test-runner's own concern.  Exactly mirrors the F401/F811/F821/silent-except
baseline-diff philosophy (see ``.pre-commit-config.yaml``).

Key: ``<rel_path>::<scope_qualname>::<ordinal>``

    scope_qualname = dotted names of enclosing FunctionDef/AsyncFunctionDef/
                     ClassDef, with a leading ``<module>`` segment.
    ordinal        = 0-based index of this subprocess call among ALL subprocess
                     calls in the SAME nearest scope, in source order.

Stable under line drift — inserting/deleting lines above a call, or adding
ordinary statements to its function, leaves the key unchanged.  Adding a new
subprocess call (guarded or not) *above* a baselined unguarded one shifts its
ordinal; that is acceptable churn absorbed by re-running ``--write-baseline``.

Scope: production code only — the ``external_llm/``, ``services/`` and
``webapp/`` trees plus EVERY root-level ``*.py``.  Root modules are covered as
a group rather than by naming ``asi.py``: all nine of them (``common.py``,
``config.py``, ``path_security.py``, ``radio.py``, …) ship in the public wheel
exactly like ``asi.py`` does, so singling one out left the other eight able to
introduce an unguarded call unblocked.  ``scripts/`` (these very gate scripts
call ruff/git unguarded — meaningless to guard under CI where they are
guaranteed present) and ``tests/`` are excluded.

Usage:
    python scripts/check_no_new_unguarded_subprocess.py            # check for new
    python scripts/check_no_new_unguarded_subprocess.py <file>.py ...  # given files
    python scripts/check_no_new_unguarded_subprocess.py --write-baseline  # regen

Explicit file args (pre-commit per-file mode) scan only those files — the
full-repo always_run scans were dropped from the hook config because they
created a multi-second window where pre-commit's run-start `git diff` vs
post-hook diff comparison false-positives on parallel-session writes.  No
args (lint.yml CI) still scans the whole repo.  Files outside the scan scope
(tests/, scripts/) are skipped, exactly like the full scan.
"""

import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "unguarded_subprocess_baseline.txt"

_SUBPROCESS_FUNCS = frozenset({
    "run", "Popen", "call", "check_call", "check_output",
})

# Exception type NAMES that catch FileNotFoundError (OSError or a supertype).
# ``subprocess.SubprocessError`` / ``CalledProcessError`` / ``TimeoutExpired``
# are deliberately NOT here — they cover exit/timeout, not a missing binary.
_OSERROR_SUPERTYPES = frozenset({
    "OSError", "IOError", "EnvironmentError", "WindowsError",
    "FileNotFoundError", "Exception", "BaseException",
})

_SCAN_ROOTS = ("external_llm", "services", "webapp")
_SKIP_DIRS = frozenset({
    "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules",
    ".venv", "venv", "env", ".tox", "dist", "build", ".eggs", ".git",
})


def _subprocess_aliases(tree: ast.AST) -> set[str]:
    """Module names bound to the ``subprocess`` module anywhere in the file.

    Covers ``import subprocess`` and local aliases ``import subprocess as _sp``
    used in this repo (``from subprocess import X`` is not used anywhere).
    """
    aliases: set[str] = {"subprocess"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "subprocess":
                    aliases.add(a.asname or "subprocess")
    return aliases


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _handler_catches_oserror(handler: ast.ExceptHandler) -> bool:
    """True iff *handler*'s type clause catches ``OSError`` or a supertype.

    Bare ``except:`` (``type is None``) catches everything.  ``except (A, B):``
    is a Tuple; any element in :data:`_OSERROR_SUPERTYPES` qualifies.
    """
    t = handler.type
    if t is None:
        return True
    elts = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(e, ast.Name) and e.id in _OSERROR_SUPERTYPES for e in elts)


def _is_suppress_call(node: ast.Call) -> bool:
    """True iff *node* is ``contextlib.suppress(...)`` or a bare ``suppress`` import.

    The sibling silent-except program converts ``try/except X: pass`` into
    ``with contextlib.suppress(X):`` (SIM105 is selected repo-wide), so the
    subprocess guard must recognize both guard shapes or the two gates fight
    each other: every suppress-wrapped subprocess call would look "unguarded".
    """
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr == "suppress" and isinstance(f.value, ast.Name) and f.value.id == "contextlib"
    return isinstance(f, ast.Name) and f.id == "suppress"


def _type_expr_catches_oserror(t) -> bool:
    """True iff type expression *t* names OSError or a supertype.

    Mirrors :func:`_handler_catches_oserror` for ``suppress(...)`` arguments,
    which are plain expressions rather than ExceptHandlers.
    """
    if t is None:
        return True
    elts = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(e, ast.Name) and e.id in _OSERROR_SUPERTYPES for e in elts)


def _is_oserror_guarded(call: ast.Call, parents: dict[int, ast.AST]) -> bool:
    """True iff an enclosing ``try``/``suppress`` whose body contains *call* catches OSError.

    Walks parent links upward.  At each enclosing ``Try`` where the call sits in
    ``try.body`` (the guarded region — NOT ``else``/``finally``/a ``handler``
    body), that Try's handlers apply.  If none catch OSError the exception
    propagates out, so we keep checking outer ancestors.  The same walk treats
    ``with contextlib.suppress(OSError, ...)`` as an equivalent guard (see
    :func:`_is_suppress_call`).
    """
    child: ast.AST = call
    while True:
        parent = parents.get(id(child))
        if parent is None:
            return False
        if (
            isinstance(parent, ast.Try)
            and any(child is s for s in parent.body)
            and any(_handler_catches_oserror(h) for h in parent.handlers)
        ):
            return True
        if isinstance(parent, ast.With) and any(child is s for s in parent.body):
            for item in parent.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and _is_suppress_call(ctx) and any(
                    _type_expr_catches_oserror(a) for a in ctx.args
                ):
                    return True
        # listed types miss OSError -> propagates past this try; keep going
        child = parent


def _is_subprocess_call(node: ast.Call, aliases: set[str]) -> bool:
    f = node.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr in _SUBPROCESS_FUNCS
        and isinstance(f.value, ast.Name)
        and f.value.id in aliases
    )


class _UnguardedSubprocessScanner(ast.NodeVisitor):
    """Collect drift-stable baseline keys for unguarded subprocess calls."""

    def __init__(self, rel: str, parents: dict[int, ast.AST], aliases: set[str]) -> None:
        self.rel = rel
        self.parents = parents
        self.aliases = aliases
        # [scope_name, call_counter]; module scope always present.
        self._scopes: list[list] = [["<module>", 0]]
        self.keys: list[str] = []

    def _qualname(self) -> str:
        return "::".join(s[0] for s in self._scopes)

    def _scope_visit(self, node) -> None:
        self._scopes.append([node.name, 0])
        self.generic_visit(node)
        self._scopes.pop()

    visit_FunctionDef = _scope_visit
    visit_AsyncFunctionDef = _scope_visit
    visit_ClassDef = _scope_visit

    def visit_Call(self, node: ast.Call) -> None:
        if _is_subprocess_call(node, self.aliases):
            scope = self._scopes[-1]
            scope[1] += 1
            if not _is_oserror_guarded(node, self.parents):
                self.keys.append(f"{self.rel}::{self._qualname()}::{scope[1] - 1}")
        self.generic_visit(node)


def _scan_source(src: str, rel: str) -> list[str]:
    """Pure analysis: return unguarded-subprocess keys for *src*.

    Factored out so detection is directly unit-testable with crafted strings
    (no temp files), mirroring the IO/analysis split in the sibling gate.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    parents = _build_parent_map(tree)
    aliases = _subprocess_aliases(tree)
    scanner = _UnguardedSubprocessScanner(rel, parents, aliases)
    scanner.visit(tree)
    return scanner.keys


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _resolve_scan_paths(args: list[str]) -> list[str] | None:
    """Normalize explicit file args to repo-relative ``*.py`` paths.

    Returns ``None`` when no file args survive (or none were given) — the
    caller then scans the whole repo, preserving the no-args (lint.yml CI)
    behaviour.  pre-commit passes absolute paths; lint.yml passes none — both
    normalize to the same repo-relative key space as the full scan.
    """
    out: list[str] = []
    for a in args:
        rel = os.path.relpath(Path(a).resolve(), Path(REPO).resolve())
        if rel.endswith(".py") and not rel.startswith(".."):
            out.append(rel)
    return out or None


def _iter_repo_py(paths: list[str] | None = None) -> list[Path]:
    if paths is not None:
        out: list[Path] = []
        for rel in paths:
            p = REPO / rel
            if not p.is_file() or p.suffix != ".py" or _should_skip(p):
                continue
            # Scope: SCAN_ROOTS trees + every root-level module, judged on the
            # REPO-RELATIVE path (an absolute path's parts[0] is "/" — see the
            # mirror predicate in the full scan; tests/ and scripts/ excluded).
            rp = Path(rel)
            if rp.parent == Path(".") or rp.parts[0] in _SCAN_ROOTS:
                out.append(p)
        return out
    full: list[Path] = []
    for root in _SCAN_ROOTS:
        d = REPO / root
        if d.is_dir():
            full.extend(p for p in d.rglob("*.py") if not _should_skip(p))
    # Every root-level module, not just asi.py — they all ship in the wheel.
    full.extend(sorted(p for p in REPO.glob("*.py") if p.is_file()))
    return full


def _scan_path(path: Path) -> list[str]:
    rel = str(path.relative_to(REPO))
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _scan_source(src, rel)


def _get_current_keys(paths: list[str] | None = None) -> set[str]:
    keys: set[str] = set()
    for p in _iter_repo_py(paths):
        keys.update(_scan_path(p))
    return keys


def _load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    out: set[str] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def _write_baseline(keys: set[str]) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# asicode unguarded-subprocess baseline — KNOWN pre-existing\n"
        "# subprocess.run/Popen/call/check_call/check_output calls that are NOT\n"
        "# inside an OSError-catching try/except.\n"
        "#\n"
        "# A missing external binary raises FileNotFoundError (an OSError).\n"
        "# Catching subprocess.SubprocessError does NOT cover it. These two sites\n"
        "# are intentional: run_bounded_subprocess defers errors to its caller;\n"
        "# TestRunner._run_cmd spawns pytest/npm whose absence is its own concern.\n"
        "#\n"
        "# Generated by `scripts/check_no_new_unguarded_subprocess.py --write-baseline`.\n"
        "# Hand-editing is allowed: sorted, one key per line, `#` comments ok.\n"
        "# Key: <rel_path>::<scope_qualname>::<ordinal>  (stable under line drift)\n"
        "#\n"
    )
    lines = sorted(keys)
    BASELINE.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    paths = _resolve_scan_paths([a for a in sys.argv[1:] if not a.startswith("--")])
    current = _get_current_keys(paths)
    if "--write-baseline" in sys.argv:
        _write_baseline(current)
        print(f"✅ Baseline written: {BASELINE} ({len(current)} entries)")
        return 0

    baseline = _load_baseline()
    new_keys = current - baseline

    if not new_keys:
        print(f"✅ No new unguarded subprocess calls ({len(current)} total, "
              f"{len(baseline)} baselined)")
        return 0

    print(f"❌ {len(new_keys)} NEW unguarded subprocess call(s) (not in baseline):\n")
    for k in sorted(new_keys):
        print(f"  {k}")
    print(f"\nTotal unguarded subprocess: {len(current)}, Baselined: {len(baseline)}")
    print(
        "\nA subprocess call to an external binary (rg/git/node/npm/…) raises"
        "\nFileNotFoundError (an OSError) when the binary is absent — which is the"
        "\ncase on a plain `pip install asicode` for every optional extra."
        "\nsubprocess.SubprocessError does NOT cover this."
        "\n\nWrap the call in try/except OSError (or add a shutil.which() guard),"
        "\nor run `--write-baseline` only if missing-binary is genuinely this"
        "\ncall's caller's responsibility (see the 2 baselined helpers)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
