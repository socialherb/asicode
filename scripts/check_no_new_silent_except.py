#!/usr/bin/env python3
"""Check no NEW silent ``except`` handlers vs baseline.

A *silent* except handler is one whose body does nothing but swallow the
exception — ``pass``, ``return None``, bare ``return``, ``continue``, or
``break`` — with no ``raise`` and no observability call (logger.* / print /
logging.exception).  Such handlers are the classic observability black hole:
this repo has a documented history (0.2.6 version_check) of a fail-open silent
disable causing a real incident, and in unattended 12h+ runs a swallowed
failure silently corrupts the self-improve loop's training signal.

This gate does NOT touch the ~1.7k pre-existing silent handlers — they are
frozen as a baseline.  It catches only NET-NEW ones, exactly mirroring the
existing F401/F811/F821 baseline-diff hooks (see ``.pre-commit-config.yaml``).

Key: ``<rel_path>::<scope_qualname>::<ordinal>``

    scope_qualname = dotted names of enclosing FunctionDef/AsyncFunctionDef/
                     ClassDef, or ``<module>`` at module level.
    ordinal        = 0-based index of this ExceptHandler among all ExceptHandlers
                     sharing the SAME nearest scope, in source order.

This key is **stable under line drift** — inserting/deleting lines above a
handler, or adding ordinary statements to its function, leaves the key
unchanged (unlike an F811-style ``file::line`` key, which would churn noisily
at the ~1.7k scale here).  Only genuinely new handlers (new ordinal) trip the
gate.  A handler that is moved to a different function or whose enclosing
function is renamed will appear as remove+add; that is acceptable churn and is
absorbed by re-running ``--write-baseline``.

Usage:
    python scripts/check_no_new_silent_except.py            # check for new
    python scripts/check_no_new_silent_except.py --write-baseline  # regen

Scope: production code only (external_llm/, services/, webapp/, root asi.py).
Test files (tests/) legitimately use bare ``except`` in scaffolding and are
intentionally excluded.
"""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "silent_except_baseline.txt"

# A handler is "observability-bearing" if it raises, or calls a logging-ish
# function.  Conservative: any such node anywhere in the handler body counts.
_LOGGING_ATTRS = frozenset({
    "debug", "info", "warning", "warn", "error", "exception", "critical", "log",
})

# Bodies that are pure suppression (do nothing but swallow).
_NOOP_STMT_TYPES = (ast.Pass, ast.Continue, ast.Break)

_SCAN_ROOTS = ("external_llm", "services", "webapp")
_SKIP_DIRS = frozenset({
    "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules",
    ".venv", "venv", "env", ".tox", "dist", "build", ".eggs", ".git",
})


def _is_none_expr(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _handler_is_silent(handler: ast.ExceptHandler) -> bool:
    """True iff *handler*'s body purely swallows the exception with no raise/log.

    Conservative on the *silent* side: a body containing any non-noop statement
    that is not a bare/None return (e.g. an assignment, a side-effecting call,
    or ``return <value>``) is NOT flagged — those handlers do *something* and are
    not the pure-black-hole pattern this gate targets.  ``return None``/bare
    ``return``/``pass``/``continue``/``break`` bodies ARE flagged.
    """
    for n in ast.walk(handler):
        if isinstance(n, ast.Raise):
            return False
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in _LOGGING_ATTRS:
                return False
            if isinstance(f, ast.Name) and f.id == "print":
                return False
    body = handler.body
    if not body:
        return False  # ast never produces empty handler bodies; defensive.
    return all(
        isinstance(s, _NOOP_STMT_TYPES)
        or (isinstance(s, ast.Return) and (s.value is None or _is_none_expr(s.value)))
        for s in body
    )


class _SilentExceptScanner(ast.NodeVisitor):
    """Collect drift-stable baseline keys for silent except handlers."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        # Module scope is always present (never popped); nested funcs/classes
        # push/pop on top of it, so their handlers get their own qualname+counter.
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

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        scope = self._scopes[-1]
        scope[1] += 1
        if _handler_is_silent(node):
            self.keys.append(f"{self.rel}::{self._qualname()}::{scope[1] - 1}")
        self.generic_visit(node)


def _scan_source(src: str, rel: str) -> list[str]:
    """Pure analysis: return silent-except baseline keys for *src*.

    Factored out of :func:`_scan_path` so detection is directly unit-testable
    with crafted strings (no temp files).  Mirrors the IO/analysis split used by
    ``test_ast_cache_mutation_guard.py``.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    scanner = _SilentExceptScanner(rel)
    scanner.visit(tree)
    return scanner.keys


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _iter_repo_py() -> list[Path]:
    paths: list[Path] = []
    for root in _SCAN_ROOTS:
        d = REPO / root
        if d.is_dir():
            paths.extend(p for p in d.rglob("*.py") if not _should_skip(p))
    asi = REPO / "asi.py"
    if asi.exists():
        paths.append(asi)
    return paths


def _scan_path(path: Path) -> list[str]:
    rel = str(path.relative_to(REPO))
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _scan_source(src, rel)


def _get_current_keys() -> set[str]:
    keys: set[str] = set()
    for p in _iter_repo_py():
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
        "# asicode silent-except baseline — KNOWN pre-existing handlers whose\n"
        "# body purely swallows the exception (pass/return None/return/continue/\n"
        "# break) with no raise and no log call.\n"
        "#\n"
        "# Generated by `scripts/check_no_new_silent_except.py --write-baseline`.\n"
        "# Hand-editing is allowed: sorted, one key per line, `#` comments ok.\n"
        "# Key: <rel_path>::<scope_qualname>::<ordinal>  (stable under line drift)\n"
        "#\n"
    )
    lines = sorted(keys)
    BASELINE.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    current = _get_current_keys()
    if "--write-baseline" in sys.argv:
        _write_baseline(current)
        print(f"✅ Baseline written: {BASELINE} ({len(current)} entries)")
        return 0

    baseline = _load_baseline()
    new_keys = current - baseline

    if not new_keys:
        print(f"✅ No new silent-except handlers ({len(current)} total, {len(baseline)} baselined)")
        return 0

    print(f"❌ {len(new_keys)} NEW silent-except handler(s) (not in baseline):\n")
    for k in sorted(new_keys):
        print(f"  {k}")
    print(f"\nTotal silent-except: {len(current)}, Baselined: {len(baseline)}")
    print("A silent except swallows failures with no log — add a logger.debug/raise,")
    print("or run `--write-baseline` if this is an intentional fallback.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
