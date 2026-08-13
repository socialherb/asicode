#!/usr/bin/env python3
"""Check no NEW "computed-and-dropped signal" patterns vs baseline.

A function computes a value (``x = <Call>``) and the very NEXT statement
raises an exception whose message f-string is the ONLY use of ``x`` — the
value is never passed as structured data (positional/keyword argument) to
the exception. That shape is usually a bug: the computed signal is discarded.

Documented incidents (P0-1 round):
  * providers.py Ollama 429 (2 sites, now fixed): ``retry_after =
    parse_retry_after(response.headers)`` was interpolated into the message
    f-string only; retry layers read ``e.retry_after`` via getattr, so the
    server's Retry-After hint was silently dropped on every Ollama 429.
    Fixed by passing ``retry_after=retry_after`` — the kwarg use excludes
    the site from this detector.
  * web_search_tools.py ``detail`` (baselined false positive): a display-only
    string built for the message, legitimately used nowhere else.

Baseline key: ``<rel_path>::<scope_qualname>::<ordinal>`` (same
drift-stability model as the silent-except gate).

Usage:
    python scripts/check_discarded_signal.py            # check (working tree)
    python scripts/check_discarded_signal.py --write-baseline  # regen
    python scripts/check_discarded_signal.py <file>.py ...      # per-file
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "discarded_signal_baseline.txt"

_SKIP_DIRS = frozenset({
    "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules",
    ".venv", "venv", "env", ".tox", "dist", "build", ".eggs", ".git",
})
_SCAN_ROOTS = ("external_llm", "services", "webapp")


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _in_scope(rel: str) -> bool:
    p = Path(rel)
    return (
        p.suffix == ".py"
        and not _should_skip(p)
        and (p == Path("asi.py") or p.parts[0] in _SCAN_ROOTS)
    )


def _name_uses_in_raise(raise_node: ast.Raise, name: str) -> tuple[list[int], list[int]]:
    """Return (fstring_lines, structured_lines) for *name* inside the raised
    Call. A use as an f-string interpolation value is 'message-only'; any
    other Load use (positional/keyword argument, nested expression) is
    'structured'. Node IDENTITY separates the two classes — an f-string and a
    kwarg on the same line must not collide (line-number dedup would merge
    them and miss the structured use)."""
    fv_nodes: set[int] = set()
    fstring_lines: list[int] = []
    for node in ast.walk(raise_node):
        if isinstance(node, ast.JoinedStr):
            for val in node.values:
                if (
                    isinstance(val, ast.FormattedValue)
                    and isinstance(val.value, ast.Name)
                    and val.value.id == name
                ):
                    fv_nodes.add(id(val.value))
                    fstring_lines.append(val.value.lineno)
    structured_lines = [
        node.lineno
        for node in ast.walk(raise_node)
        if (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
            and id(node) not in fv_nodes
        )
    ]
    return fstring_lines, structured_lines


class _DiscardedSignalScanner(ast.NodeVisitor):
    """Collect drift-stable keys for computed-and-dropped raise patterns."""

    _STMT_ATTRS = ("body", "orelse", "finalbody")

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self._scopes: list[list] = [["<module>", 0]]
        self.keys: list[str] = []

    def _qualname(self) -> str:
        return "::".join(s[0] for s in self._scopes)

    def visit(self, node) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._scopes.append([node.name, 0])
        # Check consecutive statement pairs in EVERY statement list (function
        # body, if/try/for/while/with bodies, else/orelse/finalbody) — the
        # assign-then-raise pair is usually nested inside an ``if`` guard.
        for attr in self._STMT_ATTRS:
            body = getattr(node, attr, None)
            if isinstance(body, list) and body and all(isinstance(s, ast.stmt) for s in body):
                self._check_statements(body)
        self.generic_visit(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._scopes.pop()

    def _check_statements(self, body: list[ast.stmt]) -> None:
        for i, stmt in enumerate(body[:-1]):
            nxt = body[i + 1]
            if not isinstance(stmt, ast.Assign) or not isinstance(nxt, ast.Raise):
                continue
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue
            if not isinstance(stmt.value, ast.Call):
                continue  # only `x = <Call>` shapes
            exc = nxt.exc
            if not isinstance(exc, ast.Call):
                continue
            name = stmt.targets[0].id
            fstring_lines, structured_lines = _name_uses_in_raise(nxt, name)
            if fstring_lines and not structured_lines:
                scope = self._scopes[-1]
                scope[1] += 1
                self.keys.append(f"{self.rel}::{self._qualname()}::{scope[1] - 1}")


def _scan_source(src: str, rel: str) -> list[str]:
    """Pure analysis: return discarded-signal keys for *src*."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    scanner = _DiscardedSignalScanner(rel)
    scanner.visit(tree)
    return scanner.keys


def _scan_path(path: Path) -> list[str]:
    rel = str(path.relative_to(REPO))
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _scan_source(src, rel)


def _iter_repo_py(paths: list[str] | None = None) -> list[Path]:
    if paths is not None:
        return [REPO / rel for rel in paths if _in_scope(rel)]
    full: list[Path] = []
    for root in _SCAN_ROOTS:
        d = REPO / root
        if d.is_dir():
            full.extend(p for p in d.rglob("*.py") if not _should_skip(p))
    asi = REPO / "asi.py"
    if asi.exists():
        full.append(asi)
    return full


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
        "# asicode discarded-signal baseline — KNOWN `x = <Call>` immediately\n"
        "# followed by `raise ... f\"...{x}...\"` where x appears ONLY in the\n"
        "# message f-string (never passed as structured data). Usually a bug\n"
        "# (computed signal dropped — see providers.py Ollama retry_after), but\n"
        "# display-only message strings are legitimate exceptions.\n"
        "#\n"
        "# Generated by `scripts/check_discarded_signal.py --write-baseline`.\n"
        "# Key: <rel_path>::<scope_qualname>::<ordinal> (stable under line drift)\n"
        "#\n"
    )
    lines = sorted(keys)
    BASELINE.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _resolve_scan_paths(args: list[str]) -> list[str] | None:
    out: list[str] = []
    for a in args:
        rel = os.path.relpath(Path(a).resolve(), Path(REPO).resolve())
        if rel.endswith(".py") and not rel.startswith(".."):
            out.append(rel)
    return out or None


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
        print(f"✅ No new discarded-signal patterns ({len(current)} total, {len(baseline)} baselined)")
        return 0

    print(f"❌ {len(new_keys)} NEW discarded-signal pattern(s) (not in baseline):\n")
    for k in sorted(new_keys):
        print(f"  {k}")
    print("\nA value computed then used ONLY in the exception's message f-string is a")
    print("dropped signal — pass it as a structured arg (e.g. retry_after=) or run")
    print("`--write-baseline` if it is a display-only message string.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
