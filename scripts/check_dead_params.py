#!/usr/bin/env python3
"""Check no NEW unused function/method parameters vs baseline.

A parameter that the body never reads is either a bug (the caller computed a
setting that is silently ignored — e.g. ``build_llm_context_v8`` accepted
``context_level`` from the HTTP API and never used it) or interface noise
that should be removed from the signature.

Documented incidents (P2-4 round):
  * ui_tools.build_llm_context_v8 ``context_level`` — exposed via HTTP Query,
    forwarded by the route, never read by the builder (v7 only).
  * ui_tools.build_llm_context_v7 ``context_mode`` — "kept for compat", never read.
  * agent_context_manager._build_initial_messages/_build_continuation_messages
    ``has_native_tools`` — threaded through 8 call sites, ignored by both builders.
  * context_manager.schedule_background_compress/build_context_messages
    ``system_chars`` — docstring self-confessed "(unused)", pass-through chain.
  * agent_loop._llm_call_with_tools ``read_only_request`` — pure remnant.
  All removed; the baseline below holds the legitimate interface-uniformity
  exceptions (overridden/abstract signatures must keep unused params).

Conservative filters (FP avoidance):
  * decorated functions skipped (routes/callbacks may be signature-inspected)
  * classes with a base class / metaclass skipped (override surface)
  * ``*args`` / ``**kwargs`` functions skipped (open-ended interfaces)
  * ``_``-prefixed params skipped (documented "ignored" convention)
  * bodies spanning < 4 source lines skipped (tiny callback wrappers)
  * nested function bodies are not scanned for uses (shadowing)

Baseline key: ``<rel_path>::<scope_qualname>::<param>`` (drift-stable;
same model as the silent-except / discarded-signal gates).

Usage:
    python scripts/check_dead_params.py               # check (working tree)
    python scripts/check_dead_params.py --write-baseline  # regen
    python scripts/check_dead_params.py <file>.py ...     # per-file
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "dead_params_baseline.txt"

_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".tox",
        "dist",
        "build",
        ".eggs",
        ".git",
    }
)
_SCAN_ROOTS = ("external_llm", "services", "webapp")

_MIN_BODY_LINES = 4


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _in_scope(rel: str) -> bool:
    p = Path(rel)
    return p.suffix == ".py" and not _should_skip(p) and (p == Path("asi.py") or p.parts[0] in _SCAN_ROOTS)


class _UseCollector(ast.NodeVisitor):
    """Count Load uses of *name*, descending into nested scopes UNLESS the
    nested function/lambda shadows *name* with its own parameter (then its
    body is skipped — references there bind to the inner param)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            self.found = True

    def _enter_nested(self, node) -> None:
        args = node.args
        params = [a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs) if a.arg]
        if self.name in params:
            return  # shadowed by the inner signature — do not descend
        self.generic_visit(node)

    visit_FunctionDef = _enter_nested  # noqa: N815 — NodeVisitor override idiom
    visit_AsyncFunctionDef = _enter_nested  # noqa: N815 — NodeVisitor override idiom
    visit_Lambda = _enter_nested  # noqa: N815
    # Class bodies can reference enclosing params at evaluation time — descend.


def _body_lines(body: list[ast.stmt]) -> int:
    if not body:
        return 0
    return max(getattr(s, "end_lineno", s.lineno) or s.lineno for s in body) - body[0].lineno + 1


def _param_used(body: list[ast.stmt], name: str) -> bool:
    collector = _UseCollector(name)
    for stmt in body:
        collector.visit(stmt)
    return collector.found


def _has_vararg(node: ast.arguments) -> bool:
    return node.vararg is not None or node.kwarg is not None


def _dead_params(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return unused (name) params of *func* per the conservative filters."""
    args = func.args
    posonly = list(args.posonlyargs) + list(args.args)
    all_params = posonly + list(args.kwonlyargs)
    if not all_params or _has_vararg(args):
        return []
    # Skip self/cls (first positional) — conventionally unlisted
    if posonly and posonly[0].arg in ("self", "cls"):
        posonly = posonly[1:]
        all_params = posonly + list(args.kwonlyargs)
    if _body_lines(func.body) < _MIN_BODY_LINES:
        return []
    out = []
    for p in all_params:
        name = p.arg
        if name.startswith("_") or not name:
            continue
        if not _param_used(func.body, name):
            out.append(name)
    return out


class _DeadParamScanner(ast.NodeVisitor):
    """Collect drift-stable keys for unused parameters, one pass (no dupes)."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.keys: list[str] = []

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str) -> None:
        if node.decorator_list:
            return
        for p in _dead_params(node):
            self.keys.append(f"{self.rel}::{qualname}::{p}")

    def visit_FunctionDef(self, node) -> None:
        self._record(node, node.name)
        self.generic_visit(node)  # nested functions are scanned too

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815 — NodeVisitor override idiom

    def visit_ClassDef(self, node) -> None:
        if node.bases or node.keywords:  # base class / metaclass — override surface
            return
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._record(item, f"{node.name}.{item.name}")
            else:
                self.visit(item)  # nested classes re-enter via visit_ClassDef


def _scan_source(src: str, rel: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    scanner = _DeadParamScanner(rel)
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
        "# asicode dead-parameter baseline — KNOWN function/method parameters\n"
        "# never read by the body. Usually a bug (a caller-computed setting that\n"
        "# is silently ignored) or signature noise; the documented incidents are\n"
        "# listed in check_dead_params.py. Legitimate exceptions: overridden or\n"
        "# abstract signatures whose subclasses/implementations must accept the\n"
        "# full interface (interface uniformity).\n"
        "#\n"
        "# Generated by `scripts/check_dead_params.py --write-baseline`.\n"
        "# Key: <rel_path>::<scope_qualname>::<param> (stable under line drift)\n"
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
        print(f"✅ No NEW unused parameters ({len(current)} total, {len(baseline)} baselined)")
        return 0

    print(f"❌ {len(new_keys)} NEW unused parameter(s) (not in baseline):\n")
    for k in sorted(new_keys):
        print(f"  {k}")
    print("\nA parameter the body never reads is usually a bug — remove it from the")
    print("signature and its call sites, or run `--write-baseline` if it is a")
    print("legitimate interface-uniformity exception.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
