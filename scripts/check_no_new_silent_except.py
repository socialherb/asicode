#!/usr/bin/env python3
"""Check no NEW silent ``except`` handlers vs baseline.

A *silent* except handler is one whose body does nothing but swallow the
exception — ``pass``, ``return None``, bare ``return``, ``continue``, or
``break`` — with no ``raise`` and no observability call (logger.* / print /
logging.exception).  The gate also catches ``with suppress(Exception):`` —
the ``contextlib.suppress`` form of the same black hole (broad or empty
exception lists only; narrow lists like ``suppress(KeyError)`` document the
exact failure they tolerate and are NOT flagged).  Import aliases are
resolved, so ``import contextlib as cl; cl.suppress(...)`` and
``from contextlib import suppress as cs; cs(...)`` are detected too; a
``*.suppress`` attribute on any other module is not treated as contextlib.
exact failure they tolerate and are legitimate) — and ``except Exception:
return <empty>``, where the body returns an empty container/string or a
zero/false literal (``[]``/``{}``/``()``/``set()``/``''``/``0``/``False``):
the caller cannot distinguish "nothing found" from "something broke", which
is the same black hole in a different shape (``emptyret::`` key family).
Such handlers are the classic observability black hole: this repo has a
documented history (0.2.6 version_check) of a fail-open silent disable
causing a real incident, and in unattended 12h+ runs a swallowed failure
silently corrupts the self-improve loop's training signal.

This gate does NOT touch the ~1.7k pre-existing silent handlers — they are
frozen as a baseline.  It catches only NET-NEW ones, exactly mirroring the
existing F401/F811/F821 baseline-diff hooks (see ``.pre-commit-config.yaml``).

Key: ``<rel_path>::<scope_qualname>::<ordinal>`` (silent except handlers),
``emptyret::<rel_path>::<scope_qualname>::<ordinal>`` (broad handlers whose
body only ``return <empty value>``), or
``suppress::<rel_path>::<scope_qualname>::<ordinal>`` (broad ``suppress()``
calls) — the prefixes keep the key families disjoint.

    scope_qualname = dotted names of enclosing FunctionDef/AsyncFunctionDef/
                     ClassDef, or ``<module>`` at module level.
    ordinal        = 0-based index of this ExceptHandler among all ExceptHandlers
                     (respectively: of this ``suppress`` call among all calls
                     named ``suppress``) sharing the SAME nearest scope, in
                     source order.  An ``emptyret::`` key reuses the flagged
                     handler's own ordinal.

This key is **stable under line drift** — inserting/deleting lines above a
handler, or adding ordinary statements to its function, leaves the key
unchanged (unlike an F811-style ``file::line`` key, which would churn noisily
at the ~1.7k scale here).  Only genuinely new handlers (new ordinal) trip the
gate.  A handler that is moved to a different function or whose enclosing
function is renamed will appear as remove+add; that is acceptable churn and is
absorbed by re-running ``--write-baseline``.

Usage:
    python scripts/check_no_new_silent_except.py            # check for new (worktree)
    python scripts/check_no_new_silent_except.py --index-only
        # check against the HEAD+index snapshot (git ls-files + `git show :<path>`)
        # instead of the working tree.  For pre-commit hooks this is the RIGHT
        # mode: the hook sees exactly what would be committed, so an unstaged
        # edit to a baselined file can no longer cause a false failure (the
        # stash-trap: pre-commit stashes unstaged files, so a baseline that was
        # regenerated against a half-stashed tree drifts out of sync).
    python scripts/check_no_new_silent_except.py --write-baseline  # regen
    python scripts/check_no_new_silent_except.py --index-only --write-baseline
        # regen baseline from the HEAD+index snapshot (same semantics as the
        # hook itself; keeps baseline consistent with what commits ship).
    python scripts/check_no_new_silent_except.py --observe
        # REPORT-ONLY trend monitor (exit 0 always) — prints NEW + RESOLVED
        # drift vs the committed baseline.  The reduction program regenerates
        # the baseline after each round; run this in CI to keep the "backlog
        # is shrinking" promise honest (mirrors check_plr2004_observation.py).
    python scripts/check_no_new_silent_except.py <file>.py ... [--index-only]
        # scan only the given files (pre-commit per-file mode).  The
        # full-repo always_run scans were dropped from the hook config because
        # they created a multi-second window where pre-commit's run-start
        # `git diff` vs post-hook diff comparison false-positives on
        # parallel-session writes.  No args (lint.yml CI) scans the full repo;
        # keys are repo-relative in both modes, so the baseline comparison is
        # identical for the files scanned.

Scope: production code only (external_llm/, services/, webapp/, root
production modules: asi/diff_apply/common/config/context_collector/
patch_synth/path_security/plan_compiler).
Test files (tests/) legitimately use bare ``except`` in scaffolding and are
intentionally excluded.
"""

import ast
import os
import subprocess
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

# Root-level production modules (imported by the agent/webapp/services or the
# CLI itself).  These sit outside the tree roots above, so they are enumerated
# explicitly — same protection as the trees, no blind spot for root files.
# radio.py is excluded: it is not imported by any production code.
_ROOT_FILES = (
    "asi.py",
    "common.py",
    "config.py",
    "context_collector.py",
    "diff_apply.py",
    "patch_synth.py",
    "path_security.py",
    "plan_compiler.py",
)
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


def _suppress_call_is_broad(call: ast.Call) -> bool:
    """True iff a ``suppress(...)`` call swallows an (almost) unbounded scope.

    Broad = no args at all (``suppress()`` with no exceptions never suppresses
    anything — a dead context manager), or any positional arg that is a bare
    ``Exception``/``BaseException`` name.  Narrow exception lists
    (``suppress(KeyError)``, ``suppress(OSError, ValueError)``) are legitimate
    and NOT flagged — they document the exact failure they tolerate, which is
    the same observability contract as a narrow ``except``.  Non-literal args
    (``suppress(*excs)``) are conservatively skipped.
    """
    if not call.args:
        return True
    return any(
        isinstance(a, ast.Name) and a.id in ("Exception", "BaseException")
        for a in call.args
    )


def _return_value_is_empty(node: ast.AST | None) -> bool:
    """True iff *node* is an empty/zero literal: ``[]`` ``{}`` ``()`` ``set()``
    ``''`` ``0`` ``False``.  Bare ``return`` (None) is the silent family's job.
    """
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, str):
            return v == ""
        if isinstance(v, bool):
            return v is False
        if isinstance(v, int):
            return v == 0
        return False
    if isinstance(node, ast.List) and not node.elts:
        return True
    if isinstance(node, ast.Tuple) and not node.elts:
        return True
    if isinstance(node, ast.Set) and not node.elts:
        return True
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
        and not node.args
        and not node.keywords
    )


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    """True iff *handler* catches an (almost) unbounded exception set: bare
    ``except:``, ``except Exception:``/``except BaseException:``, or a tuple
    containing either — mirroring ``_suppress_call_is_broad``."""
    t = handler.type
    if t is None:
        return True
    if isinstance(t, ast.Name):
        return t.id in ("Exception", "BaseException")
    if isinstance(t, ast.Tuple):
        return any(
            isinstance(e, ast.Name) and e.id in ("Exception", "BaseException")
            for e in t.elts
        )
    return False


def _handler_empty_returns(handler: ast.ExceptHandler) -> bool:
    """True iff a broad handler's body is nothing but ``return <empty value>``.

    ``except Exception: return []`` swallows a failure exactly like ``except
    Exception: pass`` — the caller cannot tell "nothing found" from "something
    broke" — so it is the same black hole in a different shape, tracked under
    the ``emptyret::`` key family.  Any raise or observability call anywhere
    in the body exempts the handler (same contract as ``_handler_is_silent``);
    narrow handlers (``except KeyError: return []``) document the exact
    failure they tolerate and are NOT flagged.
    """
    body = handler.body
    if not body:
        return False
    for n in ast.walk(handler):
        if isinstance(n, ast.Raise):
            return False
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in _LOGGING_ATTRS:
                return False
            if isinstance(f, ast.Name) and f.id == "print":
                return False
    return all(
        isinstance(s, ast.Return) and _return_value_is_empty(s.value)
        for s in body
    )


class _SilentExceptScanner(ast.NodeVisitor):
    """Collect drift-stable baseline keys for silent except handlers."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        # Module scope is always present (never popped); nested funcs/classes
        # push/pop on top of it, so their handlers get their own qualname+
        # counters.  Entry layout: [qualname, except_ordinal, suppress_ordinal].
        self._scopes: list[list] = [["<module>", 0, 0]]
        self.keys: list[str] = []
        # Import-alias resolution (populated in source order during the walk,
        # so a call is always classified against the imports that precede it):
        # names bound to the contextlib module (`import contextlib as cl` →
        # "cl") and names bound to contextlib.suppress (`from contextlib
        # import suppress as cs` → "cs").  Re-binding a name afterwards is a
        # conservative over-approximation (documented in _is_suppress_call).
        self._contextlib_names: set[str] = {"contextlib"}
        self._suppress_names: set[str] = {"suppress"}

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name == "contextlib":
                self._contextlib_names.add(a.asname or "contextlib")
            elif a.name == "contextlib.suppress":
                # ``import contextlib.suppress`` — rare, but legal.
                self._suppress_names.add(a.asname or "suppress")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "contextlib":
            for a in node.names:
                if a.name == "suppress":
                    self._suppress_names.add(a.asname or "suppress")
                # ``from contextlib import *`` is conservatively missed (star
                # imports cannot be resolved statically) — documented gap.
        self.generic_visit(node)

    def _qualname(self) -> str:
        return "::".join(s[0] for s in self._scopes)

    def _scope_visit(self, node) -> None:
        self._scopes.append([node.name, 0, 0])
        self.generic_visit(node)
        self._scopes.pop()

    visit_FunctionDef = _scope_visit
    visit_AsyncFunctionDef = _scope_visit
    visit_ClassDef = _scope_visit

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        scope = self._scopes[-1]
        scope[1] += 1
        ordinal = scope[1] - 1
        if _handler_is_silent(node):
            self.keys.append(f"{self.rel}::{self._qualname()}::{ordinal}")
        elif _handler_is_broad(node) and _handler_empty_returns(node):
            self.keys.append(f"emptyret::{self.rel}::{self._qualname()}::{ordinal}")
        self.generic_visit(node)

    def _is_suppress_call(self, node: ast.Call) -> bool:
        """True iff *node* is a call to something named ``suppress``.

        Resolves imports to their aliases (tracked in source order by
        ``visit_Import``/``visit_ImportFrom``), so all of these match:

        * ``contextlib.suppress(...)`` and ``import contextlib as cl;
          cl.suppress(...)``  (attribute form on the contextlib module)
        * ``from contextlib import suppress; suppress(...)`` and
          ``from contextlib import suppress as cs; cs(...)``  (name form)

        A ``*.suppress`` attribute on any OTHER module (e.g.
        ``shutil.suppress``) is NOT a contextlib suppress and is not flagged.
        Re-binding an alias name afterwards is a conservative over-
        approximation (a call through the rebound name would be flagged);
        ``from contextlib import *`` star imports are missed (documented).
        """
        f = node.func
        return (isinstance(f, ast.Name) and f.id in self._suppress_names) or (
            isinstance(f, ast.Attribute)
            and f.attr == "suppress"
            and isinstance(f.value, ast.Name)
            and f.value.id in self._contextlib_names
        )

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_suppress_call(node):
            scope = self._scopes[-1]
            scope[2] += 1
            if _suppress_call_is_broad(node):
                self.keys.append(f"suppress::{self.rel}::{self._qualname()}::{scope[2] - 1}")
        self.generic_visit(node)


def _scan_source(src: str, rel: str) -> list[str]:
    """Pure analysis: return silent-except baseline keys for *src*.

    Factored out of :func:`_scan_path` so detection is directly unit-testable
    with crafted strings (no temp files).  Mirrors the IO/analysis split used by
    the baseline-diff gate's own unit tests.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    scanner = _SilentExceptScanner(rel)
    scanner.visit(tree)
    return scanner.keys


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


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _in_scope(rel: str) -> bool:
    """True iff *rel* (repo-relative) is inside this gate's scan scope.

    Scope is production code only: the ``_ROOT_FILES`` root modules + the
    ``_SCAN_ROOTS`` trees.  Test files legitimately use bare ``except`` in
    scaffolding and are intentionally excluded (mirrors the full-scan
    predicate).
    """
    p = Path(rel)
    return (
        p.suffix == ".py"
        and not _should_skip(p)
        and (p.parts[0] in _SCAN_ROOTS or p.parts[0] in _ROOT_FILES)
    )


def _iter_repo_py(paths: list[str] | None = None) -> list[Path]:
    if paths is not None:
        return [REPO / rel for rel in paths if _in_scope(rel)]
    full: list[Path] = []
    for root in _SCAN_ROOTS:
        d = REPO / root
        if d.is_dir():
            full.extend(p for p in d.rglob("*.py") if not _should_skip(p))
    for root in _ROOT_FILES:
        f = REPO / root
        if f.exists():
            full.append(f)
    return full


def _scan_path(path: Path) -> list[str]:
    rel = str(path.relative_to(REPO))
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _scan_source(src, rel)


def _git_list_scope_py() -> list[str] | None:
    """Tracked .py paths (repo-relative) inside the scan scope, from the index.

    Returns ``None`` when git is unavailable or the directory is not a repo —
    the caller then falls back to the working-tree scan (old behavior).  Uses
    ``git ls-files`` so the file list is exactly what a commit would ship:
    untracked files and unstaged edits are invisible, which is the point of
    ``--index-only``.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out: list[str] = []
    for rel in proc.stdout.decode("utf-8", errors="replace").split("\0"):
        if not rel:
            continue
        if _in_scope(rel):
            out.append(rel)
    return out


def _read_index_blob(rel: str) -> str | None:
    """Content of *rel* as staged in the index (``git show :<path>``).

    Returns ``None`` on git failure / missing blob (caller skips the file).
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO), "show", f":{rel}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def _get_current_keys(index_only: bool = False, paths: list[str] | None = None) -> set[str]:
    keys: set[str] = set()
    if index_only:
        rels = [r for r in paths if _in_scope(r)] if paths is not None else _git_list_scope_py()
        if rels is not None:
            for rel in rels:
                src = _read_index_blob(rel)
                if src is not None:
                    keys.update(_scan_source(src, rel))
            return keys
        # git unavailable → fall back to working-tree scan (pre-git behavior).
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
        "# asicode silent-except baseline — KNOWN pre-existing handlers whose\n"
        "# body purely swallows the exception (pass/return None/return/continue/\n"
        "# break) with no raise and no log call — plus broad `suppress()` calls\n"
        "# (suppress(Exception/BaseException) or empty suppress()) and broad\n"
        "# handlers whose body is only `return <empty>` (emptyret:: keys: []/{} /\n"
        "# ()/set()/''/0/False).  Narrow suppress(KeyError) and narrow except +\n"
        "# empty return are NOT baselined (never flagged).\n"
        "#\n"
        "# Generated by `scripts/check_no_new_silent_except.py --write-baseline`.\n"
        "# Hand-editing is allowed: sorted, one key per line, `#` comments ok.\n"
        "# Key: <rel_path>::<scope_qualname>::<ordinal>  (stable under line drift)\n"
        "# emptyret keys: emptyret::<rel_path>::<scope_qualname>::<ordinal>\n"
        "# suppress keys: suppress::<rel_path>::<scope_qualname>::<ordinal>\n"
        "# NOTE: the ordinal is the handler's index among ALL except handlers in\n"
        "# the scope (silent or not), so REMOVING any handler shifts later keys.\n"
        "# After a legit removal round (e.g. try/except/pass → suppress), re-run\n"
        "# `--write-baseline` — index-shifted keys are false \"new\" hits.\n"
        "#\n"
    )
    lines = sorted(keys)
    BASELINE.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    index_only = "--index-only" in sys.argv
    observe = "--observe" in sys.argv
    paths = _resolve_scan_paths([a for a in sys.argv[1:] if not a.startswith("--")])
    current = _get_current_keys(index_only=index_only, paths=paths)
    if "--write-baseline" in sys.argv:
        _write_baseline(current)
        src = "HEAD+index" if index_only else "working tree"
        print(f"✅ Baseline written: {BASELINE} ({len(current)} entries, from {src})")
        return 0

    if observe:
        # Report-only trend monitor (mirrors check_plr2004_observation.py):
        # shows NEW + RESOLVED drift vs the committed baseline and always
        # exits 0.  The reduction program regenerates the baseline after each
        # round, so RESOLVED reflects what disappeared since the last snapshot;
        # a non-empty NEW here means the committed baseline is stale.
        baseline = _load_baseline()
        new = current - baseline
        resolved = baseline - current
        print(f"silent-except observation: total {len(current)} sites (baseline {len(baseline)})")
        for k in sorted(new):
            print(f"  ⬆ NEW  {k}")
        for k in sorted(resolved):
            print(f"  ⬇ gone {k} (re-baseline with --write-baseline when intended)")
        if not new and not resolved:
            print("  no drift vs baseline — backlog stable")
        print("report-only observation — exit 0 (trend monitor, not a gate)")
        return 0

    baseline = _load_baseline()
    new_keys = current - baseline

    if not new_keys:
        print(f"✅ No new silent-swallow sites ({len(current)} total, {len(baseline)} baselined)")
        return 0

    print(f"❌ {len(new_keys)} NEW silent-swallow site(s) (not in baseline):\n")
    for k in sorted(new_keys):
        print(f"  {k}")
    print(f"\nTotal silent-except: {len(current)}, Baselined: {len(baseline)}")
    print("A silent except / broad suppress / empty-return swallows failures with no log — add a logger.debug/raise,")
    print("or run `--write-baseline` if this is an intentional fallback.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
