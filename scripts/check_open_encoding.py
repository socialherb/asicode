#!/usr/bin/env python3
"""Check every locale-decoding call outside tests/ passes an explicit ``encoding=``.

Covers text-mode ``open()`` and ``Path.read_text()``/``Path.write_text()`` —
the twins were the follow-up gap: the original sweep fixed the 44 ``open()``
sites while 30 ``read_text``/``write_text`` sites kept decoding (e.g. Korean
config) with the process locale.

Like check_f823_none.py this gate has **no baseline**: the 44 pre-existing
sites were all fixed when it was introduced, so any hit is new. Keep it that
way — a missing encoding is never intentional.

Why this gate exists
--------------------
``open(path)`` in text mode decodes with ``locale.getpreferredencoding()``,
not UTF-8. On this developer's machine PEP 538 C-locale coercion hides that:
the preferred encoding comes back ``utf-8`` even under ``LC_ALL=C``. It does
not hide everywhere — a container without a ``C.UTF-8`` locale, or with
``PYTHONCOERCECLOCALE=0``, decodes as ASCII and every Korean comment in a
source file becomes a ``UnicodeDecodeError``. This package ships to PyPI, so
"works on my locale" is not a property we can rely on.

The repo had already settled the convention (498 call sites passed
``encoding=``, 44 did not), so this is about keeping the last 44 from coming
back rather than introducing a rule.

Why not ruff
------------
Ruff implements this as PLW1514 (unspecified-encoding), but the rule is in
**preview** and needs ``--preview``, which would also activate every other
unstable rule across the repo. An AST check costs nothing and additionally
lets us scope by directory and understand ``mode=`` passed as a keyword.

Scope
-----
Everything except ``tests/``. Test files open fixtures they just wrote into
``tmp_path``, so the portability exposure is in shipped code; the ~296 test
sites are a separate, mechanical cleanup and are deliberately not gated here.

Usage:
    python scripts/check_open_encoding.py
    python scripts/check_open_encoding.py <file>.py ...  # check only given files

Explicit file args (pre-commit per-file mode) scan only those files — the
full-repo always_run scans were dropped from the hook config because they
created a multi-second window where pre-commit's run-start `git diff` vs
post-hook diff comparison false-positives on parallel-session writes.  No
args (lint.yml CI) still scans the whole repo.
"""

import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directories that never reach a user's machine, plus caches.
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache",
             ".pytest_cache", "build", "dist", ".asicode", "tests"}


def _resolve_scan_paths(args: list[str]) -> list[str] | None:
    """Normalize explicit file args to repo-relative ``*.py`` paths.

    Returns ``None`` when no file args survive (or none were given) — the
    caller then scans the whole repo, preserving the no-args (lint.yml CI)
    behaviour.  pre-commit passes absolute paths; lint.yml passes none — both
    normalize to the same repo-relative key space as the full rglob scan.
    """
    out: list[str] = []
    for a in args:
        rel = os.path.relpath(Path(a).resolve(), Path(REPO).resolve())
        if rel.endswith(".py") and not rel.startswith(".."):
            out.append(rel)
    return out or None


def _iter_py_files(paths=None):
    if paths is None:
        for path in REPO.rglob("*.py"):
            rel = path.relative_to(REPO)
            if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
                continue
            yield rel, path
        return
    for rel in paths:
        p = REPO / rel
        if not p.is_file() or p.suffix != ".py":
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in Path(rel).parts):
            continue
        yield Path(rel), p


def _violations_in(source: str) -> list[int]:
    """Line numbers of locale-decoding calls with no ``encoding=``.

    Covers text-mode ``open()`` and the ``Path.read_text()`` /
    ``Path.write_text()`` twins, which decode/encode with the same
    locale-derived default. The attribute check is by method NAME, not
    receiver type — any same-named API (importlib Traversable etc.) shares
    the locale default, so flagging it is correct, not a false positive.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # not our problem — the syntax gates own that
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {kw.arg for kw in node.keywords}
        if "encoding" in kwargs:
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            # Binary mode takes no encoding. mode is positional arg 1 or a
            # keyword; a non-literal mode is treated as text (conservative).
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if "b" in mode:
                continue
            out.append(node.lineno)
        elif isinstance(node.func, ast.Attribute) and node.func.attr in (
            "read_text", "write_text",
        ):
            # encoding may also arrive positionally: read_text(encoding, ...)
            # is positional arg 0, write_text(data, encoding, ...) arg 1.
            positional_encoding = 1 if node.func.attr == "write_text" else 0
            if len(node.args) > positional_encoding:
                continue
            out.append(node.lineno)
    return out


def main() -> int:
    findings: list[str] = []
    paths = _resolve_scan_paths([a for a in sys.argv[1:] if not a.startswith("--")])
    for rel, path in sorted(_iter_py_files(paths)):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        findings += [f"{rel}:{line}" for line in _violations_in(source)]

    if not findings:
        print("✅ Every open()/read_text()/write_text() outside tests/ passes encoding= "
              "(0 tolerated — this gate has no baseline)")
        return 0

    print(f"❌ {len(findings)} locale-decoding call(s) (open/read_text/write_text) without encoding=:\n")
    for f in findings:
        print(f"  {f}")
    print(
        "\ntext-mode open() decodes with the process locale, not UTF-8, so these"
        "\nraise UnicodeDecodeError on any non-ASCII byte under a C/POSIX locale"
        "\n(containers without C.UTF-8, PYTHONCOERCECLOCALE=0)."
        "\n\nAdd encoding=\"utf-8\". If the file is user source whose encoding we do"
        "\nnot control, add errors=\"replace\" as well so behaviour stays lenient."
        "\nDo NOT add a baseline for this."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
