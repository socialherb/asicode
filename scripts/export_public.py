#!/usr/bin/env python3
"""Export the public (CLI-only) snapshot of this repo.

The private repo is the single source of truth; the public GitHub repo is a
filtered subset with fresh history. This script materializes that subset.

Excluded from the public snapshot:
  - webapp/          (FastAPI server/UI — not deployed)
  - tools/           (legacy verification scripts)
  - tasks/, screenshots/, .vscode/, CLAUDE.md  (internal artifacts)
  - tests that import webapp/tools (recomputed on every export, so
    newly added coupled tests are excluded automatically)
  - tests that REQUEST a fixture defined by an excluded conftest.py — a test
    need not import an excluded package itself to depend on one (see
    _fixture_coupled_tests)

Lint-baseline files (scripts/*_baseline.txt) are copied then pruned: any
entry keyed by ``<path>::...`` whose path is itself excluded from the
export (e.g. a webapp/ module, or a coupled test) is dropped, so the public
snapshot's baseline never references or names a file that isn't there.

The structural-scanner baseline is REGENERATED, not copied: the five
reference-dependent gate scanners (dead_block / public_dead_code / vulture /
broken_contract / container_reachability) judge symbol liveness, which depends
on WHICH FILES EXIST in the tree — a symbol whose only consumers are excluded
here becomes "unreferenced" in the snapshot. After copying,
``_generate_structural_baseline`` runs the snapshot's own gate in dump mode
and writes scripts/structural_scanner_baseline.txt, machine-verifying every
entry to be referenced from an excluded file; the public CI gate suppresses
exactly those (see scripts/check_structural_scanners.py).

Usage:
    python3 scripts/export_public.py <target-dir>          # export
    python3 scripts/export_public.py <target-dir> --list   # dry-run listing
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

EXCLUDE_PREFIXES = (
    "webapp/",
    "tools/",
    "tasks/",
    "screenshots/",
    ".vscode/",
)
EXCLUDE_FILES = {
    "CLAUDE.md",
    # private development history (references lane/planner internals);
    # the public repo starts its own CHANGELOG at the first release
    "CHANGELOG.md",
    # repo-shape guards over tools/ content — meaningless in the snapshot
    "tests/unit/test_config_flag_reachable.py",
    # Same category: this one asserts things about the PRIVATE tree that are
    # false by construction after export — that webapp/ exists as an excluded
    # package (it does not ship at all). It guards the pre-export release gate,
    # so it belongs upstream of the export, never inside it.
    "tests/unit/test_release_untracked_import_gate.py",
    # Third of the same family: it runs the zero-tolerance structural scanners
    # over the tree and asserts a zero count. That holds for the PRIVATE tree
    # only — public symbols whose sole consumers live under webapp/ (config.py's
    # BENCH_RAW_LLM, LEGACY_DIFF_MODE, INSTRUCTION_MODE, ALLOW_MULTIFILE, and
    # peers in patch_synth/services) are genuinely unreferenced once webapp/ is
    # dropped, so the scanner is right and the assertion is wrong. Measured on
    # the 0.2.19 export: 12 candidates. The gate belongs upstream of the export.
    "tests/unit/test_check_structural_scanners.py",
}

# Modules under these packages ship in the wheel; the release gate's import
# scan only cares about them. Defined here (not in release_public, which is
# re-executed per invocation) so the memoized scan can key on it.
FIRST_PARTY_PREFIXES = ("external_llm.", "webapp.")

# A test file is excluded when it imports (or patches into) an excluded area.
#
# The ``^\s*`` anchors matter: this repo imports lazily by convention, so an
# excluded package is routinely reached from *inside* a test function, indented.
# The anchors were previously ``^from``/``^import`` (column 0 only), which let
# `tests/unit/agent/test_rollback_shared_tree.py` ship — it does
# ``import webapp.routes.agent_stream`` inside `test_webapp_injects_file_lock_manager`,
# and webapp/ does not exist in the public snapshot, so the test failed on a
# fresh clone of the released repo.
_COUPLED_TEST_PAT = re.compile(
    r"(^\s*from webapp|^\s*import webapp\b|from webapp import|from webapp\."
    r"|^\s*from tools|^\s*import tools\b|from tools import|from tools\."
    # path-string loading of excluded dirs (importlib.spec_from_file_location,
    # subprocess script invocations, Path joins that READ an excluded file):
    # REPO / "tools" / "x.py", _R / "webapp" / "ui" / "ui.html" — any depth of
    # quoted components, but the join must END in a quoted FILENAME (dot +
    # extension). A bare quoted token after the slash also occurs in PROSE
    # (omit "tools"/"tool_choice" keys), and the earlier form
    # ["']tools["'] */ matched it — silently dropping two provider regression
    # tests from the 0.2.24 snapshot (caught only as unexpected deletions in
    # the pre-push release-delta review). Requiring the filename sibling keeps
    # genuinely webapp-reading gates excluded on principle, not by accident.
    r"|[\"'](?:tools|webapp)[\"'] */ *(?:[\"'][\w.-]+[\"'] */ *)*[\"'][\w.-]+\.\w+[\"']"
    r"|tools/[A-Za-z_]+\.py|webapp/[A-Za-z_]+\.py)",
    re.M,
)


def tracked_files() -> list[str]:
    # -z: NUL-separated so non-ASCII (e.g. Korean) filenames are exact,
    # never C-quoted (see git ls-files quoting semantics).
    # timeout: this is the release gate's first step, and a git that never
    # returns (index.lock held by another process, a stalled filesystem) would
    # hang the release instead of failing it. Fail-closed — a partial or absent
    # file list here would silently ship the wrong tree.
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, capture_output=True,
            check=True, timeout=120,
        ).stdout
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"git ls-files timed out after {exc.timeout}s in {REPO} — "
            "refusing to export from an unknown file list"
        ) from exc
    return [p.decode("utf-8") for p in out.split(b"\0") if p]


def _base_exclusion(rel: str) -> str | None:
    """Exclusion by path rule or by the test's own imports (no fixture analysis).

    Split out from :func:`is_excluded` so the fixture pass below can ask "is this
    conftest excluded?" without recursing back into itself.
    """
    if rel in EXCLUDE_FILES:
        return "internal"
    for pref in EXCLUDE_PREFIXES:
        if rel.startswith(pref):
            return pref.rstrip("/")
    if rel.startswith("tests/") and rel.endswith(".py"):
        try:
            src = (REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return None
        if _COUPLED_TEST_PAT.search(src):
            return "coupled-test"
    return None


def _fixture_names(src: str) -> set[str]:
    """Names of ``@pytest.fixture``-decorated functions in *src* (AST, not regex)."""
    names: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            # @pytest.fixture, @pytest.fixture(...), @fixture, @fixture(...)
            target = dec.func if isinstance(dec, ast.Call) else dec
            attr = getattr(target, "attr", None) or getattr(target, "id", None)
            if attr == "fixture":
                names.add(node.name)
                break
    return names


@cache
def _requested_params(src: str) -> frozenset[str]:
    """Every parameter name a test/fixture function in *src* asks pytest to inject.

    Parameters are how a test declares a fixture dependency, so this is exact
    where a substring search is not: a docstring or a string literal that merely
    mentions ``test_client`` does not make the file coupled.

    Memoized by source text and returned as a frozenset: release flows evaluate
    this for every test file on every invocation, and files do not change
    during a run, so repeated parses are deduplicated.  Callers must not mutate
    the result (it is shared).
    """
    params: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return frozenset()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if a.arg not in ("self", "cls"):
                params.add(a.arg)
    return frozenset(params)


_FIXTURE_COUPLED_CACHE: dict[str, frozenset[str]] | None = None


def _excluded_conftest_fixtures() -> dict[str, frozenset[str]]:
    """Map ``<dir>/`` -> fixture names defined by an EXCLUDED conftest.py there.

    ``tests/integration/conftest.py`` is excluded (it builds a FastAPI
    ``TestClient`` over ``webapp.main``), but its consumers —
    ``integration/sse/test_sse_streaming.py``, ``e2e/test_end_to_end_scenarios.py``,
    ``memory/test_memory_persistence.py`` — import no webapp symbol themselves, so
    the import-based rule shipped them. In the public snapshot they collected and
    then errored with ``fixture 'test_client' not found`` (12 errors, measured on
    a real export + clean venv). Fixture inheritance is a real coupling edge and
    has to be walked like one.
    """
    global _FIXTURE_COUPLED_CACHE
    if _FIXTURE_COUPLED_CACHE is not None:
        return _FIXTURE_COUPLED_CACHE
    out: dict[str, frozenset[str]] = {}
    for rel in tracked_files():
        if not (rel.startswith("tests/") and rel.endswith("/conftest.py")):
            continue
        if not _base_exclusion(rel):
            continue
        try:
            src = (REPO / rel).read_text(encoding="utf-8")
        except OSError:
            continue
        names = _fixture_names(src)
        if names:
            out[rel[: -len("conftest.py")]] = frozenset(names)
    _FIXTURE_COUPLED_CACHE = out
    return out


@cache
def is_excluded(rel: str) -> str | None:
    """Return the exclusion reason, or None if the file ships.

    Memoized per path: the release gate and the export walk every tracked file
    on every invocation, and repo files are immutable during a run — the same
    assumption the conftest-fixture cache below already makes — so the first
    evaluation per path is authoritative for the process.
    """
    reason = _base_exclusion(rel)
    if reason:
        return reason
    # A test that requests a fixture from an excluded conftest.py cannot run in
    # the snapshot even though it imports nothing excluded itself.
    if rel.startswith("tests/") and Path(rel).name.startswith("test_") and rel.endswith(".py"):
        coupled = _excluded_conftest_fixtures()
        if coupled:
            try:
                src = (REPO / rel).read_text(encoding="utf-8")
            except OSError:
                return None
            requested = _requested_params(src)
            for dirpath, names in coupled.items():
                if rel.startswith(dirpath) and (requested & names):
                    return "coupled-test"
    return None


@cache
def _first_party_imports(rel: str) -> frozenset[str]:
    """First-party module names imported by *rel* — memoized per path.

    This is the release gate's import scan (release_public delegates here):
    whole-tree ``ast.walk`` so function-level imports are found (the 0.2.14
    rich_markdown bug was a function-level import), relative imports skipped,
    non-first-party names filtered out.  The tree is immutable during a run,
    so the scan result — not just the parse — is cached; repeated gate
    invocations (gate, export, tests) pay read+parse+walk once per file.
    Missing/unreadable/parse-error files yield an empty set (the gate skips
    malformed files).
    """
    try:
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return frozenset()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return frozenset()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                mods.add(node.module)
    return frozenset(m for m in mods if m.startswith(FIRST_PARTY_PREFIXES))


def _prune_baseline_file(path: Path) -> int:
    """Drop ``<excluded-path>::...`` entries from a copied lint-baseline file.

    Baseline entries are keyed ``<file_path>::...``; flag-only baselines
    (e.g. config_flag_unreachable_baseline.txt) have no ``::`` and pass
    through untouched. Returns the number of entries dropped.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)
    kept: list[str] = []
    dropped = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "::" not in stripped:
            kept.append(line)
            continue
        ref_path = stripped.split("::", 1)[0]
        if is_excluded(ref_path):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        newline = "\n" if text.endswith("\n") else ""
        path.write_text("\n".join(kept) + newline, encoding="utf-8")
    return dropped


def prune_baseline_files(target: Path, shipped: list[str]) -> None:
    for rel in shipped:
        if not (rel.startswith("scripts/") and rel.endswith("_baseline.txt")):
            continue
        dropped = _prune_baseline_file(target / rel)
        if dropped:
            print(f"pruned {dropped} excluded-path entries from {rel}", file=sys.stderr)


def _generate_structural_baseline(target: Path, excluded_paths: list[str]) -> bool:
    """Regenerate scripts/structural_scanner_baseline.txt inside *target*.

    Runs the SNAPSHOT's own structural gate in --dump-candidates mode (its
    copy of scripts/check_structural_scanners.py resolves REPO to *target*,
    so the scan judges exactly the shipped tree, in this interpreter where
    the optional scanner deps — vulture — are installed), then keeps only
    reference-dependent-scanner candidates and verifies EVERY candidate name
    is referenced (word boundary) from at least one EXCLUDED tracked file:
    proof the candidate is an artifact of tree composition — its consumers
    were not shipped — and not dead code in the shipped subset.

    Anything else FAILS the export. The private tree's gate is green (0
    candidates, enforced by pre-commit/CI before any release), so a dumped
    candidate from a zero-tolerance scanner, or a name with no excluded-file
    reference, means a true regression or scanner drift — both need a human,
    never a silent baseline entry.
    """
    # The policy set is read from the sibling gate script (this file lives in
    # scripts/ too) — NOT from REPO, so tests that point REPO at a tmp tree
    # still get the real BASELINE_ALLOWED_SCANNERS single source.
    spec = importlib.util.spec_from_file_location(
        "check_structural_scanners", Path(__file__).resolve().parent / "check_structural_scanners.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    allowed = set(gate.BASELINE_ALLOWED_SCANNERS)

    fd, dump_path = tempfile.mkstemp(suffix=".json", prefix="structural-dump-")
    os.close(fd)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/check_structural_scanners.py",
                "--gate-only",
                "--dump-candidates",
                dump_path,
            ],
            cwd=str(target),
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,  # returncode is judged below (dump mode exits 0 on candidates)
        )
        if proc.returncode != 0:
            print(
                f"error: structural-scanner dump on the snapshot failed (exit {proc.returncode})",
                file=sys.stderr,
            )
            if proc.stdout:
                print("\n".join(proc.stdout.splitlines()[-30:]), file=sys.stderr)
            if proc.stderr:
                print(proc.stderr[-2000:], file=sys.stderr)
            return False
        payload = json.loads(Path(dump_path).read_text(encoding="utf-8"))
    finally:
        Path(dump_path).unlink(missing_ok=True)
        # The dump run warms snapshot-local analyzer caches (.cache/) and
        # __pycache__ dirs; the snapshot must ship clean.
        shutil.rmtree(target / ".cache", ignore_errors=True)

    excluded_texts: list[str] = []
    for rel in excluded_paths:
        try:
            excluded_texts.append((REPO / rel).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue

    def referenced_from_excluded(name: str) -> bool:
        pat = re.compile(rf"\b{re.escape(name)}\b")
        return any(pat.search(t) for t in excluded_texts)

    entries: set[tuple[str, str, str]] = set()
    violations: list[str] = []
    for cand in payload.get("candidates", []):
        scanner = str(cand.get("scanner") or "")
        rel = str(cand.get("file") or "")
        names = [str(n) for n in (cand.get("names") or []) if n]
        label = f"{scanner}::{rel}::{','.join(names) or '?'}"
        if scanner not in allowed:
            violations.append(f"{label} — zero-tolerance scanner has candidates in the shipped tree")
            continue
        if not rel or not names:
            violations.append(f"{label} — unkeyable candidate shape (empty file/names)")
            continue
        for n in names:
            if referenced_from_excluded(n):
                entries.add((scanner, rel, n))
            else:
                violations.append(
                    f"{scanner}::{rel}::{n} — no reference from any excluded file "
                    "(true dead code in the shipped subset, or scanner drift)"
                )
    if violations:
        for v in violations:
            print(f"error: structural baseline: {v}", file=sys.stderr)
        return False

    lines = [
        "# asicode structural-scanner export-artifact baseline — MACHINE-GENERATED.",
        "#",
        "# Regenerated by scripts/export_public.py at every export; DO NOT hand-edit.",
        "# Entries are reference-dependent-scanner candidates that exist ONLY because",
        "# this tree is a filtered snapshot: each symbol's consumers live in files the",
        "# export excludes (webapp/, tools/, tasks/, coupled tests), so the shipped",
        "# subset reports them as unreferenced while the full private tree keeps them",
        "# live. Every entry was verified at generation time to be referenced from at",
        "# least one excluded file. Zero-tolerance scanners (contradictory_logic,",
        "# duplicate_definition, unused_import, ast_similarity) never appear here —",
        "# a baseline entry naming them fails the gate.",
        "# Format: <scanner>::<file>::<symbol>",
    ]
    lines += sorted(f"{s}::{rel}::{n}" for s, rel, n in entries)
    (target / "scripts" / "structural_scanner_baseline.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    per_scanner = ", ".join(
        f"{s}x{sum(1 for e in entries if e[0] == s)}" for s in sorted({e[0] for e in entries})
    ) or "none"
    print(
        f"structural baseline: {len(entries)} verified export-artifact entries ({per_scanner}) "
        "-> scripts/structural_scanner_baseline.txt",
        file=sys.stderr,
    )
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1]).resolve()
    dry_run = "--list" in sys.argv[2:]

    shipped: list[str] = []
    excluded: dict[str, int] = {}
    excluded_paths: list[str] = []
    for rel in tracked_files():
        reason = is_excluded(rel)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            excluded_paths.append(rel)
        else:
            shipped.append(rel)

    if dry_run:
        for rel in shipped:
            print(rel)
    else:
        if target.exists() and any(target.iterdir()):
            print(f"error: target {target} exists and is not empty", file=sys.stderr)
            return 1
        for rel in shipped:
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)
        prune_baseline_files(target, shipped)
        if not _generate_structural_baseline(target, excluded_paths):
            return 1

    print(f"\nshipped: {len(shipped)} files -> {target}", file=sys.stderr)
    for reason, n in sorted(excluded.items(), key=lambda kv: -kv[1]):
        print(f"excluded [{reason}]: {n} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
