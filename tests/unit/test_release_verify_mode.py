"""P6: release_public --verify runs the public-CI gate mirror BEFORE commit.

Motivating incident (v0.2.25 hotfix, first-attempt push): the release flow
verified the structural baseline but NOT the snapshot's own gates/unit tests —
export_public.py:402 had ``errors='ignore'``, a violation of the P26-3 policy
unit gate that SHIPS in the public tree, so the failure only surfaced in
public CI *after* the push (a second push fixed it). ``--verify`` closes that
gap locally: the gates run on the STAGED release inside the public repo
(identical ``--index-only`` semantics to CI — the index holds exactly the
bytes this release would push), and any failure aborts before
commit/tag/push, breaking the mask chain at the cheapest possible point.

The same change fixes a discovered drift bug: release_public.py hand-rolled
its own snapshot copy (tracked files + baseline prune) and never regenerated
``scripts/structural_scanner_baseline.txt`` — a machine-generated export
artifact that must NOT exist in the private tree. The next release through
this script would have DELETED the baseline from the public repo (its sync
removes files outside the shipped set) and reddened the structural gate with
24 unsuppressed candidates. Snapshot construction now goes through the shared
``export_public.build_snapshot()`` used by ``export_public.main()``.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "release_public.py"
VERSION = "9.9.9"


@pytest.fixture
def relpub():
    """Load release_public.py as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("_relpub_verify_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(cwd: Path, *a: str) -> None:
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def fake_repos(relpub, tmp_path, monkeypatch) -> tuple[Path, Path]:
    """A minimal private repo (version + CHANGELOG + one shipped module) and an
    empty public git repo, with release_public pointed at the fake private.

    export_public is the SHARED sys.modules object across all tests — every
    patch goes through monkeypatch so it is restored afterwards.
    """
    priv = tmp_path / "private"
    priv.mkdir()
    (priv / "pyproject.toml").write_text(f'version = "{VERSION}"\n', encoding="utf-8")
    (priv / "CHANGELOG.md").write_text(
        f"# log\n\n## [{VERSION}] — 2026-08-19\n- test release\n", encoding="utf-8"
    )
    (priv / "alpha_shipped.py").write_text("ALPHA = 1\n", encoding="utf-8")
    _git(priv, "init", "-b", "main")
    _git(priv, "config", "user.email", "t@example.com")
    _git(priv, "config", "user.name", "t")
    _git(priv, "add", "-A")
    _git(priv, "commit", "-m", "init")

    pub = tmp_path / "public"
    pub.mkdir()
    _git(pub, "init", "-b", "main")
    _git(pub, "config", "user.email", "t@example.com")
    _git(pub, "config", "user.name", "t")

    monkeypatch.setattr(relpub, "REPO", priv)
    monkeypatch.setattr(relpub.export_public, "REPO", priv)
    return priv, pub


@pytest.fixture
def fake_baseline_gen(relpub, monkeypatch) -> list[tuple[str, tuple[str, ...]]]:
    """Replace the real structural-baseline generation (vulture + five
    scanners — slow) with one that writes the artifact the sync must ship.
    Returns the recorded call list for wiring assertions."""
    calls: list[tuple[str, tuple[str, ...]]] = []

    def gen(target: Path, excluded_paths: list[str]) -> bool:
        calls.append((target.as_posix(), tuple(excluded_paths)))
        (target / "scripts").mkdir(parents=True, exist_ok=True)
        (target / "scripts" / "structural_scanner_baseline.txt").write_text(
            "# fake machine-generated baseline\n", encoding="utf-8"
        )
        return True

    monkeypatch.setattr(relpub.export_public, "_generate_structural_baseline", gen)
    return calls


def _set_gates(monkeypatch, relpub, *, ok: bool) -> None:
    # Gate entries carry script args (the runner prepends sys.executable,
    # same as the real _VERIFY_GATES — copy-pasteable from lint.yml steps).
    code = "import sys; sys.exit(0)" if ok else "import sys; sys.exit(3)"
    monkeypatch.setattr(relpub, "_VERIFY_GATES", [("stub-gate", ["-c", code])])


def _write_suite_test(root: Path, name: str, body: str) -> None:
    d = root / "tests" / "unit"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


# ── CLI surface ────────────────────────────────────────────────────────────

def test_parser_verify_flag_forms(relpub):
    p = relpub._build_arg_parser()
    assert p.parse_args(["/tmp/pub", "--verify"]).verify == "fast"
    a = p.parse_args(["/tmp/pub", "--verify=full"])
    assert a.verify == "full" and a.public_repo == "/tmp/pub"
    a = p.parse_args(["/tmp/pub", "--verify", "full"])
    assert a.verify == "full" and a.public_repo == "/tmp/pub"
    assert p.parse_args(["/tmp/pub"]).verify is None
    a = p.parse_args(["/tmp/pub", "--tag", "--push", "--allow-dirty"])
    assert a.tag and a.push and a.allow_dirty and a.verify is None


def test_parser_rejects_unknown_verify_mode(relpub):
    with pytest.raises(SystemExit) as ei:
        relpub._build_arg_parser().parse_args(["/tmp/pub", "--verify=bogus"])
    assert ei.value.code == 2


def test_parser_verify_before_positional_fails_closed(relpub):
    # '--verify <path>' would swallow the path as the mode value — argparse's
    # choices reject it (exit 2) instead of misparsing the release target.
    with pytest.raises(SystemExit) as ei:
        relpub._build_arg_parser().parse_args(["--verify", "/tmp/pub"])
    assert ei.value.code == 2


# ── _verify_release dispatch ───────────────────────────────────────────────

def test_verify_release_reports_failing_gate(relpub, monkeypatch, tmp_path, capsys):
    _set_gates(monkeypatch, relpub, ok=False)
    monkeypatch.setattr(relpub, "_FAST_POLICY_TESTS", [])
    assert relpub._verify_release(tmp_path, "fast") is False
    assert "stub-gate" in capsys.readouterr().out


def test_verify_release_fast_runs_policy_unit_subset(relpub, monkeypatch, tmp_path):
    _set_gates(monkeypatch, relpub, ok=True)
    _write_suite_test(
        tmp_path, "test_fake_policy.py", "def test_policy():\n    assert False, 'policy violation'\n"
    )
    monkeypatch.setattr(relpub, "_FAST_POLICY_TESTS", ["tests/unit/test_fake_policy.py"])
    assert relpub._verify_release(tmp_path, "fast") is False
    _write_suite_test(
        tmp_path, "test_fake_policy.py", "def test_policy():\n    assert True\n"
    )
    assert relpub._verify_release(tmp_path, "fast") is True


def test_full_mode_runs_unit_suite_beyond_policy_subset(relpub, monkeypatch, tmp_path):
    _set_gates(monkeypatch, relpub, ok=True)
    _write_suite_test(
        tmp_path, "test_fake_suite.py", "def test_suite():\n    assert False\n"
    )
    monkeypatch.setattr(relpub, "_FAST_POLICY_TESTS", [])
    assert relpub._verify_release(tmp_path, "fast") is True   # suite not in fast scope
    assert relpub._verify_release(tmp_path, "full") is False  # full sweeps tests/unit


# ── shared snapshot construction (drift-bug fix) ───────────────────────────

def test_fast_policy_subset_actually_ships(relpub):
    """Every _FAST_POLICY_TESTS entry must ship in the public snapshot.

    A coupled-test entry (excluded from the export — the sse/tools-git/
    ui-route gates read webapp//tools/) would make --verify=fast fail with
    pytest exit 4 "file not found" on every release: a false red that blocks
    shipping. This invariant is what keeps the enumerated list honest as new
    tree-policy tests are added.
    """
    exp = relpub.export_public
    for rel in relpub._FAST_POLICY_TESTS:
        assert (REPO / rel).is_file(), f"_FAST_POLICY_TESTS entry missing locally: {rel}"
        reason = exp.is_excluded(rel)
        assert reason is None, f"_FAST_POLICY_TESTS entry does not ship ({reason}): {rel}"


def test_build_snapshot_is_the_shared_sequence(relpub, monkeypatch, tmp_path):
    exp = relpub.export_public
    calls: list[tuple[str, tuple[str, ...]]] = []

    def gen(target: Path, excluded_paths: list[str]) -> bool:
        calls.append((target.as_posix(), tuple(excluded_paths)))
        return True

    monkeypatch.setattr(exp, "_generate_structural_baseline", gen)
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "a.py").write_text("A = 1\n", encoding="utf-8")
    monkeypatch.setattr(exp, "REPO", fake_src)

    snap = tmp_path / "snap"
    assert exp.build_snapshot(snap, ["a.py"], ["x/excluded.py"]) is True
    assert (snap / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert calls == [(snap.as_posix(), ("x/excluded.py",))]

    # Generation failure must propagate — the release aborts.
    monkeypatch.setattr(exp, "_generate_structural_baseline", lambda t, e: False)
    assert exp.build_snapshot(tmp_path / "snap2", ["a.py"], []) is False


# ── end-to-end flow: verify sits between staging and the commit ────────────

def test_verify_failure_aborts_before_commit(
    fake_repos, relpub, fake_baseline_gen, monkeypatch
):
    _, pub = fake_repos
    _set_gates(monkeypatch, relpub, ok=False)
    monkeypatch.setattr(relpub, "_FAST_POLICY_TESTS", [])
    assert relpub.main([str(pub), "--verify"]) == 1
    # No commit happened — HEAD does not resolve in the unborn branch.
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=pub, capture_output=True, check=False
    )
    assert head.returncode != 0
    # The staged release is preserved for inspection, not destroyed.
    staged = subprocess.run(
        ["git", "status", "--porcelain"], cwd=pub, capture_output=True, text=True, check=False
    ).stdout
    assert "alpha_shipped.py" in staged


def test_verify_success_commits_and_ships_generated_baseline(
    fake_repos, relpub, fake_baseline_gen, monkeypatch
):
    _, pub = fake_repos
    _set_gates(monkeypatch, relpub, ok=True)
    monkeypatch.setattr(relpub, "_FAST_POLICY_TESTS", [])
    assert relpub.main([str(pub), "--verify"]) == 0
    # The regenerated export artifact ships with the release (drift-bug fix:
    # the old hand-rolled copy never produced it, and the sync would have
    # deleted it from the public repo).
    assert (pub / "scripts" / "structural_scanner_baseline.txt").exists()
    assert (pub / "alpha_shipped.py").exists()
    subject = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=pub, capture_output=True, text=True, check=False
    ).stdout.strip()
    assert subject.startswith(f"release: v{VERSION} (source snapshot ")
