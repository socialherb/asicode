"""Export-side structural baseline generation (scripts/export_public.py).

``_generate_structural_baseline`` runs the SNAPSHOT's own gate in
--dump-candidates mode, keeps only reference-dependent-scanner candidates,
and machine-verifies every candidate name is referenced (word boundary) from
at least one EXCLUDED tracked file.  These tests pin the verifier's
fail-closed contract with a stubbed subprocess: a verified artifact writes
the baseline; a zero-tolerance scanner candidate or an unverifiable name
FAILS the export (the private gate is green, so those mean regression or
scanner drift — never a silent baseline entry).
"""

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_public.py"
_spec = importlib.util.spec_from_file_location("export_public", _SCRIPT)
x = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(x)  # type: ignore[union-attr]


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_dump(monkeypatch, candidates, *, returncode=0):
    """Stub the snapshot gate subprocess: write *candidates* as the dump JSON."""

    def fake_run(args, **kwargs):
        dump_path = args[args.index("--dump-candidates") + 1]
        Path(dump_path).write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
        return _Proc(returncode=returncode)

    monkeypatch.setattr(x.subprocess, "run", fake_run)


def _private_repo(tmp_path, monkeypatch, excluded_content: str):
    """Point the export module's REPO at a tmp tree with one excluded file."""
    pkg = tmp_path / "excl_pkg"
    pkg.mkdir()
    (pkg / "consumer.py").write_text(excluded_content, encoding="utf-8")
    monkeypatch.setattr(x, "REPO", tmp_path)


def test_verified_artifact_writes_baseline(tmp_path, monkeypatch):
    _private_repo(tmp_path, monkeypatch, "from config import LEGACY_THING\nprint(LEGACY_THING)\n")
    _stub_dump(
        monkeypatch,
        [{"scanner": "public_dead_code_scanner", "file": "config.py", "names": ["LEGACY_THING"]}],
    )
    target = tmp_path / "target"
    (target / "scripts").mkdir(parents=True)
    assert x._generate_structural_baseline(target, ["excl_pkg/consumer.py"]) is True
    text = (target / "scripts" / "structural_scanner_baseline.txt").read_text(encoding="utf-8")
    assert "MACHINE-GENERATED" in text
    assert "public_dead_code_scanner::config.py::LEGACY_THING" in text
    # the dump run's snapshot-local caches must not ship
    assert not (target / ".cache").exists()


def test_zero_tolerance_scanner_candidate_fails_export(tmp_path, monkeypatch, capsys):
    _private_repo(tmp_path, monkeypatch, "import os\n")
    _stub_dump(monkeypatch, [{"scanner": "unused_import_scanner", "file": "a.py", "names": ["os"]}])
    target = tmp_path / "target"
    (target / "scripts").mkdir(parents=True)
    assert x._generate_structural_baseline(target, ["excl_pkg/consumer.py"]) is False
    assert "zero-tolerance" in capsys.readouterr().err


def test_unreferenced_name_fails_export(tmp_path, monkeypatch, capsys):
    """A candidate whose name appears in NO excluded file is not an export
    artifact — with the private gate green it can only be true dead code in
    the shipped subset or scanner drift, so the export fails."""
    _private_repo(tmp_path, monkeypatch, "unrelated = 1\n")
    _stub_dump(monkeypatch, [{"scanner": "vulture_dead_code_scanner", "file": "config.py", "names": ["BENCH_GONE"]}])
    target = tmp_path / "target"
    (target / "scripts").mkdir(parents=True)
    assert x._generate_structural_baseline(target, ["excl_pkg/consumer.py"]) is False
    assert "no reference from any excluded file" in capsys.readouterr().err


def test_word_boundary_not_substring(tmp_path, monkeypatch):
    """Verification is word-boundary, so BENCH_RAW_LLM_2 must NOT verify a
    BENCH_RAW_LLM candidate via substring noise in an excluded file."""
    _private_repo(tmp_path, monkeypatch, "BENCH_RAW_LLM_2 = 1\n")
    _stub_dump(monkeypatch, [{"scanner": "public_dead_code_scanner", "file": "config.py", "names": ["BENCH_RAW_LLM"]}])
    target = tmp_path / "target"
    (target / "scripts").mkdir(parents=True)
    assert x._generate_structural_baseline(target, ["excl_pkg/consumer.py"]) is False


def test_dump_failure_fails_export(tmp_path, monkeypatch, capsys):
    _private_repo(tmp_path, monkeypatch, "x = 1\n")
    _stub_dump(monkeypatch, [], returncode=1)
    target = tmp_path / "target"
    (target / "scripts").mkdir(parents=True)
    assert x._generate_structural_baseline(target, ["excl_pkg/consumer.py"]) is False
    assert "dump on the snapshot failed" in capsys.readouterr().err


def test_empty_dump_writes_header_only_baseline(tmp_path, monkeypatch):
    """A shipped tree with zero reference-dependent candidates still gets the
    file (header only) — presence, not emptiness, carries the contract."""
    _private_repo(tmp_path, monkeypatch, "x = 1\n")
    _stub_dump(monkeypatch, [])
    target = tmp_path / "target"
    (target / "scripts").mkdir(parents=True)
    assert x._generate_structural_baseline(target, ["excl_pkg/consumer.py"]) is True
    lines = (target / "scripts" / "structural_scanner_baseline.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#") and all(ln.startswith("#") for ln in lines)
