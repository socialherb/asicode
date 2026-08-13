"""semantic_lint availability-cache semantics (G1 fix).

The ruff availability probe must cache only POSITIVE results:
- a mid-session ruff install (this agent bootstraps tools at runtime)
  must be picked up by later calls — a cached negative would silently
  disable semantic lint for the whole session (the G1 defect);
- a broken ruff binary (nonzero rc) must not poison the cache either.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from external_llm.agent import semantic_lint


@pytest.fixture(autouse=True)
def _reset_availability_cache(monkeypatch):
    """Every test starts with an unprobed availability state."""
    monkeypatch.setattr(semantic_lint, "_RUFF_AVAILABLE", None)


def _proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_positive_result_is_cached(monkeypatch):
    """Healthy ruff (rc=0) → True, and later calls never re-probe."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _proc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert semantic_lint._check_ruff_available() is True
    assert semantic_lint._check_ruff_available() is True
    assert len(calls) == 1  # cached — no second probe


def test_missing_ruff_not_cached_mid_session_install_detected(monkeypatch):
    """FileNotFoundError → False, but a later install flips to True (G1 ②)."""
    state = {"present": False}

    def fake_run(cmd, **kwargs):
        if not state["present"]:
            raise FileNotFoundError("no ruff on PATH")
        return _proc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert semantic_lint._check_ruff_available() is False
    state["present"] = True  # ruff installed mid-session
    assert semantic_lint._check_ruff_available() is True


def test_broken_ruff_nonzero_rc_not_cached(monkeypatch):
    """rc != 0 (broken binary) → False, and a fixed binary is detected (G1 ①)."""
    state = {"rc": 1}

    def fake_run(cmd, **kwargs):
        return _proc(returncode=state["rc"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert semantic_lint._check_ruff_available() is False
    state["rc"] = 0  # binary fixed
    assert semantic_lint._check_ruff_available() is True


def test_timeout_and_permission_errors_not_cached(monkeypatch):
    """Transient probe failures also stay uncached — retried next call."""
    raised = {"times": 0}

    def fake_run(cmd, **kwargs):
        raised["times"] += 1
        if raised["times"] == 1:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 5))
        raise PermissionError("denied")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert semantic_lint._check_ruff_available() is False
    assert semantic_lint._check_ruff_available() is False  # re-probed, not cached
    assert raised["times"] == 2


def test_ruff_findings_returns_empty_when_unavailable(monkeypatch):
    """Availability gate: [] with only the cheap probe, no full run."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise FileNotFoundError("no ruff on PATH")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert semantic_lint.ruff_findings("x = 1\n") == []
    assert len(calls) == 1  # probe only — the check run is never spawned


def test_ruff_findings_normalizes_findings(monkeypatch):
    """rc=1 (findings) + JSON → normalized dicts with int line, stdin wired."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs.get("input")
        payload = json.dumps([
            {
                "code": "F401",
                "location": {"row": 3, "column": 1},
                "message": "`os` imported but unused",
            },
        ])
        return _proc(returncode=1, stdout=payload)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(semantic_lint, "_RUFF_AVAILABLE", True)  # skip probe
    findings = semantic_lint.ruff_findings("import os\n\nx = 1\n", path="t.py")
    assert findings == [
        {"code": "F401", "line": 3, "message": "`os` imported but unused"},
    ]
    assert seen["input"] == "import os\n\nx = 1\n"
    assert "--stdin-filename" in seen["cmd"] and "t.py" in seen["cmd"]


def test_ruff_findings_nonstandard_rc_returns_empty(monkeypatch):
    """rc outside {0, 1} (ruff crashed) degrades gracefully to []."""
    monkeypatch.setattr(semantic_lint, "_RUFF_AVAILABLE", True)
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: _proc(returncode=2, stderr="boom")
    )
    assert semantic_lint.ruff_findings("x = 1\n") == []


def test_ruff_findings_bad_json_returns_empty(monkeypatch):
    """Unparseable stdout degrades gracefully to []."""
    monkeypatch.setattr(semantic_lint, "_RUFF_AVAILABLE", True)
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: _proc(returncode=0, stdout="not json")
    )
    assert semantic_lint.ruff_findings("x = 1\n") == []
