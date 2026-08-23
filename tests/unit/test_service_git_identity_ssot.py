"""R4: ExternalLLMService._get_git_identity_best_effort delegates to the
agent-wide git snapshot SSOT (``agent_context_manager.get_git_snapshot``)
instead of spawning two uncached ``git rev-parse`` subprocesses per
LLM-context build.

NOTE on monkeypatch targets: ``service.py`` binds the SSOT function via a
module-level ``from ... import get_git_snapshot``, so its namespace holds its
OWN reference — patches must target ``external_llm.service.get_git_snapshot``,
not ``agent_context_manager.get_git_snapshot``. (Contrast: ``tool_failure_log``
imports the SSOT LAZILY inside ``_git_sha``, so it re-resolves the
``agent_context_manager`` attribute each call — see
``test_git_sha_delegates_to_snapshot_and_collapses_burst``.)
"""

import external_llm.service as svc_mod
from external_llm.agent import agent_context_manager as acm
from external_llm.service import ExternalLLMService


def test_git_identity_uses_snapshot_and_skips_direct_git(monkeypatch, tmp_path):
    """The SSOT path returns branch+head_commit from the snapshot and must NOT
    touch the direct ``_git_cmd_best_effort`` subprocess path at all — that is
    the whole point of consolidating onto the cached snapshot."""
    acm._clear_git_cache()
    monkeypatch.setattr(
        svc_mod,
        "get_git_snapshot",
        lambda rr: {"branch": "feature-x", "head_hash": "0123456789abcdef"},
    )
    direct = {"n": 0}

    def _direct_spy(rr, args):
        direct["n"] += 1
        return ""

    monkeypatch.setattr(ExternalLLMService, "_git_cmd_best_effort", staticmethod(_direct_spy))
    try:
        ident = ExternalLLMService._get_git_identity_best_effort(str(tmp_path))
        assert ident == {"branch": "feature-x", "head_commit": "0123456789abcdef"}
        assert direct["n"] == 0, "direct git path must be bypassed when SSOT succeeds"
    finally:
        acm._clear_git_cache()


def test_git_identity_falls_back_on_snapshot_error(monkeypatch, tmp_path):
    """If the SSOT raises (e.g. transient import failure), the method falls
    back to the direct ``_git_cmd_best_effort`` path so this best-effort
    metadata never hard-fails."""

    def _boom(rr):
        raise RuntimeError("SSOT unavailable")

    monkeypatch.setattr(svc_mod, "get_git_snapshot", _boom)
    monkeypatch.setattr(
        ExternalLLMService,
        "_git_cmd_best_effort",
        staticmethod(lambda rr, args: "main" if "--abbrev-ref" in args else "deadbeef"),
    )
    ident = ExternalLLMService._get_git_identity_best_effort(str(tmp_path))
    assert ident == {"branch": "main", "head_commit": "deadbeef"}


def test_git_identity_empty_repo_root():
    """An empty repo_root yields empty fields — get_git_snapshot("") returns {}
    (early-return guard), mapped to empty branch/head_commit."""
    acm._clear_git_cache()
    try:
        ident = ExternalLLMService._get_git_identity_best_effort("")
        assert ident == {"branch": "", "head_commit": ""}
    finally:
        acm._clear_git_cache()
