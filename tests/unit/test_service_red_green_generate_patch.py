"""RED→GREEN: ExternalLLMService.generate_patch full branch coverage.

Baseline: service.py 34% (832 stmts / 547 miss). generate_patch is the bulk
of the missing surface (lines ~787-1518). Strategy: fake LLM client (chat
capture + scripted responses) + config-driven fake PatchEngine / ASTRewriter /
symbol searcher / SemanticPatchEngine / SuperContextBuilder.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

import external_llm.service as svc_mod
from external_llm.client import LLMClientError
from external_llm.service import ExternalLLMService

_DIFF = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-old\n+new\n"
_FILE_BLOCK = "FILE: x.txt\n```python\nold\nnew\n```\n"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeClient:
    """Scripted LLM client. chat() pops the next response (last one repeats)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.chat_calls: list[dict] = []

    def chat(
        self,
        messages,
        model=None,
        temperature=None,
        max_tokens=None,
        thinking_mode=False,
        reasoning_effort=None,
        reasoning_callback=None,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


class _FakePatchEngine:
    """Config-driven stand-in for patch_engine.PatchEngine (module name)."""

    synth_result = None  # SimpleNamespace(success, patch_applied, metadata, error)
    synth_raise = None  # exception to raise from synthesize_and_apply
    repair_raise = None  # exception to raise from repair_patch
    file_blocks = ("", None)  # (patch, reason) from _try_synthesize_diff_from_file_blocks
    git_check = (True, None)  # (ok, err); callable allowed for sequencing
    repair_result = None  # SimpleNamespace(success, patch_applied) or None
    normalize = None  # (norm, err); None -> identity
    instances: ClassVar[list] = []

    def __init__(self, *a, **kw):
        _FakePatchEngine.instances.append(self)

    def synthesize_and_apply(self, llm_output, target_file, output_mode):
        if _FakePatchEngine.synth_raise is not None:
            raise _FakePatchEngine.synth_raise
        return _FakePatchEngine.synth_result

    def _try_synthesize_diff_from_file_blocks(self, repo_root, target_file, llm_text):
        return _FakePatchEngine.file_blocks

    def _git_apply_check_best_effort(self, patch_text):
        gc = _FakePatchEngine.git_check
        if callable(gc):
            return gc()
        if isinstance(gc, list):
            return gc.pop(0)()
        return gc

    def repair_patch(self, patch_text, target_file, fail_kind, llm_out):
        if _FakePatchEngine.repair_raise is not None:
            raise _FakePatchEngine.repair_raise
        return _FakePatchEngine.repair_result

    def normalize_and_validate(self, patch_text, target_file):
        if _FakePatchEngine.normalize is not None:
            return _FakePatchEngine.normalize
        return patch_text, None


class _FakeRewriter:
    """ASTRewriter stand-in; mode controls which replace_* produces a patch."""

    mode = "fail"  # fail | function | class | method | autodetect | raise

    def __init__(self, rr):
        pass

    def _patch(self):
        return _DIFF if _FakeRewriter.mode != "fail" else ""

    def replace_function(self, tgt, name, code):
        if _FakeRewriter.mode == "raise":
            raise RuntimeError("rewrite boom")
        return SimpleNamespace() if _FakeRewriter.mode == "function" else None

    def replace_class(self, tgt, name, code):
        return SimpleNamespace() if _FakeRewriter.mode == "class" else None

    def replace_method(self, tgt, class_name, method_name, code):
        return SimpleNamespace() if _FakeRewriter.mode == "method" else None

    def generate_patch(self, tgt, result):
        return self._patch()


class _FakeSearcher:
    found: ClassVar[list] = []
    fuzzy: object = None
    raise_find = False

    def __init__(self, rr):
        pass

    def find_symbol(self, name, kind=None):
        if _FakeSearcher.raise_find:
            raise RuntimeError("search boom")
        return _FakeSearcher.found

    def fuzzy_find_symbol(self, name):
        return _FakeSearcher.fuzzy


class _FakeSemantic:
    patch: object = None  # None -> raise; else generate_patch returns it

    def __init__(self, rr):
        pass

    def apply_semantic_patch(self, file_path, new_code):
        if _FakeSemantic.patch is None:
            raise RuntimeError("no semantic patch")
        return SimpleNamespace(kind="function")

    def generate_patch(self, tgt, result):
        return _FakeSemantic.patch or ""


class _FakeSuperBuilder:
    text = "SUPER"

    def __init__(self, rr):
        pass

    def build_context(self, user_request=None, target_file=None):
        return _FakeSuperBuilder.text


@pytest.fixture
def svc(monkeypatch):
    """Service with fake client + all internal collaborators replaced."""
    monkeypatch.setattr(svc_mod, "get_git_snapshot", lambda rr: {})
    monkeypatch.setattr(svc_mod, "PatchEngine", _FakePatchEngine)
    monkeypatch.setattr("external_llm.ast_rewrite.ASTRewriter", _FakeRewriter)
    monkeypatch.setattr("external_llm.agent.symbol_search.get_symbol_searcher", _FakeSearcher)
    monkeypatch.setattr("external_llm.semantic_patch.SemanticPatchEngine", _FakeSemantic)
    monkeypatch.setattr(svc_mod, "SuperContextBuilder", _FakeSuperBuilder)
    monkeypatch.setattr(svc_mod, "ContextBuilder", _FakeSuperBuilder)
    _FakePatchEngine.instances = []
    _FakePatchEngine.synth_result = None
    _FakePatchEngine.synth_raise = None
    _FakePatchEngine.file_blocks = ("", None)
    _FakePatchEngine.git_check = (True, None)
    _FakePatchEngine.repair_result = None
    _FakePatchEngine.repair_raise = None
    _FakePatchEngine.normalize = None
    _FakeRewriter.mode = "fail"
    _FakeSearcher.found = []
    _FakeSearcher.fuzzy = None
    _FakeSearcher.raise_find = False
    _FakeSemantic.patch = None
    _FakeSuperBuilder.text = "SUPER"
    fake = _FakeClient()
    monkeypatch.setattr(svc_mod, "create_llm_client", lambda **kw: fake)
    return ExternalLLMService(provider="openai", api_key="k", model="m"), fake


def _resp(content: str, tokens: int | None = None):
    return SimpleNamespace(content=content, tokens_used=tokens)


def _setup_target(tmp_path, content="old\n"):
    (tmp_path / "x.txt").write_text(content, encoding="utf-8")
    return str(tmp_path), "x.txt"


# ---------------------------------------------------------------------------
# early returns / argument normalization
# ---------------------------------------------------------------------------


def test_generate_patch_empty_repo_root(svc):
    svc_obj, _ = svc
    out = svc_obj.generate_patch("", "fix")
    assert out["success"] is False
    assert out["error"] == "repo_root is empty"
    assert out["meta"]["reason"] == "bad_input"


def test_generate_patch_unknown_mode_falls_back_to_diff(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_DIFF)]
    svc_obj.generate_patch(rr, "add a comment", tgt, output_mode="BOGUS")
    sys_prompt = fake.chat_calls[0]["messages"][0].content
    assert "unified diff" in sys_prompt  # diff-mode prompt used


def test_generate_patch_noop_precheck(tmp_path, monkeypatch):
    rr, tgt = _setup_target(tmp_path, content="marker_xyz here\n")
    svc_obj, _ = svc_fx(monkeypatch)
    out = svc_obj.generate_patch(rr, 'add "marker_xyz" to the file', tgt)
    assert out["success"] is True
    assert out["meta"]["reason"] == "noop"
    assert out["meta"]["noop_trust_level"] == "high"


def svc_fx(monkeypatch):
    """Standalone service fixture reuse (no FakeClient responses needed)."""
    monkeypatch.setattr(svc_mod, "get_git_snapshot", lambda rr: {})
    monkeypatch.setattr(svc_mod, "PatchEngine", _FakePatchEngine)
    monkeypatch.setattr("external_llm.ast_rewrite.ASTRewriter", _FakeRewriter)
    monkeypatch.setattr("external_llm.agent.symbol_search.get_symbol_searcher", _FakeSearcher)
    monkeypatch.setattr("external_llm.semantic_patch.SemanticPatchEngine", _FakeSemantic)
    monkeypatch.setattr(svc_mod, "SuperContextBuilder", _FakeSuperBuilder)
    monkeypatch.setattr(svc_mod, "ContextBuilder", _FakeSuperBuilder)
    fake = _FakeClient()
    monkeypatch.setattr(svc_mod, "create_llm_client", lambda **kw: fake)
    return ExternalLLMService(provider="openai", api_key="k", model="m"), fake


# ---------------------------------------------------------------------------
# success paths
# ---------------------------------------------------------------------------


def test_generate_patch_noop_llm_output_low_trust(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp("NOOP")]
    out = svc_obj.generate_patch(rr, "add a comment", tgt)
    assert out["success"] is True
    assert out["patch"] == ""
    assert out["meta"]["reason"] == "noop"
    assert out["meta"]["noop_trust_level"] == "low"


def test_generate_patch_noop_trivial_medium_trust(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp("NOOP")]
    out = svc_obj.generate_patch(rr, "fix the typo", tgt)
    assert out["meta"]["noop_trust_level"] == "medium"


def test_generate_patch_fast_path_success(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp("whatever")]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True
    assert out["meta"]["synth_reason"] == "patch_engine_auto:auto"
    assert out["meta"]["reason"] == "ok"


def test_generate_patch_fast_path_metadata_reason(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp("whatever")]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "fast_apply", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["meta"]["synth_reason"] == "patch_engine_auto:fast_apply"


def test_generate_patch_fast_path_failure_falls_to_legacy(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=False,
        patch_applied="",
        metadata={},
        error="boom",
    )
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True
    assert out["meta"]["reason"] == "ok_synth"


def test_generate_patch_fast_path_exception_falls_to_legacy(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.synth_raise = RuntimeError("engine crash")
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True


def test_generate_patch_legacy_ast_function(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeRewriter.mode = "function"
    fake.responses = [_resp("FUNCTION: foo\n" + _FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True
    assert out["meta"]["synth_reason"] == "ast_function"


def test_generate_patch_legacy_ast_class(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeRewriter.mode = "class"
    fake.responses = [_resp("CLASS: MyClass\n" + _FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["meta"]["synth_reason"] == "ast_class"


def test_generate_patch_legacy_ast_method(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeRewriter.mode = "method"
    fake.responses = [_resp("METHOD: MyClass.my_method\n" + _FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["meta"]["synth_reason"] == "ast_method"


def test_generate_patch_legacy_ast_autodetect(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeRewriter.mode = "autodetect"
    # header is not FUNCTION:/CLASS:/METHOD: -> is_function_def(new_code) branch;
    # new_code is "def foo():..." so extract_symbol_name finds "foo".
    llm_out = "def foo():\nFILE: x.txt\n```python\ndef foo():\n    pass\n```\n"
    fake.responses = [_resp(llm_out)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    # header is not FUNCTION:/CLASS:/METHOD: -> is_function_def(new_code) branch;
    # new_code is "def foo():" so extract_symbol_name finds "foo".
    assert out["success"] is True
    assert out["meta"]["synth_reason"] == "ast_autodetect"


def test_generate_patch_symbol_search_fallback(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeRewriter.mode = "function"
    _FakeSearcher.found = [SimpleNamespace(kind="function", name="foo", file="x.txt")]
    fake.responses = [_resp("def foo():\n" + _FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["meta"]["synth_reason"] == "ast_symbol_function"


def test_generate_patch_symbol_search_class(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeRewriter.mode = "class"
    _FakeSearcher.found = [SimpleNamespace(kind="class", name="MyClass", file="x.txt")]
    fake.responses = [_resp("class MyClass:\n" + _FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["meta"]["synth_reason"] == "ast_symbol_class"


def test_generate_patch_symbol_search_fuzzy(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeRewriter.mode = "function"
    _FakeSearcher.found = []
    _FakeSearcher.fuzzy = SimpleNamespace(kind="function", name="foo", file="x.txt")
    fake.responses = [_resp("def foo():\n" + _FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["meta"]["synth_reason"] == "ast_symbol_function"


def test_generate_patch_semantic_fallback(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeSemantic.patch = _DIFF
    fake.responses = [_resp(_FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["meta"]["synth_reason"] == "semantic_function"


def test_generate_patch_semantic_class_kind(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeSemantic.patch = _DIFF

    class _ClassSem(_FakeSemantic):
        def apply_semantic_patch(self, file_path, new_code):
            return SimpleNamespace(kind="class")

    _FakeSemantic.apply_semantic_patch = _ClassSem.apply_semantic_patch
    fake.responses = [_resp(_FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["meta"]["synth_reason"] == "semantic_class"


def test_generate_patch_file_block_diff_fallback(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakePatchEngine.file_blocks = (_DIFF, "salvaged")
    fake.responses = [_resp(_FILE_BLOCK)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True
    assert out["meta"]["reason"] == "ok_salvaged"
    assert out["meta"]["synth_reason"] == "salvaged"


# ---------------------------------------------------------------------------
# failure paths / finalize
# ---------------------------------------------------------------------------


def test_generate_patch_validate_failure(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp("this is not a diff at all")]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=False,
        patch_applied="",
        metadata={},
        error="boom",
    )
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is False
    assert out["meta"]["reason"] == "invalid_diff"
    assert out["error"].startswith("invalid_diff:")


def test_generate_patch_repair_success(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    # first apply-check fails -> repair -> re-check succeeds
    _FakePatchEngine.git_check = [lambda: (False, "hunk mismatch"), lambda: (True, None)]
    _FakePatchEngine.repair_result = SimpleNamespace(success=True, patch_applied=_DIFF)
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True  # repaired patch passes the re-check


def test_generate_patch_repair_failure(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    # small target -> git_apply_check_failed triggers a FILE-block retry, which
    # also fails; both rounds record their fail reasons.
    fake.responses = [_resp(_FILE_BLOCK), _resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    _FakePatchEngine.git_check = (False, "hunk mismatch")
    _FakePatchEngine.repair_result = SimpleNamespace(success=False, patch_applied=None)
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is False
    assert out["meta"]["reason"] == "git_apply_check_failed"
    assert "hunk mismatch" in out["error"]
    assert out["meta"]["retry_used"] is True


def test_generate_patch_no_target_empty_patch(svc, tmp_path):
    svc_obj, fake = svc
    fake.responses = [_resp("plain text")]
    out = svc_obj.generate_patch(str(tmp_path), "change it", None)
    assert out["success"] is False
    assert out["meta"]["reason"] == "invalid_diff"
    assert out["error"] == "invalid_diff: empty_patch"


def test_generate_patch_llm_client_error(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)

    def _boom(*a, **kw):
        raise LLMClientError("network down")

    fake.chat = _boom
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is False
    assert out["meta"]["reason"] == "llm_error"
    assert out["error"] == "network down"


def test_generate_patch_unexpected_exception(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    fake.chat = _boom
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is False
    assert out["meta"]["reason"] == "internal_error"
    assert "RuntimeError" in out["error"]


# ---------------------------------------------------------------------------
# retry path
# ---------------------------------------------------------------------------


def test_generate_patch_retry_success(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK), _resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    _FakePatchEngine.git_check = [lambda: (False, "boom"), lambda: (True, None)]
    _FakePatchEngine.repair_result = SimpleNamespace(success=False, patch_applied=None)
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True
    assert out["meta"]["retry_used"] is True
    assert len(fake.chat_calls) == 2
    # retry messages carry the FILE-BLOCK-ONLY marker
    assert "RETRY MODE: FILE-BLOCK ONLY" in fake.chat_calls[1]["messages"][1].content


def test_generate_patch_retry_tokens_sum(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK, tokens=10), _resp(_FILE_BLOCK, tokens=20)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    _FakePatchEngine.git_check = [lambda: (False, "boom"), lambda: (True, None)]
    _FakePatchEngine.repair_result = SimpleNamespace(success=False, patch_applied=None)
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["tokens_used"] == 30
    assert out["meta"]["tokens_used_first"] == 10
    assert out["meta"]["tokens_used_retry"] == 20


def test_generate_patch_retry_failure_second_round(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK), _resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    _FakePatchEngine.git_check = (False, "boom")
    _FakePatchEngine.repair_result = SimpleNamespace(success=False, patch_applied=None)
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is False
    assert out["meta"]["retry_used"] is True
    assert out["meta"]["second_fail_reason"].startswith("git_apply_check_failed:")
    assert out["meta"]["first_fail_reason"].startswith("git_apply_check_failed:")


def test_generate_patch_retry_invalid_diff_trigger(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK), _resp(_FILE_BLOCK)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=False,
        patch_applied="",
        metadata={},
        error="boom",
    )
    # legacy semantic fallback produces a non-diff patch -> validate fails ->
    # invalid_diff triggers the FILE-block retry (2nd round succeeds).
    _FakeSemantic.patch = "this is not a unified diff"
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True
    assert out["meta"]["retry_used"] is True


def test_generate_patch_retry_gate_large_file(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path, content="x" * 200_000)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    _FakePatchEngine.git_check = (False, "boom")
    _FakePatchEngine.repair_result = SimpleNamespace(success=False, patch_applied=None)
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is False
    assert out["meta"]["retry_used"] is False
    assert len(fake.chat_calls) == 1


def test_generate_patch_tokens_none_fallback(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK, tokens=None)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True
    assert out["tokens_used"] is None


# ---------------------------------------------------------------------------
# context variants / progress / modes
# ---------------------------------------------------------------------------


def test_generate_patch_full_file_mode(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    out = svc_obj.generate_patch(rr, "change it", tgt, output_mode="full_file")
    assert out["success"] is True
    assert out["meta"]["reason"] == "ok_synth"
    sys_prompt = fake.chat_calls[0]["messages"][0].content
    assert "FILE: x.txt" in sys_prompt


def test_generate_patch_full_file_mode_empty_blocks(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = ("", "file_rewrite_too_large")
    out = svc_obj.generate_patch(rr, "change it", tgt, output_mode="full_file")
    assert out["success"] is False
    assert out["meta"]["synth_reason"] == "file_rewrite_too_large"


def test_generate_patch_hybrid_mode(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "change it", tgt, context_variant="hybrid")
    assert out["success"] is True
    meta = out["meta"]["context_meta"]["llm_context_meta"]
    assert "hybrid_super_meta" in meta
    # SUPER_CONTEXT appendix present in the user payload
    user_payload = fake.chat_calls[0]["messages"][1].content
    assert "SUPER_CONTEXT" in user_payload


def test_generate_patch_hybrid_super_fallback_not_appended(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeSuperBuilder.text = ""  # super builder returns empty -> no appendix
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "change it", tgt, context_variant="hybrid")
    assert out["success"] is True
    meta = out["meta"]["context_meta"]["llm_context_meta"]
    assert "hybrid_super_meta" not in meta


def test_generate_patch_super_variant(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "change it", tgt, context_variant="super")
    assert out["success"] is True
    assert out["meta"]["context_meta"]["llm_context_meta"]["kind"] == "LLM_CONTEXT_SUPER"


def test_generate_patch_context_variant_invalid(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "change it", tgt, context_variant="bogus")
    assert out["success"] is True
    assert out["meta"]["context_meta"]["llm_context_meta"]["kind"] == "LLM_CONTEXT_V7"


def test_generate_patch_extra_context_provided(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "change it", tgt, extra_context="USER CTX HERE")
    assert out["success"] is True
    cm = out["meta"]["context_meta"]
    assert cm["source"] == "provided"
    user_payload = fake.chat_calls[0]["messages"][1].content
    assert "USER CTX HERE" in user_payload


def test_generate_patch_trivial_uses_focused_snippet(svc, tmp_path):
    (tmp_path / "x.txt").write_text("def _my_target():\n    pass\n", encoding="utf-8")
    rr, tgt = str(tmp_path), "x.txt"
    svc_obj, fake = svc
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "fix the typo in _my_target", tgt)
    assert out["success"] is True
    cm = out["meta"]["context_meta"]
    assert cm["llm_context_meta"]["snippet_kind"] == "focused"


def test_generate_patch_non_trivial_builds_context(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "Refactor the module to use async everywhere", tgt)
    assert out["success"] is True
    assert out["meta"]["context_meta"]["source"] == "built"


def test_generate_patch_progress_callback_stages(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    stages: list[tuple[str, int, int]] = []

    def cb(stage, message, cur, total):
        stages.append((stage, cur, total))

    out = svc_obj.generate_patch(rr, "add a comment", tgt, progress_callback=cb)
    assert out["success"] is True
    assert [s[0] for s in stages] == [
        "building_context",
        "sending_to_llm",
        "parsing_response",
        "finalizing",
    ]
    assert all(s[2] == 4 for s in stages)


def test_generate_patch_success_meta_shape(svc, tmp_path):
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_DIFF)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=True,
        patch_applied=_DIFF,
        metadata={"synth_reason": "auto", "mode": "auto"},
        error=None,
    )
    out = svc_obj.generate_patch(rr, "add a comment", tgt)
    assert out["success"] is True
    assert out["patch"] == _DIFF
    assert out["provider"] == "openai"
    assert out["model"] == "m"
    meta = out["meta"]
    assert meta["mode"] == "diff"
    assert meta["target_file"] == "x.txt"
    assert meta["retry_used"] is False
    assert "prompt_user_payload_sha256" in meta["context_meta"]
    assert "output_mode_used" in meta["context_meta"]
    assert meta["noop_trust_level"] is None


# ---------------------------------------------------------------------------
# exception / edge paths (100% completion)
# ---------------------------------------------------------------------------


def test_generate_patch_full_file_no_target_auto_prompt(svc, tmp_path):
    """full_file mode without a target falls back to the AUTO system prompt."""
    svc_obj, fake = svc
    fake.responses = [_resp("plain")]
    out = svc_obj.generate_patch(str(tmp_path), "change it", None, output_mode="full_file")
    assert out["success"] is False
    sys_prompt = fake.chat_calls[0]["messages"][0].content
    assert "FULL FILE REWRITE BLOCK FORMAT" in sys_prompt


def test_generate_patch_ast_rewriter_raises(svc, tmp_path, monkeypatch):
    """parse_file_blocks raising inside the AST rewrite attempt falls through
    to the file-block fallback (the except guard at _eval_ast_rewrite_attempt)."""

    def _boom(text):
        raise RuntimeError("parse boom")

    monkeypatch.setattr(svc_mod, "parse_file_blocks", _boom)
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True


def test_generate_patch_ast_import_failure(svc, tmp_path, monkeypatch):
    """ASTRewriter import/construction failure -> rewriter=None -> fallbacks."""
    monkeypatch.setattr("external_llm.ast_rewrite.ASTRewriter", None)
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True


def test_generate_patch_symbol_search_named_symbol(svc, tmp_path, monkeypatch):
    """find_symbol called with the extracted symbol name (kind mapping) when
    the new-code header is a real function header (if symbol_name branch)."""
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeRewriter.mode = "fail"
    _FakeSearcher.found = [SimpleNamespace(kind="function", name="foo", file="x.txt")]
    calls: list = []

    class _CapSearcher(_FakeSearcher):
        def find_symbol(self, name, kind=None):
            calls.append((name, kind))
            return _FakeSearcher.found

    monkeypatch.setattr("external_llm.agent.symbol_search.get_symbol_searcher", lambda rr: _CapSearcher(rr))
    # new_code's first line is a real function header -> symbol_name path
    llm_out = "FILE: x.txt\n```python\ndef foo():\n    pass\n```\n"
    fake.responses = [_resp(llm_out)]
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert calls == [("foo", "any")]
    assert out["success"] is False  # ast rewrite failed first (mode=fail)
    assert out["meta"]["reason"] == "invalid_diff"


def test_generate_patch_symbol_search_raises(svc, tmp_path):
    """searcher.find_symbol raising falls through to the next fallback."""
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path)
    _FakeSearcher.raise_find = True
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is True


def test_generate_patch_repair_raises(svc, tmp_path):
    """engine.repair_patch raising is swallowed by the pipeline guard."""
    svc_obj, fake = svc
    # large file -> no FILE-block retry, so the failure dict is returned
    rr, tgt = _setup_target(tmp_path, content="x" * 200_000)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.file_blocks = (_DIFF, "file_block_synth")
    _FakePatchEngine.git_check = (False, "hunk mismatch")
    _FakePatchEngine.repair_raise = RuntimeError("repair boom")
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is False
    assert out["meta"]["reason"] == "git_apply_check_failed"


def test_generate_patch_invalid_diff_failure_no_retry(svc, tmp_path):
    """invalid_diff error mapping in the failure dict (retry gated by size)."""
    svc_obj, fake = svc
    rr, tgt = _setup_target(tmp_path, content="x" * 200_000)
    fake.responses = [_resp(_FILE_BLOCK)]
    _FakePatchEngine.synth_result = SimpleNamespace(
        success=False,
        patch_applied="",
        metadata={},
        error="boom",
    )
    _FakeSemantic.patch = "this is not a unified diff"
    out = svc_obj.generate_patch(rr, "change it", tgt)
    assert out["success"] is False
    assert out["meta"]["reason"] == "invalid_diff"
    assert out["error"].startswith("invalid_diff:")
    assert out["meta"]["retry_used"] is False
