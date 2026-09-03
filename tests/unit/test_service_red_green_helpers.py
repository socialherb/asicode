"""RED→GREEN: external_llm/service.py helper surface (module-level helpers,
class helpers, __init__, system prompts, create_service_from_env).

Baseline: service.py 34% (832 stmts / 547 miss). This file covers the
non-generate_patch surface; test_service_red_green_generate_patch.py covers
generate_patch itself.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import external_llm.service as svc_mod
from external_llm.client import DEFAULT_LLM_TIMEOUT, OLLAMA_LLM_TIMEOUT
from external_llm.service import ExternalLLMService, _asrp_text, _bounded_read_text

# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


def test_asrp_text_clips_long_text():
    assert _asrp_text("a" * 100, 10) == "a" * 10 + " …[CLIPPED]"


def test_asrp_text_short_text_passthrough():
    assert _asrp_text("hello", 100) == "hello"


def test_asrp_text_normalizes_crlf_and_strips():
    assert _asrp_text("  hi\r\n", 100) == "hi"


def test_asrp_text_empty_or_none():
    assert _asrp_text(None, 100) == ""
    assert _asrp_text("", 100) == ""
    assert _asrp_text("  ", 100) == ""


def test_asrp_text_non_string_and_bad_max_chars():
    assert _asrp_text(12345, 100) == "12345"
    assert _asrp_text("x", 0) == ""
    assert _asrp_text("x", -5) == ""


def test_asrp_text_str_raises_type_error():
    class _BadStr:
        def __str__(self):
            raise TypeError("boom")

    assert _asrp_text(_BadStr(), 10) == ""


def test_bounded_read_text_small_file(tmp_path):
    p = tmp_path / "small.txt"
    p.write_text("hello\n", encoding="utf-8")
    text, truncated = _bounded_read_text(p)
    assert (text, truncated) == ("hello\n", False)


def test_bounded_read_text_large_file_truncates(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("x" * 1000, encoding="utf-8")
    text, truncated = _bounded_read_text(p, max_bytes=100)
    assert truncated is True
    assert len(text) == 100


def test_bounded_read_text_cuts_on_utf8_boundary(tmp_path):
    p = tmp_path / "utf8.txt"
    p.write_text("가" * 200, encoding="utf-8")  # 3 bytes per char
    text, truncated = _bounded_read_text(p, max_bytes=100)
    assert truncated is True
    # 100 bytes may land mid-character; trim must cut on a boundary.
    assert text.encode("utf-8", errors="strict").isalnum() is False or len(text.encode("utf-8")) <= 100
    assert len(text.encode("utf-8", errors="strict")) <= 100


def test_is_failure_summary_header_matches():
    assert svc_mod._is_failure_summary_header("failure_summary:")
    assert svc_mod._is_failure_summary_header("  Failure_Summary:  ")
    assert not svc_mod._is_failure_summary_header("failure_summary without colon")
    assert not svc_mod._is_failure_summary_header("the failure_summary: is not a header")


def test_is_section_boundary_variants():
    assert svc_mod._is_section_boundary("====")
    assert svc_mod._is_section_boundary("----")
    assert not svc_mod._is_section_boundary("=")
    assert not svc_mod._is_section_boundary("--")
    assert svc_mod._is_section_boundary("ASICODE_START")
    assert svc_mod._is_section_boundary("TIP: do something")
    assert svc_mod._is_section_boundary("END_CTX_PACK")
    assert not svc_mod._is_section_boundary("")
    assert not svc_mod._is_section_boundary("  plain text  ")


def test_extract_failure_summary_block():
    txt = "\n".join(
        [
            "some preamble",
            "failure_summary:",
            "line one",
            "line two",
            "====",
            "after",
        ]
    )
    assert svc_mod._extract_failure_summary_block(txt) == "line one\nline two"


def test_extract_failure_summary_block_no_header():
    assert svc_mod._extract_failure_summary_block("no header here\n") == ""


def test_extract_failure_summary_block_blank_line_stops():
    txt = "failure_summary:\nfirst\n\nsecond\n"
    assert svc_mod._extract_failure_summary_block(txt) == "first"


def test_extract_failure_summary_block_max_12_lines_and_clip():
    txt = "failure_summary:\n" + "\n".join(f"l{i}" for i in range(20)) + "\n"
    out = svc_mod._extract_failure_summary_block(txt)
    assert out.count("\n") == 11  # 12 lines max


def test_extract_failure_summary_block_handles_non_string():
    assert svc_mod._extract_failure_summary_block(None) == ""


def test_extract_failed_reason():
    txt = "\n".join(
        [
            "status: FAILED",
            "reason: the block was not found",
            "====",
        ]
    )
    assert svc_mod._extract_failed_reason(txt) == "reason: the block was not found"


def test_extract_failed_reason_three_max():
    txt = "\n".join(
        [
            "status: FAILED",
            "reason: one",
            "reason: two",
            "reason: three",
            "reason: four",
        ]
    )
    assert svc_mod._extract_failed_reason(txt) == "reason: one"


def test_extract_failed_reason_section_boundary_stops():
    txt = "status: FAILED\nreason: one\n----\nreason: two\n"
    assert svc_mod._extract_failed_reason(txt) == "reason: one"


def test_extract_failed_reason_empty():
    assert svc_mod._extract_failed_reason("status: OK\n") == ""
    assert svc_mod._extract_failed_reason(None) == ""


def test_extract_identifiers():
    assert svc_mod._extract_identifiers("foo_bar_baz and x") == ["foo_bar_baz", "and"]
    assert svc_mod._extract_identifiers("ab") == []
    assert svc_mod._extract_identifiers("12ab cd_") == ["cd_"]
    assert svc_mod._extract_identifiers("_x_y z") == ["_x_y"]


# ---------------------------------------------------------------------------
# class-level helpers
# ---------------------------------------------------------------------------


def test_suppress_console_noise(capsys):
    with ExternalLLMService._suppress_console_noise():
        print("hidden out")
        import sys

        print("hidden err", file=sys.stderr)
    out = capsys.readouterr()
    assert "hidden" not in out.out and "hidden" not in out.err


def test_is_trivial_edit_request_true_false():
    assert ExternalLLMService._is_trivial_edit_request("fix the typo in main.py")
    assert not ExternalLLMService._is_trivial_edit_request("")
    assert not ExternalLLMService._is_trivial_edit_request(
        "Refactor the module to use async everywhere and update tests"
    )


def test_extract_literal_needles_from_request():
    needles = ExternalLLMService._extract_literal_needles_from_request(
        "Add \"SOME_STRING_LITERAL\" and 'other literal' plus a bullet"
    )
    assert "SOME_STRING_LITERAL" in needles
    assert "other literal" in needles


def test_extract_literal_needles_bullet_lines():
    needles = ExternalLLMService._extract_literal_needles_from_request(
        "Add these rules:\n- Do NOT echo the prompt\n- Keep it short\n"
    )
    assert "- Do NOT echo the prompt" in needles
    assert "- Keep it short" in needles


def test_extract_literal_needles_empty_and_dedup():
    assert ExternalLLMService._extract_literal_needles_from_request("") == []
    assert ExternalLLMService._extract_literal_needles_from_request(None) == []
    n = ExternalLLMService._extract_literal_needles_from_request('"ABCDEF" "ABCDEF" - A very long bullet line here\n')
    assert n == sorted(n, key=len, reverse=True)


def test_extract_literal_needles_unterminated_quote():
    # opening quote without a closing quote -> the pair search breaks cleanly
    assert ExternalLLMService._extract_literal_needles_from_request("add '") == []


def test_extract_literal_needles_blank_needle_skipped():
    # a quoted run of whitespace passes the length gate but strips to "" and
    # must be skipped by the dedup loop
    n = ExternalLLMService._extract_literal_needles_from_request('add "      " please')
    assert n == []


def test_noop_precheck_literal_present(tmp_path):
    (tmp_path / "hello.txt").write_text("marker_xyz here\n", encoding="utf-8")
    assert ExternalLLMService._noop_precheck_for_literal_add(str(tmp_path), "hello.txt", 'add "marker_xyz" to the file')


def test_noop_precheck_literal_absent(tmp_path):
    (tmp_path / "hello.txt").write_text("other content\n", encoding="utf-8")
    assert not ExternalLLMService._noop_precheck_for_literal_add(
        str(tmp_path), "hello.txt", 'add "missing_marker" to the file'
    )


def test_noop_precheck_empty_args(tmp_path):
    assert not ExternalLLMService._noop_precheck_for_literal_add("", "x.txt", "req")
    assert not ExternalLLMService._noop_precheck_for_literal_add(str(tmp_path), "", "req")


def test_noop_precheck_missing_file(tmp_path):
    assert not ExternalLLMService._noop_precheck_for_literal_add(str(tmp_path), "nope.txt", 'add "SOME_STRING" please')


def test_noop_precheck_no_needles(tmp_path):
    (tmp_path / "hello.txt").write_text("x\n", encoding="utf-8")
    assert not ExternalLLMService._noop_precheck_for_literal_add(
        str(tmp_path), "hello.txt", "just fix things generally"
    )


def test_noop_precheck_resolve_error(tmp_path, monkeypatch):
    def _boom(rr, tf):
        raise ValueError("traversal")

    monkeypatch.setattr(svc_mod, "resolve_inside_repo", _boom)
    assert not ExternalLLMService._noop_precheck_for_literal_add(str(tmp_path), "x.txt", 'add "SOME_STRING"')


def test_noop_precheck_streaming_sliding_window(tmp_path):
    # needle absent but file spans multiple 64 KiB chunks -> the incremental
    # decoder buffer must slide (buf truncated to max_needle+4096) without
    # losing the trailing needle (here: none present -> False)
    p = tmp_path / "big.txt"
    p.write_text("x" * 70_000 + "\n", encoding="utf-8")
    assert not ExternalLLMService._noop_precheck_for_literal_add(str(tmp_path), "big.txt", 'add "NEEDLE_NOT_PRESENT"')


def test_noop_precheck_resolve_oserror(tmp_path, monkeypatch):
    def _boom(rr, tf):
        raise OSError("denied")

    monkeypatch.setattr(svc_mod, "resolve_inside_repo", _boom)
    assert not ExternalLLMService._noop_precheck_for_literal_add(str(tmp_path), "x.txt", 'add "SOME_STRING"')


def test_noop_precheck_read_oserror(tmp_path, monkeypatch):
    """File exists but reading it fails -> the streaming-read guard returns False."""

    class _ReadBoom:
        def exists(self):
            return True

        def is_file(self):
            return True

        def open(self, *a, **kw):
            raise OSError("read denied")

    monkeypatch.setattr(svc_mod, "resolve_inside_repo", lambda rr, tf: _ReadBoom())
    assert not ExternalLLMService._noop_precheck_for_literal_add(str(tmp_path), "x.txt", 'add "SOME_STRING"')


def test_read_target_file_snippet(tmp_path):
    (tmp_path / "t.py").write_text("print(1)\n", encoding="utf-8")
    svc = ExternalLLMService.__new__(ExternalLLMService)
    out = svc._read_target_file_snippet_best_effort(str(tmp_path), "t.py")
    assert out == "print(1)\n"


def test_read_target_file_snippet_empty_args(tmp_path):
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._read_target_file_snippet_best_effort("", "t.py") == ""
    assert svc._read_target_file_snippet_best_effort(str(tmp_path), "") == ""


def test_read_target_file_snippet_missing_file(tmp_path):
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._read_target_file_snippet_best_effort(str(tmp_path), "missing.py") == ""


def test_read_target_file_snippet_empty_file(tmp_path):
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._read_target_file_snippet_best_effort(str(tmp_path), "empty.py") == ""


def test_read_target_file_snippet_truncated_marker(tmp_path):
    p = tmp_path / "big.py"
    p.write_text("x" * 2000, encoding="utf-8")
    svc = ExternalLLMService.__new__(ExternalLLMService)
    out = svc._read_target_file_snippet_best_effort(str(tmp_path), "big.py", max_bytes=100)
    assert "SNIPPET TRUNCATED" in out


def test_read_target_file_snippet_resolve_error(tmp_path, monkeypatch):
    def _boom(rr, tf):
        raise OSError("denied")

    monkeypatch.setattr(svc_mod, "resolve_inside_repo", _boom)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._read_target_file_snippet_best_effort(str(tmp_path), "t.py") == ""


def test_extract_identifier_needles():
    ids = ExternalLLMService._extract_identifier_needles("In the _looks_like_unified_diff function fix the bug")
    assert "_looks_like_unified_diff" in ids
    assert len(ids) <= 6


def test_extract_identifier_needles_empty():
    assert ExternalLLMService._extract_identifier_needles("") == []
    assert ExternalLLMService._extract_identifier_needles("ab cd") == []


def test_read_focused_snippet_hit(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("\n".join(f"line {i}" for i in range(50)), encoding="utf-8")
    svc = ExternalLLMService.__new__(ExternalLLMService)
    out = svc._read_target_file_focused_snippet_best_effort(str(tmp_path), "t.py", needles=["line 25"], radius_lines=3)
    assert "line 25" in out
    assert "line 22" in out and "line 28" in out


def test_read_focused_snippet_miss_and_guards(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("plain\n", encoding="utf-8")
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._read_target_file_focused_snippet_best_effort(str(tmp_path), "t.py", needles=["zzz"]) == ""
    assert svc._read_target_file_focused_snippet_best_effort("", "t.py", needles=["x"]) == ""
    assert svc._read_target_file_focused_snippet_best_effort(str(tmp_path), "", needles=["x"]) == ""
    assert svc._read_target_file_focused_snippet_best_effort(str(tmp_path), "t.py", needles=[]) == ""
    assert svc._read_target_file_focused_snippet_best_effort(str(tmp_path), "missing.py", needles=["x"]) == ""


def test_read_focused_snippet_empty_file(tmp_path):
    (tmp_path / "t.py").write_text("", encoding="utf-8")
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._read_target_file_focused_snippet_best_effort(str(tmp_path), "t.py", needles=["x"]) == ""


def test_read_focused_snippet_resolve_oserror(tmp_path, monkeypatch):
    def _boom(rr, tf):
        raise OSError("denied")

    monkeypatch.setattr(svc_mod, "resolve_inside_repo", _boom)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._read_target_file_focused_snippet_best_effort(str(tmp_path), "t.py", needles=["x"]) == ""


def test_git_cmd_best_effort_success(monkeypatch):
    monkeypatch.setattr(
        svc_mod.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=" main\n"),
    )
    assert ExternalLLMService._git_cmd_best_effort("/tmp", ["rev-parse", "--abbrev-ref", "HEAD"]) == "main"


def test_git_cmd_best_effort_nonzero_and_exc(monkeypatch):
    monkeypatch.setattr(
        svc_mod.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout=""),
    )
    assert ExternalLLMService._git_cmd_best_effort("/tmp", ["x"]) == ""

    def _boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(svc_mod.subprocess, "run", _boom)
    assert ExternalLLMService._git_cmd_best_effort("/tmp", ["x"]) == ""


def test_git_cmd_best_effort_empty_root():
    assert ExternalLLMService._git_cmd_best_effort("", ["x"]) == ""


def test_classify_failure_hint():
    cls = ExternalLLMService._classify_failure_hint_best_effort
    assert cls("block not found: foo") == "block_not_found"
    assert cls("could not find block") == "block_not_found"
    assert cls("ambiguous match") == "ambiguous"
    assert cls("needs_disambiguation") == "ambiguous"
    assert cls("git apply failed") == "apply_failed"
    assert cls("hunk failed") == "apply_failed"
    assert cls("no such file or directory: x") == "path_or_header"
    assert cls("--- a/x.txt +++ b/x.txt") == "path_or_header"
    assert cls("") == ""
    assert cls("random noise") == ""


def test_extract_previous_failure_hint():
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._extract_previous_failure_hint_best_effort("") == ""
    assert svc._extract_previous_failure_hint_best_effort(None) == ""
    ctx = "preamble\nfailure_summary:\nblock was not found\n===="
    assert svc._extract_previous_failure_hint_best_effort(ctx) == "block was not found"
    ctx2 = "status: FAILED\nreason: patch failed to apply"
    assert "patch failed to apply" in svc._extract_previous_failure_hint_best_effort(ctx2)
    assert svc._extract_previous_failure_hint_best_effort("nothing here") == ""


# ---------------------------------------------------------------------------
# _build_llm_context_v7 / super
# ---------------------------------------------------------------------------


def _patch_git(monkeypatch, branch="main", head="abc123"):
    monkeypatch.setattr(
        svc_mod,
        "get_git_snapshot",
        lambda rr: {"branch": branch, "head_hash": head} if branch else {},
    )


def test_build_llm_context_v7_basic(tmp_path, monkeypatch):
    (tmp_path / "t.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    _patch_git(monkeypatch)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_llm_context_v7_best_effort(
        repo_root=str(tmp_path),
        target_file="t.py",
        user_request="fix",
        is_trivial=False,
    )
    assert meta["kind"] == "LLM_CONTEXT_V7"
    assert meta["branch"] == "main"
    assert meta["head_commit"] == "abc123"
    assert "TARGET_FILE: t.py" in text
    assert "REPO_ROOT:" in text


def test_build_llm_context_v7_no_git(tmp_path, monkeypatch):
    _patch_git(monkeypatch, branch="", head="")
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_llm_context_v7_best_effort(
        repo_root=str(tmp_path),
        target_file="t.py",
        user_request="fix",
        is_trivial=False,
    )
    assert "branch" not in meta and "head_commit" not in meta
    assert "BRANCH:" not in text


def test_build_llm_context_v7_full_file_mode(tmp_path, monkeypatch):
    _patch_git(monkeypatch)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, _ = svc._build_llm_context_v7_best_effort(
        repo_root=str(tmp_path),
        target_file="t.py",
        user_request="fix",
        is_trivial=False,
        output_mode="full_file",
    )
    assert "Output ONLY a single FILE block" in text
    text2, _ = svc._build_llm_context_v7_best_effort(
        repo_root=str(tmp_path),
        target_file="t.py",
        user_request="fix",
        is_trivial=False,
        output_mode="diff",
    )
    assert "Output ONLY a valid unified diff" in text2
    text3, _ = svc._build_llm_context_v7_best_effort(
        repo_root=str(tmp_path),
        target_file="t.py",
        user_request="fix",
        is_trivial=False,
        output_mode="auto",
    )
    assert "exactly ONE full-file rewrite block" in text3


def test_build_llm_context_v7_empty_root_and_no_snippet(monkeypatch):
    _patch_git(monkeypatch)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_llm_context_v7_best_effort(
        repo_root="",
        target_file="t.py",
        user_request="fix",
        is_trivial=True,
    )
    assert "TARGET_FILE: t.py" in text
    assert meta["snippet_chars"] == 0


def test_build_llm_context_v7_previous_failure_hint(tmp_path, monkeypatch):
    _patch_git(monkeypatch)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_llm_context_v7_best_effort(
        repo_root=str(tmp_path),
        target_file="t.py",
        user_request="fix",
        is_trivial=False,
        previous_failure_hint="some stale hint",
    )
    assert meta["previous_failure_hint_chars"] == len("some stale hint")
    assert "PREVIOUS_FAILURE_HINT" in text


def test_build_llm_context_v7_trivial_focused_and_fail_radius(tmp_path, monkeypatch):
    (tmp_path / "t.py").write_text("def _my_target():\n    pass\n", encoding="utf-8")
    _patch_git(monkeypatch)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    _text, meta = svc._build_llm_context_v7_best_effort(
        repo_root=str(tmp_path),
        target_file="t.py",
        user_request="in _my_target add a line",
        is_trivial=True,
    )
    assert meta["snippet_kind"] == "focused"
    _text2, meta2 = svc._build_llm_context_v7_best_effort(
        repo_root=str(tmp_path),
        target_file="t.py",
        user_request="fix the typo",
        is_trivial=True,
        previous_failure_hint="block not found",
    )
    assert meta2["snippet_failure_hint"] == "block_not_found"
    assert meta2["snippet_kind"] in ("focused", "head_tail")


def test_build_llm_context_super_ok(monkeypatch):
    class _FakeSuper:
        def __init__(self, rr):
            pass

        def build_context(self, user_request=None, target_file=None):
            return "SUPER TEXT"

    # service.py binds SuperContextBuilder at module level -> patch svc_mod
    monkeypatch.setattr(svc_mod, "SuperContextBuilder", _FakeSuper)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_llm_context_super_best_effort(repo_root="/tmp", target_file="t.py", user_request="fix")
    assert text == "SUPER TEXT\n"
    assert meta["kind"] == "LLM_CONTEXT_SUPER"


def test_build_llm_context_super_fallback(monkeypatch):
    class _BoomSuper:
        def __init__(self, rr):
            raise RuntimeError("no super")

    monkeypatch.setattr(svc_mod, "SuperContextBuilder", _BoomSuper)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    _text, meta = svc._build_llm_context_super_best_effort(repo_root="/tmp", target_file="t.py", user_request="fix")
    assert meta["kind"] == "LLM_CONTEXT_V7"
    assert meta["variant_fallback_from"] == "super"
    assert "super" in meta["super_error"]


def test_build_llm_context_super_empty_root(monkeypatch):
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_llm_context_super_best_effort(repo_root="", target_file=None, user_request="fix")
    assert text == ""
    assert meta["reason"] == "repo_root_empty"


# ---------------------------------------------------------------------------
# __init__ / _get_default_model
# ---------------------------------------------------------------------------


def test_init_ollama_timeout(monkeypatch):
    captured = {}

    def _fake_client(**kw):
        captured.update(kw)
        return SimpleNamespace()

    monkeypatch.setattr(svc_mod, "create_llm_client", _fake_client)
    svc = ExternalLLMService(provider="ollama", api_key="", timeout=DEFAULT_LLM_TIMEOUT)
    assert svc.provider == "ollama"
    assert captured["timeout"] == OLLAMA_LLM_TIMEOUT


def test_init_default_model_and_explicit(monkeypatch):
    monkeypatch.setattr(svc_mod, "create_llm_client", lambda **kw: SimpleNamespace())
    svc = ExternalLLMService(provider="openai", api_key="k")
    assert svc.model == "gpt-5.6-sol"
    svc2 = ExternalLLMService(provider="openai", api_key="k", model="custom-x")
    assert svc2.model == "custom-x"


def test_get_default_model():
    assert ExternalLLMService._get_default_model("openai") == "gpt-5.6-sol"
    assert ExternalLLMService._get_default_model("anthropic") == "claude-sonnet-5"
    assert ExternalLLMService._get_default_model("google") == "gemini-2.5-flash"
    assert ExternalLLMService._get_default_model("deepseek") == "deepseek-v4-flash"
    assert ExternalLLMService._get_default_model("zai") == "glm-5.3"
    assert ExternalLLMService._get_default_model("openrouter") == "deepseek/deepseek-v4-flash"
    assert ExternalLLMService._get_default_model("opencode") == "deepseek-v4-flash"
    assert ExternalLLMService._get_default_model("OLLAMA") == ""
    assert ExternalLLMService._get_default_model("bogus") == ""
    assert ExternalLLMService._get_default_model(None) == ""


def test_default_models_sync_with_client_classes():
    """The service forwards _PROVIDER_DEFAULT_MODELS into real API calls, so a
    stale entry here silently sends an outdated model. Gate: every entry must
    equal the provider client's own DEFAULT_MODEL.

    opencode is the one intentional exception: create_llm_client routes it to
    the generic OpenAIClient (whose DEFAULT_MODEL is gpt-5.6-sol), but
    the opencode.ai endpoint serves deepseek-v4-flash, so the service mapping
    deliberately diverges there.
    """
    from external_llm.anthropic_client import (
        AnthropicClient,
        ZAIAnthropicClient,
    )
    from external_llm.openai_client import (
        OpenAIClient,
        OpenRouterClient,
        ZAIClient,
    )
    from external_llm.providers import DeepSeekClient, GoogleClient

    expected = {
        "openai": OpenAIClient.DEFAULT_MODEL,
        "anthropic": AnthropicClient.DEFAULT_MODEL,
        "google": GoogleClient.DEFAULT_MODEL,
        "deepseek": DeepSeekClient.DEFAULT_MODEL,
        "zai": ZAIAnthropicClient.DEFAULT_MODEL,
        "openrouter": OpenRouterClient.DEFAULT_MODEL,
    }
    mapping = ExternalLLMService._PROVIDER_DEFAULT_MODELS
    for provider, client_default in expected.items():
        assert mapping[provider] == client_default, (
            f"{provider}: service default {mapping[provider]!r} drifted from client DEFAULT_MODEL {client_default!r}"
        )
    # Sanity: zai is served by two protocol clients (OpenAI-compatible ZAIClient
    # and Anthropic-compatible ZAIAnthropicClient). Both must carry the same
    # DEFAULT_MODEL — the dual definition silently drifted (5.2 vs 5.3 era)
    # until this gate pinned them together.
    assert ZAIClient.DEFAULT_MODEL == ZAIAnthropicClient.DEFAULT_MODEL == "glm-5.3"
    assert mapping["opencode"] == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# context building / parsing / normalization / prompts / retry gate
# ---------------------------------------------------------------------------


def test_build_context_best_effort_tuple(monkeypatch):
    class _FakeBuilder:
        def __init__(self, rr):
            pass

        def build_context(self, user_request=None, target_file=None):
            return ("CTX", {"extra": 1})

    monkeypatch.setattr(svc_mod, "ContextBuilder", _FakeBuilder)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_context_best_effort("/tmp", "t.py", "req")
    assert text == "CTX"
    assert meta["extra"] == 1
    assert meta["length"] == 3


def test_build_context_best_effort_str(monkeypatch):
    class _FakeBuilder:
        def __init__(self, rr):
            pass

        def build_context(self, user_request=None, target_file=None):
            return "plain string"

    monkeypatch.setattr(svc_mod, "ContextBuilder", _FakeBuilder)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_context_best_effort("/tmp", "t.py", "req")
    assert text == "plain string"
    assert meta["length"] == 12


def test_build_context_best_effort_exception_and_fallback(monkeypatch):
    class _BoomBuilder:
        def __init__(self, rr):
            raise RuntimeError("ctx fail")

    monkeypatch.setattr(svc_mod, "ContextBuilder", _BoomBuilder)
    svc = ExternalLLMService.__new__(ExternalLLMService)
    text, meta = svc._build_context_best_effort("/tmp", "t.py", "req")
    assert text == ""
    assert meta["fallback"] is True
    assert "ctx fail" in meta["build_error"]


def test_parse_llm_output_best_effort_ok():
    out = ExternalLLMService._parse_llm_output_best_effort("hello")
    assert set(out) == {"explanation", "patch"}


def test_parse_llm_output_best_effort_error(monkeypatch):
    def _boom(text):
        raise ValueError("parse fail")

    monkeypatch.setattr(svc_mod, "parse_llm_output", _boom)
    out = ExternalLLMService._parse_llm_output_best_effort("x")
    assert out == {"explanation": "", "patch": ""}


def test_parse_llm_output_best_effort_error_logs_cause(monkeypatch, caplog):
    """The silent-fallback regression: a parse failure used to return the empty
    dict with no trace anywhere, so the user saw a blank patch with no cause.
    The fallback must log the exception type + a raw-output head for diagnosis."""
    import logging

    def _boom(text):
        raise ValueError("parse fail")

    monkeypatch.setattr(svc_mod, "parse_llm_output", _boom)
    with caplog.at_level(logging.WARNING, logger="external_llm.service"):
        out = ExternalLLMService._parse_llm_output_best_effort("RAW LLM OUTPUT HEAD")
    assert out == {"explanation": "", "patch": ""}
    assert any("parse fail" in rec.message and "RAW LLM OUTPUT HEAD" in rec.message for rec in caplog.records), (
        f"parse failure not logged with cause: {[r.message for r in caplog.records]}"
    )


def test_normalize_candidate_patch_empty():
    normalized, error = ExternalLLMService._normalize_candidate_patch("", None)
    assert (normalized, error) == ("", None)


def test_build_patch_only_system_prompt():
    p = ExternalLLMService._build_patch_only_system_prompt()
    assert "unified diff" in p
    assert "NOOP" in p


def test_build_file_block_only_system_prompt():
    p = ExternalLLMService._build_file_block_only_system_prompt("src/x.py")
    assert "FILE: src/x.py" in p
    assert "single file rewrite block" in p


def test_build_auto_system_prompt():
    p = ExternalLLMService._build_auto_system_prompt()
    assert "FULL FILE REWRITE BLOCK FORMAT" in p
    assert "Prefer (A)" in p


def test_is_target_file_small_enough(tmp_path):
    small = tmp_path / "small.py"
    small.write_text("x", encoding="utf-8")
    big = tmp_path / "big.py"
    big.write_text("x" * 200_000, encoding="utf-8")
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "small.py")
    assert not svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "big.py")
    assert not svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "missing.py")
    assert not svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "")


def test_is_target_file_small_enough_errors(tmp_path, monkeypatch):
    svc = ExternalLLMService.__new__(ExternalLLMService)

    def _traversal(rr, tf):
        raise ValueError("escape")

    monkeypatch.setattr(svc_mod, "resolve_inside_repo", _traversal)
    assert not svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "x.py")

    def _oserr(rr, tf):
        raise OSError("denied")

    monkeypatch.setattr(svc_mod, "resolve_inside_repo", _oserr)
    assert not svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "x.py")


def test_is_target_file_small_enough_stat_oserror(tmp_path, monkeypatch):
    class _StatBoom:
        def exists(self):
            return True

        def is_file(self):
            return True

        def stat(self):
            raise OSError("stat fail")

    monkeypatch.setattr(svc_mod, "resolve_inside_repo", lambda rr, tf: _StatBoom())
    svc = ExternalLLMService.__new__(ExternalLLMService)
    assert not svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "x.py")


# ---------------------------------------------------------------------------
# create_service_from_env
# ---------------------------------------------------------------------------


def test_create_service_from_env_no_provider(monkeypatch):
    monkeypatch.delenv("EXTERNAL_LLM_PROVIDER", raising=False)
    assert svc_mod.create_service_from_env() is None


def test_create_service_from_env_unknown_provider(monkeypatch):
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "nonsense")
    assert svc_mod.create_service_from_env() is None


def test_create_service_from_env_missing_api_key(monkeypatch):
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert svc_mod.create_service_from_env() is None


def test_create_service_from_env_ollama_no_key(monkeypatch):
    captured = {}

    def _fake_client(**kw):
        captured.update(kw)
        return SimpleNamespace()

    monkeypatch.setattr(svc_mod, "create_llm_client", _fake_client)
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "ollama")
    svc = svc_mod.create_service_from_env()
    assert svc is not None
    assert svc.provider == "ollama"


def test_create_service_from_env_explicit_api_key(monkeypatch):
    monkeypatch.setattr(svc_mod, "create_llm_client", lambda **kw: SimpleNamespace())
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    svc = svc_mod.create_service_from_env(api_key="explicit-key")
    assert svc is not None


def test_create_service_from_env_model_env(monkeypatch):
    monkeypatch.setattr(svc_mod, "create_llm_client", lambda **kw: SimpleNamespace())
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EXTERNAL_LLM_MODEL", "env-model")
    svc = svc_mod.create_service_from_env()
    assert svc is not None
    assert svc.model == "env-model"


def test_create_service_from_env_construction_error(monkeypatch):
    def _fake_client(**kw):
        raise RuntimeError("client fail")

    monkeypatch.setattr(svc_mod, "create_llm_client", _fake_client)
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert svc_mod.create_service_from_env() is None
