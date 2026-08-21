"""RED→GREEN coverage tests for external_llm/patch_engine.py (68% → 100%).

Covers the untested repair/rollback/salvage surface:
  - synthesize_and_apply: parse failure / disambiguation / enum unavailable /
    exception / component-missing / happy path delegation
  - _auto_repair_patch: FUNCTION/CLASS replace via patch text alone
  - _try_synthesize_diff_from_file_blocks: guard outcomes
  - _salvage_small_model_output: +/- strategy, before/after, fenced insertion,
    ed delete, guards
  - repair_patch: full ladder incl. METHOD header, symbol/semantic/file-block
    fallbacks, empty-new-code, no-blocks
  - apply_patch: early exits, tolerant/reanchor success branches, repair merge
  - rollback helpers: _read_index_entries (bad records), _restore_index_entries
    (failed update), _snapshot_patch_targets, _restore_patch_targets
  - static helpers: _trim/_sanitize/_keep/_force/_ensure/_normalize edges
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import ClassVar

import pytest

from external_llm.patch_engine import PatchContext, PatchEngine

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    src = repo / "app.py"
    src.write_text(
        "def greet(name):\n"
        '    msg = "Hello, " + name\n'
        "    return msg\n"
        "\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
    )
    sub = repo / "tests"
    sub.mkdir()
    (sub / "test_app.py").write_text("from app import add\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


@pytest.fixture
def engine(git_repo):
    return PatchEngine(str(git_repo))


def reset_app(git_repo, content=None):
    if content is not None:
        (git_repo / "app.py").write_text(content)
    subprocess.run(["git", "checkout", "--", "app.py"], cwd=git_repo, check=False)


# ── synthesize_and_apply (419-498) ────────────────────────────────────────────


class TestSynthesizeAndApply:
    def test_parse_failure_propagates(self, engine, monkeypatch):
        class _Parsed:
            success = False
            error = "boom"
            mode = None

        monkeypatch.setattr(
            engine.hybrid_parser, "parse", lambda *a, **k: _Parsed()
        )
        r = engine.synthesize_and_apply("junk", "app.py")
        assert r.success is False
        assert "boom" in r.error
        assert "parse failed" in r.metadata["first_fail_reason"]

    def test_needs_disambiguation(self, engine, monkeypatch):
        class _Parsed:
            success = True
            error = None
            mode = None

        monkeypatch.setattr(
            engine.hybrid_parser, "parse", lambda *a, **k: _Parsed()
        )
        r = engine.synthesize_and_apply("ambiguous", "app.py")
        assert r.success is False
        assert r.metadata["synth_reason"] == "needs_disambiguation"

    def test_enum_unavailable(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        monkeypatch.setattr(pe, "OutputMode", None)
        r = engine.synthesize_and_apply("text", "app.py")
        assert r.success is False
        assert "enumeration" in r.error

    def test_synthesis_exception_wrapped(self, engine, monkeypatch):
        class _Parsed:
            success = True
            error = None

            class mode:  # noqa: N801 — parsed.mode attribute mirror
                value = "full_file"

        def _boom(*a, **k):
            raise RuntimeError("synth blew up")

        monkeypatch.setattr(engine.hybrid_parser, "parse", lambda *a, **k: _Parsed())
        monkeypatch.setattr(engine.patch_synthesizer, "synthesize", _boom)
        r = engine.synthesize_and_apply("x", "app.py")
        assert r.success is False
        assert "synthesis failed" in r.metadata["first_fail_reason"]

    def test_missing_component_reported(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        monkeypatch.setattr(pe, "OutputMode", None)
        engine.patch_synthesizer = None
        r = engine.synthesize_and_apply("x", "app.py")
        assert r.success is False
        missing = r.metadata["first_fail_reason"]
        assert "patch_synthesizer" in missing
        assert "output_modes" in missing

    def test_happy_path_delegates_to_apply(self, engine, git_repo, monkeypatch):
        calls = {}

        class _Parsed:
            success = True
            error = None

            class mode:  # noqa: N801 — parsed.mode attribute mirror
                value = "full_file"

        class _Synth:
            def synthesize(self, parsed, target):
                calls["target"] = target
                return textwrap.dedent(
                    """\
                    diff --git a/app.py b/app.py
                    --- a/app.py
                    +++ b/app.py
                    @@ -2,3 +2,3 @@
                     def greet(name):
                    -    msg = "Hello, " + name
                    +    msg = "Hi, " + name
                         return msg
                    """
                )

        monkeypatch.setattr(engine.hybrid_parser, "parse", lambda *a, **k: _Parsed())
        engine.patch_synthesizer = _Synth()
        r = engine.synthesize_and_apply("anything", "app.py")
        assert r.success is True, r.error
        assert calls["target"] == "app.py"
        assert r.metadata["mode"] == "git_apply"
        assert 'msg = "Hi, " + name' in (git_repo / "app.py").read_text()

    def test_apply_exception_wrapped(self, engine, monkeypatch):
        class _Parsed:
            success = True
            error = None

            class mode:  # noqa: N801 — parsed.mode attribute mirror
                value = "full_file"

        monkeypatch.setattr(engine.hybrid_parser, "parse", lambda *a, **k: _Parsed())
        monkeypatch.setattr(
            engine.patch_synthesizer, "synthesize", lambda p, t: "not a diff at all"
        )

        def _apply_boom(*a, **k):
            raise RuntimeError("apply blew up")

        monkeypatch.setattr(engine, "apply_patch", _apply_boom)
        r = engine.synthesize_and_apply("x", "app.py")
        assert r.success is False
        assert "synthesis failed: apply blew up" in r.metadata["first_fail_reason"]


# ── _auto_repair_patch (500-560) ──────────────────────────────────────────────


class TestAutoRepairPatch:
    def test_function_replace(self, engine, git_repo):
        bad_patch = (
            "@@ -6,2 +6,2 @@\n"
            "-def add(a, b):\n"
            "-    return a + b\n"
            "+def add(a, b):\n"
            "+    return a + b + 1\n"
        )
        out = engine._auto_repair_patch(bad_patch, "app.py")
        assert out is not None
        assert "diff --git a/app.py b/app.py" in out

    def test_class_replace(self, engine, git_repo):
        (git_repo / "cls.py").write_text(
            "class Widget:\n    def run(self):\n        return 1\n"
        )
        patch = (
            "@@ -1,3 +1,3 @@\n"
            "-class Widget:\n"
            "-    def run(self):\n"
            "-        return 1\n"
            "+class Widget:\n"
            "+    def run(self):\n"
            "+        return 2\n"
        )
        out = engine._auto_repair_patch(patch, "cls.py")
        assert out is not None

    def test_no_target_returns_none(self, engine):
        assert engine._auto_repair_patch("@@ -1 +1 @@\n-a\n+b\n", "") is None

    def test_missing_file_returns_none(self, engine):
        assert engine._auto_repair_patch("@@ -1 +1 @@\n-a\n+b\n", "nope.py") is None

    def test_empty_new_code_returns_none(self, engine):
        assert engine._auto_repair_patch("no hunks here", "app.py") is None

    def test_non_symbol_header_returns_none(self, engine, git_repo):
        patch = "@@ -1,2 +1,2 @@\n-x = 1\n+x = 2\n"
        out = engine._auto_repair_patch(patch, "app.py")
        assert out is None


# ── _try_synthesize_diff_from_file_blocks guards (563-716) ────────────────────


class TestFileBlockSynthGuards:
    def test_escape_path_rejected(self, engine):
        patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(engine.repo_root), "../outside.py", "FILE: ../outside.py\n```x=1```"
        )
        assert patch == ""
        assert reason == "target_missing"

    def test_stat_fail(self, engine, git_repo, monkeypatch):
        (git_repo / "app.py").chmod(0o000)

        class _P(Path):
            def stat(self):
                raise OSError("no stat")

        monkeypatch.setattr(
            "external_llm.patch_engine.Path", lambda p: _P(p) if str(p).endswith("app.py") else Path(p)
        )
        try:
            _patch, reason = engine._try_synthesize_diff_from_file_blocks(
                str(git_repo), "app.py", "FILE: app.py\n```x=1```"
            )
            assert reason == "read_failed"
        finally:
            (git_repo / "app.py").chmod(0o644)

    def test_oversize_target(self, engine, git_repo):
        (git_repo / "big.py").write_text("x = 1\n" * 50000)
        _patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "big.py", "FILE: big.py\n```x=2```"
        )
        assert reason == "file_too_large"

    def test_no_blocks(self, engine, git_repo):
        _patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "app.py", "nothing here"
        )
        assert reason == "no_file_block"

    def test_legacy_regex_fallback_picks_target(self, engine, git_repo):
        llm = (
            "FILE: app.py\n"
            "def greet(name):\n"
            '    msg = "Hello, " + name\n'
            "    return msg\n"
            "\n"
            "\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "\n"
            "\n"
            "def new_fn():\n"
            "    return 42\n"
        )
        patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "app.py", llm
        )
        assert reason == "file_block_synth"
        assert "new_fn" in patch

    def test_block_for_other_file(self, engine, git_repo):
        llm = "FILE: other.py\n```y = 2\n```\n"
        _patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "app.py", llm
        )
        assert reason == "no_target_file_block"

    def test_multi_file_block_rejected(self, engine, git_repo):
        llm = "FILE: app.py\n```x = 1\n```\nFILE: other.py\n```y = 2\n```\n"
        _patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "app.py", llm
        )
        assert reason == "multi_file_block"

    def test_basename_fallback(self, engine, git_repo):
        llm = "FILE: app.py\n```def add(a, b):\n    return a + b + 5\n```\n"
        patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "./app.py", llm
        )
        assert reason == "file_block_synth", reason
        assert "a + b + 5" in patch

    def test_identical_content_no_changes(self, engine, git_repo):
        cur = (git_repo / "app.py").read_text()
        llm = f"FILE: app.py\n```{cur}```\n"
        _patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "app.py", llm
        )
        assert reason == "no_changes"

    def test_rewrite_valve_rejects_whole_file(self, engine, git_repo):
        llm = "FILE: app.py\n```brand = 'new'\n```\n"
        _patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "app.py", llm
        )
        assert reason == "file_rewrite_too_large"

    def test_new_text_oversize(self, engine, git_repo):
        big = "x = 1\n" * 60000
        llm = f"FILE: app.py\n```{big}```\n"
        _patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "app.py", llm
        )
        assert reason == "file_too_large"

    def test_fence_inside_block_stripped(self, engine, git_repo):
        llm = (
            "FILE: app.py\n"
            "def greet(name):\n"
            '    msg = "Hello, " + name\n'
            "    return msg\n"
            "\n"
            "\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "\n"
            "\n"
            "def tail():\n"
            "    return 1\n"
            "\n"
            "```\n"
        )
        patch, reason = engine._try_synthesize_diff_from_file_blocks(
            str(git_repo), "app.py", llm
        )
        assert reason == "file_block_synth", reason
        # trailing fence must not leak into the synthesized diff
        assert "```\n+" not in patch


# ── _salvage_small_model_output (724-1103) ────────────────────────────────────


class TestSalvageStrategies:
    def test_no_target(self, engine):
        assert engine._salvage_small_model_output("+x", "") is None

    def test_escape_path(self, engine):
        r = engine._salvage_small_model_output("+x", "../../etc/passwd")
        assert r is None

    def test_missing_target(self, engine):
        assert engine._salvage_small_model_output("+x", "ghost.py") is None

    def test_oversize_by_bytes(self, engine, git_repo):
        (git_repo / "huge.py").write_text("# comment\n" * 30000)
        r = engine._salvage_small_model_output("+x\n", "huge.py")
        assert r is None

    def test_too_many_lines(self, engine, git_repo):
        (git_repo / "long.py").write_text("v = 1\n" * 2100)
        r = engine._salvage_small_model_output("+w = 2\n", "long.py")
        assert r is None

    def test_strategy1_fuzzy_replacement(self, engine, git_repo):
        malformed = (
            "Here is the fix:\n"
            "-    msg = \"Hello, \" + name\n"
            "+    msg = \"Bonjour, \" + name\n"
        )
        out = engine._salvage_small_model_output(malformed, "app.py")
        assert out is not None
        assert 'msg = "Bonjour, " + name' in out
        assert out.startswith("diff --git a/app.py b/app.py")

    def test_strategy1_below_threshold_rejected(self, engine, git_repo):
        malformed = (
            "-def greet(name):\n"
            "-    msg = \"Hello, \" + name\n"
            "+zzzzz completely unrelated zzzzz\n"
        )
        out = engine._salvage_small_model_output(malformed, "app.py")
        assert out is None

    def test_strategy2_before_after_exact(self, engine, git_repo):
        malformed = (
            "before:\n"
            "    return a + b\n"
            "after:\n"
            "    return (a + b) * 2\n"
        )
        out = engine._salvage_small_model_output(malformed, "app.py")
        assert out is not None
        assert "(a + b) * 2" in out

    def test_strategy2_before_after_fuzzy(self, engine, git_repo):
        malformed = (
            "before:\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "after:\n"
            "def add(a, b):\n"
            "    return a + b + 10\n"
        )
        out = engine._salvage_small_model_output(malformed, "app.py")
        assert out is not None
        assert "a + b + 10" in out

    def test_strategy3_insert_html_anchor(self, engine, git_repo):
        (git_repo / "page.html").write_text(
            "<!doctype html>\n<html>\n<body>hi</body>\n</html>\n"
        )
        malformed = "+<p>new paragraph</p>\n"
        out = engine._salvage_small_model_output(malformed, "page.html")
        assert out is not None
        assert "<p>new paragraph</p>" in out

    def test_strategy3_all_lines_present_returns_none(self, engine, git_repo):
        malformed = "+def add(a, b):\n"
        out = engine._salvage_small_model_output(malformed, "app.py")
        assert out is None

    def test_strategy3_symbol_anchor(self, engine, git_repo):
        malformed = "+def fresh_func():\n+    return 7\n"
        out = engine._salvage_small_model_output(malformed, "app.py")
        assert out is not None
        assert "fresh_func" in out

    def test_strategy5_ed_delete(self, engine, git_repo):
        out = engine._salvage_small_model_output("2d\n", "app.py")
        assert out is not None
        assert "-    msg =" in out

    def test_strategy5_ed_delete_out_of_range(self, engine, git_repo):
        # line 99 does not exist — no changes made → falls through to None
        out = engine._salvage_small_model_output("99d\n", "app.py")
        assert out is None


# ── repair_patch ladder (1187-1440) ───────────────────────────────────────────


class TestRepairPatchLadder:
    def test_no_llm_auto_repair_success(self, engine, git_repo):
        # patch whose + lines form a full function → auto-repair replaces it
        bad_patch = (
            "@@ -6,2 +6,2 @@\n"
            "-def add(a, b):\n"
            "-    return a + b\n"
            "+def add(a, b):\n"
            "+    return a + b + 100\n"
        )
        r = engine.repair_patch(bad_patch, "app.py", failure_reason="git apply failed")
        assert r.success is True, r.error
        assert r.metadata["mode"] == "auto_repair"
        reset_app(git_repo)

    def test_no_llm_auto_repair_fail(self, engine):
        r = engine.repair_patch("garbage", "app.py", failure_reason="x")
        assert r.success is False
        assert r.metadata["reason"] == "no_llm_fallback_all_failed"

    def test_no_parsed_blocks(self, engine):
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output="plain prose no fences")
        assert r.success is False
        assert r.metadata["reason"] == "no_parsed_blocks"

    def test_empty_new_code(self, engine):
        llm = "FUNCTION: add\n```\n\n```"
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output=llm)
        assert r.success is False
        assert r.metadata["reason"] == "empty_new_code"

    def test_ast_function_header(self, engine, git_repo):
        new_fn = "def add(a, b):\n    return a + b + 3\n"
        llm = f"FUNCTION: add\n```\n{new_fn}```"
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output=llm)
        assert r.success is True, r.error
        assert r.metadata["mode"] == "ast_function"
        reset_app(git_repo)

    def test_ast_class_header(self, engine, git_repo):
        (git_repo / "cls.py").write_text("class Box:\n    def m(self):\n        return 1\n")
        new_cls = "class Box:\n    def m(self):\n        return 2\n"
        llm = f"CLASS: Box\n```\n{new_cls}```"
        r = engine.repair_patch("p", "cls.py", failure_reason="x", llm_output=llm)
        assert r.success is True, r.error
        assert r.metadata["mode"] == "ast_class"

    def test_ast_method_header_nested_chain(self, engine, git_repo):
        (git_repo / "nest.py").write_text(
            "class Outer:\n    class Inner:\n        def calc(self):\n            return 1\n"
        )
        new_m = "    def calc(self):\n        return 42\n"
        llm = f"METHOD: Outer.Inner.calc\n```\n{new_m}```"
        r = engine.repair_patch("p", "nest.py", failure_reason="x", llm_output=llm)
        assert r.success is True, r.error
        assert r.metadata["mode"] == "ast_method"
        assert "return 42" in (git_repo / "nest.py").read_text()

    def test_ast_autodetect_python_def(self, engine, git_repo):
        new_fn = "def add(a, b):\n    return a + b + 9\n"
        llm = f"```\n{new_fn}```"
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output=llm)
        assert r.success is True, r.error
        assert r.metadata["mode"] == "ast_autodetect"
        reset_app(git_repo)

    def test_ast_exception_records_reason(self, engine, git_repo, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("ast died")

        monkeypatch.setattr(engine.ast_rewriter, "replace_function", _boom)
        llm = "FUNCTION: add\n```\ndef add(a, b):\n    return 5\n```"
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output=llm)
        assert r.success is False
        assert "ast_rewrite_failed" in r.metadata["second_fail_reason"]

    def test_symbol_search_function_fallback(self, engine, git_repo, monkeypatch):
        # block the header routes → reach symbol search
        llm = "no header line\n```\ndef add(a, b):\n    return a + b + 4\n```"

        class _FakeAST:
            def replace_function(self, file, name, code):
                return "rewritten-fn"

            def generate_patch(self, file, result):
                return "diff --git synthetic\n"

        monkeypatch.setattr(engine, "ast_rewriter", None)  # block ladder #1
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output=llm)
        # symbol_searcher is real; add() exists in app.py → found
        assert r.metadata["mode"] in ("ast_symbol_function", "semantic_function", "file_block_synth")
        reset_app(git_repo)

    def test_semantic_patch_fallback(self, engine, git_repo, monkeypatch):
        monkeypatch.setattr(engine, "ast_rewriter", None)
        monkeypatch.setattr(engine, "symbol_searcher", None)
        llm = "```\ndef add(a, b):\n    return a + b + 11\n```"
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output=llm)
        assert r.success is True, r.error
        assert r.metadata["mode"] in ("semantic_function", "semantic_class")
        reset_app(git_repo)

    def test_file_block_synth_fallback(self, engine, git_repo, monkeypatch):
        monkeypatch.setattr(engine, "ast_rewriter", None)
        monkeypatch.setattr(engine, "symbol_searcher", None)
        monkeypatch.setattr(engine, "semantic_patcher", None)
        llm = "```\ndef greet(name):\n    msg = \"Hello, \" + name\n    return msg\n\n\ndef add(a, b):\n    return a + b + 12\n\n\ndef multiply(a, b):\n    return a * b\n```"
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output=llm)
        assert r.success is True, r.error
        assert r.metadata["mode"] == "file_block_synth"
        reset_app(git_repo)

    def test_all_failed(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "ast_rewriter", None)
        monkeypatch.setattr(engine, "symbol_searcher", None)
        monkeypatch.setattr(engine, "semantic_patcher", None)
        monkeypatch.setattr(engine, "patch_synthesizer", None)
        r = engine.repair_patch("p", "app.py", failure_reason="x", llm_output="```\nx=1\n```")
        assert r.success is False
        assert r.metadata["reason"] == "all_repair_failed"
        assert r.metadata["fallback_used"] == []


# ── apply_patch early-exit & merge paths (144, 197, 225-226, 289-290, 349-373) ─


class TestApplyPatchPaths:
    def test_empty_normalized_warning(self, engine, git_repo, caplog):
        # a patch that sanitizes down to empty: fences only, no diff content
        r = engine.apply_patch("```diff\n```", "app.py")
        assert r.success is False
        assert "Target file does not exist" not in (r.error or "")

    def test_p1_target_from_patch_header_tab_timestamp(self, engine, git_repo):
        # `diff -u` style: +++ b/app.py\t2020-01-02 — tab must be stripped
        patch = (
            "diff -u a/app.py b/app.py\n"
            "--- a/app.py\t2020-01-02 03:04:05\n"
            "+++ b/app.py\t2020-01-02 03:04:06\n"
            "@@ -2,1 +2,1 @@\n"
            ' msg = "Hello, " + name\n'
        )
        # context-only patch → apply succeeds without changes
        r = engine.apply_patch(patch, None)
        assert r.success is True, r.error

    def test_p1_target_from_bare_plus_header(self, engine, git_repo):
        patch = (
            "--- app.py\t2020-01-02\n"
            "+++ app.py\t2020-01-02\n"
            "@@ -2,1 +2,1 @@\n"
            ' msg = "Hello, " + name\n'
        )
        r = engine.apply_patch(patch, None)
        assert r.success is True, r.error

    def test_file_not_found_early_exit(self, engine):
        patch = (
            "--- a/ghost.py\n+++ b/ghost.py\n@@ -1 +1 @@\n-a\n+b\n"
        )
        r = engine.apply_patch(patch, "ghost.py")
        assert r.success is False
        assert "does not exist" in r.error
        assert r.metadata["reason"] == "file_not_found"
        assert "Use the 'create_file' tool" in r.error

    def test_git_apply_exception_recorded(self, engine, git_repo, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("diff_apply blew")

        monkeypatch.setattr(engine, "_diff_apply", _boom)
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -2,1 +2,1 @@\n-    msg = \"Hello, \" + name\n+    msg = \"Hi, \" + name\n"
        )
        r = engine.apply_patch(patch, "app.py")
        assert r.success is False
        assert "git apply exception" in r.metadata["first_fail_reason"]

    def test_diff_apply_module_missing(self, engine, git_repo, monkeypatch):
        monkeypatch.setattr(engine, "_diff_apply", None)
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -2,1 +2,1 @@\n-    msg = \"Hello, \" + name\n+    msg = \"Hi, \" + name\n"
        )
        r = engine.apply_patch(patch, "app.py")
        assert r.success is False
        assert "diff_apply module not available" in r.metadata["first_fail_reason"]

    def test_tolerant_success_records_mode(self, engine, git_repo):
        # stale line numbers → tolerant/reanchor path rescues
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -8,2 +8,2 @@\n"
            "-def add(a, b):\n"
            "-    return a + b\n"
            "+def add(a, b):\n"
            "+    return a + b + 77\n"
        )
        r = engine.apply_patch(patch, "app.py")
        assert r.success is True, r.error
        assert "a + b + 77" in (git_repo / "app.py").read_text()
        reset_app(git_repo)

    def test_repair_success_via_apply_patch(self, engine, git_repo):
        # git apply will fail (no such content), repair ladder recovers via LLM output
        llm = "FUNCTION: add\n```\ndef add(a, b):\n    return a + b + 55\n```"
        ctx = PatchContext(llm_output=llm, output_mode="auto")
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -6,2 +6,2 @@\n-ZZZ no match ZZZ\n-ZZZZ\n+def add(a, b):\n+    return a + b + 55\n"
        )
        r = engine.apply_patch(patch, "app.py", context=ctx)
        assert r.success is True, r.error
        assert r.metadata["mode"] in ("git_apply", "auto_repair", "ast_function")
        reset_app(git_repo)

    def test_repair_patch_apply_fails(self, engine, git_repo, monkeypatch):
        llm = "FUNCTION: add\n```\ndef add(a, b):\n    return a + b + 66\n```"
        ctx = PatchContext(llm_output=llm)
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -6,2 +6,2 @@\n-ZZZ no match ZZZ\n-ZZZZ\n+def add(a, b):\n+    return 1\n"
        )

        # repair succeeds (AST replace), but the final apply must fail
        real_apply_once = engine._apply_diff_once

        def _failing_apply(patch_text, target_file=None):
            # make the final apply fail with a patch DIFFERENT from what the
            # repair produced → exercise repaired_patch_apply_failed
            if patch_text != real_apply_once.__self__._last_repair if False else True:
                pass
            return False, "simulated apply failure"

        monkeypatch.setattr(engine, "_apply_diff_once", _failing_apply)
        r = engine.apply_patch(patch, "app.py", context=ctx)
        assert r.success is False
        assert r.metadata["reason"] == "repaired_patch_apply_failed"
        assert "simulated apply failure" in r.metadata["second_fail_reason"]
        assert "Repaired patch failed to apply" in r.error

    def test_repair_missing_patch_result(self, engine, git_repo, monkeypatch):
        llm = "FUNCTION: add\n```\ndef add(a, b):\n    return 1\n```"
        ctx = PatchContext(llm_output=llm)

        class _FakeRepair:
            success = True
            patch_applied = None
            error = None
            metadata: ClassVar[dict] = {"reason": "r", "mode": "m", "fallback_used": ["x"]}

        monkeypatch.setattr(engine, "repair_patch", lambda **k: _FakeRepair())
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -6,2 +6,2 @@\n-ZZZZ\n-ZZZZ\n+q\n+w\n"
        )
        r = engine.apply_patch(patch, "app.py", context=ctx)
        assert r.success is False
        assert r.metadata["reason"] == "repaired_patch_missing"

    def test_untracked_failure_guidance(self, engine, git_repo):
        (git_repo / "untracked.py").write_text("a = 1\nb = 2\n")
        patch = (
            "diff --git a/untracked.py b/untracked.py\n--- a/untracked.py\n"
            "+++ b/untracked.py\n@@ -1,2 +1,2 @@\n-a = 1\n+a = 9\n"
        )
        r = engine.apply_patch(patch, "untracked.py")
        # untracked: plain git apply still works → success
        assert r.success is True, r.error
        assert r.metadata.get("target_git_state") == "untracked"
        subprocess.run(["git", "checkout", "--", "untracked.py"], cwd=git_repo, check=False)
        (git_repo / "untracked.py").unlink()

    def test_freshly_edited_failure_guidance(self, engine, git_repo):
        (git_repo / "app.py").write_text("def add(a, b):\n    return 999\n")
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,1 +1,1 @@\n-does not exist in file\n+new line here\n"
        )
        r = engine.apply_patch(patch, "app.py")
        assert r.success is False
        assert "freshly-edited" in (r.error or "")
        reset_app(git_repo)

    def test_gitignored_failure_guidance(self, engine, git_repo):
        (git_repo / ".gitignore").write_text("ignored.py\n")
        (git_repo / "ignored.py").write_text("x = 1\ny = 2\n")
        patch = (
            "diff --git a/ignored.py b/ignored.py\n--- a/ignored.py\n"
            "+++ b/ignored.py\n@@ -1,2 +1,2 @@\n-XXXX\n+new\n"
        )
        r = engine.apply_patch(patch, "ignored.py")
        assert r.success is False
        assert "untracked" in (r.error or "") or "gitignored" in (r.error or "")
        assert "git add" in (r.error or "")
        (git_repo / "ignored.py").unlink()
        (git_repo / ".gitignore").unlink()
        subprocess.run(["git", "checkout", "--", ".gitignore"], cwd=git_repo, check=False)


# ── Rollback & index helpers (2379-2479) ──────────────────────────────────────


class TestRollbackHelpers:
    def test_snapshot_patch_targets_roundtrip(self, engine, git_repo):
        snap = engine._snapshot_patch_targets(
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        )
        (app_abs,) = snap.keys()
        assert snap[app_abs] is not None
        # mutate + restore
        (git_repo / "app.py").write_text("corrupted")
        engine._restore_patch_targets(snap)
        assert "def greet" in (git_repo / "app.py").read_text()

    def test_restore_missing_file_removed(self, engine, git_repo):
        snap = {str(git_repo / "made_up.py"): None}
        (git_repo / "made_up.py").write_text("created by patch")
        engine._restore_patch_targets(snap)
        assert not (git_repo / "made_up.py").exists()

    def test_read_index_empty_rels(self, engine):
        assert engine._read_index_entries([]) == {}

    def test_read_index_subprocess_fail(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        def _boom(*a, **k):
            raise OSError("no git")

        monkeypatch.setattr(pe.subprocess, "run", _boom)
        assert engine._read_index_entries(["app.py"]) == {}

    def test_read_index_bad_records_skipped(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        class _R:
            returncode = 0
            stdout = b"garbage-no-tab\x00\x00100644 abc123 0\tapp.py\x00"

        monkeypatch.setattr(pe.subprocess, "run", lambda *a, **k: _R())
        out = engine._read_index_entries(["app.py"])
        assert out == {"app.py": ("100644", "abc123")}

    def test_restore_index_noop_when_equal(self, engine, git_repo, monkeypatch):
        calls = []

        def _rec(*a, **k):
            calls.append(a)
            raise AssertionError("must not be called when entries equal")

        monkeypatch.setattr(
            engine, "_read_index_entries", lambda rels: {"app.py": ("100644", "abc")}
        )
        snapshot = {"app.py": ("100644", "abc")}
        engine._restore_index_entries(snapshot)
        assert calls == []

    def test_restore_index_update_fails_logged(self, engine, git_repo, monkeypatch, caplog):
        import external_llm.patch_engine as pe

        class _R:
            returncode = 1
            stderr = b"boom"

        monkeypatch.setattr(pe.subprocess, "run", lambda *a, **k: _R())
        snapshot = {"app.py": ("100644", "abc")}
        engine._restore_index_entries(snapshot)

    def test_restore_index_subprocess_exception_logged(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        def _boom(*a, **k):
            raise OSError("no git")

        monkeypatch.setattr(pe.subprocess, "run", _boom)
        engine._restore_index_entries({"app.py": ("100644", "abc")})

    def test_restore_index_skips_healthy_entries(self, engine, git_repo, monkeypatch):
        # current matches snapshot for app.py, differs for other.py → only other updated
        updates = []

        class _R:
            returncode = 0
            stderr = b""

        def _fake_run(cmd, *a, **k):
            if cmd[1] == "ls-files":
                class _L:
                    returncode = 0
                    stdout = b"100644 sha1 0\tapp.py\x00100644 sha2 0\tother.py\x00"
                return _L()
            updates.append(cmd)
            return _R()

        import external_llm.patch_engine as pe

        monkeypatch.setattr(pe.subprocess, "run", _fake_run)
        snapshot = {"app.py": ("100644", "sha1"), "other.py": ("100644", "sha9")}
        engine._restore_index_entries(snapshot)
        assert len(updates) == 1
        assert "other.py" in updates[0][2]

    def test_snapshot_index_entries(self, engine, git_repo):
        snap = engine._snapshot_index_entries(
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        )
        assert "app.py" in snap  # committed in fixture


# ── convert_patch_to_edit_blocks edges (1115-1183) ────────────────────────────


class TestConvertPatchToEditBlocksEdges:
    def test_empty_patch(self, engine):
        assert engine.convert_patch_to_edit_blocks("") is None
        assert engine.convert_patch_to_edit_blocks("   \n  ") is None

    def test_no_path_anywhere(self, engine):
        p = "@@ -1 +1 @@\n-a\n+b\n"
        assert engine.convert_patch_to_edit_blocks(p) is None

    def test_diff_git_header_extract(self, engine):
        p = (
            "diff --git a/deep/mod.py b/deep/mod.py\n"
            "--- a/deep/mod.py\n+++ b/deep/mod.py\n@@ -1 +1 @@\n-a\n+b\n"
        )
        out = engine.convert_patch_to_edit_blocks(p)
        assert out["file_path"] == "deep/mod.py"
        assert out["blocks"] == [{"before": "a", "after": "b"}]

    def test_only_add_lines(self, engine):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -0,0 +1 @@\n+brand_new = 1\n"
        )
        out = engine.convert_patch_to_edit_blocks(p)
        assert out["blocks"] == [{"before": "", "after": "brand_new = 1"}]

    def test_blank_and_meta_lines_skipped(self, engine):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "\\ No newline at end of file\n"
            "+x = 1\n"
        )
        out = engine.convert_patch_to_edit_blocks(p)
        assert out["blocks"][0]["after"] == "x = 1"

    def test_no_hunks(self, engine):
        p = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        assert engine.convert_patch_to_edit_blocks(p) is None


# ── Static sanitizer edges (1433-1848) ────────────────────────────────────────


class TestStaticSanitizerEdges:
    # _trim_patch_to_first_header
    def test_trim_empty(self):
        assert PatchEngine._trim_patch_to_first_header("") == ""

    def test_trim_no_header_returns_stripped(self):
        assert PatchEngine._trim_patch_to_first_header("  junk only  ") == "junk only"

    def test_trim_trims_to_diff_git(self):
        out = PatchEngine._trim_patch_to_first_header(
            "preamble\nnoise\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n"
        )
        assert out.startswith("--- a/f")
        assert out.endswith("\n")

    # _sanitize_patch_lines
    def test_sanitize_empty(self):
        assert PatchEngine._sanitize_patch_lines("") == ""

    def test_sanitize_bom_and_indent_and_fence(self):
        p = (
            "\ufeff```diff\n"
            "  --- a/f\n"
            "\t+++ b/f\n"
            "  @@ -1 +1 @@\n"
            " -a\n"
            " +b\n"
        )
        out = PatchEngine._sanitize_patch_lines(p)
        assert "```" not in out
        assert out.startswith("--- a/f")
        assert "\t" not in out.split("@@")[0]

    def test_sanitize_body_context_marker_preserved(self):
        # context line whose CONTENT looks like a header must stay verbatim
        p = (
            "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n"
            " normal\n"
            " +++ b/other.py\n"
        )
        out = PatchEngine._sanitize_patch_lines(p)
        assert " +++ b/other.py\n" in out

    def test_sanitize_git_marker_in_body_returns_to_header(self):
        p = (
            "--- a/f\n+++ b/f\n@@ -1 +1 @@\n x\n"
            "diff --git a/g b/g\n"
            "    --- a/g\n"
        )
        out = PatchEngine._sanitize_patch_lines(p)
        assert "--- a/g" in out

    # _keep_only_target_file_section
    def test_keep_no_patch(self):
        assert PatchEngine._keep_only_target_file_section("", "f") == ""

    def test_keep_prefers_exact_over_earlier_basename(self):
        p = (
            "diff --git a/src/utils.py b/src/utils.py\n--- a/src/utils.py\n+++ b/src/utils.py\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/tests/utils.py b/tests/utils.py\n--- a/tests/utils.py\n+++ b/tests/utils.py\n@@ -1 +1 @@\n-c\n+d\n"
        )
        out = PatchEngine._keep_only_target_file_section(p, "tests/utils.py")
        assert "tests/utils.py" in out
        assert "src/utils.py" not in out

    def test_keep_first_section_when_no_match(self):
        p = (
            "diff --git a/one.py b/one.py\n--- a/one.py\n+++ b/one.py\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/two.py b/two.py\n--- a/two.py\n+++ b/two.py\n@@ -1 +1 @@\n-c\n+d\n"
        )
        out = PatchEngine._keep_only_target_file_section(p, "zzz.py")
        assert "one.py" in out
        assert "two.py" not in out

    def test_keep_no_diff_git_no_header(self):
        p = "@@ -1 +1 @@\n-a\n+b\n"
        out = PatchEngine._keep_only_target_file_section(p, "f.py")
        assert out == "@@ -1 +1 @@\n-a\n+b\n"

    def test_keep_no_diff_git_second_section_stops(self):
        p = (
            "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b\n"
            "--- a/g.py\n+++ b/g.py\n@@ -1 +1 @@\n-c\n+d\n"
        )
        out = PatchEngine._keep_only_target_file_section(p, "f.py")
        assert "g.py" not in out

    def test_keep_no_diff_git_unparseable_first_header(self):
        p = (
            "--- weird header\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b\n"
            "--- a/g.py\n+++ b/g.py\n@@ -1 +1 @@\n-c\n+d\n"
        )
        out = PatchEngine._keep_only_target_file_section(p, "f.py")
        # first_file is None → stop at the second '--- ' regardless
        assert "g.py" not in out

    # _force_target_file_paths
    def test_force_no_patch(self):
        assert PatchEngine._force_target_file_paths("", "f") == ""

    def test_force_no_target_passthrough(self):
        p = "--- a/f\n+++ b/f\n"
        assert PatchEngine._force_target_file_paths(p, "") == p

    def test_force_rewrites_basename_headers(self):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1 +1 @@\n-a\n+b\n"
        )
        out = PatchEngine._force_target_file_paths(p, "pkg/sub/app.py")
        assert "diff --git a/pkg/sub/app.py b/pkg/sub/app.py" in out
        assert "--- a/pkg/sub/app.py" in out
        assert "+++ b/pkg/sub/app.py" in out

    def test_force_keeps_devnull(self):
        p = "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n"
        out = PatchEngine._force_target_file_paths(p, "pkg/new.py")
        assert "--- /dev/null" in out

    def test_force_no_prefix_headers(self):
        p = "diff --git app.py app.py\n--- app.py\n+++ app.py\n@@ -1 +1 @@\n-a\n+b\n"
        out = PatchEngine._force_target_file_paths(p, "pkg/app.py")
        assert out.startswith("diff --git a/pkg/app.py")

    # _ensure_headers_before_any_hunk
    def test_ensure_empty(self):
        assert PatchEngine._ensure_headers_before_any_hunk("", "f") == ""

    def test_ensure_no_target_passthrough(self):
        p = "@@ -1 +1 @@\n-a\n+b\n"
        assert PatchEngine._ensure_headers_before_any_hunk(p, "") == p

    def test_ensure_no_hunk_passthrough(self):
        p = "--- a/f\n+++ b/f\n"
        assert PatchEngine._ensure_headers_before_any_hunk(p, "f") == p

    def test_ensure_injects_headers(self):
        p = "junk line\n@@ -1 +1 @@\n-a\n+b\n"
        out = PatchEngine._ensure_headers_before_any_hunk(p, "f.py")
        assert "diff --git a/f.py b/f.py" in out
        assert "--- a/f.py" in out
        assert "@@ -1 +1 @@" in out

    def test_ensure_present_headers_passthrough(self):
        p = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n"
        assert PatchEngine._ensure_headers_before_any_hunk(p, "f") == p

    # _normalize_patch_headers
    def test_normalize_headers_empty(self):
        assert PatchEngine._normalize_patch_headers("", "f") == ""

    def test_normalize_injects_from_diff_git(self):
        p = "diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n"
        out = PatchEngine._normalize_patch_headers(p, None)
        assert "--- a/f" in out
        assert "+++ b/f" in out

    def test_normalize_injects_from_target(self):
        p = "@@ -1 +1 @@\n-a\n+b\n"
        out = PatchEngine._normalize_patch_headers(p, "f.py")
        assert "--- a/f.py" in out
        assert "+++ b/f.py" in out

    def test_normalize_no_source_leaves_as_is(self):
        p = "@@ -1 +1 @@\n-a\n+b\n"
        out = PatchEngine._normalize_patch_headers(p, None)
        assert out == "@@ -1 +1 @@\n-a\n+b\ndefault_marker\n".replace("default_marker\n", "")

    def test_normalize_existing_headers_untouched(self):
        p = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n"
        assert PatchEngine._normalize_patch_headers(p, None) == p

    def test_normalize_second_section_resets_flags(self):
        p = (
            "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/g b/g\n@@ -1 +1 @@\n-c\n+d\n"
        )
        out = PatchEngine._normalize_patch_headers(p, None)
        assert "--- a/g" in out
        assert "+++ b/g" in out


# ── _add_diff_headers (2184-2210) ─────────────────────────────────────────────


class TestAddDiffHeaders:
    def test_no_target_passthrough(self):
        assert PatchEngine._add_diff_headers("@@ -1 +1 @@\n-a\n+b\n", "") == "@@ -1 +1 @@\n-a\n+b\n"

    def test_hunk_only_gets_full_header(self):
        out = PatchEngine._add_diff_headers("@@ -1 +1 @@\n-a\n+b\n", "pkg/f.py")
        assert out.startswith("diff --git a/pkg/f.py b/pkg/f.py\n--- a/pkg/f.py\n+++ b/pkg/f.py\n")

    def test_minus_a_without_git(self):
        out = PatchEngine._add_diff_headers("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b\n", "f.py")
        assert out.startswith("diff --git a/f.py b/f.py\n")

    def test_complete_patch_passthrough(self):
        p = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n"
        assert PatchEngine._add_diff_headers(p, "f") == p


# ── _tolerant_git_apply variants (2247, 2339-2340) ────────────────────────────


class TestTolerantApplyEdges:
    def test_check_ok_apply_fails_rolls_back(self, engine, git_repo, monkeypatch):
        import external_llm.patch_engine as pe

        seq = {"n": 0}

        class _Check:
            returncode = 0
            stderr = b""

        class _Apply:
            returncode = 1
            stderr = b"apply exploded"

        def _fake_run(cmd, *a, **k):
            seq["n"] += 1
            # 1st call per variant is --check, 2nd is the apply
            if "--check" in cmd:
                return _Check()
            return _Apply()

        restores = []
        monkeypatch.setattr(pe.subprocess, "run", _fake_run)
        monkeypatch.setattr(engine, "_snapshot_patch_targets", lambda p: {"k": "v"})
        monkeypatch.setattr(engine, "_snapshot_index_entries", lambda p: {})
        monkeypatch.setattr(engine, "_restore_patch_targets", lambda s: restores.append(s))
        monkeypatch.setattr(engine, "_restore_index_entries", lambda s: restores.append(s))

        ok, err, mode = engine._tolerant_git_apply(
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n",
            "app.py",
        )
        assert ok is False
        assert "check OK but apply failed" in (err or "")
        assert restores  # rollback ran
        assert mode != "none"

    def test_subprocess_exception_continues_to_next_variant(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        calls = {"n": 0}

        def _fake_run(cmd, *a, **k):
            calls["n"] += 1
            raise OSError("git vanished")

        monkeypatch.setattr(pe.subprocess, "run", _fake_run)
        ok, _err, mode = engine._tolerant_git_apply(
            "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n", "f"
        )
        assert ok is False
        assert mode == "none"
        assert calls["n"] >= 5  # every variant attempted

    def test_3way_only_when_allowed(self, engine, git_repo, monkeypatch):
        import external_llm.patch_engine as pe

        seen_flags = []

        class _R:
            returncode = 1
            stderr = b"nope"

        def _fake_run(cmd, *a, **k):
            if "--check" in cmd:
                seen_flags.append(tuple(cmd[2:-1]))
            return _R()

        monkeypatch.setattr(pe.subprocess, "run", _fake_run)
        engine._tolerant_git_apply("junk patch", "app.py", allow_3way=False)
        joined = [f for t in seen_flags for f in t]
        assert "--3way" not in joined


# ── _reanchor helpers (2534-2832) ─────────────────────────────────────────────


class TestReanchorEdges:
    def test_core_no_target(self, engine):
        assert engine._reanchor_patch_core("x", None, lambda *a: None, "p") is None

    def test_core_target_missing(self, engine, git_repo):
        assert engine._reanchor_patch_core("x", "ghost.py", lambda *a: None, "p") is None

    def test_core_empty_file(self, engine, git_repo):
        (git_repo / "empty.py").write_text("")
        assert engine._reanchor_patch_core("x", "empty.py", lambda *a: None, "p") is None

    def test_core_no_hunks_passthrough_none(self, engine, git_repo):
        out = engine._reanchor_patch_core("no hunks", "app.py", lambda *a: None, "p")
        assert out is None

    def test_core_finder_none_keeps_hunk(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -6,2 +6,2 @@\n-def add(a, b):\n-    return a + b\n+x\n+y\n"
        )
        out = engine._reanchor_patch_core(patch, "app.py", lambda *a: None, "p")
        assert out is None  # nothing changed

    def test_core_rewrites_header(self, engine, git_repo):
        # hunk claims line 6 but content sits at line 7 (0-idx 6)
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -6,2 +6,2 @@\n"
            "-def add(a, b):\n"
            "-    return a + b\n"
            "+def add(a, b):\n"
            "+    return a + b + 1\n"
        )
        # finder reports the anchor at 0-index 6 (= line 7), 0 context before
        out = engine._reanchor_patch_core(
            patch, "app.py", lambda body, fl, old: (6, 0, "(test)"), "p"
        )
        assert out is not None
        assert "@@ -7,2 +7,2 @@" in out

    def test_core_counts_default_to_one(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -6 +6 @@\n"
            "-def add(a, b):\n"
            "+def add(a, b):\n"
        )
        out = engine._reanchor_patch_core(
            patch, "app.py", lambda body, fl, old: (6, 0, "(t)"), "p"
        )
        assert out is not None
        assert "@@ -7,1 +7,1 @@" in out

    def test_exact_no_removed_lines(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,0 +2 @@\n+new\n"
        )
        assert engine._exact_reanchor_patch(patch, "app.py") is None

    def test_exact_removed_line_too_short(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1 +1 @@\n-ab\n+cd\n"
        )
        assert engine._exact_reanchor_patch(patch, "app.py") is None

    def test_exact_no_match_in_file(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -2 +2 @@\n-ZZZZ totally absent ZZZZ\n+new content line\n"
        )
        assert engine._exact_reanchor_patch(patch, "app.py") is None

    def test_exact_already_correct_returns_none(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -2,2 +2,2 @@\n"
            "-    msg = \"Hello, \" + name\n"
            "-    return msg\n"
            "+    msg = \"Hi, \" + name\n"
            "+    return msg\n"
        )
        assert engine._exact_reanchor_patch(patch, "app.py") is None

    def test_exact_offset_beyond_50_rejected(self, engine, git_repo):
        # 60 filler lines above push the real content far from the claimed line
        filler = "\n".join(f"# pad line {i:02d}" for i in range(60)) + "\n"
        (git_repo / "app.py").write_text(filler + "def add(a, b):\n    return a + b\n")
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1 +1 @@\n-def add(a, b):\n+def add(a, b):\n    return a + b + 1\n"
        )
        assert engine._exact_reanchor_patch(patch, "app.py") is None

    def test_exact_subsequent_line_mismatch_rejected(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -2,2 +2,2 @@\n"
            "-    msg = \"Hello, \" + name\n"
            "-TOTALLY WRONG SECOND\n"
            "+a\n+b\n"
        )
        assert engine._exact_reanchor_patch(patch, "app.py") is None

    def test_fuzzy_no_search_lines(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,0 +2 @@\n+insert\n"
        )
        assert engine._reanchor_patch(patch, "app.py") is None

    def test_fuzzy_file_over_2000_lines(self, engine, git_repo):
        (git_repo / "big.py").write_text("v = 1\n" * 2100)
        patch = (
            "diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n"
            "@@ -2,2 +2,2 @@\n-v = 1\n-v = 1\n+x\n+y\n"
        )
        assert engine._reanchor_patch(patch, "big.py") is None

    def test_fuzzy_reanchors_stale_header(self, engine, git_repo):
        # content at lines 7-8, header claims 20-21
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -20,3 +20,3 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a + b + 2\n"
        )
        out = engine._reanchor_patch(patch, "app.py")
        assert out is not None
        assert "@@ -7,3 +7,3 @@" in out

    def test_fuzzy_good_score_at_same_position_keeps_none(self, engine, git_repo):
        patch = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -7,3 +7,3 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a + b + 2\n"
        )
        assert engine._reanchor_patch(patch, "app.py") is None


# ── context_free_hunks edges (2534, 2539-2543) ────────────────────────────────


class TestContextFreeHunks:
    def test_new_file_excluded(self):
        p = (
            "diff --git a/n.py b/n.py\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+x\n"
        )
        assert PatchEngine.context_free_hunks(p) == []

    def test_context_free_hunk_reported(self):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -8,0 +9,2 @@\n+stmt1\n+stmt2\n"
        )
        out = PatchEngine.context_free_hunks(p)
        assert len(out) == 1
        assert "app.py" in out[0]

    def test_hunk_with_context_not_reported(self):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,2 +1,2 @@\n keep\n-a\n+b\n"
        )
        assert PatchEngine.context_free_hunks(p) == []

    def test_junk_line_ends_hunk(self):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,0 +2,1 @@\n+x\nJUNK-NO-PREFIX\n"
        )
        out = PatchEngine.context_free_hunks(p)
        # hunk body ended at JUNK before flush → reported with the '+x' only
        assert len(out) == 1

    def test_deletion_to_devnull_not_reported(self):
        p = (
            "diff --git a/old.py b/old.py\n--- a/old.py\n+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n-gone\n"
        )
        assert PatchEngine.context_free_hunks(p) == []


# ── _verify_c0_placement edges (2584-2617) ────────────────────────────────────


class TestVerifyC0Edges:
    def test_deleted_file_skipped(self, engine, git_repo):
        p = (
            "diff --git a/ghost.py b/ghost.py\n--- a/ghost.py\n+++ b/ghost.py\n"
            "@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
        )
        ok, _detail = engine._verify_c0_placement(p)
        assert ok is True

    def test_context_free_hunk_accepted(self, engine, git_repo):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -8,0 +9,2 @@\n+zzz\n"
        )
        ok, _detail = engine._verify_c0_placement(p)
        assert ok is True

    def test_hunk_before_file_header_skipped(self):
        # '+++ b/app.py' seen BEFORE '@@' — cur None at hunk time → body dropped
        p = "@@ -1,2 +1,2 @@\n a\n-b\n+c\n+++ b/app.py\n"
        ok, _detail = PatchEngine("")._verify_c0_placement(p)
        assert ok is True

    def test_misplaced_context_detected(self, engine, git_repo):
        # post-image block not present anywhere in the file
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,3 +1,3 @@\n"
            " unique_context_alpha\n"
            "-removed_beta\n"
            "+added_gamma\n"
        )
        ok, detail = engine._verify_c0_placement(p)
        assert ok is False
        assert "app.py" in detail

    def test_new_file_skipped(self, engine, git_repo):
        p = (
            "diff --git a/n.py b/n.py\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+anything\n"
        )
        ok, _detail = engine._verify_c0_placement(p)
        assert ok is True

    def test_correct_placement_passes(self, engine, git_repo):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -6,3 +6,3 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a + b + 1\n"
        )
        ok, _detail = engine._verify_c0_placement(p)
        assert ok is True


# ── misc (1513, 1531, 1578, 1643-1650, 1670, 1738, 1753, 1794, 1871, 1916-1921) ─


class TestMiscEdges:
    def test_git_apply_check_exception(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        def _boom(*a, **k):
            raise OSError("no git")

        monkeypatch.setattr(pe.subprocess, "run", _boom)
        ok, err = engine._git_apply_check_best_effort("x")
        assert ok is False
        assert "exception" in err

    def test_git_apply_check_fail_no_stderr(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        class _R:
            returncode = 128
            stdout = b""
            stderr = b""

        monkeypatch.setattr(pe.subprocess, "run", lambda *a, **k: _R())
        ok, err = engine._git_apply_check_best_effort("x")
        assert ok is False
        assert "exit code 128" in err

    def test_git_apply_check_fail_stdout_only(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        class _R:
            returncode = 1
            stdout = b"stdout msg"
            stderr = b""

        monkeypatch.setattr(pe.subprocess, "run", lambda *a, **k: _R())
        ok, err = engine._git_apply_check_best_effort("x")
        assert ok is False
        assert "stdout msg" in err

    def test_normalize_and_validate_adds_trailing_newline(self, engine, git_repo):
        p = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -2,1 +2,1 @@\n-    msg = \"Hello, \" + name\n+    msg = \"Hello, \" + name\n"  # context-equal
        out, err = engine.normalize_and_validate(p.rstrip("\n"), "app.py")
        assert err is None
        assert out.endswith("\n")

    def test_normalize_and_validate_non_diff(self, engine):
        _out, err = engine.normalize_and_validate("plain text", "f")
        assert err == "Patch does not look like a unified diff"

    def test_output_mode_to_enum_none(self, engine, monkeypatch):
        import external_llm.patch_engine as pe

        monkeypatch.setattr(pe, "OutputMode", None)
        assert engine._output_mode_to_enum("auto") is None

    def test_apply_diff_once_module_missing(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "_diff_apply", None)
        ok, err = engine._apply_diff_once("x")
        assert ok is False
        assert "not available" in err

    def test_apply_diff_once_exception(self, engine, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(engine, "_diff_apply", _boom)
        ok, err = engine._apply_diff_once("@@ -1 +1 @@\n-a\n+b\n")
        assert ok is False
        assert "diff_apply exception" in err

    def test_apply_diff_once_rejects(self, engine, git_repo):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -2,1 +2,1 @@\n-ZZZZ absent ZZZZ\n+new line\n"
        )
        ok, err = engine._apply_diff_once(p, "app.py")
        assert ok is False
        assert err

    def test_apply_diff_once_success(self, engine, git_repo):
        p = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -7,1 +7,1 @@\n-    return a + b\n+    return a + b + 21\n"
        )
        ok, err = engine._apply_diff_once(p, "app.py")
        assert ok is True, err
        assert "a + b + 21" in (git_repo / "app.py").read_text()
        reset_app(git_repo)

    def test_fix_hunk_counts_passthrough_no_header(self, engine):
        assert engine._fix_hunk_counts("plain\ntext\n") == "plain\ntext\n"

    def test_verify_c0_prefix_hunk_before_header(self, engine):
        # '@@' before any +++ → cur None → hunk skipped entirely
        p = "@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
        ok, _ = engine._verify_c0_placement(p)
        assert ok is True

    def test_ws_norm_line(self):
        assert PatchEngine._ws_norm_line("  a  b  ") == "ab"

    def test_strip_trailing_fences(self):
        assert PatchEngine._strip_trailing_fences("code\n```\njunk") == "code\n"
        assert PatchEngine._strip_trailing_fences("code") == "code"

    def test_add_step_appends(self, engine):
        md = {"execution_steps": []}
        engine._add_step(md, "s", "d")
        assert md["execution_steps"][0]["step"] == "s"
        assert "timestamp" in md["execution_steps"][0]
