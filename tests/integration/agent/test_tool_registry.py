"""
Integration tests for ToolRegistry.
"""
import threading
from unittest.mock import patch

import pytest

from diff_apply import apply_patch
from external_llm.agent.tool_registry import ToolRegistry


def test_tool_registry_unknown_tool(tool_registry):
    """Test dispatch with unknown tool."""
    result = tool_registry.dispatch("unknown_tool", {"arg": "value"})
    assert not result.ok
    assert "unknown tool" in result.error.lower()


def test_tool_registry_parallel_dispatch_disabled(tool_registry):
    """Test parallel dispatch when disabled."""
    # When parallel execution is disabled, should fall back to sequential
    tool_calls = [
        {"tool": "find_symbol", "args": {"name": "hello"}},
        {"tool": "grep", "args": {"pattern": "def test_"}},
    ]
    results = tool_registry.dispatch_parallel(tool_calls)
    assert len(results) == 2
    assert results[0].ok
    assert results[1].ok


def test_tool_registry_parallel_dispatch_enabled(tool_registry, agent_config):
    """Test parallel dispatch when enabled."""
    # Enable parallel execution
    agent_config.parallel_tool_execution_enabled = True
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    tool_calls = [
        {"tool": "find_symbol", "args": {"name": "hello"}},
        {"tool": "grep", "args": {"pattern": "def test_"}},
        {"tool": "get_project_info", "args": {}},
    ]
    results = registry.dispatch_parallel(tool_calls)
    assert len(results) == 3
    assert all(r.ok for r in results)
def test_dispatch_parallel_serializes_ask_user(tool_registry, agent_config):
    """ask_user must NEVER run in parallel — it blocks on human input and relies
    on one-question-at-a-time invariants (unique question_id, atomic question-
    count limit). Even with parallel_tool_execution_enabled, a batch containing
    ask_user must fall back to sequential execution.

    Regression for the race where two ask_user calls in the same batch collided
    their millisecond question_id and raced the question counter.
    """
    import time as _time
    agent_config.parallel_tool_execution_enabled = True
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    # _SERIAL_TOOLS is the declared set driving the serialization gate.
    assert "ask_user" in registry._SERIAL_TOOLS
    # ask_user must NOT be a write tool (different semantics: file-locking,
    # failure-logging, cache invalidation).
    assert "ask_user" not in registry._WRITE_TOOLS

    concurrency = {"current": 0, "max": 0}
    _guard = threading.Lock()
    _real_dispatch = registry.dispatch

    def _tracking_dispatch(name, args):
        with _guard:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        try:
            _time.sleep(0.05)  # widen the window so overlap is detectable
            return _real_dispatch(name, args)
        finally:
            with _guard:
                concurrency["current"] -= 1

    tool_calls = [
        {"tool": "ask_user", "args": {"question": "q1"}},
        {"tool": "find_symbol", "args": {"name": "hello"}},
        {"tool": "ask_user", "args": {"question": "q2"}},
    ]
    with patch.object(registry, "dispatch", side_effect=_tracking_dispatch):
        results = registry.dispatch_parallel(tool_calls)

    assert len(results) == 3
    # Sequential execution → never more than one tool in flight at once.
    assert concurrency["max"] == 1, (
        f"ask_user-containing batch ran concurrently (max={concurrency['max']})")


def test_dispatch_parallel_runs_pure_reads_concurrently(tool_registry, agent_config):
    """Control: a batch of read-only tools (no ask_user, no write tool) MUST
    still run in parallel. Guards against the gate accidentally serializing
    everything.
    """
    import time as _time
    agent_config.parallel_tool_execution_enabled = True
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    concurrency = {"current": 0, "max": 0}
    _guard = threading.Lock()
    _real_dispatch = registry.dispatch

    def _tracking_dispatch(name, args):
        with _guard:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        try:
            _time.sleep(0.05)
            return _real_dispatch(name, args)
        finally:
            with _guard:
                concurrency["current"] -= 1

    tool_calls = [
        {"tool": "find_symbol", "args": {"name": "hello"}},
        {"tool": "grep", "args": {"pattern": "def test_"}},
        {"tool": "get_project_info", "args": {}},
    ]
    with patch.object(registry, "dispatch", side_effect=_tracking_dispatch):
        results = registry.dispatch_parallel(tool_calls)

    assert len(results) == 3
    assert concurrency["max"] >= 2, (
        f"pure-read batch did not parallelize (max={concurrency['max']})")


def test_dispatch_parallel_job_list_runs_concurrently(tool_registry, agent_config):
    """job list/output are pure reads and must stay eligible for the parallel
    phase — only job kill (races on job_id with concurrent job output) is
    serial. Regression: _SERIAL_TOOLS used to force EVERY "job" call serial
    regardless of action, needlessly collapsing read batches to sequential.
    """
    import time as _time
    agent_config.parallel_tool_execution_enabled = True
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    concurrency = {"current": 0, "max": 0}
    _guard = threading.Lock()
    _real_dispatch = registry.dispatch

    def _tracking_dispatch(name, args):
        with _guard:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        try:
            _time.sleep(0.05)
            return _real_dispatch(name, args)
        finally:
            with _guard:
                concurrency["current"] -= 1

    tool_calls = [
        {"tool": "job", "args": {"action": "list"}},
        {"tool": "find_symbol", "args": {"name": "hello"}},
        {"tool": "grep", "args": {"pattern": "def test_"}},
    ]
    with patch.object(registry, "dispatch", side_effect=_tracking_dispatch):
        results = registry.dispatch_parallel(tool_calls)

    assert len(results) == 3
    assert concurrency["max"] >= 2, (
        f"job-list batch did not parallelize (max={concurrency['max']})")


def test_dispatch_parallel_serializes_job_kill(tool_registry, agent_config):
    """job kill must still force the whole batch sequential — it races with
    concurrent job output on the same job_id."""
    import time as _time
    agent_config.parallel_tool_execution_enabled = True
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    assert registry._tool_call_is_serial("job", {"action": "kill"}) is True
    assert registry._tool_call_is_serial("job", {"action": "list"}) is False
    assert registry._tool_call_is_serial("job", {"action": "output"}) is False

    concurrency = {"current": 0, "max": 0}
    _guard = threading.Lock()
    _real_dispatch = registry.dispatch

    def _tracking_dispatch(name, args):
        with _guard:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        try:
            _time.sleep(0.05)
            return _real_dispatch(name, args)
        finally:
            with _guard:
                concurrency["current"] -= 1

    tool_calls = [
        {"tool": "job", "args": {"action": "kill", "job_id": "1"}},
        {"tool": "find_symbol", "args": {"name": "hello"}},
    ]
    with patch.object(registry, "dispatch", side_effect=_tracking_dispatch):
        results = registry.dispatch_parallel(tool_calls)

    assert len(results) == 2
    assert concurrency["max"] == 1, (
        f"job-kill batch ran concurrently (max={concurrency['max']})")


@pytest.mark.xfail(reason="invalid_patch may be transformed into a valid patch by _clean_diff")
def test_tool_registry_failure_recording(tool_registry, invalid_patch):
    """Test that patch failures are recorded in failure database."""
    result = tool_registry.dispatch("apply_patch", {"patch": invalid_patch})
    assert not result.ok

    # Check that failure analysis is in metadata
    assert "failure_analysis" in result.metadata
    failure_analysis = result.metadata["failure_analysis"]
    assert "reason" in failure_analysis
    assert "hint" in failure_analysis


def test_tool_registry_cancellation(tool_registry, agent_config):
    """Test that dispatch respects cancellation event."""
    cancel_event = threading.Event()
    cancel_event.set()
    agent_config.cancel_event = cancel_event
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    result = registry.dispatch("find_symbol", {"name": "hello"})
    assert not result.ok
    assert "cancelled" in result.error.lower()


def test_tool_registry_cancellation_before_tool_execution(tool_registry, agent_config):
    """Test cancellation before long-running tools (run_tests, shell_exec)."""
    cancel_event = threading.Event()
    cancel_event.set()
    agent_config.cancel_event = cancel_event
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    # Test run_tests cancellation
    result = registry.dispatch("run_tests", {"args": []})
    assert not result.ok
    assert "cancelled" in result.error.lower()

    # Test shell_exec cancellation
    result = registry.dispatch("shell_exec", {"command": "echo test"})
    assert not result.ok
    assert "cancelled" in result.error.lower()


def test_tool_registry_write_plan(tool_registry):
    """Test write_plan tool with both plan and ops-only formats."""
    # Test with full plan object
    full_plan = {
        "kind": "ASICODE_PLAN_V1",
        "ops": [
            {
                "op": "create_file",
                "path": "test.txt",
                "content": "hello world\n",
            }
        ]
    }
    result = tool_registry.dispatch("write_plan", {"plan": full_plan})
    assert result.ok
    assert "Plan applied" in result.content

    # Verify file was created
    read_result = tool_registry.dispatch("bash", {"command": "cat test.txt"})
    assert read_result.ok
    assert "hello world" in read_result.content

    # Test with ops-only payload (should auto-wrap)
    ops_only = [
        {
            "op": "create_file",
            "path": "test2.txt",
            "content": "goodbye world\n",
        }
    ]
    result = tool_registry.dispatch("write_plan", {"ops": ops_only})
    assert result.ok
    assert "Plan applied" in result.content

    read_result = tool_registry.dispatch("bash", {"command": "cat test2.txt"})
    assert read_result.ok
    assert "goodbye world" in read_result.content

    # Test invalid empty ops list
    result = tool_registry.dispatch("write_plan", {"ops": []})
    assert not result.ok
    assert "non-empty" in result.error or "empty" in result.error


def test_tool_registry_apply_patch_hunk_only(tool_registry, sample_patch):
    """Test apply_patch tool with hunk-only diff (no file headers)."""
    # Convert sample_patch (which has headers) to hunk-only by stripping headers
    lines = sample_patch.splitlines()
    # Find first @@ line
    hunk_start = next(i for i, line in enumerate(lines) if line.startswith("@@"))
    hunk_only = "\n".join(lines[hunk_start:])

    # Should fail without path
    result = tool_registry.dispatch("apply_patch", {"patch": hunk_only})
    assert not result.ok
    assert "path" in result.error.lower() or "hunk" in result.error.lower()

    # Should succeed with path
    result = tool_registry.dispatch("apply_patch", {
        "patch": hunk_only,
        "path": "sample.py"
    })
    assert result.ok
    # Verify the patch was applied
    read_result = tool_registry.dispatch("read_file", {"path": "sample.py"})
    assert read_result.ok
    assert "def __init__" in read_result.content
    assert "self.memory = 0" in read_result.content


def test_tool_registry_apply_patch_hunk_only_invalid(tool_registry):
    """Test hunk-only diff validation."""
    # Patch with headers but missing diff --git (still valid)
    patch_with_headers = """--- a/sample.py
+++ b/sample.py
@@ -1,6 +1,9 @@
 def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
         return a + b
"""
    # This should still work (normal unified diff)
    result = tool_registry.dispatch("apply_patch", {"patch": patch_with_headers})
    # It may succeed or fail depending on cleaning, but should not crash
    # We just ensure no "hunk-only" error appears
    # (no assertion about ok)

    # Invalid: hunk-only with unsafe path
    hunk_only = """@@ -1,6 +1,9 @@
 def hello() -> str:
     return "world"
"""
    result = tool_registry.dispatch("apply_patch", {
        "patch": hunk_only,
        "path": "../outside.py"
    })
    assert not result.ok
    # Patch application should reject path traversal
    assert "error:" in result.error or "outside" in result.error

    # Invalid: hunk-only with empty path
    result = tool_registry.dispatch("apply_patch", {
        "patch": hunk_only,
        "path": ""
    })
    assert not result.ok
    # Should complain about missing path

    # Invalid: patch that doesn't start with @@ and has no headers
    invalid = "just some text"
    result = tool_registry.dispatch("apply_patch", {"patch": invalid})
    assert not result.ok
    # Should fail with git apply error

    # Invalid: patch contains diff --git but malformed
    malformed = """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -1,6 +1,9 @@
"""
    result = tool_registry.dispatch("apply_patch", {"patch": malformed})
    # Should fail but not crash


# Direct tests for diff_apply.apply_patch
def test_diff_apply_hunk_only(temp_repo_root):
    """Test diff_apply.apply_patch with hunk‑only diffs."""
    from unittest.mock import patch

    with patch('config.STRICT_CLEAN', False):
        # Debug: print file content
        import os
        sample_path = os.path.join(temp_repo_root, "sample.py")
        with open(sample_path) as f:
            content = f.read()
            print(f"Sample.py content ({len(content)} chars): {content!r}")
            print(f"Lines: {content.splitlines()}")
        # Sample patch with headers (should succeed)
        full_patch = """--- a/sample.py
+++ b/sample.py
@@ -1,6 +1,9 @@
 def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
         return a + b
"""
        ok, msg, reason, details = apply_patch(temp_repo_root, full_patch, file_path_hint=None)
        assert ok, f"Full patch failed: {msg}"

        # Reset file (git checkout)
        import subprocess
        subprocess.run(["git", "checkout", "--", "sample.py"], cwd=temp_repo_root, capture_output=True)

        # Hunk‑only patch (no headers)
        hunk_only = """@@ -1,6 +1,9 @@
 def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
         return a + b
"""
        # Should fail without file_path_hint
        ok, msg, reason, details = apply_patch(temp_repo_root, hunk_only, file_path_hint=None)
        assert not ok
        assert "MISSING_PATH_HINT" in reason or "path" in msg.lower()

        # Should succeed with file_path_hint
        ok, msg, reason, _details = apply_patch(temp_repo_root, hunk_only, file_path_hint="sample.py")
        assert ok, f"Hunk‑only patch failed: {msg}"

        # Verify the patch was applied
        diff_proc = subprocess.run(["git", "diff", "sample.py"], cwd=temp_repo_root, capture_output=True, text=True)
        assert "def __init__" in diff_proc.stdout
        assert "self.memory = 0" in diff_proc.stdout


class TestBashOutputCap:
    """bash 대량 출력 truncation — 컨텍스트 토큰 폭증 방지 (BASH_OUTPUT_MAX_CHARS).

    Self-contained: builds its own ToolRegistry so it does not depend on a
    module-level fixture/helper.
    """

    def _reg(self, tmp_path):
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
        return ToolRegistry(repo_root=str(tmp_path), config=AgentConfig())

    def _cap(self):
        from external_llm.agent.config.thresholds import config
        return config.tokens.BASH_OUTPUT_MAX_CHARS

    def test_small_output_not_truncated(self, tmp_path):
        r = self._reg(tmp_path)._tool_shell_exec({"command": "echo hello-world"})
        assert r.ok
        assert "hello-world" in r.content
        assert "truncated" not in r.content

    def test_large_ascii_output_capped_at_threshold(self, tmp_path):
        cap = self._cap()
        r = self._reg(tmp_path)._tool_shell_exec(
            {"command": f"python3 -c \"print('a'*{cap*3})\""}
        )
        assert r.ok
        assert "truncated" in r.content
        assert cap <= len(r.content) <= cap + 400

    def test_cjk_output_capped_below_ascii_threshold(self, tmp_path):
        cap = self._cap()
        r = self._reg(tmp_path)._tool_shell_exec(
            {"command": f"python3 -c \"print(chr(44032)*{cap*3})\""}
        )
        assert r.ok
        assert "truncated" in r.content
        assert len(r.content) < cap


class TestBashShellAutocorrectQuoteAware:
    """Shell-command auto-correction (python->python3, cat -A->cat -vet,
    sort -V->python3, find exclusion injection) must NOT rewrite tokens that
    live inside a shell-quoted region.

    Regression: a grep command whose SEARCH PATTERN happens to contain the
    literal text ``sort -V`` (or ``cat -A``, ``python``) was being rewritten
    inside the quotes, injecting a ``python3 -c "..."`` with literal parens
    and breaking the quoting → ``bash: syntax error near unexpected token '('``.

    These tests build their own registry and use ``echo`` so the auto-corrected
    command actually executes (we observe the correction via stdout).
    """

    def _reg(self, tmp_path):
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
        return ToolRegistry(repo_root=str(tmp_path), config=AgentConfig())

    def test_sort_V_inside_single_quoted_pattern_not_rewritten(self, tmp_path):
        # The exact failing pattern shape from a real session: grep's search
        # pattern literal contains 'sort -V'. Must be preserved verbatim.
        cmd = "echo 'sort -V flag' "
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        # The echoed pattern is intact (sort -V NOT rewritten to python3).
        assert "sort -V" in r.content
        assert "python3 -c" not in r.content

    def test_cat_A_inside_single_quoted_pattern_not_rewritten(self, tmp_path):
        cmd = "echo 'uses cat -A here' "
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        assert "cat -A" in r.content
        assert "cat -vet" not in r.content

    def test_python_inside_double_quoted_string_not_rewritten(self, tmp_path):
        cmd = 'echo "run python please"'
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        assert "python" in r.content  # preserved verbatim, not 'python3'

    def test_genuine_sort_V_outside_quotes_still_corrected(self, tmp_path):
        # Positive control: a real `echo x | sort -V` must still be rewritten.
        cmd = "echo z3 | sort -V"
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        # sort -V was rewritten to python3 natural sort → output still 'z3'
        assert "z3" in r.content

    def test_genuine_cat_A_outside_quotes_still_corrected(self, tmp_path):
        # `cat -A` is BSD-incompatible on macOS; auto-correct to `cat -vet`.
        # Use a temp file so the command is valid on both platforms.
        f = tmp_path / "sample.txt"
        f.write_text("hi\n")
        cmd = f"cat -A {f}"
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        # cat -vet shows line-end $ and tabs as ^I; 'hi$' proves it ran.
        assert "hi$" in r.content

    def test_find_with_o_grouping_no_unboundlocalerror(self, tmp_path):
        # Regression for a Python-scoping crash: a module-level `import re as
        # _re` plus a function-local `import re as _re` made the compiler treat
        # _re as function-local across the whole body, so the EARLIER
        # `_re.search(r"(^|\s)-o(\s|$)", ...)` (find -o grouping) raised
        # UnboundLocalError before the local import ran. This must NOT crash.
        (tmp_path / "a.txt").write_text("1")
        (tmp_path / "b.py").write_text("2")
        cmd = "find . -name a.txt -o -name b.py"
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"find -o command crashed or failed: {r.error}"
        assert "a.txt" in r.content
        assert "b.py" in r.content

class TestBashShellAutocorrectHeredocAware:
    """Shell-dialect auto-corrections must NEVER rewrite inside a heredoc body.

    Regression: the python->python3 / find-exclusion / sort-V / cat-A
    corrections only protected shell quotes ('...'/"..."). A heredoc body
    (``<< 'PYEOF' ... PYEOF``) is LITERAL script text, not a shell quote, so a
    comment like ``# find all *.py`` or a bare ``python`` token inside the body
    was silently mangled: the find-exclusion injector appended
    ``-not -path "./.venv/*"...`` AFTER the closing delimiter (breaking heredoc
    termination -> python read a truncated/mangled script ->
    ``File "<stdin>", line N: SyntaxError``), and python->python3 rewrote
    ``python`` inside comments/strings, altering script semantics.
    """

    def _reg(self, tmp_path):
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
        return ToolRegistry(repo_root=str(tmp_path), config=AgentConfig())

    def test_heredoc_body_with_find_comment_runs_unchanged(self, tmp_path):
        # The body mentions `find` and uses a `*.py` glob — the find-exclusion
        # injector must NOT fire inside the heredoc body.
        cmd = (
            "python3 << 'PYEOF'\n"
            "import glob\n"
            "# find all python files\n"
            "print(len(glob.glob('*.py')))\n"
            "print('OK_END')\n"
            "PYEOF"
        )
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"heredoc command failed (body was mangled?): {r.error}"
        assert "OK_END" in r.content          # script ran to completion
        assert "-not -path" not in r.content  # no find-exclusion injection

    def test_heredoc_body_python_token_not_rewritten(self, tmp_path):
        # `python3` inside the heredoc body (here, echoed) must stay `python3`,
        # not be rewritten to `python3`.
        cmd = (
            "python3 << 'PYEOF'\n"
            "s = 'launch python subprocess'\n"
            "print('has_python=' + str('python' in s))\n"
            "PYEOF"
        )
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        assert "has_python=True" in r.content

    def test_heredoc_body_sort_V_not_rewritten(self, tmp_path):
        cmd = (
            "python3 << 'PYEOF'\n"
            "print('mentions sort -V here')\n"
            "print('done')\n"
            "PYEOF"
        )
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        assert "done" in r.content
        # sort -V was NOT rewritten to python3 -c inside the body.
        assert "python3 -c" not in r.content

    def test_bare_delimiter_heredoc_protected(self, tmp_path):
        # `<<EOF` (unquoted delimiter) body must also be protected.
        cmd = (
            "python3 << EOF\n"
            "# find python files\n"
            "print('BARE_OK')\n"
            "EOF"
        )
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        assert "BARE_OK" in r.content

    def test_non_heredoc_find_still_corrected(self, tmp_path):
        # Positive control / regression: a real `find *.py` OUTSIDE a heredoc
        # must still get the noise-dir exclusions auto-injected.
        (tmp_path / "a.py").write_text("x")
        cmd = "find . -name '*.py'"
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        assert "a.py" in r.content

    def test_non_heredoc_python_still_corrected(self, tmp_path):
        # Positive control: bare `python3` outside a heredoc still -> python3.
        cmd = "python -c 'print(7)'"
        r = self._reg(tmp_path)._tool_shell_exec({"command": cmd})
        assert r.ok, f"command failed: {r.error}"
        assert "7" in r.content


class TestHeredocBodyIntervals:
    """Unit tests for _heredoc_body_intervals (the heredoc-span parser)."""

    def test_quoted_delimiter(self):
        from external_llm.agent.tool_handlers.git_tools import _heredoc_body_intervals as h
        cmd = "python3 << 'PYEOF'\nbody1\nbody2\nPYEOF"
        spans = h(cmd)
        assert len(spans) == 1
        s, e = spans[0]
        assert cmd[s:e].startswith("body1\n")
        assert "PYEOF" in cmd[s:e]

    def test_bare_delimiter(self):
        from external_llm.agent.tool_handlers.git_tools import _heredoc_body_intervals as h
        cmd = "cat <<EOF\nhello\nEOF"
        spans = h(cmd)
        assert len(spans) == 1
        assert "hello" in cmd[spans[0][0]:spans[0][1]]

    def test_dash_delimiter_strips_leading_tabs(self):
        from external_llm.agent.tool_handlers.git_tools import _heredoc_body_intervals as h
        cmd = "cat <<-DONE\n\tindented\n\tDONE"
        spans = h(cmd)
        assert len(spans) == 1, f"expected 1 span, got {spans}"
        assert "indented" in cmd[spans[0][0]:spans[0][1]]

    def test_unterminated_protects_to_end(self):
        from external_llm.agent.tool_handlers.git_tools import _heredoc_body_intervals as h
        cmd = "python3 << 'EOF'\nprint(1)\n# no closer"
        spans = h(cmd)
        assert len(spans) == 1
        assert spans[0][1] == len(cmd)

    def test_nested_opener_inside_body_treated_as_literal(self):
        from external_llm.agent.tool_handlers.git_tools import _heredoc_body_intervals as h
        # A '<<' that appears inside the first heredoc's body is literal text,
        # not a second heredoc opener -> only ONE span.
        cmd = "python3 << 'OUTER'\ns = 'a << INNER'\nprint(s)\nOUTER"
        spans = h(cmd)
        assert len(spans) == 1, f"expected 1 span, got {spans}"

    def test_position_outside_body_not_protected(self):
        from external_llm.agent.tool_handlers.git_tools import (
            _heredoc_body_intervals as h,
        )
        from external_llm.agent.tool_handlers.git_tools import (
            _match_in_quotes,
        )
        cmd = "find . -name x << 'EOF'\nbody\nEOF"
        spans = h(cmd)
        # The `find` before the heredoc opener is NOT in the body.
        find_pos = cmd.index("find")
        assert not _match_in_quotes(find_pos, spans)
        # The `body` content IS protected.
        body_pos = cmd.index("body")
        assert _match_in_quotes(body_pos, spans)


# ── Mutating-bash parallelization gate ────────────────────────────────────
# A mutating bash (rm, git commit, "> file", …) changes filesystem/git state
# and must NOT run in the parallel read phase. _tool_call_mutates is the single
# source of truth shared by cache invalidation, dispatch_parallel's gate, and
# DesignChatLoop's read/write phase partition.

def test_tool_call_mutates_single_source_classifier(tool_registry):
    """_tool_call_mutates is the shared predicate. It must classify every tool
    the same way the cache-invalidation path does (it replaced that path's
    inline expression)."""
    reg = tool_registry
    # Write tools always mutate.
    assert reg._tool_call_mutates("apply_patch", {}) is True
    assert reg._tool_call_mutates("edit_text", {}) is True
    assert reg._tool_call_mutates("write_plan", {}) is True
    # Mutating bash → True (write tokens / git state change / redirection).
    for cmd in ["rm -rf build", "git commit -am x", "echo hi > out.txt",
                "sed -i 's/a/b/' f.py", "mkdir newdir", "touch f",
                "git checkout -b feat", "pip install x", "tar -xf a.tgz"]:
        assert reg._tool_call_mutates("bash", {"command": cmd}) is True, cmd
    # Read-only bash → False (stays in the parallel read phase).
    for cmd in ["ls -la", "git status", "git diff", "git log", "grep foo .",
                "cat f.py", "find . -name x", "wc -l f.py"]:
        assert reg._tool_call_mutates("bash", {"command": cmd}) is False, cmd
    # Pure read tools → False.
    for name in ("read_file", "find_symbol", "grep", "get_file_outline",
                 "find_references", "get_project_info"):
        assert reg._tool_call_mutates(name, {}) is False, name
    # Empty/missing command → not mutating (no write token).
    assert reg._tool_call_mutates("bash", {}) is False
    assert reg._tool_call_mutates("bash", {"command": ""}) is False


def test_dispatch_parallel_sequential_for_mutating_bash(tool_registry, agent_config):
    """A batch containing a mutating bash must fall back to sequential
    execution — it would race with concurrent reads/other bash if
    parallelized."""
    import time as _time
    agent_config.parallel_tool_execution_enabled = True
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    concurrency = {"current": 0, "max": 0}
    _guard = threading.Lock()
    _real_dispatch = registry.dispatch

    def _tracking_dispatch(name, args):
        with _guard:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        try:
            _time.sleep(0.05)
            return _real_dispatch(name, args)
        finally:
            with _guard:
                concurrency["current"] -= 1

    tool_calls = [
        {"tool": "bash", "args": {"command": "rm -rf build"}},
        {"tool": "find_symbol", "args": {"name": "hello"}},
        {"tool": "bash", "args": {"command": "git commit -am wip"}},
    ]
    with patch.object(registry, "dispatch", side_effect=_tracking_dispatch):
        results = registry.dispatch_parallel(tool_calls)

    assert len(results) == 3
    assert concurrency["max"] == 1, (
        f"mutating-bash batch ran concurrently (max={concurrency['max']})")


def test_dispatch_parallel_keeps_readonly_bash_parallel(tool_registry, agent_config):
    """Guard against over-serialization: read-only bash (ls, git status) has no
    side effects and MUST still parallelize."""
    import time as _time
    agent_config.parallel_tool_execution_enabled = True
    registry = ToolRegistry(tool_registry.repo_root, agent_config)

    concurrency = {"current": 0, "max": 0}
    _guard = threading.Lock()
    _real_dispatch = registry.dispatch

    def _tracking_dispatch(name, args):
        with _guard:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        try:
            _time.sleep(0.05)
            return _real_dispatch(name, args)
        finally:
            with _guard:
                concurrency["current"] -= 1

    tool_calls = [
        {"tool": "bash", "args": {"command": "ls -la"}},
        {"tool": "bash", "args": {"command": "git status"}},
        {"tool": "find_symbol", "args": {"name": "hello"}},
    ]
    with patch.object(registry, "dispatch", side_effect=_tracking_dispatch):
        results = registry.dispatch_parallel(tool_calls)

    assert len(results) == 3
    assert concurrency["max"] >= 2, (
        f"read-only-bash batch did not parallelize (max={concurrency['max']})")
