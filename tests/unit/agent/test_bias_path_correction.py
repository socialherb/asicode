"""Tests for ToolRegistry._correct_bias_path literal-region protection.

Locks in two contracts:
  1. LLM training-data bias paths (/workspace, /app, ...) in BARE command
     arguments are rewritten to repo_root (the original intent).
  2. bias paths inside shell-quoted literals ('...' / "...") or heredoc bodies
     (<<'EOF' ... EOF) are LEFT INTACT — rewriting there corrupts literal content
     (a grep search pattern, a config/script written via heredoc, ...).

Contract (2) makes _correct_bias_path consistent with the shell-dialect
auto-corrections in git_tools, which already respect _literal_intervals
(heredoc-body protection landed in commit 1fdc264c). Before this fix
_correct_bias_path was the one remaining preprocessor running raw re.sub, so it
silently rewrote /workspace inside a heredoc body — live reproduction: a
<<'PYEOF' python fixture had its /workspace/asicode token rewritten to the
real repo path before python3 ever saw it.
"""
import types

import pytest

from external_llm.agent.tool_registry import ToolRegistry

# NOTE: REPO_ROOT must not contain any bias token (/workspace, /app, /project,
# /code, /repo) as a substring — otherwise the ``bias_token not in out`` guard
# below is spuriously satisfied/violated by the rewritten path itself.
REPO_ROOT = "/opt/work/myproj"


@pytest.fixture
def bias():
    """A bound _correct_bias_path with a fixed repo_root (no registry needed)."""
    stub = types.SimpleNamespace(repo_root=REPO_ROOT)
    return ToolRegistry._correct_bias_path.__get__(stub, type(stub))


# ── (1) bare arguments ARE rewritten — existing intent preserved ─────────────

@pytest.mark.parametrize("bias_token", ["/workspace", "/app", "/project", "/code", "/repo"])
def test_bare_bias_path_rewritten(bias, bias_token):
    out = bias(f"cat {bias_token}/myproj/tests/x.py")
    assert out == f"cat {REPO_ROOT}/tests/x.py"
    assert bias_token not in out


def test_cd_bias_path_rewritten(bias):
    assert bias("cd /workspace/myproj && pwd") == f"cd {REPO_ROOT} && pwd"


# ── (2) literal regions are PROTECTED — the fix under test ───────────────────

def test_single_quoted_search_pattern_protected(bias):
    # The path is a grep SEARCH PATTERN, not a real path — rewriting it would
    # make the grep silently match nothing.
    cmd = "grep '/workspace/myproj' file.py"
    assert bias(cmd) == cmd


def test_double_quoted_literal_protected(bias):
    cmd = 'echo "root is /workspace/myproj"'
    assert bias(cmd) == cmd


def test_heredoc_body_protected(bias):
    # A script written via heredoc that legitimately references /workspace
    # (e.g. an in-container path) must survive untouched.
    cmd = "cat > run.sh <<'EOF'\nROOT=/workspace/myproj\nEOF"
    assert bias(cmd) == cmd


def test_unquoted_heredoc_body_protected(bias):
    cmd = "python3 - <<EOF\nprint('/workspace/myproj')\nEOF"
    assert bias(cmd) == cmd


# ── mixed: rewrite + protect coexist (offset-shift safety) ───────────────────

def test_bare_rewrite_and_quoted_protect_in_same_command(bias):
    # A bare bias path is rewritten while a quoted one in the SAME command is
    # left intact. Guards against protected-interval offsets going stale after
    # the bare rewrite changes the string length mid-command.
    cmd = 'cd /workspace/myproj && grep "/workspace/myproj" *.py'
    out = bias(cmd)
    assert f"cd {REPO_ROOT}" in out
    assert 'grep "/workspace/myproj"' in out


# ── idempotency ──────────────────────────────────────────────────────────────

def test_real_repo_path_is_idempotent(bias):
    cmd = f"cd {REPO_ROOT} && echo hi"
    assert bias(cmd) == cmd


def test_empty_input(bias):
    assert bias("") == ""
