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


# ── pass 2: repo-basename correction fires on ABSOLUTE tokens only ───────────
# Training-data bias paths are virtual roots, always spelled absolute
# (/home/ubuntu/myproj, ~/myproj, /myproj). A RELATIVE token that merely ends
# with the repo basename is real data — live incident 2026-08-02: a branch
# literally named `rename/asicode` had `git rev-parse rename/asicode` rewritten
# to `git rev-parse <repo_root>`, which resolves to nothing and reads as ref
# corruption.

@pytest.mark.parametrize(
    "cmd",
    [
        "git rev-parse rename/myproj",
        "git symbolic-ref HEAD refs/heads/rename/myproj",
        "git log feature/myproj --oneline",
    ],
)
def test_relative_token_ending_in_basename_is_not_rewritten(bias, cmd):
    assert bias(cmd) == cmd


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("cat /home/ubuntu/myproj/tests/x.py", f"cat {REPO_ROOT}/tests/x.py"),
        ("ls ~/myproj/src", f"ls {REPO_ROOT}/src"),
        ("cat /myproj/tests/x.py", f"cat {REPO_ROOT}/tests/x.py"),
    ],
)
def test_absolute_embedded_basename_is_still_rewritten(bias, cmd, expected):
    assert bias(cmd) == expected


# ── idempotency ──────────────────────────────────────────────────────────────

def test_real_repo_path_is_idempotent(bias):
    cmd = f"cd {REPO_ROOT} && echo hi"
    assert bias(cmd) == cmd


def test_empty_input(bias):
    assert bias("") == ""


# ── pass 1/2: REAL existing paths are NEVER rewritten ─────────────────────────
# The pass-2 prefix regex matches ANY absolute path ending in the repo
# basename — including real user paths like /tmp/<basename> (live bug: `ls
# /tmp/asicode/files` was rewritten to `ls <repo_root>/files`, reading the
# wrong file). A matched path that actually EXISTS on disk is real user data,
# not a training-data bias path, and must be left untouched; only nonexistent
# (virtual) paths are corrected.

def test_real_existing_dir_not_rewritten(bias, tmp_path):
    real_dir = tmp_path / "myproj"
    real_dir.mkdir()
    cmd = f"ls {real_dir}/tests/x.py"
    assert bias(cmd) == cmd


def test_real_existing_dir_with_cd_not_rewritten(bias, tmp_path):
    real_dir = tmp_path / "myproj"
    real_dir.mkdir()
    cmd = f"cd {real_dir} && pwd"
    assert bias(cmd) == cmd


def test_real_existing_workspace_dir_not_rewritten(bias, monkeypatch):
    # /workspace as a REAL existing directory (e.g. inside a container) is a
    # real path — pass 1 must not redirect it into the repo either.
    import os as _os

    _real_exists = _os.path.exists

    def _fake_exists(path):
        return str(path) == "/workspace" or _real_exists(path)

    monkeypatch.setattr(_os.path, "exists", _fake_exists)
    cmd = "cat /workspace/config.yaml"
    assert bias(cmd) == cmd


# ── pass 2: scratch/temp destinations are NEVER rewritten ─────────────────────
# Live bug class (2026-08-05): `mkdir -p /tmp/myproj/out`, `tar -C /tmp/myproj`,
# `rm -rf /tmp/myproj` were rewritten to repo_root — a NOT-YET-CREATED scratch
# destination is indistinguishable from a training-data virtual root by
# exists() alone (the whole point of scratch is that it does not exist yet).
# Paths under machine scratch roots (/tmp, /var/tmp, /var/folders, ...) are
# user-intended destinations, never bias paths — rewriting sends the command
# at the real repository (tar/cp/mv/rsync have no approval gate → silent
# destructive overwrite).

@pytest.mark.parametrize(
    "cmd",
    [
        "mkdir -p /tmp/myproj/out",
        "tar -xzf pkg.tgz -C /tmp/myproj",
        "rm -rf /tmp/myproj",
        "rsync -a src/ /tmp/myproj/",
        "git worktree add /tmp/myproj HEAD",
        "python3 -m venv /tmp/myproj",
        "cp -r build/ /var/folders/myproj/",
        "cat /var/folders/myproj/files/x.txt",
    ],
)
def test_scratch_destinations_not_rewritten(bias, cmd):
    assert bias(cmd) == cmd
