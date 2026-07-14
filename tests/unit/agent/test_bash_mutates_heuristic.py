"""Regression tests for ToolRegistry._bash_command_mutates_files.

Covers the false-negative / dead-code / over-invalidation bugs found by
running the heuristic against real commands:

1. `find . -delete` / `find . -exec rm {} \\;` were NOT detected as mutating
   (the `find ` read-only prefix matched first), so a successful `find -delete`
   left the read-tool result cache intact and served stale `read_file` results.
2. `git branch -D x` (and other create/rename/delete forms) matched the bare
   `git branch` read-only prefix and was never detected as mutating.
3. `git stash list` never reached its own read-only-prefix entry because the
   `git stash` write-token substring check ran first and always matched —
   permanently dead code that also always force-cleared the cache.
4. Pure-read pipelines (`grep foo | head`) unconditionally invalidated the
   cache just for containing a `|`, even though every segment is read-only.
"""
from __future__ import annotations

from external_llm.agent.tool_registry import ToolRegistry


def _mutates(cmd: str) -> bool:
    return ToolRegistry._bash_command_mutates_files(cmd)


class TestFindDeleteExec:
    def test_find_delete_is_mutating(self):
        assert _mutates("find . -name '*.tmp' -delete") is True

    def test_find_exec_is_mutating(self):
        assert _mutates("find . -name '*.pyc' -exec rm {} \\;") is True

    def test_plain_find_is_readonly(self):
        assert _mutates("find . -name '*.py'") is False


class TestGitBranch:
    def test_git_branch_delete_is_mutating(self):
        assert _mutates("git branch -D feature-x") is True

    def test_git_branch_lowercase_delete_is_mutating(self):
        assert _mutates("git branch -d feature-x") is True

    def test_git_branch_create_is_mutating(self):
        assert _mutates("git branch new-feature") is True

    def test_git_branch_rename_is_mutating(self):
        assert _mutates("git branch -m old new") is True

    def test_bare_git_branch_is_readonly(self):
        assert _mutates("git branch") is False

    def test_git_branch_list_is_readonly(self):
        assert _mutates("git branch --list") is False

    def test_git_branch_all_is_readonly(self):
        assert _mutates("git branch -a") is False


class TestGitStash:
    def test_git_stash_list_is_readonly(self):
        assert _mutates("git stash list") is False

    def test_git_stash_show_is_readonly(self):
        assert _mutates("git stash show") is False

    def test_bare_git_stash_is_mutating(self):
        assert _mutates("git stash") is True

    def test_git_stash_push_is_mutating(self):
        assert _mutates("git stash push -m wip") is True

    def test_git_stash_pop_is_mutating(self):
        assert _mutates("git stash pop") is True

    def test_git_stash_drop_is_mutating(self):
        assert _mutates("git stash drop") is True

    def test_git_stash_list_with_redirect_is_mutating(self):
        """Regression: the git-stash special case must not bypass the
        write-token (redirect) scan just because the command starts with a
        read-only-looking `git stash list`."""
        assert _mutates("git stash list > out.txt") is True

    def test_git_stash_show_with_redirect_is_mutating(self):
        assert _mutates("git stash show -p > backup.patch") is True


class TestGitBranchChained:
    def test_git_branch_list_chained_with_rm_is_mutating(self):
        """Regression: `git branch --list && rm -rf build` must not be
        classified read-only just because the command starts with the
        read-only `git branch --list` query form."""
        assert _mutates("git branch --list && rm -rf build") is True

    def test_git_branch_all_chained_with_touch_is_mutating(self):
        assert _mutates("git branch -a; touch marker") is True


class TestPipelineSegmentation:
    def test_pure_read_pipeline_is_readonly(self):
        assert _mutates("grep -n foo file.py | head -20") is False

    def test_git_log_pipeline_is_readonly(self):
        assert _mutates("git log --oneline | head -5") is False

    def test_bare_argless_segment_is_readonly(self):
        """Regression: a read-only-prefix command with NO trailing arguments
        (e.g. bare `head` at the end of a pipeline) must still match — the
        prefix table's trailing space must not require an argument to exist."""
        assert _mutates("git log --oneline | head") is False

    def test_multi_stage_pure_read_pipeline_is_readonly(self):
        assert _mutates("cat a.txt | grep x | wc -l") is False


class TestCommandSubstitution:
    def test_dollar_paren_substitution_is_conservative(self):
        """Regression: a mutating command hidden inside $(...) must not be
        masked by the outer command looking read-only (e.g. `ls`). Also a
        regression relative to the pre-refactor heuristic, where "git stash"
        being a plain write-token substring happened to catch this by luck."""
        assert _mutates("ls $(git stash pop)") is True

    def test_backtick_substitution_is_conservative(self):
        assert _mutates("cat `git stash pop`") is True

    def test_plain_command_without_substitution_is_unaffected(self):
        assert _mutates("grep foo file.py") is False

    def test_pipeline_with_write_segment_is_mutating(self):
        assert _mutates("cat file.py | tee out.py") is True

    def test_quoted_pipe_is_not_a_separator(self):
        # tree-sitter-bash resolves the pipeline structurally, so a `|` that is
        # part of a quoted string argument is NOT mistaken for a pipeline
        # separator — the genuinely read-only command stays cached. (Pre-refactor
        # this was a conservative bail-out that always invalidated.)
        assert _mutates("grep 'a | b' file.py | head") is False
        assert _mutates('grep "foo|bar" f | head') is False
        assert _mutates('git log --format="%h %s" | head') is False

    def test_readonly_command_substitution_is_readonly(self):
        # A read-only command nested inside $(...) / backticks no longer forces a
        # wholesale bail-out — the inner command is classified too, and when it
        # is read-only the whole command stays cached.
        assert _mutates("echo $(pwd)") is False
        assert _mutates("echo $(whoami)") is False
        assert _mutates("cat $(ls)") is False
        assert _mutates("echo `date`") is True  # `date` is unknown → conservative

    def test_chained_read_commands_are_readonly(self):
        assert _mutates("git status && git diff") is False

    def test_chained_with_mutating_segment_is_mutating(self):
        assert _mutates("git status && rm foo.txt") is True


class TestRedirection:
    """Output redirection always mutates (writes/appends/truncates a file) and
    must invalidate the read-tool cache. Regression: the old fixed substring
    token ``"> "`` only matched when a space followed ``>``, so the no-space
    form escaped detection, matched the ``"echo "`` read-only prefix, and served
    stale cached data — the false negative this classifier's own contract calls
    "worse than a miss"."""

    def test_redirect_no_space_is_mutating(self):
        # The exact regression: `echo hello >out.txt` (no space after >).
        assert _mutates("echo hello >out.txt") is True

    def test_redirect_with_space_is_mutating(self):
        assert _mutates("echo hello > out.txt") is True

    def test_append_redirect_is_mutating(self):
        assert _mutates("echo hello >> out.txt") is True
        assert _mutates("echo hello >>out.txt") is True

    def test_stderr_redirect_is_mutating(self):
        # `2>err` and `2>&1` both carry a bare `>` outside quotes.
        assert _mutates("python3 -c 'print(1)' 2>err") is True
        assert _mutates("foo --bar 2>&1") is True

    def test_redirect_inside_quotes_is_not_detected(self):
        # A `>` that is part of a string literal is NOT a redirection — must
        # not be flagged (would be a harmless cache miss, but the point of the
        # quote-aware scan is to keep read-only grep/echo cached).
        assert _mutates('grep "a>b" file.py') is False
        assert _mutates("echo '> out'") is False

    def test_redirect_with_readonly_prefix_is_mutating(self):
        # Must not be masked by the read-only `echo`/`python3 -c` prefix.
        assert _mutates("echo data > log/results.txt 2>&1") is True

    def test_plain_echo_without_redirect_is_readonly(self):
        assert _mutates("echo hello world") is False


class TestTreeSitterStructuralPath:
    """The tree-sitter-bash structural segment extractor resolves $(...), quoted
    pipes, and list/loop bodies exactly — letting genuinely read-only commands
    that the text heuristic had to bail out on stay cached, without weakening the
    fail-closed contract for genuinely mutating ones."""

    def test_list_with_pipe_all_readonly(self):
        assert _mutates("git status && git diff --stat | head") is False

    def test_loop_body_with_rm_is_mutating(self):
        assert _mutates("for f in *.py; do rm $f; done") is True

    def test_segments_resolve_command_substitution(self):
        segs = ToolRegistry._bash_command_segments_via_ts("ls $(git stash pop)")
        assert segs is not None
        # Both the outer command and the inner `git stash pop` are present.
        assert any("git stash pop" in s for s in segs)
        assert any(s.startswith("ls") for s in segs)

    def test_segments_resolve_quoted_pipe(self):
        segs = ToolRegistry._bash_command_segments_via_ts('grep "a|b" f | head')
        assert segs is not None
        # The quoted `|` must NOT split — exactly two pipeline segments.
        assert len(segs) == 2

    def test_segments_none_on_parse_error(self):
        # A command tree-sitter-bash flags as having an error → None (caller
        # falls back to the conservative text heuristic).
        import external_llm.languages.tree_sitter_utils as _ts
        if not _ts.is_available():
            return  # nothing to test without tree-sitter
        # Construct input that parses with has_error: an unterminated construct.
        segs = ToolRegistry._bash_command_segments_via_ts("echo 'unterminated")
        # Either tree-sitter recovers (segments returned) or flags error (None);
        # both are acceptable — the point is no exception is raised.
        assert segs is None or isinstance(segs, list)


class TestFallbackWhenTreeSitterUnavailable:
    """When tree-sitter-bash is unavailable or yields no segments, the classifier
    must fall back to the conservative text heuristic — preserving the fail-closed
    contract (a stale cache is worse than a miss)."""

    def test_fallback_preserves_conservative_substitution(self, monkeypatch):
        monkeypatch.setattr(
            ToolRegistry, "_bash_command_segments_via_ts", staticmethod(lambda c: None)
        )
        # `$(...)` → conservative invalidate (bail-out path).
        assert _mutates("ls $(git stash pop)") is True

    def test_fallback_preserves_readonly_pipeline(self, monkeypatch):
        monkeypatch.setattr(
            ToolRegistry, "_bash_command_segments_via_ts", staticmethod(lambda c: None)
        )
        # Unquoted pure-read pipeline still recognized read-only via regex split.
        assert _mutates("git log --oneline | head") is False

    def test_fallback_preserves_quoted_pipe_bailout(self, monkeypatch):
        monkeypatch.setattr(
            ToolRegistry, "_bash_command_segments_via_ts", staticmethod(lambda c: None)
        )
        # With the fallback, a quoted pipeline bails out conservatively (the
        # structural path is what resolves it; without it we stay fail-closed).
        assert _mutates("grep 'a | b' f | head") is True


# ── 2>&1 fd-dup vs file redirect (tree-sitter-bash structural) ───────────────

from external_llm.languages import tree_sitter_utils as _ts_utils


def _ts_bash_ok() -> bool:
    try:
        return _ts_utils.is_available() and _ts_utils.get_parser("bash") is not None
    except Exception:
        return False


class TestRedirectFdDupVsFile:
    """``2>&1`` (fd duplication) is a pure in-process stream merge — extremely
    common in read-only commands like ``git log 2>&1 | head``. The raw text
    scanner treats every ``>`` as a redirect and forces a cache miss.
    tree-sitter-bash tags fd-dups and file redirects as the SAME node type
    (``file_redirect``); the node body (``&`` after ``>`` → fd target) is what
    tells them apart. These pin both the structural path and the fallback."""

    def test_fd_dup_unit(self):
        if not _ts_bash_ok():
            import pytest
            pytest.skip("tree-sitter-bash unavailable")
        is_dup = ToolRegistry._redirect_is_fd_dup
        assert is_dup("2>&1") is True
        assert is_dup(">&2") is True
        assert is_dup("1>&2") is True
        assert is_dup("2>&-") is True
        # real file redirects — NOT fd-dups
        assert is_dup("> out.txt") is False
        assert is_dup(">>out.txt") is False
        assert is_dup("2>err.log") is False
        assert is_dup("&>all") is False

    def test_fd_dup_not_treated_as_mutation(self):
        if not _ts_bash_ok():
            import pytest
            pytest.skip("tree-sitter-bash unavailable")
        # 2>&1 merges stderr into stdout — writes no file; must stay read-only.
        assert _mutates("git log 2>&1 | head") is False
        assert _mutates("git status 2>&1") is False

    def test_stderr_to_file_is_a_mutation(self):
        if not _ts_bash_ok():
            import pytest
            pytest.skip("tree-sitter-bash unavailable")
        # 2>err.log truncates/creates a file → mutate (fd-dup helper must NOT
        # misclassify it).
        assert _mutates("echo boom 2>err.log") is True

    def test_stdout_to_file_is_a_mutation(self):
        if not _ts_bash_ok():
            import pytest
            pytest.skip("tree-sitter-bash unavailable")
        assert _mutates("echo hi > out.txt") is True
        assert _mutates("echo hi >> out.txt") is True

    def test_fallback_still_invalidates_fd_dup(self, monkeypatch):
        """When tree-sitter-bash is unavailable, the conservative scanner still
        catches ``2>&1`` (over-invalidation → cache miss, never stale data).
        This pins that the fallback never loosens safety."""
        monkeypatch.setattr(
            ToolRegistry, "_has_file_redirect_via_ts", classmethod(lambda cls, c: None)
        )
        assert _mutates("git log 2>&1 | head") is True
