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

import pytest

from external_llm.agent.tool_registry import ToolRegistry


def _mutates(cmd: str) -> bool:
    return ToolRegistry._bash_command_mutates_files(cmd)


@pytest.fixture(autouse=True)
def _clear_bash_classifier_cache():
    """``_bash_command_mutates_files`` is lru_cached; a classification must not
    leak across tests because several tests monkeypatch a sub-classifier
    (``_bash_command_segments_via_ts`` / ``_has_file_redirect_via_ts``) to pin
    the fallback path — a stale cache entry would bypass the patch."""
    ToolRegistry._bash_command_mutates_files.cache_clear()
    yield
    ToolRegistry._bash_command_mutates_files.cache_clear()


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
        monkeypatch.setattr(ToolRegistry, "_bash_command_segments_via_ts", staticmethod(lambda c, _tree=None: None))
        # `$(...)` → conservative invalidate (bail-out path).
        assert _mutates("ls $(git stash pop)") is True

    def test_fallback_preserves_readonly_pipeline(self, monkeypatch):
        monkeypatch.setattr(ToolRegistry, "_bash_command_segments_via_ts", staticmethod(lambda c, _tree=None: None))
        # Unquoted pure-read pipeline still recognized read-only via regex split.
        assert _mutates("git log --oneline | head") is False

    def test_fallback_preserves_quoted_pipe_bailout(self, monkeypatch):
        monkeypatch.setattr(ToolRegistry, "_bash_command_segments_via_ts", staticmethod(lambda c, _tree=None: None))
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
        monkeypatch.setattr(ToolRegistry, "_has_file_redirect_via_ts", classmethod(lambda cls, c, _tree=None: None))
        assert _mutates("git log 2>&1 | head") is True


class TestArbitraryCodeIsNotReadOnly:
    """`python -c` / `python3 -c` / `node -e` run ARBITRARY code.

    They were on the read-only prefix whitelist, justified in a comment as
    "introspection only when via -c (no pip/install)" — which considers what the
    flag prevents (installing) but not what it permits (writing files). A
    `python3 -c "open('f','w').write(...)"` therefore classified as read-only,
    and this classifier feeds three consumers that must agree on it:

      * read-tool cache invalidation — the agent then re-reads its own
        pre-mutation content for the rest of the 120 s TTL (reproduced
        end-to-end: disk said MUTATED, read_file returned ORIGINAL with
        cache_hit=True);
      * dispatch_parallel's gate — a file-writing command batched in parallel
        with reads of that same file, the exact race the gate exists to stop;
      * DesignChatLoop's read/write phase partition.

    The classifier's own stated policy is that ambiguous commands default to
    mutating "since a stale cache is worse than a miss". Arbitrary code is the
    most ambiguous shape there is, so it now falls to that default — including
    for harmless-looking bodies, since deciding otherwise would mean parsing a
    second language to guess intent.
    """

    def test_python_c_write_is_mutating(self):
        assert _mutates("python -c \"open('a.py','w').write('x')\"") is True

    def test_python3_c_write_is_mutating(self):
        assert _mutates("python3 -c \"import pathlib;pathlib.Path('a').write_text('x')\"") is True

    def test_node_e_write_is_mutating(self):
        assert _mutates("node -e \"require('fs').writeFileSync('a','x')\"") is True

    def test_python_c_rmtree_is_mutating(self):
        assert _mutates("python -c \"import shutil;shutil.rmtree('d')\"") is True

    def test_harmless_looking_body_is_still_mutating(self):
        """No body inspection: we do not parse Python/JS to guess intent."""
        assert _mutates("python -c 'print(1)'") is True
        assert _mutates("node -e 'console.log(1)'") is True

    def test_node_check_stays_readonly(self):
        """`node --check` only parses — it must keep its cache benefit."""
        assert _mutates("node --check a.js") is False


class TestEnvPrefix:
    """`env` printed the environment, but `env <cmd>` RUNS <cmd>.

    The whitelist entry was the bare string "env" (no trailing space, unlike
    every other entry), so a plain prefix match whitelisted any command it
    wrapped — the same hole as `python -c`, reachable as `env python -c ...`.
    """

    def test_bare_env_is_readonly(self):
        assert _mutates("env") is False

    def test_env_with_assignment_only_is_readonly(self):
        assert _mutates("env FOO=1") is False
        assert _mutates("env FOO=1 BAR=2") is False

    def test_env_running_a_command_is_mutating(self):
        assert _mutates("env python -c \"open('a','w').write('x')\"") is True
        assert _mutates("env rm -rf d") is True

    def test_env_prefixed_other_binary_is_mutating(self):
        """'env' without a trailing space also matched envsubst, envdir, ..."""
        assert _mutates("envsubst < template") is True


class TestConservativeDefaultStillHolds:
    """Spot-check that the fix did not weaken or over-broaden the classifier."""

    def test_known_mutators(self):
        for cmd in (
            "sed -i 's/a/b/' f",
            "find . -delete",
            "sort -o out in",
            "dd if=/dev/zero of=f",
            "truncate -s 0 f",
            "install -m 644 a b",
            "git stash pop",
            "echo hi > f",
            "echo hi | tee f",
        ):
            assert _mutates(cmd) is True, cmd

    def test_known_readonly_still_cacheable(self):
        for cmd in (
            "cat a.py",
            "ls -la",
            "git status",
            "grep -rn x .",
            "git log --oneline",
            "pwd",
            "whoami",
            "printenv",
            "head -20 f",
            "wc -l f",
        ):
            assert _mutates(cmd) is False, cmd


class TestDevNullAndFdSinks:
    """`2>/dev/null` is not a file write.

    It was classified as one, so an ordinary read command took the full
    mutating path: the tool-result cache plus every per-root cache (walk, file
    index, non-Python symbols, prefilter memo, git snapshot) was dropped, and
    the whole batch fell out of parallel execution. Measured cost of the
    needless rebuild: ~47 ms for the walk / file-index / git-snapshot trio
    alone, on top of losing every cached read_file result.

    The direction of a miss here is safe (a false "mutating" only wastes work),
    which is exactly why nothing surfaced it — hence explicit cases.
    """

    def test_stderr_discard_is_readonly(self):
        for cmd in (
            "cat a.py 2>/dev/null",
            "git status 2>/dev/null",
            "grep -r x . 2>/dev/null | head",
            "find . -name '*.py' 2>/dev/null",
        ):
            assert _mutates(cmd) is False, cmd

    def test_stdout_discard_forms_are_readonly(self):
        # `>/dev/null 2>&1` is the single most common shape of all.
        for cmd in ("ls >/dev/null 2>&1", "ls > /dev/null", "ls &>/dev/null"):
            assert _mutates(cmd) is False, cmd

    def test_append_sink_forms_are_readonly(self):
        """`find` returns the FIRST `>`, so `>>` left a stray `>` on the target
        and no sink matched — the append forms stayed classified as writes."""
        for cmd in ("ls >>/dev/null", "cat a 2>>/dev/null", "ls &>>/dev/null", "ls >> /dev/null"):
            assert _mutates(cmd) is False, cmd

    def test_append_to_real_file_still_mutating(self):
        for cmd in ("ls >>out.txt", "cat a 2>>err.log", "ls &>>/dev/nullx"):
            assert _mutates(cmd) is True, cmd

    def test_dev_std_streams_and_fd_paths_are_readonly(self):
        for cmd in ("cat a >/dev/stdout", "cat a >/dev/stderr", "cat a 2>/dev/fd/3"):
            assert _mutates(cmd) is False, cmd

    def test_real_file_redirect_still_mutating(self):
        """The sink allowance must not swallow a genuine write in the same
        command — in either order, and for look-alike paths."""
        for cmd in (
            "echo x > /tmp/real",
            "cat a >out.txt 2>/dev/null",
            "cat a 2>/dev/null >out.txt",
            "ls > /dev/nullx",
            "ls > /dev/null/foo",
        ):
            assert _mutates(cmd) is True, cmd


class TestParseBashTreeSharedBootstrap:
    """Both structural bash classifiers share one tree-sitter bootstrap
    (_parse_bash_tree). Pin the shared contract: unavailable or parse failure →
    None (caller falls back conservatively), and both classifiers delegate to
    the same helper."""

    def test_returns_none_when_ts_unavailable(self, monkeypatch):
        import external_llm.languages.tree_sitter_utils as _ts

        monkeypatch.setattr(_ts, "is_available", lambda: False)
        assert ToolRegistry._parse_bash_tree("ls") is None

    def test_parse_failure_swallowed_and_both_delegate(self, monkeypatch):
        import types

        import external_llm.languages.tree_sitter_utils as _ts

        calls: list = []

        def _fake_parse(_cmd):
            calls.append(_cmd)
            raise RuntimeError("bootstrap failure must be swallowed by helper")

        monkeypatch.setattr(_ts, "is_available", lambda: True)
        monkeypatch.setattr(_ts, "get_parser", lambda _lang: types.SimpleNamespace(parse=_fake_parse))
        assert ToolRegistry._has_file_redirect_via_ts("ls") is None
        assert ToolRegistry._bash_command_segments_via_ts("ls") is None
        assert len(calls) == 2


# ── single-parse / shared-tree / classification cache (P1) ───────────────────


class TestSingleParseSharedTree:
    """P1: ``_bash_command_mutates_files`` used to parse the same command TWICE
    per classification (once for the redirect scan, once for segment splitting).
    Both structural classifiers now share ONE tree parsed from the ORIGINAL
    command. These pin the single-parse contract and the N1 offset pitfall."""

    def test_classification_parses_command_once(self, monkeypatch):
        calls: list = []
        real = ToolRegistry._parse_bash_tree.__func__

        def counting(cls, command):
            calls.append(command)
            return real(cls, command)

        monkeypatch.setattr(ToolRegistry, "_parse_bash_tree", classmethod(counting))
        # Read-only pipeline exercises BOTH structural classifiers (redirect
        # scan + segment split) — must still parse exactly once.
        assert _mutates("git log --oneline | head -5") is False
        assert len(calls) == 1

    def test_shared_tree_slices_original_command_offsets(self):
        # N1 pitfall: the shared tree is parsed from the ORIGINAL command (not
        # the stripped text); node byte offsets slice into that same string.
        # Leading whitespace shifts every offset — a stripped-based tree would
        # mis-slice the redirect body and the segment texts.
        assert _mutates("  echo hi > out.txt") is True
        assert _mutates("\n  git status --short") is False
        segs = ToolRegistry._bash_command_segments_via_ts("  ls $(git stash pop) | head")
        assert segs is not None
        assert any(s.startswith("ls") for s in segs)

    def test_repeated_command_served_from_classification_cache(self):
        # Repeated commands (git status / ls between edits) must not be
        # re-parsed — second classification comes from the lru_cache.
        assert _mutates("git status --short") is False
        assert _mutates("git status --short") is False
        info = ToolRegistry._bash_command_mutates_files.cache_info()
        assert info.hits >= 1
