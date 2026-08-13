"""grep: exit code 2 conflates a bad pattern with an unreadable file.

Both ripgrep and grep return 2 for "an error occurred". The handler read that
as "the regex did not compile", which is only one of the two things it means —
and the other one, a single file the process cannot open, is an ordinary state
for any repo with root-owned files (Docker bind mounts) or restricted dirs.

Measured against the live handler before the fix: one ``chmod 000`` file beside
a matching one turned the whole call into
``ok=False, "grep failed (exit=2): rg: ./locked.py: Permission denied"`` and the
real match in the readable file was never reported.
"""
import os
import pathlib
import shutil
import stat

import pytest

from external_llm.agent.tool_handlers.read_tools import (
    _search_ran_despite_errors,
    _unsupported_flag,
)


@pytest.fixture
def locked_repo(temp_repo_root):
    """A repo with one matching readable file and one unreadable file."""
    root = pathlib.Path(temp_repo_root)
    (root / "hit.py").write_text("def connect_database():\n    pass\n")
    bad = root / "locked.py"
    bad.write_text("def connect_database():\n    pass\n")
    os.chmod(bad, 0o000)
    yield root
    os.chmod(bad, stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a 0o000 file")
class TestUnreadableFileDoesNotFailTheSearch:
    def test_real_matches_survive_an_unreadable_neighbour(self, tool_registry, locked_repo):
        """The demonstrated data loss: exit 2 discarded a match it had printed."""
        result = tool_registry.dispatch("grep", {"pattern": r"connect_data\w+"})

        assert result.ok, f"grep failed despite real matches: {result.error}"
        assert "hit.py" in result.content
        assert result.metadata.get("files_skipped") is True

    def test_the_skip_is_reported_not_hidden(self, tool_registry, locked_repo):
        """Silently dropping files would make the answer look complete."""
        result = tool_registry.dispatch("grep", {"pattern": r"connect_data\w+"})

        assert "skipped" in result.content.lower()

    def test_no_matches_plus_an_unreadable_file_is_still_no_matches(
        self, tool_registry, locked_repo,
    ):
        """Exit 2 with zero output is not automatically a failed call.

        Reporting "grep failed" here costs a turn: the honest answer is that the
        pattern is absent from everything that could be read.
        """
        result = tool_registry.dispatch("grep", {"pattern": r"zzz_absent\w+"})

        assert result.ok, f"grep failed on a legitimate miss: {result.error}"
        assert "no matches" in result.content.lower()
        assert result.metadata.get("files_skipped") is True

    def test_the_regex_is_not_silently_downgraded_to_a_literal(
        self, tool_registry, locked_repo,
    ):
        r"""The old retry answered a regex query with a fixed-string search.

        ``connect_data\w+`` matches nothing as a literal, so a result that
        still finds the definition proves the pattern was run AS a regex.
        """
        result = tool_registry.dispatch("grep", {"pattern": r"connect_data\w+"})

        assert result.ok
        assert "connect_database" in result.content


def test_a_genuine_pattern_error_still_falls_back_to_a_literal(
    tool_registry, temp_repo_root,
):
    """The behaviour the exit-2 branch existed for must survive the fix."""
    pathlib.Path(temp_repo_root, "brackets.py").write_text("x = foo(1)\n")

    result = tool_registry.dispatch("grep", {"pattern": "foo("})

    assert result.ok, f"unclosed group was not retried as a literal: {result.error}"
    assert "brackets.py" in result.content


class TestStderrClassification:
    """The discriminator itself, away from the filesystem."""

    def test_access_noise_means_the_search_ran(self):
        assert _search_ran_despite_errors(
            "rg: ./locked.py: Permission denied (os error 13)"
        )

    def test_a_pattern_error_means_it_did_not(self):
        assert not _search_ran_despite_errors(
            "rg: regex parse error:\n    (?:foo()\nerror: unclosed group"
        )

    def test_an_unknown_error_is_not_assumed_benign(self):
        assert not _search_ran_despite_errors("rg: something nobody predicted")

    def test_empty_stderr_is_not_access_noise(self):
        assert not _search_ran_despite_errors("")

    def test_a_pattern_error_wins_over_a_stray_access_word(self):
        """Truncated stderr must not be read as 'the search ran'."""
        assert not _search_ran_despite_errors(
            "rg: regex parse error: permission denied"
        )


class TestUnsupportedFlagDetection:
    def test_it_names_the_rejected_flag(self):
        assert _unsupported_flag(
            "rg: unrecognized flag --max-columns-preview"
        ) == "--max-columns-preview"

    def test_a_normal_error_is_not_a_flag_problem(self):
        assert _unsupported_flag("rg: ./x: Permission denied") is None

    def test_the_gnu_wording_is_recognised_too(self):
        assert _unsupported_flag("grep: unknown flag --max-columns") == "--max-columns"


@pytest.mark.skipif(shutil.which("rg") is None, reason="needs a real rg to wrap")
def test_an_rg_too_old_for_max_columns_preview_still_searches(
    tool_registry, temp_repo_root, monkeypatch, tmp_path,
):
    """``--max-columns-preview`` is rg >= 12.0; Ubuntu 20.04 ships 11.0.2.

    Without the drop-and-retry the flag turns every grep call into exit 2 on
    those hosts, and ``use_rg`` never flips, so the system-grep fallback sitting
    right there is never reached — the tool is simply dead.
    """
    real_rg = shutil.which("rg")
    shim = tmp_path / "rg"
    shim.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        '  if [ "$a" = "--max-columns-preview" ]; then\n'
        '    echo "rg: unrecognized flag --max-columns-preview" >&2\n'
        "    exit 2\n"
        "  fi\n"
        "done\n"
        f'exec {real_rg} "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(shim) if name == "rg" else None)

    pathlib.Path(temp_repo_root, "hit.py").write_text("def connect_database():\n    pass\n")
    result = tool_registry.dispatch("grep", {"pattern": r"connect_data\w+"})

    assert result.ok, f"old rg killed the grep tool: {result.error}"
    assert "hit.py" in result.content
