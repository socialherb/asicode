"""Search output must be bounded while it is read, not after.

`grep` ran ripgrep with `capture_output=True`, so the whole output existed in
memory before any of the three budgets (`max_results`, the BM25 pre-cut, the
char cap) could apply. Measured on a 108 MB log with a match on every line:
522 MB of peak RSS to produce 24 KB of content. `glob` next door does it right
and costs nothing, and BackgroundJobManager solves the same problem with a tail
cap — the foreground search path was the outlier.

The exact match count has to survive the bounding. Reporting the DISPLAYED
count as the match count was a real defect once (50 reported for 29,871), and
the model reads that number as "I have seen everything".

The timeout has to survive it too, and did not: streaming the pipe on the
calling thread meant the deadline was only ever consulted between lines, so a
search that produced NOTHING — the exact shape of the slow search the budget
exists to bound — ran unbounded.
"""
from __future__ import annotations

import resource
import shutil
import subprocess
import threading
import time

import pytest


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


@pytest.fixture
def big_match_repo(temp_repo_root):
    """A file large enough that materialising it would be obvious in RSS."""
    line = "needle " + "x" * 100 + "\n"
    with open(f"{temp_repo_root}/big.log", "w", encoding="utf-8") as fh:
        for _ in range(200_000):  # ~21 MB, 200k matches
            fh.write(line)
    return temp_repo_root


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_grep_does_not_materialise_the_whole_match_set(tool_registry, big_match_repo):
    before = _peak_mb()
    result = tool_registry.dispatch("grep", {"pattern": "needle"})
    growth = _peak_mb() - before

    assert result.ok
    # The retained prefix is ~5000 lines of ~110 chars plus the rendered
    # content: single-digit MB. Materialising all 200k matches is ~100 MB.
    assert growth < 40, (
        f"grep grew peak RSS by {growth:.0f} MB for a 24 KB answer — the output "
        "is being materialised before the budget applies"
    )


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_grep_still_reports_the_true_match_count(tool_registry, big_match_repo):
    result = tool_registry.dispatch("grep", {"pattern": "needle", "max_results": 5})
    assert "(200000 matches)" in result.content, (
        f"the count must be the real total, not what survived the cap: "
        f"{result.content[:200]!r}"
    )


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_grep_results_are_unchanged_for_an_ordinary_search(tool_registry, temp_repo_root):
    """Bounding must not alter a normal result."""
    for name, body in (
        ("a.py", "def alpha():\n    return 1\n"),
        ("b.py", "def beta():\n    return alpha()\n"),
    ):
        with open(f"{temp_repo_root}/{name}", "w", encoding="utf-8") as fh:
            fh.write(body)
    result = tool_registry.dispatch("grep", {"pattern": "alpha"})
    assert result.ok
    assert "a.py" in result.content and "b.py" in result.content
    assert "(2 matches)" in result.content


def test_grep_reports_no_matches_distinctly(tool_registry, temp_repo_root):
    result = tool_registry.dispatch("grep", {"pattern": "zzz_no_such_token_zzz"})
    assert result.ok
    assert "no matches" in result.content


class TestSearchDeadline:
    """The budget must bind on wall-clock, not on output arriving.

    Both cases below returned at 5.0 s against a 1.0 s budget before the drain
    moved onto its own thread — the first because iterating the pipe blocks in
    read() until a line shows up, the second because one early line is enough
    to satisfy a between-lines check and then never reach it again.
    """

    @staticmethod
    def _elapsed_timeout(cmd, budget=1.0):
        from external_llm.agent.tool_handlers.read_tools import ReadToolsMixin

        t0 = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            ReadToolsMixin._run_search_bounded(cmd, ".", budget, 100)
        return time.monotonic() - t0

    def test_a_silent_search_still_times_out(self):
        elapsed = self._elapsed_timeout(["sh", "-c", "sleep 5"])
        assert elapsed < 2.5, (
            f"a search that printed nothing ran {elapsed:.2f}s against a 1.0s "
            "budget — the deadline is only checked when output arrives"
        )

    def test_a_search_that_goes_quiet_still_times_out(self):
        elapsed = self._elapsed_timeout(["sh", "-c", "echo hi; sleep 5"])
        assert elapsed < 2.5, (
            f"a search that printed once then stalled ran {elapsed:.2f}s "
            "against a 1.0s budget"
        )

    def test_a_completed_search_is_not_charged_for_a_slow_exit(self):
        """stdout EOF ends the search; exiting gets its own grace.

        Charging teardown the deadline's remainder left 0.1 s, so a process
        that had already delivered every match was killed and reported as
        "grep timed out" — a complete result thrown away.
        """
        from external_llm.agent.tool_handlers.read_tools import ReadToolsMixin

        rc, lines, total, _err = ReadToolsMixin._run_search_bounded(
            # Emits everything, closes stdout, then lingers past the budget.
            ["sh", "-c", "echo hit; exec 1>&-; sleep 0.6"], ".", 0.4, 100,
        )
        assert lines == ["hit"] and total == 1 and rc == 0

    def test_the_process_group_is_gone_after_a_timeout(self):
        """A killed search must not leave the shell's children behind."""
        from external_llm.agent.tool_handlers.read_tools import ReadToolsMixin

        marker = "asi_search_timeout_probe"
        with pytest.raises(subprocess.TimeoutExpired):
            ReadToolsMixin._run_search_bounded(
                ["sh", "-c", f"sleep 30 # {marker}"], ".", 0.5, 100,
            )
        survivors = subprocess.run(
            ["pgrep", "-f", marker], capture_output=True, text=True, timeout=10,
        )
        assert not survivors.stdout.strip(), (
            f"processes survived the search kill: {survivors.stdout!r}"
        )


class TestSearchCancel:
    """ESC has to reach a search, for the same reason it had to reach `bash`.

    A search is the other call that can hold a turn for two minutes, and until
    the poll was added nothing observed a cancel while one ran: the CLI returns
    the prompt ~1s after ESC while ripgrep keeps walking the tree, in its own
    session, owned by nobody.
    """

    @staticmethod
    def _probe(flag):
        return lambda: flag["set"]

    def test_a_running_search_observes_a_cancel(self):
        from external_llm.agent.tool_handlers.read_tools import (
            ReadToolsMixin,
            SearchCancelled,
        )

        flag = {"set": False}
        threading.Timer(0.4, lambda: flag.__setitem__("set", True)).start()
        t0 = time.monotonic()
        with pytest.raises(SearchCancelled):
            ReadToolsMixin._run_search_bounded(
                ["sh", "-c", "sleep 20"], ".", 120, 100,
                cancelled=self._probe(flag),
            )
        elapsed = time.monotonic() - t0
        assert elapsed < 2.5, (
            f"the cancel was not observed until {elapsed:.1f}s — the wait is "
            "not sliced"
        )

    def test_a_cancelled_search_leaves_nothing_behind(self):
        """`start_new_session=True` means nothing reaps it for us."""
        from external_llm.agent.tool_handlers.read_tools import (
            ReadToolsMixin,
            SearchCancelled,
        )

        marker = "asi_search_cancel_probe"
        flag = {"set": False}
        threading.Timer(0.3, lambda: flag.__setitem__("set", True)).start()
        with pytest.raises(SearchCancelled):
            ReadToolsMixin._run_search_bounded(
                ["sh", "-c", f"sleep 20 # {marker}"], ".", 120, 100,
                cancelled=self._probe(flag),
            )
        survivors = subprocess.run(
            ["pgrep", "-f", marker], capture_output=True, text=True, timeout=10,
        )
        assert not survivors.stdout.strip(), (
            f"the search survived its cancel: {survivors.stdout!r}"
        )

    def test_the_tool_reports_a_cancel_as_a_failure(self, tool_registry, monkeypatch):
        """Not as an empty answer — that reads as 'searched, found nothing'."""
        import external_llm.agent.tool_handlers.read_tools as rt

        def _cancel(cmd, cwd, timeout, retain_lines, cancelled=None):
            raise rt.SearchCancelled(cmd)

        monkeypatch.setattr(
            rt.ReadToolsMixin, "_run_search_bounded", staticmethod(_cancel), raising=True,
        )
        result = tool_registry.dispatch("grep", {"pattern": "x"})
        assert not result.ok
        assert result.error == "Operation cancelled"
        assert (result.metadata or {}).get("cancelled") is True

    def test_the_probe_reads_the_live_cancel_event(self, tool_registry, monkeypatch):
        """A captured event goes stale — the design-chat REPL swaps it per turn.

        The probe is handed down once, so the property that matters is that
        calling it re-reads config, not that it was built from the right event.
        """
        import external_llm.agent.tool_handlers.read_tools as rt

        seen: list = []

        def _capture(cmd, cwd, timeout, retain_lines, cancelled=None):
            seen.append(cancelled)
            return 1, [], 0, ""

        monkeypatch.setattr(
            rt.ReadToolsMixin, "_run_search_bounded", staticmethod(_capture), raising=True,
        )
        tool_registry.config.cancel_event = threading.Event()
        tool_registry.dispatch("grep", {"pattern": "x"})
        probe = seen[0]
        assert probe is not None, "the search was run with no cancel probe at all"
        assert probe() is False

        # A NEW event object, as the REPL installs each turn.
        tool_registry.config.cancel_event = threading.Event()
        tool_registry.config.cancel_event.set()
        assert probe() is True, (
            "the probe captured the old event — a per-turn swap would leave it "
            "watching an event nobody will set"
        )


def test_grep_surfaces_a_tool_failure(tool_registry, temp_repo_root, monkeypatch):
    """A non-search failure must still reach the caller with its stderr."""
    import external_llm.agent.tool_handlers.read_tools as rt

    def _boom(cmd, cwd, timeout, retain_lines, cancelled=None):
        return 2, [], 0, "rg: unrecognized flag"

    monkeypatch.setattr(
        rt.ReadToolsMixin, "_run_search_bounded", staticmethod(_boom), raising=True,
    )
    result = tool_registry.dispatch("grep", {"pattern": "x"})
    assert not result.ok
    assert "unrecognized flag" in (result.error or "")
