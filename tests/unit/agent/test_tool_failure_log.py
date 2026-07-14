"""Tests for write-tool failure JSONL logging (tool_failure_log module).

Covers:
  * failures are recorded with the right schema
  * success calls are a no-op
  * non-write tools are ignored
  * large payloads (patch/content/...) are redacted in args_summary
  * exception during dispatch (tr=None) path still records
  * the wrapper ``record_write_tool_failure_from_tr`` extracts ToolResult fields

All tests redirect the log to a temp file via ``ASICODE_WRITE_TOOL_FAILURE_LOG``
so the real ``~/.asicode/learning/`` log is never touched.
"""

import json
import os

from external_llm.agent.tool_failure_log import (
    _classify_from_error,
    record_write_tool_failure,
    record_write_tool_failure_from_tr,
    summarize_log,
)
from external_llm.agent.tool_registry import ToolResult


class _Recorder:
    """Tiny helper: read the redirected JSONL log as a list of dicts."""

    def __init__(self, tmp_path):
        self.path = str(tmp_path / "write_tool_failures.jsonl")
        os.environ["ASICODE_WRITE_TOOL_FAILURE_LOG"] = self.path

    def records(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def cleanup(self):
        os.environ.pop("ASICODE_WRITE_TOOL_FAILURE_LOG", None)


def test_failure_recorded_with_full_schema(tmp_path, monkeypatch):
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)  # _git_sha runs git in cwd
    try:
        record_write_tool_failure(
            tool="anchor_edit",
            ok=False,
            error="anchor_pattern 'def foo' matched 3 times; specify occurrence",
            metadata={"failure_class": "anchor_not_unique", "file_path": "src/app.py", "match_count": 3},
            args={"file_path": "src/app.py", "anchor_pattern": "def foo", "edit_mode": "insert_after"},
            model="claude-test",
            repo_root=str(tmp_path),
        )
        rows = rec.records()
        assert len(rows) == 1
        r = rows[0]
        assert r["tool"] == "anchor_edit"
        assert r["failure_class"] == "anchor_not_unique"
        assert r["ok"] is False
        assert r["partial"] is False
        assert r["file_path"] == "src/app.py"
        assert r["error"].startswith("anchor_pattern 'def foo' matched 3 times")
        assert r["model"] == "claude-test"
        assert r["match_count"] == 3
        assert "timestamp" in r and "timestamp_iso" in r
        assert "git_sha" in r
        # args_summary keeps the shape but not large payloads.
        assert r["args_summary"]["anchor_pattern"] == "def foo"
        assert r["args_summary"]["edit_mode"] == "insert_after"
    finally:
        rec.cleanup()


def test_success_is_noop(tmp_path, monkeypatch):
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        record_write_tool_failure(
            tool="edit_text",
            ok=True,
            error=None,
            metadata={},
            args={"file_path": "a.py"},
        )
        assert rec.records() == []
    finally:
        rec.cleanup()


def test_non_write_tool_ignored(tmp_path, monkeypatch):
    """read_tools / run_tests / etc. must not be logged even on failure."""
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        record_write_tool_failure(
            tool="run_tests",
            ok=False,
            error="exit 1",
            metadata={},
            args={},
        )
        assert rec.records() == []
    finally:
        rec.cleanup()


def test_large_payload_redacted_in_args_summary(tmp_path, monkeypatch):
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        big_patch = "@@ \n" + "x" * 5000
        record_write_tool_failure(
            tool="apply_patch",
            ok=False,
            error="hunk failed",
            metadata={"failure_class": "patch_apply_failed"},
            args={"patch": big_patch, "path": "src/big.py"},
        )
        rows = rec.records()
        assert len(rows) == 1
        summary = rows[0]["args_summary"]
        # patch is replaced by a size hint, NOT the raw content.
        assert summary["patch"] == f"<{len(big_patch)} chars>"
        assert summary["path"] == "src/big.py"
    finally:
        rec.cleanup()


def test_unclassified_when_no_failure_class(tmp_path, monkeypatch):
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        record_write_tool_failure(
            tool="edit_text",
            ok=False,
            error="boom",
            metadata={},
            args={"file_path": "x.py"},
        )
        rows = rec.records()
        assert rows[0]["failure_class"] == "unclassified"
    finally:
        rec.cleanup()


def test_wrapper_extracts_toolresult_fields(tmp_path, monkeypatch):
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        tr = ToolResult(
            ok=False,
            content="",
            error="symbol not found",
            metadata={"failure_class": "symbol_not_found", "file_path": "m.py"},
        )
        record_write_tool_failure_from_tr(
            tool="modify_symbol", tr=tr, args={"file_path": "m.py", "symbol": "foo"},
        )
        rows = rec.records()
        assert len(rows) == 1
        assert rows[0]["tool"] == "modify_symbol"
        assert rows[0]["failure_class"] == "symbol_not_found"
        assert rows[0]["error"] == "symbol not found"
    finally:
        rec.cleanup()


def test_wrapper_success_noop(tmp_path, monkeypatch):
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        tr = ToolResult(ok=True, content="done", metadata={})
        record_write_tool_failure_from_tr(tool="edit_text", tr=tr, args={})
        assert rec.records() == []
    finally:
        rec.cleanup()


def test_error_truncated_to_limit(tmp_path, monkeypatch):
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        huge = "E" * 5000
        record_write_tool_failure(
            tool="edit_text", ok=False, error=huge, metadata={}, args={},
        )
        rows = rec.records()
        assert len(rows[0]["error"]) <= 1300  # _MAX_ERROR_CHARS + ellipsis
    finally:
        rec.cleanup()


def test_partial_failure_recorded_even_when_ok(tmp_path, monkeypatch):
    """partial_failure=True should be logged regardless of ok flag."""
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        record_write_tool_failure(
            tool="apply_patch", ok=True, error=None, metadata={"failure_class": "partial"},
            args={}, partial_failure=True,
        )
        rows = rec.records()
        assert len(rows) == 1
        assert rows[0]["partial"] is True
        assert rows[0]["ok"] is True
    finally:
        rec.cleanup()


def test_summarize_log_aggregates_by_tool_and_failure_class(tmp_path, monkeypatch):
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        # 3 anchor_edit failures (2 not_unique, 1 miss) + 1 apply_patch failure.
        for _ in range(2):
            record_write_tool_failure(
                tool="anchor_edit", ok=False, error="e",
                metadata={"failure_class": "anchor_not_unique"}, args={"file_path": "a.py"},
            )
        record_write_tool_failure(
            tool="anchor_edit", ok=False, error="e",
            metadata={"failure_class": "anchor_miss"}, args={"file_path": "b.py"},
        )
        record_write_tool_failure(
            tool="apply_patch", ok=False, error="e",
            metadata={"failure_class": "patch_apply_failed"}, args={"path": "c.py"},
        )
        # success should NOT appear in summary.
        record_write_tool_failure(
            tool="edit_text", ok=True, error=None, metadata={}, args={},
        )
        s = summarize_log(rec.path)
        assert s["total"] == 4
        assert s["by_tool"]["anchor_edit"] == 3
        assert s["by_tool"]["apply_patch"] == 1
        assert "edit_text" not in s["by_tool"]  # success was no-op
        assert s["by_failure_class"]["anchor_not_unique"] == 2
        assert s["by_failure_class"]["anchor_miss"] == 1
        assert s["by_failure_class"]["patch_apply_failed"] == 1
        assert len(s["recent"]) == 4
    finally:
        rec.cleanup()


def test_summarize_log_missing_file(tmp_path):
    """summarize_log on a non-existent path returns empty, never raises."""
    s = summarize_log(str(tmp_path / "does_not_exist.jsonl"))
    assert s["total"] == 0
    assert s["by_tool"] == {}
    assert s["recent"] == []


# -- _summarize_args edge cases ----------------------------------------------
# These branches are exercised only indirectly via record_write_tool_failure
# above. Testing the helper directly pins the redaction + truncation contract
# without going through the JSONL write path.

from external_llm.agent.tool_failure_log import (
    _ARG_KEYS_TO_DROP,
    _MAX_ARGS_SUMMARY_CHARS,
    _MAX_ERROR_CHARS,
    _git_sha,
    _summarize_args,
    _truncate,
)


def test_summarize_args_redacts_all_payload_keys():
    """Every key in _ARG_KEYS_TO_DROP must be replaced by a size hint."""
    payload = "payload-value"
    expected = f"<{len(payload)} chars>"
    args = {k: payload for k in _ARG_KEYS_TO_DROP}
    summary = _summarize_args(args)
    for key in _ARG_KEYS_TO_DROP:
        assert summary[key] == expected, f"{key} not redacted: {summary[key]}"


def test_summarize_args_truncates_long_string():
    """Strings longer than 120 chars are truncated with an ellipsis."""
    long_val = "x" * 500
    summary = _summarize_args({"symbol": long_val})
    assert summary["symbol"].endswith("…")
    assert len(summary["symbol"]) < len(long_val)
    # The first 120 chars are preserved.
    assert summary["symbol"].startswith("x" * 120)


def test_summarize_args_keeps_short_string_verbatim():
    assert _summarize_args({"anchor_pattern": "def foo"})["anchor_pattern"] == "def foo"


def test_summarize_args_renders_list_as_json():
    """Non-scalar values (list/dict) are JSON-rendered, kept under 200 chars.
    Use a key NOT in the drop set so it goes through the render branch."""
    summary = _summarize_args({"ops": ["a", "b", "c"]})
    assert summary["ops"] == '["a", "b", "c"]'


def test_summarize_args_renders_bool_and_int_verbatim():
    summary = _summarize_args({"occurrence": 2, "dry_run": True, "val": None})
    assert summary["occurrence"] == 2
    assert summary["dry_run"] is True
    assert summary["val"] is None


def test_summarize_args_non_dict_returns_empty():
    """A non-dict args value must not raise — returns {}."""
    assert _summarize_args(None) == {}
    assert _summarize_args("not a dict") == {}


def test_summarize_args_overall_size_collapse():
    """When the WHOLE summary exceeds _MAX_ARGS_SUMMARY_CHARS it collapses to a
    keys-only dict so the log line stays bounded."""
    # Build args whose individual values are short but whose aggregate exceeds
    # the bound — forces the top-level truncation branch.
    args = {f"k{i:03d}": "short" for i in range(_MAX_ARGS_SUMMARY_CHARS // 10 + 1)}
    summary = _summarize_args(args)
    assert summary.get("_truncated") is True
    assert isinstance(summary["keys"], list)


# -- _truncate -----------------------------------------------------------------


def test_truncate_keeps_short_text():
    assert _truncate("short error") == "short error"


def test_truncate_appends_ellipsis_on_overflow():
    long_err = "e" * (_MAX_ERROR_CHARS + 100)
    out = _truncate(long_err)
    assert len(out) == _MAX_ERROR_CHARS + 1  # limit chars + '…'
    assert out.endswith("…")


def test_truncate_none_returns_empty():
    assert _truncate(None) == ""


# -- _git_sha fallback ---------------------------------------------------------


def test_git_sha_returns_unknown_on_subprocess_failure(monkeypatch, tmp_path):
    """When git is unavailable or the repo has no HEAD, _git_sha must return
    'unknown' rather than raising."""
    import subprocess


    def _boom(*a, **k):
        raise FileNotFoundError("git binary not on PATH")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.chdir(tmp_path)  # a dir with no .git
    assert _git_sha(str(tmp_path)) == "unknown"


# -- _classify_from_error fallback classifier ----------------------------------
# When a handler omits ``metadata["failure_class"]``, the logger derives a class
# from the error text so records don't bucket as "unclassified".


def test_classify_handles_none_error():
    assert _classify_from_error("edit_text", None) == "unclassified"


def test_classify_handles_empty_error():
    assert _classify_from_error("edit_text", "") == "unclassified"


def test_classify_old_string_not_found():
    assert _classify_from_error(
        "edit_text", "old_string not found in src/foo.py"
    ) == "search_string_mismatch"


def test_classify_old_string_not_unique():
    assert _classify_from_error(
        "edit_text", "Found 3 occurrences of old_string in x.py"
    ) == "search_string_mismatch"


def test_classify_closest_match_hint():
    assert _classify_from_error(
        "edit_text", "... Closest match (~77% similar) near line 55:"
    ) == "search_string_mismatch"


def test_classify_edit_text_syntax_error():
    assert _classify_from_error(
        "edit_text",
        "edit_text refused (file NOT modified): the replacement would "
        "introduce a Python syntax error in foo.py",
    ) == "syntax_invalid_after_edit"


def test_classify_anchor_edit_syntax_error():
    assert _classify_from_error(
        "anchor_edit",
        "anchor_edit introduced syntax error (file unchanged)",
    ) == "syntax_invalid_after_edit"


def test_classify_write_plan_syntax_error():
    assert _classify_from_error(
        "write_plan",
        "Plan introduced syntax errors (rolled back): test_foo.py: invalid syntax",
    ) == "syntax_invalid_after_edit"


def test_classify_apply_patch_3way_merge_blob_missing():
    assert _classify_from_error(
        "apply_patch",
        "Patch application failed and repair attempts exhausted: error: "
        "repository lacks the necessary blob to perform 3-way merge.",
    ) == "patch_apply_failed"


def test_classify_apply_patch_does_not_apply():
    assert _classify_from_error(
        "apply_patch", "error: patch failed: foo.py:42"
    ) == "patch_apply_failed"


def test_classify_anchor_multiline_pattern():
    assert _classify_from_error(
        "anchor_edit",
        "anchor_pattern contains 2 lines (embedded '\\n').",
    ) == "anchor_multiline_pattern"


def test_classify_anchor_not_unique():
    assert _classify_from_error(
        "anchor_edit", "anchor_not_unique: matched 3 times"
    ) == "anchor_not_unique"


def test_classify_missing_required_arg():
    assert _classify_from_error(
        "modify_symbol", "'code' is required"
    ) == "invalid_args"


def test_classify_unknown_error_stays_unclassified():
    assert _classify_from_error(
        "edit_text", "something completely unexpected"
    ) == "unclassified"


# ── Regression: modify_symbol resolution failures. The handler wraps every
#    symbol_modify_tool error as "modify_symbol failed for {path}@{symbol}:
#    {detail}" (write_tools.py:_tool_modify_symbol), and symbol_modify_tool
#    itself emits "All strategies failed - could not locate or replace symbol"
#    for direct callers. Neither had a classifier pattern → unclassified in the
#    production log. The "modify_failed" enum member already existed. ─────────


def test_classify_modify_symbol_failed_wrapper():
    """The handler's generic wrapper (catches symbol not found, syntax-blocked
    text splice, write-after-replace failure, etc.)."""
    assert _classify_from_error(
        "modify_symbol",
        "modify_symbol failed for src/foo.py@bar: All strategies failed - "
        "could not locate or replace symbol",
    ) == "modify_failed"


def test_classify_modify_symbol_syntax_blocked_detail():
    """The inner detail text for a re-indentation/splice that would break
    Python syntax is still a modify_failed tool-limitation, NOT a
    syntax_invalid_after_edit (the LLM's code was not applied at all)."""
    assert _classify_from_error(
        "modify_symbol",
        "modify_symbol failed for src/foo.py@bar: modify_symbol could not "
        "produce syntactically valid code for 'bar' (re-indentation/splice "
        "would break Python syntax). Use apply_patch instead.",
    ) == "modify_failed"


def test_classify_all_strategies_failed_unwrapped():
    """Direct callers of symbol_modify_tool (not via the write_tools handler)
    see the raw 'All strategies failed' string without the wrapper."""
    assert _classify_from_error(
        "modify_symbol",
        "All strategies failed - could not locate or replace symbol",
    ) == "modify_failed"


def test_classify_modify_symbol_missing_arg_still_invalid_args():
    """arg-validation errors ('is required') are emitted unwrapped by the
    handler (write_tools.py:4245-4249), so they must still classify as
    invalid_args — NOT be stolen by the 'modify_symbol failed for' pattern,
    which only matches the wrapped outcome path."""
    assert _classify_from_error(
        "modify_symbol", "'code' is required"
    ) == "invalid_args"


# ── Regression: FailureClass enum must include MODIFY_FAILED so the emitted
#    string literal is a recognized member. ────────────────────────────────────
def test_failure_class_enum_has_modify_failed_member():
    from external_llm.agent.operation_models import FailureClass

    values = [m.value for m in FailureClass]
    assert "modify_failed" in values


# ── Regression: batch edit_text occurrence errors must NOT be stolen by the
#    generic "edit_text refused" syntax wrapper (search_string_mismatch block
#    was moved BEFORE the syntax block to fix this). The real failure is a
#    search mismatch (2 occurrences), even though the message also contains
#    "edit_text refused". Found in production write_tool_failures.jsonl. ──────
def test_classify_batch_edit_text_occurrences_not_stolen_by_syntax_wrapper():
    # Verbatim wrapper from _tool_edit_text batch path: the match step runs and
    # fails with 2 occurrences BEFORE any syntax check, but the message carries
    # the generic "edit_text refused" token that the syntax patterns also match.
    assert _classify_from_error(
        "edit_text",
        "edit_text refused (file NOT modified): edit #1 (edits[0]) failed to "
        "match — no edits were applied (atomic batch).\n"
        "Found 2 occurrences of old_string in asi.py. Make old_string more "
        "unique (include 2-3 lines of surrounding context).",
    ) == "search_string_mismatch"


def test_classify_edit_text_refused_pure_syntax_still_classes_as_syntax():
    # Sanity: a pure edit_text-refused syntax error (no occurrence/match tokens)
    # must still classify as syntax, proving the reorder did not break syntax
    # classification — it only stopped the syntax wrapper from stealing match
    # failures that happen to carry "edit_text refused".
    assert _classify_from_error(
        "edit_text",
        "edit_text refused (file NOT modified): the replacement would "
        "introduce a Python syntax error in foo.py: invalid syntax at line 12.",
    ) == "syntax_invalid_after_edit"


# ── Regression: multiline anchor mismatches (pattern line N ≠ file line) were
#    a genuine unclassified gap — the "multiline_mismatch" failure_class is
#    emitted by anchor_shared.resolve_multiline_anchor but had no classifier
#    pattern. Found in production write_tool_failures.jsonl. ─────────────────
def test_classify_multiline_anchor_later_line_mismatch():
    assert _classify_from_error(
        "anchor_edit",
        'multiline anchor: pattern line 2 "ok": True, does not match file '
        "line 3242. The block starting at line 3241 does not fully match the "
        "pattern. Read the file and provide the exact block.",
    ) == "multiline_mismatch"


def test_classify_multiline_anchor_extends_past_eof():
    assert _classify_from_error(
        "anchor_edit",
        "multiline anchor: pattern has 3 lines but the file ends at line 100 "
        "(pattern line 3 extends past end of file). Re-read the file and "
        "provide the exact block.",
    ) == "multiline_mismatch"


def test_classify_multiline_anchor_distinguished_from_anchor_miss():
    # The multiline "first line not found" branch emits anchor_miss, NOT
    # multiline_mismatch — multiline_mismatch only fires when the first line
    # matched but a follow-on line did not. This guards against over-broad
    # "multiline anchor" matching stealing the anchor_miss case.
    assert _classify_from_error(
        "anchor_edit",
        "multiline anchor: first line 'def foo' not found in file (searched "
        "194 lines).",
    ) == "anchor_miss"


# ── Regression: FailureClass enum must include MULTILINE_MISMATCH so the
#    emitted string literal from anchor_shared is a recognized member. ────────
def test_failure_class_enum_has_multiline_mismatch_member():
    from external_llm.agent.operation_models import FailureClass

    values = [m.value for m in FailureClass]
    assert "multiline_mismatch" in values


# ── Regression: dead-code patterns now fire correctly ────────────────────────
# Previously the schema conflated "only_tool" with a co-substring, making the
# tool-scoped anchor entries unreachable. These now-classifiable errors were
# saved as unclassified in the production log.


def test_classify_anchor_edit_pattern_not_found_is_anchor_miss():
    """anchor_edit(delete): 'pattern ... not found in' must be anchor_miss,
    not search_string_mismatch (which is edit_text's failure mode)."""
    assert _classify_from_error(
        "anchor_edit",
        "anchor_edit(delete): pattern 'def get_foo' not found in src/foo.py "
        "(searched 758 lines)",
    ) == "anchor_miss"


def test_classify_pattern_not_found_requires_pattern_token():
    """Without the 'pattern' token, 'not found in' should NOT become
    anchor_miss — it stays unclassified (or another class). This guards the
    co_substring disambiguation against false positives."""
    # edit_text's error is already caught by search_string_mismatch above.
    assert _classify_from_error(
        "edit_text", "old_string not found in foo.py"
    ) == "search_string_mismatch"
    # A bare "not found in" with no 'pattern' and no 'old_string' → unclassified.
    assert _classify_from_error(
        "edit_text", "the thing was not found in that place"
    ) == "unclassified"


def test_classify_empty_diff_is_no_diff_generated():
    """apply_patch salvage producing an empty diff is a no-op edit, not a
    patch failure."""
    assert _classify_from_error(
        "apply_patch", "empty diff after cleaning"
    ) == "no_diff_generated"


# ── Regression: _git_sha is cached within TTL but refreshed after ────────────


def test_git_sha_caches_within_ttl(monkeypatch, tmp_path):
    """Within the TTL window, repeated _git_sha calls must hit the subprocess
    only once — failures burst, so collapsing them avoids N git rev-parse
    calls per burst."""
    import subprocess

    from external_llm.agent import tool_failure_log as mod

    calls = {"n": 0}

    def _counting_run(*a, **k):
        calls["n"] += 1
        # Simulate a successful git rev-parse --short HEAD.
        class _R:
            returncode = 0
            stdout = "abc1234\n"
        return _R()

    mod._git_sha_cache.clear()
    monkeypatch.setattr(subprocess, "run", _counting_run)
    monkeypatch.chdir(tmp_path)
    try:
        sha1 = mod._git_sha(str(tmp_path))
        sha2 = mod._git_sha(str(tmp_path))
        assert sha1 == "abc1234"
        assert sha2 == "abc1234"
        assert calls["n"] == 1, f"expected 1 subprocess call, got {calls['n']}"
    finally:
        mod._git_sha_cache.clear()


# -- handler metadata failure_class is preferred over error-text fallback ------


def test_metadata_failure_class_preferred_over_error_text(tmp_path, monkeypatch):
    """When the handler sets failure_class explicitly, the logger must use it
    even if the error text would match a different class."""
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        record_write_tool_failure(
            tool="edit_text",
            ok=False,
            error="old_string not found in foo.py",  # would → search_string_mismatch
            metadata={"failure_class": "custom_handler_class", "file_path": "foo.py"},
            args={"file_path": "foo.py"},
            repo_root=str(tmp_path),
        )
        rows = rec.records()
        assert len(rows) == 1
        assert rows[0]["failure_class"] == "custom_handler_class"
    finally:
        rec.cleanup()


def test_fallback_classifier_used_when_metadata_lacks_class(tmp_path, monkeypatch):
    """A handler that returns no failure_class still gets classified via the
    error-text fallback."""
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    try:
        record_write_tool_failure(
            tool="apply_patch",
            ok=False,
            error="repository lacks the necessary blob to perform 3-way merge",
            metadata={"file_path": "foo.py"},  # no failure_class key
            args={"path": "foo.py"},
            repo_root=str(tmp_path),
        )
        rows = rec.records()
        assert len(rows) == 1
        assert rows[0]["failure_class"] == "patch_apply_failed"
    finally:
        rec.cleanup()


# ── Bounded compaction ───────────────────────────────────────────────────────
from external_llm.agent import tool_failure_log as tfl


def test_log_compacted_when_exceeding_max_records(tmp_path, monkeypatch):
    """The log is bounded: once it exceeds _MAX_FAILURE_LOG_RECORDS the oldest
    records are evicted, keeping only the newest. Validates the call-site wiring
    inside record_write_tool_failure (compaction runs after each append)."""
    rec = _Recorder(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tfl, "_MAX_FAILURE_LOG_RECORDS", 5)
    monkeypatch.setattr(tfl, "_COMPACT_CHECK_EVERY", 1)  # check every append
    tfl._append_counter = 0
    try:
        for i in range(8):
            record_write_tool_failure(
                tool="edit_text",
                ok=False,
                error=f"old_string not found in file (i={i})",
                metadata={"failure_class": "search_string_mismatch", "file_path": "f.py"},
                args={"file_path": "f.py"},
                repo_root=str(tmp_path),
            )
        rows = rec.records()
        # Compacted to the cap; only the 5 newest survive (i=3..7).
        assert len(rows) == 5
        errors = [r["error"] for r in rows]
        assert errors[-1].endswith("i=7)"), errors
        assert not any("i=0)" in e or "i=1)" in e or "i=2)" in e for e in errors), errors
    finally:
        rec.cleanup()


def test_compaction_drops_corrupt_lines(tmp_path):
    """Compaction doubles as self-heal: unparseable lines are dropped during the
    atomic rewrite (mirrors UnifiedStore._heal_file). The newest valid records
    are kept; corrupt lines never survive a compaction pass."""
    path = str(tmp_path / "write_tool_failures.jsonl")
    tfl._append_counter = tfl._COMPACT_CHECK_EVERY - 1  # +1 → % CHECK_EVERY == 0
    # 8 lines: v1..v4, one corrupt, v5..v7. Cap = 5 → keep last 5 of the list,
    # drop the corrupt one during parse → 4 valid records survive.
    lines = [
        '{"tool":"a","failure_class":"x","i":1}',
        '{"tool":"a","failure_class":"x","i":2}',
        '{"tool":"a","failure_class":"x","i":3}',
        '{"tool":"a","failure_class":"x","i":4}',
        'THIS IS NOT JSON — corrupt line',
        '{"tool":"a","failure_class":"x","i":5}',
        '{"tool":"a","failure_class":"x","i":6}',
        '{"tool":"a","failure_class":"x","i":7}',
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    orig_max = tfl._MAX_FAILURE_LOG_RECORDS
    tfl._MAX_FAILURE_LOG_RECORDS = 5
    try:
        tfl._maybe_compact_log(path)
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
        # v1,v2,v3 evicted (oldest); corrupt dropped; v4,v5,v6,v7 kept.
        assert len(rows) == 4, rows
        kept = [r["i"] for r in rows]
        assert kept == [4, 5, 6, 7], kept
    finally:
        tfl._MAX_FAILURE_LOG_RECORDS = orig_max
        tfl._append_counter = 0
