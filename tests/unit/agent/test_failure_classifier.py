"""Unit tests for failure_classifier.py — 100% branch coverage."""

import errno

from external_llm.agent.failure_classifier import (
    FailureClassifier,
    RecoveryAction,
    _classify_by_code,
    _classify_by_text,
    _classify_by_type,
    _has_code_keyword,
    _has_text_phrase,
)

# ── Helpers ──────────────────────────────────────────────────────────────

class FakeResult:
    """Minimal stand-in for a ToolResult with an optional error attribute."""
    def __init__(self, error=None):
        self.error = error


class FakeErrorWithAttrs:
    """Error-like object with arbitrary attributes (code, error_code, errno)."""
    def __init__(self, code=None, error_code=None, errno_val=None, msg=""):
        self.code = code
        self.error_code = error_code
        self.errno = errno_val
        self.msg = msg

    def __str__(self):
        return self.msg


# ======================================================================
# _has_code_keyword
# ======================================================================

class TestHasCodeKeyword:
    def test_exact_match(self):
        assert _has_code_keyword("already_exists", {"already"})
        assert _has_code_keyword("file_missing", {"missing"})
        assert _has_code_keyword("ENOENT", {"enoent"})  # single token match
        assert _has_code_keyword("timeout_error", {"timeout"})

    def test_no_match_when_part_of_word(self):
        # "timeout" in "timeouterror" should NOT match (underscore-delimited)
        assert not _has_code_keyword("timeouterror", {"timeout"})
        # "missing" in "dismissing" should NOT match (not underscore-delimited)
        assert not _has_code_keyword("dismissing", {"missing"})

    def test_empty_code(self):
        assert not _has_code_keyword("", {"already"})
        assert not _has_code_keyword("", frozenset())

    def test_empty_keywords(self):
        assert not _has_code_keyword("already_exists", frozenset())

    def test_multiple_keywords(self):
        assert _has_code_keyword("duplicate_entry", {"already", "duplicate", "idempotent"})
        assert not _has_code_keyword("clean_run", {"already", "duplicate"})

    def test_case_insensitive(self):
        assert _has_code_keyword("Already_Exists", {"already"})
        assert _has_code_keyword("TIMEOUT_ERROR", {"timeout"})

    def test_single_word_code(self):
        # No underscores — the split yields a single token
        assert _has_code_keyword("timeout", {"timeout"})
        assert not _has_code_keyword("timeout", {"already"})

    def test_partial_underscore(self):
        assert _has_code_keyword("file_missing_error", {"missing"})
        assert _has_code_keyword("not_found", {"found"})  # "found" is a token after split


# ======================================================================
# _has_text_phrase
# ======================================================================

class TestHasTextPhrase:
    def test_exact_phrase(self):
        assert _has_text_phrase("already applied", ("already applied",))
        assert _has_text_phrase("context mismatch detected", ("context mismatch",))

    def test_case_insensitive(self):
        assert _has_text_phrase("Already Applied", ("already applied",))
        assert _has_text_phrase("CONTEXT MISMATCH", ("context mismatch",))

    def test_phrase_is_substring(self):
        assert _has_text_phrase("the patch was already applied earlier", ("already applied",))
        assert _has_text_phrase("no such file: foo.py", ("not found", "no such file"))

    def test_no_match(self):
        assert not _has_text_phrase("everything is fine", ("already applied", "not found"))

    def test_empty_text(self):
        assert not _has_text_phrase("", ("already applied",))

    def test_empty_phrases(self):
        assert not _has_text_phrase("error message", ())

    def test_multiple_phrases_any_match(self):
        assert _has_text_phrase("not found", ("already applied", "not found", "timeout"))
        assert _has_text_phrase("timed out", ("already applied", "not found", "timed out"))

    def test_partial_word_no_false_positive(self):
        # These phrases are long enough that false positives from substrings
        # are "extremely unlikely", but let's verify a clean case
        assert not _has_text_phrase("noteworthy founded company", ("not found",))


# ======================================================================
# _classify_by_type
# ======================================================================

class TestClassifyByType:
    def test_file_not_found(self):
        result = _classify_by_type(FileNotFoundError())
        assert result is not None
        assert result.action == RecoveryAction.SWITCH_TOOL
        assert result.reason == "file missing"

    def test_is_a_directory_error(self):
        result = _classify_by_type(IsADirectoryError())
        assert result is not None
        assert result.action == RecoveryAction.SWITCH_TOOL

    def test_transient_types(self):
        for exc_cls in (TimeoutError, ConnectionError, ConnectionResetError,
                        ConnectionAbortedError, BrokenPipeError):
            result = _classify_by_type(exc_cls())
            assert result is not None
            assert result.action == RecoveryAction.RETRY_SAME

    def test_permission_error(self):
        result = _classify_by_type(PermissionError())
        assert result is not None
        assert result.action == RecoveryAction.ABORT
        assert result.reason == "permission denied"

    def test_no_match(self):
        result = _classify_by_type(ValueError("foo"))
        assert result is None

    def test_none(self):
        result = _classify_by_type(None)
        assert result is None


# ======================================================================
# _classify_by_code
# ======================================================================

class TestClassifyByCode:
    def test_numeric_enoent(self):
        err = FakeErrorWithAttrs(errno_val=errno.ENOENT)
        result = _classify_by_code(err)
        assert result is not None
        assert result.action == RecoveryAction.SWITCH_TOOL

    def test_numeric_eacces(self):
        err = FakeErrorWithAttrs(errno_val=errno.EACCES)
        result = _classify_by_code(err)
        assert result is not None
        assert result.action == RecoveryAction.ABORT

    def test_numeric_transient(self):
        for en in (errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED):
            err = FakeErrorWithAttrs(errno_val=en)
            result = _classify_by_code(err)
            assert result is not None
            assert result.action == RecoveryAction.RETRY_SAME

    def test_code_attribute(self):
        err = FakeErrorWithAttrs(code="already_exists")
        result = _classify_by_code(err)
        assert result is not None
        assert result.action == RecoveryAction.SKIP

    def test_error_code_attribute(self):
        err = FakeErrorWithAttrs(error_code="missing_file")
        result = _classify_by_code(err)
        assert result is not None
        assert result.action == RecoveryAction.SWITCH_TOOL

    def test_transient_string_code(self):
        err = FakeErrorWithAttrs(code="timeout_error")
        result = _classify_by_code(err)
        assert result is not None
        assert result.action == RecoveryAction.RETRY_SAME

    def test_code_is_none(self):
        err = FakeErrorWithAttrs()  # all None
        result = _classify_by_code(err)
        assert result is None

    def test_code_not_matching(self):
        err = FakeErrorWithAttrs(code="unknown_error")
        result = _classify_by_code(err)
        assert result is None

    def test_priority_code_over_error_code(self):
        # .code should be checked before .error_code
        err = FakeErrorWithAttrs(code="already_exists", error_code="unknown")
        result = _classify_by_code(err)
        assert result is not None
        assert result.action == RecoveryAction.SKIP

    def test_priority_error_code_over_errno(self):
        # .error_code should be checked after .code but before .errno
        err = FakeErrorWithAttrs(error_code="missing_file", errno_val=0)
        result = _classify_by_code(err)
        assert result is not None
        assert result.action == RecoveryAction.SWITCH_TOOL


# ======================================================================
# _classify_by_text
# ======================================================================

class TestClassifyByText:
    def test_already_applied(self):
        result = _classify_by_text("already applied")
        assert result.action == RecoveryAction.SKIP
        assert result.reason == "patch already applied"

    def test_context_mismatch(self):
        result = _classify_by_text("hunk #1 context mismatch")
        assert result.action == RecoveryAction.READ_FIRST
        assert result.reason == "patch context mismatch"

    def test_file_missing(self):
        result = _classify_by_text("no such file: foo.py")
        assert result.action == RecoveryAction.SWITCH_TOOL
        assert result.reason == "file missing"

    def test_transient(self):
        result = _classify_by_text("connection timeout")
        assert result.action == RecoveryAction.RETRY_SAME
        assert result.reason == "transient failure"

    def test_generic_fallback(self):
        result = _classify_by_text("something went wrong")
        assert result.action == RecoveryAction.RETRY_SAME
        assert result.reason == "generic failure"

    def test_empty_string(self):
        result = _classify_by_text("")
        assert result.action == RecoveryAction.RETRY_SAME

    def test_priority_already_over_missing(self):
        # "already applied" appears before "not found" in the cascade
        result = _classify_by_text("already applied: not found")
        assert result.action == RecoveryAction.SKIP


# ======================================================================
# FailureClassifier.classify — integration path
# ======================================================================

class TestFailureClassifierClassify:
    def make_classifier(self):
        return FailureClassifier()

    def test_error_is_none(self):
        clf = self.make_classifier()
        result = FakeResult(error=None)
        classification = clf.classify("some_tool", {}, result)
        assert classification.action == RecoveryAction.RETRY_SAME
        assert classification.reason == "generic failure"

    def test_classify_by_type_file_missing(self):
        clf = self.make_classifier()
        result = FakeResult(error=FileNotFoundError("no such file"))
        classification = clf.classify("read_file", {}, result)
        assert classification.action == RecoveryAction.SWITCH_TOOL
        assert classification.reason == "file missing"

    def test_classify_by_type_permission(self):
        clf = self.make_classifier()
        result = FakeResult(error=PermissionError("access denied"))
        classification = clf.classify("write_file", {}, result)
        assert classification.action == RecoveryAction.ABORT

    def test_classify_by_type_transient(self):
        clf = self.make_classifier()
        result = FakeResult(error=TimeoutError("timed out"))
        classification = clf.classify("read_file", {}, result)
        assert classification.action == RecoveryAction.RETRY_SAME
        assert classification.reason == "transient failure"

    def test_classify_by_type_no_match_falls_to_code(self):
        clf = self.make_classifier()
        err = FakeErrorWithAttrs(code="already_exists", msg="patch already applied")
        result = FakeResult(error=err)
        classification = clf.classify("apply_patch", {}, result)
        assert classification.action == RecoveryAction.SKIP
        assert classification.reason == "patch already applied"

    def test_classify_by_code_falls_to_text(self):
        clf = self.make_classifier()
        # Error with no type match, no code match, but text match
        err = FakeErrorWithAttrs(code="some_other_error", msg="hunk #1 does not apply")
        result = FakeResult(error=err)
        classification = clf.classify("apply_patch", {}, result)
        assert classification.action == RecoveryAction.READ_FIRST
        assert classification.reason == "patch context mismatch"

    def test_classify_all_fallback_to_generic(self):
        clf = self.make_classifier()
        err = FakeErrorWithAttrs(code="unknown", msg="something weird happened")
        result = FakeResult(error=err)
        classification = clf.classify("some_tool", {}, result)
        assert classification.action == RecoveryAction.RETRY_SAME
        assert classification.reason == "generic failure"

    def test_classify_with_non_error_object(self):
        clf = self.make_classifier()
        # Some callers pass a string as result
        result = "plain string result"
        classification = clf.classify("some_tool", {}, result)
        assert classification.action == RecoveryAction.RETRY_SAME

    def test_classify_result_without_error_attr(self):
        clf = self.make_classifier()
        result = object()  # no .error attribute
        classification = clf.classify("some_tool", {}, result)
        assert classification.action == RecoveryAction.RETRY_SAME

    def test_type_match_priority_over_code(self):
        """Type-based classification is checked first and should win."""
        clf = self.make_classifier()
        class MyFileNotFound(FileNotFoundError):
            def __init__(self):
                super().__init__()
                self.code = "timeout_error"  # would trigger transient if type was checked second
        result = FakeResult(error=MyFileNotFound())
        classification = clf.classify("read_file", {}, result)
        assert classification.action == RecoveryAction.SWITCH_TOOL  # type wins

    def test_code_match_priority_over_text(self):
        """Code-based classification is checked before text-based."""
        clf = self.make_classifier()
        err = FakeErrorWithAttrs(code="already_exists", msg="hunk #1 does not apply")
        result = FakeResult(error=err)
        classification = clf.classify("apply_patch", {}, result)
        assert classification.action == RecoveryAction.SKIP  # code wins over text


# ======================================================================
# Priority 0 — the handler's own structured verdict
# ======================================================================
# The write tools compute `metadata["failure_class"]` and the classifier used
# to ignore it, re-deriving a verdict from the error string and getting it
# wrong: `_TEXT_FILE_MISSING`'s bare "not found" matched edit_text's
# `old_string not found in <path>`, so the single most common edit failure was
# labelled "file missing" and answered with "switch to a different tool" —
# contradicting the tool's own "re-read the file and include 2-3 lines of
# surrounding context". It also poisoned the persistent recall store:
# `edit_text::file missing` was this repo's top pattern with every recorded
# path present on disk.

_UNSET = object()


class ResultWithMetadata:
    """A ToolResult stand-in carrying handler-set failure metadata.

    Pass *failure_class* for the normal shape, or *metadata* to inject a
    malformed value directly.
    """
    def __init__(self, error=None, failure_class=None, metadata=_UNSET):
        self.error = error
        if metadata is not _UNSET:
            self.metadata = metadata
        else:
            self.metadata = {"failure_class": failure_class} if failure_class else None


class TestFailureClassMetadata:
    def make_classifier(self):
        return FailureClassifier()

    def test_search_string_mismatch_is_read_first_not_file_missing(self):
        """The regression this tier exists for."""
        clf = self.make_classifier()
        result = ResultWithMetadata(
            error="old_string not found in external_llm/languages/base.py",
            failure_class="search_string_mismatch",
        )
        c = clf.classify("edit_text", {}, result)
        assert c.action == RecoveryAction.READ_FIRST
        assert c.reason == "patch context mismatch"

    def test_metadata_beats_a_misleading_error_string(self):
        """Priority 0 must win even when the text tier would say otherwise."""
        clf = self.make_classifier()
        result = ResultWithMetadata(
            error="File not found: a.py",  # would text-classify as file missing
            failure_class="already_equal",
        )
        c = clf.classify("edit_text", {}, result)
        assert c.action == RecoveryAction.SKIP
        assert c.reason == "patch already applied"

    def test_a_genuinely_missing_file_still_switches_tool(self):
        clf = self.make_classifier()
        result = ResultWithMetadata(error="File not found: a.py", failure_class="file_not_found")
        c = clf.classify("edit_text", {}, result)
        assert c.action == RecoveryAction.SWITCH_TOOL
        assert c.reason == "file missing"

    def test_an_unmapped_class_falls_through_to_the_text_tiers(self):
        """The map is deliberately partial — a class with no single unambiguous
        recovery must not be forced into one."""
        clf = self.make_classifier()
        result = ResultWithMetadata(
            error="hunk #1 does not apply",
            failure_class="structural_gate_violation",
        )
        c = clf.classify("apply_patch", {}, result)
        assert c.reason == "patch context mismatch"  # from text, not from the map

    def test_an_unknown_class_string_falls_through(self):
        clf = self.make_classifier()
        result = ResultWithMetadata(error="something odd", failure_class="not_a_real_class")
        c = clf.classify("edit_text", {}, result)
        assert c.reason == "generic failure"

    def test_missing_or_malformed_metadata_is_ignored(self):
        clf = self.make_classifier()
        for md in (None, {}, {"failure_class": ""}, "not-a-dict", 42):
            result = ResultWithMetadata(error="File not found: a.py", metadata=md)
            c = clf.classify("edit_text", {}, result)
            assert c.action == RecoveryAction.SWITCH_TOOL, md


# ======================================================================
# Text tier — "not found" is only ambiguous for text-matching edit tools
# ======================================================================

class TestNotFoundIsToolAware:
    def make_classifier(self):
        return FailureClassifier()

    def test_edit_tool_not_found_means_the_search_text(self):
        """These paths return without setting failure_class, so the text tier
        is the only thing standing between them and the wrong advice."""
        clf = self.make_classifier()
        for tool, err in [
            ("anchor_edit", "anchor text not found: 'def foo():'"),
            ("anchor_edit", "anchor_edit(delete): pattern 'x' not found in a.py"),
            ("edit_text", "HINT: 'before' text not found in 'asi.py'."),
            ("modify_symbol", "symbol body not found"),
        ]:
            c = clf.classify(tool, {}, FakeResult(error=err))
            assert c.action == RecoveryAction.READ_FIRST, (tool, err)
            assert c.reason == "patch context mismatch", (tool, err)

    def test_explicit_filesystem_phrasing_still_wins_for_edit_tools(self):
        clf = self.make_classifier()
        for err in ("File not found: nope.py", "no such file or directory", "a.py does not exist"):
            c = clf.classify("edit_text", {}, FakeResult(error=err))
            assert c.action == RecoveryAction.SWITCH_TOOL, err
            assert c.reason == "file missing", err

    def test_read_tools_keep_the_filesystem_reading(self):
        clf = self.make_classifier()
        c = clf.classify("read_file", {}, FakeResult(error="Path not found or outside repo: 'nope.py'"))
        assert c.action == RecoveryAction.SWITCH_TOOL
        assert c.reason == "file missing"

    def test_a_bare_not_found_from_a_non_edit_tool_is_not_forced(self):
        """`find_symbol`'s "Symbol 'x' not found" is neither a missing file nor
        a patch mismatch; guessing either would be worse than the generic
        fallback, which the recall store excludes from hinting."""
        clf = self.make_classifier()
        c = clf.classify("find_symbol", {}, FakeResult(error="Symbol 'x' not found."))
        assert c.reason == "generic failure"
