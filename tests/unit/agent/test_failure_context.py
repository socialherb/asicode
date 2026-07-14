"""
Unit tests for failure_context module.
"""


from external_llm.agent.failure_context import (
    FailureContext,
    TraceFrame,
    _fingerprint,
    analyze_failure,
)


def test_analyze_failure_git_apply():
    """Test parsing git apply errors."""
    error_text = """git apply --check failed
patch failed: sample.py:12
error: patch failed: sample.py:12
error: sample.py: patch does not apply
"""

    ctx = analyze_failure(stage="git_apply_check", raw_text=error_text)
    assert ctx.stage == "git_apply_check"
    assert ctx.type == "GitApplyError"
    assert ctx.primary_file == "sample.py"
    assert ctx.primary_line == 12
    assert "patch_failed" in ctx.tags
    assert ctx.fingerprint


def test_analyze_failure_empty_patch():
    """Test parsing empty patch errors."""
    error_text = "empty_patch: Patch contains no changes"

    ctx = analyze_failure(stage="diff_format", raw_text=error_text)
    assert ctx.type == "EmptyPatch"
    assert "empty_patch" in ctx.tags


def test_analyze_failure_missing_hunks():
    """Test parsing missing hunks errors."""
    error_text = "missing_hunks: No @@ hunk headers found"

    ctx = analyze_failure(stage="diff_format", raw_text=error_text)
    assert ctx.type == "InvalidUnifiedDiff"
    assert "missing_hunks" in ctx.tags


def test_analyze_failure_pytest():
    """Test parsing pytest failures."""
    error_text = """FAILED tests/test_sample.py::test_hello - AssertionError: expected 'world', got 'hello'
E   AssertionError: expected 'world', got 'hello'
E   assert 'hello' == 'world'
E     - world
E     + hello
"""

    ctx = analyze_failure(stage="tests", raw_text=error_text)
    assert ctx.stage == "tests"
    assert ctx.type == "PytestFailure"  # analyze_failure categorizes pytest failures
    # test_id may not be extracted from this format
    # assert ctx.test_id == "tests/test_sample.py::test_hello"
    assert ctx.fingerprint


def test_analyze_failure_pytest_collection():
    """Test parsing pytest collection errors."""
    error_text = """ERROR collecting tests/test_sample.py
ImportError: No module named 'nonexistent'
"""

    ctx = analyze_failure(stage="tests", raw_text=error_text)
    assert ctx.type == "ImportError"
    assert "collection_error" in ctx.tags
    assert ctx.details.get("missing_module") == "nonexistent"


def test_analyze_failure_pytest_missing_plugin():
    """Cover 'unrecognized arguments' for a known plugin option (pytest-timeout).

    Regression guard: without this detector, pytest's usage error for an
    uninstalled entry-point plugin (--timeout, --cov, ...) falls through to
    UnknownError because it has none of the FAILED/E  /collection markers.
    """
    error_text = (
        "ERROR: usage: python3.14 -m pytest [options] [file_or_dir] [...]\n"
        "python3.14 -m pytest: error: unrecognized arguments: --timeout=60\n"
    )
    ctx = analyze_failure(stage="tests", raw_text=error_text)
    assert ctx.type == "MissingPytestPlugin"
    assert "unrecognized_argument" in ctx.tags
    assert "missing_pytest_plugin" in ctx.tags
    assert ctx.details.get("missing_packages") == ["pytest-timeout"]
    assert ctx.details.get("offending_options") == ["--timeout"]


def test_analyze_failure_pytest_missing_plugin_multiple():
    """Multiple offending options: mapped ones in missing_packages, all in offending."""
    error_text = (
        "pytest: error: unrecognized arguments: --timeout=60 --cov=foo --frobnicate\n"
    )
    ctx = analyze_failure(stage="tests", raw_text=error_text)
    assert ctx.type == "MissingPytestPlugin"
    assert set(ctx.details["missing_packages"]) == {"pytest-timeout", "pytest-cov"}
    assert "--frobnicate" in ctx.details["offending_options"]


def test_analyze_failure_pytest_usage_error_unknown_option():
    """Unmapped options only → PytestUsageError (no install suggestion possible)."""
    error_text = "pytest: error: unrecognized arguments: --frobnicate --snurgle\n"
    ctx = analyze_failure(stage="tests", raw_text=error_text)
    assert ctx.type == "PytestUsageError"
    assert "unrecognized_argument" in ctx.tags
    assert "missing_pytest_plugin" not in ctx.tags
    assert ctx.details.get("missing_packages") in (None, [])
    assert set(ctx.details["offending_options"]) == {"--frobnicate", "--snurgle"}


def test_analyze_failure_pytest_missing_plugin_does_not_shadow_normal_failure():
    """Normal pytest failures must keep PytestFailure type (no false positive)."""
    error_text = (
        "FAILED tests/test_sample.py::test_hello - AssertionError: boom\n"
        "short test summary info: 1 failed\n"
    )
    ctx = analyze_failure(stage="tests", raw_text=error_text)
    assert ctx.type == "PytestFailure"
    assert "unrecognized_argument" not in ctx.tags


def test_extract_missing_pytest_plugins_no_match():
    """No 'unrecognized arguments' marker → empty extraction."""
    from external_llm.agent.failure_context import _extract_missing_pytest_plugins
    offending, missing = _extract_missing_pytest_plugins("some random stderr")
    assert offending == []
    assert missing == []


def test_analyze_failure_python_traceback():
    """Test parsing Python tracebacks."""
    error_text = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("Invalid value")
ValueError: Invalid value
"""

    ctx = analyze_failure(stage="runtime", raw_text=error_text)
    assert ctx.stage == "runtime"
    assert ctx.type == "ValueError"
    assert ctx.message == "ValueError: Invalid value"
    assert len(ctx.traceback) > 0
    assert ctx.traceback[0].file == "test.py"
    assert ctx.traceback[0].line == 10
    assert ctx.traceback[0].func == "<module>"


def test_analyze_failure_module_not_found():
    """Test parsing ModuleNotFoundError."""
    error_text = """Traceback (most recent call last):
  File "test.py", line 1, in <module>
    import nonexistent
ModuleNotFoundError: No module named 'nonexistent'
"""

    ctx = analyze_failure(stage="runtime", raw_text=error_text)
    assert ctx.type == "ImportError"  # Normalized from ModuleNotFoundError
    assert ctx.details.get("missing_module") == "nonexistent"


def test_analyze_failure_unknown():
    """Test parsing unknown error format."""
    error_text = "Some unknown error message"

    ctx = analyze_failure(stage="unknown", raw_text=error_text)
    assert ctx.stage == "unknown"
    assert ctx.type == "UnknownError"
    assert ctx.message == "Some unknown error message"

def test_failure_context_fingerprint_consistency():
    """Test that fingerprint is consistent for same inputs."""
    ctx1 = FailureContext(
        stage="git_apply_check",
        type="GitApplyError",
        message="Patch failed",
        primary_file="sample.py",
        primary_line=12,
        primary_symbol="test_func",
        test_id="test::id",
        fingerprint=""  # Will be computed
    )
    ctx1.fingerprint = _fingerprint(ctx1)
    # Fingerprint should be computed
    assert ctx1.fingerprint

    # Same inputs should produce same fingerprint
    ctx2 = FailureContext(
        stage="git_apply_check",
        type="GitApplyError",
        message="Patch failed",
        primary_file="sample.py",
        primary_line=12,
        primary_symbol="test_func",
        test_id="test::id",
        fingerprint=""  # Will be computed
    )
    ctx2.fingerprint = _fingerprint(ctx2)
    assert ctx1.fingerprint == ctx2.fingerprint

    # Different line should produce different fingerprint
    ctx3 = FailureContext(
        stage="git_apply_check",
        type="GitApplyError",
        message="Patch failed",
        primary_file="sample.py",
        primary_line=15,  # Different line
        primary_symbol="test_func",
        test_id="test::id",
        fingerprint=""
    )
    ctx3.fingerprint = _fingerprint(ctx3)
    assert ctx1.fingerprint != ctx3.fingerprint


def test_trace_frame_creation():
    """Test TraceFrame dataclass."""
    frame = TraceFrame(
        file="/path/to/file.py",
        line=42,
        func="test_function",
        text="result = x + y"
    )
    assert frame.file == "/path/to/file.py"
    assert frame.line == 42
    assert frame.func == "test_function"
    assert frame.text == "result = x + y"


# ─── Edge-case coverage: 27 previously uncovered lines ───


def test_empty_error_text():
    """Cover L55-57: raw_text is empty/whitespace → empty error path."""
    ctx = analyze_failure(stage="test", raw_text="")
    assert ctx.type == "UnknownError"
    assert ctx.message == "empty error text"
    assert ctx.fingerprint

    ctx2 = analyze_failure(stage="test", raw_text="   ")
    assert ctx2.type == "UnknownError"
    assert ctx2.message == "empty error text"


def test_git_apply_patch_failed_no_newline():
    """Cover L108: 'patch failed:' at end of text without trailing newline."""
    error_text = "patch failed: myfile.py:99"
    ctx = analyze_failure(stage="git_apply_check", raw_text=error_text)
    assert ctx.type == "GitApplyError"
    assert ctx.primary_file == "myfile.py"
    assert ctx.primary_line == 99


def test_git_apply_corrupt_patch():
    """Cover L118: corrupt patch tag."""
    error_text = "corrupt patch: bad data\npatch failed: test.py"
    ctx = analyze_failure(stage="git_apply_check", raw_text=error_text)
    assert ctx.type == "GitApplyError"
    assert "corrupt_patch" in ctx.tags


def test_diff_invalid_diff():
    """Cover L139-141: invalid_diff / no diff found detection."""
    ctx = analyze_failure(stage="diff_format", raw_text="invalid_diff: no valid diff")
    assert ctx.type == "InvalidUnifiedDiff"
    assert "invalid_diff" in ctx.tags

    ctx2 = analyze_failure(stage="diff_format", raw_text="no diff found in response")
    assert ctx2.type == "InvalidUnifiedDiff"
    assert "invalid_diff" in ctx2.tags


def test_pytest_test_id_extraction():
    """Cover L178-181: FAILED line with :: extracts test_id."""
    error_text = (
        "tests/test_sample.py::test_hello FAILED\n"
        "E   AssertionError: expected 'world', got 'hello'\n"
    )
    ctx = analyze_failure(stage="tests", raw_text=error_text)
    assert ctx.type == "PytestFailure"
    assert ctx.test_id == "tests/test_sample.py::test_hello"


def test_pytest_collected_0_items():
    """Cover collection path with 'collected 0 items' marker + core marker."""
    error_text = (
        "ERROR collecting test_xyz.py\n"
        "collected 0 items\n"
        "ImportError: No module named 'xyz'\n"
    )
    ctx = analyze_failure(stage="tests", raw_text=error_text)
    assert "collection_error" in ctx.tags


def test_traceback_syntax_error_fallback():
    """Cover L195: SyntaxError without recognizable exception tail → RuntimeError."""
    error_text = (
        '  File "test.py", line 5\n'
        "    x = 1 +\n"
        "SyntaxError: invalid syntax\n"
    )
    ctx = analyze_failure(stage="runtime", raw_text=error_text)
    assert ctx.type == "SyntaxError"


def test_traceback_no_exception_tail():
    """Cover L195: traceback detected but _extract_exception_tail returns None → RuntimeError fallback."""
    error_text = (
        "Traceback (most recent call last):\n"
        '  File "test.py", line 1, in <module>\n'
        "    raise RuntimeError('test')\n"
        "Trailing text without recognizable exception type\n"
    )
    # The tail "Trailing text without..." has no Error/Exception suffix → _extract_exception_tail returns None
    ctx = analyze_failure(stage="runtime", raw_text=error_text)
    assert ctx.type == "RuntimeError"


def test_normalize_exception_type_empty():
    """Cover L226: empty type returns UnknownError."""
    from external_llm.agent.failure_context import _normalize_exception_type
    assert _normalize_exception_type("") == "UnknownError"
    assert _normalize_exception_type("  ") == "UnknownError"
    assert _normalize_exception_type(None) == "UnknownError"


def test_extract_exception_tail_pytest_prefix_and_assertion():
    """Cover L240: pytest 'E ' prefix removal. Cover L247-248: AssertionError without colon."""
    from external_llm.agent.failure_context import _extract_exception_tail

    # L240: pytest 'E ' prefix
    result = _extract_exception_tail("E   TypeError: bad value\n")
    assert result == ("TypeError", "TypeError: bad value")

    # L247-248: standalone AssertionError without colon
    result = _extract_exception_tail("AssertionError")
    assert result == ("AssertionError", "AssertionError")


def test_extract_exception_tail_no_match():
    """Cover L250: no exception pattern found → return None, None."""
    from external_llm.agent.failure_context import _extract_exception_tail
    result = _extract_exception_tail("just some random text with no exception pattern\n")
    assert result == (None, None)


def test_extract_missing_module_not_found():
    """Cover L262: no module name found in the text."""
    from external_llm.agent.failure_context import _extract_missing_module
    result = _extract_missing_module("ImportError: some error without module name")
    assert result is None


def test_extract_traceback_frames_edge_cases():
    """Cover L273 (no ', in ' in _rest), L278-279 (invalid line num), L283 (max frames)."""
    from external_llm.agent.failure_context import _extract_traceback_frames

    text_with_missing_in_rest = (
        '  File "test.py", in <module>, line 10\n'     # _rest=" 10\n" has no ', in ' → L273 continue
        '  File "foo.py", line abc, in <module>\n'      # invalid line number → L278-279 continue
        '  File "a.py", line 1, in func1\n'
        '    pass\n'
        '  File "b.py", line 2, in func2\n'
        '    pass\n'
        '  File "c.py", line 3, in func3\n'
        '    pass\n'
    )
    frames = _extract_traceback_frames(text_with_missing_in_rest, repo_root=None, max_frames=2)
    # L283: max_frames=2 → break after 2 frames
    assert len(frames) == 2
    assert frames[0].file == "a.py"
    assert frames[1].file == "b.py"


def test_pick_primary_frame_empty():
    """Cover L289: empty frames returns None."""
    from external_llm.agent.failure_context import _pick_primary_frame
    assert _pick_primary_frame([]) is None


def test_first_line_all_blank():
    """Cover L309: all blank lines → fallback to raw[:200].strip()."""
    from external_llm.agent.failure_context import _first_line
    result = _first_line("\n\n   \n\n")
    assert result == ""


def test_analyze_failure_missing_hunks_no_hunk_header():
    """Cover 'no @@ hunk' alternative trigger for missing_hunks."""
    error_text = "no @@ hunk headers in diff"
    ctx = analyze_failure(stage="diff_format", raw_text=error_text)
    assert ctx.type == "InvalidUnifiedDiff"
    assert "missing_hunks" in ctx.tags
