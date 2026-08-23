"""RED→GREEN coverage tests for C / C++ providers (59% → 100%).

Covers every uncovered surface of c_provider.py:

- _parse_cc_diagnostics: non-matching lines, column-less "fatal error"
  lines, and the filter_resolution branch
- _find_compile_commands upward walk; _extract_include_flags arguments /
  command / shlex-failure / empty / relative-path resolution;
  _collect_i_flags separated and combined spellings
- capabilities caching; tempfile-failure and toolchain-degrade paths in
  both syntax and semantic compiles
- validate_semantics: not-on-disk skip, compile_commands.json include
  injection, and the ``.h`` union retry (C compile fails → C++ accepts)
- get_symbol_patterns kind filtering; None lint/test commands
- regex fallbacks (tree-sitter forced unavailable): find_symbol_in_file,
  top-level definitions (function/struct/enum/macro incl. EOF macro),
  find_top_level_definitions, find_symbol_body_range, definition keywords
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from external_llm.languages.c_provider import (
    CppSyntaxProvider,
    CSyntaxProvider,
    _collect_i_flags,
    _extract_include_flags,
    _find_compile_commands,
    _match_compile_commands_entry,
    _parse_cc_diagnostics,
    _same_path,
)
from external_llm.languages.models import LanguageId


def _fake_proc(returncode, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


C_SRC = "int main(void) { return 0; }\n"


# ── module-level helpers ───────────────────────────────────────────────────


class TestParseCcDiagnostics:
    def test_non_matching_lines_ignored(self):
        out = "note: this is a note\nx.c:1:1: warning: unused variable 'y'\n"
        assert _parse_cc_diagnostics(out, LanguageId.C, filter_resolution=False) == []

    def test_col_optional_fatal_error(self):
        out = "x.c:3: fatal error: foo.h: No such file or directory\n"
        errs = _parse_cc_diagnostics(out, LanguageId.C, filter_resolution=False)
        assert len(errs) == 1
        assert errs[0].line == 3 and errs[0].col == 0
        assert errs[0].severity == "error"

    def test_filter_resolution_branch(self):
        out = "x.c:1:10: fatal error: foo.h: No such file or directory\n"
        filtered = _parse_cc_diagnostics(out, LanguageId.C, filter_resolution=True)
        assert filtered == []
        raw = _parse_cc_diagnostics(out, LanguageId.C, filter_resolution=False)
        assert len(raw) == 1


class TestFindCompileCommands:
    def test_walks_up_to_ccdb(self, tmp_path):
        ccdb = tmp_path / "compile_commands.json"
        ccdb.write_text("[]")
        src = tmp_path / "src" / "sub" / "main.c"
        src.parent.mkdir(parents=True)
        src.write_text("int main(void) { return 0; }")
        assert _find_compile_commands(str(src)) == str(ccdb)

    def test_not_found_returns_none(self, tmp_path):
        src = tmp_path / "main.c"
        src.write_text("int main(void) { return 0; }")
        assert _find_compile_commands(str(src)) is None


class TestExtractIncludeFlags:
    def test_arguments_array_preferred(self):
        entry = {"directory": "/proj", "arguments": ["gcc", "-Iinc", "-I", "sep", "-c", "a.c"]}
        assert _extract_include_flags(entry) == ["-I/proj/inc", "-I/proj/sep"]

    def test_command_string_fallback(self):
        entry = {"directory": "/proj", "command": "gcc -Iinc -DFOO a.c"}
        assert _extract_include_flags(entry) == ["-I/proj/inc"]

    def test_shlex_failure_falls_back_to_split(self):
        entry = {"directory": "/proj", "command": 'gcc -I"unterminated a.c'}
        flags = _extract_include_flags(entry)
        assert flags == ['-I/proj/"unterminated']  # cmd.split() keeps the token

    def test_empty_entry_returns_empty(self):
        assert _extract_include_flags({}) == []
        assert _extract_include_flags({"directory": "/proj"}) == []

    def test_relative_I_resolved_against_directory(self):
        entry = {"directory": "/proj/build", "arguments": ["gcc", "-I../inc", "-I/abs/inc", "-c", "a.c"]}
        flags = _extract_include_flags(entry)
        assert "-I/proj/inc" in flags  # ../inc → /proj/inc
        assert "-I/abs/inc" in flags  # absolute untouched


class TestCollectIFlags:
    def test_separated_and_combined(self):
        assert _collect_i_flags(["gcc", "-I", "dir", "-Idir2", "-Wall", "-c", "x.c"]) == [
            "-Idir",
            "-Idir2",
        ]

    def test_lone_trailing_dash_I_ignored(self):
        assert _collect_i_flags(["gcc", "-I"]) == []
        assert _collect_i_flags(["gcc", "-I", "-c", "x.c"]) == ["-I-c"]  # pairs with next token


class TestSamePath:
    def test_empty_rejected(self):
        assert _same_path("", "/base", "/base/x.c") is False
        assert _same_path("/base/x.c", "/base", "") is False

    def test_relative_and_absolute_spellings(self):
        assert _same_path("./x.c", "/base", "/base/x.c") is True
        assert _same_path("x.c", "/base", "/base/x.c") is True
        assert _same_path("/base/other.c", "/base", "/base/x.c") is False

    def test_match_compile_commands_entry(self):
        entries = [
            {"file": "a.c", "directory": "/p"},
            {"file": "/p/b.c", "directory": "/p"},
        ]
        assert _match_compile_commands_entry(entries, "/p/b.c") == entries[1]
        assert _match_compile_commands_entry(entries, "/p/zz.c") is None


# ── capabilities / basic commands ──────────────────────────────────────────


class TestCapabilitiesAndCommands:
    def test_capabilities_cached(self):
        p = CSyntaxProvider()
        assert p.capabilities() is p.capabilities()

    def test_symbol_patterns_kinds(self):
        p = CSyntaxProvider()
        any_kinds = {sp.kind for sp in p.get_symbol_patterns("any")}
        assert any_kinds == {"function", "struct", "enum", "typedef", "macro"}
        assert {sp.kind for sp in p.get_symbol_patterns("function")} == {"function"}
        assert {sp.kind for sp in p.get_symbol_patterns("macro")} == {"macro"}
        assert {sp.kind for sp in p.get_symbol_patterns("type")} == {"struct", "enum", "typedef"}
        assert {sp.kind for sp in p.get_symbol_patterns("struct")} == {"struct", "enum", "typedef"}

    def test_lint_and_test_commands_none(self):
        assert CSyntaxProvider().get_lint_command("x.c") is None
        assert CSyntaxProvider().get_test_command("/repo") is None
        assert CppSyntaxProvider().get_lint_command("x.cpp") is None
        assert CppSyntaxProvider().get_test_command("/repo", ["--x"]) is None

    def test_definition_keywords(self):
        assert CSyntaxProvider().get_definition_keywords() == [
            "struct ",
            "union ",
            "enum ",
            "typedef ",
            "#define ",
        ]

    def test_language_ids(self):
        assert CSyntaxProvider().language_id() is LanguageId.C
        assert CppSyntaxProvider().language_id() is LanguageId.CPP

    def test_file_globs(self):
        assert set(CSyntaxProvider().get_file_globs()) == {"*.c", "*.h"}
        assert set(CppSyntaxProvider().get_file_globs()) == {"*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh"}


# ── syntax compile: degrade paths ──────────────────────────────────────────


class TestSyntaxCompileDegrade:
    @staticmethod
    def _gcc_only():
        return patch(
            "external_llm.languages.c_provider.shutil.which",
            side_effect=lambda c: f"/usr/bin/{c}" if c in ("gcc", "clang") else None,
        )

    def test_tempfile_failure_tree_sitter_fallback(self):
        with (
            self._gcc_only(),
            patch(
                "external_llm.languages.c_provider._tempfile_for_content",
                return_value=("", lambda: None),
            ),
        ):
            r = CSyntaxProvider().validate_syntax("main.c", C_SRC)
        assert r.ok is True

    def test_file_not_found_tree_sitter_fallback(self):
        with (
            self._gcc_only(),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                side_effect=FileNotFoundError("gcc vanished"),
            ),
        ):
            r = CSyntaxProvider().validate_syntax("main.c", C_SRC)
        assert r.ok is True

    def test_timeout_tree_sitter_fallback(self):
        with (
            self._gcc_only(),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                side_effect=subprocess.TimeoutExpired("gcc", 30),
            ),
        ):
            r = CSyntaxProvider().validate_syntax("main.c", C_SRC)
        assert r.ok is True

    def test_timeout_cpp(self):
        with (
            patch(
                "external_llm.languages.c_provider.shutil.which",
                side_effect=lambda c: f"/usr/bin/{c}" if c in ("g++", "clang++") else None,
            ),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                side_effect=subprocess.TimeoutExpired("g++", 30),
            ),
        ):
            r = CppSyntaxProvider().validate_syntax("main.cpp", C_SRC)
        assert r.ok is True


# ── semantic compile: degrade + ccdb + .h union ────────────────────────────


class TestSemantics:
    @staticmethod
    def _gcc_only():
        return patch(
            "external_llm.languages.c_provider.shutil.which",
            side_effect=lambda c: f"/usr/bin/{c}" if c in ("gcc", "clang") else None,
        )

    def test_not_on_disk_skips(self):
        r = CSyntaxProvider().validate_semantics("/nonexistent/x.c")
        assert r.checked is False
        assert r.skip_reason == "the file is not on disk"

    def test_file_not_found_skips(self, tmp_path):
        f = tmp_path / "main.c"
        f.write_text(C_SRC)
        with (
            self._gcc_only(),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                side_effect=FileNotFoundError("gcc"),
            ),
        ):
            r = CSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert "is not installed" in r.skip_reason

    def test_timeout_skips(self, tmp_path):
        f = tmp_path / "main.c"
        f.write_text(C_SRC)
        with (
            self._gcc_only(),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                side_effect=subprocess.TimeoutExpired("gcc", 30),
            ),
        ):
            r = CSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert "timed out" in r.skip_reason

    def test_ccdb_include_flags_injected(self, tmp_path):
        f = tmp_path / "main.c"
        f.write_text(C_SRC)
        (tmp_path / "compile_commands.json").write_text(
            json.dumps(
                [
                    {
                        "directory": str(tmp_path),
                        "file": "main.c",
                        "arguments": ["gcc", "-Iinc", "-c", "main.c"],
                    },
                ]
            )
        )
        with (
            self._gcc_only(),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                return_value=_fake_proc(0),
            ) as run,
        ):
            r = CSyntaxProvider().validate_semantics(str(f))
        assert r.ok is True
        cmd = run.call_args.args[0]
        assert f"-I{tmp_path}/inc" in cmd  # relative -I resolved against ccdb dir

    def test_header_union_retry_cpp_accepts(self, tmp_path):
        f = tmp_path / "x.h"
        f.write_text("namespace N { int v = 1; }\n")
        with (
            patch(
                "external_llm.languages.c_provider.shutil.which",
                side_effect=lambda c: f"/usr/bin/{c}" if c in ("gcc", "g++") else None,
            ),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                side_effect=[
                    _fake_proc(1, stderr="x.h:1:10: error: expected ';' before '}'"),
                    _fake_proc(0),
                ],
            ) as run,
        ):
            r = CSyntaxProvider().validate_semantics(str(f))
        assert r.ok is True  # C compile failed, C++ accepted → valid C++ header
        assert run.call_count == 2

    def test_semantics_error_kept_for_target_file(self, tmp_path):
        f = tmp_path / "main.c"
        f.write_text(C_SRC)
        with (
            self._gcc_only(),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                return_value=_fake_proc(1, stderr="main.c:1:1: error: expected ';'\nother.c:1:1: error: boom\n"),
            ),
        ):
            r = CSyntaxProvider().validate_semantics(str(f))
        assert r.ok is False
        assert len(r.errors) == 1
        assert "expected ';'" in r.errors[0].message


# ── regex fallbacks (tree-sitter forced unavailable) ───────────────────────

C_SRC_MULTI = """\
#include <stdio.h>

int add(int a, int b) {
  return a + b;
}

struct Point {
  int x;
  int y;
};

enum Color { RED, GREEN };

#define MAX_LEN 100

typedef struct Node Node;

static int counter;
"""


class TestRegexFallbacks:
    @pytest.fixture(autouse=True)
    def _no_ts(self):
        with patch("external_llm.languages.tree_sitter_utils.is_available", return_value=False):
            yield

    def test_find_symbol_in_file_regex_fallback(self):
        r = CSyntaxProvider().find_symbol_in_file("x.c", "add", C_SRC_MULTI)
        assert r is not None and r[0] == 3

    def test_find_symbol_in_file_missing(self):
        assert CSyntaxProvider().find_symbol_in_file("x.c", "zzz", C_SRC_MULTI) is None

    def test_top_level_definitions_regex(self):
        defs = CSyntaxProvider()._find_top_level_definitions_regex(C_SRC_MULTI)
        by_name = {name: kind for name, kind, _, _ in defs}
        assert by_name["add"] == "function"
        assert by_name["Point"] == "struct"
        assert by_name["Color"] == "enum"
        assert by_name["MAX_LEN"] == "macro"
        add = next(d for d in defs if d[0] == "add")
        assert add[2] == 3 and add[3] == 5  # block end line

    def test_macro_at_eof_without_newline(self):
        defs = CSyntaxProvider()._find_top_level_definitions_regex("#define MAX 10")
        macro = next(d for d in defs if d[0] == "MAX")
        assert macro[2] == 1 and macro[3] == 1

    def test_find_top_level_definitions(self):
        defs = CSyntaxProvider().find_top_level_definitions(C_SRC_MULTI)
        by_name = {name: kind for name, kind, _, _ in defs}
        assert by_name["add"] == "function" and by_name["Point"] == "struct"

    def test_find_symbol_body_range(self):
        r = CSyntaxProvider().find_symbol_body_range(C_SRC_MULTI, "add")
        assert r == (3, 5)  # body = from the `{` on the signature line to the closing brace

    def test_find_symbol_body_range_missing(self):
        assert CSyntaxProvider().find_symbol_body_range(C_SRC_MULTI, "zzz") is None


# ── remaining branch coverage: ccdb no-match, .h union failure, TS paths ───


class TestSemanticsEdges:
    @staticmethod
    def _gcc_only():
        return patch(
            "external_llm.languages.c_provider.shutil.which",
            side_effect=lambda c: f"/usr/bin/{c}" if c in ("gcc", "clang") else None,
        )

    def test_ccdb_without_matching_entry_still_compiles(self, tmp_path):
        f = tmp_path / "main.c"
        f.write_text(C_SRC)
        (tmp_path / "compile_commands.json").write_text(
            json.dumps(
                [
                    {"directory": str(tmp_path), "file": "other.c", "arguments": ["gcc", "-c", "other.c"]},
                ]
            )
        )
        with (
            self._gcc_only(),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                return_value=_fake_proc(0),
            ),
        ):
            r = CSyntaxProvider().validate_semantics(str(f))
        assert r.ok is True

    def test_header_union_both_compilers_fail(self, tmp_path):
        f = tmp_path / "x.h"
        f.write_text("namespace N { int v = 1; }\n")
        with (
            patch(
                "external_llm.languages.c_provider.shutil.which",
                side_effect=lambda c: f"/usr/bin/{c}" if c in ("gcc", "g++") else None,
            ),
            patch(
                "external_llm.languages.c_provider.subprocess.run",
                side_effect=[
                    _fake_proc(1, stderr="x.h:1:10: error: expected ';' before '}'"),
                    _fake_proc(1, stderr="x.h:1:10: error: expected ';' before '}'"),
                ],
            ),
        ):
            r = CSyntaxProvider().validate_semantics(str(f))
        assert r.ok is False  # C++ retry also failed → keep the C verdict
        assert len(r.errors) == 1


class TestExtractIncludeFlagsEdges:
    def test_no_directory_returns_flags_unchanged(self):
        entry = {"arguments": ["gcc", "-Iinc"]}
        assert _extract_include_flags(entry) == ["-Iinc"]


class TestTreeSitterPrimaryPaths:
    def _ts_or_skip(self):
        import external_llm.languages.tree_sitter_utils as tsu

        if not tsu.is_available():
            pytest.skip("tree-sitter core not installed")

    def test_ts_available_but_no_match_falls_to_regex(self):
        self._ts_or_skip()
        with patch(
            "external_llm.languages.tree_sitter_utils.find_symbol_range",
            return_value=None,
        ):
            r = CSyntaxProvider().find_symbol_in_file("x.c", "add", C_SRC_MULTI)
        assert r is not None and r[0] == 3

    def test_symbol_body_range_ts_result(self):
        self._ts_or_skip()
        # extract_symbol_body returns None for many simple C snippets — mock it
        # so the tree-sitter primary path (non-None → return directly) is pinned.
        with patch(
            "external_llm.languages.tree_sitter_utils.extract_symbol_body",
            return_value=(2, 4),
        ):
            r = CSyntaxProvider().find_symbol_body_range(C_SRC_MULTI, "add")
        assert r == (2, 4)
