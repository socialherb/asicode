"""RED→GREEN coverage tests for TypeScriptSyntaxProvider (36% → 100%).

Covers every uncovered surface of typescript_provider.py:

- is_genuine_syntax_error band/config-dependent filtering
- _validate_syntax_impl: tempfile failure, TimeoutExpired / generic-exception
  degrade, and the full tsc diagnostic parse (genuine vs config-dependent vs
  type-band errors)
- validate_semantics / validate_semantics_batch / _batch_by_root: not-on-disk
  and no-config skips, project-mode _run_tsc_semantic (config write failure,
  toolchain degrade, rc=0, per-file diagnostic attribution incl. relative-path
  resolution, 2xxx-only band filter, non-numeric code, allow_js)
- get_symbol_patterns kind filtering, get_lint_command, get_test_directory
  (jest/vitest/package.json/convention), get_test_command vitest detection
- regex fallbacks (tree-sitter forced unavailable): top-level definitions,
  class methods (incl. no-brace class and control-keyword filter),
  symbol body range, symbol-in-file
"""

from __future__ import annotations

import builtins
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from external_llm.languages.models import LanguageId
from external_llm.languages.typescript_provider import (
    _TSC_CONFIG_DEPENDENT_1XXX,
    TypeScriptSyntaxProvider,
    is_genuine_syntax_error,
)


def _fake_proc(returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


TS_1005 = "app.ts(10,5): error TS1005: ';' expected."
TS_2304 = "app.ts(3,1): error TS2304: Cannot find name 'foo'."
TS_1259 = "app.ts(1,8): error TS1259: Can only be default-imported using esModuleInterop."


# ── is_genuine_syntax_error ────────────────────────────────────────────────


class TestIsGenuineSyntaxError:
    @pytest.mark.parametrize("code", ["", "abc", "E1005", "TS", "TS12a", "TS999", "TS2000", "TS7006"])
    def test_not_genuine(self, code):
        assert is_genuine_syntax_error(code) is False

    @pytest.mark.parametrize("code", ["TS1000", "TS1005", "TS1109", "TS1999"])
    def test_genuine_parser_band(self, code):
        assert is_genuine_syntax_error(code) is True

    @pytest.mark.parametrize("code", sorted(_TSC_CONFIG_DEPENDENT_1XXX))
    def test_config_dependent_excluded(self, code):
        assert is_genuine_syntax_error(code) is False


# ── capabilities caching ───────────────────────────────────────────────────


class TestCapabilitiesCache:
    def test_capabilities_cached_per_instance(self):
        p = TypeScriptSyntaxProvider()
        caps1 = p.capabilities()
        caps2 = p.capabilities()
        assert caps1 is caps2
        assert caps1.has_syntax_validator is True
        assert isinstance(caps1.has_tree_sitter, bool)


# ── validate_syntax: tsc path ──────────────────────────────────────────────


class TestValidateSyntaxTsc:
    def test_tempfile_failure_returns_ok(self):
        with patch(
            "external_llm.languages.typescript_provider._tempfile_for_content",
            return_value=("", lambda: None),
        ):
            r = TypeScriptSyntaxProvider().validate_syntax("app.ts", "const x = 1;")
        assert r.ok is True

    def test_timeout_falls_back_to_tree_sitter(self):
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            side_effect=subprocess.TimeoutExpired("npx tsc", 30),
        ):
            r = TypeScriptSyntaxProvider().validate_syntax("app.ts", "const x = 1;")
        assert r.ok is True  # tree-sitter accepts valid TS

    def test_generic_exception_falls_back_to_tree_sitter(self):
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            side_effect=OSError("npx exploded"),
        ):
            r = TypeScriptSyntaxProvider().validate_syntax("app.ts", "const x = 1;")
        assert r.ok is True

    def test_nonzero_mixes_genuine_and_ignored_diagnostics(self):
        out = f"garbage line\n{TS_1005}\n{TS_2304}\n{TS_1259}\n"
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            return_value=_fake_proc(1, stdout=out),
        ):
            r = TypeScriptSyntaxProvider().validate_syntax("app.ts", "const x = ;")
        assert r.ok is False
        assert len(r.errors) == 1
        e = r.errors[0]
        assert e.file == "app.ts"
        assert e.line == 10 and e.col == 5
        assert e.message == "TS1005: ';' expected."

    def test_nonzero_only_environment_diagnostics_is_clean(self):
        out = f"{TS_2304}\n{TS_1259}\n"
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            return_value=_fake_proc(1, stdout=out),
        ):
            r = TypeScriptSyntaxProvider().validate_syntax("app.ts", "const x = 1;")
        assert r.ok is True
        assert r.errors == []


# ── validate_semantics_batch / _batch_by_root / _run_tsc_semantic ──────────


class TestSemanticsBatch:
    def test_missing_files_skipped(self, tmp_path):
        p = TypeScriptSyntaxProvider()
        out = p.validate_semantics_batch(["", str(tmp_path / "nope.ts")])
        assert all(not r.checked for r in out.values())
        assert all(r.skip_reason == "the file is not on disk" for r in out.values())

    def test_no_config_skipped(self, tmp_path):
        f = tmp_path / "app.ts"
        f.write_text("const x = 1;")
        r = TypeScriptSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert "no tsconfig.json" in r.skip_reason

    def test_clean_project_mode_run(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        f = tmp_path / "app.ts"
        f.write_text("const x: number = 1;")
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            return_value=_fake_proc(0),
        ) as run:
            out = TypeScriptSyntaxProvider().validate_semantics_batch([str(f)])
        assert out[str(f)].ok is True
        assert out[str(f)].checked is True
        cmd = run.call_args.args[0]
        assert "--project" in cmd
        assert str(tmp_path) in " ".join(cmd)  # temp config lives in the root

    def test_tsc_not_installed_skips(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        f = tmp_path / "app.ts"
        f.write_text("const x = 1;")
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            side_effect=FileNotFoundError("npx"),
        ):
            r = TypeScriptSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert "tsc is not installed" in r.skip_reason

    def test_timeout_skips(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        f = tmp_path / "app.ts"
        f.write_text("const x = 1;")
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            side_effect=subprocess.TimeoutExpired("npx tsc", 30),
        ):
            r = TypeScriptSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert r.skip_reason == "tsc timed out"

    def test_generic_exception_skips(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        f = tmp_path / "app.ts"
        f.write_text("const x = 1;")
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            side_effect=OSError("npx exploded"),
        ):
            r = TypeScriptSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert r.skip_reason == "tsc could not be run"

    def test_temp_config_write_failure_skips(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        f = tmp_path / "app.ts"
        f.write_text("const x = 1;")

        real_open = builtins.open

        def raiser(path, *a, **k):
            if "tsconfig.semcheck" in str(path):
                raise OSError("disk full")
            return real_open(path, *a, **k)

        with patch("builtins.open", raiser):
            r = TypeScriptSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert "temporary tsconfig" in r.skip_reason

    def test_diagnostic_attribution_and_band_filter(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        a = tmp_path / "a.ts"
        b = tmp_path / "b.ts"
        a.write_text("export const x = 1;")
        b.write_text("export const y = 2;")
        # a.ts: relative-path error TS2304 (2xxx → kept, marks failed)
        #       sibling b.ts error TS2304 → dropped (not in batch? it IS in batch)
        #       other.ts error → dropped (not in batch)
        #       TS1005 (1xxx) → dropped by band filter
        #       TScode (non-numeric) → dropped
        out_lines = "\n".join(
            [
                "a.ts(2,5): error TS2304: Cannot find name 'foo'.",
                "b.ts(2,5): error TS2304: Cannot find name 'bar'.",
                "other.ts(1,1): error TS2304: Cannot find name 'nope'.",
                "a.ts(1,1): error TS1005: ';' expected.",
                "a.ts(1,1): error TScode: weird.",
                "",
            ]
        )
        with patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            return_value=_fake_proc(1, stdout=out_lines),
        ):
            out = TypeScriptSyntaxProvider().validate_semantics_batch([str(a), str(b)])
        assert out[str(a)].ok is False
        assert len(out[str(a)].errors) == 1
        assert out[str(a)].errors[0].code == "TS2304"
        assert out[str(b)].ok is False  # its own error was kept
        assert len(out[str(b)].errors) == 1
        # temp config cleaned up
        assert not [p for p in tmp_path.iterdir() if "semcheck" in p.name]

    def test_allow_js_forces_compiler_options(self, tmp_path):
        (tmp_path / "jsconfig.json").write_text("{}")
        f = tmp_path / "app.js"
        f.write_text("const x = 1;")
        p = TypeScriptSyntaxProvider()
        with (
            patch(
                "external_llm.languages.typescript_provider.subprocess.run",
                return_value=_fake_proc(0),
            ) as run,
            patch("json.dump") as dump_mock,
        ):
            out = p._batch_by_root(
                [str(f)],
                language=LanguageId.JAVASCRIPT,
                config_markers=("jsconfig.json",),
                config_for=lambda _root: "jsconfig.json",
                allow_js=True,
            )
        assert out[str(f)].ok is True
        # temp config is unlinked in finally — capture the body via json.dump
        body = dump_mock.call_args.args[0]
        assert body["compilerOptions"] == {"allowJs": True, "checkJs": True}
        assert body["extends"] == "./jsconfig.json"
        assert "--project" in run.call_args.args[0]


# ── symbol patterns / lint / test directory / test command ─────────────────


class TestSymbolPatternsKinds:
    def test_kind_any_has_all(self):
        kinds = {sp.kind for sp in TypeScriptSyntaxProvider().get_symbol_patterns("any")}
        assert kinds == {"function", "class", "interface", "type"}

    def test_kind_function_only(self):
        kinds = {sp.kind for sp in TypeScriptSyntaxProvider().get_symbol_patterns("function")}
        assert kinds == {"function"}

    def test_kind_class_only(self):
        kinds = {sp.kind for sp in TypeScriptSyntaxProvider().get_symbol_patterns("class")}
        assert kinds == {"class"}

    def test_kind_interface_only(self):
        kinds = {sp.kind for sp in TypeScriptSyntaxProvider().get_symbol_patterns("interface")}
        assert kinds == {"interface"}

    def test_kind_type_only(self):
        kinds = {sp.kind for sp in TypeScriptSyntaxProvider().get_symbol_patterns("type")}
        assert kinds == {"type"}


def test_get_lint_command():
    cmd = TypeScriptSyntaxProvider().get_lint_command("app.ts")
    assert cmd == ["npx", "eslint", "--format=json", "app.ts"]


class TestGetTestDirectory:
    def test_empty_repo_returns_none(self, tmp_path):
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None

    def test_jest_config_roots_prefers_test_dir(self, tmp_path):
        (tmp_path / "jest.config.js").write_text(
            "module.exports = { roots: ['<rootDir>/__tests__', '<rootDir>/spec'] };"
        )
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "__tests__"

    def test_jest_config_roots_falls_back_to_first(self, tmp_path):
        (tmp_path / "jest.config.js").write_text("module.exports = { roots: ['<rootDir>/src'] };")
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "src"

    def test_jest_config_inline_dir(self, tmp_path):
        (tmp_path / "jest.config.js").write_text("module.exports = { tests: '__tests__' };")
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "__tests__"

    def test_vitest_config_dir(self, tmp_path):
        (tmp_path / "vitest.config.ts").write_text("export default { test: { dir: 'tests' } };")
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "tests"

    def test_package_json_inline_jest_roots(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"jest": {"roots": ["<rootDir>/test"]}}))
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "test"

    def test_package_json_test_script_roots(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest --roots tests"}}))
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "tests"

    def test_convention_dirs(self, tmp_path):
        (tmp_path / "tests").mkdir()
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "tests"

    def test_convention_order(self, tmp_path):
        (tmp_path / "__tests__").mkdir()
        (tmp_path / "tests").mkdir()
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "__tests__"

    def test_invalid_json_config_skipped(self, tmp_path):
        (tmp_path / "jest.config.json").write_text("{ not json")
        (tmp_path / "spec").mkdir()
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "spec"

    def test_unreadable_config_skipped(self, tmp_path):
        (tmp_path / "jest.config.js").write_text("module.exports = {};")
        real_open = builtins.open

        def raiser(path, *a, **k):
            if str(path).endswith("jest.config.js"):
                raise OSError("permission denied")
            return real_open(path, *a, **k)

        with patch("builtins.open", raiser):
            assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None

    def test_json_config_roots(self, tmp_path):
        (tmp_path / "jest.config.json").write_text(json.dumps({"roots": ["<rootDir>/tests"]}))
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) == "tests"


class TestGetTestCommand:
    def test_no_package_json_defaults_jest(self, tmp_path):
        assert TypeScriptSyntaxProvider().get_test_command(str(tmp_path)) == [
            "npx",
            "jest",
            "--passWithNoTests",
        ]

    def test_vitest_dependency_detected(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"devDependencies": {"vitest": "^1.0.0"}}))
        assert TypeScriptSyntaxProvider().get_test_command(str(tmp_path)) == [
            "npx",
            "vitest",
            "--passWithNoTests",
        ]

    def test_test_args_appended(self, tmp_path):
        assert TypeScriptSyntaxProvider().get_test_command(str(tmp_path), ["x.test.ts"]) == [
            "npx",
            "jest",
            "--passWithNoTests",
            "x.test.ts",
        ]

    def test_unparseable_package_json_defaults_jest(self, tmp_path):
        (tmp_path / "package.json").write_text("{ nope")
        assert TypeScriptSyntaxProvider().get_test_command(str(tmp_path)) == [
            "npx",
            "jest",
            "--passWithNoTests",
        ]


# ── regex fallbacks (tree-sitter forced unavailable) ───────────────────────

TS_SRC = """\
export function foo(a: number): number {
  return a + 1;
}
export class Bar {
  run() { return 1; }
  iffy() { if (x) { return 1; } }
}
interface Baz { x: number; }
type Qux = string;
"""


class TestRegexFallbacks:
    @pytest.fixture(autouse=True)
    def _no_ts(self):
        with patch("external_llm.languages.tree_sitter_utils.is_available", return_value=False):
            yield

    def test_find_top_level_definitions(self):
        defs = TypeScriptSyntaxProvider().find_top_level_definitions(TS_SRC)
        by_name = {name: kind for name, kind, _, _ in defs}
        assert by_name == {"foo": "function", "Bar": "class", "Baz": "interface", "Qux": "type"}
        foo = next(d for d in defs if d[0] == "foo")
        assert foo[2] == 1 and foo[3] == 3  # start/end lines

    def test_type_alias_same_line_gets_end_floor(self):
        # `type Foo = string;` on one line → end_line <= start_line → +1 floor
        p = TypeScriptSyntaxProvider()
        defs = p._find_top_level_definitions_regex("type Foo = string;\n")
        foo = next(d for d in defs if d[0] == "Foo")
        assert foo[3] == foo[2] + 1

    def test_type_alias_without_semicolon_ends_at_eof(self):
        p = TypeScriptSyntaxProvider()
        defs = p._find_top_level_definitions_regex("type Foo = string")
        foo = next(d for d in defs if d[0] == "Foo")
        assert foo[3] == 2  # end_line == start_line -> +1 floor

    def test_find_class_methods(self):
        methods = TypeScriptSyntaxProvider().find_class_methods(TS_SRC, "Bar")
        names = [m[0] for m in methods]
        assert names == ["run", "iffy"]

    def test_class_without_brace_found_no_methods(self):
        # `class Foo extends` at EOF: pattern matches, no `{` anywhere
        p = TypeScriptSyntaxProvider()
        assert p._find_class_methods_regex("class Foo extends", "Foo") == []

    def test_control_keywords_filtered_from_methods(self):
        p = TypeScriptSyntaxProvider()
        methods = p._find_class_methods_regex(
            "class C {\n  if (x) { return; }\n  real() {}\n}",
            "C",
        )
        assert [m[0] for m in methods] == ["real"]

    def test_find_symbol_body_range(self):
        r = TypeScriptSyntaxProvider().find_symbol_body_range(TS_SRC, "foo")
        assert r == (1, 3)  # `{` sits on the signature line

    def test_find_symbol_in_file_regex_fallback(self):
        r = TypeScriptSyntaxProvider().find_symbol_in_file("app.ts", "foo", TS_SRC)
        assert r == (1, 3)

    def test_find_symbol_in_file_missing(self):
        assert TypeScriptSyntaxProvider().find_symbol_in_file("app.ts", "zzz", TS_SRC) is None


# ── remaining branch coverage: test-directory / test-command edge paths ─────


class TestGetTestDirectoryEdges:
    def test_testmatch_only_falls_through(self, tmp_path):
        (tmp_path / "jest.config.js").write_text("module.exports = { testMatch: ['<rootDir>/__tests__/**/*.ts'] };")
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None

    def test_first_root_empty_strip_skips_return(self, tmp_path):
        (tmp_path / "jest.config.js").write_text("module.exports = { roots: ['<rootDir>'] };")
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None

    def test_empty_roots_skips_dir(self, tmp_path):
        (tmp_path / "jest.config.js").write_text("module.exports = { roots: [] };")
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None

    def test_vitest_dir_without_test_word_falls_through(self, tmp_path):
        (tmp_path / "vitest.config.ts").write_text("export default { test: { dir: 'src' } };")
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None

    def test_package_json_jest_roots_without_test_word(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"jest": {"roots": ["<rootDir>/spec"]}}))
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None

    def test_test_script_without_hint(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None

    def test_test_script_hint_without_test_word(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "mocha spec/"}}))
        assert TypeScriptSyntaxProvider().get_test_directory(str(tmp_path)) is None


class TestGetTestCommandEdges:
    def test_package_json_without_vitest_defaults_jest(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"jest": "^29.0.0"}}))
        assert TypeScriptSyntaxProvider().get_test_command(str(tmp_path)) == [
            "npx",
            "jest",
            "--passWithNoTests",
        ]


# ── multi-line type alias: end_line > start_line (no floor applied) ─────────


class TestRegexFallbackEdges:
    @pytest.fixture(autouse=True)
    def _no_ts(self):
        with patch("external_llm.languages.tree_sitter_utils.is_available", return_value=False):
            yield

    def test_multiline_type_alias_keeps_real_end(self):
        # `;` on line 3: the old code compared a 0-based end_line
        # (line_index_at_offset) against a 1-based start_line
        # (line_at_offset), under-reporting end_line by one (2 instead of 3).
        p = TypeScriptSyntaxProvider()
        defs = p._find_top_level_definitions_regex("type Foo =\n    Bar\n    ;\n")
        foo = next(d for d in defs if d[0] == "Foo")
        assert foo[2] == 1 and foo[3] == 3


# ── tree-sitter primary path (result non-empty → return directly) ───────────


class TestTreeSitterPrimaryPath:
    def _ts_or_skip(self):
        import external_llm.languages.tree_sitter_utils as tsu

        if not tsu.is_available():
            pytest.skip("tree-sitter core not installed")

    def test_top_level_definitions_ts_result(self):
        self._ts_or_skip()
        defs = TypeScriptSyntaxProvider().find_top_level_definitions(TS_SRC)
        assert any(d[0] == "foo" and d[1] == "function" for d in defs)

    def test_class_methods_ts_result(self):
        self._ts_or_skip()
        methods = TypeScriptSyntaxProvider().find_class_methods(TS_SRC, "Bar")
        assert any(m[0] == "run" for m in methods)

    def test_symbol_body_range_ts_result(self):
        self._ts_or_skip()
        r = TypeScriptSyntaxProvider().find_symbol_body_range(TS_SRC, "foo")
        assert r is not None and r[0] >= 1
