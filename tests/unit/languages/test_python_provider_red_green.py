"""RED→GREEN coverage tests for PythonSyntaxProvider (30% → 100%).

Covers every uncovered surface of python_provider.py:

- capabilities: tree-sitter-unavailable branch (replace() not applied)
- _validate_syntax_impl: compile() raising SyntaxError / ValueError (ast
  parse passes but the stricter compile pass fails)
- validate_semantics / validate_semantics_batch: not-on-disk skip,
  per-root grouping, _run_pyright degrade paths (FileNotFoundError /
  TimeoutExpired / generic exception / non-JSON output) and the full
  diagnostic attribution (absolute-path index, file-less single-file
  attribution, file-less multi-file drop, severity handling, 0-indexed
  line/col, rule codes, malformed diagnostics)
- find_symbol_in_file fallback chain: tree-sitter raise → LibCST,
  LibCST raise/None → stdlib ast, qualified names, missing end_lineno
- find_top_level_definitions / find_class_methods / find_all_class_methods
  / find_symbol_body_range incl. SyntaxError and end_lineno-less trees
"""

from __future__ import annotations

import ast
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from external_llm.languages.python_provider import PythonSyntaxProvider


def _fake_proc(returncode, stdout=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def _empty_pyright(returncode=0):
    return _fake_proc(returncode, json.dumps({"generalDiagnostics": []}))


# ── capabilities ───────────────────────────────────────────────────────────


class TestCapabilities:
    def test_tree_sitter_unavailable_branch(self):
        with patch(
            "external_llm.languages.python_provider._tree_sitter_available",
            return_value=False,
        ):
            caps = PythonSyntaxProvider().capabilities()
        assert caps.has_tree_sitter is False

    def test_tree_sitter_available_branch(self):
        with patch(
            "external_llm.languages.python_provider._tree_sitter_available",
            return_value=True,
        ):
            caps = PythonSyntaxProvider().capabilities()
        assert caps.has_tree_sitter is True
        assert caps.has_ast_parser is True


# ── syntax validation: compile() strict pass ───────────────────────────────


class TestValidateSyntaxCompileErrors:
    @staticmethod
    def _compile_raiser(exc):
        """Raise *exc* only for the provider's own compile() call.

        ast.parse (line 74) also routes through builtins.compile with a
        flags argument; the provider's call is exactly ``compile(content,
        file_path, "exec")`` — 3 positionals, no keywords.
        """
        real_compile = builtins_compile()

        def raiser(*args, **kwargs):
            if len(args) == 3 and not kwargs:
                raise exc
            return real_compile(*args, **kwargs)

        return raiser

    def test_compile_syntax_error_appended(self, tmp_path):
        with patch(
            "builtins.compile",
            self._compile_raiser(
                SyntaxError("synthetic", (str(tmp_path / "x.py"), 3, 4, "line")),
            ),
        ):
            r = PythonSyntaxProvider().validate_syntax("x.py", "x = 1\n")
        assert r.ok is False
        assert len(r.errors) == 1
        assert r.errors[0].message == "Compile error: synthetic"
        assert r.errors[0].line == 3 and r.errors[0].col == 4

    def test_compile_value_error_appended(self, tmp_path):
        with patch("builtins.compile", self._compile_raiser(ValueError("null bytes"))):
            r = PythonSyntaxProvider().validate_syntax("x.py", "x = 1\n")
        assert r.ok is False
        assert r.errors[0].message == "Compile error: null bytes"
        assert r.errors[0].line == 0 and r.errors[0].col == 0


def builtins_compile():
    import builtins

    return builtins.compile


# ── semantic validation: batch grouping + pyright runs ─────────────────────


class TestSemanticsBatch:
    def test_missing_files_skipped(self, tmp_path):
        out = PythonSyntaxProvider().validate_semantics_batch(["", str(tmp_path / "nope.py")])
        assert all(not r.checked for r in out.values())
        assert all(r.skip_reason == "the file is not on disk" for r in out.values())

    def test_clean_run_single(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        with patch(
            "subprocess.run",
            return_value=_empty_pyright(),
        ) as run:
            r = PythonSyntaxProvider().validate_semantics(str(f))
        assert r.ok is True and r.checked is True
        assert "pyright" in run.call_args.args[0][0]

    def test_grouped_by_project_root(self, tmp_path):
        a = tmp_path / "pkg_a" / "a.py"
        b = tmp_path / "pkg_b" / "b.py"
        a.parent.mkdir()
        b.parent.mkdir()
        a.write_text("x = 1\n")
        b.write_text("y = 2\n")
        with patch(
            "subprocess.run",
            return_value=_empty_pyright(),
        ) as run:
            out = PythonSyntaxProvider().validate_semantics_batch([str(a), str(b)])
        assert out[str(a)].ok is True and out[str(b)].ok is True
        assert run.call_count == 2  # one pyright run per project root

    def test_pyright_not_installed_skips(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        with patch(
            "subprocess.run",
            side_effect=FileNotFoundError("pyright"),
        ):
            r = PythonSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert r.skip_reason == "pyright is not installed"

    def test_pyright_timeout_skips(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("pyright", 30),
        ):
            r = PythonSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert r.skip_reason == "pyright timed out"

    def test_pyright_generic_failure_skips(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        with patch(
            "subprocess.run",
            side_effect=OSError("pyright exploded"),
        ):
            r = PythonSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert r.skip_reason == "pyright could not be run"

    def test_non_json_output_skips(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        with patch(
            "subprocess.run",
            return_value=_fake_proc(1, "pyright crashed with a traceback"),
        ):
            r = PythonSyntaxProvider().validate_semantics(str(f))
        assert r.checked is False
        assert r.skip_reason == "pyright produced no readable output"

    def test_diagnostics_attribution_and_severity(self, tmp_path):
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("x = 1\n")
        b.write_text("y = 2\n")
        payload = {
            "generalDiagnostics": [
                {
                    "severity": "error",
                    "file": str(a),
                    "range": {"start": {"line": 0, "character": 2}},
                    "message": "Undefined variable 'z'",
                    "rule": "reportUndefinedVariable",
                },
                {
                    "severity": "warning",
                    "file": str(a),
                    "range": {"start": {"line": 1, "character": 0}},
                    "message": "Unused import",
                    "rule": "reportUnusedImport",
                },
                {
                    "severity": "error",
                    "file": str(tmp_path / "sibling.py"),  # not in batch → dropped
                    "range": {"start": {"line": 0, "character": 0}},
                    "message": "noise",
                },
            ],
        }
        with patch(
            "subprocess.run",
            return_value=_fake_proc(1, json.dumps(payload)),
        ):
            out = PythonSyntaxProvider().validate_semantics_batch([str(a), str(b)])
        ra, rb = out[str(a)], out[str(b)]
        assert ra.ok is False  # error diag
        assert len(ra.errors) == 2
        err = ra.errors[0]
        assert err.line == 1 and err.col == 3  # pyright is 0-indexed → +1
        assert err.code == "reportUndefinedVariable"
        assert err.severity == "error"
        assert ra.errors[1].severity == "warning"
        assert rb.ok is True and rb.errors == []  # clean file, no noise

    def test_file_less_diagnostic_single_file_attributed(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        payload = {
            "generalDiagnostics": [
                {"severity": "error", "message": "config problem"},
            ]
        }
        with patch(
            "subprocess.run",
            return_value=_fake_proc(1, json.dumps(payload)),
        ):
            r = PythonSyntaxProvider().validate_semantics(str(f))
        assert r.ok is False
        assert len(r.errors) == 1

    def test_file_less_diagnostic_multi_file_dropped(self, tmp_path):
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("x = 1\n")
        b.write_text("y = 2\n")
        payload = {
            "generalDiagnostics": [
                {"severity": "error", "message": "config problem"},
            ]
        }
        with patch(
            "subprocess.run",
            return_value=_fake_proc(1, json.dumps(payload)),
        ):
            out = PythonSyntaxProvider().validate_semantics_batch([str(a), str(b)])
        assert out[str(a)].ok is True and out[str(b)].ok is True

    def test_malformed_diagnostics_suppressed(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        payload = {"generalDiagnostics": [42, None, {"severity": None, "message": "x"}]}
        with patch(
            "subprocess.run",
            return_value=_fake_proc(1, json.dumps(payload)),
        ):
            r = PythonSyntaxProvider().validate_semantics(str(f))
        # 42 → AttributeError suppressed; None → suppressed; severity None
        # defaults to "error" with an empty range
        assert r.ok is False
        assert len(r.errors) == 1


# ── find_symbol_in_file fallback chain ─────────────────────────────────────

PY_SRC = """\
class Foo:
    def bar(self):
        return 1

def top():
    pass
"""


def _no_ts():
    return patch(
        "external_llm.languages.python_provider._tree_sitter_available",
        return_value=False,
    )


class TestFindSymbolFallbackChain:
    def test_libcst_primary_when_ts_unavailable(self):
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=(7, 9),
            ),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "top", PY_SRC)
        assert r == (7, 9)

    def test_ts_raise_falls_to_libcst(self):
        with (
            patch(
                "external_llm.languages.python_provider._tree_sitter_available",
                return_value=True,
            ),
            patch(
                "external_llm.languages.tree_sitter_utils.find_symbol_range",
                side_effect=RuntimeError("parser exploded"),
            ),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=(7, 9),
            ),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "top", PY_SRC)
        assert r == (7, 9)

    def test_libcst_raise_falls_to_ast(self):
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                side_effect=RuntimeError("libcst exploded"),
            ),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "top", PY_SRC)
        assert r == (5, 6)

    def test_libcst_none_falls_to_ast(self):
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=None,
            ),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "top", PY_SRC)
        assert r == (5, 6)

    def test_ast_qualified_name(self):
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=None,
            ),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "Foo.bar", PY_SRC)
        assert r == (2, 3)

    def test_ast_qualified_name_missing_method(self):
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=None,
            ),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "Foo.zzz", PY_SRC)
        assert r is None

    def test_ast_parse_failure_returns_none(self):
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=None,
            ),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "top", "def broken(:\n")
        assert r is None


def _fn_node(name: str, lineno: int | None = None) -> ast.FunctionDef:
    fn = ast.FunctionDef(
        name=name,
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Pass()],
        decorator_list=[],
    )
    # real parser output always carries lineno; fake trees must too, since
    # find_symbol_body_range computes `child.lineno + 1` before checking
    # end_lineno
    fn.lineno = lineno if lineno is not None else 3
    return fn


class TestFindSymbolEndLinenoMissing:
    def test_qualified_method_without_end_lineno_returns_none(self):
        cls = ast.ClassDef(name="Foo", bases=[], keywords=[], body=[_fn_node("bar")], decorator_list=[])
        fake = ast.Module(body=[cls], type_ignores=[])
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=None,
            ),
            patch("external_llm.languages.python_provider.ast.parse", return_value=fake),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "Foo.bar", "x")
        assert r is None

    def test_simple_symbol_without_end_lineno_returns_none(self):
        fake = ast.Module(body=[_fn_node("top")], type_ignores=[])
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=None,
            ),
            patch("external_llm.languages.python_provider.ast.parse", return_value=fake),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "top", "x")
        assert r is None


# ── structural queries ─────────────────────────────────────────────────────


class TestStructuralQueries:
    def test_top_level_definitions(self):
        defs = PythonSyntaxProvider().find_top_level_definitions(
            "def f():\n    pass\n\nclass C:\n    pass\n\nasync def g():\n    pass\n"
        )
        assert defs == [
            ("f", "function", 1, 2),
            ("C", "class", 4, 5),
            ("g", "function", 7, 8),
        ]

    def test_top_level_definitions_syntax_error_empty(self):
        assert PythonSyntaxProvider().find_top_level_definitions("def broken(:\n") == []

    def test_find_class_methods(self):
        p = PythonSyntaxProvider()
        assert p.find_class_methods(PY_SRC, "Foo") == [("bar", 2, 3)]
        assert p.find_class_methods(PY_SRC, "Missing") == []

    def test_find_all_class_methods(self):
        out = PythonSyntaxProvider().find_all_class_methods("class A:\n    def a1(self): pass\n\nclass B:\n    x = 1\n")
        assert out == {"A": [("a1", 2, 2)]}  # B has no methods → absent

    def test_find_all_class_methods_syntax_error_empty(self):
        assert PythonSyntaxProvider().find_all_class_methods("def broken(:\n") == {}

    def test_symbol_body_range_qualified(self):
        assert PythonSyntaxProvider().find_symbol_body_range(PY_SRC, "Foo.bar") == (3, 3)

    def test_symbol_body_range_simple(self):
        assert PythonSyntaxProvider().find_symbol_body_range(PY_SRC, "top") == (6, 6)

    def test_symbol_body_range_missing(self):
        assert PythonSyntaxProvider().find_symbol_body_range(PY_SRC, "zzz") is None

    def test_symbol_body_range_syntax_error_none(self):
        assert PythonSyntaxProvider().find_symbol_body_range("def broken(:\n", "x") is None

    def test_symbol_body_range_qualified_class_missing(self):
        assert PythonSyntaxProvider().find_symbol_body_range(PY_SRC, "Nope.x") is None

    def test_symbol_body_range_without_end_lineno_none(self):
        fake = ast.Module(body=[_fn_node("top")], type_ignores=[])
        with patch("external_llm.languages.python_provider.ast.parse", return_value=fake):
            assert PythonSyntaxProvider().find_symbol_body_range("x", "top") is None

    def test_symbol_body_range_qualified_without_end_lineno_none(self):
        cls = ast.ClassDef(name="Foo", bases=[], keywords=[], body=[_fn_node("bar")], decorator_list=[])
        fake = ast.Module(body=[cls], type_ignores=[])
        with patch("external_llm.languages.python_provider.ast.parse", return_value=fake):
            assert PythonSyntaxProvider().find_symbol_body_range("x", "Foo.bar") is None


# ── remaining branch coverage: loop-exhaust / end_lineno-less tree paths ───


class TestStructuralQueryEdges:
    def test_qualified_name_class_missing_loop_exhausts(self):
        with (
            _no_ts(),
            patch(
                "external_llm.languages.libcst_utils.find_symbol_range",
                return_value=None,
            ),
        ):
            r = PythonSyntaxProvider().find_symbol_in_file("x.py", "Nope.bar", PY_SRC)
        assert r is None

    def test_top_level_definitions_end_lineno_less_tree(self):
        assign = ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Constant(1),
        )
        cls = ast.ClassDef(name="C", bases=[], keywords=[], body=[], decorator_list=[])
        fake = ast.Module(body=[_fn_node("f"), assign, cls], type_ignores=[])
        with patch("external_llm.languages.python_provider.ast.parse", return_value=fake):
            assert PythonSyntaxProvider().find_top_level_definitions("x") == []

    def test_find_all_class_methods_end_lineno_less_method(self):
        cls = ast.ClassDef(name="Foo", bases=[], keywords=[], body=[_fn_node("a")], decorator_list=[])
        fake = ast.Module(body=[cls], type_ignores=[])
        with patch("external_llm.languages.python_provider.ast.parse", return_value=fake):
            assert PythonSyntaxProvider().find_all_class_methods("x") == {}

    def test_symbol_body_range_non_method_child_skipped(self):
        content = "class Foo:\n    x = 1\n\n    def bar(self):\n        pass\n"
        assert PythonSyntaxProvider().find_symbol_body_range(content, "Foo.bar") == (5, 5)

    def test_symbol_body_range_method_missing_breaks(self):
        assert PythonSyntaxProvider().find_symbol_body_range(PY_SRC, "Foo.zzz") is None
