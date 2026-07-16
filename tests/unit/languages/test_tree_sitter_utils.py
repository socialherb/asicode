"""Tests for tree-sitter utility functions (has_error, find_error_nodes, etc.)."""

from external_llm.languages.models import _EXT_MAP, _LANGUAGE_EXTENSION_GROUPS
from external_llm.languages.tree_sitter_utils import (
    _BASE_KIND_MAP,
    _CALL_QUERIES,
    _CSS_KIND_MAP,
    _EXT_TO_GRAMMAR_KEY,
    _IMPORT_QUERIES,
    _REFERENCE_QUERIES,
    _SYMBOL_QUERIES,
    _WALK_KIND_MAP,
    _node_kind_from_type,
    count_class_members,
    count_method_statements,
    extract_class_methods,
    extract_import_names,
    extract_imports,
    find_error_nodes,
    get_class_member_names,
    get_parser,
    has_error,
)


class TestHasError:
    """Verify has_error fast-path + full-DFS for various error types."""

    def test_valid_code_returns_false(self):
        """Valid code: fast-path triggers (has_error=False), returns False."""
        valid_c = "int main() { return 0; }"
        result = has_error(valid_c, "c")
        assert result is False

    def test_missing_semicolon_returns_true(self):
        """MISSING-only (no ERROR): has_error still returns True."""
        bad_c = "int main() { return 0 }"
        result = has_error(bad_c, "c")
        assert result is True

    def test_syntax_error_returns_true(self):
        """ERROR node present: has_error returns True."""
        bad_c = "int main() { return 0 ++ }"
        result = has_error(bad_c, "c")
        assert result is True

    def test_unbalanced_braces_returns_true(self):
        """Unmatched braces produce ERROR nodes."""
        bad_c = "int main() { return 0; "
        result = has_error(bad_c, "c")
        assert result is True

    def test_valid_java_returns_false(self):
        """Valid Java code: fast-path triggers."""
        valid = "class Foo { int x = 5; }"
        result = has_error(valid, "java")
        assert result is False

    def test_missing_semicolon_java_returns_true(self):
        """Java missing semicolon: MISSING-only, has_error=true."""
        bad = "class Foo { int x = 5 }"
        result = has_error(bad, "java")
        assert result is True


class TestFindErrorNodes:
    """Verify find_error_nodes returns correct structure."""

    def test_valid_code_returns_empty(self):
        """Valid code: returns empty list (fast-path)."""
        valid_c = "int main() { return 0; }"
        result = find_error_nodes(valid_c, "c")
        assert result == []

    def test_missing_semicolon_returns_error_node(self):
        """MISSING-only: returns SyntaxErrorNode with kind='MISSING'."""
        bad_c = "int main() { return 0 }"
        result = find_error_nodes(bad_c, "c")
        assert len(result) >= 1
        assert result[0].kind == "MISSING"
        assert result[0].missing_token == ";"

    def test_detects_syntax_error(self):
        """ERROR node: returns SyntaxErrorNode with kind='ERROR'."""
        bad_c = "int main() { return 0 ++ }"
        result = find_error_nodes(bad_c, "c")
        assert len(result) >= 1

    def test_missing_semicolon_java(self):
        """Java MISSING-only: kind='MISSING', missing_token=';'."""
        bad = "class Foo { int x = 5 }"
        result = find_error_nodes(bad, "java")
        assert len(result) >= 1
        assert result[0].kind == "MISSING"
        assert result[0].missing_token == ";"

    def test_returns_none_when_tree_sitter_unavailable(self):
        """Unsupported language returns None."""
        result = find_error_nodes("hello world", "nonexistent_lang_xyz")
        assert result is None


class TestExtractImports:
    """Regression coverage for per-language import-extraction queries.

    The ruby/lua/bash/scala queries were added without unit tests, leaving them
    unprotected against silent regressions. These pin each query's contract and
    guard the keyword-strip scope (Scala-only) fix.
    """

    def test_scala_import_module_path(self):
        # @source captures the whole import_declaration incl. the keyword;
        # extract_imports strips the keyword AND any namespace selectors/wildcards.
        assert extract_imports("import scala.collection.mutable", "scala") == [
            ("scala.collection.mutable", 1)
        ]
        # Multi-import: strip {c, d} selectors
        assert extract_imports("import a.b.{c, d}", "scala") == [
            ("a.b", 1)
        ]
        # Wildcard: strip ._ suffix
        assert extract_imports("import a.b._", "scala") == [
            ("a.b", 1)
        ]
        # Nested multi-import: only strip the final selector block
        assert extract_imports("import a.b.c.{d => e, f => g}", "scala") == [
            ("a.b.c", 1)
        ]
        # Scala 3 wildcard: strip .* suffix
        assert extract_imports("import a.b.*", "scala") == [
            ("a.b", 1)
        ]

    def test_ruby_require_and_require_relative(self):
        src = 'require "json"\nrequire_relative "./helper"\n'
        result = set(extract_imports(src, "ruby"))
        assert ("json", 1) in result
        assert ("./helper", 2) in result

    def test_lua_require_paren_and_bare(self):
        src = 'require("json")\nlocal m = require "module"\n'
        result = set(extract_imports(src, "lua"))
        assert ("json", 1) in result
        assert ("module", 2) in result

    def test_bash_source_and_dot(self):
        src = "source lib.sh\n. helper.sh\n"
        result = set(extract_imports(src, "bash"))
        assert ("lib.sh", 1) in result
        assert ("helper.sh", 2) in result

    def test_python_dotted_and_from(self):
        # Regression: @source is the dotted_name child, never includes keyword.
        assert ("os.path", 1) in set(
            extract_imports("import os.path", "python")
        )
        assert ("json", 1) in set(
            extract_imports("from json import loads", "python")
        )

    def test_c_sharp_using_keyword_not_stripped(self):
        # The keyword strip is scoped to Scala only (`if language == "scala"`),
        # so C# never enters the strip path. This pins that C#'s @source
        # capture yields the bare dotted path without the `using` keyword.
        assert extract_imports("using System.IO;", "c_sharp") == [
            ("System.IO", 1)
        ]

    def test_lua_path_starting_with_import_keyword_is_preserved(self):
        # Pins the Scala-only scope fix: a pathological lua module path that
        # literally starts with "import " must NOT be sliced by the regex.
        assert extract_imports('require("import foo")', "lua") == [
            ("import foo", 1)
        ]


class TestGrammarMapConsistency:
    """SSOT drift guard: _EXT_TO_GRAMMAR_KEY must cover all AST languages from _EXT_MAP."""

    def test_grammar_map_covers_all_ast_extensions(self):
        """Every _EXT_MAP extension with full AST support must appear in _EXT_TO_GRAMMAR_KEY."""
        non_ast = {"JSON", "CSS", "HTML"}
        missing = {
            ext for ext, lang in _EXT_MAP.items()
            if lang not in non_ast and ext not in _EXT_TO_GRAMMAR_KEY
        }
        assert not missing, f"grammar map drift (exts in _EXT_MAP but not in _EXT_TO_GRAMMAR_KEY): {sorted(missing)}"

    def test_grammar_keys_resolve_to_working_parsers(self):
        """Every entry in _EXT_TO_GRAMMAR_KEY must produce a working tree-sitter parser."""
        failures = []
        for ext, key in _EXT_TO_GRAMMAR_KEY.items():
            parser = get_parser(key)
            if parser is None:
                failures.append(f"{ext} -> {key}")
        assert not failures, f"unresolvable grammar keys: {failures}"

    def test_family_groups_match_grammar_map(self):
        """SSOT 3-way: _EXT_MAP(AST) == _EXT_TO_GRAMMAR_KEY == _LANGUAGE_EXTENSION_GROUPS membership.

        All three tables derive from the same concept (file extension → language), but
        no single table is a superset of the others — _EXT_MAP includes non-AST languages
        (JSON/CSS/HTML), groups include aliases (.scala, .sc), and grammar key is the
        narrowest (only AST languages).  The invariant is that every AST extension in
        _EXT_MAP must appear in ALL three.
        """
        non_ast = {"JSON", "CSS", "HTML"}
        ast_ext_map = {e for e, lang in _EXT_MAP.items() if lang not in non_ast}
        grammar_exts = set(_EXT_TO_GRAMMAR_KEY)
        family_exts = {e for g in _LANGUAGE_EXTENSION_GROUPS for e in g}
        assert ast_ext_map == grammar_exts, (
            f"drift between _EXT_MAP(AST) and _EXT_TO_GRAMMAR_KEY: "
            f"_EXT_MAP only: {sorted(ast_ext_map - grammar_exts)}, "
            f"_EXT_TO_GRAMMAR_KEY only: {sorted(grammar_exts - ast_ext_map)}"
        )
        assert ast_ext_map == family_exts, (
            f"drift between _EXT_MAP(AST) and _LANGUAGE_EXTENSION_GROUPS: "
            f"_EXT_MAP only: {sorted(ast_ext_map - family_exts)}, "
            f"groups only: {sorted(family_exts - ast_ext_map)}"
        )


class TestKindMapConsistency:
    """SSOT drift guard for node-type → kind mapping.

    The manual-walk path (``_node_kind`` via ``_WALK_KIND_MAP``) and the query
    path (``_node_kind_from_type`` via ``_BASE_KIND_MAP``) must agree on every
    shared node type.  They previously drifted: ``lexical_declaration`` mapped
    to "function" on the walk path but "assignment" on the query path, and
    ``object_declaration`` was missing from the walk path entirely (falling
    through to "function" instead of "class").
    """

    def test_walk_and_query_agree_on_common_keys(self):
        """Every node type in both maps must map to the same kind."""
        base = _BASE_KIND_MAP
        walk = _WALK_KIND_MAP
        disagreements = {
            k: (walk[k], base[k])
            for k in base.keys() & walk.keys()
            if walk[k] != base[k]
        }
        assert not disagreements, (
            f"kind-map drift (node_type: walk_kind != query_kind): {disagreements}"
        )

    def test_walk_map_is_base_plus_css(self):
        """Walk map is exactly the base SSOT overlaid with CSS-only entries."""
        assert _WALK_KIND_MAP == {**_BASE_KIND_MAP, **_CSS_KIND_MAP}

    def test_css_entries_are_walk_only(self):
        """CSS node types must not leak into the query path (CSS has no query)."""
        for css_type in _CSS_KIND_MAP:
            assert css_type not in _BASE_KIND_MAP

    def test_query_path_uses_base_map(self):
        """_node_kind_from_type resolves via the shared base map."""
        assert _node_kind_from_type("lexical_declaration") == "assignment"
        assert _node_kind_from_type("object_declaration") == "class"
        # CSS types are absent from the query path → generic "function" default.
        assert _node_kind_from_type("class_selector") == "function"


class TestIterativeWalkRecursionSafety:
    """Regression guard: tree-walk closures must be iterative (explicit stack).

    Six closures previously recursed over the parsed tree and blew the default
    recursion limit (1000) on deeply nested / machine-generated inputs (e.g.
    bundled JS). ``structural_hash`` was already iterative; the others were
    converted to the same explicit-stack DFS pattern.

    INPUT SHAPE MATTERS — the deep nesting must be a PRECEDING SIBLING of the
    search target, never nested *inside* the matched target's body. Rationale:
    these closures are first-match short-circuit walks; once the target is found
    they return immediately without descending into its body. Placing the depth
    chain inside the matched body yields a vacuous test that passes even on the
    old recursive code (the deep subtree is never visited). The sibling-first
    shape forces a full descent through the depth chain before the target is
    reached, so reverting any closure to recursion makes this test go red.

    ``extract_imports`` is a query-path entry point that never reaches the
    recursive walk, so the imports closure is probed directly via
    ``extract_import_names``.
    """

    DEPTH = 1200  # exceeds Python's default recursion limit (1000)

    def _deep_nested_expr(self) -> str:
        """Bare depth chain — no import/class, forces a full walk."""
        return "let x = " + "(" * self.DEPTH + "1" + ")" * self.DEPTH + ";"

    def _deep_sibling_then_class(self) -> str:
        """Depth chain as a preceding sibling of ``class C { m() {} }``.

        DFS visits the chain first and descends to full depth before reaching
        the class, so every closure below walks the entire depth chain.
        """
        chain = "let a = " + "(" * self.DEPTH + "1" + ")" * self.DEPTH + ";"
        return chain + " class C { m() {} }"

    def test_extract_import_names_no_recursion_error(self):
        # Probe the walk-based extractor directly; extract_imports() takes the
        # query path and never reaches the recursive closure.
        assert extract_import_names(self._deep_nested_expr(), "javascript") == []

    def test_count_method_statements_no_recursion_error(self):
        # m() has an empty body -> 0 statements; full descent still happens.
        assert count_method_statements(
            self._deep_sibling_then_class(), "m", "javascript",
        ) == 0

    def test_get_class_member_names_no_recursion_error(self):
        names = get_class_member_names(
            self._deep_sibling_then_class(), "C", "javascript",
        )
        assert names == ({"m"}, set())

    def test_count_class_members_no_recursion_error(self):
        assert count_class_members(
            self._deep_sibling_then_class(), "C", "javascript",
        ) == (1, 0)

    def test_extract_class_methods_no_recursion_error(self):
        methods = extract_class_methods(
            self._deep_sibling_then_class(), "C", "javascript",
        )
        assert methods == [("m", 1, 1)]


class TestQueryTableParity:
    """SSOT drift guard: the four tree-sitter query tables must stay in parity.

    A full-AST language is defined as one with ALL of symbol/call/import/
    reference queries populated. Two silent failure modes arise when one table
    drifts:
      (1) a language present in ``_SYMBOL_QUERIES`` but absent from
          ``_CALL_QUERIES``/``_IMPORT_QUERIES`` silently degrades caller/import
          search to a broad fallback extension union that excludes its own
          language family;
      (2) ``_get_language_group`` returns -1 for a language whose queries are
          incomplete, bypassing the cross-language resolution guard.
    Mirrors the extension-table 3-way pin (test_family_groups_match_grammar_map).
    """

    def test_four_query_tables_share_identical_language_keys(self):
        tables = {
            "symbol": _SYMBOL_QUERIES,
            "call": _CALL_QUERIES,
            "import": _IMPORT_QUERIES,
            "reference": _REFERENCE_QUERIES,
        }
        keysets = {name: set(t.keys()) for name, t in tables.items()}
        ref = keysets["symbol"]
        drift = {
            name: sorted(keys ^ ref)
            for name, keys in keysets.items()
            if keys != ref
        }
        assert not drift, (
            "query-table parity drift (symmetric difference vs _SYMBOL_QUERIES): "
            f"{drift}"
        )
