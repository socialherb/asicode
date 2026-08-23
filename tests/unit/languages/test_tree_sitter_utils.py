"""Tests for tree-sitter utility functions (has_error, find_error_nodes, etc.)."""

import time

import pytest

from external_llm.languages.models import (
    _EXT_MAP,
    _LANGUAGE_EXTENSION_GROUPS,
    _LANGUAGE_FAMILIES,
    LanguageId,
)
from external_llm.languages.tree_sitter_utils import (
    _BASE_KIND_MAP,
    _CALL_QUERIES,
    _CSS_KIND_MAP,
    _EXT_TO_GRAMMAR_KEY,
    _FULL_AST_GRAMMAR_KEYS,
    _GRAMMAR_KEY_OVERRIDES,
    _IMPORT_QUERIES,
    _LANG_MODULE_MAP,
    _MODULE_NAME_OVERRIDES,
    _PARSE_ONLY_GRAMMAR_KEYS,
    _REFERENCE_QUERIES,
    _SYMBOL_QUERIES,
    _WALK_KIND_MAP,
    _derive_ext_to_grammar_key,
    _derive_lang_module_map,
    _node_kind_from_type,
    extract_calls,
    extract_class_methods,
    extract_import_names,
    extract_imports,
    find_all_symbols,
    find_error_nodes,
    get_parser,
    has_error,
    parse_to_tree,
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
        assert extract_imports("import scala.collection.mutable", "scala") == [("scala.collection.mutable", 1)]
        # Multi-import: strip {c, d} selectors
        assert extract_imports("import a.b.{c, d}", "scala") == [("a.b", 1)]
        # Wildcard: strip ._ suffix
        assert extract_imports("import a.b._", "scala") == [("a.b", 1)]
        # Nested multi-import: only strip the final selector block
        assert extract_imports("import a.b.c.{d => e, f => g}", "scala") == [("a.b.c", 1)]
        # Scala 3 wildcard: strip .* suffix
        assert extract_imports("import a.b.*", "scala") == [("a.b", 1)]

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
        assert ("os.path", 1) in set(extract_imports("import os.path", "python"))
        assert ("json", 1) in set(extract_imports("from json import loads", "python"))

    def test_c_sharp_using_keyword_not_stripped(self):
        # The keyword strip is scoped to Scala only (`if language == "scala"`),
        # so C# never enters the strip path. This pins that C#'s @source
        # capture yields the bare dotted path without the `using` keyword.
        assert extract_imports("using System.IO;", "c_sharp") == [("System.IO", 1)]

    def test_lua_path_starting_with_import_keyword_is_preserved(self):
        # Pins the Scala-only scope fix: a pathological lua module path that
        # literally starts with "import " must NOT be sliced by the regex.
        assert extract_imports('require("import foo")', "lua") == [("import foo", 1)]


class TestGrammarMapConsistency:
    """_EXT_TO_GRAMMAR_KEY / _LANG_MODULE_MAP are DERIVED — no hand-written parallel maps."""

    def test_grammar_map_is_derived_from_ext_map(self):
        """Recomputing the derivation yields the live map — a hand-edit of
        _EXT_TO_GRAMMAR_KEY is drift (it must stay a pure function of _EXT_MAP,
        the query maps and the overrides)."""
        assert _derive_ext_to_grammar_key() == _EXT_TO_GRAMMAR_KEY
        # The ONLY grammar key diverging from its language's value: .tsx
        # parses with the separate "tsx" grammar (same package).  A second
        # divergence must be a deliberate, reviewed change.
        assert _GRAMMAR_KEY_OVERRIDES == {".tsx": "tsx"}
        # Parse-only languages (JSON/CSS/HTML) are excluded from the domain.
        assert not set(_EXT_TO_GRAMMAR_KEY) & {
            ".json",
            ".jsonc",
            ".css",
            ".scss",
            ".less",
            ".html",
            ".htm",
        }

    def test_grammar_map_domain_is_full_ast_query_intersection(self):
        """Full-AST keys are defined as the four query maps' intersection."""
        assert (
            frozenset(_SYMBOL_QUERIES)
            & frozenset(_CALL_QUERIES)
            & frozenset(_IMPORT_QUERIES)
            & frozenset(_REFERENCE_QUERIES)
        ) == _FULL_AST_GRAMMAR_KEYS

    def test_grammar_keys_resolve_to_working_parsers(self):
        """Every entry in _EXT_TO_GRAMMAR_KEY must produce a working tree-sitter parser."""
        failures = []
        for ext, key in _EXT_TO_GRAMMAR_KEY.items():
            parser = get_parser(key)
            if parser is None:
                failures.append(f"{ext} -> {key}")
        assert not failures, f"unresolvable grammar keys: {failures}"

    def test_family_groups_match_grammar_map(self):
        """_LANGUAGE_FAMILIES (the language-level callability partition) must
        name exactly the DERIVED full-AST domain.

        _EXT_TO_GRAMMAR_KEY is derived from _EXT_MAP, so this pins the only
        remaining hand-maintained fact: every full-AST language must have a
        family (else caller search widens to the broad fallback union and the
        cross-language guard silently bypasses), and no family may name a
        language without a grammar.  The extension sets themselves are DERIVED
        (models._derive_language_extension_groups → _EXT_MAP), so the
        extension-level equality below is a sanity consequence, not a pin.
        """
        family_names = {n for fam in _LANGUAGE_FAMILIES for n in fam}
        full_ast_names = {lang.name for lang in LanguageId if lang.value in _FULL_AST_GRAMMAR_KEYS}
        assert family_names == full_ast_names, (
            f"drift between _LANGUAGE_FAMILIES and the grammar-map domain: "
            f"families only: {sorted(family_names - full_ast_names)}, "
            f"grammar only: {sorted(full_ast_names - family_names)}"
        )
        # Derived-consequence check: families → _EXT_MAP → groups must land on
        # exactly the grammar-map extensions (same _EXT_MAP source, so a
        # mismatch would mean the derivation itself is broken).
        family_exts = {e for g in _LANGUAGE_EXTENSION_GROUPS for e in g}
        assert set(_EXT_TO_GRAMMAR_KEY) == family_exts

    def test_grammar_map_derivation_fails_fast_on_override_without_queries(self, monkeypatch):
        """An override whose key has no query maps is an import-time error, not a silent drop."""
        monkeypatch.setattr(
            "external_llm.languages.tree_sitter_utils._GRAMMAR_KEY_OVERRIDES",
            {".tsx": "no_such_grammar"},
        )
        with pytest.raises(ValueError, match="has no query maps"):
            _derive_ext_to_grammar_key()

    def test_grammar_map_derivation_fails_fast_on_unmapped_module(self, monkeypatch):
        """A derived key missing from _LANG_MODULE_MAP would read as unavailable
        to the gate probe (is_language_available) — fail-fast, not a silent map."""
        monkeypatch.setattr(
            "external_llm.languages.tree_sitter_utils._LANG_MODULE_MAP",
            {k: v for k, v in _LANG_MODULE_MAP.items() if k != "go"},
        )
        with pytest.raises(ValueError, match="has no entry in _LANG_MODULE_MAP"):
            _derive_ext_to_grammar_key()

    def test_lang_module_map_is_derived(self):
        """_LANG_MODULE_MAP domain = full-AST keys | parse-only keys, values
        follow the tree_sitter_<key> convention except the single tsx override.

        Recomputation must equal the live map — a hand-written literal or a
        partial table edit is a drift.  The parse-only set is the documented
        hand-maintained fact (html/css are parsed by editing/symbol tooling;
        JSON is deliberately absent — nothing parses .json with tree-sitter),
        and the override set mirrors _GRAMMAR_KEY_OVERRIDES' single-entry shape.
        """
        assert _derive_lang_module_map() == _LANG_MODULE_MAP
        assert frozenset(_LANG_MODULE_MAP) == (_FULL_AST_GRAMMAR_KEYS | _PARSE_ONLY_GRAMMAR_KEYS)
        assert frozenset({"html", "css"}) == _PARSE_ONLY_GRAMMAR_KEYS
        assert _MODULE_NAME_OVERRIDES == {"tsx": "typescript"}
        for key, module in _LANG_MODULE_MAP.items():
            assert module == f"tree_sitter_{_MODULE_NAME_OVERRIDES.get(key, key)}"

    def test_lang_module_map_derivation_fails_fast_on_unknown_parse_only_key(self, monkeypatch):
        """A parse-only key that is not a LanguageId value (typo) is an import-time
        error, not a silently dropped or phantom module path."""
        monkeypatch.setattr(
            "external_llm.languages.tree_sitter_utils._PARSE_ONLY_GRAMMAR_KEYS",
            frozenset({"html", "htlm"}),
        )
        with pytest.raises(ValueError, match="not a LanguageId value"):
            _derive_lang_module_map()

    def test_lang_module_map_derivation_fails_fast_on_typo_query_key(self, monkeypatch):
        """A full-AST key that is neither a LanguageId value nor the tsx alias is
        an import-time error — a typo'd query-map key would otherwise build a
        phantom tree_sitter_<typo> module path that fails only at first parse."""
        monkeypatch.setattr(
            "external_llm.languages.tree_sitter_utils._FULL_AST_GRAMMAR_KEYS",
            _FULL_AST_GRAMMAR_KEYS | frozenset({"typescrit"}),
        )
        with pytest.raises(ValueError, match="not a LanguageId value"):
            _derive_lang_module_map()

    def test_provider_globs_cover_ext_map(self):
        """SSOT 4-way: every provider's ``get_file_globs()`` must exactly cover
        the ``_EXT_MAP`` extensions for that provider's LanguageId.

        The fourth dimension of the consistency invariant (alongside _EXT_MAP /
        _LANGUAGE_EXTENSION_GROUPS / _EXT_TO_GRAMMAR_KEY).  Provider globs drive
        symbol-index discovery (``rg --files --glob`` in symbol_search); an
        extension that is mapped (resolution works on a given path) and
        parseable (grammar key present) but NOT globbed is silently dropped from
        the name→location index — find_symbol / modify_symbol by name miss it.
        This is the exact bug class where .zsh/.ksh were added to three tables
        but the bash provider's globs were forgotten (and the pre-existing
        .pyi / .mts / .cts drift of the same kind).
        """
        from collections import defaultdict

        from external_llm.languages.registry import LanguageRegistry

        # _EXT_MAP values are uppercase LanguageId *names* ("PYTHON"); match via
        # provider.language_id().name (NOT .value, which is lowercase "python").
        exts_by_lang: dict[str, set[str]] = defaultdict(set)
        for ext, lang in _EXT_MAP.items():
            exts_by_lang[lang].add(ext)

        problems: list[str] = []
        for provider in LanguageRegistry.instance()._providers.values():
            lang = provider.language_id().name
            mapped = exts_by_lang.get(lang, set())
            # "*.py" → ".py"
            globbed = {glob[1:] for glob in provider.get_file_globs()}
            if mapped != globbed:
                problems.append(f"  {lang}: _EXT_MAP={sorted(mapped)} globs={sorted(globbed)}")
        assert not problems, "provider glob drift — get_file_globs() must match _EXT_MAP per LanguageId:\n" + "\n".join(
            problems
        )

    def test_every_ext_map_language_has_provider(self):
        """SSOT 5-way: every ``_EXT_MAP`` language must have a registered provider.

        The companion ``test_provider_globs_cover_ext_map`` only iterates over
        *registered* providers, so it is blind to a language that appears in
        ``_EXT_MAP`` (resolution works) and even in ``_EXT_TO_GRAMMAR_KEY``
        (grammar queries + AST extraction work) but has **no provider at all**
        — exactly the LUA/SCALA "half-wired" state where ``_nonpy_index_for``
        (which iterates registered providers) never reached those files and
        ``find_symbol`` / ``modify_symbol`` returned empty with no signal.

        Closing this blind spot structurally: adding a language to ``_EXT_MAP``
        now requires a provider or the build fails.
        """
        from external_llm.languages.registry import LanguageRegistry

        registered = {p.language_id().name for p in LanguageRegistry.instance()._providers.values()}
        mapped = set(_EXT_MAP.values())
        missing = mapped - registered
        assert not missing, (
            f"languages in _EXT_MAP with no registered provider — "
            f"find_symbol/modify_symbol silently return empty for these: "
            f"{sorted(missing)}"
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
        disagreements = {k: (walk[k], base[k]) for k in base.keys() & walk.keys() if walk[k] != base[k]}
        assert not disagreements, f"kind-map drift (node_type: walk_kind != query_kind): {disagreements}"

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

    def test_extract_class_methods_no_recursion_error(self):
        methods = extract_class_methods(
            self._deep_sibling_then_class(),
            "C",
            "javascript",
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
        drift = {name: sorted(keys ^ ref) for name, keys in keysets.items() if keys != ref}
        assert not drift, f"query-table parity drift (symmetric difference vs _SYMBOL_QUERIES): {drift}"


# ── A cold resolve must not block unrelated cache hits ──────────────────────


class TestLanguageResolveLockScope:
    """``_LANG_CACHE_LOCK`` must not be held across grammar resolution.

    It used to be, and the first-ever resolve of a language is the expensive
    case the language-pack pays to materialise a grammar (measured 550-660 ms
    for ocaml/nim/crystal/erlang before their on-disk cache exists). While one
    thread paid that, every other thread's lookup of an ALREADY-CACHED language
    blocked behind the same global lock — 587 ms, measured, for a `python`
    cache hit.
    """

    def _slow_resolve(self, monkeypatch, delay, seen):
        import external_llm.languages.tree_sitter_utils as tsu

        def _fake(language):
            seen.append(language)
            time.sleep(delay)
            return object()

        monkeypatch.setattr(tsu, "_resolve_language_uncached", _fake)
        return tsu

    def test_cold_resolve_does_not_block_a_cached_lookup(self, monkeypatch):
        import threading
        import time as _t

        import external_llm.languages.tree_sitter_utils as tsu

        if not tsu._HAS_TREE_SITTER:
            pytest.skip("tree-sitter core not installed")

        seen: list[str] = []
        self._slow_resolve(monkeypatch, 0.5, seen)

        # A guaranteed cache hit, populated without going through the fake.
        with tsu._LANG_CACHE_LOCK:
            tsu._LANG_CACHE["__cached__"] = object()

        blocked: list[float] = []
        started = threading.Event()

        def cold():
            started.set()
            tsu._get_language("__cold__")

        def hit():
            started.wait()
            _t.sleep(0.05)  # let the cold resolve get well inside its sleep
            t0 = _t.perf_counter()
            tsu._get_language("__cached__")
            blocked.append(_t.perf_counter() - t0)

        threads = [threading.Thread(target=cold), threading.Thread(target=hit)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert blocked[0] < 0.1, (
            f"cached lookup waited {blocked[0] * 1000:.0f} ms behind an unrelated "
            f"cold resolve — the global lock is being held across resolution"
        )

    def test_same_language_is_resolved_once_under_concurrency(self, monkeypatch):
        """Per-language locking: N threads on one cold language do the work once.

        Without it, moving resolution out from under the global lock would turn
        one 600 ms resolve into N of them. ``setswitchinterval`` is forced
        because a threaded test that passes with the locking REMOVED asserts
        nothing — this repo has shipped exactly that.
        """
        import sys
        import threading

        import external_llm.languages.tree_sitter_utils as tsu

        if not tsu._HAS_TREE_SITTER:
            pytest.skip("tree-sitter core not installed")

        seen: list[str] = []
        self._slow_resolve(monkeypatch, 0.05, seen)

        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            barrier = threading.Barrier(8)

            def worker():
                barrier.wait()
                tsu._get_language("__once__")

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(old)

        assert seen == ["__once__"], f"resolved {len(seen)} times under 8 concurrent callers: {seen}"

    def test_invalidation_during_resolve_is_not_undone_by_the_store(self, monkeypatch):
        """A clear landing mid-resolve must not be resurrected by the store.

        ``invalidate_caches()`` exists so a late pip-installed grammar takes
        effect without a restart. Resolving outside the lock opens the classic
        "read -> slow collect -> store" window: the store that follows would
        re-insert the pre-invalidation Language and the new grammar would never
        be picked up.
        """
        import threading
        import time as _t

        import external_llm.languages.tree_sitter_utils as tsu

        if not tsu._HAS_TREE_SITTER:
            pytest.skip("tree-sitter core not installed")

        seen: list[str] = []
        self._slow_resolve(monkeypatch, 0.3, seen)

        def invalidate_midway():
            _t.sleep(0.1)
            tsu.invalidate_caches()

        t = threading.Thread(target=invalidate_midway)
        t.start()
        result = tsu._get_language("__raced__")
        t.join()

        assert result is not None, "the caller still gets its object"
        assert "__raced__" not in tsu._LANG_CACHE, (
            "a resolve that straddled invalidate_caches() re-populated the cache it just cleared"
        )


class TestSharedTreeParam:
    """P5 (2026-08-11): a caller-supplied tree must yield identical results.

    ``find_all_symbols`` / ``extract_calls`` / ``extract_imports`` accept an
    optional ``tree`` so one parse can serve every extraction.  Sources above
    ``_MAX_CACHED_SOURCE_CHARS`` bypass parse_to_tree's memo, so a cold
    extraction used to parse large files once per query (find_all_symbols
    alone parses twice: query phase + manual walk).  The tree param must be a
    no-op for correctness: same content + same grammar → same tree → same
    results.
    """

    _SRC = (
        "import { helper } from './helper';\n"
        "export function alpha() { helper(); }\n"
        "export class Box {\n"
        "  render() { return alpha(); }\n"
        "}\n"
    )

    def test_tree_param_matches_internal_parse(self):
        tree = parse_to_tree(self._SRC, "typescript")
        if tree is None:
            pytest.skip("tree-sitter typescript grammar not installed")
        assert find_all_symbols(self._SRC, "typescript", tree=tree) == find_all_symbols(self._SRC, "typescript")
        assert extract_calls(self._SRC, "typescript", tree=tree) == extract_calls(self._SRC, "typescript")
        assert extract_imports(self._SRC, "typescript", tree=tree) == extract_imports(self._SRC, "typescript")

    def test_tree_param_results_are_nonempty(self):
        """Guard against the parity test passing vacuously (all empty)."""
        tree = parse_to_tree(self._SRC, "typescript")
        if tree is None:
            pytest.skip("tree-sitter typescript grammar not installed")
        assert find_all_symbols(self._SRC, "typescript", tree=tree)
        assert extract_calls(self._SRC, "typescript", tree=tree)
        assert extract_imports(self._SRC, "typescript", tree=tree)
