"""Unit tests for scanner_registry.py — 100% coverage."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from external_llm.agent.scanner_registry import (
    _PYTHON_ONLY,
    _TS_LANGUAGES,
    ScannerRegistry,
    ScannerSpec,
)
from external_llm.languages import LanguageId


@pytest.fixture
def registry() -> ScannerRegistry:
    """A fresh ScannerRegistry with one mock scanner registered."""
    r = ScannerRegistry()
    r.register(
        ScannerSpec(
            name="test_scanner",
            description="A test scanner",
            input_schema={"limit": "int"},
            output_type_name="TestCandidate",
            produces_workset_kinds=["test_cluster"],
            file_filter=".py",
        ),
        MagicMock(return_value=[]),
    )
    return r


@pytest.fixture
def registry_no_filter() -> ScannerRegistry:
    """A fresh ScannerRegistry with a scanner that has no file_filter."""
    r = ScannerRegistry()
    r.register(
        ScannerSpec(
            name="no_filter_scanner",
            description="No file filter",
            file_filter="",
        ),
        MagicMock(return_value=[]),
    )
    return r


# ── Basic getters (lines 77, 81, 85) ────────────────────────────────────────

class TestGetters:
    def test_get_known(self, registry: ScannerRegistry):
        assert registry.get("test_scanner") is not None

    def test_get_unknown_returns_none(self, registry: ScannerRegistry):
        """Line 77: missing scanner returns None."""
        assert registry.get("nonexistent") is None

    def test_get_spec_known(self, registry: ScannerRegistry):
        spec = registry.get_spec("test_scanner")
        assert spec is not None
        assert spec.name == "test_scanner"

    def test_get_spec_unknown_returns_none(self, registry: ScannerRegistry):
        """Line 81: missing scanner spec returns None."""
        assert registry.get_spec("nonexistent") is None

    def test_list_scanners(self, registry: ScannerRegistry):
        """Line 85: list_scanners returns scanner specs."""
        specs = registry.list_scanners()
        assert len(specs) == 1
        assert specs[0].name == "test_scanner"

    def test_list_names(self, registry: ScannerRegistry):
        names = registry.list_names()
        assert names == ["test_scanner"]


# ── is_scanner_implementation_file ───────────────────────────────────────────

class TestIsScannerImplementationFile:
    def test_python_file_matches(self, registry: ScannerRegistry):
        assert registry.is_scanner_implementation_file("test_scanner.py") is True

    def test_python_file_does_not_match(self, registry: ScannerRegistry):
        assert registry.is_scanner_implementation_file("some_random.py") is False

    def test_non_python_file(self, registry: ScannerRegistry):
        assert registry.is_scanner_implementation_file("test_scanner.js") is False


# ── names_for_spec_target_files (line 118) ───────────────────────────────────

class TestNamesForSpecTargetFiles:
    """Coverage for line 118 — matched stem is appended."""

    def test_python_stem_matches(self, registry: ScannerRegistry):
        result = registry.names_for_spec_target_files(["test_scanner.py", "other.py"])
        assert result == ["test_scanner"]

    def test_no_python_suffix(self, registry: ScannerRegistry):
        """Stem without .py — LanguageId won't strip anything."""
        result = registry.names_for_spec_target_files(["test_scanner"])
        assert result == ["test_scanner"]

    def test_no_match(self, registry: ScannerRegistry):
        result = registry.names_for_spec_target_files(["unrelated.py"])
        assert result == []


# ── run() method (lines 143-192) ─────────────────────────────────────────────

class TestRun:
    """Coverage for ScannerRegistry.run() — all branches."""

    def test_unknown_scanner_raises_value_error(self, registry: ScannerRegistry):
        """Lines 143-146: fn is None -> ValueError."""
        with pytest.raises(ValueError, match="Unknown scanner"):
            registry.run("nonexistent")

    def test_truncated_cleanup_attribute_present(self, registry: ScannerRegistry):
        """Lines 148-152: fn has _truncated -> del succeeds."""
        fn = registry.get("test_scanner")
        fn._truncated = 5
        registry.run("test_scanner")
        assert not hasattr(fn, "_truncated")

    def test_truncated_cleanup_no_attribute(self, registry: ScannerRegistry):
        """Line 151-152: fn has no _truncated -> AttributeError caught."""
        registry.register(
            ScannerSpec(name="no_trunc", description="x"),
            lambda repo_root="", file_paths=None, **kwargs: [],
        )
        registry.run("no_trunc")  # should not raise

    def test_with_file_paths_filtered(self, registry: ScannerRegistry):
        """Lines 157-164: file_paths filtered by spec.file_filter (".py")."""
        fn = registry.get("test_scanner")
        fn.return_value = []
        registry.run("test_scanner", file_paths=["a.py", "b.txt", "c.py"])
        fn.assert_called_once()
        _args, _kwargs = fn.call_args
        assert _kwargs["file_paths"] == ["a.py", "c.py"]

    def test_with_file_paths_no_filter(self, registry_no_filter: ScannerRegistry):
        """Lines 157: file_filter is empty -> no filtering."""
        fn = registry_no_filter.get("no_filter_scanner")
        fn.return_value = []
        registry_no_filter.run("no_filter_scanner", file_paths=["a.py", "b.txt"])
        fn.assert_called_once()
        _args, _kwargs = fn.call_args
        assert _kwargs["file_paths"] == ["a.py", "b.txt"]

    def test_no_file_paths(self, registry: ScannerRegistry):
        """file_paths=None -> forwarded as []."""
        fn = registry.get("test_scanner")
        fn.return_value = []
        registry.run("test_scanner")
        _args, _kwargs = fn.call_args
        assert _kwargs["file_paths"] == []

    def test_truncated_count_non_int(self, registry: ScannerRegistry):
        """Lines 172-174: fn._truncated is not int -> treated as 0."""
        fn = registry.get("test_scanner")
        def _set_invalid_truncated(**kwargs):
            fn._truncated = "invalid"
            return []
        fn.side_effect = _set_invalid_truncated
        result = registry.run("test_scanner")
        assert result.truncated_count == 0

    def test_file_filter_normalizes_dot(self, registry_no_filter: ScannerRegistry):
        """Line 160: file_filter "py" (no dot) normalizes to ".py"."""
        r = ScannerRegistry()
        r.register(
            ScannerSpec(name="no_dot_filter", description="x", file_filter="py"),
            MagicMock(return_value=[]),
        )
        fn = r.get("no_dot_filter")
        r.run("no_dot_filter", file_paths=["a.py", "b.txt", "c.PY"])
        _args, _kwargs = fn.call_args
        # "py" normalizes to ".py" -> case-sensitive endswith
        assert _kwargs["file_paths"] == ["a.py"]

    def test_candidate_with_to_dict(self, registry: ScannerRegistry):
        """Lines 178-179: candidate has to_dict method."""
        fn = registry.get("test_scanner")
        candidate = MagicMock()
        candidate.to_dict.return_value = {"key": "value"}
        fn.return_value = [candidate]
        result = registry.run("test_scanner")
        assert result.candidates_raw == [{"key": "value"}]

    def test_candidate_is_dict(self, registry: ScannerRegistry):
        """Lines 180-181: candidate is plain dict."""
        fn = registry.get("test_scanner")
        fn.return_value = [{"a": 1}]
        result = registry.run("test_scanner")
        assert result.candidates_raw == [{"a": 1}]

    def test_candidate_other(self, registry: ScannerRegistry):
        """Lines 182-183: candidate is other object -> repr."""
        fn = registry.get("test_scanner")
        fn.return_value = [42]
        result = registry.run("test_scanner")
        assert result.candidates_raw == [{"repr": "42"}]

    def test_affected_from_attr(self, registry: ScannerRegistry):
        """Lines 187-188: candidate has .file attribute."""
        fn = registry.get("test_scanner")
        c1 = MagicMock()
        c1.file = "a.py"
        fn.return_value = [c1]
        result = registry.run("test_scanner")
        assert result.affected_files == {"a.py"}

    def test_affected_from_dict(self, registry: ScannerRegistry):
        """Lines 189-190: candidate is dict with 'file' key."""
        fn = registry.get("test_scanner")
        fn.return_value = [{"file": "b.py"}]
        result = registry.run("test_scanner")
        assert result.affected_files == {"b.py"}

    def test_affected_empty(self, registry: ScannerRegistry):
        """Neither attribute nor dict key -> empty affected_files."""
        fn = registry.get("test_scanner")
        fn.return_value = [{"no_file": True}]
        result = registry.run("test_scanner")
        assert result.affected_files == set()

    def test_scanner_result_fields(self, registry: ScannerRegistry):
        """Lines 192-199: result object fields."""
        fn = registry.get("test_scanner")
        fn.return_value = []
        result = registry.run("test_scanner")
        assert result.scanner_name == "test_scanner"
        assert result.total_candidates == 0
        assert result.truncated_count == 0
        assert result.candidates_raw == []
        assert result.affected_files == set()

    def test_concurrent_same_scanner_truncation_isolated(self):
        """Regression: concurrent run() calls for the SAME scanner must each
        report their own truncated_count. Scanners set truncation out-of-band
        on the shared function object (``fn._truncated``); without per-scanner
        serialization the reset→invoke→read sequence races and one run reads
        another's value. ``run_structural_scan`` is a read-only tool
        (``_READ_ONLY_TOOLS``) that parallelizes in the agent read phase, so
        this race is reachable in production."""
        import threading
        import time

        r = ScannerRegistry()
        local = threading.local()

        def _scan(repo_root="", file_paths=None, **kwargs):
            # Mimics real scanners (e.g. scan_dead_blocks._truncated = ...):
            _scan._truncated = local.value
            time.sleep(0.004)  # widen the set→read window (sleep releases the GIL)
            return []

        r.register(ScannerSpec(name="s", description="x", file_filter=""), _scan)

        errors = []

        def _run(value):
            local.value = value
            for _ in range(40):
                res = r.run("s")
                if res.truncated_count != value:
                    errors.append((value, res.truncated_count))
                    return

        threads = [threading.Thread(target=_run, args=(v,)) for v in (7, 13, 21)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, (
            f"truncation crosstalk under concurrent same-scanner runs: {errors} "
            "(a run reported another run's truncated_count)"
        )

    def test_concurrent_different_scanners_still_parallel(self):
        """The fix serializes same-scanner runs only; different scanners (each
        with its own lock) must still run concurrently. Guards against an
        accidental global lock that would over-serialize the registry."""
        import threading
        import time

        r = ScannerRegistry()

        def _slow(repo_root="", file_paths=None, **kwargs):
            time.sleep(0.05)
            return []

        r.register(ScannerSpec(name="a", description="x", file_filter=""), _slow)
        r.register(ScannerSpec(name="b", description="x", file_filter=""), _slow)

        t0 = time.time()

        def _run(n):
            r.run(n)

        ta = threading.Thread(target=_run, args=("a",))
        tb = threading.Thread(target=_run, args=("b",))
        ta.start()
        tb.start()
        ta.join()
        tb.join()
        elapsed = time.time() - t0
        # Parallel ~0.05s; serial ~0.10s. Allow scheduler slack.
        assert elapsed < 0.09, (
            f"different scanners ran serially ({elapsed:.3f}s) — the per-scanner "
            "lock must not collapse into a single global lock"
        )


# ── vulture_dead_code_scanner registration & requires_graph (lines 343-369) ───

class TestVultureScannerRegistration:
    """Verify vulture_dead_code_scanner is registered and graph-gated correctly.

    These run before TestAutoRegisterImportErrors (which reloads the module and
    resets registry state), so they see the real auto-registered global registry.
    """

    def test_vulture_registered_in_global_registry(self):
        """The auto-registered global registry must include vulture_dead_code_scanner."""
        from external_llm.agent.scanner_registry import get_registry
        names = get_registry().list_names()
        assert "vulture_dead_code_scanner" in names

    def test_vulture_spec_requires_graph(self):
        """vulture scanner must declare requires_graph=True so the handler injects repo_graph."""
        from external_llm.agent.scanner_registry import get_registry
        spec = get_registry().get_spec("vulture_dead_code_scanner")
        assert spec is not None
        assert spec.requires_graph is True
        assert spec.file_filter == ".py"

    def test_vulture_spec_advertises_input_schema(self):
        """Planner prompt sees min_confidence/max_per_file/exclude_patterns."""
        from external_llm.agent.scanner_registry import get_registry
        spec = get_registry().get_spec("vulture_dead_code_scanner")
        assert spec is not None
        assert "min_confidence" in spec.input_schema
        assert "max_per_file" in spec.input_schema
        # repo_graph is NOT in input_schema — graph objects are not serializable
        # and must not be advertised to the planner prompt.
        assert "repo_graph" not in spec.input_schema

    def test_run_structural_scan_enum_includes_vulture(self):
        """The run_structural_scan tool enum must expose vulture_dead_code_scanner."""
        from external_llm.agent.tool_schemas import AGENT_TOOL_SCHEMAS
        for schema in AGENT_TOOL_SCHEMAS:
            if schema["name"] == "run_structural_scan":
                enum = schema["parameters"]["properties"]["scanner"]["enum"]
                assert "vulture_dead_code_scanner" in enum
                return
        pytest.fail("run_structural_scan schema not found")

    def test_scan_vulture_dead_code_leaf_scope_uses_file_paths_only(self, tmp_path):
        """With a leaf-like graph (no importers), scope must be file_paths_only.

        This is the key runtime fix: leaf-only targets scavenge just the target
        files instead of the whole project (the historical ~90s cost driver).
        """
        from external_llm.analysis.vulture_scanner import decide_vulture_scan_scope

        class _LeafGraph:
            def get_importers(self, fp):
                return []

        assert decide_vulture_scan_scope(_LeafGraph(), ["a.py", "b.py"], 5) == "file_paths_only"

    def test_scan_vulture_dead_code_hub_scope_uses_full_project(self):
        """A hub file (>= threshold importers) forces full_project scan."""
        from external_llm.analysis.vulture_scanner import decide_vulture_scan_scope

        class _HubGraph:
            def get_importers(self, fp):
                return ["x", "y", "z", "w", "v"] if fp == "hub.py" else []

        assert decide_vulture_scan_scope(_HubGraph(), ["hub.py", "leaf.py"], 5) == "full_project"

    def test_scan_vulture_dead_code_no_graph_falls_back_to_full_project(self):
        """When repo_graph is unavailable, default to the accurate full_project scan."""
        from external_llm.analysis.vulture_scanner import decide_vulture_scan_scope
        assert decide_vulture_scan_scope(None, ["a.py"], 5) == "full_project"


# ── _auto_register import error blocks (lines 227-228, 246-247, etc.) ────────
# Run LAST (after all other tests) because importlib.reload resets module state.

class TestAutoRegisterImportErrors:
    """Cover all except ImportError blocks in _auto_register."""

    def test_all_scanners_fail_import(self):
        """When all scanner modules are unavailable, except blocks are exercised."""
        import importlib
        import sys

        from external_llm.agent import scanner_registry as sr_mod

        scanner_modules = [
            "external_llm.analysis.dead_block_scanner",
            "external_llm.analysis.duplicate_definition_scanner",
            "external_llm.analysis.unused_import_scanner",
            "external_llm.analysis.public_dead_code_scanner",
            "external_llm.analysis.contradictory_logic_scanner",
            "external_llm.analysis.ast_similarity_scanner",
            "external_llm.analysis.vulture_scanner",
            "external_llm.analysis.container_reachability_scanner",
            "external_llm.analysis.broken_contract_scanner",
        ]

        # Save original modules
        saved = {}
        for mod_name in scanner_modules:
            if mod_name in sys.modules:
                saved[mod_name] = sys.modules.pop(mod_name)

        try:
            # Patch sys.modules with None for scanner modules so imports raise ImportError
            with patch.dict(sys.modules, {m: None for m in scanner_modules}, clear=False):
                importlib.reload(sr_mod)
            # After reload with all imports failing, no scanners should be registered
            assert sr_mod.get_registry().list_names() == []
        finally:
            # Restore original modules
            sys.modules.update(saved)
            # Reload once more to restore scanner_registry with real scanners
            importlib.reload(sr_mod)


# ── supported_languages: language-aware filtering ──────────────────────────


class TestSupportedLanguages:
    """Contract tests for language-aware scanner filtering.

    Guards against the Go-repo false-positive regression: Python-only AST
    scanners must never receive (and mis-parse) Go/TS source. A scanner that
    declares ``supported_languages`` has files of other languages filtered out
    in ``run()`` *before* the scanner function runs.
    """

    def test_python_only_scanner_drops_go_and_ts_files(self):
        """run() filters out files whose LanguageId is not supported."""
        r = ScannerRegistry()
        r.register(
            ScannerSpec(
                name="py_only",
                description="python only",
                supported_languages=set(_PYTHON_ONLY),
            ),
            MagicMock(return_value=[]),
        )
        fn = r.get("py_only")
        r.run("py_only", file_paths=["a.py", "b.go", "c.ts", "d/main.go", "e.py"])
        _a, kw = fn.call_args
        # Only .py files survive; Go/TS dropped before the scanner sees them.
        assert kw["file_paths"] == ["a.py", "e.py"]

    def test_ts_scanner_admits_go_and_ts_and_py(self):
        """A tree-sitter scanner keeps all six supported languages."""
        r = ScannerRegistry()
        r.register(
            ScannerSpec(
                name="ts_scanner",
                description="multi-language",
                supported_languages=set(_TS_LANGUAGES),
            ),
            MagicMock(return_value=[]),
        )
        fn = r.get("ts_scanner")
        r.run(
            "ts_scanner",
            file_paths=["a.py", "b.go", "c.ts", "d.java", "e.kt", "f.rb", "g.rs"],
        )
        _a, kw = fn.call_args
        # Rust/Ruby are NOT in the tree-sitter set → dropped.
        assert set(kw["file_paths"]) == {"a.py", "b.go", "c.ts", "d.java", "e.kt"}

    def test_supported_languages_overrides_file_filter(self):
        """When both are set, supported_languages wins (precedence contract).

        A tree-sitter scanner with file_filter='.py' but supported_languages of
        all six must STILL receive .go/.ts files — the language set is strictly
        more precise than the extension filter.
        """
        r = ScannerRegistry()
        r.register(
            ScannerSpec(
                name="mixed",
                description="legacy ext + lang set",
                file_filter=".py",
                supported_languages=set(_TS_LANGUAGES),
            ),
            MagicMock(return_value=[]),
        )
        fn = r.get("mixed")
        r.run("mixed", file_paths=["a.py", "b.go", "c.ts"])
        _a, kw = fn.call_args
        # file_filter='.py' is IGNORED — all three kept because all are TS langs.
        assert set(kw["file_paths"]) == {"a.py", "b.go", "c.ts"}

    def test_none_supported_languages_scans_everything(self):
        """None (default) = backward compatible, no language filtering."""
        r = ScannerRegistry()
        r.register(
            ScannerSpec(
                name="unconstrained",
                description="no lang set",
                supported_languages=None,
                file_filter="",
            ),
            MagicMock(return_value=[]),
        )
        fn = r.get("unconstrained")
        r.run("unconstrained", file_paths=["a.py", "b.go", "c.rs", "d.unknown"])
        _a, kw = fn.call_args
        assert kw["file_paths"] == ["a.py", "b.go", "c.rs", "d.unknown"]

    def test_empty_file_set_after_filter_still_invokes(self):
        """When filtering removes ALL files, the scanner runs with [] (no crash).

        This is the path exercised when scanner='all' runs over a pure-Go repo
        and a Python-only scanner has nothing to scan: run() returns 0 candidates
        rather than raising.
        """
        r = ScannerRegistry()
        r.register(
            ScannerSpec(
                name="py_only",
                description="python only",
                supported_languages=set(_PYTHON_ONLY),
            ),
            MagicMock(return_value=[]),
        )
        fn = r.get("py_only")
        result = r.run("py_only", file_paths=["a.go", "b.go", "c.ts"])
        _a, kw = fn.call_args
        assert kw["file_paths"] == []
        assert result.total_candidates == 0

    # ── Registered built-in scanners have the right language sets ──────────

    def test_builtin_py_only_scanners_declare_python(self):
        """Every Python-only built-in must advertise PYTHON."""
        from external_llm.agent.scanner_registry import get_registry
        registry = get_registry()
        py_only_names = {
            "unused_import_scanner",
            "contradictory_logic_scanner",
            "ast_similarity_scanner",
            "vulture_dead_code_scanner",
            "container_reachability_scanner",
            "broken_contract_scanner",
            # Dead-code detection is Python-only — cross-reference reachability
            # is unreliable for Go/Java/Kotlin/TS/JS without native semantic
            # analysis (e.g. staticcheck for Go). Defer to language-native tools.
            "dead_block_scanner",
            "public_dead_code_scanner",
        }
        for name in py_only_names:
            spec = registry.get_spec(name)
            assert spec is not None, f"{name} not registered"
            assert spec.supported_languages == {LanguageId.PYTHON}, (
                f"{name} must declare PYTHON-only (got {spec.supported_languages})"
            )

    def test_builtin_ts_scanners_declare_six_languages(self):
        """Tree-sitter scanners must advertise all six supported languages."""
        from external_llm.agent.scanner_registry import get_registry
        registry = get_registry()
        ts_names = {
            # Only duplicate_definition_scanner stays multi-language: it is NOT
            # dead-code detection (Go receiver dedup was verified 50->0 FP).
            "duplicate_definition_scanner",
        }
        for name in ts_names:
            spec = registry.get_spec(name)
            assert spec is not None, f"{name} not registered"
            assert spec.supported_languages is not None
            assert LanguageId.PYTHON in spec.supported_languages
            assert LanguageId.GO in spec.supported_languages
            assert LanguageId.TYPESCRIPT in spec.supported_languages
            assert LanguageId.JAVASCRIPT in spec.supported_languages
            assert LanguageId.JAVA in spec.supported_languages
            assert LanguageId.KOTLIN in spec.supported_languages

    def test_unused_import_scanner_file_filter_now_py(self):
        """Regression: unused_import_scanner file_filter was '' (run on all
        files, relying on an internal guard). Now corrected to '.py' for
        clarity, matching its PYTHON-only supported_languages.
        """
        from external_llm.agent.scanner_registry import get_registry
        spec = get_registry().get_spec("unused_import_scanner")
        assert spec.file_filter == ".py"
