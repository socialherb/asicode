"""
Symbol-aware test candidate discovery and ranking.

Discovers test files by matching target symbols against test file contents
using graph relationships (imports, callers, module proximity) rather than
just filename patterns.

Deterministic, rule-based. Falls back to filename heuristic if graph unavailable.
"""

import logging
import os
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from ..languages import LanguageId

logger = logging.getLogger(__name__)

# Match type priority scores (higher = stronger match)
MATCH_SCORES = {
    "direct_symbol": 1.0,
    "module_import": 0.8,
    "same_module": 0.6,
    "impact_adjacency": 0.4,
    "filename_fallback": 0.2,
}

# Scope level limits
SCOPE_LIMITS = {
    "narrow": 5,
    "standard": 10,
    "broad": 20,
}


@dataclass
class SymbolAwareTestTarget:
    """A test target with match reasoning and priority."""

    test_path: str
    priority_score: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    matched_symbols: list[str] = field(default_factory=list)
    match_type: str = "filename_fallback"
    scope_level_hint: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_path": self.test_path,
            "priority_score": round(self.priority_score, 3),
            "reason_codes": self.reason_codes[:3],
            "matched_symbols": self.matched_symbols[:3],
            "match_type": self.match_type,
            "scope_level_hint": self.scope_level_hint,
        }


class SymbolAwareTestFinder:
    """
    Discovers and ranks test candidates using symbol graph relationships.

    Match types (in priority order):
    1. direct_symbol — test file directly references target symbol name
    2. module_import — test file imports target module
    3. same_module — test is in corresponding test directory for target module
    4. impact_adjacency — test covers files in the impact set
    5. filename_fallback — test file matches naming pattern

    Falls back to filename heuristic if graph unavailable.
    """

    def __init__(self, repo_root: str):
        self._repo_root = repo_root
        self._test_files_cache: list[str] | None = None
        self._content_cache: dict[str, str] = {}

    def discover_test_targets(
        self,
        target_symbols: list[str] | None = None,
        target_files: list[str] | None = None,
        impact_files: list[str] | None = None,
        graph_context: dict | None = None,
        scope_level: str = "standard",
    ) -> list[SymbolAwareTestTarget]:
        """
        Discover ranked test candidates for given targets.

        Never raises — returns empty list on error.
        """
        targets: dict[str, SymbolAwareTestTarget] = {}  # path → target

        try:
            # Find all test files in repo
            test_files = self._find_all_test_files()
            if not test_files:
                return []

            symbols = target_symbols or []
            files = target_files or []
            impacts = impact_files or []

            # Extract from graph_context if available
            if graph_context:
                if not symbols:
                    symbols = [
                        s.get("name", "") for s in graph_context.get("resolved_symbols", []) if isinstance(s, dict)
                    ]
                if not impacts:
                    impacts = graph_context.get("impact_files", [])
                if not files:
                    files = graph_context.get("primary_files", [])

                # P4: supplement symbols from hotspot data in graph_context
                _hotspot_syms = graph_context.get("hotspot_symbols", [])
                if _hotspot_syms:
                    _existing = set(symbols)
                    for hs_sym in _hotspot_syms[:5]:
                        if hs_sym and hs_sym not in _existing:
                            symbols.append(hs_sym)
                            _existing.add(hs_sym)

            # Guard: if nothing to search on, return empty
            if not symbols and not files and not impacts:
                return []

            # 1. Direct symbol match — search test files for symbol names
            for sym in symbols[:10]:  # limit to prevent scanning too many
                if not sym:
                    continue
                for tf in test_files:
                    if self._file_references_symbol(tf, sym):
                        t = targets.setdefault(tf, SymbolAwareTestTarget(test_path=tf))
                        t.priority_score = max(t.priority_score, MATCH_SCORES["direct_symbol"])
                        t.match_type = "direct_symbol"
                        if "DIRECT_SYMBOL_MATCH" not in t.reason_codes:
                            t.reason_codes.append("DIRECT_SYMBOL_MATCH")
                        if sym not in t.matched_symbols:
                            t.matched_symbols.append(sym)

            # 2. Module import match — test imports the target module
            target_modules = self._extract_module_names(files)
            for mod in target_modules:
                for tf in test_files:
                    if self._file_imports_module(tf, mod):
                        t = targets.setdefault(tf, SymbolAwareTestTarget(test_path=tf))
                        if t.priority_score < MATCH_SCORES["module_import"]:
                            t.priority_score = MATCH_SCORES["module_import"]
                            t.match_type = "module_import"
                        if "MODULE_IMPORT" not in t.reason_codes:
                            t.reason_codes.append("MODULE_IMPORT")

            # 3. Same module/package — test in corresponding test directory
            for f in files:
                corresponding = self._find_corresponding_test(f, test_files)
                if corresponding:
                    t = targets.setdefault(corresponding, SymbolAwareTestTarget(test_path=corresponding))
                    if t.priority_score < MATCH_SCORES["same_module"]:
                        t.priority_score = MATCH_SCORES["same_module"]
                        t.match_type = "same_module"
                    if "SAME_MODULE" not in t.reason_codes:
                        t.reason_codes.append("SAME_MODULE")

            # 4. Impact adjacency — test files in impact set
            for imp in impacts:
                if self._is_test_file(imp):
                    t = targets.setdefault(imp, SymbolAwareTestTarget(test_path=imp))
                    if t.priority_score < MATCH_SCORES["impact_adjacency"]:
                        t.priority_score = MATCH_SCORES["impact_adjacency"]
                        t.match_type = "impact_adjacency"
                    if "IMPACT_ADJACENCY" not in t.reason_codes:
                        t.reason_codes.append("IMPACT_ADJACENCY")

            # P4: boost priority for targets matching hotspot symbols
            _hs_symbols = set(graph_context.get("hotspot_symbols", [])) if graph_context else set()
            if _hs_symbols:
                for t in targets.values():
                    for ms in t.matched_symbols:
                        if ms in _hs_symbols:
                            t.priority_score = min(1.0, t.priority_score + 0.1)
                            if "HOTSPOT_BOOST" not in t.reason_codes:
                                t.reason_codes.append("HOTSPOT_BOOST")
                            break

            # 5. Filename fallback — test files matching target file names
            if not targets:
                for f in files:
                    base = os.path.splitext(os.path.basename(f))[0]
                    for tf in test_files:
                        if base and base in os.path.basename(tf):
                            t = targets.setdefault(tf, SymbolAwareTestTarget(test_path=tf))
                            if t.priority_score < MATCH_SCORES["filename_fallback"]:
                                t.priority_score = MATCH_SCORES["filename_fallback"]
                                t.match_type = "filename_fallback"
                            if "FILENAME_FALLBACK" not in t.reason_codes:
                                t.reason_codes.append("FILENAME_FALLBACK")

        except Exception as e:
            logger.debug("Symbol-aware test discovery failed: %s", e)
            return []

        # Rank and filter
        ranked = sorted(targets.values(), key=lambda t: -t.priority_score)

        # Apply scope limit
        limit = SCOPE_LIMITS.get(scope_level, SCOPE_LIMITS["standard"])
        ranked = ranked[:limit]

        # Set scope_level_hint
        for t in ranked:
            t.scope_level_hint = scope_level

        return ranked

    def to_path_list(self, targets: list[SymbolAwareTestTarget]) -> list[str]:
        """Degrade ranked targets to simple path list for TestRunner."""
        return [t.test_path for t in targets]

    def find_tests_for_symbol(
        self,
        symbol: str | None = None,
        file_path: str | None = None,
    ) -> list[str]:
        """
        Convenience wrapper around discover_test_targets().
        Returns a simple list of test file paths matching the symbol or file.
        Supports Python, TS/JS, and Go test files.
        """
        target_symbols = [symbol] if symbol else None
        target_files = [file_path] if file_path else None
        targets = self.discover_test_targets(
            target_symbols=target_symbols,
            target_files=target_files,
        )
        return self.to_path_list(targets)

    def build_summary(self, targets: list[SymbolAwareTestTarget]) -> dict[str, Any]:
        """Build concise summary for metadata."""
        return {
            "symbol_aware_targeting_used": True,
            "target_count": len(targets),
            "match_type_distribution": self._count_match_types(targets),
            "top_targets": [t.to_dict() for t in targets[:3]],
        }

    # ── Internal helpers ──

    def _find_all_test_files(self) -> list[str]:
        """Find all test files in repo (cached). Supports Python, TS/JS, Go."""
        if self._test_files_cache is not None:
            return self._test_files_cache

        test_files = []
        with suppress(OSError):  # walk on permission-restricted dirs
            for dirpath, dirnames, filenames in os.walk(self._repo_root):
                dirnames[:] = [
                    d for d in dirnames if not d.startswith(".") and d not in {"__pycache__", "node_modules"}
                ]
                for fname in filenames:
                    if self._is_test_filename(fname):
                        rel = os.path.relpath(os.path.join(dirpath, fname), self._repo_root)
                        test_files.append(rel)

        self._test_files_cache = test_files
        return test_files

    def _is_test_filename(self, filename: str) -> bool:
        """Check if filename matches any known test naming convention."""
        # Python: test_foo.py / foo_test.py
        if LanguageId.from_path(filename) is LanguageId.PYTHON:
            return filename.startswith("test_") or filename.endswith("_test.py")
        # TS/JS: foo.test.ts / foo.spec.tsx / test_foo.ts
        if filename.endswith((".ts", ".tsx", ".js", ".jsx")):
            base, _ = os.path.splitext(filename)
            if base.endswith((".test", ".spec")):
                return True
            return filename.startswith("test_")
        # Go: foo_test.go
        return bool(filename.endswith("_test.go"))

    def _is_test_file(self, path: str) -> bool:
        return self._is_test_filename(os.path.basename(path))

    def _read_test_file(self, test_file: str) -> str:
        """Whole contents of *test_file*, memoised for this finder instance.

        Read in full, not to a byte cap. The cap used to be 50 000 characters,
        which silently made the back half of a large test file invisible to
        symbol matching: this repo's ``test_shell_danger_policy.py`` is 93 KB,
        so a symbol exercised in its second half reported "no related tests" —
        the answer that reads as "none exist" rather than "not looked at".

        Memoised because the caller scans up to 10 symbols against every test
        file, which re-read each one once per symbol. One read each instead
        makes the full read cheaper than the capped one was.
        """
        cached = self._content_cache.get(test_file)
        if cached is not None:
            return cached
        try:
            abs_path = os.path.join(self._repo_root, test_file)
            if not os.path.isfile(abs_path):
                content = ""
            else:
                with open(abs_path, errors="replace", encoding="utf-8") as f:
                    content = f.read()
        except OSError:
            content = ""
        self._content_cache[test_file] = content
        return content

    def _file_references_symbol(self, test_file: str, symbol: str) -> bool:
        """Check if test file contains reference to symbol (lightweight grep)."""
        try:
            content = self._read_test_file(test_file)
            if not content:
                return False
            # Look for symbol as whole word
            idx = content.find(symbol)
            while idx != -1:
                before = idx == 0 or not (content[idx - 1].isalnum() or content[idx - 1] == "_")
                after = idx + len(symbol) >= len(content) or not (
                    content[idx + len(symbol)].isalnum() or content[idx + len(symbol)] == "_"
                )
                if before and after:
                    return True
                idx = content.find(symbol, idx + 1)
        except (OSError, UnicodeDecodeError):  # unreadable test file
            return False
        else:
            return False

    def _file_imports_module(self, test_file: str, module_name: str) -> bool:
        """Check if test file imports the given module."""
        try:
            content = self._read_test_file(test_file)
            if not content:
                return False
            # Simple import check
            return (
                f"import {module_name}" in content
                or f"from {module_name}" in content
                or f"from {module_name.replace('/', '.')}" in content
            )
        except (OSError, UnicodeDecodeError):  # unreadable test file
            return False

    def _extract_module_names(self, file_paths: list[str]) -> list[str]:
        """Extract Python module names from file paths."""
        modules = []
        for f in file_paths:
            if LanguageId.from_path(f) is LanguageId.PYTHON:
                # Convert path to module: external_llm/agent/foo.py → external_llm.agent.foo
                mod = f[:-3].replace("/", ".").replace("\\", ".")
                modules.append(mod)
                # Also add just the filename stem
                stem = os.path.splitext(os.path.basename(f))[0]
                if stem not in modules:
                    modules.append(stem)
        return modules

    def _find_corresponding_test(self, source_file: str, test_files: list[str]) -> str | None:
        """Find test file corresponding to source file by naming convention (Python, TS/JS, Go)."""
        base = os.path.splitext(os.path.basename(source_file))[0]
        candidates = [
            # Python
            f"test_{base}.py",
            f"{base}_test.py",
            # TS/JS
            f"{base}.test.ts",
            f"{base}.test.tsx",
            f"{base}.test.js",
            f"{base}.spec.ts",
            f"{base}.spec.tsx",
            f"test_{base}.ts",
            f"test_{base}.tsx",
            f"test_{base}.js",
            # Go
            f"{base}_test.go",
        ]
        for tf in test_files:
            tf_name = os.path.basename(tf)
            if tf_name in candidates:
                return tf
        return None

    def _count_match_types(self, targets: list[SymbolAwareTestTarget]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in targets:
            counts[t.match_type] = counts.get(t.match_type, 0) + 1
        return counts
