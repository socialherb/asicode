"""
Test Dependency Graph: pre-computed mapping between source modules/files and test files.

Built by scanning test files' imports once, then providing fast lookup.
"""
import ast
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ..languages import LanguageId

logger = logging.getLogger(__name__)

# Relation weights (for scoring)
RELATION_WEIGHTS = {
    "direct_import": 1.0,
    "filename_match": 0.7,
    "same_package": 0.4,
}


@dataclass
class TestDependencyEdge:
    """A single test → target relationship."""
    __test__ = False  # Not a pytest test class — domain model
    test_path: str
    target_module: Optional[str] = None
    target_file: Optional[str] = None
    relation: str = "direct_import"  # "direct_import" | "filename_match" | "same_package"
    weight: float = 1.0
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_path": self.test_path,
            "target_module": self.target_module,
            "target_file": self.target_file,
            "relation": self.relation,
            "weight": self.weight,
        }


@dataclass
class TestCoverageCandidate:
    """A test file identified as covering impacted code."""
    __test__ = False  # Not a pytest test class — domain model
    test_path: str
    coverage_score: float = 0.0
    matched_modules: list[str] = field(default_factory=list)
    matched_files: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_path": self.test_path,
            "coverage_score": round(self.coverage_score, 3),
            "matched_modules": self.matched_modules[:3],
            "matched_files": self.matched_files[:3],
            "relation_types": self.relation_types,
        }


class TestDependencyGraph:
    """Pre-computed bidirectional mapping between source code and test files.

    Provides:
    - module_to_tests: source module → list of test files that import it
    - file_to_tests: source file → list of test files related to it
    - test_to_modules: test file → list of modules it imports
    - test_to_files: test file → list of source files it relates to
    """
    __test__ = False  # Not a pytest test class — domain model

    def __init__(self):
        self.module_to_tests: dict[str, list[str]] = defaultdict(list)
        self.test_to_modules: dict[str, list[str]] = defaultdict(list)
        self.file_to_tests: dict[str, list[str]] = defaultdict(list)
        self.test_to_files: dict[str, list[str]] = defaultdict(list)
        self.edges: list[TestDependencyEdge] = []
        self.metadata: dict[str, Any] = {}

    def get_tests_for_module(self, module_name: str) -> list[str]:
        """Get test files that import or relate to a module."""
        return list(self.module_to_tests.get(module_name, []))

    def get_tests_for_file(self, file_path: str) -> list[str]:
        """Get test files related to a source file."""
        results = set(self.file_to_tests.get(file_path, []))
        # Also try module-based lookup
        module = self._file_to_module(file_path)
        if module:
            results.update(self.module_to_tests.get(module, []))
        return list(results)

    def get_tests_for_impact_set(self, impact_set) -> list[TestCoverageCandidate]:
        """
        Get test candidates covering an ImpactSet's impacted modules/files.

        Returns ranked TestCoverageCandidate list.
        """
        candidates: dict[str, TestCoverageCandidate] = {}

        try:
            # From impacted modules
            for mod in getattr(impact_set, 'impacted_modules', []):
                for test_path in self.get_tests_for_module(mod):
                    c = candidates.setdefault(test_path, TestCoverageCandidate(test_path=test_path))
                    c.matched_modules.append(mod)
                    if "direct_import" not in c.relation_types:
                        c.relation_types.append("direct_import")
                    c.coverage_score = max(c.coverage_score, RELATION_WEIGHTS["direct_import"])
                    if "IMPACTED_MODULE" not in c.reason_codes:
                        c.reason_codes.append("IMPACTED_MODULE")

            # From impacted files
            for f in getattr(impact_set, 'impacted_files', []):
                for test_path in self.get_tests_for_file(f):
                    c = candidates.setdefault(test_path, TestCoverageCandidate(test_path=test_path))
                    if f not in c.matched_files:
                        c.matched_files.append(f)
                    # Use the best relation type for this file
                    file_edges = [e for e in self.edges if e.test_path == test_path and e.target_file == f]
                    for edge in file_edges:
                        if edge.relation not in c.relation_types:
                            c.relation_types.append(edge.relation)
                        c.coverage_score = max(c.coverage_score, edge.weight)
                    if not file_edges:
                        if "filename_match" not in c.relation_types:
                            c.relation_types.append("filename_match")
                        c.coverage_score = max(c.coverage_score, RELATION_WEIGHTS["filename_match"])
                    if "IMPACTED_FILE" not in c.reason_codes:
                        c.reason_codes.append("IMPACTED_FILE")

        except Exception as e:
            logger.debug("get_tests_for_impact_set failed: %s", e)

        # Sort by coverage score descending
        ranked = sorted(candidates.values(), key=lambda c: -c.coverage_score)
        return ranked

    def get_summary(self) -> dict[str, Any]:
        """Summary for metadata."""
        relation_counts: dict[str, int] = defaultdict(int)
        for edge in self.edges:
            relation_counts[edge.relation] += 1

        return {
            "used": True,
            "total_edges": len(self.edges),
            "module_count": len(self.module_to_tests),
            "test_file_count": len(self.test_to_modules),
            "direct_import_count": relation_counts.get("direct_import", 0),
            "filename_match_count": relation_counts.get("filename_match", 0),
            "same_package_count": relation_counts.get("same_package", 0),
        }

    @staticmethod
    def _file_to_module(file_path: str) -> Optional[str]:
        if file_path and LanguageId.from_path(file_path) is LanguageId.PYTHON:
            return file_path[:-3].replace('/', '.').replace('\\', '.')
        return None


class DependencyGraphBuilder:
    """
    Builds TestDependencyGraph by scanning test files and extracting import relationships.

    Uses AST parsing for imports (with regex fallback on parse failure).
    Also adds filename-convention and package-proximity edges.
    """

    def __init__(self, repo_root: str):
        self._repo_root = repo_root
        self._cached_graph: Optional[TestDependencyGraph] = None
        self._cached_fingerprint: Optional[str] = None

    def build(self, force: bool = False) -> TestDependencyGraph:
        """
        Build or return cached test dependency graph.

        Uses a simple fingerprint (test file count + total mtime) to detect changes.
        """
        try:
            test_files = self._find_test_files()
            fingerprint = self._compute_fingerprint(test_files)

            if not force and self._cached_graph and self._cached_fingerprint == fingerprint:
                return self._cached_graph

            graph = TestDependencyGraph()
            source_files = self._find_source_files()

            for test_file in test_files:
                # 1. Extract imports via AST
                imports = self._extract_imports(test_file)
                for imp_module in imports:
                    edge = TestDependencyEdge(
                        test_path=test_file,
                        target_module=imp_module,
                        relation="direct_import",
                        weight=RELATION_WEIGHTS["direct_import"],
                        reason_codes=["AST_IMPORT"],
                    )
                    # Also resolve to file if possible
                    resolved_file = self._module_to_file(imp_module, source_files)
                    if resolved_file:
                        edge.target_file = resolved_file

                    graph.edges.append(edge)
                    graph.module_to_tests[imp_module].append(test_file)
                    graph.test_to_modules[test_file].append(imp_module)
                    if resolved_file:
                        if test_file not in graph.file_to_tests[resolved_file]:
                            graph.file_to_tests[resolved_file].append(test_file)
                        if resolved_file not in graph.test_to_files[test_file]:
                            graph.test_to_files[test_file].append(resolved_file)

                # 2. Filename convention match
                test_basename = os.path.basename(test_file)
                for src_file in source_files:
                    src_basename = os.path.basename(src_file)
                    if self._filename_matches(test_basename, src_basename):
                        if test_file not in graph.file_to_tests.get(src_file, []):
                            edge = TestDependencyEdge(
                                test_path=test_file,
                                target_file=src_file,
                                relation="filename_match",
                                weight=RELATION_WEIGHTS["filename_match"],
                                reason_codes=["FILENAME_CONVENTION"],
                            )
                            graph.edges.append(edge)
                            graph.file_to_tests[src_file].append(test_file)
                            if src_file not in graph.test_to_files[test_file]:
                                graph.test_to_files[test_file].append(src_file)

                # 3. Same package proximity
                test_pkg = self._get_package(test_file)
                if test_pkg:
                    # Find source package that corresponds to test package
                    # e.g., tests/unit/agent/ → external_llm/agent/
                    src_pkg = self._infer_source_package(test_pkg)
                    if src_pkg:
                        for src_file in source_files:
                            if src_file.startswith(src_pkg) and test_file not in graph.file_to_tests.get(src_file, []):
                                edge = TestDependencyEdge(
                                    test_path=test_file,
                                    target_file=src_file,
                                    relation="same_package",
                                    weight=RELATION_WEIGHTS["same_package"],
                                    reason_codes=["PACKAGE_PROXIMITY"],
                                )
                                graph.edges.append(edge)
                                graph.file_to_tests[src_file].append(test_file)

            graph.metadata = {
                "repo_root": self._repo_root,
                "test_file_count": len(test_files),
                "source_file_count": len(source_files),
                "fingerprint": fingerprint,
            }

            self._cached_graph = graph
            self._cached_fingerprint = fingerprint
            return graph

        except Exception as e:
            logger.debug("TestDependencyGraph build failed: %s", e)
            return TestDependencyGraph()

    def _find_test_files(self) -> list[str]:
        """Find all test files in repo."""
        test_files = []
        for dirpath, dirnames, filenames in os.walk(self._repo_root):
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__' and d != 'node_modules']
            for fname in filenames:
                if LanguageId.from_path(fname) is LanguageId.PYTHON and (fname.startswith('test_') or fname.endswith('_test.py')):
                    rel = os.path.relpath(os.path.join(dirpath, fname), self._repo_root)
                    test_files.append(rel)
        return sorted(test_files)

    def _find_source_files(self) -> list[str]:
        """Find all non-test Python source files."""
        source_files = []
        for dirpath, dirnames, filenames in os.walk(self._repo_root):
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__' and d != 'node_modules' and d != 'tests']
            for fname in filenames:
                if LanguageId.from_path(fname) is LanguageId.PYTHON and not fname.startswith('test_') and not fname.endswith('_test.py'):
                    rel = os.path.relpath(os.path.join(dirpath, fname), self._repo_root)
                    source_files.append(rel)
        return sorted(source_files)

    def _extract_imports(self, test_file: str) -> list[str]:
        """Extract imported module names from test file using AST, with regex fallback."""
        abs_path = os.path.join(self._repo_root, test_file)
        imports: set[str] = set()

        try:
            with open(abs_path, errors='ignore') as f:
                content = f.read(100000)

            # Try AST first
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
            except SyntaxError:
                # Regex fallback
                for m in re.finditer(r'^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))', content, re.MULTILINE):
                    mod = m.group(1) or m.group(2)
                    if mod:
                        imports.add(mod)
        except Exception:
            pass

        # Filter: only keep imports that look like project modules (not stdlib)
        project_imports = []
        for imp in imports:
            # Simple heuristic: keep if has dots or starts with known project prefix
            if '.' in imp or imp.startswith('external_llm') or imp.startswith('tests'):
                project_imports.append(imp)

        return sorted(project_imports)

    def _module_to_file(self, module: str, source_files: list[str]) -> Optional[str]:
        """Try to resolve module name to source file path."""
        # external_llm.agent.planner_lane.planner_agent → external_llm/agent/planner_agent.py
        candidate = module.replace('.', '/') + '.py'
        if candidate in source_files:
            return candidate
        # Try __init__.py
        init_candidate = module.replace('.', '/') + '/__init__.py'
        if init_candidate in source_files:
            return init_candidate
        return None

    def _filename_matches(self, test_name: str, source_name: str) -> bool:
        """Check if test file name matches source file by convention."""
        source_stem = source_name[:-3] if LanguageId.from_path(source_name) is LanguageId.PYTHON else source_name
        return test_name in (f"test_{source_stem}.py", f"{source_stem}_test.py")

    def _get_package(self, file_path: str) -> Optional[str]:
        """Get package directory from file path."""
        d = os.path.dirname(file_path)
        return d if d else None

    def _infer_source_package(self, test_pkg: str) -> Optional[str]:
        """Infer source package from test package path."""
        # tests/unit/agent/ → external_llm/agent/ (heuristic)
        # Remove tests/ prefix and try common patterns
        parts = test_pkg.replace('\\', '/').split('/')
        # Look for 'tests' prefix
        if 'tests' in parts:
            idx = parts.index('tests')
            # Remove tests and subdirectory (unit/integration)
            remaining = parts[idx + 1:]
            if remaining and remaining[0] in ('unit', 'integration', 'e2e'):
                remaining = remaining[1:]
            if remaining:
                return '/'.join(remaining) + '/'
        return None

    def _compute_fingerprint(self, test_files: list[str]) -> str:
        """Simple fingerprint for cache invalidation."""
        import hashlib
        total_mtime = 0
        for tf in test_files:
            try:
                total_mtime += int(os.path.getmtime(os.path.join(self._repo_root, tf)))
            except Exception:
                pass
        raw = f"{len(test_files)}:{total_mtime}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
