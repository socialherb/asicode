"""
Dependency Graph Builder for Enhanced Context

Tracks:
- Function call relationships
- Import dependencies
- Class inheritance
- Module dependencies
- Call chains
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .code_analyzer import CodeAnalysis, CodeAnalyzer
@dataclass
class CallRelation:
    """Represents a function call relationship"""
    caller: str  # function/method name
    callee: str  # called function/method
    file_path: str
    line_number: int = 0


@dataclass
class DependencyGraph:
    """Complete dependency graph for a project"""

    # Function calls: {function_name: [functions_it_calls]}
    calls: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    # Reverse: {function_name: [functions_that_call_it]}
    called_by: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    # File dependencies: {file_path: [imported_files]}
    file_imports: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    # Class inheritance: {class_name: [base_classes]}
    inheritance: dict[str, list[str]] = field(default_factory=dict)

    # Module-level: {module: [symbols_exported]}
    exports: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


class DependencyGraphBuilder:
    """
    Builds dependency graph for a Python project

    Analyzes:
    - Who calls whom
    - File import relationships
    - Class inheritance chains
    - Module exports/imports
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.analyzer = CodeAnalyzer()
        self._analysis_cache: dict[Path, CodeAnalysis] = {}

    def build_graph(self, target_file: Path, max_depth: int = 2) -> DependencyGraph:
        """
        Build dependency graph starting from target file

        Args:
            target_file: Starting point
            max_depth: How deep to traverse dependencies

        Returns:
            DependencyGraph
        """
        graph = DependencyGraph()
        visited = set()

        self._build_recursive(target_file, graph, visited, depth=0, max_depth=max_depth)

        return graph

    def _build_recursive(
        self,
        file_path: Path,
        graph: DependencyGraph,
        visited: set[Path],
        depth: int,
        max_depth: int,
    ):
        """Recursively build graph"""
        if depth > max_depth or file_path in visited:
            return

        visited.add(file_path)

        # Analyze file
        analysis = self._get_analysis(file_path)
        if not analysis:
            return

        # Track function calls within file
        self._track_internal_calls(file_path, analysis, graph)

        # Track class inheritance
        self._track_inheritance(analysis, graph)

        # Track imports and follow them
        imported_files = self._track_imports(file_path, analysis, graph)

        # Recurse into imported files
        for imported_file in imported_files:
            if imported_file.exists() and imported_file.suffix == '.py':
                self._build_recursive(imported_file, graph, visited, depth + 1, max_depth)

    def _get_analysis(self, file_path: Path) -> Optional[CodeAnalysis]:
        """Get or cache code analysis"""
        if file_path not in self._analysis_cache:
            self._analysis_cache[file_path] = self.analyzer.analyze_file(file_path)
        return self._analysis_cache[file_path]

    def _track_internal_calls(self, file_path: Path, analysis: CodeAnalysis, graph: DependencyGraph):
        """Track function calls within a file"""
        file_str = str(file_path.relative_to(self.repo_root))

        # Build map of defined functions
        defined = set()
        for func in analysis.functions:
            defined.add(func.name)
        for cls in analysis.classes:
            for method in cls.methods:
                defined.add(f"{cls.name}.{method.name}")

        # Track calls
        for func in analysis.functions:
            func_name = f"{file_str}:{func.name}"

            # We know this function exists in this file
            # The analysis.calls contains all function calls made in the entire file
            # We'd need per-function call tracking, which requires more detailed AST walking
            # For now, associate file-level calls with each function
            for call in analysis.calls:
                if call in defined:
                    # Internal call
                    callee = f"{file_str}:{call}"
                    graph.calls[func_name].append(callee)
                    graph.called_by[callee].append(func_name)

    def _track_inheritance(self, analysis: CodeAnalysis, graph: DependencyGraph):
        """Track class inheritance"""
        for cls in analysis.classes:
            if cls.bases:
                graph.inheritance[cls.name] = cls.bases

    def _track_imports(
        self,
        file_path: Path,
        analysis: CodeAnalysis,
        graph: DependencyGraph
    ) -> list[Path]:
        """Track imports and return imported file paths"""
        file_str = str(file_path.relative_to(self.repo_root))
        imported_files = []

        for imp in analysis.imports:
            # Try to resolve import to actual file
            resolved = self._resolve_import(file_path, imp.module)

            if resolved:
                imported_files.append(resolved)
                resolved_str = str(resolved.relative_to(self.repo_root))
                graph.file_imports[file_str].append(resolved_str)

                # Track what's imported from that file
                if imp.names:
                    graph.exports[resolved_str].extend(imp.names)

        return imported_files

    def _resolve_import(self, current_file: Path, module: str) -> Optional[Path]:
        """
        Resolve import to actual file path

        Examples:
            from utils import helper -> utils/helper.py or utils.py
            from .local import func -> ./local.py
        """
        if not module:
            return None

        # Relative import
        if module.startswith('.'):
            base_dir = current_file.parent
            parts = module.lstrip('.').split('.')

            # Count leading dots
            num_dots = len(module) - len(module.lstrip('.'))
            for _ in range(num_dots - 1):
                base_dir = base_dir.parent

            # Try to find the module
            for part in parts:
                if not part:
                    continue

                # Try as file
                candidate = base_dir / f"{part}.py"
                if candidate.exists():
                    return candidate

                # Try as package
                candidate = base_dir / part / "__init__.py"
                if candidate.exists():
                    return candidate

                base_dir = base_dir / part

        # Absolute import (within project)
        else:
            parts = module.split('.')

            # Try from repo root
            for i in range(len(parts), 0, -1):
                path_parts = parts[:i]

                # Try as file
                candidate = self.repo_root / '/'.join(path_parts[:-1]) / f"{path_parts[-1]}.py"
                if candidate.exists():
                    return candidate

                # Try as package
                candidate = self.repo_root / '/'.join(path_parts) / "__init__.py"
                if candidate.exists():
                    return candidate

        return None

    def find_callers(self, graph: DependencyGraph, function: str) -> list[str]:
        """Find all functions that call the given function"""
        return graph.called_by.get(function, [])

    def format_call_graph(self, graph: DependencyGraph, target_function: str, max_items: int = 10) -> str:
        """
        Format call graph as readable text

        Args:
            graph: Dependency graph
            target_function: Function to focus on
            max_items: Maximum items to show

        Returns:
            Formatted string
        """
        lines = []

        # What this function calls
        if target_function in graph.calls:
            calls = graph.calls[target_function][:max_items]
            if calls:
                lines.append(f"{target_function} calls:")
                for callee in calls:
                    lines.append(f"  ├─ {callee}")

        # What calls this function
        if target_function in graph.called_by:
            callers = graph.called_by[target_function][:max_items]
            if callers:
                if lines:
                    lines.append("")
                lines.append("Called by:")
                for caller in callers:
                    lines.append(f"  ├─ {caller}")

        return '\n'.join(lines) if lines else "No call information available"
