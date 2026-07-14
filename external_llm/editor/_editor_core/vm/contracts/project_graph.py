"""project_graph.py — Build a language-agnostic project-level symbol + import graph.

Indexes all files to produce:
- definitions: symbol → defining files
- usages: symbol → files that reference it
- import_graph: file → set of files it imports from
- dependents: file → set of files that import from it

This is the foundation for cross-file impact analysis.

Ported from ts_vm/multifile/project_graph.py with TSModule → CodeContext.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.languages import LanguageId  # S8 fix: missing module-level import


@dataclass
class ProjectGraph:
    """Global project-level symbol graph.

    Maps symbols to their defining/using files, enabling
    cross-file impact analysis.
    """

    # symbol name → list of file paths where it's defined
    definitions: dict[str, list[str]] = field(default_factory=dict)

    # symbol name → list of file paths where it's used (imported/referenced)
    usages: dict[str, list[str]] = field(default_factory=dict)

    # file path → set of files it imports from
    import_graph: dict[str, set[str]] = field(default_factory=dict)

    # file path → set of files that import from it
    dependents: dict[str, set[str]] = field(default_factory=dict)

    # file path → source code (for call-site analysis)
    file_sources: dict[str, str] = field(default_factory=dict)

    def files_using(self, symbol: str) -> list[str]:
        return self.usages.get(symbol, [])

class ProjectGraphBuilder:
    """Builds a ProjectGraph from a mapping of {path: code}."""

    def build(self, files: dict[str, str]) -> ProjectGraph:
        """Build the full project graph.

        Args:
            files: Mapping of file path → source code.

        Returns:
            ProjectGraph with definitions, usages, import/dependent graphs.
        """
        graph = ProjectGraph()
        graph.file_sources = dict(files)

        # Phase 1: Build symbol index for all files
        for path, code in files.items():
            ctx = self._build_context(code, path)
            if ctx is None:
                continue
            self._index_symbols(graph, ctx, path)
            self._index_imports(graph, ctx, path)

        # Phase 2: Build import/dependent graph
        self._build_import_graph(graph, files)

        return graph

    def _build_context(self, code: str, path: str) -> Optional[CodeContext]:
        """Create a CodeContext for a file, if the language is supported."""
        from external_llm.languages.models import LanguageId

        ext_map: dict[str, LanguageId] = {
            ".py": LanguageId.PYTHON,
            ".java": LanguageId.JAVA,
            ".kt": LanguageId.KOTLIN,
            ".kts": LanguageId.KOTLIN,
            ".go": LanguageId.GO,
            ".ts": LanguageId.TYPESCRIPT,
            ".js": LanguageId.JAVASCRIPT,
        }
        _, ext = os.path.splitext(path)
        lang = ext_map.get(ext.lower())
        if lang is None:
            return None
        return CodeContext(code, path, lang)

    def _index_symbols(
        self, graph: ProjectGraph, ctx: CodeContext, path: str,
    ) -> None:
        """Index symbol definitions from a file."""
        symbols = ctx.get_symbols_by_kind("function") + ctx.get_symbols_by_kind("class")
        for sym in symbols:
            name = sym.name
            graph.definitions.setdefault(name, [])
            if path not in graph.definitions[name]:
                graph.definitions[name].append(path)

    def _index_imports(
        self, graph: ProjectGraph, ctx: CodeContext, path: str,
    ) -> None:
        """Index imported symbols from a file."""
        imports = ctx.get_imports()
        for imp in imports:
            # Try to extract individual specifiers
            stmt = imp.statement
            # Python: "from X import Y" or "import X"
            # Java: "import X.Y"
            # Go: "import \"X\""
            names = self._extract_import_names(stmt)
            for name in names:
                graph.usages.setdefault(name, [])
                if path not in graph.usages[name]:
                    graph.usages[name].append(path)

    def _build_import_graph(
        self, graph: ProjectGraph, files: dict[str, str],
    ) -> None:
        """Build file-level import and dependent graphs."""
        path_set = set(files.keys())

        for path in files:
            ctx = self._build_context(files[path], path)
            if ctx is None:
                continue
            imports_from: set[str] = set()
            file_imports = ctx.get_imports()
            for imp in file_imports:
                source = imp.source
                resolved = self._resolve_import(path, source, path_set)
                if resolved:
                    imports_from.add(resolved)

            graph.import_graph[path] = imports_from
            for dep in imports_from:
                graph.dependents.setdefault(dep, set())
                graph.dependents[dep].add(path)

    def _extract_import_names(self, import_stmt: str) -> list[str]:
        """Extract imported symbol names from an import statement."""
        names = []
        stmt = import_stmt.strip()

        # Python: "from X import Y, Z" or "import X"
        if stmt.startswith("from "):
            parts = stmt.split(" import ", 1)
            if len(parts) == 2:
                for item in parts[1].split(","):
                    name = item.strip().split(" as ")[0].strip()
                    if name:
                        names.append(name)
        elif stmt.startswith("import "):
            rest = stmt[7:].strip()
            for item in rest.split(","):
                name = item.strip().split(" as ")[0].strip()
                # Handle "import X.Y.Z" — take the first component
                name = name.split(".")[0] if "." in name and not name.startswith('"') else name
                if name:
                    names.append(name)

        # Java: "import com.example.Foo;" or "import com.example.*;"
        # Just extract the last component
        if "import " in stmt and not stmt.startswith("from "):
            for line in stmt.split("\n"):
                line = line.strip().rstrip(";")
                if line.startswith("import "):
                    imp = line[7:].strip()
                    if imp and not imp.endswith(".*"):
                        last = imp.split(".")[-1]
                        if last:
                            names.append(last)

        return names

    def _resolve_import(
        self, from_path: str, source: str, path_set: set[str],
    ) -> Optional[str]:
        """Resolve a relative import to an actual file path."""
        if not source.startswith("."):
            return None  # external module, skip

        dir_path = os.path.dirname(from_path)
        base = os.path.normpath(os.path.join(dir_path, source))

        # Try common extensions for all languages
        for ext in ("", ".py", ".java", ".kt", ".kts", ".go",
                     ".ts", ".tsx", ".js", ".jsx",
                     "/__init__.py", "/index.ts", "/index.js"):
            candidate = base + ext
            if candidate in path_set:
                return candidate

        return None
