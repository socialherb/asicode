"""light_repo_index.py — AST-based lightweight repo index for graph-free exploration.

Provides import/call graph construction and BFS-based structural exploration
when no RepositoryGraphFacade is available. Used by UnifiedGrounder._light_grounding().

Design principles:
- Pure AST: no LLM, no keyword matching
- Import-driven BFS: follows actual code dependency edges
- Structural importance: fan-in (how many files import this) as centrality proxy
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from ..languages import LanguageId
from .config.thresholds import config as _cfg

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Per-file structural facts extracted from AST."""
    path: str
    defs: list[str] = field(default_factory=list)    # function/class names
    imports: list[str] = field(default_factory=list)  # imported module basenames
    calls: list[str] = field(default_factory=list)    # called names


class LightRepoIndex:
    """AST-based lightweight repo index.

    Replaces RepositoryGraphFacade for structural exploration when the graph
    is unavailable (e.g., fresh clone, lightweight environment).

    Usage::

        index = LightRepoIndex("/path/to/repo").build()
        important = index.score_files()        # [(relpath, score), ...]
        related = index.bfs_from(["a.py"], 2)  # BFS via imports
        found = index.find_by_identifier("find_symbol")
    """

    MAX_FILES = _cfg.counts.LIGHT_INDEX_MAX_FILES
    MAX_FILE_SIZE = _cfg.lines.LIGHT_INDEX_FILE_BYTES

    _SKIP_DIRS = frozenset({
        "venv", ".venv", "node_modules", "__pycache__", ".git",
        "dist", "build", ".eggs", ".tox", "htmlcov", "worktrees",
    })

    def __init__(self, repo_root: str):
        self._repo_root = repo_root
        self._files: dict[str, FileInfo] = {}           # relpath → FileInfo
        self._reverse_imports: dict[str, set[str]] = {} # stem → set[relpath importing it]
        self._stem_to_files: dict[str, set[str]] = {}   # stem → set[relpath with that stem]
        self._built = False

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> "LightRepoIndex":
        """Scan repo_root, build AST index. Returns self for chaining."""
        if self._built:
            return self
        py_files = self._collect_py_files()
        for abs_path in py_files:
            rel = os.path.relpath(abs_path, self._repo_root)
            info = self._parse_file(abs_path, rel)
            if info is not None:
                self._files[rel] = info

        # Build reverse-import index: stem → set of files that import it
        for rel, info in self._files.items():
            for imp_stem in info.imports:
                if imp_stem not in self._reverse_imports:
                    self._reverse_imports[imp_stem] = set()
                self._reverse_imports[imp_stem].add(rel)

        # Build stem→file index for O(1) forward-lookup (avoids O(n) per query)
        for rel in self._files:
            stem = os.path.splitext(os.path.basename(rel))[0]
            if stem not in self._stem_to_files:
                self._stem_to_files[stem] = set()
            self._stem_to_files[stem].add(rel)

        self._built = True
        logger.debug("LightRepoIndex: built %d files", len(self._files))
        return self

    def score_files(self) -> list[tuple[str, float]]:
        """Score files by structural importance.

        Score = fan_in * 0.50 + def_count * 0.30 + import_count * 0.20

        fan_in: how many OTHER files import or call symbols in this file.
        This is the AST proxy for graph centrality (hub files = high fan_in).
        """
        # Build def → file mapping (token-level for cross-file call matching)
        def_to_file: dict[str, str] = {}
        for rel, info in self._files.items():
            for d in info.defs:
                def_to_file[d.lower()] = rel

        # Compute fan_in: count calls/imports pointing to each file
        fan_in: dict[str, float] = {rel: 0.0 for rel in self._files}
        for rel, info in self._files.items():
            # call-based fan-in
            for call in info.calls:
                target = def_to_file.get(call.lower())
                if target and target != rel:
                    fan_in[target] = fan_in[target] + 1.0
            # import-based fan-in (O(1) via pre-built stem index)
            for imp_stem in info.imports:
                for other_rel in self._stem_to_files.get(imp_stem, set()):
                    if other_rel != rel:
                        fan_in[other_rel] = fan_in[other_rel] + 0.5

        scores: dict[str, float] = {}
        for rel, info in self._files.items():
            scores[rel] = (
                fan_in.get(rel, 0.0) * 0.50
                + len(info.defs) * 0.30
                + len(info.imports) * 0.20
            )

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def bfs_from(self, seeds: list[str], depth: int = 2) -> list[str]:
        """BFS from seed files via import edges.

        Follows both directions:
        - Forward: files that ``seed`` imports
        - Reverse: files that import ``seed``

        Returns files in BFS order (nearest first), capped at 20.
        """
        visited: set[str] = set()
        frontier: set[str] = set(f for f in seeds if f in self._files)
        result: list[str] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for f in frontier:
                visited.add(f)
                result.append(f)
                info = self._files.get(f)
                if not info:
                    continue
                # Forward: f imports these modules (O(1) via pre-built stem index)
                for imp_stem in info.imports:
                    for other_rel in self._stem_to_files.get(imp_stem, set()):
                        if other_rel not in visited:
                            next_frontier.add(other_rel)
                # Reverse: files that import f
                f_stem = os.path.splitext(os.path.basename(f))[0]
                for other_rel in self._reverse_imports.get(f_stem, set()):
                    if other_rel not in visited:
                        next_frontier.add(other_rel)
            frontier = next_frontier - visited

        return result[:20]

    def find_by_identifier(self, identifier: str) -> list[str]:
        """Find files containing a definition matching identifier.

        Uses token-level matching (underscore split) so "find_symbol"
        matches functions named "find_symbol" or "SymbolFinder" etc.
        """
        result: list[str] = []
        ident_lower = identifier.lower()
        id_tokens = frozenset(ident_lower.split('_'))

        for rel, info in self._files.items():
            for d in info.defs:
                d_lower = d.lower()
                d_tokens = frozenset(d_lower.split('_')) | {d_lower}
                # exact match or token-level overlap (skip ≤2 char: too many false positives)
                if ident_lower == d_lower or (len(ident_lower) > 2 and id_tokens & d_tokens):
                    result.append(rel)
                    break

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect_py_files(self) -> list[str]:
        """Walk repo_root, collect .py files sorted by recency (most recent first)."""
        py_files: list[str] = []
        for root, dirs, files in os.walk(self._repo_root):
            dirs[:] = [d for d in dirs if d not in self._SKIP_DIRS]
            for fname in files:
                if LanguageId.from_path(fname) is LanguageId.PYTHON:
                    py_files.append(os.path.join(root, fname))

        # Most recently modified first → likely more relevant
        py_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return py_files[: self.MAX_FILES]

    def _parse_file(self, abs_path: str, rel: str) -> Optional[FileInfo]:
        """Parse one file, extract defs/imports/calls via AST."""
        try:
            if os.path.getsize(abs_path) > self.MAX_FILE_SIZE:
                return None
            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                source = fh.read()
            tree = ast.parse(source, filename=abs_path)
        except (OSError, SyntaxError, ValueError):
            return None

        info = FileInfo(path=rel)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                info.defs.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    # "import external_llm.agent.agent_loop" → "agent_loop"
                    info.imports.append(alias.name.split(".")[-1])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    info.imports.append(node.module.split(".")[-1])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    info.calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    info.calls.append(node.func.attr)

        return info
