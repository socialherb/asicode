"""Analysis and exploration tool handlers for ToolRegistry."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from ...languages import LanguageId

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)


class AnalysisToolsMixin:
    """Mixin providing analysis/exploration tool implementations for ToolRegistry."""

    def _tool_get_project_info(self, args: dict[str, Any]) -> "ToolResult":
        try:
            from external_llm.project_analyzer import ProjectAnalyzer
            analyzer = ProjectAnalyzer(self.repo_root)
            structure = analyzer.analyze()

            info_parts = []
            if structure.languages:
                info_parts.append(f"Languages: {', '.join(structure.languages)}")
            if structure.frameworks:
                info_parts.append(f"Frameworks: {', '.join(structure.frameworks)}")
            elif structure.framework:
                info_parts.append(f"Framework: {structure.framework}")
            if structure.project_types:
                info_parts.append(f"Project types: {', '.join(structure.project_types)}")
            if structure.entry_points:
                # Show pyproject.toml/setup entries first (they're more informative)
                info_parts.append(f"Entry points: {', '.join(structure.entry_points[:10])}")
            if structure.test_dir:
                info_parts.append(f"Test directory: {structure.test_dir}")
            if structure.naming_style:
                info_parts.append(f"Naming style: {structure.naming_style}")
            if structure.common_imports:
                info_parts.append(f"Common imports: {', '.join(structure.common_imports[:10])}")
            if structure.directories:
                dir_summary = []
                for purpose, paths in list(structure.directories.items())[:5]:
                    # Show more entries for the catch-all 'other' bucket so real
                    # packages (e.g. webapp/, docs/) aren't hidden behind the
                    # first three noise dirs.
                    shown = paths[:8] if purpose == 'other' else paths[:3]
                    dir_summary.append(f"  {purpose}: {', '.join(str(p) for p in shown)}")
                info_parts.append("Directories:\n" + "\n".join(dir_summary))

            content = "\n".join(info_parts) if info_parts else "Unable to determine project structure"
            return self._make_result(
                ok=True,
                content=content,
                metadata={
                    "languages": structure.languages,
                    "primary_language": structure.primary_language,
                    "frameworks": structure.frameworks or ([structure.framework] if structure.framework else []),
                    "project_types": structure.project_types,
                    "entry_points": structure.entry_points,
                },
            )
        except Exception as e:
            logger.warning("get_project_info failed: %s", e)
            return self._make_result(ok=True, content=f"Project info unavailable: {e}")

    def _tool_analyze_change_impact(self, args: dict[str, Any]) -> "ToolResult":
        """Analyze impact of changing a symbol using graph traversal."""
        symbol = str(args.get("symbol", "")).strip()
        file_path = str(args.get("file_path", "")).strip() or None
        depth = int(args.get("depth", 2))
        direction = str(args.get("direction", "both")).strip().lower()
        include_importers = args.get("include_importers", True)
        limit = int(args.get("limit", 30))

        if not symbol:
            return self._make_result(ok=False, content="", error="'symbol' is required")

        if file_path:
            try:
                from pathlib import Path as _Path
                fp = _Path(file_path)
                if fp.is_absolute():
                    file_path = str(fp.relative_to(self.repo_root))
            except (ValueError, Exception):
                pass

        lines: list[str] = [f"## Impact analysis for `{symbol}`"]
        metadata: dict[str, Any] = {"symbol": symbol}

        try:
            # 1. Callers (upstream)
            callers = [] if direction == "downstream" else self._call_graph.get_callers(symbol, file_path)
            if callers:
                # Dedup by (caller_symbol, caller_file, caller_line) — graph index may
                # store the same edge under multiple keys (bare vs qualified name).
                seen_caller: set[tuple[str, str, int]] = set()
                unique_callers: list = []
                for c in callers:
                    key = (c.caller_symbol, c.caller_file or "?", c.caller_line)
                    if key not in seen_caller:
                        seen_caller.add(key)
                        unique_callers.append(c)
                lines.append(f"\n### Callers ({len(unique_callers)})")
                seen_files: set[str] = set()
                for c in unique_callers[:limit]:
                    f = c.caller_file or "?"
                    seen_files.add(f)
                    lines.append(f"  - `{c.caller_symbol}` → {f}:{c.caller_line}")
                metadata["caller_files"] = sorted(seen_files)
                metadata["caller_count"] = len(unique_callers)
            else:
                lines.append("\n### Callers (none found)")

            # 2. Callees (downstream)
            callees = [] if direction == "upstream" else self._call_graph.get_callees(symbol, file_path)
            if callees:
                # Dedup by (callee_symbol, callee_file, callee_line)
                seen_callee: set[tuple[str, str, int]] = set()
                unique_callees: list = []
                for c in callees:
                    f = c.callee_file or c.caller_file or "?"
                    key = (c.callee_symbol, f, c.callee_line)
                    if key not in seen_callee:
                        seen_callee.add(key)
                        unique_callees.append(c)
                lines.append(f"\n### Callees ({len(unique_callees)})")
                seen_files = set()
                for c in unique_callees[:limit]:
                    f = c.callee_file or c.caller_file or "?"
                    seen_files.add(f)
                    lines.append(f"  - `{c.callee_symbol}` → {f}:{c.callee_line}")
                metadata["callee_files"] = sorted(seen_files)
                metadata["callee_count"] = len(unique_callees)
            else:
                lines.append("\n### Callees (none found)")

            # 3. Importers (reverse dependencies)
            if include_importers:
                # Get the file where this symbol is defined
                sym_file = self._call_graph.get_symbol_file(symbol) if hasattr(self._call_graph, 'get_symbol_file') else None
                if not sym_file and file_path:
                    sym_file = file_path
                if sym_file:
                    try:
                        importers = self._call_graph.get_importers(sym_file)
                        if importers:
                            lines.append(f"\n### Importers ({len(importers)})")
                            for imp in sorted(importers)[:limit]:
                                lines.append(f"  - `{imp}`")
                            metadata["importer_count"] = len(importers)
                            metadata["importer_files"] = sorted(importers)[:limit]
                    except Exception:
                        pass

            # 4. File dependencies
            if include_importers and sym_file:
                try:
                    deps = self._call_graph.get_file_dependencies(sym_file)
                    if deps:
                        lines.append(f"\n### File dependencies ({len(deps)})")
                        for d in deps[:limit]:
                            lines.append(f"  - `{d.imported}` ({d.import_type})")
                except Exception:
                    pass

            # 5. Summary
            total_files = len(metadata.get("caller_files", [])) + len(metadata.get("callee_files", [])) + len(metadata.get("importer_files", []))
            lines.append(f"\n---\n**Summary**: {metadata.get('caller_count', 0)} callers, {metadata.get('callee_count', 0)} callees, ~{total_files} affected files")

            metadata["depth"] = depth
            metadata["direction"] = direction
            return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

        except Exception as e:
            logger.warning(f"analyze_change_impact error for {symbol!r}: {e}")
            return self._make_result(ok=False, content="", error=f"analyze_change_impact error: {e}")

    # Scan target extensions — languages supported by scanners via tree-sitter
    _SCAN_EXTS = (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt")
    _SCAN_SKIP_DIRS = frozenset({
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    })
    _SCAN_FILE_CAP = 4000

    def _walk_scan_files(self, root: str) -> list:
        """Collect scannable source files under *root* (repo-relative paths)."""
        out: list = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in self._SCAN_SKIP_DIRS and not d.startswith(".")
            ]
            for fn in sorted(filenames):
                if not fn.endswith(self._SCAN_EXTS):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), self.repo_root)
                out.append(rel)
                if len(out) >= self._SCAN_FILE_CAP:
                    logger.warning(
                        "[STRUCTURAL_SCAN] file cap %d reached under %s — truncating",
                        self._SCAN_FILE_CAP, root,
                    )
                    return out
        return out

    def _tool_run_structural_scan(self, args: dict[str, Any]) -> "ToolResult":
        """Run structural analysis scanner(s) from ScannerRegistry."""
        scanner_name = str(args.get("scanner", "")).strip()
        scan_path = str(args.get("path", "")).strip() or None
        max_results = int(args.get("max_results", 30))

        if not scanner_name:
            return self._make_result(ok=False, content="", error="'scanner' is required")

        try:
            from external_llm.agent.scanner_registry import get_registry

            registry = get_registry()
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Failed to load scanner registry: {e}")

        if not scanner_name or scanner_name == "all":
            scanners_to_run = [
                n for n in registry.list_names()
                if not getattr(registry.get_spec(n), "skip_in_all_mode", False)
            ]
        else:
            spec = registry.get_spec(scanner_name)
            if spec is None:
                available = ", ".join(sorted(registry.list_names()))
                return self._make_result(
                    ok=False, content="",
                    error=f"Unknown scanner: {scanner_name!r}. Available: {available}"
                )
            scanners_to_run = [scanner_name]

        # Build file path list.  Scanners iterate file_paths as FILES — they do
        # not walk directories, so a dir/empty path must be expanded here or
        # the scan silently returns 0 candidates ("scanned the project, found
        # nothing") while having scanned nothing.
        if scan_path:
            abs_scan = os.path.join(self.repo_root, scan_path) if not os.path.isabs(scan_path) else scan_path
            if os.path.isfile(abs_scan):
                file_paths = [scan_path]
            elif os.path.isdir(abs_scan):
                file_paths = self._walk_scan_files(abs_scan)
            else:
                return self._make_result(ok=False, content="", error=f"Path not found: {scan_path}")
        else:
            file_paths = self._walk_scan_files(self.repo_root)
        if not file_paths:
            return self._make_result(ok=True, content="No scannable source files found.")

        # Cross-file reachability: same signal the planner's RUN_SCANNER path
        # injects.  Without it, dead-code scanners run in private-only mode
        # AND miss cross-file imports of private symbols (false "dead").
        _cross_refs: "Optional[set]" = None
        try:
            from external_llm.analysis.cross_file_refs import (
                compute_cross_file_referenced_names_light,
            )
            _cross_refs = compute_cross_file_referenced_names_light(
                getattr(self, "_call_graph", None), self.repo_root, file_paths,
            )
        except Exception:
            logger.debug("[STRUCTURAL_SCAN] cross-file refs unavailable — conservative mode", exc_info=True)

        all_lines: list[str] = [f"Scanned {len(file_paths)} file(s)."]
        total_candidates = 0
        total_affected = set()
        per_scanner: list[dict] = []

        for name in scanners_to_run:
            _spec_n = registry.get_spec(name)

            # ── Language capability gate ────────────────────────────────────
            # When a scanner declares ``supported_languages``, skip it entirely
            # if NONE of the discovered files are in a language it can analyze.
            # This avoids wasting a scan pass (and emitting misleading
            # "Candidates: 0" lines) when e.g. ``scanner="all"`` runs over a
            # pure-Go repo and a Python-only scanner has nothing to work on.
            # run() re-checks per-file regardless, so this is a UX/efficiency
            # fast-path, not a correctness gate.
            if _spec_n is not None and _spec_n.supported_languages is not None:
                _has_supported_file = any(
                    LanguageId.from_path(p) in _spec_n.supported_languages
                    for p in file_paths
                )
                if not _has_supported_file:
                    _present_langs = sorted({
                        LanguageId.from_path(p).value for p in file_paths
                    }) or ["none"]
                    all_lines.append(
                        f"\n## {name}\n"
                        f"Description: {_spec_n.description}\n"
                        f"Skipped: scanner supports "
                        f"{sorted(_item_.value for _item_ in _spec_n.supported_languages)} "
                        f"but scan set only contains {_present_langs}"
                    )
                    per_scanner.append({
                        "scanner": name, "skipped_language_mismatch": True,
                        "supported": sorted(_item_.value for _item_ in _spec_n.supported_languages),
                        "present": _present_langs,
                    })
                    continue

            # ── Graph-required scanner skip ─────────────────────────────────
            # Scanners that declare ``graph_required_for_results=True`` hard-require
            # the call graph to produce any output (e.g. broken_contract's
            # caller-asymmetry check). Without the graph they silently return 0
            # candidates — indistinguishable from a genuine clean scan. Surface the
            # skip explicitly so the caller knows the scan was *not* performed.
            # (vulture also has requires_graph=True but degrades gracefully when the
            # graph is absent, so it does NOT set graph_required_for_results and is
            # never skipped here.)
            if (_spec_n is not None
                    and getattr(_spec_n, "requires_graph", False)
                    and getattr(_spec_n, "graph_required_for_results", False)
                    and getattr(self, "_call_graph", None) is None):
                all_lines.append(
                    f"\n## {name}\n"
                    f"Description: {_spec_n.description}\n"
                    f"Skipped: scanner requires the call graph which is unavailable "
                    f"(standalone scan has no live graph)"
                )
                per_scanner.append({
                    "scanner": name, "skipped_requires_graph": True,
                })
                continue

            _kwargs: dict = {}
            if (_cross_refs is not None and _spec_n is not None
                    and "cross_file_referenced_names" in (_spec_n.input_schema or {})):
                _kwargs["cross_file_referenced_names"] = _cross_refs
            # Scanners that need the live graph object (e.g. vulture's hub/leaf
            # scope decision) receive it via repo_graph. Unlike cross-file refs,
            # the graph is not serializable and is gated by requires_graph, not
            # input_schema.
            if _spec_n is not None and getattr(_spec_n, "requires_graph", False):
                _kwargs["repo_graph"] = getattr(self, "_call_graph", None)
            try:
                result = registry.run(name, repo_root=self.repo_root, file_paths=file_paths, **_kwargs)
            except Exception as e:
                logger.warning(f"Scanner {name} failed: {e}")
                all_lines.append(f"  - {name}: ERROR — {e}")
                continue

            candidates = result.candidates_raw[:max_results]
            total_candidates += len(candidates)
            total_affected.update(result.affected_files)
            spec = result.scanner_description

            all_lines.append(f"\n## {name}")
            all_lines.append(f"Description: {spec or '(no description)'}")
            all_lines.append(f"Files affected: {len(result.affected_files)}")
            all_lines.append(f"Candidates: {len(candidates)} (total: {result.total_candidates})")
            for c in candidates:
                c_file = c.get("file", "?")
                # Extract line number: handle lineno, line, start_line, cluster_start, occurrences
                if "occurrences" in c:
                    occ = c["occurrences"]
                    c_line = str(occ[0][0]) if occ else "?"
                else:
                    c_line = str(c.get("lineno") or c.get("line") or c.get("start_line") or c.get("cluster_start") or "?")
                # Extract symbol name: handle name, symbol, symbol_name, symbol_a/symbol_b, members
                if c.get("members"):
                    c_name = c["members"][0].get("name", "?")
                elif c.get("symbol_a") or c.get("symbol_b"):
                    c_name = f"{c.get('symbol_a','?')} ↔ {c.get('symbol_b','?')}"
                else:
                    c_name = c.get("name") or c.get("symbol") or c.get("symbol_name") or ""
                # Extract description: handle description, reason, detail, message, suggested_action, import_line_text
                c_desc = (
                    c.get("description")
                    or c.get("reason")
                    or c.get("detail")
                    or c.get("message")
                    or c.get("suggested_action")
                    or c.get("import_line_text")
                    or ""
                )
                all_lines.append(f"  - {c_file}:{c_line} {c_name} — {c_desc}")

            # When a scanner tags test-file candidates (unused_import's
            # is_test_file flag), split the count so the caller sees signal
            # density separately — test fixtures/conftests carry a high
            # false-positive rate (magic/pytest imports) that bulk-buries real
            # production findings.
            _test_candidates = [c for c in candidates if c.get("is_test_file")]
            if _test_candidates:
                _prod_count = len(candidates) - len(_test_candidates)
                all_lines.append(
                    f"  ({_prod_count} in production, {len(_test_candidates)} "
                    f"in test files — test candidates often false positives)"
                )

            per_scanner.append({
                "name": name,
                "total_candidates": result.total_candidates,
                "affected_files": len(result.affected_files),
                "reported": len(candidates),
                "test_file_candidates": len(_test_candidates),
            })

        if not all_lines:
            all_lines.append("No scanners returned results.")

        header = f"Structural scan: {len(scanners_to_run)} scanner(s)"
        if scan_path:
            header += f" on {scan_path}"
        header += f"\nTotal: {total_candidates} candidates across {len(total_affected)} files"

        metadata = {
            "scanners_run": scanners_to_run,
            "total_candidates": total_candidates,
            "affected_files": sorted(total_affected),
            "per_scanner": per_scanner,
        }

        return self._make_result(
            ok=True,
            content=header + "\n" + "\n".join(all_lines),
            metadata=metadata,
        )

    def _tool_find_import_source(self, args: dict[str, Any]) -> "ToolResult":
        """Find where a symbol is defined and determine the import statement."""
        name = str(args.get("name", "")).strip()
        from_file = str(args.get("from_file", "")).strip() or None

        if not name:
            return self._make_result(ok=False, content="", error="'name' is required")

        lines: list[str] = [f"## Import source for `{name}`"]
        metadata: dict[str, Any] = {"symbol": name}

        # 1. Check Python builtins
        import builtins
        is_builtin = hasattr(builtins, name)
        if is_builtin:
            lines.append(f"\n{name} is a Python builtin — no import needed.")
            metadata["builtin"] = True
            return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

        # 2. SymbolSearcher: find definitions
        try:
            defs = self._symbol_searcher.find_symbol(name)
            if defs:
                lines.append(f"\n### Found {len(defs)} definition(s)")
                for d in defs[:10]:
                    kind = getattr(d, "kind", "symbol")
                    sig = getattr(d, "signature", "")
                    sig_str = f" — `{sig}`" if sig else ""
                    lines.append(f"  - `{d.file}:{d.line}` ({kind}){sig_str}")

                metadata["definitions"] = [{"file": d.file, "line": d.line, "kind": getattr(d, "kind", None)} for d in defs[:10]]
            else:
                lines.append("\n### No definitions found in repository")
        except Exception as e:
            lines.append(f"\n### Symbol search error: {e}")

        # 3. Import-vs-create decision using SymbolIndex
        try:
            from external_llm.agent.symbol_index import build_repo_symbol_index, decide_import_vs_create

            sym_index = build_repo_symbol_index(self.repo_root)
            if from_file:
                result = decide_import_vs_create(name, from_file, sym_index)
            else:
                candidates = sym_index.get(name, [])
                if candidates:
                    result = {"action": "import", "symbol": name, "source_file": candidates[0].file_path, "reason": "found_via_index"}
                else:
                    result = {"action": "create", "symbol": name, "reason": "not_found_in_index"}

            action = result.get("action", "unknown")
            reason = result.get("reason", "")
            metadata["decision"] = action
            metadata["reason"] = reason
            lines.append(f"\n### Decision: {action}")

            if action == "import":
                source_file = result.get("source_file", "")
                kind = result.get("kind", "") or ""
                lines.append(f"  - Source: `{source_file}` ({kind})")
                metadata["source_file"] = source_file

                # Generate import statement for from_file
                if from_file and source_file:
                    import_statement = self._make_import_suggestion(name, source_file, from_file)
                    lines.append(f"\n### Suggested import\n```python\n{import_statement}\n```")
                    metadata["suggested_import"] = import_statement

                # Check if already imported
                if from_file:
                    abs_path = os.path.join(self.repo_root, from_file)
                    if os.path.isfile(abs_path):
                        already = self._check_import_exists(abs_path, name)
                        if already:
                            lines.append(f"\n  ℹ️ `{name}` is already imported in `{from_file}`")
                            metadata["already_imported"] = True
            elif action == "create":
                lines.append(f"  - {reason}")
            elif action == "noop":
                reason_text = result.get("reason", "")
                if reason_text == "python_builtin":
                    lines.append("  - Python builtin — no import needed")
                elif reason_text == "already_defined_here":
                    lines.append(f"  - Already defined in current file `{from_file}`")
                else:
                    lines.append(f"  - {reason_text}")
        except Exception as e:
            lines.append(f"\n### Import index error: {e}")
            logger.warning(f"find_import_source index error: {e}")

        return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

    def _check_import_exists(self, file_path: str, symbol_name: str) -> bool:
        """Check if a symbol is already imported in the given file."""
        try:
            import ast
            with open(file_path, encoding="utf-8", errors="replace") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname == symbol_name or alias.name.split(".")[-1] == symbol_name:
                            return True
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.asname == symbol_name or alias.name == symbol_name:
                            return True
        except Exception:
            pass
        return False

    def _make_import_suggestion(self, symbol: str, source_file: str, from_file: str) -> str:
        """Generate the correct import statement from source_file to from_file."""
        # Convert file path to module path
        module_path = source_file.replace(".py", "").replace("/", ".")
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]  # strip .__init__
        elif source_file.endswith("/__init__.py"):
            module_path = module_path[:-9]

        # If importing __init__ itself is unusual; check if symbol is a module-level definition
        # Try the simple case: from module import Symbol
        return f"from {module_path} import {symbol}"

    # _tool_analyze_insertion_point and _pick_best_insertion_line removed:
    # Python-only analysis tools; LLM deciding placement directly is more efficient

    def _tool_query_dependency_graph(self, args: dict[str, Any]) -> "ToolResult":
        """Query the repository dependency/call graph with BFS traversal.

        Modes:
          - importers: BFS from a FILE to find all transitive importers (who imports this file)
          - path: BFS shortest-path between two SYMBOLS (how are they connected?)
          - reachable: BFS from a SYMBOL to find all downstream reachable symbols
          - subgraph: list all symbols + edges in a FILE
        """
        mode = str(args.get("mode", "subgraph")).strip().lower()
        source = str(args.get("source", "")).strip()
        target = str(args.get("target", "")).strip()
        max_depth = int(args.get("max_depth", 5))
        limit = int(args.get("limit", 50))

        if mode == "subgraph":
            file_path = source  # source is file_path in subgraph mode
            if not file_path:
                return self._make_result(
                    ok=False, content="",
                    error="'source' (file path) is required for subgraph mode"
                )
            return self._query_subgraph(file_path, limit)
        elif mode == "importers":
            if not source:
                return self._make_result(
                    ok=False, content="",
                    error="'source' (file path) is required for importers mode"
                )
            return self._query_transitive_importers(source, max_depth, limit)
        elif mode == "reachable":
            if not source:
                return self._make_result(
                    ok=False, content="",
                    error="'source' (symbol name) is required for reachable mode"
                )
            direction = str(args.get("direction", "downstream")).strip().lower()
            return self._query_reachable(source, direction, max_depth, limit)
        elif mode == "path":
            if not source or not target:
                return self._make_result(
                    ok=False, content="",
                    error="Both 'source' and 'target' (symbol names) are required for path mode"
                )
            direction = str(args.get("direction", "downstream")).strip().lower()
            return self._query_symbol_path(source, target, direction, max_depth, limit)
        else:
            return self._make_result(
                ok=False, content="",
                error=f"Unknown mode: {mode}. Supported: importers, path, reachable, subgraph"
            )

    def _query_subgraph(self, file_path: str, limit: int) -> "ToolResult":
        """List all symbols in a file with their edges."""
        lines: list[str] = [f"## Subgraph for `{file_path}`"]
        metadata: dict[str, Any] = {"mode": "subgraph", "file_path": file_path}

        try:
            # Normalize path
            from pathlib import Path as _Path
            fp = _Path(file_path)
            if fp.is_absolute():
                file_path = str(fp.relative_to(self.repo_root))
        except Exception:
            pass

        # Symbols in file
        try:
            symbols = self._call_graph.get_symbols_in_file(file_path)
        except Exception:
            symbols = []

        if not symbols:
            lines.append("\nNo symbols found in this file via graph.")
            return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

        lines.append(f"\n**Symbols** ({len(symbols)}):")
        for sym in symbols[:limit]:
            kind = sym.kind if hasattr(sym, 'kind') else "?"
            sig = ""
            if hasattr(sym, 'signature') and sym.signature:
                sig = f" — `{sym.signature}`"
            lines.append(f"  - {kind} `{sym.name}` ({sym.start_line}-{sym.end_line}){sig}")
        metadata["symbols"] = [{"name": s.name, "kind": s.kind if hasattr(s, 'kind') else "",
                                "start_line": s.start_line, "end_line": s.end_line} for s in symbols[:limit]]
        metadata["symbol_count"] = len(symbols)

        # Edges between symbols in this file
        sym_names = {s.name for s in symbols}
        edges_found: list[str] = []
        for sym in symbols[:limit]:
            try:
                callees = self._call_graph.get_callees(sym.name, file_path=file_path)
                for c in callees[:10]:
                    c_name = c.callee_symbol
                    if c_name in sym_names:
                        edges_found.append(f"  `{sym.name}` → `{c_name}` (line {c.callee_line})")
            except Exception:
                pass

        if edges_found:
            lines.append(f"\n**Internal edges** ({len(edges_found)}):")
            lines.extend(edges_found[:limit])
            metadata["internal_edges"] = edges_found[:limit]

        # Import edges
        try:
            deps = self._call_graph.get_file_dependencies(file_path)
            if deps:
                lines.append(f"\n**Imports** ({len(deps)}):")
                for d in deps[:limit]:
                    lines.append(f"  `{d.imported}` ({d.import_type})")
                metadata["imports"] = [{"imported": d.imported, "type": d.import_type} for d in deps[:limit]]
        except Exception:
            pass

        return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

    def _query_transitive_importers(self, file_path: str, max_depth: int, limit: int) -> "ToolResult":
        """BFS from a file to find all transitive importers."""
        lines: list[str] = [f"## Transitive importers for `{file_path}`"]
        metadata: dict[str, Any] = {"mode": "importers", "source": file_path, "max_depth": max_depth}

        try:
            from pathlib import Path as _Path
            fp = _Path(file_path)
            if fp.is_absolute():
                file_path = str(fp.relative_to(self.repo_root))
        except Exception:
            pass

        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(file_path, 0)]
        import_chain: list[dict] = []

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            if depth > 0:
                import_chain.append({"file": current, "depth": depth})

            try:
                importers = self._call_graph.get_importers(current)
            except Exception:
                importers = []

            for imp in importers:
                if imp not in visited:
                    queue.append((imp, depth + 1))

        if import_chain:
            lines.append(f"\nFound {len(import_chain)} transitive importers (depth ≤{max_depth}):")
            for entry in import_chain[:limit]:
                indent = "  " * entry["depth"]
                lines.append(f"{indent}└─ {entry['file']}")
            metadata["importers"] = [e["file"] for e in import_chain[:limit]]
            metadata["importer_count"] = len(import_chain)
        else:
            lines.append("\nNo importers found.")

        return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

    def _query_reachable(self, source_symbol: str, direction: str, max_depth: int, limit: int) -> "ToolResult":
        """BFS from a symbol to find all symbols reachable in the given direction."""
        dir_label = "upstream (callers)" if direction == "upstream" else "downstream (callees)"
        lines: list[str] = [f"## Reachable symbols from `{source_symbol}` ({dir_label})"]
        metadata: dict[str, Any] = {
            "mode": "reachable", "source": source_symbol,
            "direction": direction, "max_depth": max_depth,
        }

        visited: set[str] = {source_symbol}
        queue: list[tuple[str, str, int]] = [(source_symbol, source_symbol, 0)]
        reachable: list[dict[str, Any]] = []

        while queue:
            current, origin, depth = queue.pop(0)
            # Bound is enforced at POP, but the guard is ``>=`` (not ``>``)
            # because this BFS records neighbors at DISCOVERY time — a node at
            # ``depth`` appends its children at ``depth + 1`` below. If we
            # expanded a node already AT ``max_depth`` we would record children
            # at ``max_depth + 1``, leaking one level past the requested bound
            # (the result header promises "depth ≤max_depth"). Stopping
            # expansion at ``depth >= max_depth`` keeps recorded depths in
            # 1..max_depth — matching the sibling _query_transitive_importers,
            # which records at POP time (after its own ``> max_depth`` guard).
            if depth >= max_depth:
                continue

            try:
                if direction == "upstream":
                    edges = self._call_graph.get_callers(current)
                else:
                    edges = self._call_graph.get_callees(current)
            except Exception:
                edges = []

            for edge in edges[:limit]:
                if direction == "upstream":
                    neighbor = edge.caller_symbol
                else:
                    neighbor = edge.callee_symbol

                if neighbor in visited:
                    continue
                visited.add(neighbor)
                reachable.append({
                    "symbol": neighbor,
                    "depth": depth + 1,
                    "via": current,
                    "file": edge.caller_file if direction == "upstream" else edge.callee_file or edge.caller_file or "",
                })
                queue.append((neighbor, origin, depth + 1))

        if reachable:
            lines.append(f"\nFound {len(reachable)} reachable symbols (depth ≤{max_depth}):")
            for r in reachable[:limit]:
                indent = "  " * r["depth"]
                lines.append(f"{indent}└─ `{r['symbol']}` ({r['file']})")
            metadata["reachable"] = reachable[:limit]
            metadata["reachable_count"] = len(reachable)
        else:
            lines.append(f"\nNo {dir_label} found.")

        return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

    def _query_symbol_path(self, source_sym: str, target_sym: str, direction: str, max_depth: int, limit: int) -> "ToolResult":
        """BFS shortest-path between two symbols."""
        lines: list[str] = [f"## Path from `{source_sym}` → `{target_sym}`"]
        metadata: dict[str, Any] = {
            "mode": "path", "source": source_sym, "target": target_sym,
            "direction": direction, "max_depth": max_depth,
        }
        from collections import deque

        if direction in ("downstream", "both"):
            # BFS via callees: source → target
            queue: deque = deque()
            queue.append((source_sym, [source_sym]))
            visited: set[str] = {source_sym}

            while queue:
                current, path = queue.popleft()
                if len(path) > max_depth + 1:
                    continue

                try:
                    edges = self._call_graph.get_callees(current)
                except Exception:
                    edges = []

                for edge in edges[:limit]:
                    neighbor = edge.callee_symbol
                    if neighbor == target_sym:
                        full_path = [*path, neighbor]
                        metadata["path_found"] = True
                        metadata["path"] = full_path
                        metadata["path_length"] = len(full_path) - 1
                        metadata["direction"] = "downstream"
                        lines.append(f"\nPath found (depth={len(full_path)-1} via callees):")
                        for i, sym in enumerate(full_path):
                            prefix = "→" if i > 0 else " "
                            lines.append(f"  {prefix} `{sym}`")
                        return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

                    if neighbor not in visited and len(path) < max_depth:
                        visited.add(neighbor)
                        queue.append((neighbor, [*path, neighbor]))

        if direction in ("upstream", "both"):
            # BFS via callers: source ← target (walking callers from source)
            queue = deque()
            queue.append((source_sym, [source_sym]))
            visited = {source_sym}

            while queue:
                current, path = queue.popleft()
                if len(path) > max_depth + 1:
                    continue

                try:
                    edges = self._call_graph.get_callers(current)
                except Exception:
                    edges = []

                for edge in edges[:limit]:
                    neighbor = edge.caller_symbol
                    if neighbor == target_sym:
                        full_path = [*path, neighbor]
                        metadata["path_found"] = True
                        metadata["path"] = full_path
                        metadata["path_length"] = len(full_path) - 1
                        metadata["direction"] = "upstream"
                        lines.append(f"\nPath found (depth={len(full_path)-1} via callers):")
                        for i, sym in enumerate(full_path):
                            prefix = "←" if i > 0 else " "
                            lines.append(f"  {prefix} `{sym}`")
                        return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

                    if neighbor not in visited and len(path) < max_depth:
                        visited.add(neighbor)
                        queue.append((neighbor, [*path, neighbor]))

        lines.append(f"\nNo path found (depth ≤{max_depth})")
        metadata["path_found"] = False
        return self._make_result(ok=True, content="\n".join(lines), metadata=metadata)

    def _tool_explore_codebase(self, args: dict[str, Any]) -> "ToolResult":
        """BFS over call/import graph to find relevant files starting from a symbol or file."""
        symbol = str(args.get("symbol", "")).strip() or None
        file_path = str(args.get("file_path", "")).strip() or None
        goal = str(args.get("goal", "")).strip() or "explore related code"
        int(args.get("max_depth", 3))
        max_candidates = int(args.get("max_candidates", 15))

        if not symbol and not file_path:
            return self._make_result(
                ok=False,
                content="Provide at least one of: symbol (starting symbol name) or file_path (starting file).",
            )

        from external_llm.exploration import run_exploration as _run_exploration

        # Build initial candidates from the starting point
        initial_candidates: list[dict[str, Any]] = []

        # Check for call graph / graph facade
        graph_facade = getattr(self, "_call_graph", None)

        if symbol:
            sym_file = None
            if graph_facade is not None:
                try:
                    sym_file = graph_facade.get_symbol_file(symbol)
                except Exception:
                    sym_file = None
            if sym_file:
                initial_candidates.append({"file": sym_file, "symbol": symbol, "score": 0.9})
            else:
                # Symbol not found via graph — try ripgrep-based search
                logger.info("explore_codebase: symbol '%s' not in graph, searching via rg", symbol)
                initial_candidates.append({"file": "", "symbol": symbol, "score": 0.5})

        if file_path:
            # Normalise path (strip repo_root prefix if present)
            rel_path = file_path
            if file_path.startswith(self.repo_root):
                rel_path = os.path.relpath(file_path, self.repo_root)
            # Check if this file already added via symbol
            exists = any(c.get("file") == rel_path for c in initial_candidates)
            if not exists:
                initial_candidates.append({"file": rel_path, "symbol": "", "score": 0.7})

        if not initial_candidates:
            return self._make_result(ok=False, content="Could not determine starting point for exploration.")

        if graph_facade is None:
            return self._make_result(
                ok=True,
                content="Exploration requires a repository graph (graph_facade not available). "
                        "Try 'bash (grep)' or 'get_call_graph' instead.",
            )

        try:
            exp_state = _run_exploration(
                initial_candidates=initial_candidates,
                goal=goal,
                graph_facade=graph_facade,
                repo_root=self.repo_root,
            )
        except Exception as e:
            logger.exception("explore_codebase failed")
            return self._make_result(ok=False, content=f"Exploration failed: {e}")

        # Format results
        lines: list[str] = []
        lines.append(f"## Exploration: {goal}")
        lines.append(f"  Status: {exp_state.status.value}")
        lines.append(f"  Depth: {exp_state.depth}")
        lines.append(f"  Confidence: {exp_state.confidence:.3f}")
        lines.append(f"  Candidates: {len(exp_state.candidates)}")

        top_files = exp_state.top_files[:max_candidates]
        if top_files:
            lines.append(f"\n## Top Files ({len(top_files)}):")
            # Build a quick lookup for per-file scores
            file_scores: dict[str, float] = {}
            file_symbols: dict[str, str] = {}
            for c in exp_state.candidates:
                f = c.file
                if f not in file_scores or c.score > file_scores[f]:
                    file_scores[f] = c.score
                if c.symbol and c.symbol not in file_symbols.get(f, ""):
                    file_symbols[f] = c.symbol

            for i, f in enumerate(top_files, 1):
                score = file_scores.get(f, 0.0)
                sym = file_symbols.get(f, "")
                sym_info = f"  sym={sym}" if sym else ""
                lines.append(f"  {i}. {f}  score={score:.3f}  {sym_info}")

        top_syms = exp_state.top_symbols
        if top_syms:
            lines.append(f"\n## Top Symbols ({len(top_syms)}):")
            for s in top_syms:
                lines.append(f"  · {s}")

        if exp_state.facts:
            lines.append("\n## Facts:")
            for fact in exp_state.facts:
                lines.append(f"  · {fact}")

        if exp_state.turn_history:
            lines.append("\n## Traversal History:")
            for turn in exp_state.turn_history:
                lines.append(f"  · {turn}")

        return self._make_result(
            ok=True,
            content="\n".join(lines),
            metadata={
                "status": exp_state.status.value,
                "depth": exp_state.depth,
                "confidence": exp_state.confidence,
                "candidate_count": len(exp_state.candidates),
                "top_files": top_files[:5],
                "top_symbols": top_syms[:5],
            },
        )

    def _tool_analyze_semantic_gap(self, args: dict[str, Any]) -> "ToolResult":
        """Extract behavioral trace and identify semantic gaps in Python code."""
        file_path = str(args.get("file_path", "")).strip()
        function = str(args.get("function", "")).strip() or None
        include_all = bool(args.get("include_all", False))

        if not file_path:
            return self._make_result(ok=False, content="'file_path' is required.")

        abs_path = file_path if os.path.isabs(file_path) else os.path.join(self.repo_root, file_path)
        if not os.path.isfile(abs_path):
            return self._make_result(ok=False, content=f"File not found: {file_path}")

        if LanguageId.from_path(abs_path) is not LanguageId.PYTHON:
            return self._make_result(ok=True, content="analyze_semantic_gap supports Python files only.")

        from external_llm.editor.semantic.semantic_tracer import extract_trace_from_files

        try:
            trace = extract_trace_from_files([abs_path], repo_root=self.repo_root)
        except Exception as e:
            return self._make_result(ok=True, content=f"Trace extraction failed: {e}")

        lines: list[str] = []
        gap_count = 0

        # Filter by function if requested
        func_traces = trace.function_traces
        if function:
            if function in func_traces:
                func_traces = {function: func_traces[function]}
            else:
                return self._make_result(ok=True, content=f"Function '{function}' not found in {file_path}")

        lines.append(f"## Semantic Trace: {file_path}")
        lines.append(f"  Functions: {len(func_traces)}")
        lines.append(f"  Total calls: {len(trace.all_calls)}")
        lines.append(f"  Total instantiations: {len(trace.all_instantiations)}")
        lines.append(f"  Total persists: {len(trace.all_persist_calls)}")
        lines.append("")

        # In summary mode, just show gap summary, not per-function details
        if not include_all and not function:
            gap_summary: list[str] = []
            for func_name, ft in func_traces.items():
                if ft.instantiations and not ft.persist_calls:
                    for entity in sorted(ft.instantiations):
                        gap_summary.append(f"  {func_name}: missing_persistence — {entity}")
                        gap_count += 1
                if ft.instantiations and ft.return_names and not ft.return_has_entity_ref:
                    # Filter: only flag if function creates entities AND returns
                    has_persist_entity = any(
                        p for p in ft.persist_calls
                    )
                    if not has_persist_entity:
                        gap_summary.append(f"  {func_name}: missing_output_reference — return does not reference created entity")
                        gap_count += 1
            if gap_summary:
                lines.append(f"## Gaps Found ({gap_count}):")
                lines.extend(gap_summary)
            else:
                lines.append("No semantic gaps detected.")
            lines.append("(Use function= or include_all=true for detailed per-function traces)")

            return self._make_result(
                ok=True, content="\n".join(lines),
                metadata={
                    "functions_traced": len(func_traces),
                    "gap_count": gap_count,
                    "calls": len(trace.all_calls),
                    "instantiations": len(trace.all_instantiations),
                    "persists": len(trace.all_persist_calls),
                },
            )

        for func_name, ft in func_traces.items():
            lines.append(f"### {func_name}")
            if ft.param_names:
                lines.append(f"  Params: {', '.join(ft.param_names)}")
            if ft.calls:
                lines.append(f"  Calls: {', '.join(sorted(ft.calls))}")
            if ft.instantiations:
                lines.append(f"  Instantiates: {', '.join(sorted(ft.instantiations))}")
            if ft.persist_calls:
                lines.append(f"  Persists via: {', '.join(sorted(ft.persist_calls))}")
            if ft.return_names:
                lines.append(f"  Returns: {', '.join(ft.return_names)}")
            if ft.entity_bindings:
                bindings = [f"{f}←{s}" for f, s in ft.entity_bindings]
                lines.append(f"  Entity bindings: {', '.join(bindings)}")
            if ft.has_error_branch:
                lines.append("  Has error branch: yes" + (" (before success)" if ft.error_before_success else ""))
            if ft.call_order:
                lines.append(f"  Call order: {' → '.join(ft.call_order)}")

            # Gap analysis
            gaps: list[str] = []
            if ft.instantiations and not ft.persist_calls:
                for entity in sorted(ft.instantiations):
                    gaps.append(f"[gap] missing_persistence: {entity} is instantiated but never persisted")
                    gap_count += 1
            if ft.instantiations and ft.return_names:
                # Check if any return references an instantiated entity
                has_entity_in_return = ft.return_has_entity_ref
                if not has_entity_in_return:
                    gaps.append("[gap] missing_output_reference: return does not reference created entity")
                    gap_count += 1
            if not ft.param_names and not ft.calls and not ft.instantiations:
                gaps.append("[info] trivial function: no params, calls, or instantiations")
            if gaps:
                lines.append("  Gaps:")
                for g in gaps:
                    lines.append(f"    {g}")
            lines.append("")

        # Summary
        if not include_all and not function:
            lines.append("---")
            lines.append(f"Total gaps found: {gap_count}")
            lines.append("(Use function= or include_all=true for detailed per-function traces)")

        return self._make_result(
            ok=True,
            content="\n".join(lines),
            metadata={
                "functions_traced": len(func_traces),
                "total_functions": len(trace.function_traces),
                "gap_count": gap_count,
                "calls": len(trace.all_calls),
                "instantiations": len(trace.all_instantiations),
                "persists": len(trace.all_persist_calls),
            },
        )

    def _tool_query_experience(self, args: dict[str, Any]) -> "ToolResult":
        """Query historical execution records from the learning store."""
        query_type = str(args.get("query_type", "recent")).strip().lower()
        language = str(args.get("language", "")).strip() or None
        strategy = str(args.get("strategy", "")).strip() or None
        limit = int(args.get("limit", 20))

        from external_llm.editor.learning.unified_store import get_unified_store

        try:
            store = get_unified_store(project_root=self.repo_root)
        except Exception as e:
            return self._make_result(
                ok=True,
                content=f"Learning store not available: {e}. Data accumulates as the system runs.",
            )

        lines: list[str] = []
        total_count = store.count()

        if query_type == "recent":
            records = store.get_recent(language=language, limit=limit)
            lines.append(f"## Recent Runs (total store: {total_count})")
            if language:
                lines.append(f"  Language: {language}")
            if not records:
                lines.append("  (no records found)")
            else:
                for r in records:
                    status = "✓" if r.success else "✗"
                    lang = r.language or "?"
                    strat = r.strategy or "?"
                    request_short = (r.request or "")[:80]
                    lines.append(
                        f"  {status} [{lang}] {strat} — {request_short}"
                    )

        elif query_type == "strategy_stats":
            stats = store.get_strategy_stats(limit=limit)
            lines.append(f"## Strategy Stats (total store: {total_count})")
            if not stats:
                lines.append("  (no strategy data yet)")
            else:
                lines.append(f"  {'Strategy':<30} {'OK':>5} {'Total':>6} {'Rate':>8}")
                lines.append(f"  {'─'*30} {'─'*5} {'─'*6} {'─'*8}")
                for s_name, s_data in sorted(
                    stats.items(), key=lambda x: x[1]["ok"] / max(x[1]["total"], 1), reverse=True
                )[:limit]:
                    rate = s_data["ok"] / max(s_data["total"], 1)
                    lines.append(
                        f"  {s_name:<30} {s_data['ok']:>5} {s_data['total']:>6} {rate:>7.0%}"
                    )

        elif query_type == "strategy_runs":
            if not strategy:
                return self._make_result(
                    ok=False, content="'strategy' parameter is required for strategy_runs query."
                )
            records = store.get_strategy_runs(strategy=strategy, language=language, limit=limit)
            lines.append(f"## Strategy Runs: {strategy}")
            if language:
                lines.append(f"  Language: {language}")
            if not records:
                lines.append(f"  (no runs for strategy '{strategy}')")
            else:
                for r in records:
                    status = "✓" if r.success else "✗"
                    lang = r.language or "?"
                    request_short = (r.request or "")[:80]
                    lines.append(f"  {status} [{lang}] {request_short}")

        elif query_type == "model_stats":
            stats = store.get_model_stats()
            lines.append(f"## Model Stats (total store: {total_count})")
            if not stats:
                lines.append("  (no model data yet)")
            else:
                for model, data in list(stats.items())[:limit]:
                    rate = data["ok"] / max(data["total"], 1)
                    lines.append(
                        f"  {model:<30} ok={data['ok']:<3} total={data['total']:<4} rate={rate:.0%}"
                    )

        elif query_type == "failure_patterns":
            # Aggregate by final_failure_class
            failure_counts: dict[str, int] = {}
            failure_langs: dict[str, set] = {}
            for r in store.iter_all():
                fc = r.final_failure_class or r.final_status or "unknown"
                if fc:
                    failure_counts[fc] = failure_counts.get(fc, 0) + 1
                    if fc not in failure_langs:
                        failure_langs[fc] = set()
                    if r.language:
                        failure_langs[fc].add(r.language)
            lines.append(f"## Failure Patterns (total failures: {sum(failure_counts.values())})")
            if not failure_counts:
                lines.append("  (no failure data yet)")
            else:
                sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
                for fc, count in sorted_failures[:limit]:
                    langs = ", ".join(sorted(failure_langs.get(fc, [])))
                    lang_info = f"  langs={langs}" if langs else ""
                    lines.append(f"  · {fc}: {count} occurrences {lang_info}")

        else:
            return self._make_result(
                ok=False,
                content=f"Unknown query_type: '{query_type}'. "
                        "Supported: recent, strategy_stats, strategy_runs, model_stats, failure_patterns",
            )

        return self._make_result(
            ok=True,
            content="\n".join(lines),
            metadata={"query_type": query_type, "total_records": total_count},
        )

    def _tool_suggest_edit_location(self, args: dict[str, Any]) -> "ToolResult":
        """Analyze code structure against an edit request to suggest where to make changes."""
        request = str(args.get("request", "")).strip()
        file_path = str(args.get("file_path", "")).strip() or None
        directory = str(args.get("directory", "")).strip() or None
        max_results = int(args.get("max_results", 10))

        if not request:
            return self._make_result(ok=False, content="'request' is required — describe the edit you want to make.")

        from external_llm.code_structure_utils import parse_definitions
        from external_llm.edit_localization import extract_flow_facts, score_edit_relevance

        # Collect files to analyze
        files_to_scan: list[str] = []

        if file_path:
            abs_path = file_path if os.path.isabs(file_path) else os.path.join(self.repo_root, file_path)
            if os.path.isfile(abs_path):
                files_to_scan.append(abs_path)
            else:
                return self._make_result(ok=False, content=f"File not found: {file_path}")

        elif directory:
            abs_dir = directory if os.path.isabs(directory) else os.path.join(self.repo_root, directory)
            if os.path.isdir(abs_dir):
                for root, _dirs, files in os.walk(abs_dir):
                    for fn in files:
                        if LanguageId.from_path(fn) is LanguageId.PYTHON:
                            files_to_scan.append(os.path.join(root, fn))
            else:
                return self._make_result(ok=False, content=f"Directory not found: {directory}")
        else:
            # Default: scan repo root (top-level .py files only)
            for fn in os.listdir(self.repo_root):
                if LanguageId.from_path(fn) is LanguageId.PYTHON:
                    files_to_scan.append(os.path.join(self.repo_root, fn))

        if not files_to_scan:
            return self._make_result(ok=True, content="No Python files found in the specified scope.")

        # Limit files to avoid excessive runtime
        if len(files_to_scan) > 50:
            logger.info("suggest_edit_location: scanning %d files (first 50)", len(files_to_scan))
            files_to_scan = files_to_scan[:50]

        scored_results: list[tuple[float, str, str, str]] = []  # (score, reason, file, symbol)
        scan_count = 0

        for abs_path in files_to_scan:
            try:
                with open(abs_path, encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except Exception:
                continue

            defs = parse_definitions(source)
            for d in defs:
                if d.kind not in ("function", "class") or not d.name:
                    continue
                try:
                    body_lines = source.splitlines()[d.start_line - 1:d.end_line]
                    body_source = "\n".join(body_lines)
                    facts = extract_flow_facts(body_source)
                    score, reason = score_edit_relevance(request, facts)
                    if score >= 0.01:
                        rel_path = os.path.relpath(abs_path, self.repo_root)
                        scored_results.append((score, reason, rel_path, d.name))
                except Exception:
                    continue
            scan_count += 1

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top = scored_results[:max_results]

        lines: list[str] = []
        lines.append("## Edit Location Analysis")
        lines.append(f"  Request: {request}")
        lines.append(f"  Files scanned: {scan_count}")
        lines.append(f"  Symbols analyzed: {len(scored_results)}")
        lines.append("")

        if not top:
            lines.append("No highly relevant edit locations found. Try expanding the search scope.")
        else:
            lines.append(f"## Top Candidates ({len(top)}):")
            for i, (score, reason, rel_path, sym_name) in enumerate(top, 1):
                lines.append(f"  {i}. {rel_path} → {sym_name}  score={score:.3f}")
                lines.append(f"     reason: {reason}")

        return self._make_result(
            ok=True,
            content="\n".join(lines),
            metadata={
                "files_scanned": scan_count,
                "symbols_analyzed": len(scored_results),
                "max_results": max_results,
                "top_candidate": top[0][2] + " → " + top[0][3] if top else None,
                "top_score": round(top[0][0], 3) if top else 0.0,
            },
        )
