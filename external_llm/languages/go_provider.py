"""
Go syntax provider.

Uses ``go build`` for validation and regex-based symbol detection.
Gracefully degrades when Go toolchain is not installed.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Optional

from .base import (
    SyntaxProvider,
    _compile_env,
    _filter_genuine_syntax_errors,
    _replace_last_cmd_path,
    _tempfile_for_content,
    detect_project_root,
    find_brace_block_end,
    resolve_tool_path,
    tree_sitter_syntax_fallback,
)
from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxError_,
    SyntaxValidationResult,
)

logger = logging.getLogger(__name__)


def _make_capabilities() -> LanguageCapabilities:
    from .tree_sitter_utils import is_available

    return LanguageCapabilities(
        has_ast_parser=False,
        has_syntax_validator=True,
        has_semantic_validator=True,
        has_linter=True,
        has_test_runner=True,
        has_symbol_search=True,
        has_tree_sitter=is_available(),
        supports_modify_symbol=True,
        supports_insert_after_symbol=True,
    )


# go build error: file.go:10:5: expected ';', found 'EOF'
_GO_ERROR_RE = re.compile(
    r"^(.+?):(\d+):(\d+):\s+(.+)$"
)


class GoSyntaxProvider(SyntaxProvider):
    """Go language support (regex + tree-sitter symbols, go build validation)."""

    _caps: Optional[LanguageCapabilities] = None

    def language_id(self) -> LanguageId:
        return LanguageId.GO

    def capabilities(self) -> LanguageCapabilities:
        if self._caps is None:
            self._caps = _make_capabilities()
        return self._caps

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate Go source via ``go build`` on *content* (written to temp file).

        Falls back to ``ok=True`` when go is not available.

        Resolution safety: an isolated temp file is NOT part of the file's real
        Go module, so ``go build`` would emit module/import-resolution failures
        ("no required module provides package", "go.mod file not found") for any
        non-stdlib import — and those match the ``file:line:col:`` error shape,
        so without filtering they wrongly roll back valid edits. We therefore
        run from the module root (``go.mod``) when present so imports resolve
        via the module graph (command-line-arguments build mode), and drop any
        residual resolution errors via :func:`_filter_genuine_syntax_errors`.
        Only genuine syntax errors gate the edit; the on-disk
        :meth:`validate_semantics` pass re-checks with full package context.
        """
        _tmp_path, _cleanup = _tempfile_for_content(content, ".go")
        if not _tmp_path:
            return SyntaxValidationResult(ok=True, language=LanguageId.GO)
        _cmd = _replace_last_cmd_path(
            ["go", "build", "-o", os.devnull, file_path],
            file_path, _tmp_path,
        )
        # Run from the Go MODULE root when available so imports resolve via the
        # module's dependency graph (command-line-arguments build mode). Without
        # module context `go build` emits resolution failures for every
        # non-stdlib import; those are filtered out below regardless of cwd.
        _module_root = detect_project_root(file_path, markers=("go.mod",))
        _cwd = (
            _module_root
            if os.path.isfile(os.path.join(_module_root, "go.mod"))
            else (os.path.dirname(_tmp_path) or ".")
        )
        try:
            try:
                proc = subprocess.run(
                    _cmd,
                    capture_output=True, text=True, timeout=30,
                    cwd=_cwd,
                    env=_compile_env(),
                    check=False,
                )
            except FileNotFoundError:
                logger.debug("go not installed; falling back to tree-sitter")
                return tree_sitter_syntax_fallback(content, LanguageId.GO, file_path)
            except subprocess.TimeoutExpired:
                logger.debug("go build timed out for %s", file_path)
                return tree_sitter_syntax_fallback(content, LanguageId.GO, file_path)
            except Exception as e:
                logger.debug("go build error: %s", e)
                return tree_sitter_syntax_fallback(content, LanguageId.GO, file_path)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=LanguageId.GO)

            errors: list[SyntaxError_] = []
            raw_lines = (proc.stdout + proc.stderr).splitlines()
            i = 0
            while i < len(raw_lines):
                line = raw_lines[i]
                m = _GO_ERROR_RE.match(line)
                if m:
                    msg = m.group(4)
                    # Capture multi-line detail (e.g. "have (...)\n    want (...)")
                    # that follows "not enough arguments" / "too many arguments" errors
                    if "not enough arguments" in msg or "too many arguments" in msg:
                        j = i + 1
                        while j < len(raw_lines) and (
                            raw_lines[j].startswith("\t") or raw_lines[j].startswith("    ")
                        ):
                            msg += "\n" + raw_lines[j]
                            j += 1
                        i = j - 1  # -1 because i will be incremented after loop
                    errors.append(SyntaxError_(
                        file=m.group(1),
                        line=int(m.group(2)),
                        col=int(m.group(3)),
                        message=msg,
                    ))
                i += 1
            # Drop module/import-resolution failures (no syntax error in the
            # proposed content); the isolated temp file lacks the module graph
            # the real file lives in.
            errors = _filter_genuine_syntax_errors(errors, LanguageId.GO)
            if not errors:
                return SyntaxValidationResult(ok=True, language=LanguageId.GO)

            return SyntaxValidationResult(ok=False, errors=errors, language=LanguageId.GO)
        finally:
            _cleanup()

    # ── Semantic validation ──────────────────────────────────────────────

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run ``go build`` on the **on-disk** file's package directory.

        Go is compiled per-package, so unlike pyright/tsc this runs against the
        directory containing *file_path* (from the project root detected via
        ``go.mod``). This catches real semantic errors the config-blind single-
        file syntax check misses: undefined names, wrong-arity calls, type
        mismatches, and missing imports.

        Design choices:
        - Skips if there is no ``go.mod`` (no module → no stable import graph).
        - Runs ``go build ./<pkg-dir>`` (or ``.``) from the module root so all
          sibling files in the package are compiled together.
        - Only reports diagnostics whose file path matches *file_path* (a
          package build surfaces errors from sibling files too).
        - Errors make ``ok=False``; build warnings are surfaced.

        Checking several files goes through :meth:`validate_semantics_batch`,
        which builds each package once instead of once per file.
        """
        return self.validate_semantics_batch([file_path])[file_path]

    def validate_semantics_batch(
        self, file_paths: list[str],
    ) -> dict[str, SyntaxValidationResult]:
        """Semantic-check *file_paths* with one ``go build`` per package.

        Go compiles a whole package at a time, so a single build ALREADY
        produces the diagnostics for every file in that package — the
        single-file path then discarded all but one file's worth. Two files
        edited in the same package meant running the identical command twice
        and throwing away half of each result. Grouping by package directory
        removes that duplication outright; distinct packages still need their
        own build.

        Grouped by ``(module_root, package_dir)``: a repo can contain several
        modules, and package paths are only meaningful relative to their own
        ``go.mod``.
        """
        # A fresh result per file, never one shared instance: the dataclass is
        # mutable and carries a list, so sharing would let one consumer's
        # append surface on every other skipped file.
        def _skip(reason: str) -> SyntaxValidationResult:
            return SyntaxValidationResult.unchecked(LanguageId.GO, reason)

        out: dict[str, SyntaxValidationResult] = {}
        groups: dict[tuple[str, str], list[str]] = {}
        for p in file_paths:
            if not p or not os.path.exists(p):
                out[p] = _skip("the file is not on disk")
                continue
            module_root = detect_project_root(p, markers=("go.mod",))
            if not os.path.isfile(os.path.join(module_root, "go.mod")):
                out[p] = _skip("no go.mod above this file, so `go build` has no package")
                continue
            pkg_dir = os.path.dirname(os.path.abspath(p))
            groups.setdefault((module_root, pkg_dir), []).append(p)
        for (module_root, pkg_dir), paths in groups.items():
            out.update(self._build_package(module_root, pkg_dir, paths))
        return out

    def _build_package(
        self, module_root: str, pkg_dir_abs: str, file_paths: list[str],
    ) -> dict[str, SyntaxValidationResult]:
        """One ``go build`` of the package, split back out per file in *file_paths*."""
        def _skip(reason: str) -> dict[str, SyntaxValidationResult]:
            return {
                p: SyntaxValidationResult.unchecked(LanguageId.GO, reason)
                for p in file_paths
            }
        # Package dir relative to module root
        try:
            pkg_rel = os.path.relpath(pkg_dir_abs, module_root)
        except ValueError:
            pkg_rel = "."
        pkg_target = "./" + pkg_rel if pkg_rel != "." else "."

        cmd = ["go", "build", pkg_target]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                # One build covers the whole package regardless of how many of
                # its files were edited, so the budget only needs a little head
                # room over the single-file case.
                timeout=30 + 5 * len(file_paths),
                cwd=module_root,
                env=_compile_env(),
                check=False,
            )
        except FileNotFoundError:
            logger.debug("go not installed; skipping semantic validation")
            return _skip("the go toolchain is not installed")
        except subprocess.TimeoutExpired:
            logger.debug("go build timed out for %s; skipping", pkg_target)
            return _skip("`go build` timed out")
        except Exception as e:
            logger.debug("go build semantic check failed: %s", e)
            return _skip("`go build` could not be run")

        if proc.returncode == 0:
            # The build SUCCEEDED — a real clean verdict, not a skip. Sharing
            # the skip constructor here would report a genuinely checked file
            # as unchecked, which is the same conflation in the other
            # direction.
            return {
                p: SyntaxValidationResult(ok=True, language=LanguageId.GO)
                for p in file_paths
            }

        # Parse: ./pkg/file.go:10:5: undefined: foo
        by_norm = {os.path.normpath(os.path.abspath(p)): p for p in file_paths}
        collected: dict[str, list[SyntaxError_]] = {p: [] for p in file_paths}
        failed: set[str] = set()
        for line in (proc.stdout + proc.stderr).splitlines():
            m = _GO_ERROR_RE.match(line)
            if not m:
                continue
            _file, _line, _col, _msg = m.groups()
            # Only report the files we asked about. A package build surfaces
            # siblings we were not asked to check, and go build prints paths
            # relative to its own cwd (module_root), so resolution goes through
            # the shared resolve_tool_path rather than a bare abspath — see its
            # docstring for the silent-drop failure that helper exists to
            # prevent.
            owner = by_norm.get(resolve_tool_path(module_root, _file)) if _file else None
            if owner is None:
                continue
            collected[owner].append(SyntaxError_(
                file=owner,
                line=int(_line), col=int(_col),
                message=_msg,
                severity="error",
                code="",
            ))
            failed.add(owner)
        return {
            p: SyntaxValidationResult(
                ok=p not in failed,
                errors=collected[p],
                language=LanguageId.GO,
            )
            for p in file_paths
        }

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"func\s+{name}\s*\(",
                description="Go function declaration",
            ))
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"func\s+\([^)]+\)\s+{name}\s*\(",
                description="Go method declaration (receiver)",
            ))
        if kind in ("type", "class", "any"):
            patterns.append(SymbolPattern(
                kind="type",
                regex=r"type\s+{name}\s+struct\s*\{",
                description="Go struct type",
            ))
            patterns.append(SymbolPattern(
                kind="interface",
                regex=r"type\s+{name}\s+interface\s*\{",
                description="Go interface type",
            ))
        if kind in ("variable", "constant", "any"):
            patterns.append(SymbolPattern(
                kind="variable",
                regex=r"var\s+{name}\b",
                description="Go var declaration",
            ))
            patterns.append(SymbolPattern(
                kind="constant",
                regex=r"const\s+{name}\b",
                description="Go const declaration",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.go"]

    # ── Lint / test commands ──────────────────────────────────────────────

    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        return ["golangci-lint", "run", file_path]

    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        return ["go", "test", "./..."] + (test_args or [])

    # ── Symbol finder (tree-sitter → regex fallback) ──────────────────────

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Find symbol using tree-sitter (precise) or regex + brace counting (fallback)."""
        from .tree_sitter_utils import find_symbol_range, is_available

        if is_available():
            result = find_symbol_range(content, symbol_name, "go")
            if result:
                return result

        return self._find_symbol_regex(symbol_name, content)

    _LINE_BASED_KINDS = frozenset({"variable", "constant"})

    @staticmethod
    def _find_block_end(content: str, offset: int) -> int:
        """Heuristic: find the matching closing brace from *offset*.

        Delegates to the shared :func:`find_brace_block_end` (C-family SSOT)
        which skips string/char/template literals and ``//`` / ``/* */``
        comments so braces inside them do not corrupt the depth counter.
        """
        return find_brace_block_end(content, offset)

    # ── Definition keywords ───────────────────────────────────────────────

    # ── Regex fallback for structural queries ─────────────────────────────

    def _find_top_level_definitions_regex(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Regex fallback: find all top-level Go definitions via pattern + brace counting."""
        results: list[tuple[str, str, int, int]] = []
        for m in re.finditer(r'^func\s+(\w+)\s*\(', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "function", start_line, end_line))
        for m in re.finditer(r'^func\s+\([^)]*\)\s+(\w+)\s*\(', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "method", start_line, end_line))
        for m in re.finditer(r'^type\s+(\w+)\s+(struct|interface)\s*\{', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), m.group(2), start_line, end_line))
        for m in re.finditer(r'^var\s+(\w+)\b', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            # var declarations are single-line or grouped with parens
            end_pos = content.find("\n", m.end())
            if end_pos == -1:
                end_pos = len(content)
            end_line = content[:end_pos].count("\n") + 1
            results.append((m.group(1), "variable", start_line, end_line))
        for m in re.finditer(r'^const\s+(\w+)\b', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_pos = content.find("\n", m.end())
            if end_pos == -1:
                end_pos = len(content)
            end_line = content[:end_pos].count("\n") + 1
            results.append((m.group(1), "constant", start_line, end_line))
        return results

    def _find_all_class_methods_regex(
        self, content: str,
    ) -> dict[str, list[tuple[str, int, int]]]:
        """Regex fallback: group methods by normalized receiver type (batch).

        Receiver normalization matches the tree-sitter path
        (``tree_sitter_utils._extract_go_class_methods``): last
        space-delimited token of the receiver clause, ``*`` stripped.
        Grouping once also makes the fallback agree with the tree-sitter
        receiver-type semantics instead of the old per-class ``\\b<class>\\b``
        anywhere-in-receiver match.
        """
        grouped: dict[str, list[tuple[str, int, int]]] = {}
        for m in re.finditer(
            r'^func\s+\(([^)]*)\)\s+(\w+)\s*\(',
            content, re.MULTILINE,
        ):
            _recv = m.group(1).strip()
            _parts = _recv.split()
            _recv_type = _parts[-1] if len(_parts) >= 2 else _recv
            _recv_type = _recv_type.replace("*", "").strip()
            if not _recv_type:
                continue
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            grouped.setdefault(_recv_type, []).append(
                (m.group(2), start_line, end_line)
            )
        return grouped

    # ── Structural query methods (tree-sitter → regex fallback) ────────────

    def find_top_level_definitions(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        from .tree_sitter_utils import find_all_symbols, is_available
        result = find_all_symbols(content, "go") if is_available() else None
        if result:
            return result
        return self._find_top_level_definitions_regex(content)

    def find_class_methods(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Return ``[(method_name, start_line, end_line), ...]`` for a Go struct.

        Delegates to the batch :meth:`find_all_class_methods` so per-class
        lookups share its single parse + walk (receiver-based grouping).
        """
        return self.find_all_class_methods(content).get(class_name, [])

    def find_all_class_methods(
        self, content: str,
    ) -> dict[str, list[tuple[str, int, int]]]:
        """Return ``{class_name: [(method_name, start_line, end_line), ...]}``.

        Batch variant of :meth:`find_class_methods`: parses the source exactly
        once and groups every method by receiver type, avoiding one
        tree-sitter parse per class lookup.  Tree-sitter first, regex fallback
        otherwise (same split as :meth:`find_class_methods`).
        """
        from .tree_sitter_utils import extract_all_class_methods, is_available
        if is_available():
            grouped = extract_all_class_methods(content, "go")
            if grouped is not None:
                return grouped
        return self._find_all_class_methods_regex(content)

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        from .tree_sitter_utils import extract_symbol_body, is_available
        result = extract_symbol_body(content, symbol_name, "go") if is_available() else None
        if result:
            return result
        return self._find_symbol_body_range_regex(content, symbol_name)

    def get_definition_keywords(self) -> list[str]:
        return ["func ", "type "]
