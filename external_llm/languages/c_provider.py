"""C / C++ syntax providers.

Uses ``gcc -fsyntax-only`` (C) / ``g++ -fsyntax-only`` (C++), falling back to
``clang`` / ``clang++`` when the GNU toolchain is absent. Gracefully degrades
to ``ok=True`` when no compiler is on ``$PATH``: the validator's tree-sitter
fallback (see :func:`base.tree_sitter_syntax_fallback`) then provides a
zero-toolchain syntax check, identical to the pre-provider behaviour.

Symbol detection uses tree-sitter (precise) with a regex + brace-counting
fallback, mirroring :mod:`go_provider`.

Resolution safety (why ``ok=True`` is permissive on an isolated temp file):
an isolated ``.c``/``.cpp`` temp file is NOT part of the real build, so
``#include "project.h"`` and cross-TU symbols cannot resolve. ``gcc`` therefore
emits ``fatal error: project.h: No such file or directory`` for valid code; such
messages are dropped via :func:`base._filter_genuine_syntax_errors` so they do
not roll back a valid edit. The on-disk :meth:`validate_semantics` pass (run
from the file's directory so sibling headers resolve) is the authoritative
check.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Optional

from .base import (
    SyntaxProvider,
    _compile_env,
    _filter_genuine_syntax_errors,
    _replace_last_cmd_path,
    _tempfile_for_content,
    find_brace_block_end,
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
        has_linter=False,
        has_test_runner=False,
        has_symbol_search=True,
        has_tree_sitter=is_available(),
        supports_modify_symbol=True,
        supports_insert_after_symbol=True,
    )


# gcc/clang diagnostic shape:
#   file.c:10:5: error: expected ';' after expression
#   file.c:3:10: fatal error: foo.h: No such file or directory
# Column is always emitted by modern gcc/clang but made optional for robustness.
_CC_ERROR_RE = re.compile(
    r"^(.+?):(\d+):(?:(\d+):)?\s*(error|fatal error):\s+(.+)$"
)

# Prefer GNU; fall back to LLVM. First hit on $PATH wins.
_C_COMPILERS: tuple[str, ...] = ("gcc", "clang")
_CPP_COMPILERS: tuple[str, ...] = ("g++", "clang++")




def _is_c_family_header(file_path: str) -> bool:
    """A ``.h`` header is ambiguous: equally valid as C or C++ source.

    Unlike ``.hpp``/``.hh`` (unambiguously C++) or ``.c`` (unambiguously C),
    ``.h`` is used for both. The extension router (``_EXT_MAP``) picks C, so a
    valid C++ header (``namespace``, templates, …) reports genuine C syntax
    errors under gcc-C and would roll back a valid edit. The union retry
    (gcc-C → g++-CPP) in :meth:`_CFamilySyntaxProvider._validate_syntax_impl`
    and :meth:`validate_semantics` neutralises this ambiguity.
    """
    return file_path.endswith(".h")


def _parse_cc_diagnostics(
    output: str, lang: LanguageId, filter_resolution: bool,
) -> list[SyntaxError_]:
    """Parse ``file:line:col: (error|fatal error): msg`` lines into diagnostics.

    Warnings / notes are ignored (only ``error`` / ``fatal error`` block the
    check). When *filter_resolution* is True, resolution/environment messages
    (e.g. a missing ``#include`` header) are dropped so an isolated compile of
    otherwise-valid code does not produce false-negative rollbacks.
    """
    errors: list[SyntaxError_] = []
    for line in output.splitlines():
        m = _CC_ERROR_RE.match(line)
        if not m:
            continue
        errors.append(SyntaxError_(
            file=m.group(1),
            line=int(m.group(2)),
            col=int(m.group(3)) if m.group(3) else 0,
            message=m.group(5),
            severity="error",
        ))
    if filter_resolution:
        errors = _filter_genuine_syntax_errors(errors, lang)
    return errors


def _find_compile_commands(file_path: str) -> str | None:
    """Walk upward from *file_path* looking for ``compile_commands.json``."""
    cur = os.path.dirname(os.path.abspath(file_path)) or "."
    while True:
        candidate = os.path.join(cur, "compile_commands.json")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _extract_include_flags(entry: dict) -> list[str]:
    """Extract ``-I`` include flags from a ``compile_commands.json`` entry.

    Prefers the ``arguments`` array (exact, no shell parsing); falls back to
    ``command`` string (simple shell-split via :func:`shlex.split`).

    Relative ``-I`` paths are resolved against the entry's ``directory`` field
    so they work regardless of the compiler's working directory.
    """
    base_dir = entry.get("directory", "")

    # Prefer 'arguments' array — exact tokens, no shell parsing needed.
    args = entry.get("arguments")
    if args:
        flags = _collect_I_flags(args)
    else:
        # Fall back to 'command' string — requires shell-split.
        cmd = entry.get("command", "")
        if cmd:
            import shlex
            try:
                tokens = shlex.split(cmd)
            except ValueError:
                tokens = cmd.split()
            flags = _collect_I_flags(tokens)
        else:
            return []

    # Resolve relative -I paths against the entry's directory.
    if base_dir:
        resolved: list[str] = []
        for flag in flags:
            if flag.startswith("-I") and len(flag) > 2:
                path = flag[2:]
                if path and not os.path.isabs(path):
                    resolved.append(f"-I{os.path.normpath(os.path.join(base_dir, path))}")
                    continue
            resolved.append(flag)
        return resolved
    return flags


def _collect_I_flags(tokens: list[str]) -> list[str]:
    """Extract -I<dir> and -I <dir> flags from a token list."""
    flags: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "-I" and i + 1 < len(tokens):
            flags.append(f"-I{tokens[i + 1]}")
            i += 2
            continue
        if token.startswith("-I") and len(token) > 2:
            flags.append(token)
        i += 1
    return flags


class _CFamilySyntaxProvider(SyntaxProvider):
    """Shared C/C++ provider logic. Subclasses pin the language + toolchain."""

    _lang: LanguageId
    _suffix: str
    _compilers: tuple[str, ...]
    _globs: tuple[str, ...]
    _caps: Optional[LanguageCapabilities] = None

    def capabilities(self) -> LanguageCapabilities:
        if self._caps is None:
            self._caps = _make_capabilities()
        return self._caps

    # ── Syntax validation (content → temp file) ──────────────────────────

    def _resolve_compilers(self, candidates: tuple[str, ...]) -> Optional[str]:
        """Return the first compiler in *candidates* on ``$PATH``, cached.

        Cached **per instance** keyed by *candidates*. The singleton registry
        provider reuses the cache across the agent loop (the toolchain does not
        change mid-session), while a freshly constructed provider in a test
        starts with an empty cache — so a ``shutil.which`` mock on a fresh
        instance always fires. This de-duplicates the ``$PATH`` scan across the
        un-memoised :meth:`validate_semantics` path (``validate_syntax`` is
        already memoised at the base class).
        """
        cache = self.__dict__.get("_compiler_cache")
        if cache is None:
            cache = {}
            self.__dict__["_compiler_cache"] = cache
        if candidates in cache:
            return cache[candidates]
        cc = next((c for c in candidates if shutil.which(c)), None)
        cache[candidates] = cc
        return cc

    def _run_syntax_compile(
        self, file_path: str, content: str, suffix: str,
        compilers: tuple[str, ...], lang: LanguageId,
    ) -> SyntaxValidationResult:
        """Compile *content* via one of *compilers* on an isolated temp file.

        Resolution errors (missing headers / cross-TU symbols) are filtered so
        an isolated temp-file compile does not roll back a valid edit. The temp
        file uses *suffix* so the compiler infers the language (``.c`` → C,
        ``.cpp`` → C++).
        """
        cc = self._resolve_compilers(compilers)
        if cc is None:
            logger.debug("%s not installed; falling back to tree-sitter", compilers[0])
            return tree_sitter_syntax_fallback(content, lang)

        _tmp_path, _cleanup = _tempfile_for_content(content, suffix)
        if not _tmp_path:
            return tree_sitter_syntax_fallback(content, lang)

        _cmd = _replace_last_cmd_path(
            [cc, "-fsyntax-only", "-Wall", file_path],
            file_path, _tmp_path,
        )
        try:
            try:
                proc = subprocess.run(
                    _cmd, capture_output=True, text=True, timeout=30,
                    env=_compile_env(),
                )
            except FileNotFoundError:
                logger.debug("%s vanished mid-run; falling back to tree-sitter", cc)
                return tree_sitter_syntax_fallback(content, lang)
            except subprocess.TimeoutExpired:
                logger.debug("%s timed out for %s; falling back to tree-sitter", cc, file_path)
                return tree_sitter_syntax_fallback(content, lang)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("%s error: %s; falling back to tree-sitter", cc, e)
                return tree_sitter_syntax_fallback(content, lang)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=lang)

            errors = _parse_cc_diagnostics(
                proc.stdout + proc.stderr, lang, filter_resolution=True,
            )
            if not errors:
                return SyntaxValidationResult(ok=True, language=lang)
            return SyntaxValidationResult(ok=False, errors=errors, language=lang)
        finally:
            _cleanup()

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate *content* via ``<cc> -fsyntax-only`` on an isolated temp file.

        Permissive on resolution failures (missing headers / cross-TU symbols):
        they are dropped so a valid edit is not rolled back. The post-write
        :meth:`validate_semantics` pass catches real resolution errors.

        **``.h`` union validation**: a ``.h`` header is ambiguous (C or C++),
        but ``_EXT_MAP`` routes it to the C provider. A valid C++ header
        (``namespace``, templates, …) therefore reports genuine C syntax errors
        under gcc-C. When the primary C compile fails AND the file is a ``.h``,
        we retry with the C++ toolchain; if C++ accepts it the edit is valid and
        must not roll back. Both compilers absent → tree-sitter C, then
        tree-sitter C++ fallback for ``.h``.
        """
        result = self._run_syntax_compile(
            file_path, content, self._suffix, self._compilers, self._lang,
        )
        if (
            not result.ok
            and self._lang is LanguageId.C
            and _is_c_family_header(file_path)
        ):
            cpp = self._run_syntax_compile(
                file_path, content, ".cpp", _CPP_COMPILERS, LanguageId.CPP,
            )
            if cpp.ok:
                return SyntaxValidationResult(ok=True, language=self._lang)
        return result

    # ── Semantic validation (on-disk file, sibling includes resolve) ──────

    def _run_semantics_compile(
        self, file_path: str, target: str, cwd: str,
        include_flags: list[str], compilers: tuple[str, ...], lang: LanguageId,
    ) -> SyntaxValidationResult:
        """Run ``<cc> -fsyntax-only`` on the on-disk file from *cwd*.

        Mirrors :meth:`_run_syntax_compile` but operates on the real file so
        ``#include "sibling.h"`` resolves. Only diagnostics located in *target*
        are kept (a directory build surfaces sibling-TU noise). Permissive on
        resolution failures.
        """
        cc = self._resolve_compilers(compilers)
        if cc is None:
            return SyntaxValidationResult(ok=True, language=lang)

        _cmd = [cc, "-fsyntax-only", "-Wall"]
        _cmd.extend(include_flags)
        _cmd.append(target)

        try:
            proc = subprocess.run(
                _cmd,
                capture_output=True, text=True, timeout=120, cwd=cwd,
                env=_compile_env(),
            )
        except FileNotFoundError:
            logger.debug("%s not installed; skipping semantic validation", cc)
            return SyntaxValidationResult(ok=True, language=lang)
        except subprocess.TimeoutExpired:
            logger.debug("%s timed out for %s; skipping", cc, file_path)
            return SyntaxValidationResult(ok=True, language=lang)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("%s semantic check failed: %s", cc, e)
            return SyntaxValidationResult(ok=True, language=lang)

        if proc.returncode == 0:
            return SyntaxValidationResult(ok=True, language=lang)

        errors: list[SyntaxError_] = []
        for e in _parse_cc_diagnostics(
            proc.stdout + proc.stderr, lang, filter_resolution=True,
        ):
            # A directory compile surfaces errors from sibling TUs; report only
            # diagnostics located in the file we asked about.
            if os.path.basename(e.file) != target:
                continue
            errors.append(SyntaxError_(
                file=file_path, line=e.line, col=e.col,
                message=e.message, severity=e.severity,
            ))
        return SyntaxValidationResult(
            ok=not errors, errors=errors, language=lang,
        )

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run ``<cc> -fsyntax-only`` on the **on-disk** file from its directory.

        Running from the file's own directory lets ``#include "sibling.h"``
        resolve. When a ``compile_commands.json`` is found upward, ``-I`` flags
        from the matching entry are injected so cross-directory includes also
        resolve — mirroring Go's ``go.mod``-driven module-root compilation.
        This is a best-effort check, intentionally lenient — resolution
        failures are still filtered to avoid false-negatives.

        **``.h`` union validation**: same ambiguity as
        :meth:`_validate_syntax_impl`. g++ treats a ``.h`` file as C++ by
        default, so the C++ retry parses C++ headers correctly.
        """
        if not file_path or not os.path.exists(file_path):
            return SyntaxValidationResult(ok=True, language=self._lang)

        cwd = os.path.dirname(os.path.abspath(file_path)) or "."
        target = os.path.basename(file_path)

        # ── compile_commands.json include-path injection ─────────────────
        _include_flags: list[str] = []
        _ccdb_path = _find_compile_commands(file_path)
        if _ccdb_path:
            try:
                with open(_ccdb_path, encoding="utf-8") as _f:
                    _entries = json.load(_f)
                for _entry in _entries:
                    _entry_file = _entry.get("file", "")
                    if os.path.basename(_entry_file) == target or _entry_file == file_path:
                        _include_flags = _extract_include_flags(_entry)
                        break
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                pass

        result = self._run_semantics_compile(
            file_path, target, cwd, _include_flags, self._compilers, self._lang,
        )
        if (
            not result.ok
            and self._lang is LanguageId.C
            and _is_c_family_header(file_path)
        ):
            cpp = self._run_semantics_compile(
                file_path, target, cwd, _include_flags, _CPP_COMPILERS, LanguageId.CPP,
            )
            if cpp.ok:
                return SyntaxValidationResult(ok=True, language=self._lang)
        return result

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            # <storage/return-type tokens> name(   — e.g. "static int *foo("
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"^[ \t]*[\w][\w\s\*]*?\b{name}\s*\(",
                description="C/C++ function declaration",
            ))
        if kind in ("type", "class", "struct", "any"):
            patterns.append(SymbolPattern(
                kind="struct",
                regex=r"\b(?:struct|union)\s+{name}\b",
                description="C/C++ struct/union declaration",
            ))
            patterns.append(SymbolPattern(
                kind="enum",
                regex=r"\benum\s+{name}\b",
                description="C/C++ enum declaration",
            ))
            patterns.append(SymbolPattern(
                kind="typedef",
                regex=r"\btypedef\b[\w\s\*]*?\b{name}\s*;",
                description="C/C++ typedef",
            ))
        if kind in ("variable", "constant", "macro", "any"):
            patterns.append(SymbolPattern(
                kind="macro",
                regex=r"^#define\s+{name}\b",
                description="C/C++ object-like / function-like macro",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return list(self._globs)

    # ── Lint / test commands ──────────────────────────────────────────────

    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        return None

    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        return None

    # ── Symbol finder (tree-sitter → regex fallback) ──────────────────────

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        from .tree_sitter_utils import find_symbol_range, is_available

        if is_available():
            result = find_symbol_range(content, symbol_name, self._lang.value)
            if result:
                return result
        return self._find_symbol_regex(file_path, symbol_name, content)

    def _find_symbol_regex(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        esc = re.escape(symbol_name)
        for sp in self.get_symbol_patterns("any"):
            pat = sp.regex.replace("{name}", esc)
            for m in re.finditer(pat, content, re.MULTILINE):
                start_offset = m.start()
                start_line = content[:start_offset].count("\n") + 1
                if sp.kind in ("macro", "typedef"):
                    end_pos = content.find("\n", m.end())
                    end_line = (content[:end_pos].count("\n") + 1) if end_pos != -1 else start_line
                else:
                    end_line = self._find_block_end(content, start_offset)
                return (start_line, end_line)
        return None

    @staticmethod
    def _find_block_end(content: str, offset: int) -> int:
        """Heuristic: find the matching closing brace from *offset*.

        Delegates to the shared :func:`find_brace_block_end` (C-family SSOT)
        which skips string/char literals and ``//`` / ``/* */`` comments so
        braces inside them do not corrupt the depth counter.
        """
        return find_brace_block_end(content, offset)

    # ── Regex fallback for structural queries ─────────────────────────────

    def _find_top_level_definitions_regex(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        results: list[tuple[str, str, int, int]] = []
        # functions: <type tokens> name( ... ) {
        for m in re.finditer(
            r"^[ \t]*[\w][\w\s\*]*?\b(\w+)\s*\(", content, re.MULTILINE,
        ):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "function", start_line, end_line))
        for m in re.finditer(
            r"\b(?:struct|union)\s+(\w+)\s*\{", content, re.MULTILINE,
        ):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "struct", start_line, end_line))
        for m in re.finditer(r"\benum\s+(\w+)\b", content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "enum", start_line, end_line))
        for m in re.finditer(r"^#define\s+(\w+)\b", content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_pos = content.find("\n", m.end())
            if end_pos == -1:
                end_pos = len(content)
            end_line = content[:end_pos].count("\n") + 1
            results.append((m.group(1), "macro", start_line, end_line))
        return results

    def _find_symbol_body_range_regex(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        esc = re.escape(symbol_name)
        for sp in self.get_symbol_patterns("any"):
            pat = sp.regex.replace("{name}", esc)
            for m in re.finditer(pat, content, re.MULTILINE):
                body_start = content.find("{", m.end())
                if body_start == -1:
                    continue
                body_start_line = content[:body_start].count("\n") + 1
                body_end_line = self._find_block_end(content, body_start)
                return (body_start_line, body_end_line)
        return None

    # ── Structural query methods (tree-sitter → regex fallback) ────────────

    def find_top_level_definitions(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        from .tree_sitter_utils import find_all_symbols, is_available
        result = find_all_symbols(content, self._lang.value) if is_available() else None
        if result:
            return result
        return self._find_top_level_definitions_regex(content)

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        from .tree_sitter_utils import extract_symbol_body, is_available
        result = extract_symbol_body(content, symbol_name, self._lang.value) if is_available() else None
        if result:
            return result
        return self._find_symbol_body_range_regex(content, symbol_name)

    def get_definition_keywords(self) -> list[str]:
        # C/C++ definitions are not introduced by a single keyword; callers use
        # this only as a hint, so return the common type/storage tokens.
        return ["struct ", "union ", "enum ", "typedef ", "#define "]


class CSyntaxProvider(_CFamilySyntaxProvider):
    """C language support (gcc/clang validation + tree-sitter symbols)."""

    _lang = LanguageId.C
    _suffix = ".c"
    _compilers = _C_COMPILERS
    _globs = ("*.c", "*.h")

    def language_id(self) -> LanguageId:
        return LanguageId.C


class CppSyntaxProvider(_CFamilySyntaxProvider):
    """C++ language support (g++/clang++ validation + tree-sitter symbols)."""

    _lang = LanguageId.CPP
    _suffix = ".cpp"
    _compilers = _CPP_COMPILERS
    _globs = ("*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh")

    def language_id(self) -> LanguageId:
        return LanguageId.CPP
