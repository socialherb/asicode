"""
Abstract base for language syntax providers.
"""
from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from external_llm.editor.primitives.code_context import CodeContext

from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxError_,
    SyntaxValidationResult,
)


def _sha256_hex(text: str) -> str:
    """Stable SHA-256 digest of *text* (UTF-8, replace on decode error)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _tempfile_for_content(content: str, suffix: str) -> tuple[str, Callable[[], None]]:
    """Write *content* to a temp file and return ``(tmp_path, cleanup_fn)``.

    The caller **must** call ``cleanup_fn()`` in a ``finally`` block when done
    (typically after running an external compiler/tool that requires an on-disk
    file).
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8",
        ) as _tf:
            _tf.write(content)
            _tmp_path: str = _tf.name
    except OSError:
        return "", lambda: None

    def _cleanup() -> None:
        try:
            os.unlink(_tmp_path)
        except OSError:
            pass

    return _tmp_path, _cleanup


def _replace_last_cmd_path(cmd: list[str], old_path: str, new_path: str) -> list[str]:
    """Replace *old_path* with *new_path* in the last element of *cmd*.

    The last element is always the file path in compiler/tool commands
    (``go build file.go``, ``javac File.java``, ``tsc --noEmit file.ts``, etc.).
    """
    if not cmd:
        return cmd
    result = cmd[:]
    result[-1] = cmd[-1].replace(old_path, new_path, 1)
    return result


def tree_sitter_syntax_fallback(
    content: str, lang_id: LanguageId,
) -> SyntaxValidationResult:
    """Syntax validation via tree-sitter for languages without a bundled toolchain.

    Providers for languages such as Rust, Ruby, Swift, PHP, and C# have no
    bundled compiler/linter, so their ``validate_syntax`` previously returned
    ``ok=True`` unconditionally — silently disabling the syntax gate on every
    write path.  Tree-sitter ships a grammar for these languages, so this
    helper gives them a real syntax check at zero toolchain cost.

    Returns ``SyntaxValidationResult(ok=True)`` when tree-sitter is unavailable
    (e.g. grammar not installed) to stay permissive — the same contract as
    ``SyntaxValidator.validate_syntax``'s own fallback.
    """
    from . import tree_sitter_utils as ts_utils

    has_err = ts_utils.has_error(content, lang_id.value)
    if has_err is None:
        return SyntaxValidationResult(ok=True, language=lang_id)
    if has_err:
        return SyntaxValidationResult(
            ok=False,
            errors=[SyntaxError_(
                file="", line=0, col=0,
                message="syntax error (tree-sitter)",
            )],
            language=lang_id,
        )
    return SyntaxValidationResult(ok=True, language=lang_id)


def detect_project_root(file_path: str, markers: tuple[str, ...] = ()) -> str:
    """Walk upward from *file_path*'s directory looking for a project marker.

    Returns the directory containing the first marker found, or the file's own
    directory as a last-resort fallback. Used by semantic validators that need
    a ``cwd`` so the backing tool can resolve imports / config files.

    Default markers cover the common multi-language config files. Language
    providers may pass their own (e.g. ``("go.mod",)`` for Go).
    """
    default_markers = (
        "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
        "tsconfig.json", "package.json", "go.mod", "Cargo.toml",
        "build.gradle", "pom.xml", "build.gradle.kts",
        ".git",
    )
    used_markers = markers or default_markers
    start = os.path.dirname(os.path.abspath(file_path)) or "."
    cur = start
    while True:
        for m in used_markers:
            if os.path.exists(os.path.join(cur, m)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            break
        cur = parent
    return start


# ── Resolution-error classification ─────────────────────────────────────
# Pre-write ``validate_syntax`` compiles an isolated temp file WITHOUT the
# surrounding project context (go.mod / sourcepath / classpath / node_modules).
# Compilers therefore emit symbol/module/package RESOLUTION failures for code
# that is perfectly valid inside its real project. Such failures are NOT syntax
# errors and must not block the pre-write syntax gate — the on-disk
# :meth:`SyntaxProvider.validate_semantics` pass (run from the project root with
# full context, after the write) is the authoritative check for them. Only the
# TypeScript (``is_genuine_syntax_error`` TS1xxx filter) and JavaScript
# (``node --check`` is pure-parse, never resolves) providers are inherently
# immune; Go/Java/Kotlin use the helpers below.
_RESOLUTION_ERROR_PHRASES: dict[LanguageId, tuple[str, ...]] = {
    # ``go build`` on an isolated file with non-stdlib imports.
    LanguageId.GO: (
        "no required module provides package",
        "go.mod file not found",
        "missing go.sum entry",
        "cannot find module providing package",
    ),
    # ``javac`` without -sourcepath/-classpath.
    LanguageId.JAVA: (
        "does not exist",          # "package foo does not exist"
        "cannot find symbol",
        "is public, should be declared",
    ),
    # ``kotlinc`` without classpath (incl. -script mode).
    LanguageId.KOTLIN: (
        "unresolved reference",
        "cannot find symbol",
    ),
}


def _is_resolution_error(message: str, lang_id: LanguageId) -> bool:
    """Return True if a compiler *message* is a RESOLUTION/semantic failure
    rather than a syntax error in the source itself.

    See :data:`_RESOLUTION_ERROR_PHRASES` for the per-language indicator
    phrases. Matching is case-insensitive substring.
    """
    phrases = _RESOLUTION_ERROR_PHRASES.get(lang_id, ())
    if not phrases:
        return False
    lowered = message.lower()
    return any(p in lowered for p in phrases)


def _filter_genuine_syntax_errors(
    errors: list[SyntaxError_], lang_id: LanguageId,
) -> list[SyntaxError_]:
    """Drop resolution/semantic errors, keeping only genuine SYNTAX errors.

    Used by content-based ``validate_syntax`` implementations so an isolated
    temp-file compile that cannot resolve project imports does not produce a
    false-negative rollback of an otherwise valid edit. The on-disk
    :meth:`validate_semantics` pass catches real resolution errors later.
    """
    return [e for e in errors if not _is_resolution_error(e.message, lang_id)]
class SyntaxProvider(ABC):
    """Interface that each language must implement."""

    @abstractmethod
    def language_id(self) -> LanguageId:
        ...

    @abstractmethod
    def capabilities(self) -> LanguageCapabilities:
        ...

    _SYNTAX_MEMO_MAX = 128

    def validate_syntax(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate *content* (the file body), memoised per provider instance.

        Syntax validity is a pure function of (file-path suffix, content), so
        identical ``(file_path, content)`` always reproduces the prior result.
        The instance memo de-duplicates the expensive subprocess validators
        (tsc / javac) across the agent edit-validate loop.

        The memo is **per-instance**: a freshly constructed provider (e.g. in
        tests that mock the subprocess) starts with an empty memo and always
        recomputes, so the mock fires; the registry singleton used by the agent
        loop keeps its memo across turns.

        Returns a deep copy so callers cannot mutate the cached result.
        """
        memo = self.__dict__.get("_syntax_memo")
        if memo is None:
            memo = OrderedDict()
            self.__dict__["_syntax_memo"] = memo
        key = (file_path, _sha256_hex(content))
        cached = memo.get(key)
        if cached is not None:
            memo.move_to_end(key)
            return copy.deepcopy(cached)
        result = self._validate_syntax_impl(file_path, content)
        memo[key] = result
        while len(memo) > self._SYNTAX_MEMO_MAX:
            memo.popitem(last=False)
        return copy.deepcopy(result)

    @abstractmethod
    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Per-language syntax check — implement in each subclass.

        ``validate_syntax`` memoises this. Subclasses must NOT shadow
        ``validate_syntax``; override this hook instead so the memo applies.
        """
        ...

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run a semantic check (undefined names, types, missing imports) on the
        on-disk file at *file_path*.

        Unlike :meth:`validate_syntax`, this operates on the **real file on disk**
        (not a temp file), so the backing tool can resolve imports / types against
        the surrounding project. This is what enables catching errors such as
        ``undefined: db`` or ``Import "x" could not be resolved`` that pure-syntax
        validation misses.

        Returns ``SyntaxValidationResult(ok=True)`` by default — subclasses opt in
        by setting ``LanguageCapabilities.has_semantic_validator = True`` and
        overriding this method. Errors (severity == "error") make ``ok=False``;
        warnings/info are reported but do not fail the check.
        """
        return SyntaxValidationResult(ok=True, language=self.language_id())

    @abstractmethod
    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        """Return regex patterns for finding symbol definitions.

        *kind* can be ``"function"``, ``"class"``, ``"any"``, etc.
        """
        ...

    @abstractmethod
    def get_file_globs(self) -> list[str]:
        """Glob patterns that match files of this language (e.g. ``["*.py"]``)."""
        ...

    @abstractmethod
    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        """Return the shell command to lint *file_path*, or ``None``."""
        ...

    def get_test_directory(self, repo_root: str) -> Optional[str]:
        """Return the configured test directory for this language/project.

        Returns the configured test root (e.g. "__tests__", "tests", "spec")
        as determined by project config files (jest.config.js, package.json, etc.).
        Returns ``None`` to fall back to convention-based detection
        (``tests/`` for most languages).
        """
        return None  # Default: use convention (tests/)

    @abstractmethod
    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        """Return the shell command to run tests, or ``None``."""
        ...

    @abstractmethod
    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Return ``(start_line, end_line)`` (1-indexed) of *symbol_name* in *content*, or ``None``."""
        ...

    # ── Structural query methods (Stage 2: cross-language DPB) ────────────────

    def find_top_level_definitions(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Return ``[(name, kind, start_line, end_line), ...]`` for all top-level
        definitions in *content* (functions, classes, interfaces, types, etc.).

        *kind* is one of ``"function"``, ``"class"``, ``"interface"``, ``"type"``,
        ``"enum"``, ``"assignment"``, etc.

        Default implementation delegates to ``find_all_symbols`` from
        ``tree_sitter_utils`` (tree-sitter → regex fallback).
        Subclasses may override with a faster path (e.g. Python ast.parse).

        Returns empty list when tree-sitter is unavailable and no fallback is
        implemented.
        """
        return []

    def find_class_methods(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Return ``[(method_name, start_line, end_line), ...]`` for methods of a
        class in *content*.

        Handles language-specific class body structures:
        - Python: ``class_definition → block → function_definition``
        - TS/JS: ``class_declaration → class_body → method_definition``
        - Java: ``class_declaration → class_body → method_declaration``
        - Kotlin: ``class_declaration → class_body → function_declaration``
        - Go: ``method_declaration`` with receiver matching *class_name*

        Returns empty list if the class is not found or tree-sitter is unavailable.
        """
        return []

    def find_all_class_methods(
        self, content: str,
    ) -> dict[str, list[tuple[str, int, int]]]:
        """Return ``{class_name: [(method_name, start_line, end_line), ...]}``.

        Batch variant of ``find_class_methods``: subclasses that parse the
        source once (e.g. Python ast.parse) should override this to avoid
        re-parsing the same content for every class. The default implementation
        delegates to ``find_class_symbols`` + per-class ``find_class_methods``
        for backwards compatibility with providers that have not been migrated.

        Only classes that actually contain methods need to be present in the
        returned dict; callers treat a missing key as "no methods".
        """
        classes = self.find_class_symbols(content)
        if not classes:
            return {}
        result: dict[str, list[tuple[str, int, int]]] = {}
        for class_name in classes:
            methods = self.find_class_methods(content, class_name)
            if methods:
                result[class_name] = methods
        return result

    def find_class_symbols(
        self, content: str,
    ) -> list[str]:
        """Return the names of all top-level class/type/struct/interface symbols.

        Used by the default ``find_all_class_methods`` to enumerate classes
        without re-parsing. The default implementation returns an empty list
        (providers that own a faster batch path override ``find_all_class_methods``
        directly and never reach this).
        """
        return []

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        """Return ``(body_start_line, body_end_line)`` (1-indexed) for a function
        or method's executable body (indented block or brace-delimited block).

        Returns None if the symbol is not found or tree-sitter is unavailable.
        """
        return None

    # ── SyntaxValidator dispatch helpers ───────────────────────────────────

    def find_symbol_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        """Return ``(start_line, end_line)`` (1-indexed) for *symbol_name* in *content*.

        Default implementation delegates to ``find_symbol_in_file`` with an
        empty file path. Subclasses may override for a faster path.
        """
        result = self.find_symbol_in_file("", symbol_name, content)
        if result is not None:
            return (result[0], result[1])
        return None

    def find_symbols(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Enumerate all top-level symbols in *content*.

        Returns ``[(name, kind, start_line, end_line), ...]`` or empty list.
        Default delegates to ``find_top_level_definitions``.
        """
        return self.find_top_level_definitions(content)

    def extract_symbol_body(
        self, code: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        """Return ``(body_start_line, body_end_line)`` (1-indexed) for a function/method.

        The body excludes the signature line.
        Default delegates to ``find_symbol_body_range``.
        """
        return self.find_symbol_body_range(code, symbol_name)

    def is_dead_code_introduced(self, orig: str, new: str) -> bool:
        """Check if *new* introduces dead code compared to *orig*.
        Default: conservative — validates syntax of new code.
        Subclasses may override with language-specific logic.
        """
        result = self.validate_syntax("", new)
        return not result.ok

    @abstractmethod
    def get_definition_keywords(self) -> list[str]:
        """Return keyword prefixes used to define symbols (e.g. ``["def ", "class "]``)."""
        ...

    # ── Primitive execution support ───────────────────────────────────────

    def build_code_context(self, code: str, file_path: str) -> "CodeContext":
        """Build a language-agnostic CodeContext for primitive operations.

        The default implementation uses tree-sitter (or regex fallback) via
        ``CodeContext.from_file_path()``. Language providers may override to
        provide a faster or more precise implementation.

        Args:
            code: Source code string.
            file_path: File path (used for language inference).

        Returns:
            A ``CodeContext`` instance ready for primitive execution.
        """
        from external_llm.editor.primitives.code_context import CodeContext as _CC

        return _CC.from_file_path(code, file_path)
