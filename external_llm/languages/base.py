"""
Abstract base for language syntax providers.
"""
from __future__ import annotations

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
    content: str, lang_id: LanguageId, file_path: str = "",
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

    # Map file extension → tree-sitter grammar key (".tsx" → "tsx",
    # ".mjs"/".cjs" → "javascript", etc).  When the path is unknown or
    # absent, fall back to the LanguageId-based key (e.g. "typescript").
    _lang_key = ts_utils.grammar_key_for_path(file_path or "") or lang_id.value

    err_nodes = ts_utils.find_error_nodes(content, _lang_key)
    if err_nodes is None:
        return SyntaxValidationResult(ok=True, language=lang_id)
    if err_nodes:
        return SyntaxValidationResult(
            ok=False,
            errors=[SyntaxError_(
                file="",
                line=n.line,
                col=n.column,
                message=(
                    f"syntax error (tree-sitter): expected '{n.missing_token}'"
                    if n.missing_token
                    else "syntax error (tree-sitter)"
                ),
            ) for n in err_nodes],
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
# Forced locale for compiler subprocess calls: LC_ALL=C ensures English
# error messages for reliable phrase matching in _is_resolution_error.
_COMPILE_ENV: dict[str, str] = {"LC_ALL": "C", "LANG": "C"}


def _compile_env() -> dict[str, str]:
    """Return env dict with forced C locale for stable English compiler output.

    Overrides ``LC_ALL`` and ``LANG`` while preserving the rest of the parent
    environment.  Used by all compiler/tool subprocess calls whose output is
    parsed by :func:`_is_resolution_error`.
    """
    return {**os.environ, **_COMPILE_ENV}


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
    # ``gcc -fsyntax-only`` / ``clang -fsyntax-only`` on an isolated .c file.
    # Missing project-local/system headers are environmental, not syntax
    # errors in the source itself. gcc says "No such file or directory";
    # clang (the default ``gcc`` on macOS) says "file not found" — both are
    # listed so the filter is compiler-agnostic.
    LanguageId.C: (
        "no such file or directory",
        "file not found",
        "implicit declaration of",
    ),
    # ``g++`` / ``clang++`` without include paths; undeclared symbols usually
    # stem from an unresolvable header in the isolated temp-file compile.
    LanguageId.CPP: (
        "no such file or directory",
        "file not found",
        "was not declared in this scope",
        "has not been declared",
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


# Context-dependent resolution phrases — only filtered when an actual
# resolution failure (see :data:`_RESOLUTION_CONTEXT_PHRASES`) is ALSO present
# in the compiler output.  Without that context, these represent genuine typos
# that must gate the edit.
_CONTEXT_DEPENDENT_PHRASES: dict[LanguageId, tuple[str, ...]] = {
    LanguageId.C: ("implicit declaration of",),
    LanguageId.CPP: (
        "was not declared in this scope",
        "has not been declared",
    ),
    # javac's "cannot find symbol" cascades from a failed ``import`` (package
    # X does not exist) but is ALSO how a genuine local typo (``total +=
    # valeu``) surfaces — indistinguishable except by co-occurring context.
    LanguageId.JAVA: ("cannot find symbol",),
    # kotlinc's "unresolved reference" is emitted IDENTICALLY for a failed
    # import AND a genuine typo — there is no phrase-level disambiguator (unlike
    # Java's distinct "does not exist" vs "cannot find symbol"). The only reliable
    # signal that an import failed in the isolated temp-file compile is an
    # unresolved reference ON AN IMPORT LINE (kotlinc reports the unresolved
    # package segment there). The Kotlin provider computes that line-based context
    # and passes it via ``has_resolution_context``; without it, a bare unresolved
    # reference is a genuine typo that must gate the edit.
    LanguageId.KOTLIN: ("unresolved reference",),
}

# Resolution-context indicators — the messages whose presence proves the
# compile failed to *resolve external references* (a missing header / import),
# so co-occurring context-dependent phrases are cascade noise, not typos.
#
# This is deliberately NOT "every unconditional resolution phrase": e.g. javac's
# "class X is public, should be declared in a file named X.java" is an
# unconditional-drop phrase (the isolated temp file is randomly named, so every
# public class triggers it) but is ORTHOGONAL to symbol resolution — it
# co-occurs with genuine typos in public classes.  Treating it as context would
# silently drop those typos.  Only genuinely resolution-implying messages belong
# here.
#
# Kotlin has NO phrase-based context here: kotlinc's import failure and typo
# both say "unresolved reference" (see :data:`_CONTEXT_DEPENDENT_PHRASES`). The
# Kotlin provider instead derives context LINE-BASED (an unresolved reference on
# an import line proves a dependency failed) and supplies it via the
# ``has_resolution_context`` argument of :func:`_filter_genuine_syntax_errors`.
_RESOLUTION_CONTEXT_PHRASES: dict[LanguageId, tuple[str, ...]] = {
    LanguageId.C: ("no such file", "file not found"),
    LanguageId.CPP: ("no such file", "file not found"),
    LanguageId.JAVA: ("does not exist",),  # "package foo does not exist"
}


def _filter_genuine_syntax_errors(
    errors: list[SyntaxError_], lang_id: LanguageId,
    *,
    has_resolution_context: Optional[bool] = None,
) -> list[SyntaxError_]:
    """Drop resolution/semantic errors, keeping only genuine SYNTAX errors.

    Used by content-based ``validate_syntax`` implementations so an isolated
    temp-file compile that cannot resolve project imports does not produce a
    false-negative rollback of an otherwise valid edit. The on-disk
    :meth:`validate_semantics` pass catches real resolution errors later.

    **Context-dependent filtering (C/C++/Java/Kotlin):** some compiler messages
    (e.g. "was not declared in this scope", "implicit declaration of", javac's
    "cannot find symbol", kotlinc's "unresolved reference") can mean either a
    cascade from a failed ``#include`` / ``import`` *or* a genuine typo in the
    edited source.  The filter only drops these when the compiler output also
    indicates a resolution failure: by default this is detected from the
    per-language phrases in :data:`_RESOLUTION_CONTEXT_PHRASES`
    ("no such file or directory" / "file not found" for C/C++, "package X does
    not exist" for Java).  Without that context, the undeclared-identifier
    message is a real error that must gate the edit — otherwise
    ``total += valeu;`` would silently pass the syntax gate when g++ reports
    "was not declared in this scope", javac reports "cannot find symbol", or
    kotlinc reports "unresolved reference".

    ``has_resolution_context`` lets a caller supply the resolution-context
    decision directly, bypassing the phrase-based detection.  Kotlin needs this:
    its import failure and typo are textually identical ("unresolved reference"),
    so the Kotlin provider derives the context LINE-BASED (an unresolved
    reference on an import line proves a dependency failed) and passes the result
    here.  When ``None`` (the default), the context is computed from
    :data:`_RESOLUTION_CONTEXT_PHRASES` as before.
    """
    # ── Does the output contain a resolution failure? (the context that makes
    #    context-dependent phrases resolution noise rather than real typos.)
    if has_resolution_context is None:
        _ctx_indicators = _RESOLUTION_CONTEXT_PHRASES.get(lang_id, ())
        has_resolution_context = any(
            any(ind in e.message.lower() for ind in _ctx_indicators)
            for e in errors
        )

    _ctx_phrases = _CONTEXT_DEPENDENT_PHRASES.get(lang_id, ())

    result: list[SyntaxError_] = []
    for e in errors:
        if not _is_resolution_error(e.message, lang_id):
            # Genuine syntax error — always keep.
            result.append(e)
            continue

        # Context-dependent resolution error — keep if no resolution context.
        if not has_resolution_context and _ctx_phrases:
            lowered = e.message.lower()
            if any(p in lowered for p in _ctx_phrases):
                result.append(e)
                continue

        # Resolution error (include failure or unconditional phrase) — drop.
    return result


def _skip_quoted_literal(content: str, start: int, length: int, quote: str, escapes: bool = True) -> int:
    """Skip a string/char/template literal whose opening *quote* is at *start*.

    Returns the index of the closing quote (or *length* if unterminated).
    When *escapes* is True (default), handles ``\\`` escapes so ``"\\\\"``
    does not terminate early. Set *escapes* to False for e.g. Go raw strings
    (backtick literals) where backslash is literal, not an escape.

    Known limitation: backtick (`` ` ``) is always ``escapes=False`` (Go raw
    string compat).  TS template-literal escaped backtick (``\\````) will cause
    premature closing — the caller treats this as unterminated and falls back
    to the start line (line scanner) or ``len(content)`` (offset scanner).
    Extremely rare in real code; accepted trade-off.
    """
    j = start + 1
    while j < length:
        c = content[j]
        if escapes and c == "\\":
            j += 2
            continue
        if c == quote:
            return j
        j += 1
    return length


def _is_verbatim_string_start(content: str, i: int) -> bool:
    """Whether the ``"`` at *i* opens a C# verbatim string (``@"..."``).

    C# verbatim strings treat ``\\`` as a literal backslash and use ``""`` for an
    embedded quote — the opposite of the ``escapes=True`` rule that
    :func:`_skip_quoted_literal` applies to ordinary strings. Without this
    detection a verbatim string whose closing ``"`` is preceded by ``\\``
    (e.g. ``@"C:\\x\\"``) is mis-scanned: the ``\\"`` is read as an escape,
    the scan overshoots past the real close, and downstream braces are
    mis-counted (symptom: the fail-closed pre-write gate falsely rejects a
    valid edit).

    Handles all three C# verbatim prefixes — ``@"``, ``$@"``, ``@$"`` — since
    ``@"`` is essentially C#-unique (no other scanned language prefixes a
    string literal with ``@``), so this detection is safe across the shared
    scanner.
    """
    if i >= 1 and content[i - 1] == "@":
        return True
    # @$" form (interpolated verbatim): '$' immediately precedes '"', '@' before it
    if i >= 2 and content[i - 1] == "$" and content[i - 2] == "@":
        return True
    return False


def _skip_verbatim_string(content: str, start: int, length: int) -> int:
    """Skip a C# verbatim string whose opening ``"`` is at *start*.

    In a verbatim string ``\\`` is a literal backslash (NOT an escape) and a
    doubled ``""`` represents a single embedded ``"``. The scan therefore runs
    until a ``"`` that is NOT followed by another ``"``. Returns the index of
    that closing quote (or *length* if unterminated).
    """
    j = start + 1
    while j < length:
        if content[j] == '"':
            if j + 1 < length and content[j + 1] == '"':  # doubled → embedded quote
                j += 2
                continue
            return j  # lone closing quote
        j += 1
    return length


def _consume_char_or_lifetime(content: str, i: int, length: int) -> int:
    """Given a tick ``'`` at *i*, return the index past the construct it opens.

    Disambiguates a **char literal** (``'a'``, ``'\\n'``, ``'\\u{41}'``,
    ``'\\''``) from a Rust **lifetime** (``'a``, ``'static``, ``&'a self``).
    A lifetime has NO closing tick; a char literal always closes after exactly
    one element (a single character or one escape sequence). Validating that
    element grammar distinguishes the two WITHOUT a language parameter — the
    rule is uniform across C/C++/Java/Rust/Go runes.

    Returns:

    - char literal → index past the closing ``'``
    - lifetime / lone trailing tick → ``i + 1`` (the tick's identifier is left
      as ordinary source so braces like ``struct Parser<'a> {`` are counted)

    Known limitation: multi-char char literals (C ``'ab'``) lacking a closing
    tick on the element are treated as lifetimes. They cannot contain braces
    in practice (digits/letters only), so the net count is unaffected; only
    the pathological ``'{}'`` is mis-scanned (documented, like the backtick
    xfail).
    """
    j = i + 1
    if j >= length:
        return i + 1
    c = content[j]
    if c == "\\":  # escape sequence
        if j + 1 >= length:
            return i + 1
        e = content[j + 1]
        if e in "uU" and j + 2 < length and content[j + 2] == "{":
            # \u{XXXX} / \U{XXXXXXXX} — close brace then expect closing '
            close = content.find("}", j + 3)
            if close != -1 and close + 1 < length and content[close + 1] == "'":
                return close + 2
            return i + 1
        if e == "x":  # \xHH — 1-2 hex digits
            k = j + 2
            while k < length and content[k] in "0123456789abcdefABCDEF" and k - (j + 2) < 2:
                k += 1
            if k < length and content[k] == "'":
                return k + 1
            return i + 1
        # simple escape: \n \t \\ \' \" \0 \r \a \b \f \v — one char after backslash
        if j + 2 < length and content[j + 2] == "'":
            return j + 3
        return i + 1
    # single-char element
    if j + 1 < length and content[j + 1] == "'":
        return j + 2
    return i + 1  # no closing tick within one element → lifetime


def _iter_brace_tokens(content: str, offset: int = 0):
    """Yield ``(ch, idx)`` for every ``{``/``}`` OUTSIDE literals and comments.

    SSOT literal/comment/char-vs-lifetime scanner shared by both
    :func:`_find_closing_brace` (offset of matching close) and
    :func:`net_brace_count` (net depth tally). Centralising the skip logic
    here means the Rust-lifetime disambiguation and string/comment handling
    live in exactly one place — the two consumers cannot desync (the prior
    copy-pasted loops were the recurring bug source).

    Skips: ``//`` line comments, ``/* */`` block comments, ``"..."`` strings,
    ``'...'`` char literals / Rust lifetimes (see
    :func:`_consume_char_or_lifetime`), and backtick ``\\`...\\` `` template/
    raw strings (``escapes=False`` for Go raw-string compat). An unterminated
    comment/string ends generation.
    """
    i = offset
    length = len(content)
    while i < length:
        ch = content[i]
        # ── line comment // … \n ──────────────────────────────────────
        if ch == "/" and i + 1 < length and content[i + 1] == "/":
            nl = content.find("\n", i)
            if nl == -1:
                return
            i = nl
            continue
        # ── block comment /* … */ ────────────────────────────────────
        if ch == "/" and i + 1 < length and content[i + 1] == "*":
            end = content.find("*/", i + 2)
            if end == -1:
                return
            i = end + 2
            continue
        # ── string / template literals ────────────────────────────────
        if ch == '"':
            # C# verbatim @"..." (\ literal, "" embedded quote) needs its own
            # scan rule — otherwise \" before the real close is read as an
            # escape and the scan overshoots, mis-counting downstream braces.
            if _is_verbatim_string_start(content, i):
                i = _skip_verbatim_string(content, i, length) + 1
            else:
                i = _skip_quoted_literal(content, i, length, '"', escapes=True) + 1
            continue
        if ch == "'":
            # Char literal ('x','\n',...) vs Rust lifetime ('a,'static): a
            # lifetime has no closing tick, so consume only the tick and let
            # its identifier scan normally — otherwise the '{' in
            # `struct Parser<'a> {` is swallowed between two ticks.
            i = _consume_char_or_lifetime(content, i, length)
            continue
        if ch == "`":
            i = _skip_quoted_literal(content, i, length, "`", escapes=False) + 1
            continue
        # ── brace token ───────────────────────────────────────────────
        if ch == "{" or ch == "}":
            yield (ch, i)
        i += 1


def _find_closing_brace(content: str, offset: int) -> int:
    """Core brace scanner — returns offset of matching ``}`` or -1.

    Thin consumer of :func:`_iter_brace_tokens` (the SSOT literal/comment/
    char-vs-lifetime scanner). Tracks depth and returns the offset of the
    first ``}`` that returns depth to zero after an opening ``{``.

    Returns -1 when no matching brace is found (unterminated block), letting
    each twin (:func:`find_brace_block_end` /
    :func:`find_brace_block_end_offset`) apply its own conservative fallback.

    Known limitation: backtick (`` ` ``) is always ``escapes=False`` for Go
    raw-string compat — see :func:`_iter_brace_tokens`.
    """
    depth = 0
    started = False
    for ch, idx in _iter_brace_tokens(content, offset):
        if ch == "{":
            depth += 1
            started = True
        else:  # "}"
            depth -= 1
            if started and depth == 0:
                return idx
    return -1


def net_brace_count(content: str) -> int:
    """Literal/comment-aware net brace count (``{`` minus ``}``).

    Thin consumer of :func:`_iter_brace_tokens` — shares the EXACT same
    string/char/template-literal and ``//`` / ``/* */`` comment skipping as
    :func:`_find_closing_brace`, so the two can never desync. Returns 0 for a
    balanced region.

    This is the SSOT brace tally for the pre-write safety gate in
    ``symbol_modify_tool._post_edit_syntax_ok``: brace-delimited languages that
    lack an inline compiler (Kotlin/Rust/C/C++/Java/Scala/Swift/C#) are
    validated for balance before write, so a symbol-range scan that left an
    orphan ``}`` (or dropped a brace) is rejected instead of corrupting the
    file. It is a coarse gate — a real compiler is stronger — but it catches
    the known corruption class where braces are added/removed in net terms.
    """
    depth = 0
    for ch, _idx in _iter_brace_tokens(content):
        depth += 1 if ch == "{" else -1
    return depth


def find_brace_block_end(content: str, offset: int) -> int:
    """Heuristic: 1-based line of the matching ``}`` starting from *offset*.

    Derives from the core :func:`_find_closing_brace` scanner — every newline
    in the scanned region (including those inside skipped literals) is
    accurately counted via ``content[:end].count("\\n")``, fixing the
    multi-line literal under-count regression (bug #1).

    Shared by all brace-delimited C-family languages (C/C++/Go/Java/Kotlin/
    TypeScript/JavaScript).  Conservative fallback is the start line when
    no matching brace is found.
    """
    start_line = content[:offset].count("\n") + 1
    end = _find_closing_brace(content, offset)
    if end == -1:
        return start_line
    return content[:end].count("\n") + 1


def find_brace_block_end_offset(content: str, offset: int) -> int:
    """Offset-returning twin of :func:`find_brace_block_end`.

    Returns the *offset* (exclusive) of the matching ``}`` — i.e. one past the
    closing brace — so ``content[offset:find_brace_block_end_offset(...)]`` yields
    exactly the brace-delimited block. Derives from the same core
    :func:`_find_closing_brace` scanner as its line-based twin.

    Used by the regex-fallback *class-body* range computation in Java/Kotlin/
    TypeScript providers. Conservative fallback is ``len(content)``.
    """
    end = _find_closing_brace(content, offset)
    if end == -1:
        return len(content)
    return end + 1


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

        Returns a frozen-dataclass instance; cached tuples are reconstructed
        without deepcopy (5-50x faster on repeated validation of the same
        file across the agent edit-validate loop).
        """
        memo = self.__dict__.get("_syntax_memo")
        if memo is None:
            memo = OrderedDict()
            self.__dict__["_syntax_memo"] = memo
        key = (file_path, _sha256_hex(content))
        cached = memo.get(key)
        if cached is not None:
            memo.move_to_end(key)
            # cached is stored as (ok, errors_tuple, language) — reconstruct
            # without deepcopy (5-50x faster on repeated validation of the same
            # file across the agent edit-validate loop).
            _ok, _errors_t, _lang = cached
            return SyntaxValidationResult(
                ok=_ok, errors=list(_errors_t), language=_lang,
            )
        result = self._validate_syntax_impl(file_path, content)
        # Store as immutable tuple so cache hits reconstruct cheaply
        memo[key] = (result.ok, tuple(result.errors), result.language)
        while len(memo) > self._SYNTAX_MEMO_MAX:
            memo.popitem(last=False)
        return SyntaxValidationResult(
            ok=result.ok, errors=list(result.errors), language=result.language,
        )

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
