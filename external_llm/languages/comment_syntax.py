"""Typed comment-syntax policy (SSOT) for per-language comment skipping.

Provides :class:`CommentSyntax` — the line-comment tokens and block-comment
pairs for a language — and :func:`comment_syntax_for` to look it up by
:class:`~external_llm.languages.models.LanguageId`.

Why this exists
---------------
Consumers that must skip brackets/braces inside comments (notably the
``anchor_edit`` bracket-balance guard in :mod:`external_llm.agent._shared_utils`)
derive a ``CommentSyntax`` from the target file's ``LanguageId`` ONCE and pass
it to the scanner. This replaces the prior binary ``c_style_comments`` flag,
which classified EVERY non-Python language as C-style and thus mis-counted
brackets inside ``#`` comments for Ruby / Bash / PHP — genuine ``#``-comment
languages — a latent data-loss vector (a bracket in a ``#`` comment was
counted, falsely tripping the guard and triggering the F2 multi-line expansion
that ``del`` real code).

Centralising the classification here (a typed policy, not keyword/regex
matching) means adding a language can never silently introduce the same bug:
the new language's ``LanguageId`` simply maps to its ``CommentSyntax``.

Scope note
-----------
This SSOT governs the *bracket-delta* scanners only. The brace-only ``{}``
scanner (``_brace_match_depth`` / ``_iter_brace_tokens`` in ``languages/base``)
is intentionally C-style-scoped per the two-tier brace-scanner design — it is
used for TS/JS class-body brace matching where ``#`` is never a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from external_llm.languages.models import LanguageId


@dataclass(frozen=True)
class CommentSyntax:
    """Comment syntax for bracket-scanning purposes.

    Attributes:
        line_tokens: Prefixes that start a line comment (the rest of the line is
            ignored). E.g. ``("#",)`` (Python/Ruby/Bash), ``("//",)`` (C-family),
            or both ``("#", "//")`` (PHP). Empty for languages with no line
            comment.
        block_pairs: ``(open, close)`` token pairs for block comments, e.g.
            ``(("/*", "*/"),)`` (C-family) or ``(("--[[", "]]"),)`` (Lua). Empty
            for languages with no block comment. A scanner must check block
            opens BEFORE line tokens so a block open that shares a prefix with a
            line token (Lua ``--[[`` vs ``--``) wins.
    """

    line_tokens: tuple[str, ...]
    block_pairs: tuple[tuple[str, str], ...]


# ── Canonical comment-syntax per family ───────────────────────────────────
#
# These are the ONLY correct groupings — callers MUST go through
# ``comment_syntax_for`` rather than re-deriving, so every scanner sees the same
# classification (the whole point of centralising: no binary ``is not PYTHON``
# branch can ever re-introduce the Ruby/Bash/PHP mis-count).

_HASH_ONLY = CommentSyntax(line_tokens=("#",), block_pairs=())
_C_STYLE = CommentSyntax(line_tokens=("//",), block_pairs=(("/*", "*/"),))
# PHP accepts both '#' and '//' line comments AND '/* */' block comments.
_PHP = CommentSyntax(line_tokens=("#", "//"), block_pairs=(("/*", "*/"),))
# Lua uses '--' line comments and '--[[ ... ]]' long (block) comments. (Higher
# '=' levels like --[==[ are not handled — a best-effort pre-existing gap.)
_LUA = CommentSyntax(line_tokens=("--",), block_pairs=(("--[[", "]]"),))
# CSS has only block comments; '// ' is valid only in SCSS/Less, not plain CSS.
_CSS = CommentSyntax(line_tokens=(), block_pairs=(("/*", "*/"),))
# Data / markup with no comment syntax relevant to bracket scanning.
_NONE = CommentSyntax(line_tokens=(), block_pairs=())


_COMMENT_SYNTAX: dict[LanguageId, CommentSyntax] = {
    # hash-comment family
    LanguageId.PYTHON: _HASH_ONLY,
    LanguageId.RUBY: _HASH_ONLY,
    LanguageId.BASH: _HASH_ONLY,
    # dual line-comment family
    LanguageId.PHP: _PHP,
    # Lua
    LanguageId.LUA: _LUA,
    # C-style line + block family
    LanguageId.TYPESCRIPT: _C_STYLE,
    LanguageId.JAVASCRIPT: _C_STYLE,
    LanguageId.GO: _C_STYLE,
    LanguageId.JAVA: _C_STYLE,
    LanguageId.KOTLIN: _C_STYLE,
    LanguageId.RUST: _C_STYLE,
    LanguageId.C: _C_STYLE,
    LanguageId.CPP: _C_STYLE,
    LanguageId.CSHARP: _C_STYLE,
    LanguageId.SWIFT: _C_STYLE,
    LanguageId.SCALA: _C_STYLE,
    # block-only
    LanguageId.CSS: _CSS,
    # none
    LanguageId.JSON: _NONE,
    LanguageId.HTML: _NONE,
    LanguageId.UNKNOWN: _NONE,
}


@lru_cache(maxsize=None)
def comment_syntax_for(lang_id: LanguageId) -> CommentSyntax:
    """Return the :class:`CommentSyntax` for *lang_id*.

    Unknown languages resolve to a comment-free policy (no skipping), which is
    the safest default — a bracket inside a comment is counted rather than a
    bracket inside code being wrongly skipped.
    """
    return _COMMENT_SYNTAX.get(lang_id, _NONE)
