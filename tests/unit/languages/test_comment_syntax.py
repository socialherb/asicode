"""Tests for the typed comment-syntax policy SSOT.

``comment_syntax_for`` is the single source of truth for which comment tokens
the bracket-delta scanners must skip per language. Centralising it (a typed
policy, not a binary ``is not PYTHON`` flag) means every language maps to its
correct comment syntax — pinning the classification here guards against a
future language addition silently re-introducing the Ruby/Bash/PHP mis-count
(#1): a new ``LanguageId`` member simply has to be added to the policy table.
"""

from external_llm.languages.comment_syntax import CommentSyntax, comment_syntax_for
from external_llm.languages.models import LanguageId


class TestCommentSyntaxClassification:
    """Every LanguageId MUST resolve to a correct, explicit CommentSyntax."""

    def test_hash_comment_family(self):
        # Python / Ruby / Bash use '#' line comments, no block comments.
        for lid in (LanguageId.PYTHON, LanguageId.RUBY, LanguageId.BASH):
            cs = comment_syntax_for(lid)
            assert cs.line_tokens == ("#",), f"{lid}: expected ('#',) line token"
            assert cs.block_pairs == (), f"{lid}: expected no block comments"

    def test_php_dual_line_comments(self):
        # PHP accepts BOTH '#' and '//' line comments (plus '/* */' blocks).
        cs = comment_syntax_for(LanguageId.PHP)
        assert "#" in cs.line_tokens and "//" in cs.line_tokens
        assert cs.block_pairs == (("/*", "*/"),)

    def test_c_style_family(self):
        # JS/TS/Go/Java/Kotlin/Rust/C/C++/C#/Swift/Scala: '//' line + '/* */' block.
        c_style = {
            LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT, LanguageId.GO,
            LanguageId.JAVA, LanguageId.KOTLIN, LanguageId.RUST, LanguageId.C,
            LanguageId.CPP, LanguageId.CSHARP, LanguageId.SWIFT, LanguageId.SCALA,
        }
        for lid in c_style:
            cs = comment_syntax_for(lid)
            assert cs.line_tokens == ("//",), f"{lid}: expected ('//',) line token"
            assert cs.block_pairs == (("/*", "*/"),), f"{lid}: expected /* */ block"

    def test_lua(self):
        cs = comment_syntax_for(LanguageId.LUA)
        assert cs.line_tokens == ("--",)
        assert cs.block_pairs == (("--[[", "]]"),)

    def test_css_block_only(self):
        # Plain CSS has ONLY block comments ('//' is valid only in SCSS/Less).
        cs = comment_syntax_for(LanguageId.CSS)
        assert cs.line_tokens == ()
        assert cs.block_pairs == (("/*", "*/"),)

    def test_no_comment_languages(self):
        for lid in (LanguageId.JSON, LanguageId.HTML, LanguageId.UNKNOWN):
            cs = comment_syntax_for(lid)
            assert cs.line_tokens == () and cs.block_pairs == (), f"{lid}: expected no comments"

    def test_every_language_id_is_classified(self):
        """Completeness: the policy table MUST list every LanguageId member.

        A newly-added LanguageId that is forgotten here silently falls through
        to the UNKNOWN (skip-nothing) default — which for a '#'- or '//'-comment
        language re-introduces the #1 mis-count. Comparing the table's key set
        to ``set(LanguageId)`` catches that omission at test time.
        """
        from external_llm.languages.comment_syntax import _COMMENT_SYNTAX

        table_keys = set(_COMMENT_SYNTAX.keys())
        enum_members = set(LanguageId)
        missing = enum_members - table_keys
        assert not missing, (
            f"LanguageId members missing from _COMMENT_SYNTAX table: {missing} — "
            "add each to the policy with its correct comment syntax"
        )

    def test_python_does_not_list_slash_slash(self):
        """The floor-division hazard: Python MUST NOT treat '//' as a comment."""
        cs = comment_syntax_for(LanguageId.PYTHON)
        assert "//" not in cs.line_tokens


class TestCommentSyntaxIsFrozenAndCached:
    def test_returns_frozen_instance(self):
        cs = comment_syntax_for(LanguageId.JAVASCRIPT)
        assert isinstance(cs, CommentSyntax)

    def test_same_language_returns_identical_instance(self):
        # lru_cache: same LanguageId → same object (cheap repeated lookups).
        assert comment_syntax_for(LanguageId.PYTHON) is comment_syntax_for(LanguageId.PYTHON)

    def test_unknown_is_safe_default(self):
        # Unknown languages resolve to skip-nothing (safest: count brackets in
        # comments rather than wrongly skipping brackets in real code).
        cs = comment_syntax_for(LanguageId.UNKNOWN)
        assert cs.line_tokens == () and cs.block_pairs == ()
