"""Regression tests for the comment-aware bracket-delta guard (anchor_edit).

Background: ``_tool_anchor_edit`` (tool path) and ``_handle_anchor_edit``
(editor path) both run a single-line ``replace_line`` bracket-balance guard.
Historically its inline scanner only skipped Python ``#`` comments; later a
binary ``c_style_comments`` flag handled the C-family but mis-classified
Ruby / Bash / PHP (genuine ``#``-comment languages) as C-style, so brackets
inside their ``#`` comments were counted — falsely tripping the guard or, worse,
triggering a spurious multi-line expansion that ``del`` real code.

The fix centralises comment classification in a typed policy,
``languages.comment_syntax.CommentSyntax`` (looked up per-file via
``comment_syntax_for(lang_id)``), consumed by the SSOT scanners
``_net_bracket_delta`` (per-line) and ``_scan_line_brackets_delta`` (stateful,
multi-line). These tests pin the SSOT helper behaviour across all comment
families and the two call sites' language dispatch.
"""
import pytest

from external_llm.agent._shared_utils import _net_bracket_delta, _scan_line_brackets_delta
from external_llm.languages.comment_syntax import comment_syntax_for
from external_llm.languages.models import LanguageId

JS = comment_syntax_for(LanguageId.JAVASCRIPT)
PY = comment_syntax_for(LanguageId.PYTHON)


# ── SSOT helper: comment/string awareness ─────────────────────────────────

class TestNetBracketDeltaCommentAware:
    def test_js_line_comment_brackets_ignored(self):
        # Unbalanced '(' inside a // comment must not count (the original bug).
        assert _net_bracket_delta("return compute(x); // note (see below", JS) == 0

    def test_js_block_comment_brackets_ignored(self):
        assert _net_bracket_delta("foo(); /* open { never closed inline", JS) == 0

    def test_js_single_line_block_comment_balanced(self):
        assert _net_bracket_delta("a = /* { } */ b;", JS) == 0

    def test_real_js_open_brace_still_counts(self):
        assert _net_bracket_delta("function foo() {", JS) == 1

    def test_python_hash_comment_brackets_ignored(self):
        assert _net_bracket_delta("x = 1  # note (unbalanced", PY) == 0

    def test_python_string_with_hash_not_comment(self):
        assert _net_bracket_delta('s = "# not a comment {",', PY) == 0

    def test_real_python_open_paren_still_counts(self):
        assert _net_bracket_delta("def foo(", PY) == 1

    def test_mixed_bracket_families(self):
        # ( { [ ] } )  — all balanced
        assert _net_bracket_delta("func({a: [1, 2]})", JS) == 0
        # ( { [  with ] closed -> net +2
        assert _net_bracket_delta("func({a: [1, 2]", JS) == 2


class TestPythonFloorDivisionNotComment:
    """CRITICAL: in Python `//` is floor division, not a line comment.

    The Python policy must NOT list ``//`` as a comment token, otherwise
    ``q = a // b`` would have its second operand swallowed.
    """

    def test_floor_division_not_treated_as_comment(self):
        assert _net_bracket_delta("q = a // b", PY) == 0

    def test_floor_division_with_brackets_after(self):
        # The brackets after // are real code in Python (floor-div result).
        assert _net_bracket_delta("q = a // (b + 1)", PY) == 0

    def test_same_text_is_comment_under_c_style(self):
        # Same textual content, but under C-style rules // starts a comment.
        assert _net_bracket_delta("q = a // (b + 1)", JS) == 0


class TestHashCommentLanguages:
    """Regression for the #1 data-loss bug: Ruby / Bash / PHP use '#' line
    comments but the prior binary ``is not PYTHON`` flag classified them as
    C-style, so a bracket inside a '#' comment was counted → false bracket-
    delta mismatch → F2 multi-line expansion → ``del`` real code.
    """

    @pytest.mark.parametrize("lang", [LanguageId.RUBY, LanguageId.BASH, LanguageId.PHP])
    def test_hash_comment_brackets_not_counted(self, lang):
        cs = comment_syntax_for(lang)
        assert _net_bracket_delta("foo(a)  # note (", cs) == 0, (
            f"{lang}: '#' comment bracket must be ignored (the #1 bug)"
        )

    def test_ruby_real_bracket_outside_comment_counts(self):
        cs = comment_syntax_for(LanguageId.RUBY)
        assert _net_bracket_delta("foo(a, b)", cs) == 0  # balanced real brackets

    def test_php_accepts_both_hash_and_slash_comments(self):
        cs = comment_syntax_for(LanguageId.PHP)
        assert _net_bracket_delta("$x = f(a);  # note (", cs) == 0
        assert _net_bracket_delta("$x = f(a);  // note (", cs) == 0
        # PHP block comment too
        assert _net_bracket_delta("$x = /* ( */ f(a);", cs) == 0

    def test_php_division_not_swallowed(self):
        # '/' as division must NOT be skipped (only '//' and '/*').
        cs = comment_syntax_for(LanguageId.PHP)
        assert _net_bracket_delta("$q = $a / ($b + 1)", cs) == 0

    @pytest.mark.parametrize("ext", [".zsh", ".ksh"])
    def test_zsh_ksh_resolve_to_bash_comment_syntax(self, ext):
        """Gap #2: .zsh/.ksh previously fell to UNKNOWN (no comment skipping),
        so a '#' comment bracket was over-counted — the SAME F2 multi-line-
        expansion data-loss class this family prevents for Ruby/Bash/PHP.  These
        shells share the bash grammar, so they must resolve to BASH end-to-end.
        """
        lid = LanguageId.from_path("script" + ext)
        assert lid is LanguageId.BASH, f"{ext}: must map to BASH, got {lid}"
        cs = comment_syntax_for(lid)
        # end-to-end: '#' comment bracket must be ignored (was delta +1)
        assert _net_bracket_delta("foo(a)  # note (", cs) == 0, ext


class TestLuaComments:
    """Lua uses '--' line comments and '--[[ ]]' long (block) comments."""

    def test_lua_line_comment_brackets_not_counted(self):
        cs = comment_syntax_for(LanguageId.LUA)
        assert _net_bracket_delta("foo(a) -- note (", cs) == 0

    def test_lua_long_comment_brackets_not_counted(self):
        cs = comment_syntax_for(LanguageId.LUA)
        assert _net_bracket_delta("x = --[[ ( ]] y", cs) == 0

    def test_lua_block_carries_close_token(self):
        # The stateful scanner must carry the Lua close token ']]' (not '*/').
        d, _s, _t, block_close = _scan_line_brackets_delta(
            "foo( --[[ open (", None, False, None, comment_syntax_for(LanguageId.LUA)
        )
        assert d == 1 and block_close == "]]", "Lua block comment must carry ']]' close token"


class TestTripleQuoteAndEscape:
    def test_triple_quote_brackets_ignored(self):
        assert _net_bracket_delta('x = """ ( { """ + y', PY) == 0

    def test_single_quoted_string_brackets_ignored(self):
        # Brackets inside any quote family are skipped.
        assert _net_bracket_delta("x = '( [ {' + y", PY) == 0
        assert _net_bracket_delta('x = "( [ {" + y', JS) == 0

    def test_real_bracket_outside_string_counts(self):
        assert _net_bracket_delta('x = "(" + (', PY) == 1  # trailing ( is real code


class TestEscapeHandling:
    """Regression: the escape check was a naive ``prev != '\\'`` look-back,
    which mis-counts an ESCAPED backslash run before the closing quote.

    In ``"C:\\"`` (a Windows path — two backslashes, the first escaping the
    second) the trailing ``"`` is a REAL closer, but the look-back saw a
    backslash and treated it as escaped, leaving the literal "open" and
    swallowing ``foo(``  →  delta 0 instead of 1. With an even-length
    backslash run the quote is real; only an ODD run escapes it.

    These inputs use raw strings so the backslash count is unambiguous.
    """

    def test_windows_path_escaped_backslash(self):
        # "C:\\"  (2 backslashes = escaped backslash) → literal closes, '(' counts.
        assert _net_bracket_delta(r'path = "C:\\"; foo(', PY) == 1

    def test_regex_with_escaped_backslashes(self):
        # "\\\\"  (4 backslashes = 2 escaped backslashes) → closes, '(' counts.
        assert _net_bracket_delta(r're = "\\\\"; bar(', PY) == 1

    def test_six_backslashes_then_close(self):
        # 6 backslashes = 3 escaped backslashes → even run, quote is real.
        assert _net_bracket_delta(r's = "\\\\\\"; baz(', PY) == 1

    def test_escaped_quote_odd_run(self):
        # "a\"" (1 backslash escapes the quote) → literal still closes on the
        # FINAL quote, '(' counts. (Pin: the single-backslash escape case the
        # old look-back happened to get right — must not regress.)
        assert _net_bracket_delta(r'x = "a\""; qux(', PY) == 1

    def test_unterminated_when_quote_actually_escaped(self):
        # "C:\"  (1 backslash escapes the quote) → literal is UNTERMINATED,
        # so the '(' after is inside the string and must NOT count (delta 0).
        # This is the symmetric counterpart: odd run → quote is escaped.
        assert _net_bracket_delta(r'bad = "C:\"; foo(', PY) == 0

    def test_js_single_quote_escape(self):
        assert _net_bracket_delta(r"s = 'it\'s'; fn(", JS) == 1

    def test_js_template_literal_escape(self):
        # Backtick template: \${ is an escaped interpolation, literal stays open
        # only until the real backtick close, then '(' counts.
        assert _net_bracket_delta(r'const s = `\${a}`; fn(', JS) == 1

    def test_string_then_comment_bracket_still_ignored(self):
        # Escape fix must not regress comment awareness.
        assert _net_bracket_delta(r's = "ok"  # ( note', PY) == 0


# ── Call-site language dispatch (source contract) ─────────────────────────
# Verifies both consumers derive a CommentSyntax policy via comment_syntax_for
# (not a binary ``is not PYTHON`` flag) without importing the heavy modules
# (avoids import-time side effects). Mirrors the source-contract test pattern
# used elsewhere in the suite.

import re

WT = "external_llm/agent/tool_handlers/write_tools.py"
# P2-2: the anchor_edit tool path now lives in the patch mixin module.
WT_PATCH = "external_llm/agent/tool_handlers/write_tools_patch_mixin.py"


def _read(p):
    with open(p) as f:
        return f.read()


class TestCallSiteLanguageDispatch:
    def test_tool_path_uses_typed_comment_policy(self):
        src = _read(WT_PATCH)
        # helper is imported
        assert "_net_bracket_delta" in src, "tool path must import _net_bracket_delta"
        # the binary flag must be GONE — replaced by a typed CommentSyntax policy
        assert re.search(r"_c_style_brackets", src) is None, (
            "tool path must NOT use the binary _c_style_brackets flag"
        )
        assert re.search(r"comment_syntax_for\s*\(\s*lang_id\s*\)", src), (
            "tool path must derive a CommentSyntax via comment_syntax_for(lang_id)"
        )
        # the old inline nested def is gone
        assert "def _bracket_delta(" not in src, "inline _bracket_delta def must be removed"

    def test_expansion_scans_delegate_to_stateful_helper(self):
        # Both expansion scans must delegate to _scan_line_brackets_delta (the
        # stateful SSOT in _shared_utils), NOT keep a hand-rolled inline
        # ``while _j < len(_sl)`` scanner.
        # The editor-side call site (operation_handlers) went with the PLANNER
        # lane; the tool path is the only remaining consumer.
        for p in (WT_PATCH,):
            src = _read(p)
            assert "_scan_line_brackets_delta(" in src, (
                f"{p}: expansion scan must call _scan_line_brackets_delta"
            )
            # the inline bracket-counting scanner is gone from the call site
            assert re.search(r"while _j < len\(_sl\)", src) is None, (
                f"{p}: inline expansion scanner must be removed (now in helper)"
            )
            assert re.search(r"_scan_balance \+= 1", src) is None, (
                f"{p}: inline bracket tally must be removed (now in helper)"
            )


# ── SSOT helper: stateful multiline bracket scan ─────────────────────────
# _scan_line_brackets_delta threads string/block-comment state across lines
# so the F2 expansion scan never mis-counts a bracket inside a multi-line
# construct (triple-quoted string, /* */ block comment) or a line comment.

class TestScanLineBracketsDelta:
    def _run(self, lines, cs, start=+1, target=0):
        """Mirror the call-site loop: accumulate delta, carry state, return
        the 1-based index (within ``lines``) where target balance is hit."""
        bal = start
        in_str, in_triple, block_close = None, False, None
        for i, ln in enumerate(lines, start=1):
            ld, in_str, in_triple, block_close = _scan_line_brackets_delta(
                ln, in_str, in_triple, block_close, cs
            )
            bal += ld
            if bal == target:
                return i
        return None

    def test_python_comment_bracket_not_counted(self):
        # Regression: a ')' inside a '#' comment used to terminate the scan
        # early (close_line mis-identified), then ``del`` deleted the real
        # function arguments. Now the comment is skipped.
        lines = ["    x,", "    y,  # returns tuple) here", ")"]
        # start +1 (anchor 'foo('), target 0 → must reach the real ')'
        assert self._run(lines, PY) == 3

    def test_python_floor_division_brackets_counted(self):
        # Python '// ' is NOT a comment — brackets around it ARE counted.
        # 'a // (b)' → net 0 on that line; start +1, scan finds nothing → None
        lines = ["    a // (b)", "    c,"]
        assert self._run(lines, PY, start=+1, target=0) is None

    def test_cfamily_line_comment_bracket_not_counted(self):
        # JS/TS/Go: '//' comment — a ')' inside it must not close.
        lines = ["  x,", "  y,  // note )", ")"]
        assert self._run(lines, JS) == 3

    def test_cfamily_block_comment_carries_across_lines(self):
        # /* ... ( ... */ spanning lines: the '(' inside the comment is skipped,
        # state carries, and only the real ')' closes.
        lines = ["  /* open ( paren", "     still in comment ) */", "  );"]
        assert self._run(lines, JS) == 3

    def test_cfamily_block_comment_single_line(self):
        # '/* ( */' on one line: the '(' is inside the comment, net 0 for line.
        # start +1, line net 0 → balance stays +1, never hits 0 → None.
        lines = ["  /* ( */", "  other;"]
        assert self._run(lines, JS, start=+1, target=0) is None

    def test_ruby_hash_comment_bracket_not_counted(self):
        # #1 regression: Ruby '#' comment ')' must not close the scan.
        lines = ["  x,", "  y,  # note )", ")"]
        assert self._run(lines, comment_syntax_for(LanguageId.RUBY)) == 3

    def test_php_hash_comment_bracket_not_counted(self):
        lines = ["  $a,", "  $b,  # note )", ")"]
        assert self._run(lines, comment_syntax_for(LanguageId.PHP)) == 3

    def test_lua_long_comment_carries_across_lines(self):
        # Lua --[[ ... ]] spanning lines: carry ']]' close token.
        lines = ["  --[[ open ( paren", "     still in comment ) ]]", "  )"]
        assert self._run(lines, comment_syntax_for(LanguageId.LUA)) == 3

    def test_triple_quoted_string_carries_across_lines(self):
        # Python triple-quoted string spanning lines: braces inside are skipped.
        # anchor 'd = {' (+1, '{'), then a docstring with '{' '}', then '}'.
        lines = [
            '  """doc with { and } inside"""',  # opens+closes triple on same line, net 0
            "}",
        ]
        assert self._run(lines, PY, start=+1, target=0) == 2

    def test_string_with_braces_not_counted(self):
        # JS: "a { b }" string — braces cancel, net 0 for the line.
        lines = ['  return "{ not real }";', "}"]
        assert self._run(lines, JS, start=+1, target=0) == 2

    def test_no_false_close_then_real_close(self):
        # Mixed: comment-with-paren line must be skipped, real close found.
        lines = ["  a,", "  b,  # )", "  c,", ")"]
        assert self._run(lines, PY) == 4

    def test_escape_handling_single_line(self):
        # Regression: an escaped backslash before the closing quote must not
        # leave the literal "open" — the '(' after the closed string counts.
        d, in_str, _t, _bc = _scan_line_brackets_delta(
            r'x = "C:\\"; foo(', None, False, None, PY
        )
        assert d == 1 and in_str is None, "escaped-backslash literal must close"

    def test_escape_not_carried_across_line_in_triple(self):
        # A trailing backslash inside a triple-quoted string is a LINE
        # CONTINUATION (it escapes the newline, not the first char of the next
        # line), so the next line must start UNESCAPED. The closing triple on
        # line 2 therefore closes the literal and the real '(' counts.
        l1 = 'x = """abc' + chr(92)            # trailing backslash (continuation)
        l2 = '""" + ('                         # closing triple + real code
        d1, s1, t1, bc1 = _scan_line_brackets_delta(l1, None, False, None, PY)
        d2, s2, _t2, _bc2 = _scan_line_brackets_delta(l2, s1, t1, bc1, PY)
        assert (d1, s1, t1) == (0, '"', True), "line 1 opens triple, swallows nothing"
        assert (d2, s2) == (1, None), "line 2 closes triple, '(' counts (escape not carried)"

    def test_returns_carried_state(self):
        # Directly verify the returned state for a C-family block comment carry.
        d, _s, _t, block_close = _scan_line_brackets_delta("/* open (", None, False, None, JS)
        assert d == 0 and block_close == "*/", "entered block comment, '(' not counted"
        d2, _s2, _t2, block_close2 = _scan_line_brackets_delta("close */ + foo(", None, False, "*/", JS)
        assert d2 == 1 and block_close2 is None, "closed block comment, then '(' counted"


# ── Anchor initial-state seed (prior-line context) ───────────────────────
# Regression: a replace_line whose ANCHOR sits inside a multi-line block
# comment or triple-quoted string. The per-line tally is stateless, so without
# seeding it with the prior-line state (_scan_to_line_state) it counted the
# anchor's brackets as REAL code, falsely tripping the F2 expansion and
# ``del``-ing the real code after the comment. Seeding makes the anchor tally
# aware that it is already inside a literal/comment, so its brackets are
# ignored and the expansion does not trigger.

from external_llm.agent._shared_utils import _scan_to_line_state


class TestAnchorInitialStateSeed:
    def test_anchor_inside_block_comment_seeded_to_zero(self):
        # Anchor line 2 is inside a /* */ block; its ')' must be treated as
        # comment content (delta 0), not real code (stateless gave -1).
        lines = ["/* multi-line comment", "   with ( a paren", "   and ) another", "*/"]
        anchor = 2
        stateless = _net_bracket_delta(lines[anchor], JS)
        s0, t0, bc0 = _scan_to_line_state(lines, anchor, JS)
        seeded = _net_bracket_delta(lines[anchor], JS, in_str=s0, in_triple=t0, block_close=bc0)
        assert stateless == -1, "stateless tally sees the ')' as real (the bug)"
        assert bc0 == "*/", "seed detected we are inside an open block comment"
        assert seeded == 0, "seeded tally correctly ignores the in-comment ')'"

    def test_anchor_inside_block_comment_prevents_false_expansion(self):
        # End-to-end mirror of the call-site logic: seeded old==new deltas => no
        # F2 expansion => the 'function foo() {' line after the comment is NOT
        # deleted (the confirmed data-loss vector).
        lines = ["/* comment ( opens", "   still comment", "*/", "function foo() {"]
        anchor = 1  # inside the block comment
        s0, t0, bc0 = _scan_to_line_state(lines, anchor, JS)
        old_d = _net_bracket_delta(lines[anchor], JS, in_str=s0, in_triple=t0, block_close=bc0)
        new_d = _net_bracket_delta("   changed comment line\n", JS, in_str=s0, in_triple=t0, block_close=bc0)
        assert old_d == new_d == 0, "both deltas zero => no spurious F2 expansion"

    def test_anchor_inside_triple_quoted_string_seeded(self):
        # Python triple-quoted docstring spanning lines; anchor inside it.
        lines = ['def f():', '    """doc with ( paren', '    and ) close', '    """', '    return (']
        anchor = 2  # inside the docstring
        s0, t0, bc0 = _scan_to_line_state(lines, anchor, PY)
        assert s0 == '"' and t0 is True, "seed detected we are inside a triple-quoted string"
        seeded = _net_bracket_delta(lines[anchor], PY, in_str=s0, in_triple=t0, block_close=bc0)
        assert seeded == 0, "in-docstring ')' is string content, not real code"
        # The stateless tally would have counted it:
        assert _net_bracket_delta(lines[anchor], PY) == -1

    def test_anchor_in_normal_code_seed_is_empty(self):
        # Common case: anchor in normal code => seed is the empty state and the
        # seeded tally equals the stateless tally (no behavior change, just the
        # O(anchor) pre-scan cost).
        lines = ["def foo():", "    bar()", "    baz()"]
        s0, t0, bc0 = _scan_to_line_state(lines, 2, PY)
        assert (s0, t0, bc0) == (None, False, None), "normal code => empty seed"
        line = "    if (a or b):"
        assert _net_bracket_delta(line, PY) == _net_bracket_delta(
            line, PY, in_str=s0, in_triple=t0, block_close=bc0
        )

    def test_seed_bounded_by_available_lines(self):
        # end_lineno beyond len(lines) must not raise.
        s0, t0, bc0 = _scan_to_line_state(["x = 1"], 99, PY)
        assert (s0, t0, bc0) == (None, False, None)

    # ── Rust lifetime residual-risk closure ─────────────────────────────────
    # A single ' opens a char/string literal for every language, so a Rust
    # lifetime ('a, 'static) — a ' with NO closer — leaves the scanner inside an
    # unterminated "string" at end-of-line. When an ODD number of lifetimes sits
    # ABOVE the anchor (with no intervening '), that poisoned state reaches the
    # anchor and mis-directs the F2 forward scan to `del` a victim line below it
    # (confirmed data-loss vector). _scan_to_line_state now falls back to the
    # empty seed for any non-triple ' / " open at a line boundary: no supported
    # language has a legit multi-line single/double-quote literal (multi-line
    # literals use triple-quotes, tracked via in_triple, or backticks).

    def test_rust_odd_lifetime_seed_falls_back_to_empty(self):
        # fn foo<'a>( has exactly ONE ' -> without the fallback the scanner would
        # be inside an unterminated 'string' entering the anchor.
        RUST = comment_syntax_for(LanguageId.RUST)
        lines = ["fn foo<'a>(", "    x: i32,", ") {", "    let z = 1;"]
        s0, t0, bc0 = _scan_to_line_state(lines, 3, RUST)
        assert (s0, t0, bc0) == (None, False, None), (
            "odd-count Rust lifetime must not poison the seed; a non-triple ' "
            "open at a line boundary is a lifetime/syntax-error, not a literal"
        )

    def test_rust_even_lifetime_seed_is_clean_without_fallback(self):
        # Even lifetime count self-closes (the second ' ends the literal); this
        # documents the non-poisoned baseline independently of the fallback.
        RUST = comment_syntax_for(LanguageId.RUST)
        lines = ["fn foo<'a>(x: &'a str) {", "    let z = 1;"]
        s0, t0, bc0 = _scan_to_line_state(lines, 1, RUST)
        assert (s0, t0, bc0) == (None, False, None)

    def test_rust_lifetime_seed_fallback_prevents_data_loss(self):
        # End-to-end mirror of the confirmed data-loss vector: an odd-count
        # lifetime above the anchor + an anchor line containing a ' then '(' +
        # a later line with ' then the balancing ') previously made the F2
        # forward scan `del` the victim line `real_code();`. With the fallback
        # the seed is empty, old==new deltas, and no expansion triggers.
        RUST = comment_syntax_for(LanguageId.RUST)
        lines = [
            "fn foo<'a>(",
            "    x: i32,",
            ") {",
            "    m'a(",          # anchor: ' closes poison, then ( -> +1
            "    real_code();",  # victim
            "    n');",          # ' closes, ) -> -1 (false close)
            "}",
        ]
        anchor = 3
        s0, t0, bc0 = _scan_to_line_state(lines, anchor, RUST)
        assert s0 is None, "fallback cleared the poisoned seed"
        old_d = _net_bracket_delta(lines[anchor], RUST, in_str=s0, in_triple=t0, block_close=bc0)
        new_d = _net_bracket_delta("    replaced\n", RUST, in_str=s0, in_triple=t0, block_close=bc0)
        # Without the fallback old_d would be +1 != new_d 0 -> F2 expansion ->
        # del real_code(). With it, both are scanned from the empty seed and the
        # guard does not mis-fire.
        assert old_d == new_d, f"guard must not mis-fire under cleared seed (old={old_d} new={new_d})"

    def test_non_triple_double_quote_open_also_resets(self):
        # Symmetry: an open " (non-triple) at a line boundary is likewise not a
        # legit multi-line literal in any supported language — it too must reset
        # (e.g. an unterminated string from a genuine syntax error must not
        # poison the seed either).
        lines = ['x = "unterminated', "    bar("]
        s0, t0, bc0 = _scan_to_line_state(lines, 1, PY)
        assert (s0, t0, bc0) == (None, False, None), "non-triple \" open must reset"

    def test_triple_quote_seed_NOT_reset_by_fallback(self):
        # Regression guard: the fallback targets ONLY non-triple ' / ". A legit
        # triple-quoted multi-line string (in_triple=True) must still seed.
        lines = ['x = """multi', "    bar("]
        s0, t0, _bc0 = _scan_to_line_state(lines, 1, PY)
        assert s0 == '"' and t0 is True, "triple-quote seed is a legit multi-line literal, kept"

    def test_backtick_seed_NOT_reset_by_fallback(self):
        # Regression guard: JS/TS template literals (backtick) legitimately span
        # lines and must still seed (backtick is not in the ' / " fallback set).
        JS_TB = comment_syntax_for(LanguageId.JAVASCRIPT)
        lines = ["const x = `multi-line", "    bar("]
        s0, _t0, _bc0 = _scan_to_line_state(lines, 1, JS_TB)
        assert s0 == "`", "backtick template-literal seed is legit multi-line, kept"

    def test_block_comment_seed_NOT_reset_by_fallback(self):
        # Regression guard: a multi-line block comment is a legit construct and
        # must still seed its close token (the fallback only touches strings).
        lines = ["/* multi-line comment ( opens", "    bar("]
        _s0, _t0, bc0 = _scan_to_line_state(lines, 1, JS)
        assert bc0 == "*/", "block-comment seed is legit multi-line, kept"
