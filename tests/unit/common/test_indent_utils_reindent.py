"""Regression tests for indent_utils.reindent_to_match.

Covers the bug where a new (content-unmatched) line in the ``after`` block that
the LLM emitted flush-left stayed at column 0 while its re-indented siblings
moved to the match site's indent — producing invalid, "unexpected indent" code.
``reindent_to_match`` is live via plan_compiler's edit_blocks fuzzy-match path.
"""
import ast
import itertools

from external_llm.common.indent_utils import (
    _continuation_rows,
    _first_logical_indent,
    _indent_of,
    detect_indent_char,
    indent_unit,
    normalize_indent_char_to_file,
    reindent_block,
    reindent_to_anchor,
    reindent_to_match,
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def test_flush_left_after_reindents_new_lines_to_match_site():
    """LLM emits 'after' flush-left; match site is a method body at 8 spaces.

    A brand-new multi-line call (not present in 'before') must shift up with the
    rest instead of being left at column 0.
    """
    actual_before = (
        "        if not args:\n"
        "            return None\n"
        "        x = compute()"
    )
    after = (
        "if not args:\n"
        "    return None\n"
        "y = transform(a,\n"
        "              b)\n"
        "x = compute()"
    )
    fixed = reindent_to_match(after, actual_before)
    lines = fixed.split("\n")

    # All top-level logical lines land at the match site's indent (8), not 0.
    assert _indent(lines[0]) == 8, lines           # if not args:
    assert _indent(lines[2]) == 8, lines           # y = transform(a,  (NEW line)
    assert _indent(lines[4]) == 8, lines           # x = compute()

    # The whole block parses as valid Python inside a real method.
    full = "class C:\n    def m(self, args, a, b):\n" + fixed
    ast.parse(full)

    # The bracket-aligned continuation tracks the open paren of its owner.
    owner = lines[2]
    cont = lines[3]
    assert _indent(cont) == owner.index("(") + 1, (owner, cont)


def test_unit_conversion_preserved():
    """4-space 'after' into a 2-space file still halves nesting (ratio path)."""
    after = "    function foo() {\n        return 2;\n    }"
    before = "  function foo() {\n    return 1;\n}"
    fixed = reindent_to_match(after, before).split("\n")
    assert _indent(fixed[0]) == 2   # function -> 2sp (matched line)
    assert _indent(fixed[1]) == 4   # return 2 -> 4sp (new line, ratio 0.5 of 8)
    assert _indent(fixed[2]) == 0   # closing brace -> 0 (matched)


def test_bracket_continuation_does_not_pollute_indent_unit():
    """A bracket-aligned continuation line must not collapse the GCD unit to 1.

    ``result = foo(a,\\n             b)`` has a 13-col continuation line.  Counting
    it would make indent_unit return 1 (GCD(2,13)), inflating ratios downstream.
    """
    assert indent_unit("result = foo(a,\n             b)", " ") == 4
    assert _continuation_rows("x = compute(a,\n   b)\ny = 2") == {2}


def test_continuation_detection_is_language_agnostic():
    """Only ()/[] open a continuation — a JS ``{`` block is real nesting."""
    js = "function foo() {\n  if (x) {\n    bar();\n  }\n}"
    assert _continuation_rows(js) == set()
    assert indent_unit(js, " ") == 2


def test_no_indent_explosion_on_continuation_in_scaled_path():
    """Continuation lines align under their open paren instead of exploding.

    Regression: a 2-space 'after' block (whose unit was mis-detected as 1 by a
    bracket-continuation line) re-indented into a 4-space file ratio-scaled the
    continuation column to ~80 chars and over-indented real code lines.
    """
    after = (
        "  if cond:\n"
        "    result = compute(a,\n"
        "                     b)\n"
        "    return result\n"
    )
    before = "    if cond:\n        old = 1\n"
    lines = reindent_to_match(after, before).split("\n")
    assert _indent(lines[0]) == 4              # if cond: (matched)
    assert _indent(lines[1]) == 8              # result = compute(  (one level deeper)
    assert _indent(lines[3]) == 8              # return result
    # Continuation 'b)' tracks the open paren, not a ratio-scaled column.
    assert _indent(lines[2]) == lines[1].index("(") + 1


def test_indent_unit_accepts_precomputed_cont_rows():
    """Passing _continuation_rows is equivalent to computing it internally."""
    text = "x = foo(a,\n        b)\ny = 2"
    assert indent_unit(text, " ", _continuation_rows(text)) == indent_unit(text, " ")


def test_normalize_indent_char_preserves_continuation_alignment():
    """A space-aligned continuation in a tab file stays as-is, not re-tabbed.

    Regression: the conversion re-quantized every wrong-char line via
    ``round(len(lead)/unit) * file_unit``; for a bracket-continuation that
    collapsed its alignment column into N tabs and broke the layout.
    """
    old = "def f():\n\tx = 1\n\treturn x\n"
    new = "def f():\n    y = compute(a,\n               b)\n    return y\n"
    out = normalize_indent_char_to_file(new, old).split("\n")
    assert out[1] == "\ty = compute(a,"   # real nesting line → 1 tab
    assert out[2] == "               b)"  # continuation alignment → untouched
    assert out[3] == "\treturn y"


def test_normalize_indent_char_converts_plain_nesting():
    """Sanity: ordinary nested space lines still convert to the file's tabs."""
    old = "def f():\n\tx = 1\n"
    new = "def f():\n    y = 2\n        z = 3\n"
    out = normalize_indent_char_to_file(new, old).split("\n")
    assert out[1] == "\ty = 2"      # depth 1 → 1 tab
    assert out[2] == "\t\tz = 3"    # depth 2 → 2 tabs


def test_first_logical_indent_skips_bracket_continuation():
    """The helper picks the logical line's indent, not a continuation's."""
    src = "    result = some_func(\n        arg1=v1,\n    )"
    assert _first_logical_indent(src) == "    "
    assert _first_logical_indent("") is None


def test_ambiguous_content_does_not_flatten_control_flow():
    """Same stripped line at two indents must not be content-mapped.

    Regression: the content→indent map keyed on stripped text with "last
    occurrence wins". When the match site (``actual_before``) contained the same
    line at two depths — e.g. ``return None`` inside an ``if`` (depth 8) and at
    the block base (depth 4) — the in-block copy in ``after`` was mapped to the
    *last* indent (4), escaping the ``if`` body. The result still parsed, so the
    syntax gate never caught it (silent control-flow corruption).

    The fix excludes any stripped text seen at more than one indent from the
    content map, so those lines fall through to the depth-remap path which
    preserves ``after``'s relative nesting.
    """
    actual_before = (
        "    if x:\n"
        "        return None\n"      # depth 8 — inside the if
        "    return None"            # depth 4 — block base
    )
    after = (
        "if x:\n"
        "    do_something()\n"
        "    return None\n"          # intended: depth 8 (inside the if)
        "return None"                # intended: depth 4 (block base)
    )
    fixed = reindent_to_match(after, actual_before)
    lines = fixed.split("\n")

    # The in-block `return None` stays INSIDE the if (depth 8), and the base
    # `return None` sits at the block base (depth 4) — not both collapsed to 4.
    assert _indent(lines[0]) == 4, lines            # if x:        -> base (4)
    assert _indent(lines[1]) == 8, lines            # do_something -> one level in (8)
    assert _indent(lines[2]) == 8, lines            # return None  -> STILL in the if (8)
    assert _indent(lines[3]) == 4, lines            # return None  -> block base (4)

    # And the result is valid Python whose meaning matches the intent.
    tree = ast.parse("def f():\n" + fixed)
    if_node = tree.body[0].body[0]
    assert isinstance(if_node, ast.If)
    # if body has two statements (do_something + return), then a base return.
    assert len(if_node.body) == 2
    assert isinstance(if_node.body[1], ast.Return)
    assert isinstance(tree.body[0].body[1], ast.Return)


def test_unambiguous_content_still_uses_fast_path():
    """A stripped line seen at a single indent keeps the exact-match fast path.

    Guards against over-broadly disabling the content map: the common case (each
    line appears once in the match site) must still map to the exact file indent.
    """
    actual_before = "    a = 1\n    b = 2"
    after = "a = 1\nb = 2"
    fixed = reindent_to_match(after, actual_before)
    assert fixed == "    a = 1\n    b = 2"


def test_cross_unit_continuation_no_explosion():
    """Tab-indented file + space-indented LLM snippet with bracket continuation.

    Regression: ``last_delta = len(new_lead) - leading`` mixed units (tabs vs
    spaces), causing the continuation line to explode from 4 spaces of alignment
    to 6 tabs.  The content-map path now runs BEFORE the continuation guard so
    the exact mixed tab+space indent from the file is used verbatim.
    """
    # File uses tabs for nesting + spaces for bracket alignment.
    actual_before = "\t\tfoo(a,\n\t\t    b)\n\t\tbar()"
    # LLM emits everything with spaces.
    after = "    foo(a,\n        b)\n    bar()"

    fixed = reindent_to_match(after, actual_before)
    lines = fixed.split("\n")

    assert lines[0] == "\t\tfoo(a,", f"line 0: {lines[0]!r}"
    assert lines[1] == "\t\t    b)", f"line 1: {lines[1]!r}"
    assert lines[2] == "\t\tbar()", f"line 2: {lines[2]!r}"

    # Must parse as valid Python.
    import ast
    ast.parse("def f():\n" + fixed)


def test_flush_left_into_tab_file_no_explosion():
    """LLM emits a block flush-left (col-0 base) into a TAB-indented match site.

    Regression (``fix/reindent-indent-explosion``): when ``after``'s first
    logical line sits at column 0, ``after_base`` is empty, so the old additive
    path shifted via ``leading + len(actual_base)`` — adding 4 *spaces* of nesting
    depth as 4 *tabs*.  A 2-level body exploded to 5 and 9 tabs.  The flush-left
    case now routes through the unit-scaled path so a 4-space level maps to one
    tab level regardless of char width.
    """
    after = "def f():\n    if x:\n        return 1"   # 4-space levels, col-0 base
    before = "\tif q:\n\t\told = 1"                    # tab-indented, base = 1 tab
    fixed = reindent_to_match(after, before).split("\n")
    assert fixed[0] == "\tdef f():"      # base level → 1 tab (not exploded)
    assert fixed[1] == "\t\tif x:"       # +1 level → 2 tabs (was 5)
    assert fixed[2] == "\t\t\treturn 1"  # +2 levels → 3 tabs (was 9)
    ast.parse("class C:\n" + "\n".join(fixed))


# ── Property / invariant tests ──────────────────────────────────────────────
# ``reindent_to_match`` is a pure ``(after, before) -> str``.  Case-based tests
# pin individual fixes, but every past indent fix spawned the next edge case
# (see git log: consolidate engines → reindent content-map → indent-explosion).
# These invariants kill the *class* of regressions across a generated matrix of
# (indent char × unit width × nesting depth × match-site base) rather than one
# example at a time.  Generated with stdlib only — no hypothesis dependency.

def _max_indent_levels(text: str) -> float:
    """Deepest non-continuation indent, normalised to nesting *levels*.

    Levels (not chars) so tab files (1 char/level) and 4-space files compare on
    one scale.  Continuation rows carry alignment, not depth, so are excluded.
    """
    lines = text.split("\n")
    if not any(_item_.strip() for _item_ in lines):
        return 0.0
    unit = indent_unit(text, detect_indent_char(lines)) or 1
    cont = _continuation_rows(text)
    best = 0.0
    for i, _item_ in enumerate(lines, start=1):
        if not _item_.strip() or i in cont:
            continue
        best = max(best, len(_indent_of(_item_)) / unit)
    return best


def _base_levels(before: str) -> float:
    base = _indent_of(next((_item_ for _item_ in before.split("\n") if _item_.strip()), ""))
    unit = indent_unit(before, "\t" if "\t" in before else " ") or 1
    return len(base) / unit


def _render(levels, char, width):
    one = char * width if char == " " else "\t"
    return "\n".join(one * lv + code for lv, code in levels)


_AFTER_SHAPES = [
    [(0, "def f():"), (1, "if x:"), (2, "return 1")],   # col-0 base, 2 levels deep
    [(0, "def f():"), (1, "x = 1"), (1, "return x")],    # col-0 base, flat body
    [(1, "if x:"), (2, "return 1")],                     # already-indented base
    [(0, "a = foo(b,"), (0, "c)"), (0, "d = 2")],        # bracket continuation
]
_AFTERS = [
    _render(shape, char, width)
    for shape in _AFTER_SHAPES
    for char, width in [(" ", 2), (" ", 4), ("\t", 1)]
]
# Match sites whose per-level unit is *detectable* (>1 logical indent depth), so
# the level-based no-explosion measurement below is reliable.
_BEFORES_NESTED = [
    "\tif q:\n\t\told = 1",                 # tab, base 1
    "        if q:\n            old = 1",   # 4-space, base 8
    "    if q:\n      old = 1",             # 2-space, base 4
    "if q:\n    old = 1",                   # col-0 match site
    "\t\t\tif q:\n\t\t\t\told = 1",         # tab, base 3
]
# A *flat* site (one logical level at 8sp) whose only extra raw "depth" is a
# bracket continuation.  Its gcd "unit" is a bogus 8, so the level-based
# measurement is itself unreliable here — it's covered by the exact-output
# regression ``test_flush_left_flat_continuation_site_no_explosion`` instead, and
# kept in the idempotence corpus (which is char-exact, not measurement-based).
_FLAT_CONT_BEFORE = "        result = foo(a,\n                     b)\n        return result"

_PAIRS_NESTED = list(itertools.product(_AFTERS, _BEFORES_NESTED))
_PAIRS_ALL = _PAIRS_NESTED + list(itertools.product(_AFTERS, [_FLAT_CONT_BEFORE]))


def test_property_no_indent_explosion():
    """Output is never deeper than the input's depth plus the match-site base.

    A direct assertion that indent cannot *explode*: re-indenting can shift a
    block to the file's base and rescale its unit, but the deepest line must stay
    within ``input_levels + base_levels`` (+1 level slack for rounding at unit
    boundaries).  The pre-fix flush-left-into-tabs bug overshot this by 3–6
    levels; legitimate re-indents land at or under it.  Restricted to nested
    sites, where the level measurement (via ``indent_unit``) is trustworthy.
    """
    for after, before in _PAIRS_NESTED:
        out = reindent_to_match(after, before)
        bound = _max_indent_levels(after) + _base_levels(before) + 1.0
        assert _max_indent_levels(out) <= bound + 1e-9, (
            f"indent explosion: out={_max_indent_levels(out):.2f} > {bound:.2f}\n"
            f"after={after!r}\nbefore={before!r}\nout={out!r}"
        )


def test_property_idempotence():
    """Re-indenting an already-aligned block is a no-op — the load-bearing test.

    ``reindent(reindent(x, b), b) == reindent(x, b)``.  This is char-exact (no
    ``indent_unit`` measurement, so it can't move in lockstep with a unit
    mis-detection the way the no-explosion bound can).  If the first pass over- or
    under-shoots (e.g. an explosion), the second pass — now on file-aligned input
    — moves it again, breaking equality.  Idempotence therefore certifies the
    alignment is a stable fixed point.  Runs over the full corpus, including the
    flat-continuation site that the no-explosion measurement can't cover.
    """
    for after, before in _PAIRS_ALL:
        once = reindent_to_match(after, before)
        twice = reindent_to_match(once, before)
        assert once == twice, (
            f"not idempotent\nafter={after!r}\nbefore={before!r}\n"
            f"once={once!r}\ntwice={twice!r}"
        )


def test_flush_left_flat_continuation_site_no_explosion():
    """Flush-left block into a flat match site whose only depth is a continuation.

    Regression (advisor-found): the match-site unit detector saw the raw depths
    ``{8, 21}`` (8 = base, 21 = a bracket-continuation alignment) and read a bogus
    unit of 8, so a nested 4-space ``after`` ratio-scaled to 8/4 = 2× — exploding
    a one-level body to 16 spaces.  The unit detector now excludes continuation
    rows (agreeing with ``indent_unit``), so the body lands one real level in.
    """
    before = "        result = foo(a,\n                     b)\n        return result"
    after = "def f():\n    y = transform()\n    return y"   # 4-space, col-0 base
    fixed = reindent_to_match(after, before).split("\n")
    assert fixed[0] == "        def f():"          # base → 8 spaces
    assert fixed[1] == "            y = transform()"  # +1 level → 12 (not 16)
    assert fixed[2] == "            return y"          # +1 level → 12
    # The 8-space base is a method body — parse it in that context.
    ast.parse("class C:\n    def m(self):\n" + "\n".join(fixed))


def test_cross_unit_continuation_content_changed():
    """Tab-indented file + space-indented LLM snippet where continuation content changes.

    Regression: when the continuation line's stripped content differs from the
    match site (normal edit — LLM actually modifies a multi-line argument), the
    content-map fast-path is a miss.  The continuation guard then mixed units
    (``leading`` in spaces + ``last_delta`` in tabs), causing indent explosion
    (8 tabs instead of a reasonable position).  The fix normalises ``leading``
    into file-char units before adding ``last_delta``.
    """
    # File uses tabs for nesting + spaces for bracket alignment.
    actual_before = "\t\tfoo(a,\n\t\t    OLD)\n\t\tbar()"
    # LLM emits with spaces, AND changes the continuation content.
    after = "    foo(a,\n        b_new)\n    bar()"

    fixed = reindent_to_match(after, actual_before)
    lines = fixed.split("\n")

    assert lines[0] == "\t\tfoo(a,", f"line 0: {lines[0]!r}"
    # Continuation content changed → no content-map hit → guard path.
    # The flat tab site has one logical level (``foo``/``bar`` both at 2 tabs), so
    # its per-level unit is the conventional 1 tab/level (``_match_site_unit``),
    # giving indent_ratio = 1/4 = 0.25.  The content-matched ``foo(a,`` sets
    # last_delta = 2 - round(4*0.25) = 1, so round(8*0.25) + 1 = 3 tabs.
    # The exact mixed tab+space indent is unrecoverable for changed content
    # (information only exists in the content-map), but 3 tabs is a
    # syntactically valid approximation — not the pre-fix explosion.
    assert lines[1] == "\t\t\tb_new)", f"line 1: {lines[1]!r}"
    assert lines[2] == "\t\tbar()", f"line 2: {lines[2]!r}"

    # Must parse as valid Python.
    import ast
    ast.parse("def f():\n" + fixed)


# ══════════════════════════════════════════════════════════════════════════════
# B2: reindent_block / reindent_to_anchor must respect the destination file's
# chars-per-level (dest_unit) so a tab/4-space block dropped into a 2-space file
# maps each level to 2 spaces, not the legacy hardcoded 4.  Without dest_unit the
# legacy behaviour is preserved (regression guard).
# ══════════════════════════════════════════════════════════════════════════════


def test_reindent_block_tab_into_2space_file_uses_dest_unit():
    """A tab-indented block re-based into a 2-space file must expand each level to
    2 spaces when dest_unit is supplied (the fix), not 4 (legacy)."""
    tab_block = "\tx\n\t\ty\n"  # base 1 tab, body 2 tabs -> 1 level deeper
    out = reindent_block(tab_block, "  ", dest_unit=2)
    assert out == "  x\n    y\n", repr(out)  # base 2sp, body 2+2=4sp


def test_reindent_block_legacy_default_preserves_hardcoded_4():
    """Without dest_unit, the tab->space mapping stays at 4 spaces/level (legacy)."""
    tab_block = "\tx\n\t\ty\n"
    out = reindent_block(tab_block, "  ")  # no dest_unit
    assert out == "  x\n      y\n", repr(out)  # base 2sp, body 2+4=6sp


def test_reindent_block_4space_file_unaffected_by_dest_unit():
    """For a 4-space file, dest_unit=4 is identical to the legacy default -- no
    regression for the dominant Python convention."""
    tab_block = "\tx\n\t\ty\n"
    assert reindent_block(tab_block, "    ", dest_unit=4) == reindent_block(tab_block, "    ")


def test_reindent_block_same_style_round_trips():
    """Same-style space->space blocks must reproduce exactly even with dest_unit."""
    sp_block = "    a\n        b\n"  # 4-space, 1 level deeper
    out = reindent_block(sp_block, "    ", dest_unit=4)
    assert out == "    a\n        b\n", repr(out)


def test_reindent_to_anchor_tab_into_2space_file_uses_dest_unit():
    """reindent_to_anchor: tab snippet into a 2-space space-anchor must expand each
    level to 2 spaces when dest_unit is supplied."""
    snip = ["\tfoo()\n", "\t\tbar()\n"]
    out = reindent_to_anchor(snip, "    if x:\n", dest_unit=2)
    # anchor_indent = 4 spaces; foo level 0 -> 4sp; bar level 1 -> 4+2=6sp
    assert out == ["    foo()\n", "      bar()\n"], out


def test_reindent_to_anchor_legacy_default_preserves_hardcoded_4():
    """Without dest_unit the tab->space mapping stays at 4 spaces/level (legacy)."""
    snip = ["\tfoo()\n", "\t\tbar()\n"]
    out = reindent_to_anchor(snip, "    if x:\n")
    # bar level 1 -> 4+4=8 spaces
    assert out == ["    foo()\n", "        bar()\n"], out


def test_dest_unit_threaded_from_file_detection():
    """The realistic caller pattern: detect dest_unit from the file content and
    pass it through. A 2-space file yields dest_unit=2."""
    two_space_file = "def f():\n  if x:\n    return 1\n"
    dest_unit = indent_unit(two_space_file, detect_indent_char(two_space_file.splitlines()))
    assert dest_unit == 2, dest_unit
    tab_block = "\tx\n\t\ty\n"
    assert reindent_block(tab_block, "  ", dest_unit=dest_unit) == "  x\n    y\n"


# ══════════════════════════════════════════════════════════════════════════════
# F1 invariant: reindent_block & reindent_to_anchor share ONE level→char core
# ══════════════════════════════════════════════════════════════════════════════

def _leading_and_stripped(text_or_lines):
    """[(leading_whitespace, stripped_content)] for every non-empty line."""
    out = []
    lines = text_or_lines if isinstance(text_or_lines, list) else text_or_lines.split("\n")
    for ln in lines:
        body = ln.rstrip("\n")
        if body.strip():
            lead = body[:len(body) - len(body.lstrip())]
            out.append((lead, body.lstrip()))
    return out


def test_reindent_block_anchor_level_char_invariant():
    """reindent_block and reindent_to_anchor MUST agree on level→char mapping.

    Both delegate to the shared ``_block_levels`` / ``_resolve_space_unit`` core;
    this matrix test is the regression guard for the B2 bug class (the two paths
    drifted on space_unit — tab→space hardcoded to 4 in one, detected-unit in
    the other). Across source styles × destinations × dest_unit hints, every
    output line's (leading-whitespace, stripped-content) must match.
    """
    sources = {
        "tab":    ["\tfoo()\n", "\t\tbar()\n", "\t\t\tbaz()\n"],
        "2space": ["  foo()\n", "    bar()\n", "      baz()\n"],
        "4space": ["    foo()\n", "        bar()\n", "            baz()\n"],
    }
    dests = [
        ("        if x:\n", 2),    # 8-space space anchor, dest_unit=2
        ("        if x:\n", 4),    # 8-space space anchor, dest_unit=4
        ("        if x:\n", None), # legacy default
        ("\tif x:\n", 2),          # tab anchor (dest_unit irrelevant for tab emit)
        ("\tif x:\n", None),
    ]
    for sname, slines in sources.items():
        for anchor_line, dest_unit in dests:
            base_indent = anchor_line[:len(anchor_line) - len(anchor_line.lstrip())]
            block_out = reindent_block("".join(slines), base_indent, dest_unit=dest_unit)
            anchor_out = reindent_to_anchor(list(slines), anchor_line, dest_unit=dest_unit)
            b = _leading_and_stripped(block_out)
            a = _leading_and_stripped(anchor_out)
            assert len(b) == len(a), (sname, anchor_line, dest_unit, b, a)
            for (lb, sb), (la, sa) in zip(b, a, strict=False):
                assert sb == sa, ("content", sname, anchor_line, dest_unit, sb, sa)
                assert lb == la, ("indent", sname, anchor_line, dest_unit, lb, la)


def test_block_levels_and_resolve_space_unit_core():
    """The shared core returns the expected (min, unit, block_char) and unit."""
    from external_llm.common.indent_utils import _block_levels, _resolve_space_unit
    # 4-space source: min 4, unit 4, space.
    assert _block_levels(["    a", "        b"]) == (4, 4, " ")
    # tab source: min 1, unit 1, tab.
    assert _block_levels(["\ta", "\t\tb"]) == (1, 1, "\t")
    # single level: unit defaults to 4 (space) / 1 (tab).
    assert _block_levels(["    a"]) == (4, 4, " ")
    assert _block_levels(["\ta"]) == (1, 1, "\t")
    # all-empty → None.
    assert _block_levels(["", "  "]) is None
    # dest_unit wins; else block's own unit (space) / 4 (tab).
    assert _resolve_space_unit(2, 4, " ") == 2
    assert _resolve_space_unit(None, 4, " ") == 4
    assert _resolve_space_unit(None, 1, "\t") == 4
    assert _resolve_space_unit(2, 1, "\t") == 2


# ══════════════════════════════════════════════════════════════════════════════
# F2: file-wide unit hint for flat (single-level) match sites
# ══════════════════════════════════════════════════════════════════════════════

def test_file_unit_hint_fixes_flat_2space_overindent():
    """A flat single-level 2-space site can't reveal the file's unit; without a
    hint a 4-space LLM snippet over-indents. ``file_unit=2`` scales it down."""
    # 4-space snippet: base 4sp, body 8sp (one level deeper). GCD(4,8)=4.
    after = "    y = 2\n        z = 3\n"
    flat_2sp_site = "  x = 1\n  return x\n"
    hinted = reindent_to_match(after, flat_2sp_site, file_unit=2)
    lines = hinted.split("\n")
    assert lines[0] == "  y = 2", lines
    # base 2sp + one 2-space level = 4sp (was 6sp without the hint)
    assert lines[1] == "    z = 3", lines


def test_file_unit_hint_none_preserves_legacy():
    """``file_unit=None`` keeps the historic behaviour and reproduces the bug."""
    after = "    y = 2\n        z = 3\n"
    flat_2sp_site = "  x = 1\n  return x\n"
    legacy = reindent_to_match(after, flat_2sp_site)
    none_hint = reindent_to_match(after, flat_2sp_site, file_unit=None)
    assert legacy == none_hint
    # legacy over-indents the body to 6sp (the pre-F2 behaviour)
    assert legacy.split("\n")[1] == "      z = 3", legacy


def test_file_unit_hint_4space_file_unaffected():
    """A 4-space file's unit already matches the snippet → hint changes nothing."""
    after = "    y = 2\n        z = 3\n"
    flat_4sp_site = "    x = 1\n    return x\n"
    assert reindent_to_match(after, flat_4sp_site) == reindent_to_match(after, flat_4sp_site, file_unit=4)


# ══════════════════════════════════════════════════════════════════════════════
# F3: content-map last_delta domain must match the continuation consumer
# ══════════════════════════════════════════════════════════════════════════════

def test_same_char_diff_unit_content_match_continuation_alignment():
    """Same indent char (space) but different unit (2sp snippet -> 4sp file).

    The owner ``foo(a,`` is content-matched to the file (col 8); its bracket
    continuation ``b)`` (a NEW line, not in ``before``) must consume the owner's
    ``last_delta`` in the SAME unit domain the consumer reads.  Before F3 the
    content-map producer gated ``last_delta`` on ``use_scaled_additive`` alone
    (file-char domain) while the consumer gated on ``cross_char`` (raw domain);
    with same-char/diff-unit (cross_char=False, use_scaled_additive=True) the
    domains disagreed and the continuation collapsed to col 7 — shallower than
    its own opener at col 8.  F3 gates the producer on
    ``use_scaled_additive and cross_char`` so both producers agree with the
    consumer; the continuation lands at col 11 (the file's own continuation
    column).
    """
    after  = "  if c:\n    foo(a,\n       b)\n"          # 2sp: owner col4, cont col7
    before = "    if c:\n        foo(a,\n           OLD)\n"  # 4sp: owner col8, cont col11
    fixed = reindent_to_match(after, before)
    lines = fixed.split("\n")

    def col(i):
        return len(lines[i]) - len(lines[i].lstrip(" "))

    assert lines[0] == "    if c:", lines
    assert lines[1] == "        foo(a,", lines          # owner content-matched to file col 8
    cont_col = col(2)
    opener_col = col(1)
    assert cont_col == 11, (
        f"continuation misaligned: col {cont_col} (owner opener col {opener_col}, "
        f"file expects 11)\n{fixed}"
    )
    assert cont_col >= opener_col, (          # never shallower than its opener
        f"continuation col {cont_col} < opener col {opener_col}\n{fixed}"
    )
    import ast
    ast.parse("def f():\n" + fixed)            # must stay syntactically valid
