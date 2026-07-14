"""Unit tests for loop_variable pre-derivation in _attach_edit_contracts.

When a guard_add IR has insert_scope="for_loop" and the target function contains
multiple for-loops, the planner must derive loop_variable at planning time so the
executor does not encounter ≥2-candidates ambiguity.

Spec:
  Single for-loop   → loop_variable omitted (executor derive-at-apply-time is safe)
  Multi-loop, 1 match  → loop_variable embedded in IR
  Multi-loop, 0 matches → LLM route (preferred_mode left unset)
  Multi-loop, ≥2 matches → LLM route (preferred_mode left unset)
  File parse error  → graceful fallback (derive-at-apply-time, deterministic path kept)
"""
import ast
import textwrap

# ─── Isolated derivation logic (mirrors _attach_edit_contracts internals) ──────

def _derive_loop_var_pre(
    func_src: str,
    guard_stmt: str,
) -> tuple[str, bool]:
    """Replicate the planning-time loop_variable derivation from _attach_edit_contracts.

    Returns:
        (loop_var, ambiguous)
        - loop_var:   non-empty str  → unique match found, embed in IR
        - ambiguous:  True           → ≥2 or 0 matches, route to LLM
        - ("", False)                → single loop, let executor derive
    """
    try:
        tree = ast.parse(func_src)
        func_node = next(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    except (SyntaxError, StopIteration):
        return "", False  # parse error → graceful fallback

    for_loops = [n for n in ast.walk(func_node) if isinstance(n, ast.For)]

    if len(for_loops) <= 1:
        # Single loop: derive-at-apply-time is always safe
        return "", False

    # Multiple loops → derive now
    try:
        gs_names: set = {
            n.id
            for n in ast.walk(ast.parse(guard_stmt, mode="exec"))
            if isinstance(n, ast.Name)
        }
    except SyntaxError:
        gs_names = set()

    loop_target_vars: set = set()
    for fl in for_loops:
        for tn in ast.walk(fl.target):
            if isinstance(tn, ast.Name):
                loop_target_vars.add(tn.id)

    matches = gs_names & loop_target_vars
    if len(matches) == 1:
        return matches.pop(), False   # unique → embed
    return "", True                   # 0 or ≥2 → ambiguous


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _func(body: str) -> str:
    """Wrap body lines in a minimal function definition."""
    indented = textwrap.indent(textwrap.dedent(body).strip(), "    ")
    return f"def process(items, records):\n{indented}\n"


# ─── Single for-loop → no pre-derivation needed ───────────────────────────────

class TestSingleLoop:

    def test_single_loop_no_pre_derive(self):
        src = _func("""
            for item in items:
                if item is None:
                    continue
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if item is None: continue")
        assert lv == ""
        assert ambiguous is False

    def test_single_loop_guard_uses_loop_var(self):
        src = _func("""
            for record in records:
                if record.status == "skip":
                    continue
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if record.status == 'skip': continue")
        assert lv == ""
        assert ambiguous is False


# ─── Multiple for-loops, unique match ─────────────────────────────────────────

class TestMultiLoopUniqueMatch:

    def test_guard_uses_inner_loop_var(self):
        src = _func("""
            for item in items:
                for sub in item.subs:
                    if sub is None:
                        continue
                    do_work(sub)
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if sub is None: continue")
        assert lv == "sub"
        assert ambiguous is False

    def test_guard_uses_outer_loop_var(self):
        src = _func("""
            for item in items:
                process(item)
                for record in records:
                    if record.ok:
                        continue
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if item.skip: continue")
        assert lv == "item"
        assert ambiguous is False

    def test_three_loops_unique_match(self):
        src = _func("""
            for a in items:
                for b in a.children:
                    for c in b.leaves:
                        if c.expired:
                            continue
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if c.expired: continue")
        assert lv == "c"
        assert ambiguous is False

    def test_sequential_loops_guard_refs_second(self):
        """Sequential (not nested) loops — guard references only second loop's var."""
        src = _func("""
            for x in items:
                handle(x)
            for y in records:
                if y is None:
                    continue
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if y is None: continue")
        assert lv == "y"
        assert ambiguous is False


# ─── Multiple for-loops, ambiguous (0 matches) ────────────────────────────────

class TestMultiLoopZeroMatches:

    def test_guard_uses_no_loop_var(self):
        """Guard uses only module-level names — 0 intersection with loop targets."""
        src = _func("""
            for item in items:
                handle(item)
            for record in records:
                process(record)
        """)
        # Guard references neither `item` nor `record`
        lv, ambiguous = _derive_loop_var_pre(src, "if config.skip: continue")
        assert lv == ""
        assert ambiguous is True

    def test_guard_uses_only_constants(self):
        src = _func("""
            for x in items:
                pass
            for y in records:
                pass
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if True: continue")
        assert lv == ""
        assert ambiguous is True


# ─── Multiple for-loops, ambiguous (≥2 matches) ───────────────────────────────

class TestMultiLoopTwoMatches:

    def test_guard_uses_both_loop_vars(self):
        """Guard references variables from both loops → ambiguous."""
        src = _func("""
            for item in items:
                for sub in item.subs:
                    if item and sub:
                        continue
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if item and sub: continue")
        assert lv == ""
        assert ambiguous is True

    def test_guard_references_all_three_vars(self):
        src = _func("""
            for a in items:
                for b in a.children:
                    for c in b.leaves:
                        if a and b and c:
                            continue
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if a and b and c: continue")
        assert lv == ""
        assert ambiguous is True


# ─── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_tuple_unpack_loop_target_unique(self):
        """for (k, v) in pairs: guard uses v → unique."""
        src = textwrap.dedent("""\
            def process(pairs, items):
                for k, v in pairs:
                    pass
                for x in items:
                    pass
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if v is None: continue")
        assert lv == "v"
        assert ambiguous is False

    def test_syntax_error_in_guard_stmt_treated_as_zero_match(self):
        src = _func("""
            for x in items:
                pass
            for y in records:
                pass
        """)
        _lv, ambiguous = _derive_loop_var_pre(src, "if :::broken:::")
        # SyntaxError in guard → gs_names = {} → 0 intersection → ambiguous
        assert ambiguous is True

    def test_file_parse_error_graceful_fallback(self):
        """If the source itself is unparseable, no ambiguity raised (executor handles)."""
        lv, ambiguous = _derive_loop_var_pre("def broken(:", "if x: continue")
        assert lv == ""
        assert ambiguous is False

    def test_no_for_loops_at_all(self):
        """Function with no for-loops → single-loop path → no pre-derivation."""
        src = textwrap.dedent("""\
            def process(items):
                while items:
                    items.pop()
        """)
        lv, ambiguous = _derive_loop_var_pre(src, "if True: continue")
        assert lv == ""
        assert ambiguous is False
