"""Unit tests for ASTOpExecutor — typed AST operation executor.

Covers every op handler (replace_expr, add_import, remove_import_name,
add_class_field, add_guard, delete_stmt, list_append, list_remove), the apply()
orchestrator (symbol parsing, dispatch, syntax rollback, changed flag), the
module-level guard-idempotency helper, and the name-safety machinery.

All ops are pure AST transformations on source strings — no I/O, no mocking.
"""
from __future__ import annotations

import ast

import external_llm.agent.tool_handlers.ast_op_executor as _aoe
from external_llm.agent.tool_handlers.ast_op_executor import (
    ASTOpExecutor,
    _guard_already_present,
    _safe_unparse_iter,
)

EX = ASTOpExecutor()


# ── module-level helpers ────────────────────────────────────────────────────


class TestSafeUnparseIter:
    def test_normal_for(self):
        tree = ast.parse("for x in range(10):\n    pass\n")
        node = tree.body[0]
        assert _safe_unparse_iter(node) == "range(10)"

    def test_complex_iterable(self):
        tree = ast.parse("for k, v in obj.items():\n    pass\n")
        assert _safe_unparse_iter(tree.body[0]) == "obj.items()"

    def test_invalid_node_returns_empty(self):
        # Non-For node / malformed iter → silent fallback "".
        # _safe_unparse_iter catches (SyntaxError, TypeError, AttributeError);
        # use AttributeError so the except clause swallows it.
        class _Bad:
            @property
            def iter(self):
                raise AttributeError("no iter")

        assert _safe_unparse_iter(_Bad()) == ""


class TestGuardAlreadyPresent:
    SRC = (
        "def f(x):\n"
        "    if x is None:\n"
        "        return None\n"
        "    return x + 1\n"
    )

    def test_present_returns_true(self):
        stmt = "if x is None:\n    return None"
        assert _guard_already_present(self.SRC, stmt, "f") is True

    def test_absent_returns_false(self):
        stmt = "if x < 0:\n    raise ValueError()"
        assert _guard_already_present(self.SRC, stmt, "f") is False

    def test_non_if_statement_returns_false(self):
        assert _guard_already_present(self.SRC, "x = 1", "f") is False

    def test_condition_without_terminal_action(self):
        # if-condition matches but body has no raise/return/continue/break
        src = "def f(x):\n    if x is None:\n        pass\n    return x\n"
        stmt = "if x is None:\n    return None"
        assert _guard_already_present(src, stmt, "f") is False

    def test_loop_body_scope(self):
        src = (
            "def f(xs):\n"
            "    for x in xs:\n"
            "        if x is None:\n"
            "            continue\n"
            "        print(x)\n"
        )
        stmt = "if x is None:\n    continue"
        assert _guard_already_present(src, stmt, "f") is True

    def test_unknown_function_returns_false(self):
        assert _guard_already_present(self.SRC, "if x is None:\n    return None", "nope") is False

    def test_unparseable_stmt_returns_false(self):
        assert _guard_already_present(self.SRC, ")))not code(((", "f") is False

    def test_qualified_symbol_resolves_class_scoped(self):
        # A same-named method in a different class must NOT satisfy the check.
        src = (
            "class A:\n"
            "    def m(self, x):\n"
            "        if x is None:\n"
            "            return None\n"
            "class B:\n"
            "    def m(self, x):\n"
            "        return x\n"
        )
        stmt = "if x is None:\n    return None"
        assert _guard_already_present(src, stmt, "A.m") is True
        assert _guard_already_present(src, stmt, "B.m") is False
        # Explicit parent_class form (bare symbol) resolves the same way.
        assert _guard_already_present(src, stmt, "m", "A") is True
        assert _guard_already_present(src, stmt, "m", "B") is False

    def test_first_function_with_bare_name_still_matches(self):
        # Legacy bare-name behaviour: first function with that name matches.
        src = "def f(x):\n    if x is None:\n        return None\n"
        assert _guard_already_present(src, "if x is None:\n    return None", "f") is True


# ── apply() orchestrator ────────────────────────────────────────────────────


class TestApplyDispatch:
    def test_unknown_op_type_recorded_as_failed(self):
        src = "x = 1\n"
        r = EX.apply(src, [{"type": "bogus"}])
        assert r.success is False
        assert r.ops_applied == 0
        assert any("unknown op type" in f for f in r.ops_failed)

    def test_empty_ops_returns_not_success(self):
        src = "x = 1\n"
        r = EX.apply(src, [])
        assert r.success is False and r.ops_applied == 0 and r.changed is False

    def test_qualified_symbol_parsing(self):
        src = (
            "class A:\n"
            "    def m(self):\n"
            "        return 1\n"
            "class B:\n"
            "    def m(self):\n"
            "        return 2\n"
        )
        # Bare "m" matches FIRST (A.m). "B.m" must resolve to B.m.
        r_bare = EX.apply(src, [{"type": "replace_expr", "old": "return 1", "new": "return 10"}], symbol="m")
        assert r_bare.success and "return 10" in r_bare.new_source
        r_qual = EX.apply(src, [{"type": "replace_expr", "old": "return 2", "new": "return 20"}], symbol="B.m")
        assert r_qual.success and "return 20" in r_qual.new_source

    def test_syntax_error_rolls_back(self):
        # replace_expr replaces 'pass' with 'continue' inside a non-loop function.
        # ast.parse accepts it; compile() rejects ('continue not properly in loop').
        src = "def f():\n    pass\n"
        r = EX.apply(src, [{"type": "replace_expr", "old": "pass", "new": "continue"}], symbol="f")
        assert r.success is False
        assert r.new_source == src           # rolled back to original
        assert r.ops_applied == 0            # reported as 0 on rollback
        assert r.ops_failed and "syntax error" in r.ops_failed[0]

    def test_changed_flag_distinguishes_effect(self):
        src = "def f():\n    return 1\n"
        # Effective change
        r_eff = EX.apply(src, [{"type": "replace_expr", "old": "return 1", "new": "return 2"}], symbol="f")
        assert r_eff.changed is True
        # Idempotent op (guard already present) → changed False
        guarded = "def f(x):\n    if x is None:\n        return None\n    return x\n"
        r_idem = EX.apply(
            guarded,
            [{"type": "add_guard", "statement": "if x is None:\n    return None"}],
            symbol="f",
        )
        assert r_idem.success and r_idem.changed is False

    def test_op_exception_caught(self):
        # Malformed op dict that triggers an internal exception path
        r = EX.apply("x = 1\n", [{"type": "list_append", "list_name": None}])  # type: ignore[arg-type]
        assert r.success is False
        assert any("list_append" in f for f in r.ops_failed)


# ── _find_func_node ─────────────────────────────────────────────────────────


class TestFindFuncNode:
    def test_bare_match_returns_first(self):
        tree = ast.parse("def a():\n    pass\ndef a():\n    return 2\n")
        node = ASTOpExecutor._find_func_node(tree, "a")
        assert isinstance(node, ast.FunctionDef)

    def test_parent_class_scoping(self):
        tree = ast.parse(
            "class A:\n"
            "    def m(self):\n        return 1\n"
            "class B:\n"
            "    def m(self):\n        return 2\n"
        )
        node = ASTOpExecutor._find_func_node(tree, "m", parent_class="B")
        assert isinstance(node, ast.FunctionDef)
        # Verify it's B.m by checking body
        assert isinstance(node.body[0], ast.Return)

    def test_nested_class_chain(self):
        tree = ast.parse(
            "class Outer:\n"
            "    class Inner:\n"
            "        def m(self):\n            return 1\n"
        )
        node = ASTOpExecutor._find_func_node(tree, "m", parent_class="Outer.Inner")
        assert isinstance(node, ast.FunctionDef)

    def test_empty_symbol_returns_none(self):
        tree = ast.parse("x = 1\n")
        assert ASTOpExecutor._find_func_node(tree, "") is None

    def test_missing_class_returns_none(self):
        tree = ast.parse("class A:\n    def m(self):\n        pass\n")
        assert ASTOpExecutor._find_func_node(tree, "m", parent_class="Z") is None


# ── _ws_tolerant_span ───────────────────────────────────────────────────────


class TestWsTolerantSpan:
    def test_exact_unique_match(self):
        text = "line one   \nline two\n"
        span = ASTOpExecutor._ws_tolerant_span(text, "line one")
        assert span == "line one   "   # original span incl trailing ws

    def test_multi_match_returns_none(self):
        text = "dup\nfoo\ndup\n"
        assert ASTOpExecutor._ws_tolerant_span(text, "dup") is None

    def test_no_match_returns_none(self):
        assert ASTOpExecutor._ws_tolerant_span("abc\n", "xyz") is None


# ── replace_expr ────────────────────────────────────────────────────────────


class TestReplaceExpr:
    def test_file_level_replace(self):
        src = "x = 1\ny = 2\n"
        new, ok = EX._replace_expr(src, {"old": "x = 1", "new": "x = 10"}, "")
        assert ok and "x = 10" in new

    def test_scoped_to_function(self):
        src = "def f():\n    return 1\ndef g():\n    return 1\n"
        new, ok = EX._replace_expr(src, {"old": "return 1", "new": "return 2"}, "f")
        assert ok
        # Only f's return changed; g's untouched
        assert new.count("return 2") == 1
        assert "def g():\n    return 1" in new

    def test_ambiguous_in_scope_returns_false(self):
        src = "def f():\n    x = 1\n    x = 1\n"
        new, ok = EX._replace_expr(src, {"old": "x = 1", "new": "x = 2"}, "f")
        assert ok is False and new == src

    def test_empty_old_returns_false(self):
        new, ok = EX._replace_expr("x = 1\n", {"old": "", "new": "y"}, "")
        assert ok is False and new == "x = 1\n"

    def test_ws_tolerant_fallback(self):
        src = "def f():\n    return 1  \n"   # trailing spaces
        new, ok = EX._replace_expr(src, {"old": "return 1", "new": "return 2"}, "f")
        assert ok and "return 2" in new

    def test_symbol_not_found_fails_loudly(self):
        # Symbol miss must NOT fall back to a file-wide replace (which would
        # silently mutate a different function) — mirrors delete_stmt's
        # loud-failure contract.
        src = "def foo():\n    return 111\ndef bar():\n    return 1\n"
        op = {"old": "return 111", "new": "return 999"}
        new, ok = EX._replace_expr(src, op, "nonexistent_func")
        assert ok is False and new == src
        assert "_error" in op and "not found" in op["_error"]
        assert "return 999" not in new

    def test_symbol_not_found_no_file_wide_mutation(self):
        # Even a unique file-level pattern must stay untouched when the scoped
        # symbol is missing — no silent cross-function mutation.
        src = "def foo():\n    return 111\ndef bar():\n    return 1\n"
        new, ok = EX._replace_expr(src, {"old": "return 111", "new": "return 999"}, "nope")
        assert ok is False and new == src


# ── add_import ──────────────────────────────────────────────────────────────


class TestAddImport:
    def test_plain_append_after_imports(self):
        src = "import os\n\ndef f():\n    pass\n"
        new, ok = EX._add_import(src, {"import": "import sys"})
        assert ok and new.index("import sys") < new.index("def f")

    def test_idempotent_importfrom(self):
        src = "from os import path\n"
        new, ok = EX._add_import(src, {"import": "from os import path"})
        assert ok and new == src

    def test_idempotent_import(self):
        src = "import os\n"
        new, ok = EX._add_import(src, {"import": "import os"})
        assert ok and new == src

    def test_merge_into_existing_from_import(self):
        src = "from os import path\n"
        new, ok = EX._add_import(src, {"import": "from os import getcwd"})
        assert ok
        assert "getcwd" in new and "path" in new
        # merged into a single line, not two imports
        assert new.count("from os import") == 1

    def test_module_grouped_placement_shadowed_by_idempotency(self):
        # Documented behavior: when standalone ``import os`` exists, adding
        # ``from os import path`` is treated as idempotent (the module is already
        # importable), so it returns (source, True) WITHOUT inserting a new line.
        # The module-grouped-placement branch is therefore shadowed by the
        # idempotency short-circuit for this input shape.
        src = "import os\nimport json\n"
        new, ok = EX._add_import(src, {"import": "from os import path"})
        assert ok and new == src  # idempotent — nothing inserted

    def test_multiline_parenthesized_import_end_detection(self):
        src = "from os import (\n    path,\n    getcwd,\n)\n\nx = 1\n"
        new, ok = EX._add_import(src, {"import": "import sys"})
        assert ok
        # sys inserted AFTER the closing paren, not inside it
        assert new.index("import sys") > new.index(")")

    def test_empty_import_returns_false(self):
        _new, ok = EX._add_import("x = 1\n", {"import": ""})
        assert ok is False

    def test_local_import_not_anchored(self):
        # Indented (function-local) import must NOT be an insertion anchor.
        src = "def f():\n    import os\n    pass\n"
        new, ok = EX._add_import(src, {"import": "import sys"})
        assert ok
        # sys at module level, NOT inside f
        assert new.lstrip().startswith("import sys") or "\nsys" not in new
        assert "    import sys" not in new

    def test_asname_alias_does_not_satisfy_idempotency(self):
        # ``from os import path as p`` binds p, NOT path — adding
        # ``from os import path`` must actually insert (and preserve the alias).
        src = "from os import path as p\n"
        new, ok = EX._add_import(src, {"import": "from os import path"})
        assert ok and new != src
        assert "path" in new and "path as p" in new
        assert new.count("from os import") == 1

    def test_import_as_alias_does_not_satisfy_idempotency(self):
        # ``import os as o`` binds o, not os — adding ``import os`` must insert.
        src = "import os as o\n"
        new, ok = EX._add_import(src, {"import": "import os"})
        assert ok and new != src
        assert "import os\n" in new

    def test_merge_preserves_asname_aliases(self):
        # Merging a new name into an aliased import must keep ``A as B`` intact
        # (rendering only names would drop the alias → B undefined at runtime).
        src = "from os import path as p\n"
        new, ok = EX._add_import(src, {"import": "from os import getcwd"})
        assert ok
        assert "path as p" in new and "getcwd" in new
        assert new.count("from os import") == 1


# ── remove_import_name ──────────────────────────────────────────────────────


class TestRemoveImportName:
    def test_single_name_deletes_line(self):
        src = "from os import path\n"
        new, ok = EX._remove_import_name(src, {"module": "os", "name": "path"})
        assert ok and "path" not in new

    def test_multi_name_rewrite(self):
        src = "from os import path, getcwd\n"
        new, ok = EX._remove_import_name(src, {"module": "os", "name": "path"})
        assert ok and "getcwd" in new and "path" not in new

    def test_relative_import_dots_preserved(self):
        src = "from .pkg import field\n"
        new, ok = EX._remove_import_name(src, {"module": ".pkg", "name": "field"})
        assert ok and "field" not in new

    def test_asname_match(self):
        src = "from os import path as p\n"
        new, ok = EX._remove_import_name(src, {"module": "os", "name": "p"})
        assert ok and "p" not in new.replace("import", "")  # 'p' gone except in 'import'

    def test_name_not_found_returns_false(self):
        src = "from os import path\n"
        new, ok = EX._remove_import_name(src, {"module": "os", "name": "nope"})
        assert ok is False and new == src

    def test_empty_name_returns_false(self):
        assert EX._remove_import_name("x = 1\n", {"name": ""})[1] is False


# ── add_class_field ─────────────────────────────────────────────────────────


class TestAddClassField:
    def test_add_after_existing_field(self):
        src = "class C:\n    a: int = 0\n"
        new, ok = EX._add_class_field(src, {"class_name": "C", "field_name": "b", "field_type": "str"})
        assert ok and "b: str" in new

    def test_add_after_docstring(self):
        src = 'class C:\n    """doc."""\n\n    def m(self):\n        pass\n'
        new, ok = EX._add_class_field(src, {"class_name": "C", "field_name": "x", "field_type": "float", "field_default": "1.0"})
        assert ok and "x: float = 1.0" in new

    def test_idempotent(self):
        src = "class C:\n    a: int = 0\n"
        new, ok = EX._add_class_field(src, {"class_name": "C", "field_name": "a", "field_type": "int"})
        assert ok and new == src

    def test_class_not_found(self):
        src = "class C:\n    a: int = 0\n"
        new, ok = EX._add_class_field(src, {"class_name": "Z", "field_name": "b", "field_type": "int"})
        assert ok is False and new == src

    def test_missing_required_field(self):
        _new, ok = EX._add_class_field("class C:\n    pass\n", {"class_name": "C", "field_name": "", "field_type": "int"})
        assert ok is False


# ── add_guard ───────────────────────────────────────────────────────────────


class TestAddGuard:
    def test_function_body_after_docstring(self):
        src = 'def f(x):\n    """doc."""\n    return x\n'
        new, ok = EX._add_guard(src, {"statement": "if x is None:\n    return None"}, "f")
        assert ok
        # guard inserted AFTER docstring
        doc_line = new.index('"""doc."""')
        guard_line = new.index("if x is None")
        ret_line = new.index("return x")
        assert doc_line < guard_line < ret_line

    def test_idempotent_verbatim(self):
        src = "def f(x):\n    if x is None:\n        return None\n    return x\n"
        new, ok = EX._add_guard(src, {"statement": "if x is None:\n    return None"}, "f")
        assert ok and new == src

    def test_name_safety_deferred_insertion(self):
        # guard references 'y' which is defined later → insertion after y's def
        src = "def f():\n    y = compute()\n    return y\n"
        new, ok = EX._add_guard(src, {"statement": "if y:\n    return"}, "f")
        assert ok
        assert new.index("y = compute()") < new.index("if y:")

    def test_name_safety_undefined_returns_false(self):
        src = "def f():\n    return 1\n"
        new, ok = EX._add_guard(src, {"statement": "if undefined_name:\n    return"}, "f")
        assert ok is False and new == src

    def test_for_loop_scope(self):
        src = "def f(xs):\n    for x in xs:\n        print(x)\n"
        new, ok = EX._add_guard(src, {"statement": "if x is None:\n    continue", "insert_scope": "for_loop", "loop_variable": "x"}, "f")
        assert ok
        # guard is first statement of loop body
        loop_body = new.split("for x in xs:")[1]
        assert loop_body.lstrip().startswith("if x is None")

    def test_while_loop_scope(self):
        # 'done' must be a parameter so name-safety treats it as always-available
        # (a guard referencing an undefined name is rejected as unsafe).
        src = "def f(done):\n    while running:\n        print(1)\n"
        new, ok = EX._add_guard(src, {"statement": "if done:\n    break", "insert_scope": "while_loop"}, "f")
        assert ok and "if done" in new

    def test_ambiguous_loops_returns_false(self):
        src = "def f():\n    for x in a:\n        pass\n    for x in b:\n        pass\n"
        new, ok = EX._add_guard(src, {"statement": "if x:\n    continue", "insert_scope": "for_loop", "loop_variable": "x"}, "f")
        assert ok is False and new == src

    def test_loop_iterable_src_disambiguates(self):
        src = "def f():\n    for x in a:\n        pass\n    for x in b:\n        pass\n"
        op = {
            "statement": "if x:\n    continue",
            "insert_scope": "for_loop",
            "loop_variable": "x",
            "loop_iterable_src": "b",
        }
        new, ok = EX._add_guard(src, op, "f")
        assert ok
        # guard landed in the 'for x in b' loop (the one before it in source)
        b_loop_idx = new.index("for x in b")
        guard_idx = new.index("if x:")
        assert guard_idx > b_loop_idx

    def test_ir_dict_compact_form(self):
        src = "def f(x):\n    return x\n"
        ir = {"compact": "if x is None:\n    return None", "insert_scope": "function_body", "loop_variable": ""}
        new, ok = EX._add_guard(src, {"ir": ir}, "f")
        assert ok and "if x is None" in new

    def test_empty_statement_returns_false(self):
        _new, ok = EX._add_guard("def f():\n    pass\n", {"statement": ""}, "f")
        assert ok is False

    def test_unknown_function_returns_false(self):
        new, ok = EX._add_guard("def f():\n    pass\n", {"statement": "if x:\n    return"}, "nope")
        assert ok is False and new == "def f():\n    pass\n"

    def test_verbatim_in_other_function_not_idempotent(self):
        # A single-line guard text inside a DIFFERENT function must not satisfy
        # the idempotency check for the target function.
        src = "def a():\n    if not x: return\n    pass\ndef b(x):\n    pass\n"
        new, ok = EX._add_guard(src, {"statement": "if not x: return"}, "b")
        assert ok and new != src
        b_part = new.split("def b(x):")[1]
        assert "if not x: return" in b_part

    def test_same_named_method_other_class_not_idempotent(self):
        # Guard present in A.m must not satisfy the check for B.m (apply()
        # resolves "B.m" → bare "m" + parent class "B").
        src = (
            "class A:\n"
            "    def m(self, x):\n"
            "        if x is None:\n"
            "            return None\n"
            "class B:\n"
            "    def m(self, x):\n"
            "        return x\n"
        )
        op = {"type": "add_guard", "statement": "if x is None:\n    return None"}
        res = EX.apply(src, [op], symbol="B.m")
        assert res.success and res.changed
        b_part = res.new_source.split("class B:")[1]
        assert "if x is None" in b_part
        # A.m keeps its guard untouched; re-adding to A.m is idempotent.
        assert _guard_already_present(res.new_source, "if x is None:\n    return None", "A.m") is True
        res2 = EX.apply(res.new_source, [op], symbol="A.m")
        assert res2.success and not res2.changed


# ── add_guard single-parse (P2) ─────────────────────────────────────────────


class TestAddGuardSingleParse:
    """P2: the whole add_guard path parses the source exactly once.

    The parsed tree is threaded through _guard_already_present /
    _insert_at_function_body / _insert_at_loop_body via the keyword-only
    _src_tree parameter; None falls back to a self-parse so standalone helper
    calls keep working.

    ast.parse(source) calls are distinguished from guard-statement parses
    (which always pass mode="exec") by the absence of the mode kwarg.
    """

    @staticmethod
    def _source_parse_count(monkeypatch, src, op, symbol, *, _src_tree=None):
        real_parse = _aoe.ast.parse
        seen: list[tuple] = []

        def _counting(text, *args, **kwargs):
            seen.append((text, kwargs.get("mode")))
            return real_parse(text, *args, **kwargs)

        monkeypatch.setattr(_aoe.ast, "parse", _counting)
        kwargs = {} if _src_tree is None else {"_src_tree": _src_tree}
        _new, ok = EX._add_guard(src, op, symbol, **kwargs)
        assert ok, "guard op must succeed for the parse-count pin"
        return sum(1 for _t, _m in seen if _m is None)

    def test_function_body_parses_source_once(self, monkeypatch):
        src = "def f(x):\n    return x\n"
        op = {"statement": "if x is None:\n    return None"}
        assert self._source_parse_count(monkeypatch, src, op, "f") == 1

    def test_for_loop_parses_source_once(self, monkeypatch):
        src = "def f(xs):\n    for x in xs:\n        print(x)\n"
        op = {"statement": "if x is None:\n    continue", "insert_scope": "for_loop", "loop_variable": "x"}
        assert self._source_parse_count(monkeypatch, src, op, "f") == 1

    def test_while_loop_parses_source_once(self, monkeypatch):
        src = "def f(done):\n    while running:\n        print(1)\n"
        op = {"statement": "if done:\n    break", "insert_scope": "while_loop"}
        assert self._source_parse_count(monkeypatch, src, op, "f") == 1

    def test_name_safety_deferred_path_parses_source_once(self, monkeypatch):
        # Deferred insertion still exercises _find_safe_insertion_point →
        # _compute_name_safety_info, which must reuse the shared tree.
        src = "def f():\n    y = compute()\n    return y\n"
        op = {"statement": "if y:\n    return"}
        assert self._source_parse_count(monkeypatch, src, op, "f") == 1

    def test_preseeded_tree_skips_all_source_parses(self, monkeypatch):
        # A caller-provided tree must be reused: 0 source parses, same result.
        src = "def f(x):\n    return x\n"
        tree = ast.parse(src)
        op = {"statement": "if x is None:\n    return None"}
        assert self._source_parse_count(monkeypatch, src, op, "f", _src_tree=tree) == 0

    def test_guard_already_present_src_tree_parity(self):
        # Pre-parsed tree gives identical verdicts to the self-parse path.
        src = "def f(x):\n    if x is None:\n        return None\n    return x\n"
        stmt = "if x is None:\n    return None"
        tree = ast.parse(src)
        assert _guard_already_present(src, stmt, "f") is True
        assert _guard_already_present(src, stmt, "f", _src_tree=tree) is True
        absent = "if x < 0:\n    raise ValueError()"
        assert _guard_already_present(src, absent, "f") is False
        assert _guard_already_present(src, absent, "f", _src_tree=tree) is False


# ── add_guard diagnostics (B1 exception surfacing + F1 precise failure reasons) ──


class TestAddGuardDiagnostics:
    """B1: ``_add_guard`` must surface the real exception (op['_error'] prefixed
    'add_guard:') instead of swallowing it into a silent 'no match found'.

    F1: ``_insert_at_loop_body`` / ``_insert_at_function_body`` report the
    precise failure reason via op['_error'] so the LLM can recover (supply
    loop_variable/loop_iterable_src) instead of a blind fallback.
    """

    def test_b1_exception_surfaces_not_silent(self, monkeypatch):
        # Force an exception inside the try block; the real reason must reach
        # op["_error"], not the generic "no match found" fallback in apply().
        def _boom(*a, **kw):
            raise RuntimeError("internal kaboom")
        monkeypatch.setattr(EX, "_find_func_node", _boom)
        op = {"statement": "if x:\n    return"}
        _new, ok = EX._add_guard("def f():\n    pass\n", op, "f")
        assert ok is False
        assert op["_error"] == "add_guard: internal kaboom"

    def test_f1_no_loop_reports_precise_reason(self):
        # function with no for-loop → distinct "no for-loop found" message
        src = "def f():\n    x = 1\n    return x\n"
        op = {"statement": "if x:\n    continue", "insert_scope": "for_loop", "loop_variable": "x"}
        _new, ok = EX._add_guard(src, op, "f")
        assert ok is False
        assert "no for-loop found" in op["_error"]

    def test_f1_ambiguous_loops_reports_disambiguation(self):
        # ≥2 matching loops → distinct "N for-loops match … disambiguate" message
        src = "def f():\n    for x in a:\n        pass\n    for x in b:\n        pass\n"
        op = {"statement": "if x:\n    continue", "insert_scope": "for_loop", "loop_variable": "x"}
        _new, ok = EX._add_guard(src, op, "f")
        assert ok is False
        assert "for-loops match" in op["_error"]
        assert "disambiguate" in op["_error"]

    def test_f1_loop_name_safety_forward_ref_reports_reason(self):
        # guard references a name defined AFTER the loop → name-safety violation
        src = "def f():\n    for x in xs:\n        print(x)\n    later = 1\n"
        op = {"statement": "if later:\n    continue", "insert_scope": "for_loop", "loop_variable": "x"}
        _new, ok = EX._add_guard(src, op, "f")
        assert ok is False
        assert "name-safety violation" in op["_error"]

    def test_f1_function_body_undefined_name_reports_reason(self):
        # function_body scope: undefined guard name → "no safe insertion point"
        src = "def f():\n    return 1\n"
        op = {"statement": "if undefined_name:\n    return"}
        _new, ok = EX._add_guard(src, op, "f")
        assert ok is False
        assert "no safe insertion point" in op["_error"]
# ── delete_stmt ─────────────────────────────────────────────────────────────


class TestDeleteStmt:
    def test_scoped_deletion(self):
        src = "def f():\n    DEBUG = True\n    return 1\ndef g():\n    DEBUG = True\n    return 2\n"
        new, ok = EX._delete_stmt(src, {"pattern": "DEBUG"}, "f")
        assert ok
        # only f's DEBUG removed
        assert new.count("DEBUG") == 1

    def test_loud_failure_symbol_not_found(self):
        op = {"pattern": "DEBUG"}
        src = "def f():\n    pass\n"
        new, ok = EX._delete_stmt(src, op, "nope")
        assert ok is False and new == src
        assert "_error" in op and "not found" in op["_error"]

    def test_file_wide_deletion(self):
        src = "a = 1\nb = 2\n"
        new, ok = EX._delete_stmt(src, {"pattern": "a = 1"}, "")
        assert ok and "a = 1" not in new and "b = 2" in new

    def test_nothing_removed_returns_false(self):
        src = "def f():\n    x = 1\n"
        new, ok = EX._delete_stmt(src, {"pattern": "zzz"}, "f")
        assert ok is False and new == src

    def test_empty_pattern_returns_false(self):
        assert EX._delete_stmt("x = 1\n", {"pattern": ""}, "")[1] is False


# ── list_append / list_remove ───────────────────────────────────────────────


class TestResolveModuleLevelList:
    def test_module_level_assignment_found(self):
        src = '__all__ = ["a"]\n'
        lines, node = EX._resolve_module_level_list(src, "__all__")
        assert node is not None and node.value.elts[0].value == "a"
        assert lines[0] == '__all__ = ["a"]\n'

    def test_nested_only_returns_none(self):
        # Module-level contract: a same-named list inside a function must not
        # be treated as the target (shared front-end of _list_append/_list_remove).
        src = 'def f():\n    __all__ = ["a"]\n'
        assert EX._resolve_module_level_list(src, "__all__") is None

    def test_unparsable_source_returns_none(self):
        assert EX._resolve_module_level_list("def f(:\n", "__all__") is None

    def test_missing_name_returns_none(self):
        assert EX._resolve_module_level_list('x = 1\n', "__all__") is None
class TestListAppend:
    def test_inline_list_append(self):
        src = '__all__ = ["a", "b"]\n'
        new, ok = EX._list_append(src, {"list_name": "__all__", "value": "c"})
        assert ok and '"c"' in new

    def test_multiline_list_append_no_trailing_comma(self):
        # Multiline list whose last element has NO trailing comma: the previous
        # element gets a comma appended so elements stay separate (no implicit
        # string concatenation), and the new element lands on its own line.
        src = '__all__ = [\n    "a",\n    "b"\n]\n'
        new, ok = EX._list_append(src, {"list_name": "__all__", "value": "c"})
        assert ok
        tree = ast.parse(new)
        all_node = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__all__"
        )
        elts = [e.value for e in all_node.value.elts]
        assert elts == ["a", "b", "c"]  # no "bc" implicit concatenation

    def test_multiline_list_append_trailing_comma(self):
        # Regression: previously produced '"c",,' (double comma) → ok=False.
        src = '__all__ = [\n    "a",\n    "b",\n]\n'
        new, ok = EX._list_append(src, {"list_name": "__all__", "value": "c"})
        assert ok and '"c"' in new and ",," not in new
        elts = [
            e.value for e in ast.parse(new).body[0].value.elts
        ]
        assert elts == ["a", "b", "c"]

    def test_empty_inline_list_append(self):
        # Regression: previously produced '[, "z"]' (leading comma) → ok=False.
        src = '__all__ = []\n'
        new, ok = EX._list_append(src, {"list_name": "__all__", "value": "z"})
        assert ok and '"z"' in new and "[," not in new
        elts = [e.value for e in ast.parse(new).body[0].value.elts]
        assert elts == ["z"]

    def test_empty_tuple_append_keeps_tuple(self):
        # Empty tuple () needs a trailing comma to remain a single-element tuple
        # (() → ("z",)); without it the assignment collapses to a plain string.
        src = '__all__ = ()\n'
        new, ok = EX._list_append(src, {"list_name": "__all__", "value": "z"})
        assert ok
        val = ast.parse(new).body[0].value
        assert isinstance(val, ast.Tuple)  # not a bare Constant
        assert [e.value for e in val.elts] == ["z"]

    def test_idempotent(self):
        src = '__all__ = ["a"]\n'
        new, ok = EX._list_append(src, {"list_name": "__all__", "value": "a"})
        assert ok and new == src

    def test_list_not_found_returns_false(self):
        assert EX._list_append('x = 1\n', {"list_name": "__all__", "value": "z"})[1] is False

    def test_missing_args_returns_false(self):
        assert EX._list_append('__all__ = []\n', {"list_name": "", "value": "z"})[1] is False

    def test_nested_scope_list_ignored(self):
        # Regression: ast.walk descended into function bodies, so a nested
        # `__all__` was treated as the module-level target — the op mutated
        # the local list and reported success though the documented
        # "module-level" contract was unmet. Only tree.body is scanned now.
        src = 'def f():\n    __all__ = ["a"]\n    return __all__\n'
        new, ok = EX._list_append(src, {"list_name": "__all__", "value": "z"})
        assert ok is False and new == src


class TestListRemove:
    def test_multiline_own_line_removal(self):
        src = '__all__ = [\n    "a",\n    "b",\n]\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "a"})
        assert ok and '"a"' not in new and '"b"' in new

    def test_inline_removal(self):
        src = '__all__ = ["a", "b", "c"]\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "b"})
        assert ok and '"b"' not in new and '"a"' in new and '"c"' in new

    def test_inline_tuple_removal_keeps_tuple(self):
        # Regression: removing one element of a 2-element inline tuple left
        # ("a") which parses as the bare string "a" — the assignment silently
        # changed type from tuple to str. Must stay a 1-tuple ("a",).
        src = '__all__ = ("a", "b")\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "b"})
        assert ok
        val = ast.parse(new).body[0].value
        assert isinstance(val, ast.Tuple)  # not a bare Constant
        assert [e.value for e in val.elts] == ["a"]

    def test_inline_tuple_remove_first_keeps_tuple(self):
        # Same collapse when removing the FIRST element of a 2-tuple.
        src = '__all__ = ("a", "b")\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "a"})
        assert ok
        val = ast.parse(new).body[0].value
        assert isinstance(val, ast.Tuple)
        assert [e.value for e in val.elts] == ["b"]

    def test_inline_tuple_three_elements_stays_tuple(self):
        # Control: a 3-element tuple reduced to 2 keeps its tuple kind without
        # any comma fix-up (2 elements need no trailing comma).
        src = '__all__ = ("a", "b", "c")\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "b"})
        assert ok
        val = ast.parse(new).body[0].value
        assert isinstance(val, ast.Tuple)
        assert [e.value for e in val.elts] == ["a", "c"]

    def test_single_inline_tuple_removal_to_empty(self):
        # Control: ("a",) → () stays a tuple (trailing-comma pattern matches).
        src = '__all__ = ("a",)\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "a"})
        assert ok
        val = ast.parse(new).body[0].value
        assert isinstance(val, ast.Tuple) and val.elts == []

    def test_single_inline_list_removal_to_empty(self):
        # Regression: removing the sole element of an inline list ["a"]
        # returned ok=False (no-op) because no comma adjoined the element.
        # Should produce an empty list [], matching the multiline path.
        src = '__all__ = ["a"]\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "a"})
        assert ok
        val = ast.parse(new).body[0].value
        assert isinstance(val, ast.List) and val.elts == []

    def test_idempotent_not_present(self):
        src = '__all__ = ["a"]\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "zzz"})
        assert ok and new == src

    def test_missing_args_returns_false(self):
        assert EX._list_remove('__all__ = ["a"]\n', {"list_name": "", "value": "a"})[1] is False

    def test_nested_scope_list_ignored(self):
        # Same module-level contract as _list_append: a nested `__all__` must
        # not be treated as the target (ast.walk regression).
        src = 'def f():\n    __all__ = ["a"]\n    return __all__\n'
        new, ok = EX._list_remove(src, {"list_name": "__all__", "value": "a"})
        assert ok is False and new == src
    class TestRewriteModuleLevelListContract:
        """Direct tests of the shared _list_append/_list_remove skeleton."""

        def test_rewrite_none_rolls_back(self):
            src = '__all__ = ["a"]\n'
            new, ok = EX._rewrite_module_level_list(
                src, {"list_name": "__all__", "value": "z"},
                lambda lines, node, value: None,
            )
            assert ok is False and new == src

        def test_rewrite_noop_is_idempotent_success(self):
            src = '__all__ = ["a"]\n'
            new, ok = EX._rewrite_module_level_list(
                src, {"list_name": "__all__", "value": "z"},
                lambda lines, node, value: (lines, False),
            )
            assert ok is True and new == src

        def test_rewrite_apply_validates(self):
            src = '__all__ = ["a"]\n'
            new, ok = EX._rewrite_module_level_list(
                src, {"list_name": "__all__", "value": "z"},
                lambda lines, node, value: ([lines[0].rstrip() + ', "z"\n'], True),
            )
            assert ok is True and '"z"' in new

        def test_rewrite_invalid_lines_roll_back(self):
            src = '__all__ = ["a"]\n'
            new, ok = EX._rewrite_module_level_list(
                src, {"list_name": "__all__", "value": "z"},
                lambda lines, node, value: (["def f(:\n"], True),
            )
            assert ok is False and new == src

        def test_rewrite_callback_exception_rolls_back(self):
            src = '__all__ = ["a"]\n'

            def boom(lines, node, value):
                raise TypeError("boom")

            new, ok = EX._rewrite_module_level_list(
                src, {"list_name": "__all__", "value": "z"}, boom,
            )
            assert ok is False and new == src

        def test_rewrite_missing_op_keys_roll_back(self):
            src = '__all__ = ["a"]\n'
            new, ok = EX._rewrite_module_level_list(
                src, {"list_name": "", "value": "z"},
                lambda lines, node, value: (lines, True),
            )
            assert ok is False and new == src
