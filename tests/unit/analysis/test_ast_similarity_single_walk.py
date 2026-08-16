"""Single-walk feature extraction: fused walk ≡ the four former ast.walk passes.

``_walk_function_features`` replaced four separate traversals per function
(``_collect_calls`` / ``_exit_shapes`` / ``_extract_result_keys`` over the
body; ``_ident_tokens`` over the whole node).  These tests re-derive the old
semantics with straightforward ``ast.walk`` reference implementations and pin
the fused walk to them — especially the scope boundary: ident tokens cover the
signature (decorators/defaults/annotations) while body-only features do not.
"""
from __future__ import annotations

import ast
import textwrap

from external_llm.analysis.ast_similarity_scanner import (
    _BUILTINS,
    _call_shape,
    _walk_function_features,
    normalise_function,
)

# Exercises every scope boundary at once: decorator call, default-value call,
# annotation names, nested def, yield/raise, result.get/result[...] subscripts,
# keyword args (incl. **kwargs with arg=None), and short strings (<3 chars).
_FIXTURE = '''
@decorator_factory(timeout=30)
def process(result, queue, *, retries=make_default(), **kwargs):
    """Docstring with tokens."""
    data = result.get("alpha") or {}
    beta = result["beta"]
    for item in queue.iter(timeout=1):
        if item is None:
            continue
        yield transform(item, mode="fast")
    raise ValueError("no;pe")

    def nested_helper(x):
        return json.dumps(x)
'''


def _first_func(src: str) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(src))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))


# ── Reference implementations (the pre-fusion semantics) ─────────────────────

def _ref_collect_calls(body: list[ast.stmt]) -> list[str]:
    shapes = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            s = _call_shape(node)
            if s:
                shapes.append(s)
    return shapes


def _ref_exit_shapes(body: list[ast.stmt]) -> list[str]:
    exits = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Return):
            exits.append("return")
        elif isinstance(node, ast.Raise):
            exits.append("raise")
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            exits.append("yield")
    return list(dict.fromkeys(exits))


def _ref_result_keys(body: list[ast.stmt], var_name: str = "result") -> list[str]:
    keys: set = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == var_name
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == var_name
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
    return sorted(keys)


def _ref_ident_tokens(func_node: ast.FunctionDef) -> list[str]:
    args = func_node.args
    params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    toks: set = set()
    for n in ast.walk(func_node):
        if isinstance(n, ast.Attribute) and n.attr not in _BUILTINS:
            toks.add(f"attr:{n.attr}")
        elif isinstance(n, ast.Name):
            if n.id in params or n.id in _BUILTINS or n.id in ("self", "cls"):
                continue
            toks.add(f"name:{n.id}")
        elif (isinstance(n, ast.Constant)
                and isinstance(n.value, str) and len(n.value) >= 3):
            toks.add(f"str:{n.value[:30]}")
        elif isinstance(n, ast.keyword) and n.arg:
            toks.add(f"kw:{n.arg}")
    return sorted(toks)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fused_walk_matches_reference_on_boundary_fixture():
    fn = _first_func(_FIXTURE)
    calls, exits, result_keys, idents = _walk_function_features(fn)

    assert sorted(calls) == sorted(_ref_collect_calls(fn.body))
    assert sorted(exits) == sorted(_ref_exit_shapes(fn.body))
    assert result_keys == _ref_result_keys(fn.body)
    assert idents == _ref_ident_tokens(fn)


def test_signature_calls_are_not_body_calls():
    """decorator_factory(...) / make_default() live outside the body."""
    fn = _first_func(_FIXTURE)
    calls, _, _, idents = _walk_function_features(fn)

    joined = " ".join(calls)
    assert "decorator_factory" not in joined
    # builtin receiver/name shapes: make_default() → CALL.local
    assert calls.count("CALL.local") >= 2  # transform + nested json.dumps… body calls
    assert "name:decorator_factory" in idents
    assert "name:make_default" in idents


def test_result_keys_capture_get_and_subscript():
    _, _, result_keys, _ = _walk_function_features(_first_func(_FIXTURE))
    assert result_keys == ["alpha", "beta"]


def test_exits_include_yield_and_raise():
    _, exits, _, _ = _walk_function_features(_first_func(_FIXTURE))
    # nested_helper's `return` is inside the body subtree → collected too
    assert exits == ["raise", "return", "yield"]


def test_ident_tokens_exclude_params_and_short_strings():
    _, _, _, idents = _walk_function_features(_first_func(_FIXTURE))
    # parameter names are measured separately (param_role_similarity)
    assert not any(t in ("name:result", "name:queue", "name:retries") for t in idents)
    # **kwargs contributes no kw: token (arg is None) but stays out of names too
    assert "kw:kwargs" not in idents
    # strings shorter than 3 chars ("no;pe" is 5 → in; the 1-char timeout=1 is int)
    assert "str:no;pe" in idents
    # nested-def body identifiers are part of the domain tokens
    assert "name:json" in idents and "attr:dumps" in idents


def test_normalise_function_wires_fused_features(tmp_path):
    fn = _first_func(_FIXTURE)
    n = normalise_function(fn, "pkg.process")
    assert n.qualname == "pkg.process"
    assert sorted(n.call_shapes) == sorted(_ref_collect_calls(fn.body))
    assert sorted(n.exit_shapes) == sorted(_ref_exit_shapes(fn.body))
    assert n.result_keys == _ref_result_keys(fn.body)
    assert n.ident_tokens == _ref_ident_tokens(fn)
    assert n.try_present is False
    assert n.line_count >= 10
