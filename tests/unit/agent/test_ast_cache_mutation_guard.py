"""Static guard: detect mutation of shared ``ast_cache.parse_cached`` trees.

``parse_cached`` hands out **shared references** (an ``lru_cache`` of
``ast.Module``). Any caller that mutates the returned tree — via an
``ast.NodeTransformer``, ``fix_missing_locations``, ``increment_lineno``, or
direct node-attribute writes — poisons every subsequent cache hit, producing
heisenbugs that are extremely hard to trace.

This test statically scans the codebase for the dangerous pattern so a future
violation fails CI instead of silently corrupting the parse cache.

Precision model
---------------
Only ``ast.NodeTransformer`` (which rewrites nodes in place) is unsafe on a
shared cached tree. ``ast.NodeVisitor`` is read-only — walking a cached tree
is the *intended* use, so it is NOT flagged. The guard resolves the receiver
class of each ``.visit()`` / ``.generic_visit()`` call and flags the call
only when:

  * the argument is a **bare** (non-``deepcopy``-wrapped) ``parse_cached()``
    result (or a variable holding one), AND
  * the receiver is a **``NodeTransformer`` subclass** (resolved from local
    class definitions; an imported/unknown receiver is conservatively left
    unflagged, matching the local-transformer risk profile).

``fix_missing_locations`` / ``increment_lineno`` mutate unconditionally and
are always flagged on a cached argument regardless of receiver.

Variable tracking is **per-scope** (module or function body, not descending
into nested functions) so a common name like ``tree`` in one function does
not false-trigger on an unrelated ``.visit(tree)`` elsewhere.

Today there are ZERO violations: every local ``NodeTransformer`` operates on
a ``copy.deepcopy()`` result or a direct ``ast.parse()``, and ``parse_cached``
results feed only read-only ``NodeVisitor`` walkers. This test keeps it that way.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_CACHED_PARSE_NAMES = frozenset(
    {"parse_cached", "parse_cached_optional", "parse_expr_cached_optional"}
)
_MUTATING_METHODS = frozenset({"visit", "generic_visit"})
_MUTATING_FREEFUNCS = frozenset({"fix_missing_locations", "increment_lineno"})
_TRANSFORMER_BASES = frozenset({"NodeTransformer"})
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCAN_ROOTS = [_PROJECT_ROOT / "external_llm"]

# Documented allowlist of (relpath, lineno) sites where a cached var feeds a
# mutating primitive INTENTIONALLY with a PROVEN read-only receiver. Empty today.
_INTENTIONAL_ALLOWLIST: set[tuple[str, int]] = set()


def _is_cached_parse_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name) and f.id in _CACHED_PARSE_NAMES:
        return True
    if isinstance(f, ast.Attribute) and f.attr in _CACHED_PARSE_NAMES:
        return True
    return False


def _walk_no_descend(node: ast.AST):
    """Yield descendants of ``node`` WITHOUT entering nested scopes."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, _SCOPE_NODES):
            yield from _walk_no_descend(child)


def _collect_transformer_classes(tree: ast.Module) -> set[str]:
    """Names of ClassDefs whose bases include ``ast.NodeTransformer``."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for b in node.bases:
            nm = b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else None)
            if nm in _TRANSFORMER_BASES:
                out.add(node.name)
    return out


def _ctor_class_name(call: ast.Call) -> str | None:
    """Class name for a constructor call ``ClassName(...)`` / ``mod.ClassName(...)``."""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _scope_locals(scope: ast.AST) -> tuple[dict[str, int], dict[str, str]]:
    """Return (bare_cached_vars, instance_vars) for ``scope``'s OWN body.

    ``bare_cached_vars``: var → lineno for ``var = parse_cached(...)`` (the
    ``deepcopy(parse_cached(...))`` form is excluded because ``node.value`` is
    the ``deepcopy`` call, not a bare cached parse, so the target holds a copy).
    ``instance_vars``: var → classname for ``var = ClassName(...)``.
    """
    bare: dict[str, int] = {}
    inst: dict[str, str] = {}
    for stmt in getattr(scope, "body", []):
        if isinstance(stmt, _SCOPE_NODES):
            continue  # nested scope — has its own _iter_scopes() entry
        for node in (stmt, *_walk_no_descend(stmt)):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if _is_cached_parse_call(node.value):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        bare.setdefault(tgt.id, node.lineno)
            else:
                cn = _ctor_class_name(node.value)
                if cn:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            inst.setdefault(tgt.id, cn)
    return bare, inst


def _iter_scopes(tree: ast.Module):
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _receiver_class(func_value: ast.AST, instance_vars: dict[str, str]) -> str | None:
    """Resolve the class name of a ``.visit()`` receiver expression."""
    if isinstance(func_value, ast.Call):
        return _ctor_class_name(func_value)  # inline ClassName().visit(...)
    if isinstance(func_value, ast.Name):
        return instance_vars.get(func_value.id)  # var.visit(...)
    return None


def _arg_is_cached(arg: ast.AST, bare: dict[str, int]) -> bool:
    if isinstance(arg, ast.Name) and arg.id in bare:
        return True
    return isinstance(arg, ast.Call) and _is_cached_parse_call(arg)


def _scan_source(src: str, rel: str) -> list[tuple[str, int, str]]:
    """Pure analysis: return mutation violations in ``src``.

    ``rel`` is the display path embedded in violation tuples. Factored out of
    :func:`_scan_path` so the detection logic is directly unit-testable with
    crafted source strings instead of temp files — this is what the parametrized
    precision tests below exercise, protecting the scanner itself from a vacuous
    regression (e.g. ``return []`` would keep ``test_no_mutation_of_cached_ast``
    green while detection silently vanished).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    transformers = _collect_transformer_classes(tree)
    violations: list[tuple[str, int, str]] = []
    for scope in _iter_scopes(tree):
        bare, instance_vars = _scope_locals(scope)
        for stmt in getattr(scope, "body", []):
            if isinstance(stmt, _SCOPE_NODES):
                continue  # nested scope — has its own _iter_scopes() entry
            for node in (stmt, *_walk_no_descend(stmt)):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in _MUTATING_METHODS and node.args:
                    # visit()/generic_visit(): dangerous only on a NodeTransformer.
                    recv_cls = _receiver_class(func.value, instance_vars)
                    if recv_cls not in transformers:
                        continue  # read-only NodeVisitor, or unknown → safe-by-default
                    arg = node.args[0]
                    if _arg_is_cached(arg, bare):
                        violations.append((rel, node.lineno, f"{func.attr}() on cached tree via NodeTransformer '{recv_cls}'"))
                else:
                    # fix_missing_locations / increment_lineno — mutate unconditionally.
                    # Match both bare (imported) and ast.<name> attribute forms.
                    fname = None
                    if isinstance(func, ast.Name):
                        fname = func.id
                    elif isinstance(func, ast.Attribute):
                        fname = func.attr
                    if fname in _MUTATING_FREEFUNCS and node.args:
                        arg = node.args[0]
                        if _arg_is_cached(arg, bare):
                            violations.append((rel, node.lineno, f"{fname}() mutates cached tree"))
    return violations


def _scan_path(path: pathlib.Path) -> list[tuple[str, int, str]]:
    src = path.read_text()
    rel = str(path.relative_to(_PROJECT_ROOT))
    return _scan_source(src, rel)


def _collect_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in _SCAN_ROOTS:
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return sorted(out)


def test_no_mutation_of_cached_ast() -> None:
    """Fail if a NodeTransformer (or fix_missing_locations) mutates a shared cached tree."""
    all_v: list[tuple[str, int, str]] = []
    for f in _collect_files():
        all_v.extend(_scan_path(f))
    real = [v for v in all_v if (v[0], v[1]) not in _INTENTIONAL_ALLOWLIST]
    assert not real, (
        "ast_cache.parse_cached hands out SHARED references — mutating one poisons "
        "every cache hit. These sites feed a cached tree into a NodeTransformer or "
        "fix_missing_locations/increment_lineno. Fix by either (a) copy.deepcopy() "
        "before transforming, (b) calling ast.parse() directly instead of "
        "parse_cached, or (c) if the receiver is provably read-only, add "
        "(relpath, lineno) to _INTENTIONAL_ALLOWLIST with a justifying comment.\n  "
        + "\n  ".join(f"{f}:{ln} — {why}" for f, ln, why in real)
    )


# --- Scanner precision: protect the GUARD itself from a vacuous regression. ---
# ``test_no_mutation_of_cached_ast`` only asserts real code = 0 violations, so a
# future weakening of ``_scan_source`` (e.g. dropping NodeTransformer resolution)
# would stay green. These crafted snippets encode the documented precision model
# and FAIL if detection drifts: dangerous patterns must flag, safe ones must not.
_PRECISION_DANGEROUS = [
    pytest.param(
        "import ast\nclass X(ast.NodeTransformer): ...\n"
        "def f():\n    tree = parse_cached(\"x\")\n    X().visit(tree)\n",
        id="transformer-on-var",
    ),
    pytest.param(
        "import ast\nclass X(ast.NodeTransformer): ...\n"
        "def f():\n    X().visit(parse_cached(\"x\"))\n",
        id="transformer-inline",
    ),
    pytest.param(
        "def f():\n    fix_missing_locations(parse_cached(\"x\"))\n",
        id="fix_missing_locations-bare",
    ),
    pytest.param(
        "import ast\ndef f():\n    ast.fix_missing_locations(parse_cached(\"x\"))\n",
        id="fix_missing_locations-attr",
    ),
    pytest.param(
        "def f():\n    increment_lineno(parse_cached(\"x\"))\n",
        id="increment_lineno",
    ),
]

_PRECISION_SAFE = [
    pytest.param(
        "import ast\nclass V(ast.NodeVisitor): ...\n"
        "def f():\n    V().visit(parse_cached(\"x\"))\n",
        id="nodevisitor-read-only",
    ),
    pytest.param(
        "import ast, copy\nclass X(ast.NodeTransformer): ...\n"
        "def f():\n    tree = copy.deepcopy(parse_cached(\"x\"))\n    X().visit(tree)\n",
        id="deepcopy-then-transform",
    ),
    pytest.param(
        "import ast\nclass X(ast.NodeTransformer): ...\n"
        "def f():\n    X().visit(ast.parse(\"x\"))\n",
        id="direct-ast-parse",
    ),
]


@pytest.mark.parametrize("src", _PRECISION_DANGEROUS)
def test_scan_flags_dangerous_patterns(src: str) -> None:
    """True positive: a mutation-of-shared-cache pattern MUST be detected."""
    violations = _scan_source(src, "synthetic.py")
    assert violations, f"scanner FAILED to flag dangerous pattern:\n{src}"


@pytest.mark.parametrize("src", _PRECISION_SAFE)
def test_scan_passes_safe_patterns(src: str) -> None:
    """True negative: read-only / copied / non-cached patterns MUST NOT flag."""
    violations = _scan_source(src, "synthetic.py")
    assert not violations, f"scanner FALSE-flagged a safe pattern:\n{src}"
