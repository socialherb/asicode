"""Container reachability scanner — detects structurally unreachable keys in
class-level and module-level dict literals.

B1 scope (this version):
  - Collects private dict literals at class-level and module-level
  - Collects read sites: ``.get(var, ...)`` and ``[var]`` access patterns
  - Infers key domain for simple single-constant-return method chains
  - Emits three reachability grades per key:
      structurally_unreachable — domain fully known, key absent
      possibly_unreachable     — domain partially known or heuristic
      reachable                — key found in inferred domain
  - Single-file, intra-class scope only (conservative; no false positives from
    cross-file dynamics)

The scanner is *evidence-first*: it always emits read_sites and key_domain so
downstream LLM (FixSpec / summarize) can reason even when the structural verdict
is only "possibly_unreachable".  Hard ``structurally_unreachable`` judgment is
reserved for cases where intra-class constant-domain propagation is complete.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from external_llm.agent.config.thresholds import config as _cfg

from . import parse_cache

logger = logging.getLogger(__name__)


# ── Candidate model ────────────────────────────────────────────────────────────

@dataclass
class ContainerReadSite:
    """One call site that reads from a container."""
    access_kind: str        # "get" | "subscript"
    key_expr_kind: str      # "literal" | "name" | "call" | "other"
    key_expr_text: str      # e.g. "intent" or '"general"'
    in_method: str          # enclosing method name (empty = module-level)
    lineno: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_kind": self.access_kind,
            "key_expr_kind": self.key_expr_kind,
            "key_expr_text": self.key_expr_text,
            "in_method": self.in_method,
            "lineno": self.lineno,
        }


@dataclass
class ContainerReachabilityCandidate:
    """Reachability analysis result for one dict literal."""
    file: str
    container_symbol: str   # bare name, e.g. ``_INTENT_STRATEGIES``
    qualified_name: str     # qualified, e.g. ``AgentLoop._INTENT_STRATEGIES``
    enclosing_class: Optional[str]
    container_kind: str     # always "dict_literal" in this version
    lineno: int
    end_lineno: int
    all_keys: list[str]
    keys_unreachable: list[str]          # structurally_unreachable
    keys_possibly_unreachable: list[str]
    keys_reachable: list[str]
    read_sites: list[dict[str, Any]]
    key_domain: Optional[list[str]]      # inferred domain; None = unknown
    confidence: float
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "container_symbol": self.container_symbol,
            "qualified_name": self.qualified_name,
            "enclosing_class": self.enclosing_class,
            "container_kind": self.container_kind,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "all_keys": list(self.all_keys),
            "keys_unreachable": list(self.keys_unreachable),
            "keys_possibly_unreachable": list(self.keys_possibly_unreachable),
            "keys_reachable": list(self.keys_reachable),
            "read_sites": list(self.read_sites),
            "key_domain": list(self.key_domain) if self.key_domain is not None else None,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "reason": _reachability_reason(self),
        }


def _reachability_reason(cand: "ContainerReachabilityCandidate") -> str:
    """Human-readable one-line summary consumed by the SCAN tool's ``reason``.

    The SCAN handler's candidate→line renderer looks up ``description`` then
    ``reason`` (see ``_tool_run_structural_scan``); without this key, container
    candidates rendered as a bare dash (``file:line symbol — ``).  Summarise the
    actionable keys so the LLM can triage without opening the file.
    """
    total = len(cand.all_keys)
    parts: list[str] = []
    if cand.keys_unreachable:
        n = len(cand.keys_unreachable)
        sample = ", ".join(repr(k) for k in cand.keys_unreachable[:3])
        more = f" (+{n - 3} more)" if n > 3 else ""
        of_total = f" of {total}" if total else ""
        parts.append(f"{n} structurally-unreachable key(s){of_total}: {sample}{more}")
    if cand.keys_possibly_unreachable:
        parts.append(f"{len(cand.keys_possibly_unreachable)} possibly-unreachable key(s)")
    if not parts:
        parts.append(f"all {total} key(s) reachable" if total else "no keys")
    return "; ".join(parts)
# ── AST helpers ───────────────────────────────────────────────────────────────

def _is_private_name(name: str) -> bool:
    """True for ``_foo`` but not ``__foo__`` dunders."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _extract_string_keys(dict_node: ast.Dict) -> Optional[list[str]]:
    """Extract all keys from a Dict node if every key is a string constant.

    Returns None when any key is not a string constant (e.g. integer keys,
    variable keys, **-unpacking) — we skip those containers entirely.
    """
    keys: list[str] = []
    for k in dict_node.keys:
        if k is None:
            return None  # **-unpacking present
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.append(k.value)
        else:
            return None  # non-string or non-literal key
    return keys


def _build_lineno_to_method(tree: ast.Module) -> dict[int, str]:
    """Map every source line to the innermost enclosing function/method name."""
    result: dict[int, str] = {}

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def _visit_func(self, node: ast.FunctionDef) -> None:
            self._stack.append(node.name)
            qualified_name = ".".join(self._stack)
            end = getattr(node, "end_lineno", node.lineno)
            for ln in range(node.lineno, end + 1):
                result[ln] = qualified_name
            self.generic_visit(node)
            self._stack.pop()

        visit_FunctionDef = _visit_func
        visit_AsyncFunctionDef = _visit_func

    _Visitor().visit(tree)
    return result


def _classify_key_expr(node: ast.expr) -> tuple:
    """Return (kind, text) describing a key expression node.

    Kinds:
      literal   — string constant literal (e.g. "foo")
      name      — local variable (e.g. intent)
      attr_const — self.ATTR or cls.ATTR where ATTR may be a class constant
      call      — arbitrary call expression
      other     — anything else
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ("literal", repr(node.value))
    if isinstance(node, ast.Name):
        return ("name", node.id)
    # self.ATTR / cls.ATTR — may be an enum-like class constant
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in ("self", "cls")
    ):
        return ("attr_const", node.attr)
    if isinstance(node, ast.Call):
        try:
            return ("call", ast.unparse(node))
        except Exception:
            return ("call", "call(...)")
    try:
        return ("other", ast.unparse(node))
    except Exception:
        return ("other", "expr")


def _is_container_ref(
    node: ast.expr,
    container_name: str,
    enclosing_class: Optional[str],
) -> bool:
    """True when *node* references the container.

    Matches three access spellings:
      - bare name        — ``_FOO`` (module-level containers)
      - self/cls attr    — ``self._FOO`` / ``cls._FOO``
      - class-name attr  — ``MyClass._FOO``
    """
    if isinstance(node, ast.Name):
        return node.id == container_name
    if isinstance(node, ast.Attribute) and node.attr == container_name:
        if isinstance(node.value, ast.Name):
            return node.value.id in ("self", "cls") or node.value.id == enclosing_class
    return False


def _collect_dict_literals(
    tree: ast.Module,
    cross_file_referenced_names: Optional[set] = None,
) -> list[tuple]:
    """Collect class-level and module-level dict literals for reachability scan.

    With graph-backed ``cross_file_referenced_names``: includes non-private dict
    literals whose names are NOT in the set (i.e., not externally referenced).

    Without graph data (conservative): only private (``_``-prefixed) dicts.

    Returns list of (name, enclosing_class, lineno, end_lineno, keys).
    Only includes dicts where every key is a string constant.
    """
    results = []

    def _scan_body(body: list, enclosing_class: Optional[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                _scan_body(list(node.body), enclosing_class=node.name)
                continue
            # Assign: foo = {...}
            if isinstance(node, ast.Assign):
                value = node.value
                for tgt in node.targets:
                    if not isinstance(tgt, ast.Name):
                        continue
                    name = tgt.id
                    # Cross-file referenced names are skipped regardless of
                    # privacy — private dicts get imported across modules too
                    # (e.g. `from .edit_prompts import _USER_PROMPT_BUILDERS`).
                    if cross_file_referenced_names and name in cross_file_referenced_names:
                        continue
                    if not _is_private_name(name):
                        if cross_file_referenced_names is None:
                            continue  # conservative: private only
                    if not isinstance(value, ast.Dict):
                        continue
                    keys = _extract_string_keys(value)
                    if keys is None:
                        continue
                    end = getattr(node, "end_lineno", node.lineno)
                    results.append((name, enclosing_class, node.lineno, end, keys))
            # AnnAssign: foo: Dict[...] = {...}
            elif isinstance(node, ast.AnnAssign):
                if not isinstance(node.target, ast.Name):
                    continue
                name = node.target.id
                if cross_file_referenced_names and name in cross_file_referenced_names:
                    continue  # externally referenced (incl. imported private names)
                if not _is_private_name(name):
                    if cross_file_referenced_names is None:
                        continue  # conservative: private only
                if node.value is None or not isinstance(node.value, ast.Dict):
                    continue
                keys = _extract_string_keys(node.value)
                if keys is None:
                    continue
                end = getattr(node, "end_lineno", node.lineno)
                results.append((name, enclosing_class, node.lineno, end, keys))

    _scan_body(list(tree.body), enclosing_class=None)
    return results


def _collect_read_sites(
    tree: ast.Module,
    container_name: str,
    lineno_to_method: dict[int, str],
    enclosing_class: Optional[str] = None,
) -> tuple:
    """Find all read sites for *container_name* in the tree.

    Returns ``(sites, has_dynamic_use)``.  ``has_dynamic_use`` is True when the
    container is referenced outside the keyed ``get``/subscript patterns —
    iteration, ``in`` containment, ``.keys()/.values()/.items()``, being passed
    as an argument, returned, aliased, etc.  Any such use means every key may
    be consumed dynamically, so per-key reachability verdicts are unsound.
    """
    sites: list[ContainerReadSite] = []
    consumed: set = set()  # id() of container-ref nodes matched by keyed patterns

    for node in ast.walk(tree):
        ln = getattr(node, "lineno", 0)

        # Pattern A: <container>.get(key, default)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_container_ref(node.func.value, container_name, enclosing_class)
            and len(node.args) >= 1
        ):
            consumed.add(id(node.func.value))
            key_kind, key_text = _classify_key_expr(node.args[0])
            sites.append(ContainerReadSite(
                access_kind="get",
                key_expr_kind=key_kind,
                key_expr_text=key_text,
                in_method=lineno_to_method.get(ln, ""),
                lineno=ln,
            ))

        # Pattern B: <container>[key]
        elif isinstance(node, ast.Subscript) and _is_container_ref(
            node.value, container_name, enclosing_class
        ):
            consumed.add(id(node.value))
            slice_node = node.slice
            if isinstance(slice_node, ast.Index):  # Python 3.8 compat
                slice_node = slice_node.value  # type: ignore[attr-defined]
            key_kind, key_text = _classify_key_expr(slice_node)
            sites.append(ContainerReadSite(
                access_kind="subscript",
                key_expr_kind=key_kind,
                key_expr_text=key_text,
                in_method=lineno_to_method.get(ln, ""),
                lineno=ln,
            ))

    has_dynamic_use = False
    for node in ast.walk(tree):
        if not _is_container_ref(node, container_name, enclosing_class):
            continue
        if id(node) in consumed:
            continue
        # The definition / rebind target itself (Store context) is not a read.
        if isinstance(getattr(node, "ctx", None), ast.Store):
            continue
        has_dynamic_use = True
        break

    return sites, has_dynamic_use


def _collect_constant_returns(func_node: ast.FunctionDef) -> Optional[set[str]]:
    """If every ``return`` in func_node yields a string constant, return that set.

    Returns None when any return is non-constant or there are no returns.
    Bare ``return`` (no value) is treated as non-constant.
    """
    constants: set[str] = set()
    found_return = False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return):
            found_return = True
            if node.value is None:
                return None  # bare return
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                constants.add(node.value.value)
            else:
                return None  # non-constant return
    return constants if found_return else None


def _find_method_in_class(
    class_node: ast.ClassDef,
    method_name: str,
) -> Optional[ast.FunctionDef]:
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == method_name:
                return node  # type: ignore[return-value]
    return None


def _resolve_class_constant(class_node: ast.ClassDef, attr_name: str) -> Optional[str]:
    """Return the string value of a class-level constant for *attr_name*, or None.

    Only resolves simple single-level assignments at class body scope:
      ``TYPE_A = "type_a"`` or ``TYPE_A: str = "type_a"``.
    Returns None for non-string values or missing attributes.
    """
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == attr_name:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == attr_name:
                if (
                    node.value is not None
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
    return None


def _find_class_node(tree: ast.Module, class_name: str) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _binds_name(target: ast.expr, var_name: str) -> bool:
    """True if an assignment/loop target binds *var_name* (incl. tuple unpack)."""
    if isinstance(target, ast.Name):
        return target.id == var_name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_binds_name(e, var_name) for e in target.elts)
    if isinstance(target, ast.Starred):
        return _binds_name(target.value, var_name)
    return False


def _domain_of_rhs(
    rhs: ast.expr,
    tree: ast.Module,
    enclosing_class: Optional[str],
) -> Optional[set[str]]:
    """Domain of one assignment RHS, or None when not statically determinable."""
    # Case 1: direct string constant
    if isinstance(rhs, ast.Constant) and isinstance(rhs.value, str):
        return {rhs.value}

    # Case 2: self._helper(...) call — trace the method
    if (
        isinstance(rhs, ast.Call)
        and isinstance(rhs.func, ast.Attribute)
        and isinstance(rhs.func.value, ast.Name)
        and rhs.func.value.id in ("self", "cls")
        and enclosing_class is not None
    ):
        class_node = _find_class_node(tree, enclosing_class)
        if class_node is not None:
            helper_method = _find_method_in_class(class_node, rhs.func.attr)
            if helper_method is not None:
                return _collect_constant_returns(helper_method)

    return None


def _infer_key_domain_for_var(
    tree: ast.Module,
    method_node: ast.FunctionDef,
    var_name: str,
    enclosing_class: Optional[str],
    target_lineno: int,
) -> Optional[set[str]]:
    """Try to determine the string value domain for ``var_name`` in ``method_node``.

    The domain is the UNION of every reaching assignment (lineno <=
    target_lineno) — branch assignments like ``if c: var = "a" else: var = "b"``
    each contribute.  Picking only the lexically-last assignment would drop the
    earlier branch and misreport its key as unreachable.

    Returns None (unknown) when:
      - ``var_name`` is a parameter of the method (callers control the value)
      - any reaching assignment has a non-constant RHS
      - the name is bound via tuple unpack, for-target, walrus, or AugAssign
      - there is no reaching assignment at all
    """
    args = method_node.args
    param_names = {a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)}
    if args.vararg:
        param_names.add(args.vararg.arg)
    if args.kwarg:
        param_names.add(args.kwarg.arg)
    if var_name in param_names:
        return None

    domains: list[set[str]] = []
    for stmt in ast.walk(method_node):
        if isinstance(stmt, ast.Assign):
            simple_hit = any(
                isinstance(t, ast.Name) and t.id == var_name for t in stmt.targets
            )
            complex_hit = any(
                _binds_name(t, var_name)
                for t in stmt.targets
                if not isinstance(t, ast.Name)
            )
            if complex_hit:
                return None  # tuple unpack etc. — value unknown
            if simple_hit and stmt.lineno <= target_lineno:
                d = _domain_of_rhs(stmt.value, tree, enclosing_class)
                if d is None:
                    return None
                domains.append(d)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == var_name:
                if stmt.value is None:
                    continue
                if stmt.lineno <= target_lineno:
                    d = _domain_of_rhs(stmt.value, tree, enclosing_class)
                    if d is None:
                        return None
                    domains.append(d)
        elif isinstance(stmt, ast.AugAssign):
            if _binds_name(stmt.target, var_name):
                return None
        elif isinstance(stmt, ast.NamedExpr):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == var_name:
                return None
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            if _binds_name(stmt.target, var_name):
                return None

    if not domains:
        return None
    out: set[str] = set()
    for d in domains:
        out |= d
    return out


def _compute_reachability(
    all_keys: list[str],
    read_sites: list[ContainerReadSite],
    tree: ast.Module,
    enclosing_class: str | None,
    lineno_to_method: dict[int, str],
    method_nodes: dict[str, ast.FunctionDef],
) -> tuple:
    """Return (keys_unreachable, keys_possibly_unreachable, keys_reachable, domain, evidence).

    Reachability strategy:
      1. Collect all key expressions from read sites.
      2. For literal keys: directly mark as reachable.
      3. For name-type keys: infer domain via _infer_key_domain_for_var.
      4. Classify remaining keys.
    """
    evidence: list[str] = []
    known_reachable: set[str] = set()
    full_domain: set[str] | None = None
    partial_domain: set[str] = set()
    domain_fully_known = False

    # Collect domains from all read sites
    inferred_domains: list[set[str] | None] = []
    for site in read_sites:
        if site.key_expr_kind == "literal":
            raw = site.key_expr_text.strip("'\"")
            known_reachable.add(raw)
            inferred_domains.append({raw})
        elif site.key_expr_kind == "name":
            var = site.key_expr_text
            method_name = site.in_method
            method_node = method_nodes.get(method_name)
            if method_node is None:
                inferred_domains.append(None)
                evidence.append(f"key_domain:unknown_method:{method_name}")
                continue
            domain = _infer_key_domain_for_var(
                tree, method_node, var, enclosing_class, site.lineno
            )
            inferred_domains.append(domain)
            if domain is not None:
                evidence.append(
                    f"key_domain:{var}:{{{','.join(sorted(domain))}}}"
                )
            else:
                evidence.append(f"key_domain:{var}:unknown")
        elif site.key_expr_kind == "attr_const":
            attr_name = site.key_expr_text
            if enclosing_class is not None:
                class_node = _find_class_node(tree, enclosing_class)
                if class_node is not None:
                    const_val = _resolve_class_constant(class_node, attr_name)
                    if const_val is not None:
                        inferred_domains.append({const_val})
                        evidence.append(
                            f"key_domain:class_const:{attr_name}={const_val!r}"
                        )
                        continue
            inferred_domains.append(None)
            evidence.append(f"key_domain:attr_const:unresolved:{attr_name}")
        else:
            # call or other — can't determine domain
            inferred_domains.append(None)
            evidence.append(f"key_domain:dynamic_expr:{site.key_expr_text[:40]}")

    # Aggregate domains across all sites
    if all(d is not None for d in inferred_domains) and inferred_domains:
        aggregated: set[str] = set()
        for d in inferred_domains:
            aggregated |= d  # type: ignore[operator]
        full_domain = aggregated
        domain_fully_known = True
        evidence.append(f"key_domain:fully_determined:{{{','.join(sorted(full_domain))}}}")
    else:
        for d in inferred_domains:
            if d is not None:
                partial_domain |= d
        if partial_domain:
            evidence.append(
                f"key_domain:partial:{{{','.join(sorted(partial_domain))}}}"
            )

    # Classify keys
    keys_unreachable: list[str] = []
    keys_possibly_unreachable: list[str] = []
    keys_reachable: list[str] = []

    # A read site whose key domain could not be determined (e.g. `.get(var)`
    # where var's domain is unknown) may read ANY key — per-key "possibly
    # unreachable" verdicts are unsound there, same rationale as the
    # has_dynamic_use skip.  Sample verification (2026-06-12) found 6/8 of
    # such verdicts were noise.  No read sites at all is different: the dict
    # is genuinely never read in this file, so possibly_unreachable stands.
    has_unknown_domain_site = any(d is None for d in inferred_domains)
    if has_unknown_domain_site:
        evidence.append("unknown_domain_site:possibly_unreachable_suppressed")

    for key in all_keys:
        if domain_fully_known:
            if full_domain and key in full_domain:
                keys_reachable.append(key)
            else:
                keys_unreachable.append(key)
        elif key in known_reachable or key in partial_domain:
            keys_reachable.append(key)
        elif has_unknown_domain_site:
            keys_reachable.append(key)
        else:
            keys_possibly_unreachable.append(key)

    domain_out = sorted(full_domain) if full_domain is not None else None
    return keys_unreachable, keys_possibly_unreachable, keys_reachable, domain_out, evidence


# ── Public scan API ───────────────────────────────────────────────────────────

def scan_container_reachability(
    *,
    repo_root: str,
    file_paths: list[str],
    max_per_file: int = _cfg.counts.SCANNER_CONTAINER_REACH_MAX,
    min_unreachable_keys: int = 1,
    cross_file_referenced_names: Optional[set] = None,
) -> list[ContainerReachabilityCandidate]:
    """Scan Python files for dict literals with structurally unreachable keys.

    Args:
        repo_root: Repository root (used to resolve relative paths).
        file_paths: List of ``.py`` file paths to scan.
        max_per_file: Max candidates emitted per file.
        min_unreachable_keys: Minimum unreachable key count to emit a candidate.
            Set to 0 to also emit evidence-only (all keys reachable) candidates.
        cross_file_referenced_names: Names that have cross-file references.
            When provided, non-private dicts NOT in this set are also scanned.
            Without this, only private (``_``-prefixed) dicts are analyzed.

    Returns:
        List of ContainerReachabilityCandidate.  Files that fail to parse are
        silently skipped — this scanner is supplementary signal, never blocking.
    """
    candidates: list[ContainerReachabilityCandidate] = []
    _truncated_total = 0  # containers dropped by max_per_file

    for rel_path in file_paths or []:
        abs_path = (
            rel_path if os.path.isabs(rel_path)
            else os.path.join(repo_root or "", rel_path)
        )
        tree = parse_cache.parse_ast(abs_path)
        if tree is None:
            continue

        lineno_to_method = _build_lineno_to_method(tree)

        # Build method_nodes map: qualified_name → FunctionDef
        method_nodes: dict[str, ast.FunctionDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual_name = lineno_to_method.get(node.lineno, node.name)
                method_nodes[qual_name] = node

        dict_literals = _collect_dict_literals(tree, cross_file_referenced_names)

        per_file_count = 0
        for (sym_name, enclosing_class, lineno, end_lineno, keys) in dict_literals:
            if per_file_count >= max_per_file:
                _truncated_total += len(dict_literals) - per_file_count
                logger.warning(
                    "[CONTAINER_REACH] %s: hit max_per_file=%d, truncating %d remaining container(s)",
                    rel_path, max_per_file, len(dict_literals) - per_file_count,
                )
                break
            if not keys:
                continue

            qualified = (
                f"{enclosing_class}.{sym_name}" if enclosing_class else sym_name
            )

            read_sites, has_dynamic_use = _collect_read_sites(
                tree, sym_name, lineno_to_method, enclosing_class,
            )
            if has_dynamic_use:
                # Iteration / containment / whole-dict pass-through — every key
                # may be consumed dynamically; per-key verdicts would be unsound.
                logger.debug(
                    "[CONTAINER_REACH] %s: %s has dynamic (non-keyed) use — skipping",
                    rel_path, qualified,
                )
                continue

            (
                keys_unreachable,
                keys_possibly_unreachable,
                keys_reachable,
                domain_out,
                reach_evidence,
            ) = _compute_reachability(
                keys, read_sites, tree, enclosing_class,
                lineno_to_method, method_nodes,
            )

            # Skip if nothing interesting (caller can lower min_unreachable_keys=0)
            total_actionable = len(keys_unreachable) + len(keys_possibly_unreachable)
            if total_actionable < min_unreachable_keys and min_unreachable_keys > 0:
                continue

            # Confidence: high when domain fully determined, lower otherwise
            if keys_unreachable:
                confidence = 0.90
            elif keys_possibly_unreachable:
                confidence = 0.55
            else:
                confidence = 0.30  # evidence-only, all reachable

            evidence: list[str] = [
                f"container:dict_literal:{qualified}",
                f"keys_total:{len(keys)}",
                f"read_sites:{len(read_sites)}",
            ]
            if keys_unreachable:
                evidence.append(
                    f"structurally_unreachable:{{{','.join(keys_unreachable)}}}"
                )
            evidence.extend(reach_evidence)

            candidates.append(ContainerReachabilityCandidate(
                file=rel_path,
                container_symbol=sym_name,
                qualified_name=qualified,
                enclosing_class=enclosing_class,
                container_kind="dict_literal",
                lineno=lineno,
                end_lineno=end_lineno,
                all_keys=list(keys),
                keys_unreachable=keys_unreachable,
                keys_possibly_unreachable=keys_possibly_unreachable,
                keys_reachable=keys_reachable,
                read_sites=[s.to_dict() for s in read_sites],
                key_domain=domain_out,
                confidence=confidence,
                evidence=evidence,
            ))
            per_file_count += 1

    if candidates:
        unreachable_count = sum(len(c.keys_unreachable) for c in candidates)
        logger.info(
            "[CONTAINER_REACH] %d container(s) across %d file(s); "
            "structurally_unreachable keys=%d",
            len(candidates),
            len(set(c.file for c in candidates)),
            unreachable_count,
        )

    if _truncated_total:
        # Function attribute consumed by ScannerRegistry.run() (reset via
        # `del` before each invocation).
        scan_container_reachability._truncated = _truncated_total
    return candidates
