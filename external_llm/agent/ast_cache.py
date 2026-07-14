"""Shared AST parse cache — de-duplicates ``ast.parse()`` within a single
verification flow.

Motivation
----------
A single ``SemanticVerifier.verify_edit()`` call re-parses the same
``content_after`` string ten or more times: ``_validate_ast_structure``,
``_verify_symbol_preservation``, ``_verify_signature_compatibility``,
``_verify_dead_code``, ``_verify_partial_implementation``,
``_verify_incomplete_return_path``, ``_verify_return_shape``,
``_verify_decorator_preservation``, ``_verify_insertion_position_safety``,
etc. Similarly ``rewrite_transaction.verify_rewrite()`` parses
``original_content`` and ``new_content`` four to six times each across
Gates 1–5. On a 6 000-line file each parse is tens of milliseconds — the
redundancy is measurable.

A module-level LRU cache is the simplest fix: callers invoke
``parse_cached(content)`` and any repeat call within cache residency
returns the already-parsed tree.

Mutation contract — READ ONLY
-----------------------------
The cache hands out **shared references**. Any caller that mutates the
returned tree leaks that mutation into every subsequent cache hit.
**Do not feed this cache from code that:**

* uses ``ast.NodeTransformer`` — walks-and-rewrites in place
* calls ``ast.fix_missing_locations`` / ``ast.increment_lineno``
* writes to node attributes (``node.body = ...``, ``node.name = ...``)

Verifier / inspector code that only walks the tree and reads attributes
(``ast.walk``, ``ast.dump``, attribute reads, structural comparisons) is
safe. Transform code should call ``ast.parse()`` directly — the cost is
immaterial relative to the transform itself, and mutation isolation is
worth the extra parse.

Sizing
------
maxsize is intentionally small (see ``_CACHE_MAX``). The workload we're
optimizing is *within-call* redundancy (same content parsed 5–10 times in
one flow), not *across-call* persistence. A few entries suffice for the
2–4 distinct contents in a verification flow, and keeping the cap low
bounds worst-case memory: a parsed AST of a large file can be several MB,
so 16 slots ≈ low-tens of MB in the pathological case.

Stats
-----
``cache_info()`` delegates to ``functools.lru_cache``'s ``cache_info``,
exposing hits / misses / maxsize / currsize. Useful for telemetry and for
validating cache effectiveness in tests.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from typing import Optional

from .config.thresholds import config as _cfg

# Empirically: verify_edit parses (content_before, content_after) plus at
# most 2 body fragments → 4 entries per flow. 16 gives headroom for 3-4
# concurrent flows without noticeable memory growth.
_CACHE_MAX = _cfg.counts.AST_CACHE_MAX


@lru_cache(maxsize=_CACHE_MAX)
def _parse_cached_impl(content: str) -> ast.Module:
    """LRU-cached ``ast.parse``. Raises ``SyntaxError`` on failure."""
    return ast.parse(content)


@lru_cache(maxsize=_CACHE_MAX)
def _parse_expr_cached_impl(content: str) -> ast.Expression:
    """LRU-cached ``ast.parse(content, mode='eval')``.

    Separate cache from ``_parse_cached_impl`` because the return type
    differs (``ast.Expression`` vs ``ast.Module``) and the same string
    could plausibly be passed to both modes.
    """
    return ast.parse(content, mode="eval")


def parse_cached(content: str) -> ast.Module:
    """Return a cached parse of ``content``; raise ``SyntaxError`` on failure.

    **Returned tree MUST NOT be mutated** — see module docstring.
    """
    return _parse_cached_impl(content)


def parse_cached_optional(content: str) -> Optional[ast.Module]:
    """``parse_cached`` that swallows ``SyntaxError`` and returns ``None``.

    Use this at call-sites that already treat a parse failure as "skip
    this verifier" rather than a hard error.
    """
    try:
        return _parse_cached_impl(content)
    except SyntaxError:
        return None


def parse_expr_cached_optional(content: str) -> Optional[ast.Expression]:
    """Return a cached ``mode='eval'`` parse of ``content`` or ``None``.

    Use when the caller needs an ``ast.Expression`` (e.g. small
    annotation / default / condition fragments). Swallows
    ``SyntaxError`` and returns ``None``. **Returned tree MUST NOT be
    mutated** — see module docstring.
    """
    try:
        return _parse_expr_cached_impl(content)
    except SyntaxError:
        return None


def cache_info() -> dict:
    """Return combined ``{hits, misses, maxsize, currsize}`` across both caches.

    Individual caches are exposed as the ``module`` / ``expr`` sub-dicts
    for finer telemetry.
    """
    mod = _parse_cached_impl.cache_info()
    expr = _parse_expr_cached_impl.cache_info()
    return {
        "hits": mod.hits + expr.hits,
        "misses": mod.misses + expr.misses,
        "maxsize": mod.maxsize + expr.maxsize,
        "currsize": mod.currsize + expr.currsize,
        "module": {
            "hits": mod.hits,
            "misses": mod.misses,
            "maxsize": mod.maxsize,
            "currsize": mod.currsize,
        },
        "expr": {
            "hits": expr.hits,
            "misses": expr.misses,
            "maxsize": expr.maxsize,
            "currsize": expr.currsize,
        },
    }


def clear_cache() -> None:
    """Drop all cached parses. Call between unrelated test flows."""
    _parse_cached_impl.cache_clear()
    _parse_expr_cached_impl.cache_clear()
