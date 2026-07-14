"""Cross-symbol score propagation via call graph.

Solves: when A → B → C and the edit is in C, A alone scores low.
By propagating callee scores upstream, A's score reflects that it
leads to the actual edit target.

Two modes:
1. propagate_scores(): adjust scores using callee dataflow scores
2. expand_candidates(): discover additional edit targets along call chains
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Decay factor: how much of callee score propagates to caller
_CALLEE_DECAY = 0.35
# Mutating callees (db.save, session.commit, etc.) carry stronger propagation —
# a function that calls a write API is more relevant to state-change requests.
_CALLEE_DECAY_MUTATING = 0.52

# Maximum chain depth for propagation
_MAX_DEPTH = 2


@dataclass
class PropagatedScore:
    """Score with propagation context."""
    symbol: str
    path: str
    base_score: float
    propagated_score: float
    chain: list[str]  # symbols in the propagation chain
    reason: str


def propagate_scores(
    symbol_scores: dict[str, tuple[float, str]],
    graph_context: Optional[dict[str, Any]],
    *,
    request: str = "",
    extract_and_score_fn: Any = None,
) -> dict[str, PropagatedScore]:
    """Propagate callee dataflow scores to callers.

    For each symbol with a low base score, checks if any of its callees
    have a high score. If so, boosts the caller's score proportionally.

    Args:
        symbol_scores: {symbol_name: (score, reason)} from individual scoring
        graph_context: graph dict with "callees" and "callers" keys
        request: original edit request (for scoring new candidates)
        extract_and_score_fn: callable(symbol, path) -> (score, reason)
            for scoring callees not in the initial set

    Returns:
        {symbol_name: PropagatedScore} with adjusted scores
    """
    if not graph_context or not isinstance(graph_context, dict):
        return {
            sym: PropagatedScore(
                symbol=sym, path="", base_score=score,
                propagated_score=score, chain=[], reason=reason,
            )
            for sym, (score, reason) in symbol_scores.items()
        }

    callees_map = graph_context.get("callees", {})
    if not isinstance(callees_map, dict):
        callees_map = {}

    results: dict[str, PropagatedScore] = {}

    # Snapshot keys to avoid mutation-during-iteration
    initial_symbols = list(symbol_scores.keys())

    for sym in initial_symbols:
        base_score, base_reason = symbol_scores[sym]
        # Find callees of this symbol
        callee_infos = _get_callees(sym, callees_map)

        if not callee_infos:
            results[sym] = PropagatedScore(
                symbol=sym, path="", base_score=base_score,
                propagated_score=base_score, chain=[], reason=base_reason,
            )
            continue

        # Check callee scores — use known scores or score on demand
        best_callee_score = 0.0
        best_chain: list[str] = []
        best_chain_mutating = False

        for callee_sym, callee_file, callee_mutating in callee_infos:
            if callee_sym in symbol_scores:
                # Callee already scored
                callee_score = symbol_scores[callee_sym][0]
            elif extract_and_score_fn:
                # Score on demand
                try:
                    callee_score, _ = extract_and_score_fn(callee_sym, callee_file)
                except Exception:
                    callee_score = 0.5

                # Cache the score for other callers
                symbol_scores[callee_sym] = (callee_score, "on_demand")
            else:
                continue

            if callee_score > best_callee_score:
                best_callee_score = callee_score
                best_chain = [callee_sym]
                best_chain_mutating = callee_mutating

                # Check one more level deep (callee of callee)
                if best_callee_score > 0.15:
                    deeper_callees = _get_callees(callee_sym, callees_map)
                    for deep_sym, deep_file, deep_mutating in deeper_callees[:2]:
                        if deep_sym in symbol_scores:
                            deep_score = symbol_scores[deep_sym][0]
                        elif extract_and_score_fn:
                            try:
                                deep_score, _ = extract_and_score_fn(deep_sym, deep_file)
                            except Exception:
                                deep_score = 0.5
                            symbol_scores[deep_sym] = (deep_score, "on_demand")
                        else:
                            continue

                        if deep_score > best_callee_score:
                            best_callee_score = deep_score
                            best_chain = [callee_sym, deep_sym]
                            best_chain_mutating = callee_mutating or deep_mutating

        # Propagate: caller gets boost from callee.
        # Mutating callees (db.save, session.commit, etc.) carry stronger
        # propagation — functions calling write APIs are more relevant to
        # state-change requests than functions calling pure reads.
        decay = _CALLEE_DECAY_MUTATING if best_chain_mutating else _CALLEE_DECAY
        boost = best_callee_score * decay
        propagated = base_score + boost

        # Cap at 1.0
        propagated = min(propagated, 1.0)

        mutating_tag = "[mut]" if best_chain_mutating else ""
        chain_str = " → ".join(best_chain) if best_chain else ""
        reason = base_reason
        if boost > 0.01:
            reason += f" | +callee{mutating_tag}:{boost:.2f}({chain_str})"
            logger.debug(
                "[GRAPH_PROP] %s: %.3f → %.3f (callee chain: %s%s, boost=%.3f)",
                sym, base_score, propagated, chain_str, mutating_tag, boost,
            )

        results[sym] = PropagatedScore(
            symbol=sym, path="", base_score=base_score,
            propagated_score=propagated, chain=best_chain,
            reason=reason,
        )

    return results


def expand_candidates(
    current_symbols: set[str],
    graph_context: Optional[dict[str, Any]],
    target_files: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """Discover additional edit candidates along call chains.

    For each current symbol, follows callee edges to find functions
    that might be the actual edit target (delegated edit).

    Args:
        current_symbols: symbols already in the candidate set
        graph_context: graph dict with "callees" key
        target_files: if set, only return candidates in these files

    Returns:
        List of (symbol, file) pairs not already in current_symbols.
    """
    if not graph_context or not isinstance(graph_context, dict):
        return []

    callees_map = graph_context.get("callees", {})
    if not isinstance(callees_map, dict):
        return []

    target_set = set(target_files) if target_files else None
    expanded: list[tuple[str, str]] = []
    visited: set[str] = set(current_symbols)

    # BFS expansion along callee edges, max _MAX_DEPTH
    frontier = list(current_symbols)
    for _depth in range(_MAX_DEPTH):
        next_frontier = []
        for sym in frontier:
            for callee_sym, callee_file, _mut in _get_callees(sym, callees_map):
                if callee_sym in visited:
                    continue
                visited.add(callee_sym)

                # File filter
                if target_set and callee_file not in target_set:
                    continue

                expanded.append((callee_sym, callee_file))
                next_frontier.append(callee_sym)

        frontier = next_frontier
        if not frontier:
            break

    return expanded


def _get_callees(
    symbol: str,
    callees_map: dict[str, Any],
) -> list[tuple[str, str, bool]]:
    """Extract (callee_symbol, callee_file, is_mutating) triples from callees_map."""
    callee_list = callees_map.get(symbol, [])
    if not callee_list:
        # Try bare name
        bare = symbol.split(".")[-1] if "." in symbol else symbol
        callee_list = callees_map.get(bare, [])

    result: list[tuple[str, str, bool]] = []
    for c in callee_list:
        if isinstance(c, dict):
            c_sym = c.get("symbol", "")
            c_file = c.get("file", "")
            c_mutating = bool(c.get("is_mutating", False))
            if c_sym and c_file:
                result.append((c_sym, c_file, c_mutating))
    return result
