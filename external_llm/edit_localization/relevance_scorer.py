"""Edit relevance scoring — code-centric multi-factor scoring.

Scores how relevant a symbol is to an edit request based on its structural
role, not keyword overlap. Three scoring dimensions:

1. Direct mention (0.30): request references values/names in the function
2. Flow relevance (0.40): function's dataflow graph connects to requested change
3. Role signal (0.30): structural role (delegation vs value-determination)

Design principle: code-centric, not request-centric.
  - Extract what the code does → check if it matches what the request asks.
  - NOT: extract tokens from request → search in code.
"""

from __future__ import annotations

import logging

from external_llm.agent.rag_configs import CodeTokenizer

from .dataflow_extractor import SymbolFlowFacts, resolve_alias_root

# Shared tokenizer instance (module-level singleton) — replaces ad-hoc
# ``_tokenize_request`` regex + ``_STOP_WORDS`` frozenset.
_TOKENIZER = CodeTokenizer()

logger = logging.getLogger(__name__)

# Weight allocation — 3 dimensions (semantic match dimension removed)
_W_DIRECT = 0.30
_W_FLOW = 0.40
_W_ROLE = 0.30


def score_edit_relevance(
    request: str,
    facts: SymbolFlowFacts,
) -> tuple[float, str]:
    """Score how relevant a symbol is for the given edit request.

    Args:
        request: The edit request text.
        facts: Extracted dataflow facts for the symbol.

    Returns:
        (score, reason) where score is 0.0..1.0 and reason explains the score.
    """
    req_lower = request.lower()
    req_tokens = _tokenize_request(req_lower)

    # Dimension 1: Direct mention — request references function's values/names
    direct_score, direct_hits = _score_direct_mention(req_lower, req_tokens, facts)

    # Dimension 2: Flow relevance — function's dataflow relates to request
    flow_score, flow_reason = _score_flow_relevance(req_tokens, facts)

    # Dimension 3: Role signal — structural role appropriateness
    role_score, role_reason = _score_role_signal(facts)

    final = (
        _W_DIRECT * direct_score
        + _W_FLOW * flow_score
        + _W_ROLE * role_score
    )

    reason_parts = []
    if direct_hits:
        reason_parts.append(f"direct:{direct_score:.2f}({','.join(list(direct_hits)[:3])})")
    else:
        reason_parts.append(f"direct:{direct_score:.2f}")
    reason_parts.append(f"flow:{flow_score:.2f}({flow_reason})")
    reason_parts.append(f"role:{role_score:.2f}({role_reason})")

    reason = " | ".join(reason_parts)

    logger.debug(
        "[EDIT_LOC] score=%.3f [direct=%.2f flow=%.2f role=%.2f] %s",
        final, direct_score, flow_score, role_score, reason,
    )

    return final, reason


def _score_direct_mention(
    req_lower: str,
    req_tokens: set[str],
    facts: SymbolFlowFacts,
) -> tuple[float, set[str]]:
    """Score based on direct mention of function's values in the request.

    Checks: string literals, assigned variable names, constructor fields,
    attribute writes — all the "things this function touches".

    Object identity: also checks call_sites literal args + alias chains.
    e.g. u2 = u1; get_user(1) + "user 1" in request → identity hit.
    """
    # Pool of matchable tokens from the function
    code_tokens: set[str] = set()

    # String literals (highest signal — these are concrete values)
    for lit in facts.string_literals:
        code_tokens.add(lit.lower())

    # Assigned variable names
    for name in facts.assigned_names:
        code_tokens.add(name.lower())

    # Constructor field names
    for _cls, fields in facts.constructor_calls.items():
        for f in fields:
            code_tokens.add(f.lower())

    # Attribute writes
    for attr in facts.attribute_writes:
        code_tokens.add(attr.lower())

    # ── Object identity via call_sites ────────────────────────────────
    # Literal positional args recorded at each call site.
    # e.g. get_user(1) → call_sites["get_user"] = [["1"]]
    # If the request mentions "1" (or "user 1"), this is an identity hit.
    identity_hits: set[str] = set()
    for _callee, arg_sets in facts.call_sites.items():
        for arg_list in arg_sets:
            for raw_arg in arg_list:
                # raw_arg is repr(value): '1', "'admin'", etc.
                # Strip Python repr wrapping to get the plain value
                plain = raw_arg.strip("'\"")
                if len(plain) < 1:
                    continue
                if plain in req_lower or plain in req_tokens:
                    identity_hits.add(plain)

    if not code_tokens and not identity_hits:
        return 0.0, set()

    # Match: how many code tokens appear in the request?
    hits: set[str] = set()
    for token in code_tokens:
        if len(token) < 2:
            continue
        if token in req_lower:
            hits.add(token)
        # Also check if request tokens overlap (handles morphological variants)
        elif token in req_tokens:
            hits.add(token)

    # Identity hits: scored alongside regular hits but capped separately
    # to prevent a single-arg call site from dominating
    hits |= identity_hits

    if not hits:
        return 0.0, set()

    denom = max(len(code_tokens) + (1 if identity_hits else 0), 1)
    score = min(len(hits) / denom, 1.0)
    return score, hits


def _score_flow_relevance(
    req_tokens: set[str],
    facts: SymbolFlowFacts,
) -> tuple[float, str]:
    """Score based on dataflow graph connection to the request.

    Key insight: even if the request doesn't mention a variable directly,
    if it mentions something in the derivation chain, the function is relevant.

    Example:
        Request: "kind unification"
        derives_from = {kind: {is_async}, is_async: {node}}
        → "kind" mentioned → all upstream (is_async, node) are relevant
        → high flow score
    """
    if not facts.derives_from and not facts.constructor_calls:
        # No dataflow to analyze
        if facts.delegation_calls and not facts.assigned_names:
            return 0.0, "no_flow+delegation"
        return 0.2, "no_flow_data"

    # Build reachability: which tokens are in the derivation graph?
    flow_tokens: set[str] = set()

    # All variables in derivation chains
    for target, sources in facts.derives_from.items():
        flow_tokens.add(target.lower())
        for src in sources:
            flow_tokens.add(src.lower())

    # Constructor fields are flow endpoints
    for cls_name, fields in facts.constructor_calls.items():
        flow_tokens.add(cls_name.lower())
        for f in fields:
            flow_tokens.add(f.lower())

    # Return variables are flow endpoints
    for name in facts.return_names:
        flow_tokens.add(name.lower())

    if not flow_tokens:
        return 0.2, "empty_flow"

    # How many request tokens connect to the flow graph?
    flow_hits = req_tokens & flow_tokens
    if not flow_hits:
        # Substring check for compound tokens (e.g., "is_async" ↔ "async")
        for rt in req_tokens:
            if len(rt) < 3:
                continue
            for ft in flow_tokens:
                if rt in ft or ft in rt:
                    flow_hits.add(ft)

    if not flow_hits:
        return 0.1, "no_flow_hit"

    # Score: proportion of flow graph that connects to request
    score = min(len(flow_hits) / max(len(flow_tokens), 1) * 2.0, 1.0)

    # Bonus: if a derivation chain target is mentioned, extra relevance
    chain_target_hits = 0
    for target in facts.derives_from:
        if target.lower() in req_tokens:
            chain_target_hits += 1
        # Substring match for derived variables
        for rt in req_tokens:
            if len(rt) >= 3 and (rt in target.lower() or target.lower() in rt):
                chain_target_hits += 1
                break

    if chain_target_hits > 0:
        score = min(score + 0.2 * chain_target_hits, 1.0)

    # ── Object identity: call-site callee / alias-resolved attr writes ────
    # Checked as an additive bonus AFTER main ratio, to avoid inflating the
    # denominator and penalizing functions where callee names don't match.
    identity_bonus = 0.0

    # Callee name bonus: request references the function this code calls
    #e.g. "get_user result correction" + call_sites["get_user"] → strong identity signal
    for callee in facts.call_sites:
        if callee.lower() in req_tokens:
            identity_bonus += 0.15
            break  # one hit is enough

    # Alias-chain attr-write bonus: attribute writes through an alias are
    # attributed to the alias-root (original object).
    # e.g. u2 = u1; u2.email = x → if u1 is in flow graph, "email" is relevant.
    if identity_bonus == 0.0 and facts.alias_chains and facts.attribute_writes:
        for _alias, original in facts.alias_chains.items():
            root = resolve_alias_root(original, facts.alias_chains)
            if root.lower() in flow_tokens or original.lower() in flow_tokens:
                for attr in facts.attribute_writes:
                    if attr.lower() in req_tokens:
                        identity_bonus += 0.10
                        break
                if identity_bonus > 0:
                    break

    score = min(score + identity_bonus, 1.0)
    reason = f"{len(flow_hits)}_hits"
    if identity_bonus > 0:
        reason += f"+identity:{identity_bonus:.2f}"
    return score, reason


def _score_role_signal(facts: SymbolFlowFacts) -> tuple[float, str]:
    """Score based on the function's structural role.

    Pure delegation functions score low.
    Value-determining functions score high.
    """
    tags = facts.tags

    if "pure_delegation" in tags:
        return 0.1, "pure_delegation"

    if "pass_through" in tags:
        return 0.15, "pass_through"

    score = 0.5  # baseline for non-trivial functions
    reasons = []

    if "value_determiner" in tags:
        score += 0.25
        reasons.append("value_det")

    if "conditional_logic" in tags:
        score += 0.15
        reasons.append("conditional")

    if "field_constructor" in tags:
        score += 0.1
        reasons.append("field_ctor")

    if "collection_builder" in tags:
        score += 0.1
        reasons.append("collection")

    return min(score, 1.0), "+".join(reasons) if reasons else "baseline"

def _tokenize_request(request: str) -> set[str]:
    """Tokenize request into meaningful tokens for matching.

    Delegates to ``CodeTokenizer`` (``rag_configs``) — the single shared
    tokenizer that handles CamelCase, snake_case, Korean, and stop-word
    filtering consistently across the codebase.

    Replaces the previous inline regex + ``_STOP_WORDS`` frozenset approach.
    """
    return set(_TOKENIZER.tokenize(request))
