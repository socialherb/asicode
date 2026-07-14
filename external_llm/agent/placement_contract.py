"""Placement Contract Layer (PCL) — first-class placement constraints.

Placement contracts define WHERE code must be inserted and flow unchanged
through the entire pipeline: Planner -> Developer LLM -> Verifier -> Repair.

Architecture:
  - PlacementContract is the IR (intermediate representation)
  - Planner builds it via builder functions
  - Developer LLM receives it as intent_hint (natural language) + structured JSON
  - Verifier checks it via verify_placement_contract() (AST-based)
  - Repair uses it to guide retries

Design rules:
  1. placement_contract is a MANDATORY rule, not an LLM hint
  2. LLM proposes where; Verifier has final authority
  3. Only deterministic (AST-expressible) checks — no LLM in verification
  4. Edits without a contract skip placement verification entirely

Supported contract kinds:
  - after_anchor: code must appear after specific assignment(s)
  - before_return: code must appear before return statement(s)
  - at_function_entry: code must be at function body start
  - inside_block: code must be inside a specific block (for/while/if)

Future extensible to: insert_call, update_callers, logging, error_handling, etc.
"""
from __future__ import annotations

import ast
import builtins as _builtins_module
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from . import ast_cache

logger = logging.getLogger(__name__)


# Python builtins are never meaningful placement anchors (len, print, range,
# True/False/None, …). Caching the set keeps auto-extraction O(|names|).
_PYTHON_BUILTINS: frozenset = frozenset(dir(_builtins_module))


def extract_module_level_names(file_content: str) -> set[str]:
    """Extract module-level binding names from ``file_content``.

    Captures names that are NOT local data — imports, module-level
    assignments, top-level function/class definitions, and the bindings
    they create when wrapped in ``if`` / ``try`` / ``with`` / ``for`` /
    ``while`` (compound statements that don't create new lexical scope).
    The set is fed to ``extract_read_names`` so the auto-anchor
    extraction does not pick up things like ``JSONResponse`` / ``logger``
    / ``yaml`` (try/except import) / ``ResolvedExecutionSpec``
    (TYPE_CHECKING import) when they appear in a guard payload — they
    are not local dependencies the verifier should chase, and including
    them creates impossible-to-satisfy contracts ("placement must come
    after assignment of JSONResponse" — there is no such assignment).

    "Module scope" here is the textual top-level lexical scope of the
    file PLUS the bodies of any compound statements that don't introduce
    a new scope (``if``, ``try``, ``with``, ``for``, ``while``).
    ``FunctionDef`` / ``AsyncFunctionDef`` / ``ClassDef`` DO introduce a
    new scope; their inner bindings are captured for the def/class name
    itself but the bodies are not entered.  This matches Python's actual
    name resolution: ``try: import yaml; except: yaml = None`` makes
    ``yaml`` a module-level name regardless of which branch ran.

    Returns an empty set on parse failure (safe degradation: builder
    falls back to legacy behaviour).
    """
    if not file_content:
        return set()
    try:
        tree = ast_cache.parse_cached(file_content)
    except Exception:
        return set()  # non-critical — never block execution

    # Single source of the wrapper-aware module-scope walker — see
    # code_structure_utils.iter_module_scope_nodes.  Importing here keeps
    # the bug-fix surface unified across placement_contract, intent_verifier,
    # and code_structure_utils itself (Set 1 / Set 5 / Set 6 root cause).
    from ..code_structure_utils import iter_module_scope_nodes

    names: set = set()
    for sub in iter_module_scope_nodes(tree):
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in getattr(sub, "names", []):
                # ``from a.b import c as d`` → bind ``d``.
                # ``import a.b`` → bind ``a`` (head segment is the
                bound = (
                    getattr(alias, "asname", None)
                    or getattr(alias, "name", "")
                    or ""
                )
                if bound and bound != "*":
                    names.add(bound.split(".")[0])
        elif isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                for nd in ast.walk(tgt):
                    if isinstance(nd, ast.Name):
                        names.add(nd.id)
        elif isinstance(sub, ast.AnnAssign):
            tgt2 = sub.target
            if isinstance(tgt2, ast.Name):
                names.add(tgt2.id)
        elif isinstance(sub, ast.NamedExpr):
            # Module-level walrus: ``if (n := compute()): pass``
            # binds ``n`` at module scope.  Rare but legal.
            if isinstance(sub.target, ast.Name):
                names.add(sub.target.id)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Captured both at top-level and when nested inside a
            # non-scope-creating compound (``if X: def helper(): ...``
            # still binds ``helper`` at module scope).
            names.add(sub.name)

    return names


def extract_read_names(
    target_statement: str,
    *,
    module_level_names: Optional[set[str]] = None,
) -> list[str]:
    """Extract Load-context identifier names from ``target_statement``.

    Returns identifiers the statement READS (not assigns), filtered for
    Python builtins AND (when supplied) module-level names from the
    enclosing source.  Used by placement contract builders to auto-derive
    anchors from what the payload actually uses — LLM-specified
    anchor_names routinely miss secondary reads, so auto-extraction is
    the PRIMARY path (opt-out via ``auto_extract_uses=False``).

    Rules:
      * ``ast.Name`` with ``Load`` ctx → collected.
      * ``ast.Attribute`` chain → its base ``Name`` is collected (dotted
        reads like ``obj.attr.method()`` count as a read of ``obj``).
      * ``ast.AugAssign`` target → collected (``x += y`` reads x too,
        even though AST marks the target as ``Store``).
      * Python builtins are excluded.
      * When ``module_level_names`` is supplied, those names are also
        excluded — imports / module globals / top-level def-class are
        not local data dependencies and should not become anchors.
      * Store-only names (assignment targets, ``for x in …``,
        comprehension loop vars, keyword argument names) are NOT
        collected — ctx-based filtering handles them.

    Comprehension-scoped locals and other names that have no
    function-level assignment will be classified as ``presence="none"``
    at verify time and treated as unfindable (safe), so this function
    deliberately does no scope analysis of its own.

    Returns a deterministic sorted list (empty on parse failure).
    """
    if not target_statement:
        return []
    try:
        tree = ast_cache.parse_cached(target_statement)
    except Exception:
        return []  # non-critical — never block execution
    if not tree.body:
        return []

    names: set = set()
    for stmt in tree.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    names.add(node.id)
            elif isinstance(node, ast.Attribute):
                base = node.value
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name) and isinstance(base.ctx, ast.Load):
                    names.add(base.id)
            elif isinstance(node, ast.AugAssign):
                tgt = node.target
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
                elif isinstance(tgt, ast.Attribute):
                    base = tgt.value
                    while isinstance(base, ast.Attribute):
                        base = base.value
                    if isinstance(base, ast.Name):
                        names.add(base.id)

    _excluded = _PYTHON_BUILTINS | (module_level_names or set())
    return sorted(n for n in names if n not in _excluded)


# ---------------------------------------------------------------------------
# IR: PlacementContract
# ---------------------------------------------------------------------------

class AnchorRole(str, Enum):
    """Declarative role of ``anchor_names`` for a placement contract.

    Different contract kinds use ``anchor_names`` in structurally different
    ways. Historically the verifier inferred the role from the contract
    shape — e.g. ``inside_block`` + ``block_type='for'`` + non-empty
    ``anchor_names`` was implicitly treated as "iter_var anchors must
    match the for-loop iteration target". That implicit inference made
    the contract author's intent unrecoverable from the contract itself
    and caused divergence between builder and verifier.

    A contract now declares the role explicitly. The verifier dispatches
    on the declared role rather than re-inferring from block_type.

    Roles:
      * ``ITER_VAR`` — anchor names must match for-loop iteration targets
        (``for X in ...:`` → ``X``). Used by ``inside_block`` + ``for``.
      * ``UNSPECIFIED`` (empty string) — no structural role; anchors are
        either consumed by a kind-specific verifier (``after_anchor``
        ordering, ``before_return`` ordering) or treated as advisory.

    New roles can be added as additional verification paths come online
    (e.g. ``CONTEXT_BINDING`` for ``with ... as X:``).
    """
    UNSPECIFIED = ""
    ITER_VAR = "iter_var"


# Closed table mapping ``inside_block`` block_type → default anchor_role.
# The builder uses this so callers do not have to repeat the mapping at
# every site; explicit ``anchor_role=`` arguments still override.
_DEFAULT_ANCHOR_ROLE_BY_BLOCK_TYPE: dict[str, str] = {
    "for": AnchorRole.ITER_VAR.value,
}


@dataclass
class PlacementAnchor:
    """A single anchor point that the placement depends on."""
    name: str                    # variable or expression name (e.g. "candidates", "x.attr")
    access_path: str = ""        # dotted path if needed (e.g. "results.data")
    match: str = "assignment"    # "assignment" | "call" | "return" | "raise"
    strength: str = "effective"  # "effective" (skip weak init) | "any"


@dataclass
class PlacementScope:
    """Scope rules for anchor search."""
    mode: str = "nearest_dominating"  # "strict" | "nearest_dominating" | "global"
    allow_nested: bool = True         # search inside if/try/with blocks


@dataclass
class PlacementConstraints:
    """Hard constraints on placement."""
    forbid_before_anchor: bool = True
    forbid_reassignment_before_use: bool = True
    max_distance: Optional[int] = None  # max stmts between anchor and target


@dataclass
class PlacementVerification:
    """How to verify this contract.

    Ordering semantics for multi-anchor contracts (after_anchor,
    before_return-with-anchors): the payload must come after
    max-of-anchors — lexicographic max over ``(frame_depth,
    index_in_frame)`` of each anchor's nearest dominating effective
    definition. See ``build_after_assignment_contract`` for the full
    statement.
    """
    assertion_type: str = "AFTER_ASSIGNMENT"  # maps to verification function
    mode: str = "relaxed"                     # "strict" | "relaxed"
    # P6: Manage block_type as a 1st-class field instead of embedding it in the
    # assertion_type string ("INSIDE_BLOCK:for"). Empty string = unset (non-block contracts).
    block_type: str = ""


@dataclass
class PlacementRepair:
    """How to handle violations."""
    on_violation: str = "force_body_only"   # repair strategy
    escalation: str = "blocking"            # "blocking" | "warning"


@dataclass
class PlacementContract:
    """First-class placement constraint that flows through the entire pipeline.

    This is the canonical representation. All placement-sensitive edits should
    express their WHERE constraint as a PlacementContract.
    """
    kind: str                                      # "after_anchor" | "before_return" | "at_function_entry" | "inside_block" | "top_level"
    anchors: list[PlacementAnchor] = field(default_factory=list)
    scope: PlacementScope = field(default_factory=PlacementScope)
    constraints: PlacementConstraints = field(default_factory=PlacementConstraints)
    intent_hint: str = ""                          # natural language for LLM
    verification: PlacementVerification = field(default_factory=PlacementVerification)
    repair: PlacementRepair = field(default_factory=PlacementRepair)
    # convenience fields for verifier
    target_statement: str = ""                     # the code being placed
    anchor_names: list[str] = field(default_factory=list)  # flat list of anchor names
    # Declarative role for ``anchor_names`` — see ``AnchorRole``. Set by the
    # builder (closed table from contract shape) so the verifier dispatches
    # on the declared role instead of re-inferring from ``kind`` /
    # ``block_type``. Empty string == UNSPECIFIED.
    anchor_role: str = ""
    # Provenance tag — distinguishes who emitted this contract.  Default "llm"
    # keeps LLM-parsed contracts (Step 1 shadow) and ad-hoc test contracts on
    source: str = "llm"                            # "llm" | "deterministic" | "manual"
    # AST-derived shape metadata for the target_statement.  Populated by the
    # planner when building placement candidates so guard recovery and candidate
    statement_kind: str = ""
    control_flow_kind: str = ""
    # Container scope for the inserted symbol.
    # "" = existing behaviour, "module" = must be at module level,
    # "class:ClassName" = must be a member of the named class.
    insertion_container: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for metadata storage."""
        return {
            "kind": self.kind,
            "anchors": [
                {"name": a.name, "access_path": a.access_path,
                 "match": a.match, "strength": a.strength}
                for a in self.anchors
            ],
            "scope": {"mode": self.scope.mode, "allow_nested": self.scope.allow_nested},
            "constraints": {
                "forbid_before_anchor": self.constraints.forbid_before_anchor,
                "forbid_reassignment_before_use": self.constraints.forbid_reassignment_before_use,
                "max_distance": self.constraints.max_distance,
            },
            "intent_hint": self.intent_hint,
            "verification": {
                "assertion_type": self.verification.assertion_type,
                "mode": self.verification.mode,
                "block_type": self.verification.block_type,
            },
            "repair": {
                "on_violation": self.repair.on_violation,
                "escalation": self.repair.escalation,
            },
            "target_statement": self.target_statement,
            "anchor_names": self.anchor_names,
            "anchor_role": self.anchor_role,
            "source": self.source,
            "statement_kind": self.statement_kind,
            "control_flow_kind": self.control_flow_kind,
            "insertion_container": self.insertion_container,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlacementContract":
        """Deserialize from metadata dict.

        ``position`` (int): LLM-requested "insert at position N" signal.
            Preserved in ``intent_hint`` when ``anchor_names`` is absent
            for divergence logger / shadow logging use.
        """
        anchors = [
            PlacementAnchor(**a) for a in (d.get("anchors") or [])
        ]
        scope_d = d.get("scope") or {}
        constraints_d = d.get("constraints") or {}
        verification_d = d.get("verification") or {}
        repair_d = d.get("repair") or {}

        return cls(
            kind=d.get("kind", "after_anchor"),
            anchors=anchors,
            scope=PlacementScope(
                mode=scope_d.get("mode", "nearest_dominating"),
                allow_nested=scope_d.get("allow_nested", True),
            ),
            constraints=PlacementConstraints(
                forbid_before_anchor=constraints_d.get("forbid_before_anchor", True),
                forbid_reassignment_before_use=constraints_d.get("forbid_reassignment_before_use", True),
                max_distance=constraints_d.get("max_distance"),
            ),
            intent_hint=d.get("intent_hint", "") or (
                f"[position={d.get('position')}]" if d.get("position") is not None else ""
            ),
            verification=PlacementVerification(
                assertion_type=verification_d.get("assertion_type", "AFTER_ASSIGNMENT"),
                mode=verification_d.get("mode", "relaxed"),
                block_type=verification_d.get("block_type", ""),
            ),
            repair=PlacementRepair(
                on_violation=repair_d.get("on_violation", "force_body_only"),
                escalation=repair_d.get("escalation", "blocking"),
            ),
            target_statement=d.get("target_statement", ""),
            anchor_names=d.get("anchor_names", []),
            anchor_role=d.get("anchor_role", ""),
            source=d.get("source", "llm"),
            statement_kind=d.get("statement_kind", ""),
            control_flow_kind=d.get("control_flow_kind", ""),
            insertion_container=d.get("insertion_container", ""),
        )

    @property
    def is_blocking(self) -> bool:
        return self.repair.escalation == "blocking"


# ---------------------------------------------------------------------------
# Builders — one per contract family
# ---------------------------------------------------------------------------

def build_after_assignment_contract(
    target_statement: str,
    anchor_names: list[str],
    placement_mode: str = "relaxed",
    *,
    auto_extract_uses: bool = True,
    module_level_names: Optional[set[str]] = None,
) -> PlacementContract:
    """Build contract for code that must appear AFTER anchor assignment(s).

    Generalizes LOCAL_GUARD_AFTER_ASSIGNMENT to any placement-sensitive edit:
    guard_add, insert_call, logging insertion, etc.

    Args:
        target_statement: The code to be placed (e.g. "if not candidates: return None")
        anchor_names: Variable names that must be assigned before the target.
            Callers (planner / LLM) supply what they believe are the
            required anchors; this list is typically incomplete.
        placement_mode: "strict" (immediately after) or "relaxed" (allow intermediate stmts)
        auto_extract_uses: When True (default), the builder parses
            ``target_statement`` and unions its Load-context Names
            (filtered for Python builtins) into ``anchor_names``.

            This is the PRIMARY path, not a safety net — the LLM rarely
            enumerates every read the payload makes, so silent
            ordering bugs are otherwise the norm. Opt out with
            ``auto_extract_uses=False`` only when callers need strictly
            LLM-specified anchors (e.g. contract-level tests that
            inject adversarial inputs).

    Anchor ordering semantics (multi-anchor):
        The verifier admits the edit iff the payload appears after
        ``max(anchor_matches)`` where ``max`` is lexicographic over
        ``(frame_depth, index_in_frame)`` — i.e. the most deeply
        dominating, latest-in-block anchor. For same-frame anchors
        this reduces to "after ``max(lineno)`` of the anchor
        definitions". Multi-anchor contracts therefore REQUIRE
        placement after EVERY anchor's effective definition.

        Example:
            ``x = 1 ; … ; x = 2 ; … ; if x > 0: use()`` →
            insertion must come after line-of(``x = 2``), not just
            after the first ``x = 1``. The nearest-dominating search
            finds ``x = 2`` per anchor, then max selects it.
    """
    # Primary path: LLM-specified ∪ AST-extracted reads. Preserves the
    # caller's order (they may have ranked anchors by importance) and
    # appends auto-extracted names at the tail. Deterministic dedup.
    combined: list[str] = []
    seen: set = set()
    for n in (anchor_names or []):
        if n not in seen:
            combined.append(n)
            seen.add(n)
    if auto_extract_uses:
        # Skip anchors already covered by a dotted variant the caller
        # supplied — ``candidates`` vs ``candidates.filter`` target the
        # same variable at the verifier level, so adding both is noise.
        dotted_bases = {n.split(".", 1)[0] for n in combined if "." in n}
        for n in extract_read_names(
            target_statement, module_level_names=module_level_names
        ):
            if n in seen:
                continue
            if n in dotted_bases:
                # Base already implied by a dotted anchor; skip.
                continue
            combined.append(n)
            seen.add(n)
    # Belt-and-suspenders: even caller-supplied anchors get filtered
    # against module_level_names.  LLM occasionally emits an import
    if module_level_names:
        combined = [n for n in combined if n.split(".", 1)[0] not in module_level_names]
        seen = set(combined)

    anchor_list_str = ", ".join(f"`{n}`" for n in combined) or "(none)"
    intent_hint = (
        f"LOCAL VARIABLE PLACEMENT RULE\n"
        f"The statement `{target_statement}` depends on local variable(s) "
        f"{anchor_list_str} (not parameters).\n"
        f"DO NOT insert it at function entry.\n"
        f"You MUST insert it immediately AFTER the nearest "
        f"dominating assignment where ALL required variables are "
        f"already defined with actual values (not None / empty "
        f"sentinel initializations).\n"
        f"For multiple anchors, placement must come after the LATEST of "
        f"their effective assignments (verifier uses max-of-anchors).\n"
        f"Search within the same block first, then outer blocks if needed.\n"
        f"If an equivalent statement already exists after "
        f"that assignment, prefer reusing/adjusting it rather than "
        f"adding a redundant one.\n"
        f"Do NOT change the function's return contract unless "
        f"the request explicitly requires it."
    )

    anchors = [
        PlacementAnchor(
            name=n,
            access_path=n,
            match="assignment",
            strength="effective",
        )
        for n in combined
    ]

    return PlacementContract(
        kind="after_anchor",
        anchors=anchors,
        scope=PlacementScope(mode="nearest_dominating", allow_nested=True),
        constraints=PlacementConstraints(
            forbid_before_anchor=True,
            forbid_reassignment_before_use=True,
            max_distance=None,
        ),
        intent_hint=intent_hint,
        verification=PlacementVerification(
            assertion_type="AFTER_ASSIGNMENT",
            mode=placement_mode,
        ),
        repair=PlacementRepair(
            on_violation="force_body_only",
            escalation="blocking",
        ),
        target_statement=target_statement,
        anchor_names=combined,
    )


def build_before_return_contract(
    target_statement: str,
    *,
    anchor_names: Optional[list[str]] = None,
    auto_extract_uses: bool = True,
    module_level_names: Optional[set[str]] = None,
) -> PlacementContract:
    """Build contract for code that must appear BEFORE return statement(s).

    Use for: cleanup code, logging, metrics collection, etc.

    Args:
        target_statement: The payload being placed before a return.
        anchor_names: Optional explicit anchors. If non-empty (either
            explicit or auto-extracted), the verifier additionally
            enforces that every anchor has an effective definition
            BEFORE the target — cleanup that reads ``x`` must not run
            before ``x`` is bound even if it sits before a return.
        auto_extract_uses: When True (default), parses
            ``target_statement`` for its Load-context Names and unions
            them into ``anchor_names``. Matches ``after_anchor``
            semantics: cleanup statements routinely reference variables
            defined earlier, and LLMs rarely enumerate every read.

    Anchor ordering semantics, if any anchors are present:
        Same max-of-anchors rule as ``after_anchor`` — placement must
        come after ``max(anchor_matches)`` over
        ``(frame_depth, index_in_frame)``. On top of that, at least
        one ``return`` with ``lineno > target.lineno`` must exist in
        the same function so the "before-return" side is satisfied.
    """
    combined: list[str] = []
    seen: set = set()
    for n in (anchor_names or []):
        if n not in seen:
            combined.append(n)
            seen.add(n)
    if auto_extract_uses:
        dotted_bases = {n.split(".", 1)[0] for n in combined if "." in n}
        for n in extract_read_names(
            target_statement, module_level_names=module_level_names
        ):
            if n in seen or n in dotted_bases:
                continue
            combined.append(n)
            seen.add(n)
    # Belt-and-suspenders parity with build_after_assignment_contract —
    # see the comment there for rationale.
    if module_level_names:
        combined = [n for n in combined if n.split(".", 1)[0] not in module_level_names]
        seen = set(combined)

    anchor_hint = (
        f"\nThe statement reads {', '.join(f'`{n}`' for n in combined)}; "
        f"placement must also come AFTER the latest effective "
        f"assignment of those names (max-of-anchors)."
        if combined else ""
    )
    intent_hint = (
        f"BEFORE-RETURN PLACEMENT RULE\n"
        f"The statement `{target_statement}` must be placed "
        f"immediately BEFORE the return statement(s).\n"
        f"If there are multiple return paths, place it before each one.\n"
        f"Do NOT place it at function entry or after return."
        f"{anchor_hint}"
    )

    anchors = [
        PlacementAnchor(
            name=n,
            access_path=n,
            match="assignment",
            strength="effective",
        )
        for n in combined
    ]

    return PlacementContract(
        kind="before_return",
        anchors=anchors,
        scope=PlacementScope(mode="global", allow_nested=True),
        constraints=PlacementConstraints(
            # When anchors are present, ordering matters (same rule as
            # after_anchor); otherwise we preserve the legacy "no anchor
            # constraint" defaults so contracts that deliberately skip
            # auto-extract stay backwards compatible.
            forbid_before_anchor=bool(combined),
            forbid_reassignment_before_use=bool(combined),
        ),
        intent_hint=intent_hint,
        verification=PlacementVerification(
            assertion_type="BEFORE_RETURN",
            mode="relaxed",
        ),
        repair=PlacementRepair(
            on_violation="force_body_only",
            escalation="blocking",
        ),
        target_statement=target_statement,
        anchor_names=combined,
    )


def build_at_function_entry_contract(
    target_statement: str,
) -> PlacementContract:
    """Build contract for code that must appear at function body start.

    Use for: parameter validation, input sanitization, entry logging.
    """
    intent_hint = (
        f"FUNCTION ENTRY PLACEMENT RULE\n"
        f"The statement `{target_statement}` must be the first "
        f"executable statement in the function body (after docstring if present).\n"
        f"Do NOT place it after any other logic."
    )

    return PlacementContract(
        kind="at_function_entry",
        anchors=[],
        scope=PlacementScope(mode="strict", allow_nested=False),
        constraints=PlacementConstraints(
            forbid_before_anchor=False,
            forbid_reassignment_before_use=False,
        ),
        intent_hint=intent_hint,
        verification=PlacementVerification(
            assertion_type="ENTRY_GUARD",
            mode="strict",
        ),
        repair=PlacementRepair(
            on_violation="force_body_only",
            escalation="blocking",
        ),
        target_statement=target_statement,
        anchor_names=[],
    )


def build_inside_block_contract(
    target_statement: str,
    block_type: str = "try",
    block_anchor: str = "",
    candidate_anchors: Optional[list[str]] = None,
    escalation: str = "blocking",
    *,
    anchor_role: Optional[str] = None,
) -> PlacementContract:
    """Build contract for code that must appear inside a specific block type.

    Use for: error handling (try/except), loop bodies, conditional blocks.

    Args:
        target_statement: The code to be placed
        block_type: "try" | "for" | "while" | "if" | "with"
        block_anchor: Optional identifier to match one specific block
                      (e.g. loop variable name, context manager name).
                      Authoritative when set.
        candidate_anchors: Optional list of acceptable anchors when the
                      correct block is structurally ambiguous (e.g. multiple
                      nested for-loops — the LLM picks; verifier accepts
                      any one of them).  Ignored when ``block_anchor`` is set.
        escalation: "blocking" or "warning"
        anchor_role: Declarative role for ``anchor_names`` — see
                      ``AnchorRole``. When omitted, derived from
                      ``block_type`` via ``_DEFAULT_ANCHOR_ROLE_BY_BLOCK_TYPE``
                      (e.g. ``for`` → ``ITER_VAR``). The verifier reads this
                      field directly instead of re-inferring from
                      ``block_type``; declaring it here is what migrates
                      the legacy implicit "for-loop iter target" check
                      onto a named role.
    """
    block_desc = {
        "try": "a try/except block",
        "for": "a for loop body",
        "while": "a while loop body",
        "if": "an if block",
        "with": "a with/context-manager block",
    }.get(block_type, f"a {block_type} block")

    if block_anchor:
        anchor_hint = f" associated with `{block_anchor}`"
    elif candidate_anchors:
        _cands = ", ".join(f"`{a}`" for a in candidate_anchors)
        anchor_hint = (
            f" (choose the loop whose iteration variable is one of: "
            f"{_cands})"
        )
    else:
        anchor_hint = ""
    if block_type == "for":
        _creation_hint = (
            f"The loop already exists in the function body. "
            f"Insert `{target_statement}` as the FIRST statement inside the "
            f"EXISTING loop body. Do NOT create a new loop."
        )
    else:
        _creation_hint = (
            "If the block does not exist yet, create it first and place the "
            "statement inside."
        )
    intent_hint = (
        f"INSIDE-BLOCK PLACEMENT RULE\n"
        f"The statement `{target_statement}` must be placed "
        f"inside {block_desc}{anchor_hint}.\n"
        f"Do NOT place it outside the block.\n"
        f"{_creation_hint}"
    )

    anchors = []
    _anchor_names: list[str] = []
    if block_anchor:
        anchors = [PlacementAnchor(
            name=block_anchor,
            match="call" if block_type == "with" else "assignment",
            strength="any",
        )]
        _anchor_names = [block_anchor]
    elif candidate_anchors:
        # Multiple candidates — verifier should accept a match against any.
        anchors = [
            PlacementAnchor(
                name=a,
                match="call" if block_type == "with" else "assignment",
                strength="any",
            )
            for a in candidate_anchors
        ]
        _anchor_names = list(candidate_anchors)

    # Closed-table default for anchor_role: explicit ``anchor_role=`` from
    # the caller wins (including ``""`` to opt out), otherwise the
    if anchor_role is None:
        _resolved_role = _DEFAULT_ANCHOR_ROLE_BY_BLOCK_TYPE.get(
            block_type, AnchorRole.UNSPECIFIED.value
        )
    else:
        _resolved_role = anchor_role

    return PlacementContract(
        kind="inside_block",
        anchors=anchors,
        scope=PlacementScope(mode="nearest_dominating", allow_nested=True),
        constraints=PlacementConstraints(
            forbid_before_anchor=False,
            forbid_reassignment_before_use=False,
        ),
        intent_hint=intent_hint,
        verification=PlacementVerification(
            assertion_type=f"INSIDE_BLOCK:{block_type}",
            mode="relaxed",
            block_type=block_type,  # P6: 1st-class field — no string parsing needed
        ),
        repair=PlacementRepair(
            on_violation="force_body_only",
            escalation=escalation,
        ),
        target_statement=target_statement,
        anchor_names=_anchor_names,
        anchor_role=_resolved_role,
    )


# ---------------------------------------------------------------------------
# Generic builder — dispatches by placement_kind metadata
# ---------------------------------------------------------------------------

def build_contract_from_metadata(
    metadata: dict[str, Any],
    *,
    file_content: str = "",
) -> Optional[PlacementContract]:
    """Build a PlacementContract from operation metadata fields.

    This is the GENERIC entry point for contract creation.
    The planner/DPB sets metadata fields, and this function dispatches
    to the appropriate builder.

    Expected metadata keys:
      placement_kind: "after_anchor" | "before_return" | "at_function_entry" | "inside_block"
      target_statement: str — the code being placed (required)
      anchor_names: List[str] — variable names (optional; auto-extracted
          from target_statement when absent and auto_extract_uses=True)
      auto_extract_uses: bool — opt-out of read-name auto-extraction.
          Defaults to True (PRIMARY path) for after_anchor and
          before_return; ignored by at_function_entry / inside_block.
      placement_mode: "strict" | "relaxed" (default: "relaxed")
      block_type: str — for inside_block (default: "try")
      block_anchor: str — for inside_block (optional)
      placement_escalation: "blocking" | "warning" (default: "blocking")

    Returns None if placement_kind is not set or target_statement is empty.
    """
    placement_kind = (metadata.get("placement_kind") or "").strip()
    if not placement_kind:
        return None

    target_statement = (metadata.get("target_statement") or "").strip()
    if not target_statement:
        return None

    anchor_names = metadata.get("anchor_names") or []
    # Pre-compute module-level names from the file (when caller provided
    # source) so all builder paths below can pass them down to
    # extract_read_names — closes the JSONResponse-style anchor leak.
    _module_level_names: Optional[set[str]] = (
        extract_module_level_names(file_content) if file_content else None
    )
    # auto_extract_uses defaults True; explicit False is the opt-out
    # (escape hatch for callers that need strictly LLM-specified anchors).
    auto_extract_uses = bool(metadata.get("auto_extract_uses", True))
    placement_mode = (metadata.get("placement_mode") or "relaxed").strip()
    escalation = (metadata.get("placement_escalation") or "blocking").strip()

    # build_contract_from_metadata is the canonical deterministic entry
    # point — anything coming through metadata fields has been authored by
    contract: Optional[PlacementContract]
    if placement_kind == "after_anchor":
        contract = build_after_assignment_contract(
            target_statement=target_statement,
            anchor_names=anchor_names,
            placement_mode=placement_mode,
            auto_extract_uses=auto_extract_uses,
            module_level_names=_module_level_names,
        )
        # If neither explicit anchors nor auto-extraction produced any
        # anchor, there's nothing for the verifier to enforce — drop the
        # contract so we don't attach a noop.
        if not contract.anchor_names:
            return None
    elif placement_kind == "before_return":
        contract = build_before_return_contract(
            target_statement=target_statement,
            anchor_names=anchor_names,
            auto_extract_uses=auto_extract_uses,
            module_level_names=_module_level_names,
        )
    elif placement_kind == "at_function_entry":
        contract = build_at_function_entry_contract(
            target_statement=target_statement,
        )
    elif placement_kind == "inside_block":
        block_type = (metadata.get("block_type") or "try").strip()
        block_anchor = (metadata.get("block_anchor") or "").strip()
        # Distinguish "key absent" (→ builder default from block_type)
        # from "key present and empty string" (→ explicit UNSPECIFIED).
        # ``or ""``-style coercion would conflate the two and silently
        # turn an opt-out into the default.
        if "anchor_role" in metadata:
            _raw_role = metadata.get("anchor_role")
            _meta_anchor_role: Optional[str] = (
                _raw_role.strip() if isinstance(_raw_role, str) else ""
            )
        else:
            _meta_anchor_role = None
        contract = build_inside_block_contract(
            target_statement=target_statement,
            block_type=block_type,
            block_anchor=block_anchor,
            escalation=escalation,
            anchor_role=_meta_anchor_role,
        )
    else:
        logger.warning("build_contract_from_metadata: unknown placement_kind=%s", placement_kind)
        return None

    if contract is not None:
        contract.source = "deterministic"
    return contract


# ---------------------------------------------------------------------------
# Top-level (module-level) contract builder and verifier
# ---------------------------------------------------------------------------

def build_top_level_contract(new_symbol_name: str) -> "PlacementContract":
    """Return a contract asserting that new_symbol_name is defined at module level."""
    return PlacementContract(
        kind="top_level",
        anchor_names=[new_symbol_name],
        insertion_container="module",
        intent_hint=f"Insert '{new_symbol_name}' as a module-level definition (not inside any class or function).",
        source="deterministic",
    )


def verify_top_level_placement(new_symbol_name: str, source_code: str) -> tuple[bool, str]:
    """Return (ok, message) — True if new_symbol_name is defined at module level."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return False, f"parse_error: {exc}"

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == new_symbol_name:
                return True, f"'{new_symbol_name}' found at module level"

    return False, f"'{new_symbol_name}' not found at module level in parsed AST"


# ---------------------------------------------------------------------------
# Verification — AST-based, deterministic
# ---------------------------------------------------------------------------

def verify_placement_contract(
    source_code: str,
    target_symbol: str,
    contract: PlacementContract,
    *,
    strict_match: bool = False,
    patch_strategy: str = "",
) -> tuple[bool, str]:
    """Verify that placement contract is satisfied in the given source code.

    This is the SINGLE entry point for all placement verification.
    Returns (ok, message). If not ok and contract.is_blocking,
    caller should treat as blocking failure.

    Dispatches to kind-specific verification functions.

    ``strict_match=True``: tightens ``_stmt_satisfies_signature`` for
    pre-edit idempotency checks (ALL target calls must be present in
    the candidate statement, not just any overlap).  Post-LLM placement
    verification keeps ``False`` so that LLM paraphrasing is still
    accepted.  Callers should only pass ``True`` when asking "is the
    guard ALREADY present?", not "did the LLM place it correctly?".

    ``patch_strategy``: the EditInstructionKind value of the edit that
    produced this source (e.g. ``"replace_symbol_body"``, ``"surgical_edit"``,
    ``"ast_op"``).  When the LLM rewrites an entire function body
    (``replace_symbol_body``), the ``after_anchor`` ordering check is
    skipped — the old code's anchor line numbers are irrelevant to the
    new body, and the LLM is responsible for internal ordering.  All
    other contract kinds (``at_function_entry``, ``before_return``,
    ``inside_block``) are always verified regardless of strategy.
    """
    # top_level contracts bypass the intra-function search entirely.
    if contract.kind == "top_level":
        _tl_sym = contract.anchor_names[0] if contract.anchor_names else target_symbol
        return verify_top_level_placement(_tl_sym, source_code)

    try:
        tree = ast_cache.parse_cached(source_code)
    except Exception as e:
        return False, f"parse_error: {e}"

    func = _find_target_function(tree, target_symbol)
    if func is None:
        return False, f"target function '{target_symbol}' not found"

    if contract.kind == "after_anchor":
        if patch_strategy == "replace_symbol_body":
            # Full-body rewrite: anchor line numbers from the old code no longer
            # apply.  The LLM produced a complete new body; ordering within that
            # body is its responsibility.  Skipping avoids false-positive
            # placement_violations that block correct rewrites.
            return True, "skipped: replace_symbol_body — after_anchor ordering not verified for full-body rewrites"
        return _verify_after_anchor(func, contract, strict_call_matching=strict_match)
    elif contract.kind == "at_function_entry":
        return _verify_at_function_entry(func, contract)
    elif contract.kind == "before_return":
        return _verify_before_return(func, contract, strict_call_matching=strict_match)
    elif contract.kind == "inside_block":
        return _verify_inside_block(func, contract)
    else:
        logger.warning("verify_placement_contract: unsupported kind=%s", contract.kind)
        return True, f"no-op (unsupported contract kind: {contract.kind})"


# ---------------------------------------------------------------------------
# Representation × placement compatibility

# Score in [0.0, 1.0].  0.0 = incompatible, 1.0 = perfectly suited.
# Missing entries default to 1.0 (no signal — selector falls back to other
# heuristics).  Add new representations / placement kinds here only when we
# have a concrete reason from verifier behaviour or empirical failure data.
_REPRESENTATION_COMPATIBILITY: dict[tuple[str, str], float] = {
    # after_anchor:  verifier skips ordering for replace_symbol_body
    ("after_anchor", "replace_symbol_body"): 0.0,
    ("after_anchor", "ast_op"):              1.0,
    ("after_anchor", "ast_direct_body"):     0.7,
    ("after_anchor", "surgical_edit"):       0.9,

    # before_return: full rewrite is allowed by the verifier but loses the
    # structural intent in practice.
    ("before_return", "replace_symbol_body"): 0.3,
    ("before_return", "ast_op"):              1.0,
    ("before_return", "ast_direct_body"):     0.7,
    ("before_return", "surgical_edit"):       0.9,

    # inside_block: same logic as before_return — block-level slot is fragile
    # under full-body rewrites.
    ("inside_block", "replace_symbol_body"): 0.2,
    ("inside_block", "ast_op"):              1.0,
    ("inside_block", "ast_direct_body"):     0.7,
    ("inside_block", "surgical_edit"):       0.9,

    # at_function_entry: prepending at entry is well-defined for surgical
    # and ast paths; full-body rewrite is acceptable since the entire
    # function is regenerated and the LLM controls the entry.
    ("at_function_entry", "replace_symbol_body"): 0.8,
    ("at_function_entry", "ast_op"):              1.0,
    ("at_function_entry", "ast_direct_body"):     0.9,
    ("at_function_entry", "surgical_edit"):       0.9,
}

# Below this score the representation is treated as a hard ban for the given
# placement contract.  Tuned so 0.0 (the verifier-skip case) becomes a ban
# while 0.2-0.3 (discouraged) remains a soft demotion handled by the
# selector's scoring.
_REPRESENTATION_INCOMPATIBLE_THRESHOLD = 0.1


def representation_compatibility_score(
    contract: Optional["PlacementContract"], representation: str
) -> float:
    """Return a [0.0, 1.0] score for (placement_kind, representation).

    Returns 1.0 when there is no contract or no specific entry — absence of
    signal is *not* a hard ban, just a "no preference" from the placement
    layer.
    """
    if contract is None or not representation:
        return 1.0
    pk = (getattr(contract, "kind", "") or "").strip().lower()
    if not pk:
        return 1.0
    return _REPRESENTATION_COMPATIBILITY.get((pk, representation), 1.0)


def representations_incompatible_with(
    contract: Optional["PlacementContract"],
) -> list[str]:
    """Return the list of representations that the selector should ban for
    the given contract.  Empty when no contract is supplied."""
    if contract is None:
        return []
    pk = (getattr(contract, "kind", "") or "").strip().lower()
    if not pk:
        return []
    banned: list[str] = []
    for (placement_kind, rep), score in _REPRESENTATION_COMPATIBILITY.items():
        if placement_kind != pk:
            continue
        if score < _REPRESENTATION_INCOMPATIBLE_THRESHOLD:
            banned.append(rep)
    return banned


# ---------------------------------------------------------------------------
# Pre-edit feasibility — ground-truth labeling for shadow records


def check_candidate_feasibility(
    contract: PlacementContract,
    target_symbol: str,
    source_code: Optional[str],
) -> tuple[str, str]:
    """Pre-edit feasibility check for a placement candidate.

    Different from two sibling functions in this module:

    * ``verify_placement_contract`` — POST-edit: "is the target statement
      placed correctly *now*?"  Needs the target_statement which only
      exists after the developer LLM has emitted code.
    * ``precheck_placement_feasibility`` — LIVE preflight: "should we burn
      an LLM call?"  Bias toward soft-pass (returns True on parse_error
      and missing target function) to avoid false rejects.  Covers fewer
      kinds (only inside_block + after_anchor get a real check).

    This function — GROUND-TRUTH labeling: "could this candidate
    structurally apply to ``source_code``?"  Bias toward an explicit
    3-state classification so shadow corpora can split feasible /
    infeasible / unknown without conflating "we don't know" with
    "definitely wrong".  Covers all four contract kinds.

    Returns ``(status, reason)`` where ``status`` is one of:
      - ``"feasible"``   structural prerequisites met
      - ``"infeasible"`` structural prerequisites missing (anchor absent,
                         no return, etc.)
      - ``"unknown"``    cannot determine (no source / parse error / no
                         target symbol).  Aggregation should exclude these
                         from feasibility-rate denominators.

    ``reason`` is a short machine-readable token (e.g.
    ``"anchor_not_found:foo"``, ``"no_return"``, ``"target_not_found"``)
    suitable for jsonl logging and bucket aggregation.
    """
    if not source_code:
        return "unknown", "no_source_code"
    if not target_symbol:
        return "unknown", "no_target_symbol"

    try:
        tree = ast_cache.parse_cached(source_code)
    except Exception as e:
        return "unknown", f"parse_error:{type(e).__name__}"

    func = _find_target_function(tree, target_symbol)
    if func is None:
        return "infeasible", "target_not_found"

    kind = contract.kind

    if kind == "at_function_entry":
        # Always feasible if the function exists.  Empty body / docstring-only
        # are valid insertion points (verifier handles those at apply time).
        return "feasible", "ok"

    if kind == "before_return":
        # Feasible if at least one Return statement is reachable from the
        # function body (including in nested try/if/with — verifier handles
        # the lineno ordering at apply time).
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                return "feasible", "ok"
        return "infeasible", "no_return"

    if kind == "after_anchor":
        anchor_names = list(contract.anchor_names) or [
            a.name for a in contract.anchors if getattr(a, "name", None)
        ]
        if not anchor_names:
            # Contract claims after_anchor but specifies no anchor — degenerate
            # but not structurally infeasible (verifier returns "no anchors to
            # verify" → True).  Mark feasible with a tag so aggregation can
            # filter if needed.
            return "feasible", "no_anchors_specified"

        # Collect every Name reference (Load OR Store) inside the function
        # body — for feasibility, even a use-only reference proves the anchor
        # exists in this scope.  Dotted names (``x.attr``) match against the
        # base segment, mirroring _verify_after_anchor.
        present: set = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Name):
                present.add(node.id)
            elif isinstance(node, ast.arg):
                present.add(node.arg)
            elif isinstance(node, ast.Attribute):
                # Walk the .value chain to find the base Name (a.b.c → a).
                base = node.value
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    present.add(base.id)

        missing = [n for n in anchor_names if n.split(".", 1)[0] not in present]
        if not missing:
            return "feasible", "ok"
        # Partial: SOME anchors present is still feasible (verifier picks any
        # satisfied anchor).  Only fully-missing is infeasible.
        if len(missing) < len(anchor_names):
            return "feasible", f"partial_anchors_missing:{','.join(missing)}"
        return "infeasible", f"anchor_not_found:{','.join(missing)}"

    if kind == "inside_block":
        # P6: block_type 1st-class field preferred; legacy format fallback.
        block_type = contract.verification.block_type
        if not block_type:
            assertion = contract.verification.assertion_type or "INSIDE_BLOCK:try"
            block_type = assertion.split(":", 1)[1] if ":" in assertion else "try"
        type_map = {
            "try":   (ast.Try,),
            "for":   (ast.For, ast.AsyncFor),
            "while": (ast.While,),
            "if":    (ast.If,),
            "with":  (ast.With, ast.AsyncWith),
        }
        required = type_map.get(block_type)
        if required is None:
            return "unknown", f"unsupported_block_type:{block_type}"
        for node in ast.walk(func):
            if isinstance(node, required):
                return "feasible", "ok"
        return "infeasible", f"no_{block_type}_block"

    # Unknown kind → cannot evaluate, don't pollute feasibility rate.
    return "unknown", f"unsupported_kind:{kind}"


# ---------------------------------------------------------------------------
# Round-trip: verify failure → structured repair hint
# ---------------------------------------------------------------------------

# Message → repair_action lookup table. Each entry is a (substring, action)
# pair; the FIRST matching substring wins. Substrings are phrases the
_PLACEMENT_REPAIR_ACTION_TABLE: tuple[tuple[str, str], ...] = (
    ("assigned after target",              "placement.reorder_target_after_anchor_def"),
    ("no effective assignment found",      "placement.add_effective_assignment_before_target"),
    ("no statement matching target signature",
                                           "placement.insert_target_in_function"),
    ("no return statement appears after target",
                                           "placement.place_target_before_existing_return"),
    ("no statement after anchor",          "placement.move_target_immediately_after_anchor"),
    ("FAIL strict:",                       "placement.move_target_immediately_after_anchor"),
    ("reassignment of",                    "placement.prevent_anchor_reassignment_before_target"),
    ("not before any return",              "placement.place_target_before_existing_return"),
    ("target is not at function entry",    "placement.move_target_to_function_entry"),
    ("function body has only docstring",   "placement.add_statements_after_docstring"),
    ("not found inside any",               "placement.wrap_target_in_required_block"),
    ("target statement not parseable",     "placement.emit_syntactically_valid_target"),
    ("target statement empty",             "placement.provide_non_empty_target"),
    ("parse_error",                        "placement.emit_syntactically_valid_source"),
    ("target function",                    "placement.ensure_target_function_defined"),
)


def _classify_placement_repair_action(verify_message: str) -> str:
    """Map a ``verify_placement_contract`` failure message to a short
    repair_action code that repair candidates can dispatch on.

    Returns ``"placement.inspect_rule"`` for unmatched messages — the
    generic fallback instructs the LLM to re-read the contract rule
    itself (surfaced via ``contract.intent_hint`` in the round-trip
    payload).
    """
    if not verify_message:
        return "placement.inspect_rule"
    for needle, action in _PLACEMENT_REPAIR_ACTION_TABLE:
        if needle in verify_message:
            return action
    return "placement.inspect_rule"


def build_violation_from_verify_result(
    contract: PlacementContract,
    ok: bool,
    verify_message: str,
    symbol: str = "",
) -> Optional[dict[str, Any]]:
    """Convert a ``verify_placement_contract`` result into a structured
    ``ContractViolation`` dict consumable by repair candidates.

    Returns ``None`` when the contract passed (``ok=True``) — passing
    verifications carry no information the Developer LLM needs. On
    failure, returns a dict in the cross-layer schema (generic so that
    future IntentAssertion / EditContract / Dataflow Cross-Ref
    integrations can reuse the same shape):

        {
            "layer": "placement",
            "kind": str,              # contract.kind
            "symbol": str,
            "anchor_names": List[str],
            "target_statement": str,
            "verify_message": str,    # verbatim verifier output
            "repair_action": str,     # classification (see table above)
            "intent_hint": str,       # contract.intent_hint (rule in NL)
            "is_blocking": bool,
        }

    The dict is JSON-serializable (contract.to_dict() is already proven
    for metadata round-trips) so repair_engine can pass it through
    ``instruction.metadata["_contract_violations"]`` without custom
    serializers.
    """
    if ok:
        return None
    if contract is None:
        return None
    return {
        "layer": "placement",
        "kind": contract.kind,
        "symbol": symbol,
        "anchor_names": list(contract.anchor_names or []),
        "target_statement": contract.target_statement,
        "verify_message": verify_message,
        "repair_action": _classify_placement_repair_action(verify_message),
        "intent_hint": contract.intent_hint,
        "is_blocking": contract.is_blocking,
    }


# ---------------------------------------------------------------------------
# Internal: after_anchor verification
# ---------------------------------------------------------------------------

@dataclass
class _BlockFrame:
    """Single statement block (function body / if body / try body etc.)."""
    stmts: list          # List[ast.stmt]
    parent_idx: Optional[int]  # index of block entry in parent frame (None for root)


@dataclass
class _AnchorMatch:
    """Effective assignment location."""
    name: str
    frame: _BlockFrame
    index_in_frame: int
    frame_depth: int     # depth in block path (0 = function body)


def _find_target_function(tree: ast.AST, name: str) -> Optional[ast.FunctionDef]:
    """Find function/method by name (supports Class.method dotted names)."""
    parts = name.split(".")
    if len(parts) == 1:
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                return n
    elif len(parts) == 2:
        class_name, method_name = parts
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == class_name:
                for child in ast.walk(n):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                        return child
    return None


def _normalize_stmt(code: str) -> Optional[str]:
    """Normalize a statement to canonical AST representation."""
    try:
        node = ast_cache.parse_cached(code).body[0]
        return ast.unparse(node)
    except (SyntaxError, TypeError, AttributeError):
        return None


def _stmt_matches(stmt: ast.stmt, norm: str) -> bool:
    """Check if a statement matches the normalized form."""
    try:
        return ast.unparse(stmt) == norm
    except (SyntaxError, TypeError, AttributeError):
        return False


def _is_weak_value(value: Optional[ast.AST]) -> bool:
    """True if value is a sentinel initializer (None, [], {}, (), "", 0, False)."""
    if value is None:
        return True
    if isinstance(value, ast.Constant):
        return value.value in (None, "", 0, False)
    if isinstance(value, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
        elts = getattr(value, "elts", None) or getattr(value, "keys", None) or []
        return len(elts) == 0
    return False


def _assign_strength(stmt: ast.stmt, name: str) -> str:
    """Classify how a statement assigns to a name.

    Supports dotted anchors (x.attr):
      1. x.attr = value  (attribute assignment) -> preferred
      2. x = value       (base object creation) -> fallback

    Returns: "none" | "weak" | "effective" | "update"
    """
    is_dotted = "." in name
    if is_dotted:
        base, attr = name.rsplit(".", 1)
    else:
        base = name
        attr = None

    def _name_matches(target: ast.AST, check_name: str) -> bool:
        return isinstance(target, ast.Name) and target.id == check_name

    def _attr_matches(target: ast.AST, check_base: str, check_attr: str) -> bool:
        return (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == check_base
                and target.attr == check_attr)

    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            if is_dotted and _attr_matches(t, base, attr):
                return "weak" if _is_weak_value(stmt.value) else "effective"
            if _name_matches(t, base):
                return "weak" if _is_weak_value(stmt.value) else "effective"
        return "none"

    if isinstance(stmt, ast.AnnAssign):
        t = stmt.target
        if is_dotted and _attr_matches(t, base, attr):
            return "weak" if _is_weak_value(stmt.value) else "effective"
        if _name_matches(t, base):
            return "weak" if _is_weak_value(stmt.value) else "effective"
        return "none"

    if isinstance(stmt, ast.AugAssign):
        t = stmt.target
        if is_dotted and _attr_matches(t, base, attr):
            return "update"
        if _name_matches(t, base):
            return "update"
        return "none"

    return "none"


def _find_stmt_block_path(func: ast.AST, pred) -> tuple[list[_BlockFrame], int]:
    """Find the block path to a statement matching pred.

    Returns (path, guard_idx) where:
      - path: list of _BlockFrame from outermost to innermost
      - guard_idx: index of the matched statement in the last frame
      - each frame's parent_idx records the entry point in the parent frame
        (i.e. the index of the compound statement that contains this block)

    Returns ([], -1) if not found.
    """
    def _visit(node, current_path, entry_idx_in_parent=None):
        for field_name in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field_name, None)
            if not isinstance(stmts, list):
                continue

            frame = _BlockFrame(stmts=stmts, parent_idx=entry_idx_in_parent)
            new_path = [*current_path, frame]

            for i, s in enumerate(stmts):
                if pred(s):
                    return new_path, i
                found = _visit(s, new_path, entry_idx_in_parent=i)
                if found:
                    return found
        return None

    result = _visit(func, [])
    return result if result else ([], -1)


def _first_effective_assignment_lineno(func: ast.AST, name: str) -> Optional[int]:
    """Return the line number of the first effective assignment of ``name``
    anywhere inside ``func``, or None if none found.

    Used to produce actionable repair hints when an ordering violation is
    detected: "insert after line N" is far more useful than "force_body_only".
    """
    priority = {"none": 0, "weak": 1, "update": 2, "effective": 3}
    best_lineno: Optional[int] = None
    best_strength = "none"
    for n in ast.walk(func):
        if not isinstance(n, ast.stmt):
            continue
        s = _assign_strength(n, name)
        if priority[s] > priority[best_strength]:
            best_strength = s
            best_lineno = getattr(n, "lineno", None)
            if best_strength == "effective" and best_lineno is not None:
                # First effective assignment found; but we want the EARLIEST,
                # so keep scanning for lower line numbers.
                pass
    # Re-scan for the earliest effective assignment line.
    if best_strength == "effective":
        earliest: Optional[int] = None
        for n in ast.walk(func):
            if not isinstance(n, ast.stmt):
                continue
            if _assign_strength(n, name) == "effective":
                lineno = getattr(n, "lineno", None)
                if lineno is not None and (earliest is None or lineno < earliest):
                    earliest = lineno
        return earliest
    return None


def _strongest_assignment_in_scope(
    path: list[_BlockFrame],
    guard_idx: int,
    name: str,
) -> str:
    """Strongest assignment kind for ``name`` within the ancestor scope of
    the target statement.

    Unlike ``_strongest_assignment`` (which walks the entire function via
    ast.walk and is loop-context-blind), this function only inspects
    statements that are VISIBLE to the target's execution context:

      - Statements that precede the target in the target's own block frame
      - Statements that precede the enclosing compound in each outer frame
        (the same backward range that ``_find_nearest_anchor`` searches)

    Assignments in SIBLING blocks — e.g., a prior ``for op in X:`` loop
    at the same depth as the target's ``for op in Y:`` loop — are
    intentionally excluded.  Those loops execute in separate iteration
    contexts; their variable bindings are NOT visible at the target's
    execution point.  Using the global ast.walk caused false-positive
    ordering violations when the same attribute name (e.g. ``op.depends_on``)
    appeared in both a prior loop body and the new code being verified.

    Used in ``_verify_after_anchor`` and ``_verify_before_return`` as a
    drop-in replacement for ``_strongest_assignment(func, name)`` in the
    ordering-violation fallback path.
    """
    priority = {"none": 0, "weak": 1, "update": 2, "effective": 3}
    best = "none"

    for depth in range(len(path) - 1, -1, -1):
        frame = path[depth]
        stmts = frame.stmts

        if depth == len(path) - 1:
            rng = range(guard_idx - 1, -1, -1)
        else:
            upper = path[depth + 1].parent_idx
            if upper is not None:
                rng = range(upper - 1, -1, -1)
            else:
                rng = range(len(stmts) - 1, -1, -1)

        for i in rng:
            s = stmts[i]
            strength = _assign_strength(s, name)
            if strength == "none" and isinstance(
                s, (ast.Try, ast.With, ast.AsyncWith,
                    ast.If, ast.For, ast.While, ast.AsyncFor)
            ):
                strength = _strongest_in_compound(s, name)
            if priority[strength] > priority[best]:
                best = strength
                if best == "effective":
                    return best

    return best


# Nested scope boundaries — walking into these would cross a lexical
# scope, so ``_find_nearest_anchor`` and the compound-block helpers
# must stop at them.
_NESTED_SCOPE_STOP = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _strongest_in_stmts(
    stmts: list,
    name: str,
) -> str:
    """Strongest assignment kind for ``name`` within a block of
    statements, descending recursively into any compound statements
    (try / with / if / for / while etc.) but stopping at nested
    function/class scopes.  Used by ``_strongest_in_compound`` to
    aggregate across the branches of a compound.
    """
    priority = {"none": 0, "weak": 1, "update": 2, "effective": 3}
    best = "none"
    for s in stmts:
        if isinstance(s, _NESTED_SCOPE_STOP):
            continue
        strength = _assign_strength(s, name)
        if priority[strength] > priority[best]:
            best = strength
            if best == "effective":
                return best
        for field_name in ("body", "orelse", "finalbody"):
            nested = getattr(s, field_name, None)
            if isinstance(nested, list):
                sub = _strongest_in_stmts(nested, name)
                if priority[sub] > priority[best]:
                    best = sub
                    if best == "effective":
                        return best
        handlers = getattr(s, "handlers", None)
        if isinstance(handlers, list):
            for h in handlers:
                h_body = getattr(h, "body", None)
                if isinstance(h_body, list):
                    sub = _strongest_in_stmts(h_body, name)
                    if priority[sub] > priority[best]:
                        best = sub
                        if best == "effective":
                            return best
    return best


def _strongest_in_compound(node: ast.AST, name: str) -> str:
    """Strongest assignment kind for ``name`` attributable to running
    ``node`` (a compound statement) to completion in the guard's
    normal flow.

    Semantics by node kind:
      * ``Try`` / ``With`` / ``AsyncWith`` — all sub-blocks run before
        the guard in any path that reaches it, so any effective assign
        in body / orelse / handlers / finalbody counts.  We walk them
        all and take the strongest.
      * ``If`` — the guard is only guaranteed to see the name bound
        when BOTH branches assign it.  Take the weakest of the two
        branches (the conservative one).
      * ``For`` / ``While`` / ``AsyncFor`` — body may execute zero
        times.  Only the ``else:`` branch is guaranteed to run when
        the loop completes normally; treat else-only assigns as the
        effective source.  Body-only assigns are not sufficient.
    Anything else → ``"none"``.
    """
    priority = {"none": 0, "weak": 1, "update": 2, "effective": 3}

    if isinstance(node, (ast.Try,)):
        # Try: body + orelse + handlers + finalbody are all pre-guard.
        branches = [
            getattr(node, "body", None) or [],
            getattr(node, "orelse", None) or [],
            getattr(node, "finalbody", None) or [],
        ]
        handlers = getattr(node, "handlers", None) or []
        for h in handlers:
            branches.append(getattr(h, "body", None) or [])
        best = "none"
        for stmts in branches:
            sub = _strongest_in_stmts(stmts, name)
            if priority[sub] > priority[best]:
                best = sub
                if best == "effective":
                    return best
        return best

    if isinstance(node, (ast.With, ast.AsyncWith)):
        return _strongest_in_stmts(getattr(node, "body", None) or [], name)

    if isinstance(node, ast.If):
        body_s = _strongest_in_stmts(getattr(node, "body", None) or [], name)
        else_s = _strongest_in_stmts(getattr(node, "orelse", None) or [], name)
        if body_s == "none" or else_s == "none":
            # One branch doesn't bind — guard cannot rely on the name.
            return "none"
        # Weakest of two (both must bind for guaranteed availability).
        return body_s if priority[body_s] < priority[else_s] else else_s

    if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
        # Body may never run. Only the else-branch (executed on normal
        # completion) is a safe source of binding.
        return _strongest_in_stmts(getattr(node, "orelse", None) or [], name)

    return "none"


def _find_nearest_anchor(
    path: list[_BlockFrame],
    guard_idx: int,
    name: str,
    strength_required: str = "effective",
) -> Optional[_AnchorMatch]:
    """Find nearest dominating anchor assignment for a name.

    Walks from the guard frame outward along the block path, looking
    for effective (non-weak) assignments.  When a pre-guard statement
    is a compound (try / with / if / for / while), ``_strongest_in_compound``
    inspects its sub-blocks to see whether the name is bound along
    every path that reaches the guard — otherwise the verifier would
    miss an anchor that lives inside a try body (common pattern for
    ``_result = subprocess.run(...)``-style resource setup).
    """
    priority = {"none": 0, "weak": 1, "update": 2, "effective": 3}

    for depth in range(len(path) - 1, -1, -1):
        frame = path[depth]
        stmts = frame.stmts

        if depth == len(path) - 1:
            rng = range(guard_idx - 1, -1, -1)
        else:
            upper = path[depth + 1].parent_idx
            if upper is not None:
                rng = range(upper - 1, -1, -1)
            else:
                rng = range(len(stmts) - 1, -1, -1)

        for i in rng:
            s = stmts[i]
            strength = _assign_strength(s, name)
            if strength == "none":
                # Compound-block-aware peek (Fix E): the name might be
                # bound inside a try/with/if branch that executes
                # before the guard.
                if isinstance(s, (ast.Try, ast.With, ast.AsyncWith,
                                  ast.If, ast.For, ast.While, ast.AsyncFor)):
                    strength = _strongest_in_compound(s, name)
            if strength == "none":
                continue
            if strength_required == "effective" and (
                strength == "weak"
                or priority[strength] < priority[strength_required]
            ):
                continue
            return _AnchorMatch(name=name, frame=frame,
                                index_in_frame=i, frame_depth=depth)

    return None


def _target_in_subtree(node: ast.AST, target_norm: str) -> bool:
    """Check if target statement exists anywhere in the subtree of a compound stmt."""
    for field_name in ("body", "orelse", "finalbody", "handlers"):
        stmts = getattr(node, field_name, None)
        if not isinstance(stmts, list):
            continue
        for s in stmts:
            if _stmt_matches(s, target_norm):
                return True
            if _target_in_subtree(s, target_norm):
                return True
    return False


@dataclass
class _TargetSignature:
    """Structural fingerprint of the PCL target_statement.

    Used by ``_verify_after_anchor`` to recognize semantically
    equivalent statements the LLM may generate.  The identity we check
    is deliberately not ``ast.unparse`` equality — LLMs phrase the same
    guard in many valid ways (adding defensive short-circuits, using
    different loop styles, etc.).  What the contract actually demands
    is "after the anchors are bound, a statement runs that references
    the anchors and carries the target's statement-kind +
    literal/call/control-flow fingerprint."
    """
    stmt_type: type            # Outer stmt AST class (If, Return, Raise, For, ...)
    load_names: set            # Name ids + Attribute base names loaded
    literals: set              # Non-empty string/numeric constants
    calls: set                 # Function/method names invoked
    has_early_exit: bool       # contains Return or Raise
    returns_none: bool         # contains a bare ``return None`` (Return with None constant)
    condition_names: set       # Name ids that appear in the test/condition of an If
    condition_attrs: set       # Attribute names (e.g. READ_FILE_SEGMENT from OperationKind.X)
                               # in the test/condition — used for strict idempotency matching


def _extract_target_signature(node: ast.AST) -> _TargetSignature:
    load_names: set = set()
    literals: set = set()
    calls: set = set()
    has_early_exit = False
    condition_names: set = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load):
                load_names.add(n.id)
        elif isinstance(n, ast.Attribute):
            base = n.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                load_names.add(base.id)
        elif isinstance(n, ast.Constant):
            v = n.value
            # Skip boolean / None / empty / trivial numeric constants
            # — they appear in too many statements to be discriminating.
            if isinstance(v, str) and v:
                literals.add(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
                literals.add(v)
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                calls.add(f.id)
            elif isinstance(f, ast.Attribute):
                calls.add(f.attr)
        elif isinstance(n, (ast.Return, ast.Raise)):
            has_early_exit = True
    returns_none = any(
        isinstance(n, ast.Return)
        and (n.value is None or (isinstance(n.value, ast.Constant) and n.value.value is None))
        for n in ast.walk(node)
    )
    # Extract names that appear specifically in the condition (test) of an If
    # statement.  These are the strongest discriminators: a guard checking
    # ``if kind == X and not _path:`` has entirely different semantics from
    # ``if not obj or 'key' not in obj:`` even though both use logger.warning.
    condition_attrs: set = set()
    if isinstance(node, ast.If):
        for cnd_n in ast.walk(node.test):
            if isinstance(cnd_n, ast.Name) and isinstance(cnd_n.ctx, ast.Load):
                condition_names.add(cnd_n.id)
            elif isinstance(cnd_n, ast.Attribute):
                # Collect both the base name and the attribute name so
                # OperationKind.READ_FILE_SEGMENT contributes both "OperationKind"
                # (to condition_names) and "READ_FILE_SEGMENT" (to condition_attrs).
                condition_attrs.add(cnd_n.attr)
                cnd_base = cnd_n.value
                while isinstance(cnd_base, ast.Attribute):
                    condition_attrs.add(cnd_base.attr)
                    cnd_base = cnd_base.value
                if isinstance(cnd_base, ast.Name):
                    condition_names.add(cnd_base.id)
    return _TargetSignature(
        stmt_type=type(node),
        load_names=load_names,
        literals=literals,
        calls=calls,
        has_early_exit=has_early_exit,
        returns_none=returns_none,
        condition_names=condition_names,
        condition_attrs=condition_attrs,
    )


def _stmt_satisfies_signature(
    stmt: ast.AST,
    target_sig: _TargetSignature,
    anchor_bases: set,
    *,
    strict_call_matching: bool = False,
) -> bool:
    """Is this statement plausibly the one the contract is describing?

    Rules (in order):
      1. Outer stmt kind must match target's kind (If vs Return vs Try vs …)
         — prevents e.g. a bare ``return candidates`` from being accepted
         as a substitute for ``if not candidates: return None``.
      2. All anchor base names appear as loads in the statement (≥1
         overlap required if anchors supplied — we accept a single
         referencing stmt for multi-anchor contracts because the LLM
         may split the guard).
      3. At least one literal OR call name from the target signature
         appears in the statement's subtree.  If the target signature
         is empty on both literal and call axes, fall back to rules
         1+2 only.
         When ``strict_call_matching=True`` (idempotency checks): ALL
         target calls must be present in the candidate — missing a call
         means the statement is semantically different (e.g. a guard
         checking ``len(data.split(...))`` is distinct from one checking
         ``len(data)``; the ``split`` absence is discriminating).
      4. If the target includes an early exit (return/raise), the
         statement's subtree must also contain one.
      5. If the target is an ``If`` and has condition_names that overlap
         with anchor_bases, the candidate's condition must also reference
         at least one of those names.  This prevents unrelated guards that
         share only body-level names (e.g. ``logger.warning``) from being
         accepted as placement matches — which causes false ordering
         violations when such an unrelated guard happens to appear before
         the anchor assignments.

    ``strict_call_matching`` should be ``True`` only for pre-edit idempotency
    checks (is the guard ALREADY there?).  Post-edit placement verification
    keeps ``False`` so that LLM paraphrasing (different but semantically
    equivalent guards) is still accepted.
    """
    # Rule 1 — stmt kind
    if target_sig.stmt_type is not None and not isinstance(stmt, target_sig.stmt_type):
        return False

    stmt_sig = _extract_target_signature(stmt)

    # Rule 2 — anchor presence
    if anchor_bases and not (stmt_sig.load_names & anchor_bases):
        return False

    # Rule 3 — discriminating content
    has_discriminator = bool(target_sig.literals) or bool(target_sig.calls)
    if has_discriminator:
        lit_overlap = stmt_sig.literals & target_sig.literals
        call_overlap = stmt_sig.calls & target_sig.calls
        if not lit_overlap and not call_overlap:
            return False
        # Strict idempotency: ALL target calls must be present in the candidate.
        # ANY overlap is intentionally kept for post-LLM placement verification
        # (LLM may rephrase the guard) but is too permissive for idempotency
        # (a guard missing ``split`` is a different operation, not the same one).
        if strict_call_matching and target_sig.calls and not target_sig.calls.issubset(stmt_sig.calls):
            return False

    # Rule 4 — early-exit preservation
    if target_sig.has_early_exit and not stmt_sig.has_early_exit:
        return False

    # Rule 4b — return-None specificity (strict idempotency only)
    # When the target has a bare ``return None`` and there is no other
    if (
        strict_call_matching
        and target_sig.returns_none
        and not has_discriminator
        and not stmt_sig.returns_none
    ):
        return False

    # Rule 5 — condition anchor overlap (If-specific)
    # If the target's condition references any anchor names, the candidate's
    if (target_sig.condition_names and anchor_bases
            and (target_sig.condition_names & anchor_bases)):
        cond_anchor_overlap = target_sig.condition_names & anchor_bases
        if not (stmt_sig.condition_names & cond_anchor_overlap):
            return False

    # Rule 6 — condition attribute matching (strict idempotency only)
    # When checking "is this EXACT guard already present?" (strict_call_matching=True),
    if strict_call_matching and target_sig.condition_attrs:
        if not (stmt_sig.condition_attrs & target_sig.condition_attrs):
            return False

    return True


def _signature_in_subtree(
    node: ast.AST,
    target_sig: _TargetSignature,
    anchor_bases: set,
    *,
    strict_call_matching: bool = False,
) -> bool:
    """Recursively search compound blocks for a signature-satisfying stmt."""
    for field_name in ("body", "orelse", "finalbody", "handlers"):
        stmts = getattr(node, field_name, None)
        if not isinstance(stmts, list):
            continue
        for s in stmts:
            if isinstance(s, ast.ExceptHandler):
                if _signature_in_subtree(s, target_sig, anchor_bases,
                                         strict_call_matching=strict_call_matching):
                    return True
                continue
            if _stmt_satisfies_signature(s, target_sig, anchor_bases,
                                         strict_call_matching=strict_call_matching):
                return True
            if _signature_in_subtree(s, target_sig, anchor_bases,
                                     strict_call_matching=strict_call_matching):
                return True
    return False


def _verify_after_anchor(
    func: ast.AST,
    contract: PlacementContract,
    *,
    strict_call_matching: bool = False,
) -> tuple[bool, str]:
    """Verify that, after every anchor assignment, a statement exists
    whose structural signature matches the contract's target.

    This is a **semantic** check (see ``_TargetSignature``).
    ``contract.target_statement`` is treated as a representative hint of
    what the payload looks like, not as an AST-identity spec.  Identity
    matching would produce false-positive REJECTs whenever the LLM
    phrased the same guard differently (e.g. added an ``X and ...``
    short-circuit in front of the same condition).

    ``strict_call_matching=True`` is forwarded to ``_stmt_satisfies_signature``
    for pre-edit idempotency checks — see its docstring for details.
    """
    try:
        target_tree = ast_cache.parse_cached(contract.target_statement) if contract.target_statement else None
    except SyntaxError:
        return False, "target statement not parseable"
    if target_tree is None or not target_tree.body:
        return False, "target statement empty"

    target_sig = _extract_target_signature(target_tree.body[0])

    anchor_names = contract.anchor_names
    if not anchor_names:
        return True, "no anchors to verify"

    # Dotted anchors (``x.attr``) use the base name for Load matching.
    anchor_bases = {n.split(".", 1)[0] for n in anchor_names}

    mode = contract.verification.mode

    def _pred(s: ast.stmt) -> bool:
        return _stmt_satisfies_signature(s, target_sig, anchor_bases,
                                         strict_call_matching=strict_call_matching)

    path, guard_idx = _find_stmt_block_path(func, _pred)
    if not path or guard_idx < 0:
        return False, "no statement matching target signature found in function"

    strength_req = "effective"
    for anchor in contract.anchors:
        if anchor.strength == "any":
            strength_req = "any"
            break

    anchor_matches = []
    unfindable_anchors: list = []
    for name in anchor_names:
        am = _find_nearest_anchor(path, guard_idx, name, strength_req)
        if am is not None:
            anchor_matches.append(am)
            continue
        # Distinguish three None reasons by scanning only the ancestor scope:
        #   (a) anchor has an 'effective' assignment, but after the
        presence = _strongest_assignment_in_scope(path, guard_idx, name)
        if presence == "effective":
            assign_ln = _first_effective_assignment_lineno(func, name)
            ln_hint = f" first assigned at line {assign_ln}" if assign_ln else ""
            return False, f"'{name}' assigned after target (ordering violation{ln_hint}; insert AFTER line {assign_ln})"
        if presence in ("update", "weak"):
            # Both augmented-assign (update) and sentinel-init (weak) patterns
            # may be valid anchors when they appear BEFORE the target.
            am_relaxed = _find_nearest_anchor(path, guard_idx, name, "any")
            if am_relaxed is not None:
                anchor_matches.append(am_relaxed)
                continue
            # The assign exists somewhere in the function but not before the
            # target → genuine ordering violation.
            kind_str = "augmented" if presence == "update" else "sentinel"
            return False, (
                f"no effective assignment found for '{name}' before target "
                f"({kind_str} assign exists but appears after the insertion point)"
            )
        unfindable_anchors.append(name)

    if not anchor_matches:
        # No anchor has a function-scope assignment at all. The
        # signature match in ``_pred`` already proved the stmt loads
        return True, (
            f"PASS: signature match (anchors are comprehension/param-level: "
            f"{unfindable_anchors})"
        )

    # max-of-anchors ordering (see build_after_assignment_contract docstring).
    # Lexicographic comparison over (frame_depth, index_in_frame) picks the
    latest = max(anchor_matches, key=lambda a: (a.frame_depth, a.index_in_frame))

    if mode == "strict":
        next_idx = latest.index_in_frame + 1
        if next_idx >= len(latest.frame.stmts):
            return False, "no statement after anchor"
        if _pred(latest.frame.stmts[next_idx]):
            return True, "PASS strict: target signature satisfied immediately after anchor"
        return False, "FAIL strict: target signature not immediately after anchor"

    # relaxed: allow intermediate non-assignment statements, and also
    # search inside compound statements (if/for/while/try/with) for a
    # signature-satisfying stmt.
    for j in range(latest.index_in_frame + 1, len(latest.frame.stmts)):
        stmt = latest.frame.stmts[j]

        if _pred(stmt):
            return True, f"PASS relaxed: signature match at offset +{j - latest.index_in_frame}"

        if _signature_in_subtree(stmt, target_sig, anchor_bases,
                                  strict_call_matching=strict_call_matching):
            return True, f"PASS relaxed: signature match nested at offset +{j - latest.index_in_frame}"

        if contract.constraints.forbid_reassignment_before_use:
            for am in anchor_matches:
                if _assign_strength(stmt, am.name) not in ("none", "weak"):
                    return False, f"FAIL relaxed: reassignment of '{am.name}' before target"

    return False, "FAIL relaxed: no signature match after anchor"


def _verify_at_function_entry(func: ast.AST, contract: PlacementContract) -> tuple[bool, str]:
    """Verify that target statement is the first executable statement."""
    target_norm = _normalize_stmt(contract.target_statement)
    if not target_norm:
        return False, "target statement not parseable"

    body = func.body
    if not body:
        return False, "function has no body"

    # Skip docstring
    start_idx = 0
    if (isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        start_idx = 1

    if start_idx >= len(body):
        return False, "function body has only docstring"

    if _stmt_matches(body[start_idx], target_norm):
        return True, "PASS: target is first executable statement"

    return False, "FAIL: target is not at function entry"


def _verify_before_return(
    func: ast.AST,
    contract: PlacementContract,
    *,
    strict_call_matching: bool = False,
) -> tuple[bool, str]:
    """Verify that target statement appears before at least one return.

    Two paths depending on whether the contract carries anchors:

    * **No anchors** (legacy): exact AST-identity match (``_stmt_matches``)
      with the target statement, and the next immediate sibling must be
      an ``ast.Return``. This is intentionally strict — it preserves the
      historical no-anchor contract behavior so callers that opted out
      of auto-extract don't regress.
    * **With anchors** (default under ``auto_extract_uses=True``): uses
      the same signature-based matching as ``after_anchor`` so that
      paraphrased payloads still validate, then enforces
      (a) max-of-anchors ordering — every anchor has an effective
      definition before the target — and (b) at least one ``ast.Return``
      with ``lineno`` strictly greater than the target's line. The
      return may live in an outer block (e.g. the target is inside a
      ``try`` and the return follows the block), which is a common
      cleanup/logging pattern.
    """
    # Legacy path: no anchors → exact-match + immediate-next-return.
    if not contract.anchor_names:
        target_norm = _normalize_stmt(contract.target_statement)
        if not target_norm:
            return False, "target statement not parseable"

        def _check_block(stmts: list) -> bool:
            for i, s in enumerate(stmts):
                if _stmt_matches(s, target_norm):
                    for j in range(i + 1, len(stmts)):
                        if isinstance(stmts[j], ast.Return):
                            return True
                        break  # only check immediate next
            return False

        def _walk_blocks(node) -> bool:
            for field_name in ("body", "orelse", "finalbody"):
                stmts = getattr(node, field_name, None)
                if isinstance(stmts, list):
                    if _check_block(stmts):
                        return True
                    for s in stmts:
                        if _walk_blocks(s):
                            return True
            return False

        if _walk_blocks(func):
            return True, "PASS: target appears before return"
        return False, "FAIL: target not found before any return statement"

    # Anchor-aware path: signature match + max-of-anchors ordering +
    # return-after-target in the enclosing function.
    try:
        target_tree = ast_cache.parse_cached(contract.target_statement) if contract.target_statement else None
    except SyntaxError:
        return False, "target statement not parseable"
    if target_tree is None or not target_tree.body:
        return False, "target statement empty"

    target_sig = _extract_target_signature(target_tree.body[0])
    anchor_names = contract.anchor_names
    anchor_bases = {n.split(".", 1)[0] for n in anchor_names}

    def _pred(s: ast.stmt) -> bool:
        return _stmt_satisfies_signature(s, target_sig, anchor_bases,
                                         strict_call_matching=strict_call_matching)

    path, target_idx = _find_stmt_block_path(func, _pred)
    if not path or target_idx < 0:
        return False, "no statement matching target signature found in function"

    strength_req = "effective"
    for anchor in contract.anchors:
        if anchor.strength == "any":
            strength_req = "any"
            break

    anchor_matches = []
    unfindable_anchors: list = []
    for name in anchor_names:
        am = _find_nearest_anchor(path, target_idx, name, strength_req)
        if am is not None:
            anchor_matches.append(am)
            continue
        presence = _strongest_assignment_in_scope(path, target_idx, name)
        if presence == "effective":
            assign_ln = _first_effective_assignment_lineno(func, name)
            ln_hint = f" first assigned at line {assign_ln}" if assign_ln else ""
            return False, f"'{name}' assigned after target (ordering violation{ln_hint}; insert AFTER line {assign_ln})"
        if presence in ("update", "weak"):
            am_relaxed = _find_nearest_anchor(path, target_idx, name, "any")
            if am_relaxed is not None:
                anchor_matches.append(am_relaxed)
                continue
            kind_str = "augmented" if presence == "update" else "sentinel"
            return False, (
                f"no effective assignment found for '{name}' before target "
                f"({kind_str} assign exists but appears after the insertion point)"
            )
        unfindable_anchors.append(name)

    # max-of-anchors is implicitly enforced: _find_nearest_anchor returned
    # the nearest dominating assignment for each name, and we have
    # already rejected any post-target assignment above. The remaining
    # check is the "before-return" side.
    target_stmt = path[-1].stmts[target_idx]
    target_line = getattr(target_stmt, "lineno", 0) or 0

    for n in ast.walk(func):
        if isinstance(n, ast.Return):
            ret_line = getattr(n, "lineno", 0) or 0
            if ret_line > target_line:
                if unfindable_anchors:
                    return True, (
                        f"PASS: signature+return match "
                        f"(param/comprehension anchors: {unfindable_anchors})"
                    )
                return True, "PASS: target before return (anchor ordering satisfied)"

    return False, "FAIL: no return statement appears after target"


def _verify_inside_block(func: ast.AST, contract: PlacementContract) -> tuple[bool, str]:
    """Verify that target statement is inside a specific block type.

    Supported block_types from verification.assertion_type == "INSIDE_BLOCK":
      - "try" → ast.Try (or ast.TryStar)
      - "for" → ast.For / ast.AsyncFor
      - "while" → ast.While
      - "if" → ast.If
      - "with" → ast.With / ast.AsyncWith
    """
    target_norm = _normalize_stmt(contract.target_statement)
    if not target_norm:
        return False, "target statement not parseable"

    # P6: block_type 1st-class field preferred; legacy assertion_type string is fallback.
    block_type = contract.verification.block_type
    if not block_type:
        _assertion = contract.verification.assertion_type or "INSIDE_BLOCK:try"
        block_type = _assertion.split(":", 1)[1] if ":" in _assertion else "try"

    _BLOCK_TYPE_MAP = {
        "try": (ast.Try,),
        "for": (ast.For, ast.AsyncFor),
        "while": (ast.While,),
        "if": (ast.If,),
        "with": (ast.With, ast.AsyncWith),
    }
    required_types = _BLOCK_TYPE_MAP.get(block_type, (ast.Try,))

    def _search_in_node(node: ast.AST) -> bool:
        """Recursively search for target inside required block types."""
        for field_name in ("body", "orelse", "finalbody", "handlers"):
            stmts = getattr(node, field_name, None)
            if not isinstance(stmts, list):
                continue
            for s in stmts:
                if isinstance(s, required_types):
                    # This is the block we're looking for — search inside it
                    if _target_in_subtree(s, target_norm):
                        return True
                    # Also check direct body
                    for bf in ("body", "orelse", "finalbody", "handlers"):
                        inner = getattr(s, bf, None)
                        if isinstance(inner, list):
                            for inner_s in inner:
                                if _stmt_matches(inner_s, target_norm):
                                    return True
                # Recurse into nested structures
                if _search_in_node(s):
                    return True
        return False

    if _search_in_node(func):
        return True, f"PASS: target found inside {block_type} block"

    # Fuzzy fallback: when the LLM adapts the guard to use the actual loop
    # variable (e.g. guard_statement='if not path.name: continue' but code
    _anchor_names_flat: list[str] = list(contract.anchor_names or [])
    if not _anchor_names_flat:
        _anchor_names_flat = [a.name for a in (contract.anchors or []) if a.name]

    _pcl_fuzzy_blocked = False
    _pcl_target_stmt = contract.target_statement or ""
    if _pcl_target_stmt:
        try:
            _pcl_gs_tree = ast.parse(_pcl_target_stmt.strip(), mode="exec")
            _pcl_attr_pairs = [
                (_nd.value.id, _nd.attr)
                for _nd in ast.walk(_pcl_gs_tree)
                if isinstance(_nd, ast.Attribute) and isinstance(_nd.value, ast.Name)
            ]
            if _pcl_attr_pairs:
                _pcl_func_names: set = {
                    _nd.id for _nd in ast.walk(func) if isinstance(_nd, ast.Name)
                }
                for _pcl_obj, _pcl_attr in _pcl_attr_pairs:
                    if _pcl_obj not in _pcl_func_names:
                        _pcl_fuzzy_blocked = True
                        logger.info(
                            "[PCL_FUZZY_BLOCKED] attribute base %r not in function scope "
                            "— fuzzy PASS disabled (guard scope mismatch, "
                            "target_stmt=%r)",
                            _pcl_obj, _pcl_target_stmt,
                        )
                        break
        except Exception:
            pass  # non-critical — never block execution

    def _guard_op_class(expr: ast.expr) -> str:
        """Classify root operator of a guard condition into a broad category string."""
        if isinstance(expr, ast.UnaryOp):
            return f"unary_{type(expr.op).__name__.lower()}"
        if isinstance(expr, ast.BoolOp):
            return f"bool_{type(expr.op).__name__.lower()}"
        if isinstance(expr, ast.Compare) and expr.ops:
            return f"cmp_{type(expr.ops[0]).__name__.lower()}"
        return f"other_{type(expr).__name__.lower()}"

    _target_op_class: str = ""
    try:
        _g = ast.parse(contract.target_statement or "")
        if _g.body and isinstance(_g.body[0], ast.If):
            _target_op_class = _guard_op_class(_g.body[0].test)
    except (SyntaxError, TypeError, AttributeError):
        pass

    def _name_in_expr(expr: ast.expr, names: list[str]) -> bool:
        """True if any identifier in names appears in expr (AST-level check)."""
        for _n in ast.walk(expr):
            if isinstance(_n, ast.Name) and _n.id in names:
                return True
            if isinstance(_n, ast.Attribute) and _n.attr in names:
                return True
        return False

    def _has_anchor_guard_in_block(node: ast.AST) -> bool:
        """Recursively look for any if-continue matching anchor + operator class."""
        for _fn in ("body", "orelse", "finalbody", "handlers"):
            stmts = getattr(node, _fn, None)
            if not isinstance(stmts, list):
                continue
            for s in stmts:
                if isinstance(s, required_types):
                    for _sub in ast.walk(s):
                        if isinstance(_sub, ast.If):
                            _has_continue = any(
                                isinstance(_b, ast.Continue) for _b in _sub.body
                            )
                            if not _has_continue:
                                continue
                            if not _name_in_expr(_sub.test, _anchor_names_flat):
                                continue
                            # Root operator class must agree with target guard
                            if _target_op_class and _guard_op_class(_sub.test) != _target_op_class:
                                continue
                            return True
                if _has_anchor_guard_in_block(s):
                    return True
        return False

    if not _pcl_fuzzy_blocked and _anchor_names_flat and _has_anchor_guard_in_block(func):
        logger.info(
            "[PCL_INSIDE_BLOCK_FUZZY] exact match failed but fuzzy anchor guard found "
            "inside %s block (anchors=%s op_class=%r) — treating as PASS",
            block_type, _anchor_names_flat, _target_op_class,
        )
        return True, (
            f"PASS: fuzzy anchor guard found inside {block_type} block "
            f"(anchors={_anchor_names_flat} op_class={_target_op_class!r})"
        )

    # Diagnostic: on FAIL, emit the closest-matching statement actually
    # present inside any block of the required type so debuggers can
    # see whether the LLM paraphrased the guard (strict _stmt_matches
    # miss) vs genuinely placed it outside any loop.
    try:
        import difflib as _dl_pcl
        _candidates: list[str] = []

        def _collect_stmts(node: ast.AST) -> None:
            for field_name in ("body", "orelse", "finalbody", "handlers"):
                stmts = getattr(node, field_name, None)
                if not isinstance(stmts, list):
                    continue
                for s in stmts:
                    if isinstance(s, required_types):
                        for bf in ("body", "orelse", "finalbody", "handlers"):
                            inner = getattr(s, bf, None)
                            if isinstance(inner, list):
                                for inner_s in inner:
                                    # inner_s is an AST node — unparse directly
                                    try:
                                        _norm = ast.unparse(inner_s)
                                        if _norm:
                                            _candidates.append(_norm)
                                    except (SyntaxError, TypeError, AttributeError):
                                        pass
                    _collect_stmts(s)

        _collect_stmts(func)
        _close = _dl_pcl.get_close_matches(
            target_norm, _candidates, n=3, cutoff=0.5,
        ) if _candidates else []
        logger.warning(
            "[PCL_INSIDE_BLOCK_MISS] target_norm=%r block_type=%r "
            "stmt_count_in_blocks=%d close_matches=%s",
            target_norm[:120], block_type, len(_candidates),
            [m[:120] for m in _close],
        )
    except Exception as _diag_exc:
        logger.debug("[PCL_INSIDE_BLOCK_MISS] diagnostics failed: %s", _diag_exc)

    _scope_tag = " [anchor_scope_mismatch]" if _pcl_fuzzy_blocked else ""
    return False, f"FAIL: target not found inside any {block_type} block{_scope_tag}"


# ---------------------------------------------------------------------------
# Utility: extract contract from operation metadata
# ---------------------------------------------------------------------------

def get_placement_contract(metadata: Optional[dict[str, Any]]) -> Optional[PlacementContract]:
    """Extract PlacementContract from operation/instruction metadata.

    Returns None if no contract exists. Handles both dict and dataclass forms.
    """
    if not metadata:
        return None
    raw = metadata.get("placement_contract")
    if raw is None:
        return None
    if isinstance(raw, PlacementContract):
        return raw
    if isinstance(raw, dict):
        return PlacementContract.from_dict(raw)
    return None


# ---------------------------------------------------------------------------
# Placement feasibility pre-flight
# ---------------------------------------------------------------------------

def precheck_placement_feasibility(
    source_code: str,
    target_symbol: str,
    contract: PlacementContract,
) -> tuple[bool, str]:
    """Pre-flight: verify the placement LOCATION exists before an LLM call.

    ``verify_placement_contract`` answers *"is the target statement placed
    correctly?"* — it needs the target statement, which only exists after
    the developer LLM has emitted code. This function answers the strictly
    weaker, pre-generation question *"is it structurally possible for any
    target statement to satisfy this contract?"* by checking only the
    PLACEMENT LOCATION requirements against the current function body.

    Returns (True, reason) when feasible (or when we cannot conclusively
    reject). Returns (False, reason) only when we can prove no LLM output
    could satisfy the contract — e.g. the contract requires the guard to
    sit inside a ``for`` loop but the target function contains no for
    loops at all.

    This is a conservative check: a True result does not guarantee
    success, but a False result guarantees failure without burning an
    LLM call (SL28 symptom: two 60s LLM retries that both hit
    ``placement_violation`` on the same unsatisfiable anchor).
    """
    if contract is None:
        return True, "no contract"
    try:
        tree = ast_cache.parse_cached(source_code)
    except Exception as e:
        # Don't fail-fast on parse errors — let the normal flow surface
        # the syntax issue via verify_placement_contract.
        return True, f"parse_error (soft-pass): {e}"

    func = _find_target_function(tree, target_symbol)
    if func is None:
        # Symbol absence is handled by the caller's own not-found path.
        return True, f"target function '{target_symbol}' not found (soft-pass)"

    if contract.kind == "inside_block":
        # P6: block_type 1st-class field preferred; legacy format fallback.
        block_type = contract.verification.block_type
        if not block_type:
            _assertion = contract.verification.assertion_type or "INSIDE_BLOCK:try"
            block_type = _assertion.split(":", 1)[1] if ":" in _assertion else "try"
        _BLOCK_TYPE_MAP = {
            "try": (ast.Try,),
            "for": (ast.For, ast.AsyncFor),
            "while": (ast.While,),
            "if": (ast.If,),
            "with": (ast.With, ast.AsyncWith),
        }
        required_types = _BLOCK_TYPE_MAP.get(block_type)
        if not required_types:
            return True, f"unknown block_type {block_type} (soft-pass)"

        _matching_blocks = [
            n for n in ast.walk(func) if isinstance(n, required_types)
        ]
        if not _matching_blocks:
            return False, (
                f"placement infeasible: contract requires target inside a "
                f"'{block_type}' block, but '{target_symbol}' contains no "
                f"{block_type} statement"
            )

        # Iter-target check is gated on the contract's declared anchor_role
        # (ITER_VAR), NOT on ``block_type == "for"``. The role is declared
        _anchor_names = contract.anchor_names or []
        if _anchor_names and contract.anchor_role == AnchorRole.ITER_VAR.value:
            _iter_targets: set[str] = set()
            for n in _matching_blocks:
                _target = getattr(n, "target", None)
                if _target is None:
                    continue
                for sub in ast.walk(_target):
                    if isinstance(sub, ast.Name):
                        _iter_targets.add(sub.id)
            _matched = [a for a in _anchor_names if a in _iter_targets]
            if not _matched:
                return False, (
                    f"placement infeasible: contract requires target inside a "
                    f"for-loop whose iteration variable is one of "
                    f"{_anchor_names}, but '{target_symbol}' has for-loop "
                    f"targets {sorted(_iter_targets) if _iter_targets else '[]'}"
                )
        return True, (
            f"{len(_matching_blocks)} '{block_type}' block(s) present"
            + (f"; matched anchors={contract.anchor_names}" if _anchor_names else "")
        )

    if contract.kind == "after_anchor":
        _anchor_names = contract.anchor_names or []
        if not _anchor_names:
            return True, "no anchor names to check"
        try:
            _src_segment = ast.get_source_segment(source_code, func) or ""
        except Exception:
            _src_segment = source_code  # non-critical — never block execution
        _missing = [a for a in _anchor_names if a not in _src_segment]
        if len(_missing) == len(_anchor_names):
            return False, (
                f"placement infeasible: none of the anchors {_anchor_names} "
                f"appear anywhere in '{target_symbol}' body"
            )
        return True, (
            f"{len(_anchor_names) - len(_missing)}/{len(_anchor_names)} "
            f"anchors present in body"
        )

    # after_assignment / before_return / at_function_entry have no pre-flight
    # structural requirement that can be checked without the target statement.
    return True, f"no pre-flight check for kind={contract.kind}"


# ---------------------------------------------------------------------------
# Planning helper: resolve a concrete insertion line from PCL anchors
# ---------------------------------------------------------------------------

@dataclass
class AnchorInsertionPoint:
    """Resolved placement for an ``after_anchor`` PCL contract — **function body** scope.

    Distinct from :class:`ast_placement_engine.InsertionPoint`, which
    models module/class-level insertion (different fields, different
    semantics). The old shared name ``InsertionPoint`` collided with that
    module-level type; importing both under the same name in one file
    silently shadowed one definition, so this type was renamed to
    ``AnchorInsertionPoint`` to make the scope explicit.

    Splits two concepts that used to be conflated under "anchor line":

    * ``after_line`` — the 1-indexed line *after* which the new code
      should be inserted.  This is the file-level position.
    * ``body_indent`` — the indentation (in spaces) that the inserted
      statement must start with.  This is the *scope* the insertion
      belongs to, which is NOT the indent of ``after_line`` whenever
      block-scope promotion has occurred (e.g. ``after_line`` is the
      last line of a try/except, deeply indented, but the new guard
      belongs to the enclosing function body).
    * ``promoted_from_block`` — True when the returned point was
      promoted out of a compound block (try/if/for/while/with) rather
      than taken verbatim from a direct-child assignment.

    Callers MUST honor ``body_indent`` rather than re-computing indent
    from ``after_line``'s source; the whole point of this struct is that
    those two disagree.
    """
    after_line: int
    body_indent: int
    promoted_from_block: bool = False


def resolve_after_anchor_insertion(
    source_code: str,
    target_symbol: str,
    anchor_names: list[str],
) -> Optional[AnchorInsertionPoint]:
    """Resolve where to insert a guard for an ``after_anchor`` contract.

    Used by the large-symbol redirect path to translate a PCL contract
    into a concrete line + indent so ``_handle_anchor_edit`` can place
    the payload without re-deriving indentation from whatever statement
    happens to sit at the anchor line.

    **Anchor line vs. insertion scope.** For statements that are direct
    children of the function body, the two coincide: the insertion goes
    right after the assignment, at the function-body indent, and the
    assignment's own line has that same indent.  But when an assignment
    lives inside a compound block (``try``/``if``/``for``/``while``/
    ``with``), we bubble up to the block's ``end_lineno`` so the
    inserted code sits *after* the whole block — and the correct
    indent is then the function body's indent, not the deep indent of
    whatever statement closes the block (often an ``except`` body line
    or a ``return``).

    Returns ``None`` when the function cannot be located, the function
    body is empty, or no effective assignment for any anchor name is
    found.
    """
    if not source_code or not target_symbol or not anchor_names:
        return None
    try:
        tree = ast_cache.parse_cached(source_code)
    except SyntaxError:
        return None

    func = _find_target_function(tree, target_symbol)
    if func is None or not func.body:
        return None

    # Function body indent = col_offset of the first body statement.
    # (Docstring Expr also has a col_offset matching body indent.)
    body_indent = int(getattr(func.body[0], "col_offset", 0) or 0)

    best_line: Optional[int] = None
    best_promoted: bool = False

    def _scan(stmts: list, enclosing_end: Optional[int]) -> None:
        """Walk ``stmts``. ``enclosing_end`` is the end_lineno of the
        outermost compound block of the function body that contains
        these statements; ``None`` while iterating the function body
        itself. When an effective assignment is hit we record
        ``enclosing_end`` (block-promotion) if set, else the
        statement's own end line.
        """
        nonlocal best_line, best_promoted
        for s in stmts:
            for name in anchor_names:
                strength = _assign_strength(s, name)
                if strength in ("effective", "update"):
                    if enclosing_end is not None:
                        line = enclosing_end
                        promoted = True
                    else:
                        line = getattr(s, "end_lineno", None) or getattr(s, "lineno", None)
                        promoted = False
                    if line is not None and (best_line is None or line > best_line):
                        best_line = line
                        best_promoted = promoted
                    break  # one record per statement

            # Recurse into child containers.  The first time we descend
            # below a function-body statement we freeze enclosing_end to
            # *that* compound's end_lineno so deeper nesting keeps
            # promoting to the outermost block.
            child_end = enclosing_end if enclosing_end is not None else getattr(s, "end_lineno", None)
            for field_name in ("body", "orelse", "finalbody"):
                nested = getattr(s, field_name, None)
                if isinstance(nested, list):
                    _scan(nested, child_end)
            handlers = getattr(s, "handlers", None)
            if isinstance(handlers, list):
                for h in handlers:
                    h_body = getattr(h, "body", None)
                    if isinstance(h_body, list):
                        _scan(h_body, child_end)

    _scan(func.body, None)
    if best_line is None:
        return None
    return AnchorInsertionPoint(
        after_line=best_line,
        body_indent=body_indent,
        promoted_from_block=best_promoted,
    )


def find_last_effective_assignment_lineno(
    source_code: str,
    target_symbol: str,
    anchor_names: list[str],
) -> Optional[int]:
    """Thin wrapper returning only the ``after_line`` of
    :func:`resolve_after_anchor_insertion`, preserved for callers that
    only need a line number. New code should use
    :func:`resolve_after_anchor_insertion` to also get the correct
    ``body_indent``.
    """
    ip = resolve_after_anchor_insertion(source_code, target_symbol, anchor_names)
    return ip.after_line if ip is not None else None


def find_pre_return_lineno(
    source_code: str,
    function_name: str,
) -> Optional[AnchorInsertionPoint]:
    """Find the insertion point immediately before the last return in a function.

    Used by the large-symbol anchor_edit redirect to position 'store field before
    return' insertions accurately — avoids the default "insert at function top"
    behaviour of the bare ``def func(`` anchor.

    Returns an AnchorInsertionPoint with:
        after_line  — line number of the statement just before the last return
                      (insert *after* this line, i.e. between it and the return)
        body_indent — function body indentation level (col_offset of first stmt)
        promoted_from_block — always False (return is a top-level body statement)

    Returns None when:
        - The function cannot be located in the source.
        - The function body has no return statement (e.g. returns None implicitly).
        - The return is the very first body statement (nothing to insert before it).
    """
    if not source_code or not function_name:
        return None
    try:
        tree = ast_cache.parse_cached(source_code)
    except SyntaxError:
        return None

    func = _find_target_function(tree, function_name)
    if func is None or not func.body:
        return None

    body_indent = int(getattr(func.body[0], "col_offset", 0) or 0)

    # Collect top-level return statements only (not nested inside if/for/try).
    last_return: Optional[ast.Return] = None
    for stmt in func.body:
        if isinstance(stmt, ast.Return):
            last_return = stmt

    if last_return is None:
        return None

    # Find the statement immediately before the last return (same top-level body).
    prev_stmt: Optional[ast.stmt] = None
    for stmt in func.body:
        if stmt is last_return:
            break
        prev_stmt = stmt

    if prev_stmt is None:
        # Return is the very first body statement — nowhere to insert before it.
        return None

    after_line = int(getattr(prev_stmt, "end_lineno", None) or getattr(prev_stmt, "lineno", 0))
    return AnchorInsertionPoint(
        after_line=after_line,
        body_indent=body_indent,
        promoted_from_block=False,
    )
