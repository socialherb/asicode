"""primitive_registry.py — Phase F: 12 Generic Semantic Primitives.

Domain-independent behavioral building blocks. Each primitive describes
WHAT behavior should exist, not HOW or in which domain.
"""
from __future__ import annotations

from external_llm.editor.semantic.primitive_models import SemanticPrimitive

# ── 12 Semantic Primitives ────────────────────────────────────────────────────

INPUT_BIND = SemanticPrimitive(
    name="input_bind",
    category="io",
    description="Request/params/body bound to logic or entity fields",
    required_for_actions=["create", "update", "send", "upload", "login", "signup", "register"],
    typical_signals=[
        "keyword_arg_from_param",  # Entity(field=param)
        "param_used_in_body",      # function param referenced in logic
    ],
)

VALIDATE = SemanticPrimitive(
    name="validate",
    category="control",
    description="Input/condition/state validation",
    required_for_actions=["login", "signup", "create", "update", "delete"],
    typical_signals=[
        "validate", "validate_input", "check", "check_password",
        "verify", "verify_password", "is_valid",
        "raise.*ValueError", "raise.*HTTPException.*4",
    ],
)

LOOKUP = SemanticPrimitive(
    name="lookup",
    category="data",
    description="Existing entity/state retrieval",
    required_for_actions=["login", "update", "delete", "get"],
    typical_signals=[
        "get", "get_", "find", "find_", "load", "load_",
        "fetch", "fetch_", "lookup", "lookup_",
        "query", "filter", "select",
    ],
)

CREATE_ENTITY = SemanticPrimitive(
    name="create_entity",
    category="data",
    description="New entity instance creation",
    required_for_actions=["create", "send", "upload", "signup", "register"],
    typical_signals=[
        "ClassName(",  # Instantiation
        "= ClassName(",
    ],
)

UPDATE_ENTITY = SemanticPrimitive(
    name="update_entity",
    category="data",
    description="Existing entity modification",
    required_for_actions=["update", "edit", "modify", "patch"],
    typical_signals=[
        ".update(", "setattr(", "merge(", "entity.field =",
    ],
)

DELETE_ENTITY = SemanticPrimitive(
    name="delete_entity",
    category="data",
    description="Entity deletion/deactivation",
    required_for_actions=["delete", "remove", "cancel"],
    typical_signals=[
        ".delete(", "remove(", "pop(", "discard(", "del ", ".is_active = False",
    ],
)

PERSIST_STATE = SemanticPrimitive(
    name="persist_state",
    category="data",
    description="Save state change to storage",
    required_for_actions=["create", "update", "delete", "send", "upload", "signup"],
    typical_signals=[
        "db.add", "session.add", ".commit()", ".append(",
        ".save(", "store(", "write(", "put(", "insert(",
    ],
)

LIST_OR_QUERY = SemanticPrimitive(
    name="list_or_query",
    category="data",
    description="List/search/query result return",
    required_for_actions=["list", "browse", "search", "feed", "index", "get_all"],
    typical_signals=[
        "return.*list", "return.*[]", "query.all()",
        "filter(", ".all(", "List[",
        "list(", "search(", "find_all(",
    ],
)

AUTHORIZE = SemanticPrimitive(
    name="authorize",
    category="control",
    description="Permission/identity verification",
    required_for_actions=["login", "signup"],
    typical_signals=[
        "create_access_token", "jwt", "bearer", "token",
        "sign_token", "encode",
    ],
)

BRANCH_ON_FAILURE = SemanticPrimitive(
    name="branch_on_failure",
    category="control",
    description="Failure path blocks success output",
    required_for_actions=["login", "create", "update", "delete"],
    typical_signals=[
        "if not", "raise HTTPException", "raise ValueError",
        "return.*error", "return.*401", "return.*404",
    ],
)

PRODUCE_OUTPUT = SemanticPrimitive(
    name="produce_output",
    category="io",
    description="Return value references actual processed result",
    required_for_actions=["create", "login", "send", "upload", "update", "delete", "get", "list"],
    typical_signals=[
        "return.*entity", "return.*created", "return.*token",
        "return.*id", "return.*result",
    ],
)

DELEGATE_ACTION = SemanticPrimitive(
    name="delegate_action",
    category="structure",
    description="Route→service→repository delegation",
    required_for_actions=[],  # Not required, but recognized
    typical_signals=[
        "service.", "repository.", "handler.", "manager.",
    ],
)

# ── Extended Primitives (#9) ──────────────────────────────────────────────────
# Additional primitives for common patterns not covered by the base 12.

PAGINATE = SemanticPrimitive(
    name="paginate",
    category="data",
    description="Pagination/offset/limit/cursor on query results",
    required_for_actions=["list", "browse", "search", "feed", "index"],
    typical_signals=[
        "offset", "limit", "page", "cursor", "skip",
        "paginate", "pagination", "next_page",
    ],
)

CACHE = SemanticPrimitive(
    name="cache",
    category="data",
    description="Cache lookup/store to avoid redundant computation",
    required_for_actions=[],  # Optional enhancement
    typical_signals=[
        "cache", "redis", "memcache", "lru_cache", "cached",
        "cache_key", "get_cached", "set_cache",
    ],
)

RATE_LIMIT = SemanticPrimitive(
    name="rate_limit",
    category="control",
    description="Rate limiting / throttling guard",
    required_for_actions=[],  # Optional enhancement
    typical_signals=[
        "rate_limit", "throttle", "too_many_requests", "429",
        "bucket", "window", "limiter",
    ],
)

TRANSFORM = SemanticPrimitive(
    name="transform",
    category="io",
    description="Data transformation / serialization / mapping",
    required_for_actions=[],  # Optional enhancement
    typical_signals=[
        "serialize", "to_dict", "to_json", "schema", "map(",
        "transform", "convert", "format",
    ],
)

# ── Registry ──────────────────────────────────────────────────────────────────

ALL_PRIMITIVES: list[SemanticPrimitive] = [
    # Base 12
    INPUT_BIND,
    VALIDATE,
    LOOKUP,
    CREATE_ENTITY,
    UPDATE_ENTITY,
    DELETE_ENTITY,
    PERSIST_STATE,
    LIST_OR_QUERY,
    AUTHORIZE,
    BRANCH_ON_FAILURE,
    PRODUCE_OUTPUT,
    DELEGATE_ACTION,
    # Extended 4 (#9)
    PAGINATE,
    CACHE,
    RATE_LIMIT,
    TRANSFORM,
]

PRIMITIVE_MAP: dict[str, SemanticPrimitive] = {p.name: p for p in ALL_PRIMITIVES}


def get_required_primitives(action_type: str) -> list[SemanticPrimitive]:
    """Get primitives required for a given action type.

    When ``action_type`` is ``"unknown"``, returns a minimal set of
    domain-agnostic primitives (input_bind, produce_output) so that
    coverage is computed meaningfully instead of always 1.0 (empty
    required set).  This prevents domain mismatch — e.g. game functions
    classified as ``"create"`` (from ``_create_initial_objects``) from
    being penalized for missing web-CRUD primitives like ``validate``,
    ``persist_state``, or ``branch_on_failure``.
    """
    if action_type == "unknown":
        return [INPUT_BIND, PRODUCE_OUTPUT]

    return [p for p in ALL_PRIMITIVES if action_type in p.required_for_actions]


def get_primitive(name: str) -> SemanticPrimitive:
    """Get primitive by name."""
    return PRIMITIVE_MAP.get(name, SemanticPrimitive(name=name, category="unknown"))


# ── Action Type Inference ─────────────────────────────────────────────────────

# Function name patterns → action type.
#
# Matching rules (word-boundary):
# - Function name is split on underscores: "upload_file" → {"upload", "file"}
# - A pattern matches only if it is one of those complete segments, not a
#   substring.  This prevents "import" in "is_import_boundary" from triggering
#   "upload", and "add" in "is_add_member" from triggering "create".
# - Multi-word patterns (sign_up, get_all, …) are kept as full strings and
#   matched against the original lowercased name for backward compat.
#
# NOTE: "import" intentionally excluded from "upload" — Python AST utility
# functions often have "import" in their names (is_import_boundary, etc.) but
# they are not file-upload/data-pipeline actions.
_ACTION_PATTERNS: dict[str, list[str]] = {
    "create": ["create", "add", "new", "register", "signup"],
    "login": ["login", "authenticate", "auth"],
    "send": ["send", "publish", "emit", "broadcast"],
    "upload": ["upload", "ingest"],
    "list": ["list", "browse", "search", "feed", "index"],
    "get": ["get", "read", "fetch", "retrieve", "show", "view"],
    "update": ["update", "edit", "modify", "patch", "change"],
    "delete": ["delete", "remove", "cancel", "destroy"],
}

# Multi-word patterns that need full-name substring matching (snake_case combos)
_MULTIWORD_PATTERNS: dict[str, list[str]] = {
    "create": ["sign_up"],
    "login": ["sign_in", "signin"],
    "list": ["get_all", "find_all"],
    "get": ["detail"],
    "delete": ["drop"],
}

import re as _re


def infer_action_type(func_name: str) -> str:
    """Infer action type from function name using word-boundary segment matching.

    Splits the name on underscore/CamelCase boundaries and matches each word
    segment against the pattern vocabulary.  Substring matching is
    intentionally avoided to prevent false positives like 'import' → upload
    on compound identifiers such as is_import_boundary / find_import_boundary.

    Special rule: "import" → upload only when it is the LEADING verb segment
    (e.g. import_data, import_csv) — not a descriptive qualifier in the middle
    of a utility function name.
    """
    # Split into word segments handling snake_case, CamelCase, and ALL_CAPS.
    # Strategy: split on underscore first, then split each token on CamelCase.
    #   "is_import_boundary" → ["is", "import", "boundary"]
    #   "DeleteItem"         → ["delete", "item"]
    #   "CREATE_USER"        → ["create", "user"]
    raw_tokens = func_name.split("_")
    camel_parts: list = []
    for token in raw_tokens:
        if not token:
            continue
        if token.isupper():
            # ALL_CAPS segment: "CREATE" → "create"
            camel_parts.append(token.lower())
        else:
            # CamelCase or lowercase: "DeleteItem" → ["Delete", "Item"]
            words = _re.findall(r'[A-Z]?[a-z]+|[A-Z]+$', token) or [token]
            camel_parts.extend(w.lower() for w in words)
    segments = set(camel_parts)

    name_lower = func_name.lower()

    # Multi-word patterns checked FIRST (higher specificity wins over single-word)
    for action_type, patterns in _MULTIWORD_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower:
                return action_type

    # Segment (word-boundary) matching
    for action_type, patterns in _ACTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in segments:
                return action_type

    # "import" → upload only when it is the first (leading verb) word segment
    # Handles: import_data, import_csv → upload
    # Excludes: is_import_boundary, find_import_boundary → unknown
    if camel_parts and camel_parts[0] == "import":
        return "upload"

    return "unknown"
