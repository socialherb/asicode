"""Shared string normalization utilities.

Provides a single SSOT for identifier/key normalization patterns
that were previously duplicated across the codebase in multiple variants.

Usage:
    from external_llm.languages._normalize import normalize_key

    key = normalize_key("Hello-World ")   # → "hello_world"
"""

_NORMALIZE_TABLE = str.maketrans(" -", "__")


def normalize_key(s: str) -> str:
    """Normalize identifier: strip, lowercase, translate spaces/dashes to underscores.

    Deliberately NOT equivalent to the legacy inline chains it replaced —
    it strips FIRST, fixing their trailing-space bug::

        s.lower().replace("-", "_").replace(" ", "_").strip()
        # legacy: "read only "  → "read_only_"  (trailing space became _,
        #         then .strip() no longer removes it → lookup miss)
        # here:   "read only "  → "read_only"

    Migration note: callers whose old chain lacked the dash/space
    substitution (e.g. bare ``.lower().replace(" ", "_")``) get MORE
    normalization than before — for persisted keys this orphans entries
    written under the old form (accepted for EMA-decayed stores, which
    self-heal; do not migrate exact-match persistent keys without a
    key-migration step).

    NOTE: Unlike plain ``.strip().lower()``, this also transforms **internal**
    spaces and dashes to underscores. Use ``strip_lower()`` when you only
    want trimming + lowercasing without internal character substitution.

    Examples:
        >>> normalize_key("Hello-World ")
        'hello_world'
        >>> normalize_key("foo/bar/baz")
        'foo/bar/baz'
        >>> normalize_key("  Mixed CASE  ")
        'mixed_case'
    """
    return s.strip().lower().translate(_NORMALIZE_TABLE)


def strip_lower(s: str) -> str:
    """Strip whitespace and lowercase — no internal character transformation.

    Use this instead of ``normalize_key()`` when you only want external
    trimming and case folding (e.g. user input, env vars) without converting
    internal spaces or dashes to underscores.

    Examples:
        >>> strip_lower("  Hello World  ")
        'hello world'
        >>> strip_lower("Read-Only")
        'read-only'
    """
    return s.strip().lower()
