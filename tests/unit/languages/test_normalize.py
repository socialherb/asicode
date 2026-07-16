"""
Unit tests for external_llm/languages/_normalize.py (P1-3 normalization SSOT).

The migration report claimed all 7 replaced sites were "equivalent to the
original pattern" — verification (2026-07-16) showed that is false for three
variant families, by design. These tests lock the INTENDED semantics and
document exactly where behavior diverges from the legacy inline chains, so a
future "simplification" cannot silently reintroduce the trailing-space bug or
un-document the persisted-key caveat.
"""
from external_llm.languages._normalize import normalize_key, strip_lower

# ─── core semantics ──────────────────────────────────────────────────────────

def test_basic_normalization():
    assert normalize_key("Hello-World") == "hello_world"
    assert normalize_key("Mixed CASE") == "mixed_case"
    assert normalize_key("already_ok") == "already_ok"


def test_strip_happens_first():
    # The whole point of the strip-first ordering: surrounding whitespace is
    # removed, not converted to underscores.
    assert normalize_key("  Hello-World  ") == "hello_world"
    assert normalize_key("read only ") == "read_only"


def test_internal_whitespace_and_dashes_translate():
    assert normalize_key("a b-c") == "a_b_c"


def test_slash_paths_pass_through():
    assert normalize_key("foo/bar/baz") == "foo/bar/baz"


def test_empty_and_whitespace_only():
    assert normalize_key("") == ""
    assert normalize_key("   ") == ""


# ─── documented divergence from the legacy chains ────────────────────────────

def test_diverges_from_legacy_trailing_space_chain():
    # Legacy variant (replace-then-strip) left a trailing underscore —
    # normalize_key deliberately fixes this. Locks the bug-fix direction.
    legacy = "read only ".lower().replace("-", "_").replace(" ", "_").strip()
    assert legacy == "read_only_"          # the legacy bug
    assert normalize_key("read only ") == "read_only"  # the fix


def test_diverges_from_bare_space_only_chain():
    # graph_failure_memory's old chain did NOT touch dashes; normalize_key
    # does. Persisted keys written under the old form are orphaned (accepted
    # for EMA-decayed stores — see the docstring migration note).
    legacy = "tree-sitter term".lower().replace(" ", "_")
    assert legacy == "tree-sitter_term"
    assert normalize_key("tree-sitter term") == "tree_sitter_term"


def test_equivalent_to_strip_first_legacy_chain():
    # execution_mode_classifier's old chain stripped first — for that variant
    # normalize_key IS a drop-in equivalent.
    for s in ("Read-Only ", "  edit text", "ANALYZE", "a b-c "):
        legacy = s.lower().strip().replace("-", "_").replace(" ", "_")
        assert normalize_key(s) == legacy


# ─── strip_lower ─────────────────────────────────────────────────────────────

def test_strip_lower_no_internal_transformation():
    assert strip_lower("  Hello World  ") == "hello world"
    assert strip_lower("Read-Only") == "read-only"
    assert strip_lower("") == ""
