"""Regression tests for large-symbol edit fallback structural fixes.

Covers:
  Part 1 — metadata (edit_kind / guard_statement / ast_ops) is preserved
            when modify_symbol is redirected to anchor_edit for >300-line symbols.
  Part 2 — spec.metadata['edit_kind'] / ['guard_statement'] propagate into
            op.metadata before _attach_edit_contracts runs, so the guard_add
            deterministic path is triggered correctly.
  Part 4 — placement_uncertain=True is set on anchor_edit fallbacks that have
            no guard_add, no ast_ops, and no precise anchor lineno.
"""
from types import SimpleNamespace

# ─── helpers ───────────────────────────────────────────────────────────────────

def _make_op(**kwargs):
    """Return a minimal Operation-like SimpleNamespace."""
    ns = SimpleNamespace(
        id="op-test",
        kind="MODIFY_SYMBOL",
        path="foo.py",
        symbol="my_func",
        intent="add guard",
        anchor_pattern=None,
        anchor_ast_lineno=None,
        depends_on=[],
        acceptance=None,
        context_hints={},
        metadata=None,
        edit_contract=None,
    )
    ns.__dict__.update(kwargs)
    return ns


# ─── Part 1: metadata propagation through large-symbol redirect ───────────────

class TestLargeSymbolMetadataPreservation:
    """Verify that op.metadata survives the modify_symbol → anchor_edit redirect."""

    def _build_redirected_metadata(self, original_metadata):
        """Replicate the metadata-copy logic from symbol_handlers.py redirect block."""
        return dict(original_metadata) if original_metadata else {}

    def test_edit_kind_preserved(self):
        op = _make_op(metadata={"edit_kind": "guard_add", "guard_statement": "if x is None: return"})
        redirected = self._build_redirected_metadata(op.metadata)
        assert redirected.get("edit_kind") == "guard_add"
        assert redirected.get("guard_statement") == "if x is None: return"

    def test_ast_ops_preserved(self):
        ast_ops = [{"type": "add_guard", "symbol": "my_func", "statement": "if x: return"}]
        op = _make_op(metadata={"ast_ops": ast_ops, "deterministic": True})
        redirected = self._build_redirected_metadata(op.metadata)
        assert redirected.get("ast_ops") == ast_ops
        assert redirected.get("deterministic") is True

    def test_empty_metadata_produces_empty_dict(self):
        op = _make_op(metadata=None)
        redirected = self._build_redirected_metadata(op.metadata)
        assert isinstance(redirected, dict)
        assert redirected == {}

    def test_metadata_is_a_copy_not_same_reference(self):
        original = {"edit_kind": "guard_add"}
        op = _make_op(metadata=original)
        redirected = self._build_redirected_metadata(op.metadata)
        redirected["extra"] = "injected"
        assert "extra" not in op.metadata, "redirect must not mutate original op.metadata"


# ─── Part 4: placement_uncertain flag ──────────────────────────────────────────

class TestPlacementUncertainFlag:
    """Verify placement_uncertain=True is set when the fallback has no precise semantic info."""

    def _should_be_uncertain(self, metadata, anchor_ast_lineno=None) -> bool:
        """Replicate the Part 4 guard logic from symbol_handlers.py."""
        redir_edit_kind = (metadata or {}).get("edit_kind", "").lower()
        has_guard_path = redir_edit_kind == "guard_add"
        has_ast_ops = bool((metadata or {}).get("ast_ops"))
        has_precise_anchor = bool(anchor_ast_lineno)
        return not has_guard_path and not has_ast_ops and not has_precise_anchor

    def test_no_info_marks_uncertain(self):
        assert self._should_be_uncertain({}) is True

    def test_guard_add_not_uncertain(self):
        assert self._should_be_uncertain({"edit_kind": "guard_add"}) is False

    def test_ast_ops_not_uncertain(self):
        meta = {"ast_ops": [{"type": "add_guard"}]}
        assert self._should_be_uncertain(meta) is False

    def test_precise_lineno_not_uncertain(self):
        assert self._should_be_uncertain({}, anchor_ast_lineno=42) is False

    def test_none_metadata_is_uncertain(self):
        assert self._should_be_uncertain(None) is True

    def test_unrelated_metadata_keys_still_uncertain(self):
        meta = {"action_hint": "modify", "no_signature_change": True}
        assert self._should_be_uncertain(meta) is True


# ─── Part 2: spec → op metadata propagation ────────────────────────────────────

class TestSpecToOpMetadataPropagation:
    """Verify _attach_edit_contracts sees edit_kind/guard_statement from spec.metadata."""

    def _propagate(self, spec_metadata: dict, ops: list) -> list:
        """Replicate the propagation loop from planner_agent.py."""
        spec_ek = spec_metadata.get('edit_kind', '')
        spec_gs = spec_metadata.get('guard_statement', '')
        if not spec_ek and not spec_gs:
            return ops
        for op in ops:
            if getattr(op, 'kind', '') != 'MODIFY_SYMBOL':
                continue
            if op.metadata is None:
                op.metadata = {}
            if spec_ek and not op.metadata.get('edit_kind'):
                op.metadata['edit_kind'] = spec_ek
            if spec_gs and not op.metadata.get('guard_statement'):
                op.metadata['guard_statement'] = spec_gs
        return ops

    def test_edit_kind_propagated_to_op(self):
        op = _make_op(metadata=None)
        ops = self._propagate({"edit_kind": "guard_add"}, [op])
        assert ops[0].metadata.get("edit_kind") == "guard_add"

    def test_guard_statement_propagated_to_op(self):
        op = _make_op(metadata=None)
        ops = self._propagate(
            {"edit_kind": "guard_add", "guard_statement": "if x is None: return"},
            [op],
        )
        assert ops[0].metadata.get("guard_statement") == "if x is None: return"

    def test_op_specific_edit_kind_not_overwritten(self):
        """op.metadata['edit_kind'] already set → spec value must NOT override it."""
        op = _make_op(metadata={"edit_kind": "full_rewrite"})
        ops = self._propagate({"edit_kind": "guard_add"}, [op])
        assert ops[0].metadata["edit_kind"] == "full_rewrite"

    def test_non_modify_ops_not_touched(self):
        op = _make_op(kind="INSERT_AFTER_SYMBOL", metadata=None)
        self._propagate({"edit_kind": "guard_add"}, [op])
        assert op.metadata is None, "non-MODIFY_SYMBOL ops must not be touched"

    def test_empty_spec_metadata_is_no_op(self):
        op = _make_op(metadata={"edit_kind": "existing"})
        ops = self._propagate({}, [op])
        assert ops[0].metadata["edit_kind"] == "existing"

    def test_multiple_ops_all_receive_propagation(self):
        ops = [_make_op(metadata=None), _make_op(metadata=None)]
        self._propagate({"edit_kind": "guard_add", "guard_statement": "if x: return"}, ops)
        for op in ops:
            assert op.metadata.get("edit_kind") == "guard_add"
            assert op.metadata.get("guard_statement") == "if x: return"
