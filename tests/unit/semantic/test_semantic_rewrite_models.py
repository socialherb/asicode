"""Tests for semantic_rewrite_models.py."""
from external_llm.editor.semantic.semantic_rewrite_models import (
    RewriteOperation,
    RewriteOpType,
    RewritePlan,
    RewriteResult,
)


class TestRewriteOpType:
    def test_constants(self):
        assert RewriteOpType.REORDER_CALLS == "reorder_calls"
        assert RewriteOpType.REPLACE_CALL_ARGS == "replace_call_args"
        assert RewriteOpType.REWRITE_RETURN == "rewrite_return"
        assert RewriteOpType.MOVE_STATEMENT == "move_statement"


class TestRewriteOperation:
    def test_creation(self):
        op = RewriteOperation(
            op_type=RewriteOpType.REORDER_CALLS,
            target_function="create_user",
            payload={"order": ["get_user", "hash_password"]},
        )
        assert op.op_type == "reorder_calls"
        assert op.target_function == "create_user"
        assert op.payload["order"] == ["get_user", "hash_password"]

    def test_optional_fields_default(self):
        op = RewriteOperation(op_type="reorder_calls", target_function="f")
        assert op.description == ""
        assert op.contract_name == ""
        assert op.payload == {}


class TestRewritePlan:
    def test_empty_plan(self):
        plan = RewritePlan(file_path="/repo/service.py")
        assert plan.is_empty is True
        assert plan.operations == []

    def test_non_empty_plan(self):
        op = RewriteOperation(op_type="reorder_calls", target_function="f")
        plan = RewritePlan(file_path="/repo/service.py", operations=[op])
        assert plan.is_empty is False
        assert len(plan.operations) == 1


class TestRewriteResult:
    def test_default_failure(self):
        r = RewriteResult()
        assert r.success is False
        assert r.applied_ops == []
        assert r.skipped_ops == []
        assert r.files_modified == []
        assert r.error == ""

    def test_success(self):
        r = RewriteResult(
            success=True,
            applied_ops=["reorder_calls"],
            files_modified=["/repo/service.py"],
        )
        assert r.success is True
        assert "/repo/service.py" in r.files_modified

    def test_to_dict(self):
        r = RewriteResult(success=True, applied_ops=["op1"])
        d = r.to_dict()
        assert d["success"] is True
        assert "applied_ops" in d
        assert "skipped_ops" in d
        assert "files_modified" in d
