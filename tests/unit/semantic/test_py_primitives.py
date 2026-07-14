"""Tests for Python semantic primitive detection and IR building system.

Covers:
1. PRIMITIVE_MAP (12 primitives)
2. get_required_primitives() for different action types
3. infer_action_type() from various raw requests
4. Each per-primitive detector
5. draft_parser: parse_draft() extracting actions and entities
6. IR builder: build_primitive_ir() with code that has various primitives
7. PrimitiveIR: coverage calculation, all_missing list, summary()
"""
from __future__ import annotations

import os
import tempfile
import textwrap

import pytest

from external_llm.editor.semantic.draft_parser import (
    DraftAction,
    DraftResult,
    parse_draft,
)
from external_llm.editor.semantic.primitive_detector import (
    _DETECTORS,
    _detect_authorize,
    _detect_branch_on_failure,
    _detect_create_entity,
    _detect_delegate_action,
    _detect_delete_entity,
    _detect_input_bind,
    _detect_list_or_query,
    _detect_lookup,
    _detect_persist_state,
    _detect_produce_output,
    _detect_unknown,
    _detect_update_entity,
    _detect_validate,
    detect_primitives,
)
from external_llm.editor.semantic.primitive_ir_builder import build_primitive_ir
from external_llm.editor.semantic.primitive_models import (
    PrimitiveIR,
    PrimitiveMatch,
    PrimitiveSequence,
)
from external_llm.editor.semantic.primitive_registry import (
    ALL_PRIMITIVES,
    PRIMITIVE_MAP,
    get_primitive,
    get_required_primitives,
    infer_action_type,
)
from external_llm.editor.semantic.semantic_tracer import (
    FunctionTrace,
    SemanticTrace,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_temp_py(content: str) -> str:
    """Write a temporary .py file and return its absolute path."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(content))
    return path


def _empty_trace() -> SemanticTrace:
    return SemanticTrace()


def _action(
    name: str = "do_something",
    action_type: str = "create",
    entity: str = "Item",
    params: list | None = None,
    calls: list | None = None,
    has_return: bool = False,
    has_decorator: bool = False,
) -> DraftAction:
    return DraftAction(
        name=name,
        action_type=action_type,
        entity=entity,
        params=params or [],
        calls=calls or [],
        has_return=has_return,
        has_decorator=has_decorator,
    )


# =============================================================================
# 1. PRIMITIVE_MAP has exactly 12 primitives
# =============================================================================

class TestPrimitiveRegistry:

    def test_primitive_map_has_at_least_12(self):
        # Base 12 + any composites added by evolution at runtime
        assert len(PRIMITIVE_MAP) >= 12

    def test_all_primitives_has_at_least_12(self):
        assert len(ALL_PRIMITIVES) >= 12

    def test_all_primitive_names(self):
        expected = {
            "input_bind", "validate", "lookup", "create_entity",
            "update_entity", "delete_entity", "persist_state",
            "list_or_query", "authorize", "branch_on_failure",
            "produce_output", "delegate_action",
        }
        # Base 12 must be present (composites from evolution may also exist)
        assert expected.issubset(set(PRIMITIVE_MAP.keys()))

    def test_each_primitive_has_category(self):
        valid_categories = {"io", "control", "data", "structure"}
        for p in ALL_PRIMITIVES:
            assert p.category in valid_categories, f"{p.name} has invalid category {p.category}"

    def test_get_primitive_known(self):
        p = get_primitive("validate")
        assert p.name == "validate"
        assert p.category == "control"

    def test_get_primitive_unknown_returns_fallback(self):
        p = get_primitive("nonexistent")
        assert p.name == "nonexistent"
        assert p.category == "unknown"


# =============================================================================
# 2. get_required_primitives() for different action types
# =============================================================================

class TestGetRequiredPrimitives:

    def test_create_action(self):
        prims = get_required_primitives("create")
        names = {p.name for p in prims}
        assert "input_bind" in names
        assert "create_entity" in names
        assert "persist_state" in names
        assert "produce_output" in names
        assert "validate" in names
        assert "branch_on_failure" in names

    def test_login_action(self):
        prims = get_required_primitives("login")
        names = {p.name for p in prims}
        assert "validate" in names
        assert "lookup" in names
        assert "authorize" in names
        assert "input_bind" in names
        assert "branch_on_failure" in names

    def test_update_action(self):
        prims = get_required_primitives("update")
        names = {p.name for p in prims}
        assert "update_entity" in names
        assert "persist_state" in names
        assert "lookup" in names
        assert "validate" in names

    def test_delete_action(self):
        prims = get_required_primitives("delete")
        names = {p.name for p in prims}
        assert "delete_entity" in names
        assert "lookup" in names
        assert "validate" in names
        assert "branch_on_failure" in names

    def test_list_action(self):
        prims = get_required_primitives("list")
        names = {p.name for p in prims}
        assert "list_or_query" in names
        assert "produce_output" in names

    def test_get_action(self):
        prims = get_required_primitives("get")
        names = {p.name for p in prims}
        assert "lookup" in names
        assert "produce_output" in names

    def test_unknown_action_returns_empty_or_few(self):
        prims = get_required_primitives("nonexistent_action")
        assert len(prims) == 0

    def test_delegate_not_required_for_any(self):
        """delegate_action has empty required_for_actions."""
        p = get_primitive("delegate_action")
        assert p.required_for_actions == []


# =============================================================================
# 3. infer_action_type() from various function names
# =============================================================================

class TestInferActionType:

    @pytest.mark.parametrize("name,expected", [
        ("create_user", "create"),
        ("add_item", "create"),
        ("register_account", "create"),
        ("signup", "create"),
        ("new_post", "create"),
    ])
    def test_create_variants(self, name, expected):
        assert infer_action_type(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("login", "login"),
        ("authenticate", "login"),
        ("sign_in", "login"),
    ])
    def test_login_variants(self, name, expected):
        assert infer_action_type(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("get_user", "get"),
        ("fetch_data", "get"),
        ("retrieve_order", "get"),
        ("read_config", "get"),
        ("show_profile", "get"),
    ])
    def test_read_variants(self, name, expected):
        assert infer_action_type(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("update_profile", "update"),
        ("edit_user", "update"),
        ("modify_settings", "update"),
        ("patch_user", "update"),
    ])
    def test_update_variants(self, name, expected):
        assert infer_action_type(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("delete_user", "delete"),
        ("remove_item", "delete"),
        ("cancel_order", "delete"),
        ("destroy_session", "delete"),
    ])
    def test_delete_variants(self, name, expected):
        assert infer_action_type(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("list_users", "list"),
        ("search_items", "list"),
        ("browse_catalog", "list"),
        ("find_all_orders", "list"),
        ("get_all_items", "list"),
    ])
    def test_list_variants(self, name, expected):
        assert infer_action_type(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("send_email", "send"),
        ("publish_event", "send"),
        ("emit_signal", "send"),
    ])
    def test_send_variants(self, name, expected):
        assert infer_action_type(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("upload_file", "upload"),
        ("import_data", "upload"),
    ])
    def test_upload_variants(self, name, expected):
        assert infer_action_type(name) == expected

    def test_unknown_func_name(self):
        assert infer_action_type("foobar") == "unknown"

    def test_case_insensitive(self):
        assert infer_action_type("CREATE_USER") == "create"
        assert infer_action_type("DeleteItem") == "delete"


# =============================================================================
# 4. Per-primitive detectors
# =============================================================================

class TestDetectInputBind:

    def test_with_entity_bindings_in_trace(self):
        ft = FunctionTrace(name="create_user", entity_bindings=[("name", "name")])
        trace = _empty_trace()
        action = _action(params=["name"])
        m = _detect_input_bind("input_bind", action, ft, trace)
        assert m.present is True
        assert m.confidence >= 0.9

    def test_with_params_and_calls(self):
        action = _action(params=["title", "body"], calls=["Post"])
        m = _detect_input_bind("input_bind", action, None, _empty_trace())
        assert m.present is True
        assert m.confidence >= 0.5

    def test_with_params_only(self):
        action = _action(params=["x"])
        m = _detect_input_bind("input_bind", action, None, _empty_trace())
        assert m.present is True
        assert m.confidence == 0.4

    def test_no_params(self):
        action = _action(params=[])
        m = _detect_input_bind("input_bind", action, None, _empty_trace())
        assert m.present is False


class TestDetectValidate:

    def test_validate_in_trace_calls(self):
        ft = FunctionTrace(name="login", calls={"validate", "other_call"})
        action = _action(action_type="login")
        m = _detect_validate("validate", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence >= 0.9

    def test_validate_in_action_calls(self):
        action = _action(calls=["utils.verify"])
        m = _detect_validate("validate", action, None, _empty_trace())
        assert m.present is True

    def test_check_password_in_trace(self):
        ft = FunctionTrace(name="login", calls={"check_password"})
        action = _action(action_type="login")
        m = _detect_validate("validate", action, ft, _empty_trace())
        assert m.present is True

    def test_no_validation(self):
        action = _action(calls=["save", "commit"])
        m = _detect_validate("validate", action, None, _empty_trace())
        assert m.present is False


class TestDetectLookup:

    def test_get_user_in_trace(self):
        ft = FunctionTrace(name="login", calls={"get_user"})
        action = _action(action_type="login")
        m = _detect_lookup("lookup", action, ft, _empty_trace())
        assert m.present is True

    def test_find_in_action_calls(self):
        action = _action(calls=["db.find"])
        m = _detect_lookup("lookup", action, None, _empty_trace())
        assert m.present is True

    def test_no_lookup(self):
        action = _action(calls=["save"])
        m = _detect_lookup("lookup", action, None, _empty_trace())
        assert m.present is False


class TestDetectCreateEntity:

    def test_instantiation_in_trace(self):
        ft = FunctionTrace(name="create_user", instantiations={"User"})
        action = _action(action_type="create")
        m = _detect_create_entity("create_entity", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence >= 0.9

    def test_class_call_in_action(self):
        trace = SemanticTrace(all_classes={"User"})
        action = _action(calls=["User"])
        m = _detect_create_entity("create_entity", action, None, trace)
        assert m.present is True

    def test_no_instantiation(self):
        action = _action(calls=["save"])
        m = _detect_create_entity("create_entity", action, None, _empty_trace())
        assert m.present is False


class TestDetectUpdateEntity:

    def test_update_call_in_trace(self):
        ft = FunctionTrace(name="update_profile", calls={"update"})
        action = _action(action_type="update")
        m = _detect_update_entity("update_entity", action, ft, _empty_trace())
        assert m.present is True

    def test_setattr_in_trace(self):
        ft = FunctionTrace(name="edit", calls={"setattr"})
        action = _action(action_type="update")
        m = _detect_update_entity("update_entity", action, ft, _empty_trace())
        assert m.present is True

    def test_no_update(self):
        action = _action(calls=["save"])
        m = _detect_update_entity("update_entity", action, None, _empty_trace())
        assert m.present is False


class TestDetectDeleteEntity:

    def test_delete_in_trace(self):
        ft = FunctionTrace(name="remove_item", calls={"delete"})
        action = _action(action_type="delete")
        m = _detect_delete_entity("delete_entity", action, ft, _empty_trace())
        assert m.present is True

    def test_remove_in_trace(self):
        ft = FunctionTrace(name="cancel", calls={"remove"})
        action = _action(action_type="delete")
        m = _detect_delete_entity("delete_entity", action, ft, _empty_trace())
        assert m.present is True

    def test_no_delete(self):
        action = _action(calls=["save"])
        m = _detect_delete_entity("delete_entity", action, None, _empty_trace())
        assert m.present is False


class TestDetectPersistState:

    def test_persist_calls_in_trace(self):
        ft = FunctionTrace(name="create_user", persist_calls={"db.add"})
        action = _action(action_type="create")
        m = _detect_persist_state("persist_state", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence >= 0.9

    def test_save_in_action_calls(self):
        action = _action(calls=["db.save"])
        m = _detect_persist_state("persist_state", action, None, _empty_trace())
        assert m.present is True

    def test_commit_in_action_calls(self):
        action = _action(calls=["session.commit"])
        m = _detect_persist_state("persist_state", action, None, _empty_trace())
        assert m.present is True

    def test_append_in_action_calls(self):
        action = _action(calls=["items.append"])
        m = _detect_persist_state("persist_state", action, None, _empty_trace())
        assert m.present is True

    def test_no_persist(self):
        action = _action(calls=["validate"])
        m = _detect_persist_state("persist_state", action, None, _empty_trace())
        assert m.present is False


class TestDetectListOrQuery:

    def test_filter_in_trace(self):
        ft = FunctionTrace(name="list_users", calls={"filter"})
        action = _action(action_type="list")
        m = _detect_list_or_query("list_or_query", action, ft, _empty_trace())
        assert m.present is True

    def test_all_in_trace(self):
        ft = FunctionTrace(name="list_items", calls={"all"})
        action = _action(action_type="list")
        m = _detect_list_or_query("list_or_query", action, ft, _empty_trace())
        assert m.present is True

    def test_return_names_fallback(self):
        ft = FunctionTrace(name="list_things", return_names={"items"})
        action = _action(action_type="list")
        m = _detect_list_or_query("list_or_query", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence == 0.5

    def test_no_list_pattern(self):
        action = _action(calls=["save"])
        m = _detect_list_or_query("list_or_query", action, None, _empty_trace())
        assert m.present is False


class TestDetectAuthorize:

    def test_create_access_token_in_trace(self):
        ft = FunctionTrace(name="login", calls={"create_access_token"})
        action = _action(action_type="login")
        m = _detect_authorize("authorize", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence >= 0.9

    def test_jwt_in_action_calls(self):
        action = _action(calls=["security.jwt"])
        m = _detect_authorize("authorize", action, None, _empty_trace())
        assert m.present is True

    def test_no_auth(self):
        action = _action(calls=["save"])
        m = _detect_authorize("authorize", action, None, _empty_trace())
        assert m.present is False


class TestDetectBranchOnFailure:

    def test_error_branch_before_success(self):
        ft = FunctionTrace(name="login", has_error_branch=True, error_before_success=True)
        action = _action(action_type="login")
        m = _detect_branch_on_failure("branch_on_failure", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence >= 0.9

    def test_error_branch_only(self):
        ft = FunctionTrace(name="login", has_error_branch=True, error_before_success=False)
        action = _action(action_type="login")
        m = _detect_branch_on_failure("branch_on_failure", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence == 0.6

    def test_no_error_branch(self):
        ft = FunctionTrace(name="login", has_error_branch=False)
        action = _action(action_type="login")
        m = _detect_branch_on_failure("branch_on_failure", action, ft, _empty_trace())
        assert m.present is False


class TestDetectProduceOutput:

    def test_return_has_entity_ref(self):
        ft = FunctionTrace(name="create_user", return_has_entity_ref=True)
        action = _action(action_type="create")
        m = _detect_produce_output("produce_output", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence >= 0.9

    def test_return_names_fallback(self):
        ft = FunctionTrace(name="create_user", return_names={"user"})
        action = _action(action_type="create")
        m = _detect_produce_output("produce_output", action, ft, _empty_trace())
        assert m.present is True
        assert m.confidence == 0.5

    def test_has_return_weak(self):
        action = _action(has_return=True)
        m = _detect_produce_output("produce_output", action, None, _empty_trace())
        assert m.present is True
        assert m.confidence == 0.3

    def test_no_return(self):
        action = _action(has_return=False)
        m = _detect_produce_output("produce_output", action, None, _empty_trace())
        assert m.present is False


class TestDetectDelegateAction:

    def test_service_delegation(self):
        action = _action(calls=["service.create_user"])
        m = _detect_delegate_action("delegate_action", action, None, _empty_trace())
        assert m.present is True

    def test_repository_delegation(self):
        action = _action(calls=["repository.save"])
        m = _detect_delegate_action("delegate_action", action, None, _empty_trace())
        assert m.present is True

    def test_no_delegation(self):
        action = _action(calls=["save", "validate"])
        m = _detect_delegate_action("delegate_action", action, None, _empty_trace())
        assert m.present is False


class TestDetectUnknown:

    def test_always_missing(self):
        action = _action()
        m = _detect_unknown("mystery", action, None, _empty_trace())
        assert m.present is False
        assert m.missing_reason == "unknown primitive"


class TestDetectorsRegistry:

    def test_all_12_detectors_registered(self):
        # Base 12 detectors; PRIMITIVE_MAP may have composites from evolution
        assert len(_DETECTORS) >= 12
        # All detectors must have a corresponding primitive
        for key in _DETECTORS:
            assert key in PRIMITIVE_MAP, f"detector {key!r} not in PRIMITIVE_MAP"


# =============================================================================
# 5. draft_parser: parse_draft()
# =============================================================================

class TestDraftParser:

    def test_parse_creates_actions(self):
        code = """\
        class User:
            pass

        def create_user(name: str, email: str):
            user = User()
            return user

        def get_user(user_id: int):
            return user_id
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            assert len(result.actions) >= 2
            names = [a.name for a in result.actions]
            assert "create_user" in names
            assert "get_user" in names
        finally:
            os.unlink(path)

    def test_parse_extracts_entities(self):
        code = """\
        class Product:
            pass

        class Order:
            pass

        def create_order():
            return Order()
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            assert "Product" in result.entities
            assert "Order" in result.entities
        finally:
            os.unlink(path)

    def test_parse_infers_action_type(self):
        code = """\
        def delete_item(item_id):
            pass

        def update_profile(user_id, data):
            pass
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            action_map = {a.name: a for a in result.actions}
            assert action_map["delete_item"].action_type == "delete"
            assert action_map["update_profile"].action_type == "update"
        finally:
            os.unlink(path)

    def test_parse_extracts_params(self):
        code = """\
        def create_user(name: str, email: str, age: int):
            pass
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            action = result.actions[0]
            assert set(action.params) == {"name", "email", "age"}
        finally:
            os.unlink(path)

    def test_parse_extracts_calls(self):
        code = """\
        def create_user(name):
            validate(name)
            user = User(name=name)
            db.save(user)
            return user
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            action = next(a for a in result.actions if a.name == "create_user")
            assert "validate" in action.calls
            assert "User" in action.calls
            assert "db.save" in action.calls
        finally:
            os.unlink(path)

    def test_parse_detects_return(self):
        code = """\
        def create_user():
            return "ok"

        def do_nothing():
            pass
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            action_map = {a.name: a for a in result.actions}
            assert action_map["create_user"].has_return is True
            assert action_map["do_nothing"].has_return is False
        finally:
            os.unlink(path)

    def test_parse_detects_decorator(self):
        code = """\
        def my_decorator(f):
            return f

        @my_decorator
        def create_user():
            pass
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            action_map = {a.name: a for a in result.actions}
            assert action_map["create_user"].has_decorator is True
        finally:
            os.unlink(path)

    def test_private_functions_skipped(self):
        code = """\
        def _internal_helper():
            pass

        def public_function():
            pass
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            names = [a.name for a in result.actions]
            assert "_internal_helper" not in names
            assert "public_function" in names
        finally:
            os.unlink(path)

    def test_class_methods_extracted(self):
        code = """\
        class UserService:
            def create_user(self, name):
                pass

            def __init__(self):
                pass
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            names = [a.name for a in result.actions]
            assert "create_user" in names
            assert "__init__" not in names
        finally:
            os.unlink(path)

    def test_file_role_inference(self):
        code = """\
        def create_user():
            pass
        """
        # Write to a file with "route" in the path
        fd, path = tempfile.mkstemp(suffix=".py", prefix="route_")
        with os.fdopen(fd, "w") as f:
            f.write(textwrap.dedent(code))
        try:
            result = parse_draft([path])
            assert "route" in result.files_by_role
        finally:
            os.unlink(path)

    def test_non_py_files_skipped(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("not python")
        try:
            result = parse_draft([path])
            assert len(result.actions) == 0
        finally:
            os.unlink(path)

    def test_syntax_error_file_skipped(self):
        code = "def broken(:\n"
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            assert len(result.actions) == 0
        finally:
            os.unlink(path)

    def test_entity_assignment_from_calls(self):
        code = """\
        class User:
            pass

        def create_user(name):
            user = User(name=name)
            return user
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path])
            action = next(a for a in result.actions if a.name == "create_user")
            assert action.entity == "User"
        finally:
            os.unlink(path)

    def test_context_tags_passed_through(self):
        code = """\
        def create_user():
            pass
        """
        path = _write_temp_py(code)
        try:
            result = parse_draft([path], context_tags=["web", "crud"])
            assert "web" in result.context_tags
            assert "crud" in result.context_tags
        finally:
            os.unlink(path)

    def test_draft_result_summary(self):
        dr = DraftResult(
            actions=[
                DraftAction(name="create_user", action_type="create", entity="User"),
                DraftAction(name="delete_user", action_type="delete", entity="User"),
            ],
            entities=["User"],
            files_by_role={"route": ["routes.py"]},
        )
        s = dr.summary()
        assert s["entities"] == ["User"]
        assert len(s["actions"]) == 2
        assert s["roles"] == {"route": 1}


# =============================================================================
# 6. detect_primitives() integration
# =============================================================================

class TestDetectPrimitivesIntegration:

    def test_create_action_with_full_trace(self):
        ft = FunctionTrace(
            name="create_user",
            calls={"validate", "User", "db.add", "commit"},
            instantiations={"User"},
            persist_calls={"db.add"},
            return_has_entity_ref=True,
            entity_bindings=[("name", "name")],
            has_error_branch=True,
            error_before_success=True,
        )
        trace = SemanticTrace(
            function_traces={"create_user": ft},
            all_classes={"User"},
        )
        action = _action(
            name="create_user",
            action_type="create",
            entity="User",
            params=["name", "email"],
            calls=["validate", "User", "db.add"],
            has_return=True,
        )
        seq = detect_primitives(action, trace)
        assert seq.action_name == "create_user"
        assert seq.coverage > 0.5
        present_names = seq.present_names
        assert "input_bind" in present_names
        assert "validate" in present_names
        assert "create_entity" in present_names
        assert "persist_state" in present_names
        assert "produce_output" in present_names
        assert "branch_on_failure" in present_names

    def test_login_action_missing_auth(self):
        trace = _empty_trace()
        action = _action(
            name="login",
            action_type="login",
            params=["username", "password"],
            calls=["validate"],
        )
        seq = detect_primitives(action, trace)
        assert "authorize" in seq.missing_names

    def test_unknown_action_type_with_decorator(self):
        """Actions with unknown type but has_decorator should still be processed."""
        trace = _empty_trace()
        # unknown type with decorator: detect_primitives still runs;
        # universal primitives (input_bind, produce_output) are detected as missing
        action = _action(name="custom_endpoint", action_type="unknown", has_decorator=True)
        seq = detect_primitives(action, trace)
        assert len(seq.present) == 0
        # universal primitives are detected as missing for any action type
        assert len(seq.missing) == 2


# =============================================================================
# 7. PrimitiveIR model and build_primitive_ir()
# =============================================================================

class TestPrimitiveIRModel:

    def test_empty_ir_coverage_is_1(self):
        ir = PrimitiveIR()
        assert ir.overall_coverage == 1.0

    def test_all_present_coverage_is_1(self):
        seq = PrimitiveSequence(
            action_name="test",
            action_type="create",
            present=[
                PrimitiveMatch(primitive="validate", present=True),
                PrimitiveMatch(primitive="persist_state", present=True),
            ],
            missing=[],
        )
        ir = PrimitiveIR(sequences=[seq])
        assert ir.overall_coverage == 1.0

    def test_half_missing_coverage(self):
        seq = PrimitiveSequence(
            action_name="test",
            action_type="create",
            present=[PrimitiveMatch(primitive="validate", present=True)],
            missing=[PrimitiveMatch(primitive="persist_state", present=False)],
        )
        ir = PrimitiveIR(sequences=[seq])
        assert ir.overall_coverage == 0.5

    def test_all_missing_from_multiple_sequences(self):
        seq1 = PrimitiveSequence(
            action_name="a",
            action_type="create",
            missing=[PrimitiveMatch(primitive="validate", present=False)],
        )
        seq2 = PrimitiveSequence(
            action_name="b",
            action_type="update",
            missing=[
                PrimitiveMatch(primitive="validate", present=False),
                PrimitiveMatch(primitive="persist_state", present=False),
            ],
        )
        ir = PrimitiveIR(sequences=[seq1, seq2])
        assert set(ir.all_missing) == {"validate", "persist_state"}
        # all_missing preserves order and deduplicates
        assert ir.all_missing[0] == "validate"

    def test_summary_structure(self):
        seq = PrimitiveSequence(
            action_name="create_user",
            action_type="create",
            entity="User",
            present=[PrimitiveMatch(primitive="validate", present=True)],
            missing=[PrimitiveMatch(primitive="persist_state", present=False)],
        )
        ir = PrimitiveIR(
            sequences=[seq],
            entities=["User"],
            context_tags=["web"],
        )
        s = ir.summary()
        assert s["actions"] == 1
        assert s["coverage"] == 0.5
        assert s["missing_primitives"] == ["persist_state"]
        assert s["entities"] == ["User"]
        assert len(s["sequences"]) == 1
        assert s["sequences"][0]["action"] == "create_user"
        assert s["sequences"][0]["type"] == "create"
        assert s["sequences"][0]["entity"] == "User"
        assert s["sequences"][0]["present"] == ["validate"]
        assert s["sequences"][0]["missing"] == ["persist_state"]

    def test_multiple_sequences_coverage_avg(self):
        s1 = PrimitiveSequence(
            action_name="a", action_type="create",
            present=[PrimitiveMatch(primitive="x", present=True)],
            missing=[PrimitiveMatch(primitive="y", present=False)],
        )
        s2 = PrimitiveSequence(
            action_name="b", action_type="get",
            present=[
                PrimitiveMatch(primitive="x", present=True),
                PrimitiveMatch(primitive="y", present=True),
                PrimitiveMatch(primitive="z", present=True),
            ],
            missing=[PrimitiveMatch(primitive="w", present=False)],
        )
        ir = PrimitiveIR(sequences=[s1, s2])
        # s1 coverage = 0.5, s2 coverage = 0.75, avg = 0.625
        assert abs(ir.overall_coverage - 0.625) < 0.001


class TestBuildPrimitiveIR:

    def test_build_from_create_code(self):
        code = """\
        class User:
            pass

        def create_user(name: str, email: str):
            validate(name)
            user = User(name=name, email=email)
            db.add(user)
            db.commit()
            return user
        """
        path = _write_temp_py(code)
        try:
            # context_tags must confirm "create" domain — spec.request_type is primary signal
            ir = build_primitive_ir([path], context_tags=["create"])
            assert len(ir.sequences) >= 1
            seq = next(s for s in ir.sequences if s.action_name == "create_user")
            assert seq.action_type == "create"
            assert seq.coverage > 0
            assert "User" in ir.entities
        finally:
            os.unlink(path)

    def test_build_from_login_code(self):
        code = """\
        class User:
            pass

        def login(username: str, password: str):
            user = get_user(username)
            if not user:
                raise ValueError("not found")
            verify_password(password, user.hashed_password)
            token = create_access_token(user.id)
            return token
        """
        path = _write_temp_py(code)
        try:
            # context_tags must confirm "auth.login" domain
            ir = build_primitive_ir([path], context_tags=["auth.login"])
            seq = next(s for s in ir.sequences if s.action_name == "login")
            assert seq.action_type == "login"
            # Should detect: lookup (get_user), validate (verify_password),
            # authorize (create_access_token), produce_output (return token)
            present_names = seq.present_names
            assert "lookup" in present_names
            assert "authorize" in present_names
        finally:
            os.unlink(path)

    def test_build_from_list_code(self):
        code = """\
        def list_users():
            users = db.query(User).all()
            return users
        """
        path = _write_temp_py(code)
        try:
            ir = build_primitive_ir([path])
            assert len(ir.sequences) >= 1
            seq = next(s for s in ir.sequences if s.action_name == "list_users")
            assert seq.action_type == "list"
        finally:
            os.unlink(path)

    def test_build_skips_unknown_without_decorator(self):
        code = """\
        def helper_function():
            pass

        def create_user():
            return "ok"
        """
        path = _write_temp_py(code)
        try:
            # context_tags confirms "create" domain → create_user analyzed
            ir = build_primitive_ir([path], context_tags=["create"])
            names = [s.action_name for s in ir.sequences]
            assert "helper_function" not in names
            assert "create_user" in names
        finally:
            os.unlink(path)

    def test_build_skips_create_without_context_confirmation(self):
        """infer_action_type is a hint only: if context_tags don't confirm
        the create domain, add_member / new_parser should NOT trigger
        create entity contracts (the latent keyword gate scenario)."""
        code = """\
        def add_member(group_id: int, user_id: int):
            group.members.append(user_id)
            return group

        def new_parser(text: str):
            return text.split()
        """
        path = _write_temp_py(code)
        try:
            # No context_tags → spec.request_type=modify, not create
            ir = build_primitive_ir([path], context_tags=[])
            names = [s.action_name for s in ir.sequences]
            # add_member / new_parser must NOT be analyzed for create contracts
            assert "add_member" not in names
            assert "new_parser" not in names
        finally:
            os.unlink(path)

    def test_build_keeps_unknown_with_decorator(self):
        code = """\
        def decorator(f):
            return f

        @decorator
        def custom_endpoint():
            pass
        """
        path = _write_temp_py(code)
        try:
            ir = build_primitive_ir([path])
            names = [s.action_name for s in ir.sequences]
            assert "custom_endpoint" in names
        finally:
            os.unlink(path)

    def test_build_with_multiple_files(self):
        code1 = """\
        class User:
            pass

        def create_user(name):
            user = User(name=name)
            return user
        """
        code2 = """\
        def delete_user(user_id):
            user = get_user(user_id)
            db.delete(user)
        """
        p1 = _write_temp_py(code1)
        p2 = _write_temp_py(code2)
        try:
            # context_tags confirms "create" domain
            ir = build_primitive_ir([p1, p2], context_tags=["create"])
            names = [s.action_name for s in ir.sequences]
            assert "create_user" in names
            assert "delete_user" in names
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_build_empty_file_list(self):
        ir = build_primitive_ir([])
        assert len(ir.sequences) == 0
        assert ir.overall_coverage == 1.0

    def test_build_with_context_tags(self):
        code = """\
        def create_user():
            return "ok"
        """
        path = _write_temp_py(code)
        try:
            ir = build_primitive_ir([path], context_tags=["api", "v2"])
            assert "api" in ir.context_tags
            assert "v2" in ir.context_tags
        finally:
            os.unlink(path)

    def test_build_scope_filters_out_of_scope_actions(self):
        code = """\
        def in_scope_create(name):
            user = User(name=name)
            db.save(user)
            return user

        def unrelated_helper(x):
            return x + 1

        def create_post(title):
            post = Post(title=title)
            db.save(post)
            return post
        """
        path = _write_temp_py(code)
        try:
            ir = build_primitive_ir(
                [path],
                context_tags=["create"],
                scope={"in_scope_create"},
            )
            names = [s.action_name for s in ir.sequences]
            assert "in_scope_create" in names
            assert "unrelated_helper" not in names
            assert "create_post" not in names
        finally:
            os.unlink(path)

    def test_build_scope_none_preserves_legacy(self):
        code = """\
        def create_user(name):
            user = User(name=name)
            db.save(user)
            return user
        """
        path = _write_temp_py(code)
        try:
            ir = build_primitive_ir([path], context_tags=["create"], scope=None)
            names = [s.action_name for s in ir.sequences]
            assert "create_user" in names
        finally:
            os.unlink(path)

    def test_build_scope_empty_yields_no_sequences(self):
        code = """\
        def create_user(name):
            user = User(name=name)
            db.save(user)
            return user
        """
        path = _write_temp_py(code)
        try:
            ir = build_primitive_ir([path], context_tags=["create"], scope=set())
            assert ir.sequences == []
        finally:
            os.unlink(path)

    def test_build_ir_summary_roundtrip(self):
        code = """\
        class Item:
            pass

        def create_item(name):
            item = Item(name=name)
            db.save(item)
            return item

        def delete_item(item_id):
            item = get_item(item_id)
            db.delete(item)
        """
        path = _write_temp_py(code)
        try:
            ir = build_primitive_ir([path])
            s = ir.summary()
            assert isinstance(s["actions"], int)
            assert isinstance(s["coverage"], float)
            assert isinstance(s["missing_primitives"], list)
            assert isinstance(s["entities"], list)
            assert isinstance(s["sequences"], list)
        finally:
            os.unlink(path)


# =============================================================================
# 8. PrimitiveSequence properties
# =============================================================================

class TestPrimitiveSequenceProperties:

    def test_coverage_all_present(self):
        seq = PrimitiveSequence(
            action_name="test", action_type="create",
            present=[PrimitiveMatch(primitive="a", present=True)],
            missing=[],
        )
        assert seq.coverage == 1.0

    def test_coverage_all_missing(self):
        seq = PrimitiveSequence(
            action_name="test", action_type="create",
            present=[],
            missing=[PrimitiveMatch(primitive="a", present=False)],
        )
        assert seq.coverage == 0.0

    def test_coverage_empty(self):
        seq = PrimitiveSequence(action_name="test", action_type="create")
        assert seq.coverage == 1.0

    def test_missing_names(self):
        seq = PrimitiveSequence(
            action_name="test", action_type="create",
            missing=[
                PrimitiveMatch(primitive="validate", present=False),
                PrimitiveMatch(primitive="persist_state", present=False),
            ],
        )
        assert seq.missing_names == ["validate", "persist_state"]

    def test_present_names(self):
        seq = PrimitiveSequence(
            action_name="test", action_type="create",
            present=[
                PrimitiveMatch(primitive="input_bind", present=True),
                PrimitiveMatch(primitive="create_entity", present=True),
            ],
        )
        assert seq.present_names == ["input_bind", "create_entity"]
