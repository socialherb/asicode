"""Unit tests for session_state.py — SessionState + SessionStateManager."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from external_llm.agent.session_state import SessionState, SessionStateManager


class TestSessionState:
    """Tests for SessionState data model."""

    def test_init_sets_defaults(self):
        state = SessionState(session_id="test-123")
        assert state.session_id == "test-123"
        assert state.edit_history == []
        assert state.plan is None

    def test_save_delegates_to_manager(self):
        state = SessionState(session_id="save-1")
        with patch.object(SessionStateManager, "save_state") as mock_save:
            state.save()
            mock_save.assert_called_once_with(state)

    def test_load_state_restores_data(self):
        loaded = SessionState(session_id="load-1")
        loaded.edit_history = [{"file": "a.py", "op": "edit"}]
        loaded.plan = {"steps": ["read", "write"]}

        mock_mgr = MagicMock()
        mock_mgr.load_state.return_value = loaded

        with patch.object(SessionStateManager, "load_state", return_value=loaded):
            state = SessionState(session_id="load-1")
            state.load_state()
            assert state.edit_history == [{"file": "a.py", "op": "edit"}]
            assert state.plan == {"steps": ["read", "write"]}

    def test_load_state_returns_none_no_modify(self):
        with patch.object(SessionStateManager, "load_state", return_value=None):
            state = SessionState(session_id="none-1")
            state.load_state()
            assert state.edit_history == []
            assert state.plan is None


class TestSessionStateManager:
    """Tests for SessionStateManager persistence."""

    def test_default_base_dir(self, tmp_path):
        with patch.object(Path, "cwd", return_value=tmp_path):
            mgr = SessionStateManager()
            expected = tmp_path / ".asicode" / "sessions"
            assert mgr.base_dir == expected
            assert expected.exists()

    def test_custom_base_dir(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        expected = tmp_path / ".asicode" / "sessions"
        assert mgr.base_dir == expected
        assert expected.exists()

    def test_session_path(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        p = mgr._session_path("abc123")
        assert p == tmp_path / ".asicode" / "sessions" / "abc123.json"

    def test_save_and_load_roundtrip(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        state = SessionState(session_id="roundtrip-1")
        state.edit_history = [{"file": "x.py", "change": "add"}]
        state.plan = {"description": "Add feature"}

        mgr.save_state(state)

        loaded = mgr.load_state("roundtrip-1")
        assert loaded is not None
        assert loaded.session_id == "roundtrip-1"
        assert loaded.edit_history == [{"file": "x.py", "change": "add"}]
        assert loaded.plan == {"description": "Add feature"}

    def test_save_io_error_raises(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        state = SessionState(session_id="io-err")

        # Simulate write error via unwritable path
        mgr.base_dir = tmp_path / "nonexistent" / "deep" / "path"
        with pytest.raises(RuntimeError, match="Failed to save session state"):
            mgr.save_state(state)

    def test_load_missing_returns_none(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        result = mgr.load_state("does-not-exist")
        assert result is None

    def test_load_corrupted_json_raises(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        p = mgr._session_path("corrupt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Failed to load session state"):
            mgr.load_state("corrupt")

    def test_load_io_error_raises(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        p = mgr._session_path("io-load")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Create an unwritable file situation — make it a directory
        p.mkdir(parents=True, exist_ok=True)

        with pytest.raises(RuntimeError, match="Failed to load session state"):
            mgr.load_state("io-load")

    def test_persistence_across_instances(self, tmp_path):
        mgr1 = SessionStateManager(base_dir=str(tmp_path))
        state = SessionState(session_id="persist-1")
        state.edit_history = [{"file": "a.py"}]
        mgr1.save_state(state)

        mgr2 = SessionStateManager(base_dir=str(tmp_path))
        loaded = mgr2.load_state("persist-1")
        assert loaded is not None
        assert loaded.edit_history == [{"file": "a.py"}]

    def test_save_unicode_content(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        state = SessionState(session_id="unicode-1")
        state.edit_history = [{"description": "안녕하세요 — 유니코드"}]
        mgr.save_state(state)

        loaded = mgr.load_state("unicode-1")
        assert loaded is not None
        assert loaded.edit_history[0]["description"] == "안녕하세요 — 유니코드"

    def test_file_content_is_valid_json(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        state = SessionState(session_id="json-valid")
        state.edit_history = [{"op": "read"}]
        mgr.save_state(state)

        content = (mgr.base_dir / "json-valid.json").read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["session_id"] == "json-valid"
        assert data["edit_history"] == [{"op": "read"}]

    def test_blank_plan_handling(self, tmp_path):
        """plan=None should roundtrip correctly (not become empty dict)."""
        mgr = SessionStateManager(base_dir=str(tmp_path))
        state = SessionState(session_id="none-plan")
        state.plan = None
        mgr.save_state(state)

        loaded = mgr.load_state("none-plan")
        assert loaded is not None
        assert loaded.plan is None


class TestSessionStateManagerEdgeCases:
    """Additional edge case tests."""

    def test_creates_nested_dir(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        assert not deep.exists()
        mgr = SessionStateManager(base_dir=str(deep))
        assert mgr.base_dir.exists()
        # base_dir = deep/.asicode/sessions → parent.parent == deep
        assert mgr.base_dir.parent.parent == deep

    def test_overwrite_existing_session(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        state1 = SessionState(session_id="overwrite")
        state1.edit_history = [{"v": 1}]
        mgr.save_state(state1)

        state2 = SessionState(session_id="overwrite")
        state2.edit_history = [{"v": 2}]
        mgr.save_state(state2)

        loaded = mgr.load_state("overwrite")
        assert loaded.edit_history == [{"v": 2}]

    def test_empty_edit_history(self, tmp_path):
        mgr = SessionStateManager(base_dir=str(tmp_path))
        state = SessionState(session_id="empty-history")
        mgr.save_state(state)

        loaded = mgr.load_state("empty-history")
        assert loaded.edit_history == []
