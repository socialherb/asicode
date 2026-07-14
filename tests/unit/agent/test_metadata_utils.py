"""Tests for metadata_utils — _safe_metadata."""

from __future__ import annotations

from unittest.mock import MagicMock

from external_llm.agent.metadata_utils import _safe_metadata


class TestSafeMetadata:
    """Tests for _safe_metadata(obj) -> dict."""

    def test_object_with_metadata(self):
        obj = MagicMock()
        obj.metadata = {"key": "value"}
        assert _safe_metadata(obj) == {"key": "value"}

    def test_object_without_metadata(self):
        obj = object()
        assert _safe_metadata(obj) == {}

    def test_non_dict_metadata_returns_empty(self):
        obj = MagicMock()
        obj.metadata = "not a dict"
        assert _safe_metadata(obj) == {}

    def test_none_metadata_returns_empty(self):
        obj = MagicMock()
        obj.metadata = None
        assert _safe_metadata(obj) == {}

    def test_none_object(self):
        """None is handled gracefully by getattr default."""
        assert _safe_metadata(None) == {}
