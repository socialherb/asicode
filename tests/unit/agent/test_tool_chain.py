"""Tests for tool_chain.py — ScopedToolFilter file access control."""

from __future__ import annotations

from external_llm.agent.tool_chain import ScopedToolFilter


class TestScopedToolFilter:
    """Coverage for ScopedToolFilter — can_write, can_read, and init branches."""

    def test_default_no_restriction(self):
        """Default constructor: allowed_write=None, readonly_files=set()."""
        f = ScopedToolFilter()
        assert f.can_write("any_path.py")
        assert f.can_read("any_path.py")

    def test_allowed_write_set(self):
        """allowed_write restricts writable paths."""
        f = ScopedToolFilter(allowed_write={"/tmp", "/home"})
        assert f.can_write("/tmp")
        assert f.can_write("/home")
        assert not f.can_write("/etc")

    def test_readonly_blocks_write_when_allowed_write_none(self):
        """readonly files are blocked even when allowed_write is None."""
        f = ScopedToolFilter(readonly_files={"locked.py"})
        assert not f.can_write("locked.py")
        assert f.can_write("free.py")

    def test_readonly_blocks_write_when_in_allowed_set(self):
        """readonly files are blocked even if in allowed_write."""
        f = ScopedToolFilter(allowed_write={"shared.py"}, readonly_files={"shared.py"})
        assert not f.can_write("shared.py")

    def test_allowed_write_excludes_readonly(self):
        """Path in both allowed_write and readonly → not writable."""
        f = ScopedToolFilter(allowed_write={"a.py", "b.py"}, readonly_files={"b.py"})
        assert f.can_write("a.py")
        assert not f.can_write("b.py")

    def test_can_read_always_true(self):
        """can_read always returns True regardless of allowed_write/readonly."""
        f = ScopedToolFilter(allowed_write=set(), readonly_files={"x.py"})
        assert f.can_read("x.py")
        assert f.can_read("y.py")
        assert f.can_read("")

    def test_readonly_defaults_to_empty_set(self):
        """readonly_files=None → defaults to empty set, no restrictions."""
        f = ScopedToolFilter(readonly_files=None)
        assert f.can_write("anything.py")

    def test_allowed_write_empty_blocks_all(self):
        """allowed_write=set() blocks write to any path."""
        f = ScopedToolFilter(allowed_write=set())
        assert not f.can_write("x.py")
        assert not f.can_write("y.py")
