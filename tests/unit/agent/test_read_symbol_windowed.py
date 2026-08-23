"""read_symbol must stream its line window, never load the whole file.

P25-1: ``_tool_read_symbol`` read the ENTIRE target file (``abs_path.read_text``)
just to slice the symbol body window — the same unbounded-read class as
P22-4/P23-1/P24-2, and the only read tool path that bypassed the
``READ_FILE_MAX_CHARS`` SSOT: it bounded the OUTPUT via ``_apply_char_budget``
but never the INPUT. A 100 MB file with the symbol near the end was fully
materialised in memory on every read_symbol call.

The windowed read is exact: line numbers come from the AST symbol index, so
only ``[start, start+count)`` is needed — O(window) memory for any file size.
"""

from __future__ import annotations

from pathlib import Path

from external_llm.agent.symbol_search import SymbolDef


def _boom(*args, **kwargs):
    raise AssertionError("read_text must not be called — read_symbol must stream its window")


class _StubSearcher:
    def __init__(self, defs):
        self._defs = defs

    def find_symbol(self, name, search_path=None):
        return self._defs


def test_read_symbol_streams_window_not_whole_file(tool_registry, temp_repo_root, monkeypatch):
    """A symbol at the end of a 60k-line file: only the window is read."""
    target = Path(temp_repo_root) / "big_module.py"
    with open(target, "w", encoding="utf-8") as fh:
        for i in range(60_000):
            fh.write(f"filler_{i:05d} = {i}\n")
        fh.write("\ndef target_fn():\n    return 42\n")
    defs = [
        SymbolDef(
            file="big_module.py",
            line=60_002,
            end_line=60_004,
            kind="function",
            name="target_fn",
        )
    ]
    monkeypatch.setattr(tool_registry, "_symbol_searcher", _StubSearcher(defs))
    monkeypatch.setattr(Path, "read_text", _boom)

    result = tool_registry._tool_read_symbol({"name": "target_fn"})

    assert result.ok
    assert "def target_fn" in result.content
    assert "return 42" in result.content
    # Window: context_lines=10 → 0-based start 59991 → first shown 59992.
    assert "filler_59991" in result.content
    assert "filler_59990" not in result.content


def test_read_symbol_fallback_window_without_end_line(tool_registry, temp_repo_root, monkeypatch):
    """No end_line in the index → fixed ±context window, still streamed."""
    target = Path(temp_repo_root) / "mid_module.py"
    with open(target, "w", encoding="utf-8") as fh:
        for i in range(200):
            fh.write(f"line_{i:03d}\n")
        fh.write("def mid_fn():\n    pass\n")
    defs = [
        SymbolDef(file="mid_module.py", line=202, end_line=None, kind="function", name="mid_fn"),
    ]
    monkeypatch.setattr(tool_registry, "_symbol_searcher", _StubSearcher(defs))
    monkeypatch.setattr(Path, "read_text", _boom)

    result = tool_registry._tool_read_symbol({"name": "mid_fn"})

    assert result.ok
    assert "def mid_fn" in result.content
    # 0-based start 191 → first shown 192 ("line_191").
    assert "line_191" in result.content
    assert "line_190" not in result.content
