"""Tests for the post-edit declaration-loss guard (tool_safety).

Covers summarize_decl_losses (pure) and its integration into
WriteSafetyManager.summarize_change ([POST-EDIT DIFF] block).
"""
import pytest

from external_llm.agent.tool_safety import (
    WriteSafetyManager,
    _python_decl_sets,
    summarize_decl_losses,
)

BASE = '''\
import os
from typing import List as TList

def keep_me():
    pass

def drop_me():
    pass

class Service:
    def method_a(self):
        pass

    def method_b(self):
        pass
'''


class TestPythonDeclSets:
    def test_extracts_symbols_and_imports(self):
        symbols, imports = _python_decl_sets(BASE)
        assert {"keep_me", "drop_me", "Service", "Service.method_a", "Service.method_b"} == symbols
        assert imports == {"os", "TList"}

    def test_unparseable_returns_none(self):
        assert _python_decl_sets("def broken(:\n") is None


class TestSummarizeDeclLosses:
    def test_dropped_function_flagged(self):
        current = BASE.replace("def drop_me():\n    pass\n\n", "")
        out = summarize_decl_losses(BASE, current)
        assert "removed symbols [drop_me]" in out
        assert "verify this was intended" in out

    def test_dropped_method_flagged_with_class_prefix(self):
        current = BASE.replace("    def method_b(self):\n        pass\n", "")
        out = summarize_decl_losses(BASE, current)
        assert "Service.method_b" in out

    def test_dropped_import_flagged(self):
        current = BASE.replace("import os\n", "")
        out = summarize_decl_losses(BASE, current)
        assert "imports [os]" in out

    def test_rename_lists_added_names_without_guessing(self):
        current = BASE.replace("def drop_me", "def drop_me_v2")
        out = summarize_decl_losses(BASE, current)
        assert "removed symbols [drop_me]" in out
        assert "newly added: [drop_me_v2]" in out
        assert "(rename?)" in out

    def test_no_loss_returns_empty(self):
        current = BASE + "\ndef extra():\n    pass\n"
        assert summarize_decl_losses(BASE, current) == ""

    def test_unparseable_side_skips_check(self):
        assert summarize_decl_losses(BASE, "def broken(:\n") == ""
        assert summarize_decl_losses("def broken(:\n", BASE) == ""

    def test_many_losses_capped(self):
        pre = "\n".join(f"def f{i}():\n    pass" for i in range(12))
        out = summarize_decl_losses(pre, "x = 1\n")
        assert "(+4 more)" in out


def _ts_available() -> bool:
    try:
        from external_llm.languages.tree_sitter_utils import get_parser
        return get_parser("javascript") is not None
    except ImportError:
        return False


JS_BASE = """\
import { helper } from './helper.js';
function drawBlock() {}
const renderWorld = () => {};
class Game {
    update() {}
    render() {}
}
"""


@pytest.mark.skipif(not _ts_available(), reason="tree-sitter unavailable")
class TestTreeSitterLanguages:
    def test_js_dropped_function_flagged(self):
        current = JS_BASE.replace("function drawBlock() {}\n", "")
        out = summarize_decl_losses(JS_BASE, current, ".js")
        assert "removed symbols [drawBlock]" in out

    def test_js_dropped_method_flagged(self):
        current = JS_BASE.replace("    render() {}\n", "")
        out = summarize_decl_losses(JS_BASE, current, ".js")
        assert "render" in out

    def test_js_rename_lists_added(self):
        current = JS_BASE.replace("function drawBlock", "function drawVoxel")
        out = summarize_decl_losses(JS_BASE, current, ".js")
        assert "removed symbols [drawBlock]" in out
        assert "drawVoxel" in out and "(rename?)" in out

    def test_js_broken_post_skips_check(self):
        # tree-sitter parses broken code tolerantly — has_error guard must
        # suppress phantom "removed" warnings.
        current = JS_BASE.replace("function drawBlock() {}", "function drawBlock( {")
        assert summarize_decl_losses(JS_BASE, current, ".js") == ""

    def test_js_no_loss_returns_empty(self):
        current = JS_BASE + "\nfunction extra() {}\n"
        assert summarize_decl_losses(JS_BASE, current, ".js") == ""

    def test_ts_class_removal_flagged(self):
        ts = "interface Opts { x: number }\nclass Store { get() {} }\nexport function load() {}\n"
        current = "interface Opts { x: number }\nexport function load() {}\n"
        out = summarize_decl_losses(ts, current, ".ts")
        assert "Store" in out


class TestUnsupportedLanguage:
    def test_unmapped_extension_skipped(self):
        assert summarize_decl_losses("a\n", "b\n", ".css") == ""


class TestSummarizeChangeIntegration:
    def _summarize(self, tmp_path, original: str, current: str, name: str = "mod.py"):
        f = tmp_path / name
        f.write_text(current, encoding="utf-8")
        mgr = WriteSafetyManager(str(tmp_path))
        return mgr.summarize_change({str(f): original})

    def test_loss_warning_appended_to_diff(self, tmp_path):
        current = BASE.replace("def drop_me():\n    pass\n\n", "")
        out = self._summarize(tmp_path, BASE, current)
        assert "[POST-EDIT DIFF]" in out
        assert "+0/-" in out or "lines; changed" in out
        assert "removed symbols [drop_me]" in out

    def test_no_warning_for_safe_edit(self, tmp_path):
        current = BASE.replace("pass", "return 1", 1)
        out = self._summarize(tmp_path, BASE, current)
        assert "removed" not in out

    @pytest.mark.skipif(not _ts_available(), reason="tree-sitter unavailable")
    def test_js_file_loss_flagged_in_diff(self, tmp_path):
        current = JS_BASE.replace("function drawBlock() {}\n", "")
        out = self._summarize(tmp_path, JS_BASE, current, name="game.js")
        assert "[POST-EDIT DIFF]" in out
        assert "removed symbols [drawBlock]" in out

    def test_unsupported_language_file_skipped(self, tmp_path):
        out = self._summarize(tmp_path, "a { color: red }\n", "b { color: blue }\n", name="app.css")
        assert out is not None
        assert "removed" not in out
