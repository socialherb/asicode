"""Unit tests for repair_helpers — self-repair strip helpers.

Covers _strip_redundant_inline_imports (module-import collection incl.
asnames, redundant-import removal across Import/ImportFrom shapes, asname
reference preservation, and the broken-strip guard) and
_strip_redundant_dataclass_decorator (Name/Call decorator detection in both
file and new body, and the RH-B1 dotted-decorator crash fix).
"""
from __future__ import annotations

from external_llm.agent.repair_helpers import (
    _strip_redundant_dataclass_decorator,
    _strip_redundant_inline_imports,
)

# ── _strip_redundant_inline_imports ─────────────────────────────────────────

class TestStripRedundantInlineImports:
    def test_bad_file_source_returns_unchanged(self):
        body = "import os\n"
        assert _strip_redundant_inline_imports(body, "def broken(:") == body

    def test_import_alias_collected(self):
        # asname must be collected too: import os as operating_system
        file_src = "import os as operating_system\n"
        assert _strip_redundant_inline_imports("import operating_system\n", file_src) == ""

    def test_from_import_asname_collected(self):
        file_src = "from os import path as p\n"
        assert _strip_redundant_inline_imports("import p\n", file_src) == ""

    def test_no_module_imports_returns_unchanged(self):
        # File source parses but has no import statements at all → bail early.
        file_src = "x = 1\n"
        body = "import os\n"
        assert _strip_redundant_inline_imports(body, file_src) == body

    def test_simple_redundant_import_removed(self):
        file_src = "import os\n"
        assert _strip_redundant_inline_imports("import os\nprint(1)\n", file_src) == "print(1)\n"

    def test_import_asname_referenced_elsewhere_kept(self):
        # Local rename whose asname is used outside the import span must stay.
        file_src = "import os\n"
        body = "import os as os2\nprint(os2)\n"
        assert _strip_redundant_inline_imports(body, file_src) == body

    def test_from_import_module_match_removed(self):
        file_src = "from os import path\n"
        assert _strip_redundant_inline_imports(
            "from os import path\nprint(path)\n", file_src) == "print(path)\n"

    def test_from_import_asname_used_keeps_whole_stmt(self):
        file_src = "from os import path\n"
        body = "from os import path as p\nprint(p)\n"
        assert _strip_redundant_inline_imports(body, file_src) == body

    def test_from_other_module_name_match_removed(self):
        # Module differs but the imported NAME is module-level → redundant.
        file_src = "from time import sleep\n"
        assert _strip_redundant_inline_imports(
            "from other import sleep\nprint(sleep)\n", file_src) == "print(sleep)\n"

    def test_from_other_module_asname_used_kept(self):
        file_src = "from time import sleep\n"
        body = "from other import sleep as s\nprint(s)\n"
        assert _strip_redundant_inline_imports(body, file_src) == body

    def test_multiline_import_full_span_removed(self):
        file_src = "import os\n"
        body = "from os import (\n    path,\n    sep,\n)\nprint(path)\n"
        assert _strip_redundant_inline_imports(body, file_src) == "print(path)\n"

    def test_broken_strip_guard_keeps_original(self):
        # Removing the indented import would orphan 'if True:' → the guard
        # must bail back to the original body instead of returning broken text.
        file_src = "import os\n"
        body = "if True:\n    import os\n"
        assert _strip_redundant_inline_imports(body, file_src) == body

    def test_unparseable_body_returns_unchanged(self):
        file_src = "import os\n"
        body = "def broken(:\n"
        assert _strip_redundant_inline_imports(body, file_src) == body


# ── _strip_redundant_dataclass_decorator ────────────────────────────────────

class TestStripRedundantDataclassDecorator:
    FILE = "@dataclass\nclass A:\n    x: int\n"

    def test_bad_file_source_returns_unchanged(self):
        body = "@dataclass\nclass A:\n    x: int\n"
        assert _strip_redundant_dataclass_decorator(body, "def broken(:") == body

    def test_file_call_decorator_collected(self):
        # @dataclass(frozen=True) in the FILE counts the class.
        file_src = "@dataclass(frozen=True)\nclass A:\n    x: int\n"
        body = "@dataclass\nclass A:\n    x: int\n"
        assert _strip_redundant_dataclass_decorator(body, file_src) == "class A:\n    x: int\n"

    def test_bad_new_body_returns_unchanged(self):
        assert _strip_redundant_dataclass_decorator("def broken(:", self.FILE) == "def broken(:"

    def test_no_dataclass_classes_in_file_unchanged(self):
        # File source parses but carries no @dataclass class → bail early.
        file_src = "class A:\n    x: int\n"
        body = "@dataclass\nclass A:\n    x: int\n"
        assert _strip_redundant_dataclass_decorator(body, file_src) == body

    def test_class_not_in_file_kept(self):
        body = "@dataclass\nclass B:\n    y: int\n"
        assert _strip_redundant_dataclass_decorator(body, self.FILE) == body

    def test_bare_dataclass_stripped(self):
        body = "@dataclass\nclass A:\n    x: int\n"
        assert _strip_redundant_dataclass_decorator(body, self.FILE) == "class A:\n    x: int\n"

    def test_call_dataclass_stripped(self):
        body = "@dataclass()\nclass A:\n    x: int\n"
        assert _strip_redundant_dataclass_decorator(body, self.FILE) == "class A:\n    x: int\n"

    def test_no_dataclass_in_new_body_unchanged(self):
        body = "class A:\n    x: int\n"
        assert _strip_redundant_dataclass_decorator(body, self.FILE) == body

    def test_dotted_decorator_does_not_crash_rh_b1(self):
        # RH-B1: ast.Attribute decorators (e.g. @pydantic.dataclasses.dataclass)
        # have no `.id` attribute — must not crash, and must not be stripped
        # (the removal loop only handles bare @dataclass / @dataclass(...)).
        body = "@pydantic.dataclasses.dataclass\nclass A:\n    x: int\n"
        assert _strip_redundant_dataclass_decorator(body, self.FILE) == body

    def test_dotted_decorator_with_bare_keeps_both_lines(self):
        # Mixed: dotted decorator preserved, bare @dataclass still stripped.
        body = "@pydantic.dataclasses.dataclass\n@dataclass\nclass A:\n    x: int\n"
        out = _strip_redundant_dataclass_decorator(body, self.FILE)
        assert "@pydantic.dataclasses.dataclass" in out
        assert "\n@dataclass\n" not in out
