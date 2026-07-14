"""Unit tests for external_llm.agent.symbol_locator.PythonAstLocator."""

from external_llm.agent.symbol_locator import PythonAstLocator


def _by_qual(spans):
    return {s.qualname: s for s in spans}


class TestPythonAstLocator:
    def test_top_level_functions_and_classes(self):
        src = (
            "def foo():\n"
            "    return 1\n"
            "\n"
            "class Bar:\n"
            "    def m(self):\n"
            "        return 2\n"
        )
        spans = _by_qual(PythonAstLocator().locate(src))
        assert set(spans) == {"foo", "Bar", "Bar.m"}
        assert spans["foo"].kind == "function" and spans["foo"].top_level
        assert spans["Bar"].kind == "class" and spans["Bar"].top_level
        assert spans["Bar.m"].kind == "method" and not spans["Bar.m"].top_level
        assert spans["Bar.m"].name == "m"

    def test_span_line_ranges(self):
        src = "def foo():\n    a = 1\n    return a\n"
        s = _by_qual(PythonAstLocator().locate(src))["foo"]
        assert s.start_line == 1
        assert s.end_line == 3

    def test_decorator_included_in_span(self):
        src = (
            "@decorator\n"
            "def foo():\n"
            "    return 1\n"
        )
        s = _by_qual(PythonAstLocator().locate(src))["foo"]
        assert s.start_line == 1  # decorator line, not the def line

    def test_syntax_error_returns_empty(self):
        assert PythonAstLocator().locate("def (:\n") == []

    def test_async_function(self):
        spans = _by_qual(PythonAstLocator().locate("async def foo():\n    return 1\n"))
        assert "foo" in spans and spans["foo"].kind == "function"
