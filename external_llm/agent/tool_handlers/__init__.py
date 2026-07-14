"""
Tool handler mixins for ToolRegistry.

Each module contains a Mixin class with a group of _tool_* methods.
ToolRegistry inherits from all mixins via multiple inheritance.
"""
from .agent_tools import AgentToolsMixin
from .analysis_tools import AnalysisToolsMixin
from .browser_tools import BrowserActionToolsMixin
from .git_tools import ShellToolsMixin
from .read_tools import ReadToolsMixin
from .test_tools import TestToolsMixin
from .web_search_tools import WebSearchToolsMixin
from .write_tools import WriteToolsMixin

__all__ = [
    "AgentToolsMixin",
    "AnalysisToolsMixin",
    "BrowserActionToolsMixin",
    "ReadToolsMixin",
    "ShellToolsMixin",
    "TestToolsMixin",
    "WebSearchToolsMixin",
    "WriteToolsMixin",
]
