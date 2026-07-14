"""
Output Mode Enumeration for asicode

Defines the different output formats that LLMs can produce.
Used by hybrid_parser, patch_synthesizer, and intelligent routing.
"""
from enum import Enum


class OutputMode(str, Enum):
    """Output formats supported by asicode LLM integration"""

    # Unified diff format (git apply compatible)
    UNIFIED_DIFF = "unified_diff"

    # Cursor-like full file rewrite blocks
    FULL_FILE = "full_file"

    # asicode specific block format (BEFORE/AFTER blocks)
    ASICODE_BLOCK = "asicode_block"

    # Targeted insertion blocks (FUNCTION: + INSERT_AFTER: + code)
    TARGETED_BLOCK = "targeted_block"

    # JSON-based execution plans for multi-file operations
    PLAN_JSON = "plan_json"

    # Special mode for disambiguation requests
    NEEDS_DISAMBIGUATION = "needs_disambiguation"
