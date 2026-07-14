"""
Argument Repairer for asicode Agent

Lightweight layer that corrects common argument naming mistakes
before tool dispatch. Prevents tool errors due to minor naming variations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RepairResult:
    """Result of argument repair attempt."""
    repaired: bool
    original_args: dict[str, Any]
    repaired_args: dict[str, Any]
    repairs_applied: list[str]


# Canonical argument names for each tool.
#
# The tool schemas are split between two file-argument conventions: some tools
# name it "path" (create_file, read_file, edit_file, grep, run_lint) and others
# "file_path" (modify_symbol, edit_ast, edit_text, read_symbol,
# analyze_change_impact). LLMs routinely carry the wrong convention across
# look-alike tools — e.g. sending "file_path" to read_file — which otherwise
# surfaces as a bare "'path' is required" failure even though the path was given.
# These aliases absorb both directions of that mix-up before dispatch.
_ARG_ALIASES: dict[str, dict[str, str]] = {
    "apply_patch": {
        "diff": "patch",
        "content": "patch",
    },
    # Schema uses "path"; accept the "file_path" family.
    "read_file": {"file_path": "path", "filepath": "path", "target_file": "path"},
    "edit_file": {"file_path": "path", "filepath": "path"},
    "create_file": {"file_path": "path", "filepath": "path"},
    "grep": {"file_path": "path", "filepath": "path"},
    "run_lint": {"file_path": "path", "filepath": "path"},
    # Schema uses "file_path"; accept the "path" family.
    "modify_symbol": {"path": "file_path"},
    "anchor_edit": {"path": "file_path"},
    "edit_ast": {"path": "file_path"},
    "edit_text": {"path": "file_path"},
    "read_symbol": {"path": "file_path"},
    "analyze_change_impact": {"path": "file_path"},
    # Schema uses "path"; accept the "file_path" family.
    "get_file_outline": {"file_path": "path", "filepath": "path", "target_file": "path"},
    # Schema uses "query"; accept common alternatives.
    "find_relevant_files": {"q": "query", "search": "query", "keyword": "query"},
    # Schema uses "name"; tool is called "find_symbol" so LLMs commonly send "symbol".
    "find_symbol": {"symbol": "name"},
}


class ArgumentRepairer:
    """Lightweight argument repair layer.

    Only repairs arguments when the canonical name is missing.
    If canonical name already present, aliases are ignored.
    """

    def __init__(self, custom_aliases: dict[str, dict[str, str]] | None = None):
        """Initialize with optional custom aliases.

        Args:
            custom_aliases: Additional or overriding alias mappings.
        """
        self.aliases: dict[str, dict[str, str]] = {}
        self.aliases.update(_ARG_ALIASES)
        if custom_aliases:
            for tool, mapping in custom_aliases.items():
                self.aliases.setdefault(tool, {}).update(mapping)

    def repair(self, tool_name: str, args: dict[str, Any]) -> RepairResult:
        """Attempt to repair argument names.

        Args:
            tool_name: Name of the tool being called.
            args: Original arguments dict.

        Returns:
            RepairResult with repair status and repaired arguments.
        """
        original_args = args.copy()
        repaired_args = args.copy()
        repairs_applied: list[str] = []

        # Get alias mapping for this tool
        tool_aliases = self.aliases.get(tool_name, {})
        if not tool_aliases:
            return RepairResult(
                repaired=False,
                original_args=original_args,
                repaired_args=repaired_args,
                repairs_applied=repairs_applied
            )

        # Check each alias
        for alias, canonical in tool_aliases.items():
            # Alias present and canonical name missing
            if alias in repaired_args and canonical not in repaired_args:
                # Move value from alias to canonical
                repaired_args[canonical] = repaired_args[alias]
                # Remove alias to avoid confusion
                del repaired_args[alias]
                repairs_applied.append(f"{alias} → {canonical}")

        repaired = len(repairs_applied) > 0

        return RepairResult(
            repaired=repaired,
            original_args=original_args,
            repaired_args=repaired_args,
            repairs_applied=repairs_applied
        )
