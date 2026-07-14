"""Shell execution policy constants shared by tool_registry and git_tools.

Default-open model: ALL shell commands are allowed by default.
Only specific dangerous operations require user approval.
"""

# Commands that are allowed but have certain forbidden flags.
FORBIDDEN_FLAGS: dict = {
    "sed": {"-i", "--in-place"},  # in-place edit bypasses apply_patch pipeline
}

# Commands that require user approval before execution (destructive operations).
# When the LLM requests one of these, the system blocks it and asks the LLM
# to obtain explicit user consent before proceeding.
DANGEROUS_SHELL_COMMANDS: frozenset = frozenset({
    "rm",
})
