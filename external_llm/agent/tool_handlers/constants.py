"""Shared tool-handler constants.

Leaf module (no internal imports) so both ``tool_registry`` and the handler
mixins can import these without creating a circular import. Mirrors the
``shell_policy`` pattern (shared policy constants imported by multiple modules).
"""

# ask_user default timeout (seconds) — single shared threshold across all entry points.
# On timeout, proceeds autonomously with the provided default.
ASK_USER_DEFAULT_TIMEOUT = 60
