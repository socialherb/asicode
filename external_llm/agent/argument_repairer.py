"""
Argument Repairer for asicode Agent

Lightweight layer that corrects common argument naming mistakes
before tool dispatch. Prevents tool errors due to minor naming variations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepairResult:
    """Result of argument repair attempt."""

    repaired: bool
    original_args: dict[str, Any]
    repaired_args: dict[str, Any]
    repairs_applied: list[str]
    # Arguments whose type is wrong AND not safely coercible. The caller must
    # refuse the call with these messages instead of invoking the handler —
    # see ``coerce_types``.
    errors: list[str] = field(default_factory=list)


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
    "glob": {"file_path": "path", "filepath": "path", "glob": "pattern"},
    "run_lint": {"file_path": "path", "filepath": "path"},
    # Schema uses "file_path"; accept the "path" family.
    "modify_symbol": {"path": "file_path"},
    "anchor_edit": {"path": "file_path"},
    "edit_ast": {"path": "file_path"},
    "edit_text": {"path": "file_path", "old_text": "old_string", "new_text": "new_string"},
    "read_symbol": {"path": "file_path"},
    "analyze_change_impact": {"path": "file_path"},
    # Schema uses "path"; accept the "file_path" family.
    "get_file_outline": {"file_path": "path", "filepath": "path", "target_file": "path"},
    # Schema uses "query"; accept common alternatives.
    "find_relevant_files": {"q": "query", "search": "query", "keyword": "query"},
    # Schema uses "name"; tool is called "find_symbol" so LLMs commonly send "symbol".
    "find_symbol": {"symbol": "name"},
}


def _schema_param_types() -> dict[str, dict[str, str]]:
    """``{tool: {param: json_type}}`` from the tool schemas the model is given.

    Read lazily and cached: importing tool_schemas at module import time would
    invert the existing dependency direction (tool_registry imports this).
    """
    global _PARAM_TYPES
    if _PARAM_TYPES is None:
        table: dict[str, dict[str, str]] = {}
        from .tool_schemas import AGENT_TOOL_SCHEMAS

        for schema in AGENT_TOOL_SCHEMAS:
            fn = schema.get("function", schema)
            name = fn.get("name")
            props = (fn.get("parameters") or {}).get("properties") or {}
            if not name or not props:
                continue
            table[name] = {p: spec.get("type") for p, spec in props.items() if spec.get("type")}
        _PARAM_TYPES = table
    return _PARAM_TYPES


_PARAM_TYPES: dict[str, dict[str, str]] | None = None

# Only these are checked. "object"/"array" params (write_plan's `plan`) carry
# structure this layer has no business second-guessing, and the handlers that
# take them already validate shape and report it well.
_COERCIBLE_TYPES = frozenset({"string", "integer", "number", "boolean"})

_TRUE_STRINGS = frozenset({"true", "yes", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "no", "0", "off"})


class ArgumentRepairer:
    """Lightweight argument repair layer.

    Two classes of mistake an LLM makes with tool arguments:

      * the wrong NAME for the right value (``file_path`` for ``path``) —
        absorbed by ``_ARG_ALIASES``;
      * the wrong TYPE for the right name (``max_results=[3]``, ``path=12345``,
        ``pattern=None``) — absorbed by ``coerce_types``.

    Only the first was handled, so the second reached the handlers, where
    ``args.get("x", "").strip()`` and ``int(args.get("y", 30))`` raised. The
    outer catch in ``_dispatch_impl`` turned those into a ToolResult, so nothing
    crashed — but the model's only feedback was a Python exception string naming
    no argument and no expected type. Measured before this layer existed: 67 of
    138 malformed-argument dispatches (48.6%) across 15 of 24 tools came back
    that way, write tools included.

    Repairs are conservative on purpose. A wrong type is only rewritten when
    exactly one reading is plausible (``"30"`` for an integer, ``30`` for a
    string); anything else is REFUSED with a message naming the parameter, what
    arrived and what the schema wants, which is strictly more actionable than
    the value this layer would otherwise have to invent.
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

        # ── Name repair ──────────────────────────────────────────────────────
        # Runs FIRST so the type pass below sees canonical names and can look
        # their declared types up in the schema.
        for alias, canonical in self.aliases.get(tool_name, {}).items():
            # Alias present and canonical name missing
            if alias in repaired_args and canonical not in repaired_args:
                # Move value from alias to canonical
                repaired_args[canonical] = repaired_args[alias]
                # Remove alias to avoid confusion
                del repaired_args[alias]
                repairs_applied.append(f"{alias} → {canonical}")

        # ── Type repair ──────────────────────────────────────────────────────
        repaired_args, type_repairs, errors = self.coerce_types(tool_name, repaired_args)
        repairs_applied.extend(type_repairs)

        repaired = len(repairs_applied) > 0

        return RepairResult(
            repaired=repaired,
            original_args=original_args,
            repaired_args=repaired_args,
            repairs_applied=repairs_applied,
            errors=errors,
        )

    def coerce_types(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        """Bring *args* in line with the tool schema's declared types.

        Returns ``(args, repairs, errors)``. A non-empty *errors* means the call
        must be refused — the value cannot be read as the declared type without
        guessing, and guessing here would turn "wrong argument" into "wrong
        answer", which is the harder failure to notice.

        Rules, in the order they matter:

        * ``None`` is treated as ABSENT (the key is dropped) rather than as a
          bad value. A model emitting a JSON null for an optional argument means
          "no opinion"; dropping it lets the handler's own default and its own
          "x is required" message take over, which reads far better than
          ``AttributeError: 'NoneType' object has no attribute 'strip'``.
        * ``bool`` is checked BEFORE ``int``. In Python ``isinstance(True, int)``
          is True, so an unguarded numeric branch would silently accept
          ``recursive=True`` as ``max_results=1``.
        * Containers are never stringified. ``str(["a"])`` is ``"['a']"`` — a
          path/pattern that cannot match anything, failing later and further
          from the cause.
        """
        param_types = _schema_param_types().get(tool_name)
        if not param_types:
            return args, [], []

        out = dict(args)
        repairs: list[str] = []
        errors: list[str] = []

        for name, value in list(args.items()):
            declared = param_types.get(name)
            if declared not in _COERCIBLE_TYPES:
                continue

            if value is None:
                del out[name]
                repairs.append(f"{name}=null dropped (treated as omitted)")
                continue

            if declared == "string":
                if isinstance(value, str):
                    continue
                if isinstance(value, (bool, list, dict, tuple, set)):
                    errors.append(_type_error(name, value, "a string"))
                elif isinstance(value, (int, float)):
                    out[name] = str(value)
                    repairs.append(f"{name}={value!r} → {str(value)!r} (string)")
                else:
                    errors.append(_type_error(name, value, "a string"))

            elif declared in ("integer", "number"):
                if isinstance(value, bool):
                    errors.append(_type_error(name, value, f"a{'n' if declared[0] == 'i' else ''} {declared}"))
                elif isinstance(value, int) or (declared == "number" and isinstance(value, float)):
                    continue
                elif isinstance(value, float):  # integer param given 3.0
                    if value.is_integer():
                        out[name] = int(value)
                        repairs.append(f"{name}={value!r} → {int(value)} (integer)")
                    else:
                        errors.append(_type_error(name, value, "an integer"))
                elif isinstance(value, str):
                    try:
                        out[name] = int(value.strip()) if declared == "integer" else float(value.strip())
                        repairs.append(f"{name}={value!r} → {out[name]!r} ({declared})")
                    except ValueError:
                        # Unreadable NUMBER, not an unreadable shape: drop it and
                        # let the handler's own default stand. Several handlers
                        # document a tolerant contract for exactly this —
                        # bash's timeout clamps "abc" to the default rather than
                        # failing the command ("model-supplied garbage must not
                        # raise", test_shell_danger_policy) — and refusing here
                        # would override a deliberate decision this layer cannot
                        # see. Containers below still refuse: no scalar reading
                        # of them exists at all.
                        del out[name]
                        repairs.append(f"{name}={value!r} dropped (not {declared}; handler default applies)")
                else:
                    errors.append(_type_error(name, value, f"a{'n' if declared[0] == 'i' else ''} {declared}"))

            elif declared == "boolean":
                if isinstance(value, bool):
                    continue
                if isinstance(value, str) and value.strip().lower() in _TRUE_STRINGS:
                    out[name] = True
                    repairs.append(f"{name}={value!r} → True (boolean)")
                elif isinstance(value, str) and value.strip().lower() in _FALSE_STRINGS:
                    out[name] = False
                    repairs.append(f"{name}={value!r} → False (boolean)")
                else:
                    errors.append(_type_error(name, value, "a boolean (true/false)"))

        return out, repairs, errors


def _type_error(name: str, value: Any, expected: str) -> str:
    """Message that names the parameter, what arrived, and what was wanted."""
    got = type(value).__name__
    shown = repr(value)
    if len(shown) > 60:
        shown = shown[:57] + "…"
    return f"'{name}' must be {expected}, got {got} ({shown})"
