from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SlashCommand:
    name: str
    description: str
    template: str
    aliases: list[str] = field(default_factory=list)
    category: str = "general"
    default_params: dict[str, Any] = field(default_factory=dict)

    def expand(self, args: str = "", context: str = "") -> str:
        return self.template.format(args=args, context=context) if args else self.template


class SlashCommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        builtins = [
            SlashCommand("fix", "Fix bugs or issues in code",
                         "Fix the following issue: {args}", ["f", "fixit", "bug"], "code"),
            SlashCommand("refactor", "Refactor code for better structure",
                         "Refactor the following: {args}", ["rf"], "code"),
            SlashCommand("test", "Add or improve tests",
                         "Add tests for: {args}", ["t"], "code"),
            SlashCommand("explain", "Explain what code does",
                         "Explain this: {args}", ["ex"], "analysis"),
            SlashCommand("review", "Review code for issues",
                         "Review the following for issues: {args}", ["rv"], "analysis"),
        ]
        for cmd in builtins:
            self.register(cmd)

    def register(self, cmd: SlashCommand) -> None:
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd

    def get_command(self, name: str) -> Optional[SlashCommand]:
        return self._commands.get(name)

    def all_commands(self) -> list[SlashCommand]:
        seen: set = set()
        result = []
        for cmd in self._commands.values():
            if id(cmd) not in seen:
                seen.add(id(cmd))
                result.append(cmd)
        return result

    def detect_slash_command(self, text: str) -> Optional[str]:
        """Detect a slash command in text, return command name or None."""
        if not text:
            return None
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped[1:].split(None, 1)
        name = parts[0].lower() if parts else ""
        return name if self.get_command(name) else None

    def generate_prompt(self, command: str, original: str) -> str:
        """Generate an expanded prompt for a slash command."""
        cmd = self.get_command(command)
        if not cmd:
            return original
        parts = original.strip().split(None, 1)
        args = parts[1] if len(parts) > 1 else ""
        return cmd.expand(args)


_registry: Optional[SlashCommandRegistry] = None


def get_registry() -> SlashCommandRegistry:
    global _registry
    if _registry is None:
        _registry = SlashCommandRegistry()
    return _registry


def list_commands() -> list[SlashCommand]:
    return get_registry().all_commands()
