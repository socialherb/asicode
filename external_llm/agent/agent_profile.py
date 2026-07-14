"""AgentProfile — per-request agent customisation loaded from .asicode/agents/{name}.json."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from external_llm.agent.tool_registry import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    """Lightweight customisation layer applied on top of AgentConfig.

    Profile JSON schema (.asicode/agents/{name}.json):
    {
        "name": "reviewer",
        "description": "Read-only code review agent",
        "allowed_tools": ["find_symbol", "find_references",
                          "get_project_info"],
        "blocked_tools": [],
        "model": null,
        "provider": null,
        "system_prompt_prefix": "You are a code reviewer. Avoid modifying files.",
        "max_turns": 10,
        "planning_enabled": false
    }

    Rules:
    - allowed_tools: empty list → no restriction (all tools allowed).
    - blocked_tools: tools always denied regardless of allowed_tools.
    - model / provider: override AgentConfig if set.
    - system_prompt_prefix: prepended to the system prompt.
    - max_turns / planning_enabled: override corresponding AgentConfig fields if set.
    """

    name: str
    description: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    model: Optional[str] = None
    provider: Optional[str] = None
    system_prompt_prefix: Optional[str] = None
    max_turns: Optional[int] = None
    planning_enabled: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, name: str, repo_root: str) -> "AgentProfile":
        """Load a profile from .asicode/agents/{name}.json.

        Args:
            name: Profile name (filename without .json).
            repo_root: Repository root directory.

        Returns:
            AgentProfile instance.

        Raises:
            FileNotFoundError: If the profile file does not exist.
            ValueError: If the JSON is malformed.
        """
        profile_path = Path(repo_root) / ".asicode" / "agents" / f"{name}.json"
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Agent profile '{name}' not found at {profile_path}"
            )
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed agent profile JSON at {profile_path}: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentProfile":
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            allowed_tools=data.get("allowed_tools") or [],
            blocked_tools=data.get("blocked_tools") or [],
            model=data.get("model"),
            provider=data.get("provider"),
            system_prompt_prefix=data.get("system_prompt_prefix"),
            max_turns=data.get("max_turns"),
            planning_enabled=data.get("planning_enabled"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
            "model": self.model,
            "provider": self.provider,
            "system_prompt_prefix": self.system_prompt_prefix,
            "max_turns": self.max_turns,
            "planning_enabled": self.planning_enabled,
        }

    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #

    def apply(self, agent_config: "AgentConfig") -> None:
        """Override AgentConfig fields from this profile (non-None values only)."""
        if self.model is not None:
            agent_config.model = self.model
            logger.debug("AgentProfile '%s': model → %s", self.name, self.model)
        if self.provider is not None:
            agent_config.provider = self.provider
            logger.debug("AgentProfile '%s': provider → %s", self.name, self.provider)
        if self.max_turns is not None:
            agent_config.max_turns = self.max_turns
            logger.debug("AgentProfile '%s': max_turns → %d", self.name, self.max_turns)
        if self.planning_enabled is not None:
            agent_config.planning_enabled = self.planning_enabled
            logger.debug(
                "AgentProfile '%s': planning_enabled → %s",
                self.name,
                self.planning_enabled,
            )


# ------------------------------------------------------------------ #
# Built-in profiles                                                    #
# ------------------------------------------------------------------ #

BUILTIN_PROFILES: dict[str, dict] = {
    "reviewer": {
        "name": "reviewer",
        "description": "Read-only code review — no file modifications allowed",
        "allowed_tools": [
            "find_symbol", "find_references",
            "get_project_info", "find_relevant_files",
        ],
        "blocked_tools": [],
        "planning_enabled": False,
    },
    "patcher": {
        "name": "patcher",
        "description": "Edit-focused agent — read + patch only, no tests or shell",
        "allowed_tools": [
            "find_symbol", "find_references",
            "write_plan", "apply_patch", "find_relevant_files",
        ],
        "blocked_tools": ["bash"],
    },
    "tester": {
        "name": "tester",
        "description": "Test-focused agent — runs tests and reports results",
        "allowed_tools": [
            "find_symbol", "get_project_info",
            # run_tests removed from LLM tool set (internal dispatch only; use bash("pytest ..."))
        ],
        "blocked_tools": ["apply_patch", "bash"],
        "planning_enabled": False,
    },
}


def get_builtin_profile(name: str) -> Optional[AgentProfile]:
    """Return a built-in profile by name, or None if not found."""
    data = BUILTIN_PROFILES.get(name)
    return AgentProfile.from_dict(data) if data else None


def load_profile(name: str, repo_root: str) -> AgentProfile:
    """Load a profile: file-based first, then built-ins, then error.

    Args:
        name: Profile name.
        repo_root: Repository root.

    Returns:
        AgentProfile instance.

    Raises:
        ValueError: If not found in files or built-ins.
    """
    try:
        return AgentProfile.load(name, repo_root)
    except FileNotFoundError:
        pass
    builtin = get_builtin_profile(name)
    if builtin is not None:
        return builtin
    raise ValueError(
        f"Agent profile '{name}' not found. "
        f"Create .asicode/agents/{name}.json or use a built-in: {list(BUILTIN_PROFILES)}"
    )
