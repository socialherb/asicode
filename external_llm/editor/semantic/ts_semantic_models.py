"""ts_semantic_models.py — Data models for TS/JS semantic analysis.

Mirrors Python's FunctionTrace / SemanticTrace with frontend-specific
concepts: components, hooks, state variables, event handlers, JSX trees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TSImport:
    """A single import statement."""

    source: str  # e.g. 'react', './utils'
    specifiers: list[str] = field(default_factory=list)  # named imports
    default_import: Optional[str] = None  # default import name
    is_type_only: bool = False  # `import type { ... }`


@dataclass
class TSStateVar:
    """A React state variable declared via useState."""

    name: str  # e.g. 'count'
    setter: str  # e.g. 'setCount'
    initial_value: Optional[str] = None  # e.g. '0'


@dataclass
class TSHook:
    """A React hook call (useState, useEffect, useMemo, etc.)."""

    name: str  # e.g. 'useState', 'useEffect'
    deps: Optional[list[str]] = None  # dependency array items (useEffect, useMemo)


@dataclass
class TSEventHandler:
    """An event handler wired in JSX."""

    event_name: str  # e.g. 'onClick', 'onChange'
    handler_expr: str  # handler expression text


@dataclass
class TSProp:
    """A component prop."""

    name: str
    type_annotation: Optional[str] = None
    has_default: bool = False


@dataclass
class TSComponent:
    """A React component extracted from source."""

    name: str
    props: list[TSProp] = field(default_factory=list)
    state_vars: list[TSStateVar] = field(default_factory=list)
    hooks: list[TSHook] = field(default_factory=list)
    events: list[TSEventHandler] = field(default_factory=list)
    jsx_root: Optional[str] = None  # root JSX element tag name
    is_exported: bool = False
    start_line: int = 0
    end_line: int = 0


@dataclass
class TSFunction:
    """A non-component function/arrow function."""

    name: str
    params: list[str] = field(default_factory=list)
    is_exported: bool = False
    is_async: bool = False
    start_line: int = 0
    end_line: int = 0


@dataclass
class TSModuleSemantic:
    """Complete semantic model for a single TS/JS module.

    Analogous to Python's SemanticTrace but structured around
    frontend concepts: components, hooks, state, events, JSX.
    """

    file_path: str
    components: list[TSComponent] = field(default_factory=list)
    functions: list[TSFunction] = field(default_factory=list)
    imports: list[TSImport] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)  # exported names

    # ── convenience helpers ──────────────────────────────────────────────

    @property
    def all_state_vars(self) -> list[TSStateVar]:
        """All state variables across all components."""
        return [sv for c in self.components for sv in c.state_vars]

    @property
    def all_hooks(self) -> list[str]:
        """Unique hook names used across all components."""
        return list({h.name for c in self.components for h in c.hooks})

    @property
    def all_events(self) -> list[str]:
        """Unique event names across all components."""
        return list({e.event_name for c in self.components for e in c.events})
