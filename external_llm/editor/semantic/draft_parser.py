"""draft_parser.py — Phase F: LLM Output Draft Parser.

Parses LLM-generated files as a structural draft:
- What actions (functions/endpoints) exist
- What entities are referenced
- What file roles are present (route/service/model)
- What action types are implied
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.editor.semantic.primitive_registry import infer_action_type
from external_llm.languages import LanguageId

logger = logging.getLogger(__name__)

_FORBIDDEN = ("venv", "site-packages", "node_modules", ".git", "__pycache__")


@dataclass
class DraftAction:
    """An action (function) found in the draft."""
    name: str
    action_type: str         # Inferred: "create", "login", "send", etc.
    file_path: str = ""
    entity: str = ""         # Primary entity this action works with
    params: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    has_return: bool = False
    has_decorator: bool = False  # HTTP endpoint decorator


@dataclass
class DraftResult:
    """Parsed draft of LLM output."""
    actions: list[DraftAction] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    files_by_role: dict[str, list[str]] = field(default_factory=dict)
    # "route": [...], "service": [...], "model": [...]
    context_tags: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "actions": [(a.name, a.action_type, a.entity) for a in self.actions],
            "entities": self.entities,
            "roles": {k: len(v) for k, v in self.files_by_role.items()},
        }


def parse_draft(
    file_paths: list[str],
    repo_root: str = ".",
    trace: Any = None,
    context_tags: Optional[list[str]] = None,
) -> DraftResult:
    """Parse LLM output files as a structural draft."""
    result = DraftResult(context_tags=list(context_tags or []))
    seen_entities: set[str] = set()

    for path in file_paths:
        abs_path = path if os.path.isabs(path) else os.path.join(repo_root, path)
        if LanguageId.from_path(abs_path) is not LanguageId.PYTHON or not os.path.isfile(abs_path):
            continue
        if any(p in abs_path for p in _FORBIDDEN):
            continue

        _parse_file(abs_path, repo_root, result, seen_entities)

    # Infer entities from trace if available
    if trace:
        for cls in getattr(trace, 'all_classes', set()):
            if cls not in seen_entities and cls[0].isupper():
                result.entities.append(cls)
                seen_entities.add(cls)

    # Assign entities to actions
    _assign_entities_to_actions(result)

    logger.debug(
        "[DRAFT] %d actions, %d entities, roles=%s",
        len(result.actions), len(result.entities),
        {k: len(v) for k, v in result.files_by_role.items()},
    )

    return result


def _parse_file(abs_path: str, repo_root: str, result: DraftResult, seen_entities: set[str]) -> None:
    """Parse a single file."""
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception:
        return

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    rel_path = os.path.relpath(abs_path, repo_root) if repo_root else abs_path
    role = _infer_file_role(rel_path)
    result.files_by_role.setdefault(role, []).append(rel_path)

    # Extract classes (entities)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if node.name not in seen_entities and node.name[0].isupper():
                result.entities.append(node.name)
                seen_entities.add(node.name)

    # Extract functions (actions)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            action = _parse_function(node, rel_path)
            if action:
                result.actions.append(action)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name != "__init__":
                        action = _parse_function(item, rel_path)
                        if action:
                            result.actions.append(action)


def _parse_function(node: ast.FunctionDef, file_path: str) -> Optional[DraftAction]:
    """Parse a function into a DraftAction."""
    name = node.name
    if name.startswith("_") and name != "__init__":
        return None

    action_type = infer_action_type(name)
    params = [a.arg for a in node.args.args if a.arg != "self"]

    calls: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(f"{_get_name(child.func.value)}.{child.func.attr}")

    has_return = any(isinstance(s, ast.Return) for s in ast.walk(node))
    has_decorator = bool(node.decorator_list)

    return DraftAction(
        name=name,
        action_type=action_type,
        file_path=file_path,
        params=params,
        calls=calls,
        has_return=has_return,
        has_decorator=has_decorator,
    )


def _assign_entities_to_actions(result: DraftResult) -> None:
    """Assign the most likely entity to each action based on calls/name."""
    entity_lower = {e.lower(): e for e in result.entities}

    for action in result.actions:
        # Check if action calls an entity constructor
        for call in action.calls:
            bare = call.split(".")[-1]
            if bare in entity_lower.values():
                action.entity = bare
                break

        # Fallback: infer from action name
        if not action.entity:
            for entity in result.entities:
                if entity.lower() in action.name.lower():
                    action.entity = entity
                    break

        # Fallback: first entity
        if not action.entity and result.entities:
            action.entity = result.entities[0]


def _infer_file_role(rel_path: str) -> str:
    """Infer file role from path."""
    path_lower = rel_path.lower()
    if "route" in path_lower or "view" in path_lower or "controller" in path_lower or "endpoint" in path_lower:
        return "route"
    if "service" in path_lower or "handler" in path_lower or "manager" in path_lower:
        return "service"
    if "model" in path_lower or "schema" in path_lower or "entity" in path_lower:
        return "model"
    if "test" in path_lower:
        return "test"
    return "other"


def _get_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
