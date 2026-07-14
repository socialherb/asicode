"""
Shared enum definitions for asicode Agent.

Extracted from task_router.py to break circular import dependencies and
provide a single source of truth for Scope and Complexity enums used
across intent_models.py, intent_resolver.py, spec_resolver.py, etc.
"""

from __future__ import annotations

from enum import Enum


class Complexity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Scope(str, Enum):
    SINGLE_FILE = "single_file"
    MULTI_FILE = "multi_file"
    PROJECT_WIDE = "project_wide"

class EstimatedScope(str, Enum):
    TINY = 'tiny'
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'


def estimated_scope_to_score(scope: EstimatedScope) -> float:
    mapping = {
        EstimatedScope.TINY: 0.0,
        EstimatedScope.SMALL: 0.1,
        EstimatedScope.MEDIUM: 0.4,
        EstimatedScope.LARGE: 0.7,
    }
    return mapping[scope]


def scope_to_score(scope: Scope) -> float:
    mapping = {
        Scope.SINGLE_FILE: 0.1,
        Scope.MULTI_FILE: 0.4,
        Scope.PROJECT_WIDE: 0.7,
    }
    return mapping[scope]
