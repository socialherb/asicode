"""generation_planner.py — Phase D: Generation Planning (disabled).

Fragment generation is disabled (Phase 5).  This module exists to preserve
the import chain; build_generation_plan always returns [].
"""
from __future__ import annotations
from typing import Optional

from external_llm.editor.semantic.fragment_generator import GeneratedFragment
from external_llm.editor.semantic.semantic_gap_analyzer import SemanticGap
from external_llm.editor.semantic.semantic_tracer import SemanticTrace
def build_generation_plan(
    gaps: list[SemanticGap],
    trace: SemanticTrace,
    repo_root: str = ".",
    context_tags: Optional[list[str]] = None,
) -> list[GeneratedFragment]:
    """DISABLED — fragment generation is code generation, not repair.  Returns []."""
    return []
