"""fragment_generator.py — Phase D: Minimal Fragment Generation.

Converts semantic gaps into concrete code fragments using templates.
Does NOT write files — only produces GeneratedFragment objects.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeneratedFragment:
    """A minimal code fragment ready for integration."""
    fragment_type: str     # "entity" | "persistence" | "schema" | "output_ref"
    target_file: str       # Relative path to write to
    target_role: str       # "model" | "service" | "route" | "schema"
    content: str           # Code content
    unique_key: str        # Idempotency marker
    description: str = ""
    patch_kind: str = "insert_body"  # SectionPatcher kind
    priority: int = 0      # Lower = applied first
