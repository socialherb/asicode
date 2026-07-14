"""
Context packs for different agent roles.

HelperContextBuilder – minimal function/snippet-level context.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
@dataclass
class ContextPack:
    """Container for a rendered context string and metadata."""
    content: str
    metadata: dict[str, Any]


class HelperContextBuilder:
    """Build minimal context for helper (small code generation) phase."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def build(
        self,
        task: str,
        function_signature: Optional[str] = None,
        local_snippet: Optional[str] = None,
        constraints: Optional[str] = None,
    ) -> ContextPack:
        """Build helper context."""
        lines: list[str] = []
        metadata: dict[str, Any] = {}

        lines.append("## Task")
        lines.append(task)
        lines.append("")

        if function_signature:
            lines.append("### Function Signature")
            lines.append(f"```python\n{function_signature}\n```")
            lines.append("")

        if local_snippet:
            lines.append("### Local Context")
            lines.append(f"```python\n{local_snippet}\n```")
            lines.append("")

        if constraints:
            lines.append("### Constraints")
            lines.append(constraints)
            lines.append("")

        metadata.update({
            "has_signature": bool(function_signature),
            "has_snippet": bool(local_snippet),
            "has_constraints": bool(constraints),
        })

        return ContextPack(content="\n".join(lines), metadata=metadata)
