from __future__ import annotations

import ast
import difflib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SemanticPatchResult:
    old_text: str
    new_text: str
    start_line: int
    end_line: int
    symbol: str
    kind: str


class SemanticPatchEngine:
    """
    MVP semantic patch engine.

    Goal:
    - Accept a code block produced by the LLM
    - Parse it as Python
    - Find the matching symbol in the target file
    - Replace only that symbol block
    - Generate a unified diff server-side

    Supported in MVP:
    - def
    - async def
    - class
    """

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    # ---------------------------------------------------------
    # public API
    # ---------------------------------------------------------

    def apply_semantic_patch(
        self,
        file_path: str,
        new_code: str,
    ) -> Optional[SemanticPatchResult]:
        """
        Try semantic patching for a single top-level Python symbol.

        Returns SemanticPatchResult on success, else None.
        """
        try:
            normalized = str(new_code or "").strip()
            if not normalized:
                return None

            source, tree = self._load_ast(file_path)
            new_tree = ast.parse(normalized)

            if not new_tree.body:
                return None

            node0 = new_tree.body[0]

            if isinstance(node0, ast.AsyncFunctionDef):
                return self._replace_function_like(
                    source=source,
                    tree=tree,
                    new_code=normalized,
                    function_name=node0.name,
                    kind="async_function",
                )

            if isinstance(node0, ast.FunctionDef):
                return self._replace_function_like(
                    source=source,
                    tree=tree,
                    new_code=normalized,
                    function_name=node0.name,
                    kind="function",
                )

            if isinstance(node0, ast.ClassDef):
                return self._replace_class_like(
                    source=source,
                    tree=tree,
                    new_code=normalized,
                    class_name=node0.name,
                )

            return None

        except Exception as e:
            logger.debug("Semantic patch apply failed for %s: %s", file_path, e)
            return None

    def generate_patch(self, file_path: str, result: SemanticPatchResult) -> str:
        rel = file_path

        diff = difflib.unified_diff(
            result.old_text.splitlines(True),
            result.new_text.splitlines(True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )

        body = "".join(diff)
        return f"diff --git a/{rel} b/{rel}\n{body}"

    # ---------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------

    def _load_ast(self, file_path: str):
        path = self.repo_root / file_path
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        return source, tree

    def _replace_function_like(
        self,
        *,
        source: str,
        tree: ast.AST,
        new_code: str,
        function_name: str,
        kind: str,
    ) -> Optional[SemanticPatchResult]:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return self._replace_node(
                    source=source,
                    node=node,
                    new_code=new_code,
                    symbol=function_name,
                    kind=kind,
                )
        return None

    def _replace_class_like(
        self,
        *,
        source: str,
        tree: ast.AST,
        new_code: str,
        class_name: str,
    ) -> Optional[SemanticPatchResult]:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return self._replace_node(
                    source=source,
                    node=node,
                    new_code=new_code,
                    symbol=class_name,
                    kind="class",
                )
        return None

    def _replace_node(
        self,
        *,
        source: str,
        node: ast.AST,
        new_code: str,
        symbol: str,
        kind: str,
    ) -> SemanticPatchResult:
        start = node.lineno - 1
        end = node.end_lineno

        lines = source.splitlines()
        new_lines = new_code.splitlines()

        updated = lines[:start] + new_lines + lines[end:]
        new_text = "\n".join(updated) + "\n"

        return SemanticPatchResult(
            old_text=source,
            new_text=new_text,
            start_line=start,
            end_line=end,
            symbol=symbol,
            kind=kind,
        )
