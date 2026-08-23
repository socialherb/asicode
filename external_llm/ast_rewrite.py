from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from pathlib import Path

from common import EDIT_TARGET_MAX_BYTES

# -------------------------------------------------------------
# Result container
# -------------------------------------------------------------


@dataclass
class RewriteResult:
    old_text: str
    new_text: str
    start_line: int
    end_line: int
    symbol: str


# -------------------------------------------------------------
# AST Rewriter
# -------------------------------------------------------------


class ASTRewriter:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    # P21-3: same policy as the P19-4 rewrite guard (webapp) — refuse before
    # the full read so a multi-hundred-MB target cannot OOM the AST fallback
    # path (read_text + ast.parse would otherwise load it entirely).
    # P9-2: value flows from the shared SSOT (external_llm/common); the class
    # attribute keeps its name for existing callers but is no longer a local
    # hardcoded copy.
    _MAX_EDIT_BYTES = EDIT_TARGET_MAX_BYTES  # 64 MiB (SSOT: external_llm/common)

    # ---------------------------------------------------------
    # public API
    # ---------------------------------------------------------

    def replace_function(self, file_path: str, function_name: str, new_code: str) -> RewriteResult:

        source, tree = self._load_ast(file_path)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return self._replace_node(source, node, new_code, function_name)

        raise ValueError(f"Function not found: {function_name}")

    def replace_class(self, file_path: str, class_name: str, new_code: str) -> RewriteResult:

        source, tree = self._load_ast(file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return self._replace_node(source, node, new_code, class_name)

        raise ValueError(f"Class not found: {class_name}")

    def replace_method(self, file_path: str, class_name: str, method_name: str, new_code: str) -> RewriteResult:
        """Replace a method inside a class.

        ``class_name`` may be a dotted path for nested classes, e.g.
        ``"OuterClass.InnerClass"``.  Each component is resolved in order
        through the AST class hierarchy.
        """
        source, tree = self._load_ast(file_path)

        # Walk the class chain (supports nested classes like "A.B")
        class_chain = class_name.split(".")
        current_body: list = tree.body  # type: ignore[attr-defined]  # ast.Module.body present at runtime
        for cls_name in class_chain:
            found = None
            for node in current_body:
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    found = node
                    break
            if found is None:
                raise ValueError(f"Class not found: {cls_name} (in chain {class_name!r})")
            current_body = found.body

        for item in current_body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                return self._replace_node(source, item, new_code, f"{class_name}.{method_name}")

        raise ValueError(f"Method not found: {class_name}.{method_name}")

    # ---------------------------------------------------------
    # fallback anchor replace
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # fuzzy fallback
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # patch generation
    # ---------------------------------------------------------

    def generate_patch(self, file_path: str, result: RewriteResult) -> str:

        rel = file_path

        diff = difflib.unified_diff(
            result.old_text.splitlines(True),
            result.new_text.splitlines(True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="\n",
        )

        # lineterm="\n" gives the control lines (---/+++/@@) their newline so a
        # bare "".join is well-formed. (lineterm="" + "\n".join double-newlined
        # every body line — body lines keep their newline from splitlines(True)
        # — producing a patch git apply rejects as corrupt.)
        body = "".join(diff)
        if body:
            body += "\n"

        return f"diff --git a/{rel} b/{rel}\n{body}"

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------

    def _load_ast(self, file_path: str) -> tuple[str, ast.AST]:

        path = self.repo_root / file_path

        try:
            if path.stat().st_size > self._MAX_EDIT_BYTES:
                raise ValueError(f"file too large for AST rewrite (>{self._MAX_EDIT_BYTES // (1024 * 1024)}MiB)")
        except OSError as e:
            raise ValueError(f"cannot stat {file_path}: {e}") from e

        source = path.read_text(encoding="utf-8")

        tree = ast.parse(source)

        return source, tree

    def _replace_node(self, source: str, node: ast.AST, new_code: str, symbol: str) -> RewriteResult:
        # node.lineno points at the `def`/`class` line — decorators live ABOVE
        # it and are NOT covered by lineno. Slicing at `lineno - 1` would leave
        # the original decorators in `lines[:start]`, and since new_code carries
        # a complete symbol (decorators included), they would be re-introduced
        # and applied twice (silently wrong for side-effecting decorators like
        # @property / @lru_cache / @app.route). Start the replacement at the
        # topmost decorator when one is present. Same policy as semantic_patch.py
        # and symbol_modify_tool.py.
        decorator_list = getattr(node, "decorator_list", []) or []
        start = (min(d.lineno for d in decorator_list) - 1) if decorator_list else (node.lineno - 1)  # type: ignore[attr-defined]  # AST stmt node
        end = node.end_lineno  # type: ignore[attr-defined]  # AST stmt node

        lines = source.splitlines()

        new_lines = new_code.splitlines()

        updated = lines[:start] + new_lines + lines[end:]

        new_text = "\n".join(updated) + "\n"

        return RewriteResult(old_text=source, new_text=new_text, start_line=start, end_line=end, symbol=symbol)
