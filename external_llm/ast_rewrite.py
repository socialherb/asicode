from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from pathlib import Path

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

    # ---------------------------------------------------------
    # public API
    # ---------------------------------------------------------

    def replace_function(
        self,
        file_path: str,
        function_name: str,
        new_code: str
    ) -> RewriteResult:

        source, tree = self._load_ast(file_path)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    return self._replace_node(source, node, new_code, function_name)

        raise ValueError(f"Function not found: {function_name}")

    def replace_class(
        self,
        file_path: str,
        class_name: str,
        new_code: str
    ) -> RewriteResult:

        source, tree = self._load_ast(file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == class_name:
                    return self._replace_node(source, node, new_code, class_name)

        raise ValueError(f"Class not found: {class_name}")

    def replace_method(
        self,
        file_path: str,
        class_name: str,
        method_name: str,
        new_code: str
    ) -> RewriteResult:
        """Replace a method inside a class.

        ``class_name`` may be a dotted path for nested classes, e.g.
        ``"OuterClass.InnerClass"``.  Each component is resolved in order
        through the AST class hierarchy.
        """
        source, tree = self._load_ast(file_path)

        # Walk the class chain (supports nested classes like "A.B")
        class_chain = class_name.split(".")
        current_body: list = tree.body
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
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == method_name:
                    return self._replace_node(
                        source,
                        item,
                        new_code,
                        f"{class_name}.{method_name}"
                    )

        raise ValueError(f"Method not found: {class_name}.{method_name}")

    def replace_by_line_range(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        new_code: str,
        symbol: str = "",
    ) -> RewriteResult:
        """Replace lines [start_line..end_line] (1-indexed inclusive) with new_code.

        Bypasses AST name matching entirely — uses line numbers from ``SymbolDef``
        (``SymbolDef.line`` → ``start_line``, ``SymbolDef.end_line`` → ``end_line``).
        Safe for same-named methods in different classes and nested classes.
        """
        path = self.repo_root / file_path
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()

        s = start_line - 1   # convert to 0-indexed
        e = end_line         # 0-indexed exclusive (SymbolDef.end_line is inclusive)

        new_lines = new_code.splitlines()
        updated = lines[:s] + new_lines + lines[e:]
        new_text = "\n".join(updated) + "\n"

        return RewriteResult(
            old_text=source,
            new_text=new_text,
            start_line=s,
            end_line=e,
            symbol=symbol or f"lines:{start_line}-{end_line}",
        )

    def replace_symbol(
        self,
        file_path: str,
        symbol: str,
        new_code: str
    ) -> RewriteResult:

        """
        Auto detect symbol type
        """

        source, tree = self._load_ast(file_path)

        for node in ast.walk(tree):

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == symbol:
                    return self._replace_node(source, node, new_code, symbol)

            if isinstance(node, ast.ClassDef):
                if node.name == symbol:
                    return self._replace_node(source, node, new_code, symbol)

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol:
                        return self._replace_node(source, node, new_code, symbol)

            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == symbol:
                    return self._replace_node(source, node, new_code, symbol)

        raise ValueError(f"Symbol not found: {symbol}")

    # ---------------------------------------------------------
    # fallback anchor replace
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # fuzzy fallback
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # patch generation
    # ---------------------------------------------------------

    def generate_patch(
        self,
        file_path: str,
        result: RewriteResult
    ) -> str:

        rel = file_path

        diff = difflib.unified_diff(
            result.old_text.splitlines(True),
            result.new_text.splitlines(True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm=""
        )

        body = "".join(diff)

        patch = f"diff --git a/{rel} b/{rel}\n{body}"

        return patch

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------

    def _load_ast(self, file_path: str) -> tuple[str, ast.AST]:

        path = self.repo_root / file_path

        source = path.read_text(encoding="utf-8")

        tree = ast.parse(source)

        return source, tree

    def _replace_node(
        self,
        source: str,
        node: ast.AST,
        new_code: str,
        symbol: str
    ) -> RewriteResult:

        start = node.lineno - 1
        end = node.end_lineno

        lines = source.splitlines()

        new_lines = new_code.splitlines()

        updated = lines[:start] + new_lines + lines[end:]

        new_text = "\n".join(updated) + "\n"

        return RewriteResult(
            old_text=source,
            new_text=new_text,
            start_line=start,
            end_line=end,
            symbol=symbol
        )
