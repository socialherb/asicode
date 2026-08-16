"""
AST-based Code Analyzer for Enhanced Context

Analyzes Python code to extract:
- Functions and classes with signatures
- Type hints and return types
- Docstrings
- Dependencies (imports and calls)
- Code patterns
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    args: list[str]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    line_number: int = 0
    is_async: bool = False
    type_hints: dict[str, str] = field(default_factory=dict)
    # Call targets made within THIS function's own scope (not file-level).
    # Used by DependencyGraphBuilder._track_internal_calls for caller-accurate
    # edges instead of the lossy whole-file call set. (DG-B1)
    calls: set[str] = field(default_factory=set)


@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    bases: list[str]
    methods: list[FunctionInfo] = field(default_factory=list)
    docstring: Optional[str] = None
    line_number: int = 0
    decorators: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Information about an import"""
    module: str
    names: list[str] = field(default_factory=list)
    alias: Optional[str] = None


@dataclass
class CodeAnalysis:
    """Complete analysis of a Python file"""
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    global_vars: dict[str, str] = field(default_factory=dict)
    calls: set[str] = field(default_factory=set)  # Functions called
    module_docstring: Optional[str] = None


class CodeAnalyzer:
    """
    Analyzes Python code using AST

    Extracts detailed information about code structure,
    types, dependencies, and patterns.
    """

    def analyze_file(self, file_path: Path) -> Optional[CodeAnalysis]:
        """
        Analyze a Python file

        Args:
            file_path: Path to Python file

        Returns:
            CodeAnalysis or None if parse fails
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))
            return self._analyze_ast(tree)
        except Exception as e:
            logger.debug("Failed to analyze %s: %s", file_path, e)
            return None

    def _analyze_ast(self, tree: ast.Module) -> CodeAnalysis:
        """Analyze AST tree"""
        analysis = CodeAnalysis()

        # Module docstring
        analysis.module_docstring = ast.get_docstring(tree)

        # Identity set of top-level body nodes. ``ast.walk`` visits every node
        # (n); the prior ``_is_top_level`` re-scanned ``tree.body`` (m items) on
        # every qualifying node — O(n*m). A precomputed id-set gives O(1)
        # membership tests with identical identity semantics (AST nodes compare
        # by identity, so ``item == node`` was already ``item is node``).
        # (CA-P1: O(n*m) -> O(n).)
        top_level_ids = {id(item) for item in tree.body}

        # Walk through all nodes
        for node in ast.walk(tree):
            # Functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if id(node) in top_level_ids:
                    func_info = self._extract_function_info(node)
                    analysis.functions.append(func_info)

            # Classes
            elif isinstance(node, ast.ClassDef):
                if id(node) in top_level_ids:
                    class_info = self._extract_class_info(node)
                    analysis.classes.append(class_info)

            # Imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis.imports.append(ImportInfo(
                        module=alias.name,
                        alias=alias.asname,
                    ))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Encode the relative-import level as leading dots so downstream
                # consumers can tell a relative import (``from .sibling import x``)
                # from an absolute one (``from sibling import x``). ``node.module``
                # alone is always dot-free; the level lives in ``node.level``.
                # Without this, DependencyGraphBuilder._resolve_import takes the
                # absolute branch and silently mis-resolves in-package relative
                # imports (or matches a stray same-named file at repo root).
                # Mirrors the canonical pattern at ast_op_executor.py:1018 and
                # tool_safety.py:634 — code_analyzer was the sole outlier.
                if node.level:
                    module = "." * node.level + module
                names = [alias.name for alias in node.names]
                analysis.imports.append(ImportInfo(
                    module=module,
                    names=names,
                ))

            # Function calls
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    analysis.calls.add(call_name)

            # Global assignments (constants, config). Two node kinds share a
            # ``.value`` but differ in targets:
            #   Assign    `X = 5`       — ``node.targets`` may name several
            #   AnnAssign `X: int = 5`  — single ``node.target``; ``.value`` is
            #       Optional (None for an annotation-only `x: int` declaration,
            #       excluded by the ``node.value is not None`` guard below).
            # Both feed the LLM-facing "Type aliases/Constants" block; AnnAssign
            # was dropped entirely before. (CA-B3)
            elif (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and id(node) in top_level_ids
                and node.value is not None  # skip annotation-only `x: int`
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        analysis.global_vars[target.id] = self._node_to_string(node.value)

        return analysis

    def _extract_function_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
        """Extract function information, preserving every parameter kind.

        Assembles positional-only (before ``/``), positional-or-keyword,
        ``*vararg``, keyword-only (after ``*``), and ``**kwarg`` so the rendered
        signature is faithful to the source. Separator markers (``/`` and ``*``)
        are stored as pseudo-arg strings; ``format_function_signature`` appends
        unknown strings verbatim, so they render as the correct Python
        separators. (CA-B2: previously only ``node.args.args`` was iterated,
        silently dropping every other parameter kind from the LLM-facing
        signature.)
        """
        args: list[str] = []
        type_hints: dict[str, str] = {}

        sig_args = node.args

        # positional-only (before '/')
        for arg in sig_args.posonlyargs:
            args.append(arg.arg)
            if arg.annotation:
                type_hints[arg.arg] = self._node_to_string(arg.annotation)
        if sig_args.posonlyargs:
            args.append("/")

        # positional-or-keyword
        for arg in sig_args.args:
            args.append(arg.arg)
            if arg.annotation:
                type_hints[arg.arg] = self._node_to_string(arg.annotation)

        # *vararg, or a bare '*' separator when keyword-only args follow without one
        if sig_args.vararg:
            marker = f"*{sig_args.vararg.arg}"
            args.append(marker)
            if sig_args.vararg.annotation:
                type_hints[marker] = self._node_to_string(sig_args.vararg.annotation)
        elif sig_args.kwonlyargs:
            args.append("*")

        # keyword-only (after '*')
        for arg in sig_args.kwonlyargs:
            args.append(arg.arg)
            if arg.annotation:
                type_hints[arg.arg] = self._node_to_string(arg.annotation)

        # **kwarg
        if sig_args.kwarg:
            marker = f"**{sig_args.kwarg.arg}"
            args.append(marker)
            if sig_args.kwarg.annotation:
                type_hints[marker] = self._node_to_string(sig_args.kwarg.annotation)

        # Per-function calls (DG-B1): Call targets in THIS function's body only,
        # without descending into nested function/class definitions.
        func_calls: set[str] = set()
        self._collect_calls(node, func_calls)

        # Return type
        return_type = None
        if node.returns:
            return_type = self._node_to_string(node.returns)

        # Decorators
        decorators = [self._node_to_string(dec) for dec in node.decorator_list]

        # Docstring
        docstring = ast.get_docstring(node)

        return FunctionInfo(
            name=node.name,
            args=args,
            return_type=return_type,
            docstring=docstring,
            decorators=decorators,
            line_number=node.lineno,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            type_hints=type_hints,
            calls=func_calls,
        )

    def _collect_calls(self, node: ast.AST, out: set[str]) -> None:
        """Collect function-call target names within ``node``'s own scope.

        Skips nested function/class definitions so their calls are not
        mis-attributed to the enclosing scope (DG-B1: caller-accurate edges).
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # nested scope — its calls belong to it, not to this scope
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if call_name:
                    out.add(call_name)
            self._collect_calls(child, out)

    def _extract_class_info(self, node: ast.ClassDef) -> ClassInfo:
        """Extract class information"""
        # Base classes
        bases = [self._node_to_string(base) for base in node.bases]

        # Methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._extract_function_info(item)
                methods.append(method_info)

        # Decorators
        decorators = [self._node_to_string(dec) for dec in node.decorator_list]

        # Docstring
        docstring = ast.get_docstring(node)

        return ClassInfo(
            name=node.name,
            bases=bases,
            methods=methods,
            docstring=docstring,
            line_number=node.lineno,
            decorators=decorators,
        )

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get the name of a function call"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _node_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string"""
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for older Python
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Constant):
                return repr(node.value)
            if isinstance(node, ast.Attribute):
                return f"{self._node_to_string(node.value)}.{node.attr}"
            return str(type(node).__name__)

    def format_function_signature(self, func: FunctionInfo) -> str:
        """Format function signature with types"""
        # Decorators
        lines = [f"@{dec}" for dec in func.decorators]

        # Build signature
        prefix = "async def" if func.is_async else "def"

        # Arguments with types
        args_str = []
        for arg in func.args:
            if arg in func.type_hints:
                args_str.append(f"{arg}: {func.type_hints[arg]}")
            else:
                args_str.append(arg)

        signature = f"{prefix} {func.name}({', '.join(args_str)})"

        # Return type
        if func.return_type:
            signature += f" -> {func.return_type}"

        signature += ":"
        lines.append(signature)

        # Docstring (first line only)
        if func.docstring:
            first_line = func.docstring.split('\n')[0].strip()
            lines.append(f'    """{first_line}"""')

        return '\n'.join(lines)

    def format_class_signature(self, cls: ClassInfo) -> str:
        """Format class signature"""

        # Decorators
        lines = [f"@{dec}" for dec in cls.decorators]

        # Class definition
        if cls.bases:
            bases_str = ', '.join(cls.bases)
            lines.append(f"class {cls.name}({bases_str}):")
        else:
            lines.append(f"class {cls.name}:")

        # Docstring
        if cls.docstring:
            first_line = cls.docstring.split('\n')[0].strip()
            lines.append(f'    """{first_line}"""')

        # Methods (just signatures)
        if cls.methods:
            lines.append("")
            for method in cls.methods[:5]:  # Show first 5 methods
                method_sig = self.format_function_signature(method)
                # Indent
                indented = '\n'.join('    ' + line for line in method_sig.split('\n'))
                lines.append(indented)

        return '\n'.join(lines)



