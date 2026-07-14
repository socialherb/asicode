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
    is_from_import: bool = False


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
            logger.debug(f"Failed to analyze {file_path}: {e}")
            return None

    def _analyze_ast(self, tree: ast.Module) -> CodeAnalysis:
        """Analyze AST tree"""
        analysis = CodeAnalysis()

        # Module docstring
        analysis.module_docstring = ast.get_docstring(tree)

        # Walk through all nodes
        for node in ast.walk(tree):
            # Functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_top_level(node, tree):
                    func_info = self._extract_function_info(node)
                    analysis.functions.append(func_info)

            # Classes
            elif isinstance(node, ast.ClassDef):
                if self._is_top_level(node, tree):
                    class_info = self._extract_class_info(node)
                    analysis.classes.append(class_info)

            # Imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis.imports.append(ImportInfo(
                        module=alias.name,
                        alias=alias.asname,
                        is_from_import=False
                    ))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                analysis.imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from_import=True
                ))

            # Function calls
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    analysis.calls.add(call_name)

            # Global assignments (constants, config)
            elif isinstance(node, ast.Assign):
                if self._is_top_level(node, tree):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            value_str = self._node_to_string(node.value)
                            analysis.global_vars[target.id] = value_str

        return analysis

    def _is_top_level(self, node: ast.AST, tree: ast.Module) -> bool:
        """Check if node is at module level (not nested)"""
        for item in tree.body:
            if item == node:
                return True
        return False

    def _extract_function_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
        """Extract function information"""
        # Arguments
        args = []
        type_hints = {}

        for arg in node.args.args:
            args.append(arg.arg)
            if arg.annotation:
                type_hints[arg.arg] = self._node_to_string(arg.annotation)

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
        )

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
        elif isinstance(node.func, ast.Attribute):
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
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Attribute):
                return f"{self._node_to_string(node.value)}.{node.attr}"
            return str(type(node).__name__)

    def format_function_signature(self, func: FunctionInfo) -> str:
        """Format function signature with types"""
        # Decorators
        lines = []
        for dec in func.decorators:
            lines.append(f"@{dec}")

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
        lines = []

        # Decorators
        for dec in cls.decorators:
            lines.append(f"@{dec}")

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



