"""semantic_tracer.py — Phase C.1: AST-based Semantic Trace Extraction.

Extracts behavioral traces from Python source code:
- Which functions are called within each function
- Which entities (classes) are instantiated
- Which persist-like operations occur (db.add, append, save, etc.)
- What each function returns
- Data bindings (function params → entity constructor args)

Phase C.1 scope: Python AST analysis only, no runtime tracing.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.languages import LanguageId

logger = logging.getLogger(__name__)


@dataclass
class FunctionTrace:
    """Trace of a single function's behavior."""
    name: str
    file_path: str = ""
    calls: set[str] = field(default_factory=set)
    # Functions/classes called within this function
    instantiations: set[str] = field(default_factory=set)
    # Classes instantiated (e.g., Message(...), Video(...))
    persist_calls: set[str] = field(default_factory=set)
    # Persistence patterns: db.add, .append, .save, .commit, store[...] = ...
    return_names: set[str] = field(default_factory=set)
    # Variable names that appear in return statements
    return_has_entity_ref: bool = False
    # Whether return references an instantiated entity
    param_names: list[str] = field(default_factory=list)
    # Function parameter names
    entity_bindings: list[tuple[str, str]] = field(default_factory=list)
    # (entity_field, source) pairs — e.g., ("content", "content") means entity.content = param.content
    has_error_branch: bool = False
    # Whether function has if/raise HTTPException or similar error handling
    error_before_success: bool = False
    # Whether error branch appears before success path (return)
    call_order: list[str] = field(default_factory=list)
    # Ordered list of significant calls (for ordering checks)


@dataclass
class SemanticTrace:
    """Aggregate trace across all analyzed files."""
    function_traces: dict[str, FunctionTrace] = field(default_factory=dict)
    # func_name → FunctionTrace
    all_calls: set[str] = field(default_factory=set)
    all_instantiations: set[str] = field(default_factory=set)
    all_persist_calls: set[str] = field(default_factory=set)
    all_classes: set[str] = field(default_factory=set)
    all_functions: set[str] = field(default_factory=set)

    @property
    def created_entities(self) -> set[str]:
        return self.all_instantiations

    @property
    def persisted_entities(self) -> set[str]:
        return self.all_persist_calls

    @property
    def return_vars(self) -> set[str]:
        result: set[str] = set()
        for ft in self.function_traces.values():
            result.update(ft.return_names)
        return result


# ── Persist-like patterns ─────────────────────────────────────────────────────

_PERSIST_METHODS = {
    "add", "save", "commit", "insert", "put", "store",
    "append", "extend", "update", "create", "write",
}

_PERSIST_OBJECTS = {
    "db", "session", "database", "conn", "cursor", "repo",
    "repository", "store", "collection", "table",
}


# ── AST Extraction ────────────────────────────────────────────────────────────

class _FunctionVisitor(ast.NodeVisitor):
    """Visit a function body and extract behavioral trace."""

    def __init__(self, func_name: str, param_names: list[str], all_classes: set[str]):
        self.trace = FunctionTrace(name=func_name, param_names=param_names)
        self._all_classes = all_classes
        self._local_vars: dict[str, str] = {}  # var_name → class_name (if instantiation)
        self._seen_error_branch = False
        self._seen_return = False

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._extract_call_name(node)
        if call_name:
            self.trace.calls.add(call_name)
            self.trace.call_order.append(call_name)

            # Check if this is a class instantiation
            if call_name in self._all_classes or call_name[0].isupper():
                self.trace.instantiations.add(call_name)
                # Track entity bindings from constructor args
                self._extract_bindings(call_name, node)

            # Check for persist patterns
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in _PERSIST_METHODS:
                    obj_name = self._extract_obj_name(node.func.value)
                    if obj_name in _PERSIST_OBJECTS or method in {"save", "commit"}:
                        self.trace.persist_calls.add(f"{obj_name}.{method}")

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track local variable assignments from instantiations
        if isinstance(node.value, ast.Call):
            call_name = self._extract_call_name(node.value)
            if call_name and (call_name in self._all_classes or call_name[0].isupper()):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._local_vars[target.id] = call_name

        # Check for dict/list store pattern: store[key] = value
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                obj_name = self._extract_obj_name(target.value) if isinstance(target.value, ast.Name) else ""
                if obj_name and ("store" in obj_name.lower() or "db" in obj_name.lower()):
                    self.trace.persist_calls.add(f"{obj_name}[]=")

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Track annotated assignments from instantiations (e.g. obj: MyClass = MyClass(...))
        if node.value and isinstance(node.value, ast.Call):
            call_name = self._extract_call_name(node.value)
            if call_name and (call_name in self._all_classes or call_name[0].isupper()):
                if isinstance(node.target, ast.Name):
                    self._local_vars[node.target.id] = call_name
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self._seen_return = True
        if node.value:
            self._extract_return_names(node.value)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        # Check for error branches (if not ... raise / if ... raise HTTPException)
        has_raise = self._body_has_raise(node.body)
        has_raise_else = self._body_has_raise(node.orelse) if node.orelse else False

        if has_raise or has_raise_else:
            self.trace.has_error_branch = True
            if not self._seen_return:
                self.trace.error_before_success = True

        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.trace.has_error_branch = True
        if not self._seen_return:
            self.trace.error_before_success = True
        self.generic_visit(node)

    def _extract_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _extract_obj_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _extract_bindings(self, class_name: str, call_node: ast.Call) -> None:
        """Extract parameter bindings from constructor call."""
        params = set(self.trace.param_names)
        # Keyword args: Message(content=content, sender_id=sender_id)
        for kw in call_node.keywords:
            if kw.arg and isinstance(kw.value, ast.Name):
                if kw.value.id in params:
                    self.trace.entity_bindings.append((kw.arg, kw.value.id))
        # Positional args that match param names
        for arg in call_node.args:
            if isinstance(arg, ast.Name) and arg.id in params:
                self.trace.entity_bindings.append(("_positional", arg.id))

    def _extract_return_names(self, node: ast.expr) -> None:
        """Extract variable names from return value."""
        if isinstance(node, ast.Name):
            self.trace.return_names.add(node.id)
            # Check if return var is an instantiated entity
            if node.id in self._local_vars:
                self.trace.return_has_entity_ref = True
        elif isinstance(node, ast.Dict):
            for val in node.values:
                if val:
                    self._extract_return_names(val)
            # Check if dict references entity vars
            for val in node.values:
                if isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name):
                    if val.value.id in self._local_vars:
                        self.trace.return_has_entity_ref = True
                elif isinstance(val, ast.Name) and val.id in self._local_vars:
                    self.trace.return_has_entity_ref = True
        elif isinstance(node, ast.Call):
            call_name = self._extract_call_name(node)
            if call_name:
                self.trace.return_names.add(call_name)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                self.trace.return_names.add(node.value.id)
                if node.value.id in self._local_vars:
                    self.trace.return_has_entity_ref = True

    def _body_has_raise(self, body: list[ast.stmt]) -> bool:
        for stmt in body:
            if isinstance(stmt, ast.Raise):
                return True
            if isinstance(stmt, ast.If):
                if self._body_has_raise(stmt.body):
                    return True
        return False


def extract_trace_from_files(
    file_paths: list[str],
    repo_root: str = ".",
) -> SemanticTrace:
    """Build SemanticTrace from Python source files using AST."""
    trace = SemanticTrace()

    for path in file_paths:
        abs_path = path if os.path.isabs(path) else os.path.join(repo_root, path)
        if LanguageId.from_path(abs_path) is not LanguageId.PYTHON or not os.path.isfile(abs_path):
            continue

        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception:
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        # Collect all class names for instantiation detection
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                trace.all_classes.add(node.name)

        # Process each function
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _process_function(node, abs_path, trace)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _process_function(item, abs_path, trace)

    logger.info(
        "[TRACER] %d functions traced, %d calls, %d instantiations, %d persists",
        len(trace.function_traces), len(trace.all_calls),
        len(trace.all_instantiations), len(trace.all_persist_calls),
    )
    return trace


def extract_trace_cross_file(
    file_paths: list,
    repo_root: str = ".",
    resolver: Any = None,
    max_depth: int = 2,
) -> "SemanticTrace":
    """Build SemanticTrace with cross-file trace merging.

    When a function calls another function in a different file,
    the callee's trace (persist_calls, instantiations, entity_bindings)
    is merged into the caller's trace. This allows contract evaluation
    to see through delegation patterns like:
        route.create_user() -> service.create_user() -> db.add()

    Args:
        file_paths: Files to analyze.
        repo_root: Repository root.
        resolver: CrossFileFlowResolver instance (builds import graph).
        max_depth: Maximum call chain depth for merging (default 2).
    """
    # Step 1: Build base trace (single-file analysis)
    trace = extract_trace_from_files(file_paths, repo_root)

    if not resolver or max_depth <= 0:
        return trace

    # Step 2: Build resolver graph if not already built
    try:
        if not getattr(resolver, '_built', False):
            resolver.build(file_paths)
    except Exception:
        return trace

    # Step 3: For each function, merge callee traces
    for _func_name, ft in list(trace.function_traces.items()):
        _merge_callee_traces(ft, trace, resolver, max_depth, set())

    # Update aggregate sets
    for ft in trace.function_traces.values():
        trace.all_calls.update(ft.calls)
        trace.all_instantiations.update(ft.instantiations)
        trace.all_persist_calls.update(ft.persist_calls)

    logger.info(
        "[TRACER_CROSS] merged cross-file traces for %d functions",
        len(trace.function_traces),
    )
    return trace


def _merge_callee_traces(
    caller_ft: "FunctionTrace",
    trace: "SemanticTrace",
    resolver: Any,
    remaining_depth: int,
    visited: set,
) -> None:
    """Recursively merge callee FunctionTraces into caller.

    For each call in the caller, if the callee has a FunctionTrace,
    merge its persist_calls, instantiations, and entity_bindings into
    the caller. This propagates deep behaviors upward.
    """
    if remaining_depth <= 0:
        return

    visit_key = (caller_ft.name, caller_ft.file_path)
    if visit_key in visited:
        return
    visited.add(visit_key)

    for call_name in list(caller_ft.calls):
        # Extract bare function name from attribute calls (e.g., "service.create_user" -> "create_user")
        bare_name = call_name.split(".")[-1] if "." in call_name else call_name

        # Find callee trace
        callee_ft = trace.function_traces.get(bare_name)
        if not callee_ft:
            # Try via resolver
            callee_ft = _find_callee_trace_via_resolver(
                bare_name, caller_ft, trace, resolver,
            )

        if callee_ft and callee_ft.name != caller_ft.name:
            # Recursively merge the callee first (depth-1)
            _merge_callee_traces(callee_ft, trace, resolver, remaining_depth - 1, visited)

            # Merge callee's behavioral data into caller
            caller_ft.persist_calls.update(callee_ft.persist_calls)
            caller_ft.instantiations.update(callee_ft.instantiations)
            caller_ft.entity_bindings.extend(callee_ft.entity_bindings)
            # If callee has return entity ref, propagate
            if callee_ft.return_has_entity_ref:
                caller_ft.return_has_entity_ref = True
            # Propagate error branch info
            if callee_ft.has_error_branch:
                caller_ft.has_error_branch = True


def _find_callee_trace_via_resolver(
    func_name: str,
    caller_ft: "FunctionTrace",
    trace: "SemanticTrace",
    resolver: Any,
) -> Optional["FunctionTrace"]:
    """Try to find a callee's FunctionTrace using the cross-file resolver."""
    try:
        graph = resolver.graph
        # Look up the function in the resolver's function_files mapping
        for key, _file_path in graph.function_files.items():
            if ":" in key and key.endswith(f":{func_name}"):
                # Found it — check if we have its trace
                if func_name in trace.function_traces:
                    return trace.function_traces[func_name]
    except Exception:
        pass
    return None


def _process_function(
    node: ast.FunctionDef, file_path: str, trace: SemanticTrace,
) -> None:
    """Process a single function definition."""
    func_name = node.name
    trace.all_functions.add(func_name)

    # Extract parameter names
    param_names = []
    for arg in node.args.args:
        if arg.arg != "self":
            param_names.append(arg.arg)

    visitor = _FunctionVisitor(func_name, param_names, trace.all_classes)
    visitor.trace.file_path = file_path
    visitor.visit(node)

    ft = visitor.trace
    trace.function_traces[func_name] = ft
    trace.all_calls.update(ft.calls)
    trace.all_instantiations.update(ft.instantiations)
    trace.all_persist_calls.update(ft.persist_calls)
