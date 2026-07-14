"""virtual_graph.py — Build a virtual graph from plan CREATE_FILE operations.

When plans create new files (new project or adding modules), the real graph
is empty because those files don't exist on disk yet.  This module parses
the **planned** code content (or infers structure from intent/path heuristics)
to produce virtual SymbolNodes, ImportEdges, and CallEdges that are merged
into the real RepositoryGraph so that downstream graph features work:

- _graph_topo_sort() sees import edges between new files
- check_graph_execution_contracts() finds symbols in route files
- expand_plan_with_dependencies() detects missing imports
- build_graph_context_for_planner() shows virtual structure
"""
from __future__ import annotations

import ast
import enum
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ..languages import LanguageId
from .models import ImportEdge, SymbolNode
from .repository_graph import CallEdge as RepoCallEdge
from .repository_graph import RepositoryGraph


class FileRole(str, enum.Enum):
    """Architectural roles for file classification.

    Each role represents a conventional architectural layer.
    Used as a typed key in role-to-metadata mappings.
    """
    ROUTE = 'route'
    MODEL = 'model'
    DATABASE = 'database'
    SERVICE = 'service'
    AUTH = 'auth'
    CONFIG = 'config'
    MAIN = 'main'
    TEST = 'test'
    MIDDLEWARE = 'middleware'
    UTIL = 'util'
    UNKNOWN = 'unknown'
_AST_ROLE_PATTERNS: list[tuple[FileRole, tuple[str, str, str], ...]] = [(FileRole.ROUTE, (('Import', 'APIRouter', ''), ('Import', 'Blueprint', ''), ('Call', 'add_url_rule', ''), ('Call', 'route', ''))), (FileRole.MODEL, (('Import', 'Model', 'sqlalchemy'), ('Import', 'Base', 'sqlalchemy'), ('Class', 'Model', ''), ('Import', 'models', ''))), (FileRole.DATABASE, (('Import', 'Session', ''), ('Import', 'engine', 'sqlalchemy'), ('Import', 'connection', ''), ('Import', 'migrate', ''))), (FileRole.AUTH, (('Import', 'jwt', ''), ('Import', 'oauth', ''), ('Import', 'login', ''), ('Import', 'permission', ''), ('Import', 'authenticate', ''))), (FileRole.SERVICE, (('Import', 'Service', ''),)), (FileRole.CONFIG, (('Import', 'config', ''), ('Import', 'settings', ''), ('Import', 'env', ''))), (FileRole.MAIN, (('Call', 'create_app', ''), ('Call', 'run', ''), ('Call', 'uvicorn.run', ''))), (FileRole.TEST, (('Import', 'pytest', ''), ('Import', 'unittest', ''), ('Import', 'TestCase', ''), ('Class', 'TestCase', ''), ('Import', 'TestClient', ''))), (FileRole.MIDDLEWARE, (('Import', 'middleware', ''), ('Class', 'Middleware', ''), ('Call', 'add_middleware', ''))), (FileRole.UTIL, (('Import', 'decorator', ''), ('Import', 'functools', ''), ('Import', 'typing', ''), ('Class', 'Mixin', '')))]

def _match_role_by_ast(content: str) -> str:
    """Classify file role using AST structural signals from actual code content.

    Parses *content* with ``ast.parse`` and checks for import/call/class nodes
    matching known architectural patterns. Returns the best-matching role, or
    ``'unknown'`` if no structural signal is found.

    This is a pure structural-query approach: no keywords, no regex.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 'unknown'
    role_scores: dict[str, int] = defaultdict(int)
    for role, patterns_tuple in _AST_ROLE_PATTERNS:
        for node_type, attr, _val in patterns_tuple:
            _matched = False
            for node in ast.walk(tree):
                if node_type == 'Import':
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in getattr(node, 'names', []):
                            _name = getattr(alias, 'name', '') or ''
                            if attr.lower() in _name.lower():
                                _matched = True
                                break
                elif node_type == 'Call':
                    if isinstance(node, ast.Call):
                        _func = getattr(node.func, 'attr', '') or getattr(node.func, 'id', '') or ''
                        if attr.lower() in _func.lower():
                            _matched = True
                elif node_type == 'Class':
                    if isinstance(node, ast.ClassDef):
                        _bases = [getattr(b, 'id', '') or getattr(b, 'attr', '') for b in node.bases]
                        if attr in node.name or attr in _bases:
                            _matched = True
                if _matched:
                    role_scores[role.value] += 1
                    break
    if role_scores:
        return max(role_scores, key=lambda r: role_scores[r])
    return 'unknown'

@dataclass
class RolePattern:
    """Typed pattern for file role classification.

    Replaces the ad-hoc ``_ROLE_KEYWORDS`` dict with a structured dataclass.
    Each pattern combines:

    - **Keyword signals**: found in filename stems, parent directory names,
      or intent text (conventional architectural terms, no regex).
    - **Suffix signals**: conventional file suffixes (e.g., ``_test.go``, ``.spec.ts``).
    - **AST patterns**: structural code patterns (imports, calls, class definitions).
    """
    role: FileRole
    keywords: set[str] = field(default_factory=set)
    suffixes: list[str] = field(default_factory=list)
    priority: int = 100
logger = logging.getLogger('asicode.graph.virtual')

def enrich_facade_with_plan(facade: Any, plan: Any) -> Any:
    """Enrich an existing RepositoryGraphFacade with virtual symbols from plan.

    Mutates the facade's internal graph by injecting virtual data extracted
    from CREATE_FILE operations.  Returns the same facade instance.
    """
    from ..agent.operation_models import OperationKind
    create_ops = [op for op in plan.operations or [] if getattr(op, 'kind', None) == OperationKind.CREATE_FILE and getattr(op, 'path', None)]
    if not create_ops:
        return facade
    repo_root = getattr(facade, 'repo_root', '') or os.getcwd()
    graph = facade._ensure_graph()
    injected_files = 0
    for op in create_ops:
        rel_path = _normalize_path(op.path, repo_root)
        if graph.file_symbols.get(rel_path):
            continue
        content = _get_content(op)
        if content and LanguageId.from_path(rel_path) is LanguageId.PYTHON:
            if _inject_from_python_content(graph, rel_path, content, repo_root):
                injected_files += 1
                continue
        if _inject_from_heuristics(graph, op, rel_path, repo_root):
            injected_files += 1
    if injected_files:
        logger.info('Virtual graph: injected %d file(s) into graph', injected_files)
    _infer_cross_file_imports(graph, create_ops, repo_root)
    _infer_layer_call_edges(graph, create_ops, repo_root)
    return facade

def _get_content(op: Any) -> Optional[str]:
    """Extract code content from operation metadata."""
    md = getattr(op, 'metadata', None) or {}
    content = md.get('content') or md.get('patch_segment') or ''
    if content and (not content.strip().startswith('---')):
        return content
    return None

def _normalize_path(path: str, repo_root: str) -> str:
    """Normalize operation path to relative path."""
    if os.path.isabs(path):
        try:
            return os.path.relpath(path, repo_root)
        except ValueError:
            return path
    return path

def _inject_from_python_content(graph: RepositoryGraph, rel_path: str, content: str, repo_root: str) -> bool:
    """Parse Python content and inject symbols/imports/calls into graph."""
    try:
        tree = ast.parse(content, filename=rel_path)
    except SyntaxError:
        return False
    from .repository_graph import GraphVisitor
    visitor = GraphVisitor(rel_path, repo_root)
    visitor.visit(tree)
    if not visitor.symbols and (not visitor.imports):
        return False
    for symbol in visitor.symbols:
        unique_id = f'{rel_path}:{symbol.qualname}'
        if unique_id not in graph.symbols:
            graph.symbols[unique_id] = symbol
            graph.file_symbols[rel_path].append(unique_id)
            graph._symbol_locations[symbol.qualname, rel_path] = unique_id
    for call in visitor.calls:
        graph.call_edges.append(call)
    for imp in visitor.imports:
        graph.import_edges.append(imp)
    logger.debug('Virtual graph: parsed %s → %d symbols, %d calls, %d imports', rel_path, len(visitor.symbols), len(visitor.calls), len(visitor.imports))
    return True
_ROLE_PATTERNS: list[RolePattern] = [RolePattern(role=FileRole.ROUTE, keywords={'route', 'router', 'api', 'endpoint', 'view'}, priority=10), RolePattern(role=FileRole.MODEL, keywords={'model', 'schema', 'entity'}, priority=20), RolePattern(role=FileRole.DATABASE, keywords={'database', 'db', 'session', 'connection'}, priority=30), RolePattern(role=FileRole.SERVICE, keywords={'service', 'business', 'logic', 'handler'}, priority=40), RolePattern(role=FileRole.AUTH, keywords={'auth', 'login', 'security', 'permission'}, priority=50), RolePattern(role=FileRole.CONFIG, keywords={'config', 'settings', 'env'}, priority=60), RolePattern(role=FileRole.MAIN, keywords={'main', 'app', 'server', 'run'}, priority=70), RolePattern(role=FileRole.TEST, keywords={'test'}, suffixes=['_test', '.spec', '.test.'], priority=5), RolePattern(role=FileRole.MIDDLEWARE, keywords={'middleware'}, priority=80), RolePattern(role=FileRole.UTIL, keywords={'util', 'helper', 'common'}, priority=90)]
_ROLE_DEFAULT_SYMBOLS: dict[FileRole, list[str]] = {FileRole.ROUTE: ['register_routes', 'router'], FileRole.MODEL: ['Base', 'Model'], FileRole.DATABASE: ['get_db', 'SessionLocal', 'Base', 'engine'], FileRole.SERVICE: ['Service'], FileRole.AUTH: ['authenticate', 'verify_token', 'get_current_user'], FileRole.CONFIG: ['Settings', 'settings'], FileRole.MAIN: ['create_app', 'app'], FileRole.TEST: ['test_main'], FileRole.MIDDLEWARE: ['Middleware'], FileRole.UTIL: ['helper']}
_GENERIC_PARENTS: set[str] = {'app', 'src', 'lib', 'pkg', 'core', 'api', 'server', 'backend'}

def _match_role_keywords(text: str, skip_roles: set[str]=frozenset()) -> str:
    """Match text against role patterns, return the best-matching role or 'unknown'.

    Uses typed ``_ROLE_PATTERNS`` (``RolePattern`` dataclass instances) instead
    of the old ad-hoc ``_ROLE_KEYWORDS`` dict.  Returns the role with the most
    keyword matches in *text*. Ties are broken by pattern priority (lower =
    higher priority).

    Also checks ``suffixes`` when keyword count is 0 — e.g. ``_test.go``,
    ``.spec.ts`` patterns are matched against *text*.
    """
    best_role = 'unknown'
    best_count = 0
    best_priority = 999
    for pat in _ROLE_PATTERNS:
        role_str = pat.role.value
        if role_str in skip_roles:
            continue
        count = sum(1 for kw in pat.keywords if kw in text)
        if count == 0 and pat.suffixes:
            count = sum(1 for sfx in pat.suffixes if sfx in text)
        if count > 0 and (count > best_count or (count == best_count and pat.priority < best_priority)):
            best_count = count
            best_role = role_str
            best_priority = pat.priority
    return best_role

def _classify_file_role_from_imports(rel_path: str, graph: Optional[RepositoryGraph]) -> str:
    """Infer file role using actual import graph evidence.

    Checks which roles the file imports from and which roles import this file,
    then scores candidate roles using the conventional import flow table.
    Returns 'unknown' when graph has no relevant import edges.
    """
    if not graph or not hasattr(graph, 'import_edges') or (not graph.import_edges):
        return 'unknown'
    imported_role_counts: dict[str, int] = defaultdict(int)
    importing_role_counts: dict[str, int] = defaultdict(int)
    for edge in graph.import_edges:
        importer_path = getattr(edge, 'importer', '') or ''
        imported_path = getattr(edge, 'imported', '') or ''
        if importer_path == rel_path:
            seg = imported_path.split('.')[0].lower() if '.' in imported_path else imported_path.lower()
            r = _match_role_keywords(seg)
            if r != 'unknown':
                imported_role_counts[r] += 1
        if imported_path == rel_path or rel_path in imported_path:
            seg = importer_path.rsplit('/', 1)[-1].rsplit('.', 1)[0].lower()
            r = _match_role_keywords(seg)
            if r != 'unknown':
                importing_role_counts[r] += 1
    if not imported_role_counts and (not importing_role_counts):
        return 'unknown'
    _role_upstream: dict[str, set[str]] = defaultdict(set)
    for role, deps in _ROLE_IMPORT_FLOW.items():
        for dep in deps:
            _role_upstream[dep].add(role)
    scores: dict[str, int] = defaultdict(int)
    for imp_role, count in imported_role_counts.items():
        for candidate_role, deps in _ROLE_IMPORT_FLOW.items():
            if imp_role in deps:
                scores[candidate_role] += count * 2
    for importer_role, count in importing_role_counts.items():
        for dep_role, upstreams in _role_upstream.items():
            if importer_role in upstreams:
                scores[dep_role] += count
    if not scores:
        return 'unknown'
    best_role = max(scores, key=lambda r: scores[r])
    return best_role

def _classify_file_role(path: str, intent: str='', graph: Optional[RepositoryGraph]=None, content: Optional[str]=None) -> str:
    """Classify a file's architectural role.

    Priority:
      0. AST structural signals from *content* (when available — most reliable)
      1. Import graph evidence (for files with existing import edges)
      2. Parent directory name (path heuristic, no keywords in code)
      3. Filename stem match (path heuristic)
      4. Intent text match (fallback)

    Uses ``_match_role_by_ast(content)`` for Priority\u202f0 — pure structural
    queries via ``ast.parse`` with no keyword/regex dependency.  This
    replaces the old keyword-set-based approach for all cases where code
    content is available.
    """
    if content is not None:
        ast_role = _match_role_by_ast(content)
        if ast_role != 'unknown':
            return ast_role
    if graph is not None:
        role = _classify_file_role_from_imports(path, graph)
        if role != 'unknown':
            return role
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    parent = os.path.basename(os.path.dirname(path)).lower() if os.path.dirname(path) else ''
    if parent and parent not in _GENERIC_PARENTS:
        role = _match_role_keywords(parent)
        if role != 'unknown':
            return role
    role = _match_role_keywords(stem)
    if role != 'unknown':
        return role
    intent_lower = (intent or '').lower()
    return _match_role_keywords(intent_lower)
_ROLE_IMPORT_FLOW: dict[str, list[str]] = {'main': ['route', 'database', 'config', 'middleware', 'service'], 'route': ['service', 'model', 'auth', 'database'], 'service': ['model', 'database'], 'auth': ['model', 'database'], 'model': ['database']}
_ROLE_CALL_FLOW: dict[str, list[str]] = {'route': ['service', 'auth', 'database'], 'service': ['model', 'database'], 'auth': ['database']}

def _extract_symbols_from_intent(intent: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    if not intent:
        return results
    tokens = []
    current = []
    for ch in intent:
        if ch.isalnum() or ch == '_':
            current.append(ch)
        elif current:
            tokens.append(''.join(current))
            current = []
    if current:
        tokens.append(''.join(current))
    for name in tokens:
        if name[0].isupper() and name.isidentifier():
            if name in {'Create', 'User', 'Login', 'Signup', 'Delete', 'Update', 'API', 'REST', 'CRUD', 'HTTP', 'JSON', 'SQL', 'ORM', 'Flask', 'FastAPI', 'Django', 'Express', 'SQLAlchemy', 'True', 'False', 'None', 'The', 'This', 'Use'}:
                if name in {'User', 'Login', 'Session', 'Token', 'Cart', 'Item', 'Product', 'Order', 'Message', 'Chat', 'Video', 'Comment'}:
                    results.append((name, 'class'))
                continue
            results.append((name, 'class'))
    _SKIP_SNAKE = {'create_file', 'modify_symbol', 'read_symbol', 'insert_after', 'hashed_password', 'created_at', 'updated_at', 'is_active', 'access_token', 'refresh_token', 'token_type', 'file_path', 'user_id', 'item_id', 'order_id', 'video_id', 'chat_id', 'message_id', 'product_id', 'session_id', 'primary_key', 'foreign_key', 'not_null', 'auto_increment'}
    for name in tokens:
        if name[0].islower() and '_' in name and name.isidentifier() and (name not in _SKIP_SNAKE):
            results.append((name, 'function'))
    return results

def _inject_from_heuristics(graph: RepositoryGraph, op: Any, rel_path: str, repo_root: str) -> bool:
    """Inject virtual symbols based on intent text + path conventions."""
    intent = getattr(op, 'intent', '') or ''
    symbol_name = getattr(op, 'symbol', '') or ''
    role = _classify_file_role(rel_path, intent)
    symbols_to_add: list[tuple[str, str]] = []
    if symbol_name:
        kind = 'class' if symbol_name[0].isupper() else 'function'
        symbols_to_add.append((symbol_name, kind))
    symbols_to_add.extend(_extract_symbols_from_intent(intent))
    if not symbols_to_add:
        try:
            role_key = FileRole(role) if role != 'unknown' else None
        except ValueError:
            role_key = None
        defaults = _ROLE_DEFAULT_SYMBOLS.get(role_key, []) if role_key else []
        for dname in defaults:
            kind = 'class' if dname[0].isupper() else 'function'
            symbols_to_add.append((dname, kind))
    if not symbols_to_add:
        return False
    module = _path_to_module(rel_path)
    seen: set[str] = set()
    for name, kind in symbols_to_add:
        if name in seen:
            continue
        seen.add(name)
        unique_id = f'{rel_path}:{name}'
        if unique_id in graph.symbols:
            continue
        node = SymbolNode(name=name, qualname=name, module=module, file_path=rel_path, kind=kind, start_line=1, end_line=10)
        graph.symbols[unique_id] = node
        graph.file_symbols[rel_path].append(unique_id)
        graph._symbol_locations[name, rel_path] = unique_id
    logger.debug('Virtual graph: heuristic %s (role=%s) → %d symbols', rel_path, role, len(seen))
    return True

def _infer_cross_file_imports(graph: RepositoryGraph, create_ops: list, repo_root: str) -> None:
    """Infer import edges between new files based on role conventions.

    E.g., route files import from model/service files.
    """
    role_files: dict[str, list[str]] = defaultdict(list)
    for op in create_ops:
        rel_path = _normalize_path(op.path, repo_root)
        intent = getattr(op, 'intent', '') or ''
        content = _get_content(op)
        role = _classify_file_role(rel_path, intent, graph, content=content)
        role_files[role].append(rel_path)
    for op in create_ops:
        rel_path = _normalize_path(op.path, repo_root)
        intent = (getattr(op, 'intent', '') or '').lower()
        content = _get_content(op)
        role = _classify_file_role(rel_path, intent, graph, content=content)
        target_roles = _ROLE_IMPORT_FLOW.get(role, [])
        for target_role in target_roles:
            for target_path in role_files.get(target_role, []):
                if target_path == rel_path:
                    continue
                exists = any(e.importer == rel_path and e.imported == _path_to_module(target_path) for e in graph.import_edges)
                if not exists:
                    module = _path_to_module(target_path)
                    graph.import_edges.append(ImportEdge(importer=rel_path, imported=module, import_type='from'))
        for other_op in create_ops:
            other_path = _normalize_path(other_op.path, repo_root)
            if other_path == rel_path:
                continue
            other_name = os.path.splitext(os.path.basename(other_path))[0]
            if other_name and other_name in intent:
                exists = any(e.importer == rel_path and e.imported == _path_to_module(other_path) for e in graph.import_edges)
                if not exists:
                    graph.import_edges.append(ImportEdge(importer=rel_path, imported=_path_to_module(other_path), import_type='from'))
    added = sum(1 for _ in graph.import_edges)
    logger.debug('Virtual graph: cross-file import inference complete (%d total edges)', added)

def _infer_layer_call_edges(graph: RepositoryGraph, create_ops: list, repo_root: str) -> None:
    """Infer call edges between architectural layers.

    Critical for check_graph_execution_contracts() BFS:
    route handler → service function → db operation (persistence).
    """
    role_symbols: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for op in create_ops:
        rel_path = _normalize_path(op.path, repo_root)
        intent = getattr(op, 'intent', '') or ''
        content = _get_content(op)
        role = _classify_file_role(rel_path, intent, graph, content=content)
        symbol_ids = graph.file_symbols.get(rel_path, [])
        for sid in symbol_ids:
            sym = graph.symbols.get(sid)
            if sym and sym.kind in ('function', 'method'):
                role_symbols[role].append((sym.qualname, rel_path))
    existing_calls = {(e.caller, e.callee) for e in graph.call_edges}
    for caller_role, callee_roles in _ROLE_CALL_FLOW.items():
        for caller_name, caller_file in role_symbols.get(caller_role, []):
            for callee_role in callee_roles:
                for callee_name, _callee_file in role_symbols.get(callee_role, []):
                    if (caller_name, callee_name) not in existing_calls:
                        graph.call_edges.append(RepoCallEdge(caller=caller_name, callee=callee_name, file_path=caller_file, line=1))
                        existing_calls.add((caller_name, callee_name))

def _path_to_module(rel_path: str) -> str:
    """Convert relative file path to Python module name."""
    p = rel_path
    if LanguageId.from_path(p) is LanguageId.PYTHON:
        p = p[:-3]
    if p.endswith('/__init__'):
        p = p[:-9]
    module = p.replace('/', '.').replace('\\', '.')
    if module.startswith('.'):
        module = module[1:]
    return module
