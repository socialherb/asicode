"""AST utilities extracted from OperationExecutor for reuse and testability.

All functions in this module are pure — they accept values and return values
without depending on OperationExecutor state.  This makes them testable and
safe to use from any executor-adjacent module.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tree-sitter availability ─────────────────────────────────────────────
try:
    from ..languages.tree_sitter_utils import (
        _CALL_QUERIES as _TS_CALL_QUERIES,
    )
    from ..languages.tree_sitter_utils import (
        _IMPORT_QUERIES as _TS_IMPORT_QUERIES,
    )
    from ..languages.tree_sitter_utils import (
        find_symbol_range as _ts_find_symbol_range,
    )
    from ..languages.tree_sitter_utils import (
        get_node_text as _ts_get_node_text,
    )
    from ..languages.tree_sitter_utils import (  # type: ignore
        parse_to_tree as _ts_parse_to_tree,
    )
    from ..languages.tree_sitter_utils import (
        query_matches as _ts_query_matches,
    )
    _HAS_TS = True
except ImportError:
    _HAS_TS = False


# ── Constants ────────────────────────────────────────────────────────────

_AST_NODE_DELETE_KINDS: frozenset = frozenset({"If", "While", "Assert"})

# How many lines to search around anchor_lineno when exact match fails.
# Prior ops on the same file shift line numbers; a small radius recovers most cases.
_AST_ANCHOR_SEARCH_RADIUS: int = 5

_AST_KIND_MAP: dict = {"If": ast.If, "While": ast.While, "Assert": ast.Assert}


# ── AST parsing helpers ──────────────────────────────────────────────────


def ast_parse_result(result: str) -> tuple[Optional[str], str]:
    """Return (result, 'ok') if parseable, else (None, 'syntax_broken')."""
    try:
        ast.parse(result)
    except SyntaxError:
        return None, "syntax_broken"
    return result, "ok"


# ── Symbol / definition helpers ──────────────────────────────────────────


def extract_ast_call_names(source: str, bare_name: str) -> set:
    """Return method names called by *bare_name* inside *source*.

    Uses tree-sitter call queries (Python) with AST fallback.
    Finds self.X() and X() style calls that reference class methods.
    Falls back gracefully on parse failure.
    """
    called: set = set()

    # ── Primary: tree-sitter ────────────────────────────
    if _HAS_TS:
        try:
            tree = _ts_parse_to_tree(source, "python")
            if tree is not None:
                source.encode("utf-8")
                # Find function_definition matching bare_name
                fn_query = f"""
(function_definition name: (identifier) @name
  (#eq? @name "{bare_name}"))
"""
                matches = _ts_query_matches(source, "python", fn_query)
                if not matches:
                    return called
                # Get the matched function node
                fn_node = matches[0].get("name", [None])[0]
                if fn_node is None:
                    return called

                # Now find calls inside that function's body using call query
                call_query = _TS_CALL_QUERIES.get("python", "")
                if call_query:
                    call_matches = _ts_query_matches(source, "python", call_query)
                    for match in call_matches:
                        callee_caps = match.get("callee", [])
                        for cap in callee_caps:
                            called.add(cap.text)
                return called
        except Exception:
            pass  # fall through to AST

    # ── Fallback: AST ───────────────────────────────────
    try:
        tree = ast.parse(source)
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if func.name != bare_name:
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Attribute):
                        called.add(fn.attr)
                    elif isinstance(fn, ast.Name):
                        called.add(fn.id)
    except SyntaxError:
        pass
    return called


def find_symbol_line_range_for_delete(src: str, symbol: str) -> Optional[tuple[int, int]]:
    """Find (start_line, end_line) of *symbol* in *src*.

    Uses tree-sitter (Python) with AST fallback.
    Handles functions, classes, imports, constants (Assign/AnnAssign),
    and dotted names (class.method).
    Returns None if the symbol cannot be found or the source is unparseable.
    All line numbers are 1-based.
    """
    # ── Primary: tree-sitter ────────────────────────────
    if _HAS_TS:
        try:
            result = _ts_find_symbol_range(src, symbol, "python")
            if result:
                return result
        except Exception:
            pass  # fall through to AST

    # ── Fallback: AST ───────────────────────────────────
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    parts = symbol.split(".", 1)

    # Pass 1: function/class/method definitions.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if len(parts) == 1 and node.name == symbol:
                _start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                return (_start, node.end_lineno or node.lineno)
            if len(parts) == 2 and node.name == parts[0]:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == parts[1]:
                        _start = child.decorator_list[0].lineno if child.decorator_list else child.lineno
                        return (_start, child.end_lineno or child.lineno)

    # Pass 2: import statements.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                match = (
                    alias.name == symbol or alias.asname == symbol
                    or (len(parts) == 1 and (alias.name == parts[0] or alias.asname == parts[0]))
                )
                if match:
                    return (node.lineno, node.end_lineno or node.lineno)

    # Pass 3: module-level/class-level constants
    if len(parts) == 1:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol:
                        return (node.lineno, node.end_lineno or node.lineno)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == symbol:
                    return (node.lineno, node.end_lineno or node.lineno)

    return None


def get_symbol_start_line_ast(name: str, file_path: str) -> Optional[int]:
    """Lookup a definition symbol's 1-indexed start line (tree-sitter + AST fallback).

    Returns the line number of the class/function/async-function definition
    matching *name*, or None if not found or unparseable.
    """
    # ── Primary: tree-sitter ────────────────────────────
    if _HAS_TS:
        try:
            if not os.path.isfile(file_path):
                return None
            with open(file_path, encoding="utf-8", errors="replace") as _fh:
                _src = _fh.read()
            result = _ts_find_symbol_range(_src, name, "python")
            if result:
                return result[0]
        except Exception:
            pass  # fall through to AST

    # ── Fallback: AST ───────────────────────────────────
    try:
        if not os.path.isfile(file_path):
            return None
        with open(file_path, encoding="utf-8", errors="replace") as _fh:
            _src = _fh.read()
        _tree = ast.parse(_src)
        _bare = name.split(".")[-1] if "." in name else name
        for _node in ast.walk(_tree):
            if isinstance(_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if _node.name == _bare:
                    _ln = _node.decorator_list[0].lineno if _node.decorator_list else getattr(_node, 'lineno', None)
                    if _ln is not None:
                        return _ln
    except Exception:
        logger.debug("[except_pass] get_symbol_start_line_ast: ast.parse/walk failed", exc_info=True)
    return None


def class_declared_fields(cls: ast.ClassDef) -> set[str]:
    """Return fields assigned or annotated directly in a class body."""
    fields: set[str] = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if isinstance(target, ast.Name):
                fields.add(target.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    fields.add(target.id)
    return fields


def resolve_imported_facade_file(
    source_file: str, facade_symbol: str, repo_root: str = ""
) -> str:
    """Return the Python file that defines an imported facade symbol, if direct.

    Uses tree-sitter with AST fallback.
    """
    if not source_file or not facade_symbol or not os.path.isfile(source_file):
        return ""

    # ── Primary: tree-sitter ────────────────────────────
    if _HAS_TS:
        try:
            with open(source_file, encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            query = _TS_IMPORT_QUERIES.get("python", "")
            if query:
                matches = _ts_query_matches(content, "python", query)
                code_bytes = content.encode("utf-8")
                source_dir = os.path.dirname(source_file)
                for match_group in matches:
                    source_caps = match_group.get("source", [])
                    for sc in source_caps:
                        module = sc.text
                        # Check if this import matches facade_symbol
                        import_caps = match_group.get("import", [])
                        for ic in import_caps:
                            import_text = code_bytes[ic.start_byte:ic.end_byte].decode("utf-8")
                            if facade_symbol in import_text:
                                # Resolve module path
                                if module.startswith("."):
                                    level = len(module) - len(module.lstrip("."))
                                    rel_base = source_dir
                                    for _ in range(max(level - 1, 0)):
                                        rel_base = os.path.dirname(rel_base)
                                    rel_parts = module.lstrip(".").split(".")
                                    candidate = os.path.join(rel_base, *rel_parts) + ".py"
                                else:
                                    candidate = os.path.join(repo_root, *module.split(".")) + ".py" if repo_root and module else ""
                                if candidate and os.path.isfile(candidate):
                                    return candidate
        except Exception:
            pass  # fall through to AST

    # ── Fallback: AST ───────────────────────────────────
    try:
        with open(source_file, encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return ""
    source_dir = os.path.dirname(source_file)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not any(alias.asname == facade_symbol or alias.name == facade_symbol for alias in node.names):
            continue
        module = node.module or ""
        if node.level:
            rel_base = source_dir
            for _ in range(max(node.level - 1, 0)):
                rel_base = os.path.dirname(rel_base)
            rel_parts = module.split(".") if module else []
            candidate = os.path.join(rel_base, *rel_parts) + ".py"
        else:
            candidate = os.path.join(repo_root, *module.split(".")) + ".py" if repo_root and module else ""
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


def _extract_def_name(node, code_bytes: bytes) -> Optional[str]:
    """Extract the definition name from a tree-sitter definition node."""
    # Standard: node has a "name" field
    name_node = node.child_by_field_name("name")
    if name_node:
        return code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
    # Python decorated_definition: unwrap
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition",
                              "async_function_definition"):
                return _extract_def_name(child, code_bytes)
    # lexical_declaration / variable_declaration: find variable_declarator
    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                nn = child.child_by_field_name("name")
                if nn:
                    return code_bytes[nn.start_byte:nn.end_byte].decode("utf-8")
    return None


# ── Type-definition import scanning ─────────────────────────────────────


def _ts_collect_type_def_imports(abs_path: str, repo_root: str, pkg_dir: str, _TYPE_BASE_NAMES: frozenset) -> list[str]:
    """Tree-sitter version: collect type-definition imports."""
    if not _HAS_TS:
        return []
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as _fh:
            src = _fh.read()
    except OSError:
        return []

    # ── Step 1: collect candidate import paths via tree-sitter ──
    tree = _ts_parse_to_tree(src, "python")
    if tree is None:
        return []
    candidates: list[str] = []
    source_bytes = src.encode("utf-8")

    def _add_candidate(module_text: str, is_relative: bool, level: int):
        """Resolve import to candidate .py path."""
        mod_path = module_text.replace(".", os.sep)
        if is_relative:
            candidate = os.path.normpath(
                os.path.join(pkg_dir, "../" * (level - 1), mod_path + ".py")
            )
        else:
            candidate = os.path.join(repo_root, mod_path + ".py")
        candidates.append(candidate)

    # Walk tree for import statements
    def _walk_imports(node):
        if node is None:
            return
        nt = node.type
        if nt == "import_statement":
            # import X, import X.Y, import X as Z
            for c in (node.children or []):
                if c.type == "dotted_name":
                    text = _ts_get_node_text(source_bytes, c)
                    _add_candidate(text, False, 0)
        elif nt == "import_from_statement":
            # from X import Y, from .X import Y
            module_node = node.child_by_field_name("module_name")
            rel_dots = 0
            for c in (node.children or []):
                if c.type == ".":
                    rel_dots += 1
            if module_node:
                text = _ts_get_node_text(source_bytes, module_node)
                is_rel = rel_dots > 0
                level = rel_dots or 1
                _add_candidate(text, is_rel, level)
            else:
                # Relative import like "from .. import X" (no module name)
                if rel_dots > 0:
                    _add_candidate("", True, rel_dots)
        for c in (node.children or []):
            _walk_imports(c)

    _walk_imports(tree.root_node)

    if not candidates:
        return []

    # ── Step 2: check each candidate for type-definition content ──
    _ts_query = """
(class_definition
  name: (identifier) @name
  body: (block
    (expression_statement (assignment
      left: (attribute attribute: (identifier) @deco_name) @deco_attr
      right: (call function: (identifier) @call_name))) ?)
  (decorator (identifier) @decorator_name)?
  (decorator (attribute attribute: (identifier) @deco_attr2))?
) @class
"""
    _deco_query = """
(call function: (identifier) @call_name) @call
"""

    result: list[str] = []
    seen: set = set()
    for cand_abs in candidates:
        cand_abs = os.path.normpath(cand_abs)
        if not os.path.isfile(cand_abs):
            continue
        try:
            rel = os.path.relpath(cand_abs, repo_root)
        except ValueError:
            continue
        if rel in seen or rel == os.path.relpath(abs_path, repo_root):
            continue
        seen.add(rel)
        try:
            with open(cand_abs, encoding="utf-8", errors="replace") as _cf:
                cand_src = _cf.read()
        except OSError:
            continue

        cand_tree = _ts_parse_to_tree(cand_src, "python")
        if cand_tree is None:
            continue

        has_type_def = False
        cand_bytes = cand_src.encode("utf-8")

        # Find all class_definition nodes and check their bases/decorators
        def _scan_class_defs(node):
            nonlocal has_type_def
            if has_type_def or node is None:
                return
            if node.type == "class_definition":
                # Check bases (parenthesized list after name)
                for c in (node.children or []):
                    if c.type == "argument_list":
                        for arg in (c.children or []):
                            if arg.type in ("identifier", "attribute"):
                                base_text = _ts_get_node_text(cand_bytes, arg)
                                base_name = base_text.split(".")[-1] if "." in base_text else base_text
                                if base_name in _TYPE_BASE_NAMES:
                                    has_type_def = True
                                    return
                # Check decorators
                for c in (node.children or []):
                    if c.type == "decorator":
                        dec_text = _ts_get_node_text(cand_bytes, c)
                        dec_name = dec_text.lstrip("@").strip()
                        # @dataclass, @dataclass()
                        if dec_name == "dataclass" or dec_name.startswith("dataclass("):
                            has_type_def = True
                            return
                        # @attr.dataclass, @attr.s, etc.
                        if "." in dec_name:
                            _, _, attr = dec_name.rpartition(".")
                            if attr == "dataclass" or attr == "s":
                                has_type_def = True
                                return
            for c in (node.children or []):
                _scan_class_defs(c)
                if has_type_def:
                    return

        _scan_class_defs(cand_tree.root_node)

        if has_type_def:
            result.append(rel)
        if len(result) >= 3:
            break

    return result


def collect_type_def_imports(file_path: str, repo_root: str) -> list[str]:
    """Return repo-relative paths (max 3) of same-package files defining Enum/dataclass/TypedDict.

    Uses tree-sitter (primary) with AST fallback.
    """
    if not file_path or not repo_root:
        return []
    abs_path = file_path if os.path.isabs(file_path) else os.path.join(repo_root, file_path)
    if not os.path.isfile(abs_path):
        return []

    pkg_dir = os.path.dirname(abs_path)

    _TYPE_BASE_NAMES = frozenset({
        "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
        "TypedDict", "NamedTuple",
    })

    # ── Primary: tree-sitter ──
    result = _ts_collect_type_def_imports(abs_path, repo_root, pkg_dir, _TYPE_BASE_NAMES)
    if result:
        return result

    # ── Fallback: AST ──
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as _fh:
            src = _fh.read()
        tree = ast.parse(src, filename=abs_path)
    except (OSError, SyntaxError, ValueError):
        return []

    candidates: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level and node.level > 0:
                rel_mod = node.module.replace(".", os.sep)
                candidate = os.path.normpath(
                    os.path.join(pkg_dir, "../" * (node.level - 1), rel_mod + ".py")
                )
                candidates.append(candidate)
            else:
                candidate = os.path.join(
                    repo_root, node.module.replace(".", os.sep) + ".py"
                )
                candidates.append(candidate)

    # Step 2: AST fallback
    result = []
    seen = set()
    for cand_abs in candidates:
        cand_abs = os.path.normpath(cand_abs)
        if not os.path.isfile(cand_abs):
            continue
        try:
            rel = os.path.relpath(cand_abs, repo_root)
        except ValueError:
            continue
        if rel in seen or rel == os.path.relpath(abs_path, repo_root):
            continue
        seen.add(rel)
        try:
            with open(cand_abs, encoding="utf-8", errors="replace") as _cf:
                cand_src = _cf.read()
            cand_tree = ast.parse(cand_src, filename=cand_abs)
        except (OSError, SyntaxError, ValueError):
            continue

        has_type_def = False
        for cnode in ast.walk(cand_tree):
            if not isinstance(cnode, ast.ClassDef):
                continue
            for base in cnode.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in _TYPE_BASE_NAMES:
                    has_type_def = True
                    break
            if not has_type_def:
                for dec in cnode.decorator_list:
                    dec_name = ""
                    if isinstance(dec, ast.Name):
                        dec_name = dec.id
                    elif isinstance(dec, ast.Attribute):
                        dec_name = dec.attr
                    elif isinstance(dec, ast.Call):
                        fn = dec.func
                        dec_name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
                    if dec_name == "dataclass":
                        has_type_def = True
                        break
            if has_type_def:
                break

        if has_type_def:
            result.append(rel)
        if len(result) >= 3:
            break

    return result


# ── Branch / statement manipulation ─────────────────────────────────────


def _ts_expand_to_branch_body_end(src: str, start_line: int) -> Optional[int]:
    """Tree-sitter version: return last line of branch body at *start_line*."""
    if not _HAS_TS:
        return None
    tree = _ts_parse_to_tree(src, "python")
    if tree is None:
        return None
    root = tree.root_node
    target = start_line - 1  # 1→0-indexed

    def _find_node(n):
        if n is None or n.start_point[0] > target or n.end_point[0] < target:
            return None
        if n.start_point[0] == target and n.type in ("if_statement", "while_statement"):
            return n
        for c in (n.children or []):
            r = _find_node(c)
            if r is not None:
                return r
        return None

    node = _find_node(root)
    if node is None:
        return None

    def _body_end(n) -> Optional[int]:
        """Return end line (1-indexed) of the body block in *n*."""
        for c in (n.children or []):
            if c.type == "block":
                stmts = [ch for ch in (c.children or []) if ch.type not in ("comment", "newline")]
                if not stmts:
                    return c.start_point[0] + 1
                return max(s.end_point[0] for s in stmts) + 1
        return None

    # For if_statement: find the branch whose condition starts on target line
    if node.type == "if_statement":
        # First branch is the "if" itself
        cond = node.child_by_field_name("condition")
        if cond and cond.start_point[0] == target:
            result = _body_end(node)
            if result is not None:
                return result
        # Check elif/else clauses
        for c in (node.children or []):
            if c.type == "elif_clause":
                c_cond = c.child_by_field_name("condition")
                if c_cond and c_cond.start_point[0] == target:
                    result = _body_end(c)
                    if result is not None:
                        return result
            elif c.type == "else_clause":
                if c.start_point[0] == target:
                    result = _body_end(c)
                    if result is not None:
                        return result
    elif node.type == "while_statement":
        result = _body_end(node)
        if result is not None:
            return result
    return None


def py_expand_to_branch_body_end(src: str, start_line: int) -> Optional[int]:
    """Return the last line of the LOCAL branch body of the If/While at *start_line*.

    Uses tree-sitter (primary) with AST fallback.

    "Local branch body" means the body stmts of that specific if/elif/while branch,
    NOT including subsequent elif/else clauses (those belong to separate branches).

    Used by DELETE_SYMBOL_RANGE to expand a single-line condition hint into the
    full statement that must be removed to avoid orphaned indented bodies.

    Returns the expanded end line, or None if not found / not applicable.
    All line numbers are 1-based (AST convention).
    """
    # ── Primary: tree-sitter ──
    result = _ts_expand_to_branch_body_end(src, start_line)
    if result is not None:
        return result

    # ── Fallback: AST ──
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != start_line:
            continue
        if not isinstance(node, (ast.If, ast.While)):
            continue
        body = list(node.body or [])
        if not body:
            return start_line
        # Max end_lineno across all body statements (excludes orelse/else)
        return max(getattr(stmt, "end_lineno", start_line) for stmt in body)

    return None


def node_condition_dump(node: ast.stmt) -> str:
    """Return ast.dump of the condition of an If/While/Assert node, or ''."""
    if isinstance(node, (ast.If, ast.While)):
        return ast.dump(node.test, annotate_fields=False)
    if isinstance(node, ast.Assert):
        return ast.dump(node.test, annotate_fields=False)
    return ""


def find_statement_node_by_anchor(
    tree: ast.Module,
    node_kind: str,
    anchor_lineno: int,
    node_fingerprint: str = "",
    search_radius: int = _AST_ANCHOR_SEARCH_RADIUS,
) -> Optional[ast.stmt]:
    """Find the best-matching ast.If/While/Assert node near *anchor_lineno*.

    Strategy:
    1. Collect all nodes of the target kind within anchor_lineno ± search_radius.
    2. If *node_fingerprint* is provided, prefer nodes whose condition dump matches.
    3. Among equally-ranked candidates, pick the one closest to anchor_lineno.

    Returns None if no candidate is found within the radius.
    """
    target_type = _AST_KIND_MAP.get(node_kind)
    if target_type is None:
        return None

    # Collect (distance, fingerprint_match, node)
    candidates: list = []
    for node in ast.walk(tree):
        if not isinstance(node, target_type):
            continue
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            continue
        dist = abs(lineno - anchor_lineno)
        if dist > search_radius:
            continue
        fp_match = (
            bool(node_fingerprint)
            and node_condition_dump(node) == node_fingerprint
        )
        candidates.append((dist, not fp_match, node))  # sort: dist ASC, fp_match DESC

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]  # type: ignore[return-value]


def rewrite_source_without_node(
    src: str,
    node: ast.stmt,
    node_kind: str,
) -> str:
    """Splice *node* out of *src* using its line boundaries.

    For ``If`` nodes: removes the condition line + LOCAL branch body only —
    subsequent elif/else clauses (in node.orelse) are preserved as-is, since
    they become the remaining valid branches of the enclosing if statement.

    For ``While`` / ``Assert``: removes the full node extent.
    """
    lines = src.splitlines(keepends=True)
    total = len(lines)
    start = node.lineno  # 1-based

    if node_kind == "If":
        body = list(node.body or [])
        end = max(
            (getattr(s, "end_lineno", start) for s in body),
            default=start,
        )
    else:
        end = getattr(node, "end_lineno", start)

    end = min(end, total)
    return "".join(lines[: start - 1] + lines[end:])


def promote_if_true_body_python(
    src: str,
    anchor_lineno: int,
    node_fingerprint: str = "",
    search_radius: int = _AST_ANCHOR_SEARCH_RADIUS,
) -> tuple[Optional[str], str]:
    """Replace ``if True: <body>`` with the body de-indented one level.

    else clause is dropped (dead code).  Returns (new_src, reason).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, "parse_fail"

    # Collect truthy-constant If nodes within radius; rank by distance then fingerprint.
    _candidates: list = []
    for _n in ast.walk(tree):
        if not isinstance(_n, ast.If):
            continue
        _dist = abs(_n.lineno - anchor_lineno)
        if _dist > search_radius:
            continue
        if node_fingerprint and node_condition_dump(_n) != node_fingerprint:
            continue
        if isinstance(_n.test, ast.Constant) and _n.test.value:
            # If node_fingerprint was set and we passed the guard above, fingerprint matched.
            _fp_match = bool(node_fingerprint)
            _candidates.append((_dist, not _fp_match, _n))

    if not _candidates:
        return None, "not_found"

    _candidates.sort(key=lambda t: (t[0], t[1]))
    target: Optional[ast.If] = _candidates[0][2]
    if not target.body:
        return None, "empty_body"

    lines = src.splitlines(keepends=True)
    total = len(lines)

    if_lineno = target.lineno
    if_block_end = getattr(target, "end_lineno", if_lineno)  # includes orelse
    body_start = target.body[0].lineno
    body_end = max(getattr(s, "end_lineno", s.lineno) for s in target.body)

    if_line_text = lines[if_lineno - 1] if if_lineno <= total else ""
    if_indent = len(if_line_text) - len(if_line_text.lstrip())

    first_body_text = lines[body_start - 1] if body_start <= total else ""
    first_body_indent = len(first_body_text) - len(first_body_text.lstrip())
    delta = first_body_indent - if_indent
    if delta <= 0:
        delta = 4

    # Inline form: "if True: stmt"
    if body_start == if_lineno:
        colon_pos = if_line_text.find(":")
        if colon_pos == -1:
            return None, "not_found"
        stmt_text = if_line_text[colon_pos + 1:].lstrip(" \t")
        if not stmt_text.strip():
            return None, "empty_body"
        promoted_line = " " * if_indent + stmt_text
        new_lines = [*lines[:if_lineno - 1], promoted_line, *lines[if_block_end:]]
        return ast_parse_result("".join(new_lines))

    def _de_indent(line: str) -> str:
        if not line.strip():
            return line
        if len(line) >= delta and line[:delta] == " " * delta:
            return line[delta:]
        stripped = line.lstrip()
        original_indent = len(line) - len(stripped)
        new_indent = max(0, original_indent - delta)
        return " " * new_indent + stripped

    body_lines = [_de_indent(lines[i]) for i in range(body_start - 1, body_end)]

    new_lines = lines[: if_lineno - 1] + body_lines + lines[if_block_end:]
    return ast_parse_result("".join(new_lines))


def apply_ast_node_delete_python(
    src: str,
    node_kind: str,
    anchor_lineno: int,
    node_fingerprint: str = "",
) -> tuple[Optional[str], str]:
    """Delete an If/While/Assert statement from Python *src* by AST identity.

    Workflow:
    1. Parse source into AST.
    2. Locate the target node near *anchor_lineno* (with fingerprint verification
       and a small search radius to tolerate stale line numbers from prior ops).
    3. Splice it out of the original text (preserving orelse for If nodes).
    4. Validate that the result still parses.

    Returns (new_src, reason):
      - (str, "ok")          : success
      - (None, "bad_kind")   : node_kind not in supported set
      - (None, "parse_fail") : source is unparseable
      - (None, "not_found")  : no matching node within search radius
      - (None, "syntax_broken") : result doesn't parse — structural issue in source
    """
    if node_kind not in _AST_NODE_DELETE_KINDS:
        return None, "bad_kind"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, "parse_fail"

    target = find_statement_node_by_anchor(
        tree, node_kind, anchor_lineno, node_fingerprint=node_fingerprint,
    )
    if target is None:
        return None, "not_found"

    new_src = rewrite_source_without_node(src, target, node_kind)

    try:
        ast.parse(new_src)
    except SyntaxError:
        return None, "syntax_broken"

    return new_src, "ok"


# ── Import manipulation ─────────────────────────────────────────────────


def remove_name_from_import_text(import_text: str, name: str) -> str:
    """Remove *name* (and adjacent comma/space) from an import statement text.

    Handles ``from X import A, B, C``, ``from X import (A, B, C)``,
    and ``import A, B, C``.
    Uses ``[ \\t]`` (not ``\\s``) for intra-line matching to avoid consuming
    newlines, then cleans up cross-line artifacts (double commas, dangling
    comma before ``)``) with ``\\s``-based post-processing.
    """
    escaped = re.escape(name)
    # Comma before name (non-first names): `, name`
    result = re.sub(
        r"[ \t]*,[ \t]*\b" + escaped + r"\b[ \t]*",
        "", import_text, count=1,
    )
    if result != import_text:
        result = re.sub(r",\s*,", ",", result)
        result = re.sub(r",\s*\)", ")", result)
        return result

    # Name with trailing comma (first name): `name, `
    result = re.sub(
        r"\b" + escaped + r"\b[ \t]*,[ \t]*",
        "", import_text, count=1,
    )
    if result != import_text:
        result = re.sub(r",\s*,", ",", result)
        result = re.sub(r",\s*\)", ")", result)
        return result

    return import_text


def delete_unused_import_from_python_source(
    src: str,
    unused_name: str,
    start_line: int,
) -> Optional[str]:
    """Delete an unused import name from Python source via AST-aware logic.

    Multi-name imports: remove just the unused name, keeping the import.
    Single-name imports: remove the entire import node; insert ``pass`` if the
    enclosing block becomes empty.

    Returns the modified source, or None when the target import is not found.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno == start_line:
            target = node
            break
    if target is None:
        return None

    unused_alias = None
    for alias in target.names:
        local_name = alias.asname or alias.name.split(".")[0]
        if local_name == unused_name:
            unused_alias = alias
            break
    if unused_alias is None:
        return None

    remaining = [a for a in target.names if a is not unused_alias]
    lines = src.splitlines(keepends=True)

    if remaining:
        import_start = target.lineno
        import_end = getattr(target, "end_lineno", target.lineno)
        import_text = "".join(lines[import_start - 1 : import_end])
        # For alias imports ("original as local"), search for the full "original as local"
        # text so the entire alias entry is removed rather than just the local name part.
        search_name = (
            f"{unused_alias.name} as {unused_alias.asname}"
            if unused_alias.asname
            else unused_name
        )
        new_import_text = remove_name_from_import_text(import_text, search_name)
        if new_import_text == import_text:
            return None
        result_lines = [*lines[:import_start - 1], new_import_text, *lines[import_end:]]
        return "".join(result_lines)

    node_end = getattr(target, "end_lineno", target.lineno)
    new_lines = lines[: target.lineno - 1] + lines[node_end:]
    new_src = "".join(new_lines)
    try:
        ast.parse(new_src)
    except SyntaxError:
        # Import removal left an invalid structure (e.g. empty try block).
        # Try to remove the enclosing try/except/finally entirely.
        enclosed = remove_enclosing_try_except(src, target.lineno)
        if enclosed is not None:
            return enclosed
        # Fallback: insert pass to maintain valid syntax
        target_line = lines[target.lineno - 1]
        indent = target_line[:len(target_line) - len(target_line.lstrip())]
        pass_line = f"{indent}pass\n"
        new_lines = [*lines[:target.lineno - 1], pass_line, *lines[node_end:]]
        new_src = "".join(new_lines)
    return new_src


def remove_enclosing_try_except(src: str, target_lineno: int) -> Optional[str]:
    """When target_lineno is the only statement in a try block, remove the
    entire try/except/finally block.

    E.g. ``try: import X`` with no other statements → the try-except guard
    is unnecessary after the import is removed.  Returns modified source or
    None when the import is not inside a try, or the try body has other stmts.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        try_start = node.lineno
        try_end = getattr(node, "end_lineno", node.lineno)
        if try_end <= try_start:
            continue
        if not (try_start <= target_lineno <= try_end):
            continue
        if len(node.body) == 1 and node.body[0].lineno == target_lineno:
            lines = src.splitlines(keepends=True)
            new_lines = lines[:try_start - 1] + lines[try_end:]
            new_src = "".join(new_lines)
            try:
                ast.parse(new_src)
                return new_src
            except SyntaxError:
                # Removal of this try breaks outer structure
                # (e.g. nested try-except), fall back to pass insertion.
                return None
        # Target is inside this try block but isn't the only body
        # statement — could be in a nested try. Continue walking.
        continue
    return None
