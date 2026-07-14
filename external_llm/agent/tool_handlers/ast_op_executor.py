"""AST-typed operation executor for Python code modifications.

Eliminates SANITIZE_FAIL by never embedding multi-line Python code inside JSON
strings.  Each op value is a single line or short expression that JSON handles
safely.

Supported op types:

  replace_expr  — replace a single-line expression/call within a function
  add_import    — add an import statement if not already present
  add_guard     — insert a guard statement at the start of a function body
  delete_stmt   — delete lines matching a text pattern (scoped to function)
"""

from __future__ import annotations

import ast
import builtins as _builtins
from dataclasses import dataclass
from typing import Any, Optional

from external_llm.code_structure_utils import is_module_level_import_present, iter_module_scope_nodes


def _safe_unparse_iter(loop_node: ast.For) -> str:
    """ast.unparse the iterable of a For loop node, with silent fallback."""
    try:
        return ast.unparse(loop_node.iter)
    except (SyntaxError, TypeError, AttributeError):
        return ""


def _guard_already_present(source: str, stmt: str, symbol: str) -> bool:
    """AST-based guard idempotency check.

    Returns True when the function ``symbol`` already contains an ``if``
    statement whose *condition* is AST-equivalent to the guard statement's
    condition, AND whose body contains a terminal action (raise/return/
    continue/break).

    Scans both the function body entry (first 5 stmts) AND first stmts of
    every for/while loop body, covering all insert_scope variants.
    """
    try:
        # Parse the guard statement to extract its condition.
        guard_tree = ast.parse(stmt, mode="exec")
        guard_if: ast.If | None = None
        for node in ast.walk(guard_tree):
            if isinstance(node, ast.If):
                guard_if = node
                break
        if guard_if is None:
            return False
        guard_cond_dump = ast.dump(guard_if.test)

        # Parse the source and find the target function.
        src_tree = ast.parse(source)
        func_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        bare = symbol.split(".")[-1] if "." in symbol else symbol
        for node in ast.walk(src_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == bare:
                func_node = node
                break
        if func_node is None:
            return False

        def _has_terminal_if(body_stmts: list, max_check: int = 5) -> bool:
            """Return True if any of the first max_check stmts is a matching guard."""
            for s in body_stmts[:max_check]:
                if not isinstance(s, ast.If):
                    continue
                if ast.dump(s.test) != guard_cond_dump:
                    continue
                for child in ast.walk(ast.Module(body=s.body, type_ignores=[])):
                    if isinstance(child, (ast.Raise, ast.Return, ast.Continue, ast.Break)):
                        return True
            return False

        # Scan entry of function body (skip docstring).
        body = func_node.body
        start_idx = 0
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)):
            start_idx = 1
        if _has_terminal_if(body[start_idx:]):
            return True

        # Scan entry of every loop body inside the function (for_loop / while_loop).
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.While)) and node.body:
                if _has_terminal_if(node.body):
                    return True

    except Exception:
        pass
    return False


@dataclass
class ASTOpResult:
    success: bool
    new_source: str
    ops_applied: int
    ops_failed: list[str]
    # True only when new_source actually differs from original source.
    # Idempotent ops (guard already present, import already exists, etc.)
    # set ops_applied > 0 but changed=False.
    changed: bool = False


class ASTOpExecutor:
    """Apply a list of typed AST ops to Python source, return modified source."""

    @staticmethod
    def _find_func_node(
        tree: ast.AST,
        bare_sym: str,
        parent_class: str = "",
    ):
        """Find a FunctionDef/AsyncFunctionDef node, optionally scoped to a class.

        When ``parent_class`` is provided, only returns a function that is a direct
        child of the named class — prevents matching a same-named method in a different
        class earlier in the file (e.g., two ``__init__`` methods in different classes).

        Returns the first matching AST node, or None.
        """
        if not bare_sym:
            return None

        if parent_class:
            # Walk the full class chain (supports nested: "OuterClass.InnerClass")
            class_chain = parent_class.split(".")
            current_body = tree.body
            for cls_name in class_chain:
                found_cls = None
                for node in current_body:
                    if isinstance(node, ast.ClassDef) and node.name == cls_name:
                        found_cls = node
                        break
                if found_cls is None:
                    return None
                current_body = found_cls.body
            # Search method in the innermost class body
            for child in current_body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == bare_sym
                ):
                    return child
            return None

        # No class scope — return first match anywhere (legacy behaviour)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == bare_sym
            ):
                return node
        return None

    def apply(
        self,
        source: str,
        ops: list[dict[str, Any]],
        symbol: str = "",
    ) -> ASTOpResult:
        # Parse qualified symbol "ClassName.method" → parent_class + bare_sym.
        # Passing only the bare name causes _find_func_node to match the FIRST
        # function with that name in the file, which may belong to a different class.
        _parts = symbol.split(".") if symbol else []
        _bare_sym = _parts[-1] if _parts else ""
        _parent_class = _parts[-2] if len(_parts) >= 2 else ""

        current = source
        applied = 0
        failed: list[str] = []
        _last_op_type: str = ""

        for op in ops:
            op_type = (op.get("type") or "").strip()
            _last_op_type = op_type
            try:
                if op_type == "replace_expr":
                    new_src, ok = self._replace_expr(current, op, _bare_sym, _parent_class)
                elif op_type == "add_import":
                    new_src, ok = self._add_import(current, op)
                elif op_type == "remove_import_name":
                    new_src, ok = self._remove_import_name(current, op)
                elif op_type == "add_class_field":
                    new_src, ok = self._add_class_field(current, op)
                elif op_type == "add_guard":
                    new_src, ok = self._add_guard(current, op, _bare_sym, _parent_class)
                elif op_type == "delete_stmt":
                    new_src, ok = self._delete_stmt(current, op, _bare_sym, _parent_class)
                elif op_type == "list_append":
                    new_src, ok = self._list_append(current, op)
                elif op_type == "list_remove":
                    new_src, ok = self._list_remove(current, op)
                else:
                    failed.append(f"unknown op type: {op_type!r}")
                    continue

                if ok:
                    current = new_src
                    applied += 1
                else:
                    # Prefer a specific error message set by the op handler over
                    # the generic "no match found" fallback.
                    _op_err = op.get("_error") or f"{op_type}: no match found"
                    failed.append(_op_err)
            except Exception as exc:
                failed.append(f"{op_type}: {exc}")

        # Final syntax + semantic validation — roll back completely on failure.
        # compile() catches semantic errors ast.parse() misses (e.g. 'continue'
        # outside a loop, 'return' outside a function, duplicate args).
        try:
            compile(current, "<ast_op_result>", "exec")
        except SyntaxError as se:
            # Build context: show surrounding lines from the broken source
            # so the LLM can see what the result looked like.
            _ctx_lines = current.splitlines()
            _err_lineno = se.lineno or 0
            _start = max(0, _err_lineno - 5)
            _end = min(len(_ctx_lines), _err_lineno + 4)
            _ctx_parts: list[str] = []
            for i in range(_start, _end):
                _marker = ">" if (i + 1) == _err_lineno else " "
                _ctx_parts.append(f"  {_marker} {i+1:4d}| {_ctx_lines[i]}")
            _ctx = "\n".join(_ctx_parts)

            return ASTOpResult(
                success=False,
                new_source=source,
                ops_applied=0,
                ops_failed=[
                    f"syntax error after applying {applied}/{len(ops)} ops "
                    f"(last op: {_last_op_type}): {se.msg} at line {_err_lineno}\n"
                    f"Result context around error:\n{_ctx}"
                ],
            )

        return ASTOpResult(
            success=applied > 0,
            new_source=current,
            ops_applied=applied,
            ops_failed=failed,
            changed=current != source,
        )

    # ── Op implementations ──────────────────────────────────────────────────

    @staticmethod
    def _ws_tolerant_span(text: str, old: str) -> Optional[str]:
        """Locate ``old`` in ``text`` tolerating per-line trailing-whitespace and
        line-ending differences, returning the EXACT original span (so a
        subsequent ``str.replace`` is guaranteed to hit real text), or None.

        Mirrors edit_text's whitespace-tolerant fallback: matches whole-line
        blocks by their rstrip()'d form (indentation preserved exactly, so the
        reconstructed span keeps correct indentation). Only meaningful when
        ``old`` spans complete lines — sub-line expressions are handled by the
        exact-match path and fall through here harmlessly.
        """
        norm_old = [ln.rstrip() for ln in old.splitlines()]
        if not norm_old:
            return None
        orig = text.splitlines(keepends=True)
        norm_text = [ln.rstrip() for ln in orig]
        n = len(norm_old)
        matches = []
        for i in range(len(norm_text) - n + 1):
            if norm_text[i:i + n] == norm_old:
                span = "".join(orig[i:i + n])
                # Honor old's trailing-newline intent: the keepends join carries
                # the last line's terminator, but if old had none, swallowing it
                # would merge the replacement with the following line.
                if span.endswith("\n") and not old.endswith(("\n", "\r")):
                    if span.endswith("\r\n"):
                        span = span[:-2]
                    else:
                        span = span[:-1]
                matches.append(span)
        if len(matches) == 1:
            return matches[0]
        return None

    def _replace_expr(
        self, source: str, op: dict[str, Any], symbol: str, parent_class: str = ""
    ) -> tuple[str, bool]:
        """Replace first occurrence of ``old`` with ``new``.

        When ``symbol`` is given, the replacement is scoped to that function's
        line range.  Falls back to file-level replace if the function is not
        found or the pattern is absent inside it.

        Matching is exact first; on miss, a whitespace-tolerant fallback retries
        ignoring per-line trailing whitespace / line endings (the common reason
        an LLM-supplied ``old`` fails to match an otherwise-identical block).
        """
        old: str = op.get("old") or ""
        new: str = op.get("new") or ""
        if not old:
            return source, False

        if symbol:
            try:
                tree = ast.parse(source)
                lines = source.splitlines(keepends=True)
                node = self._find_func_node(tree, symbol, parent_class)
                if node is not None:
                    s = node.lineno - 1        # 0-based inclusive start
                    e = node.end_lineno        # 0-based exclusive end
                    func_text = "".join(lines[s:e])
                    _match = old if old in func_text else self._ws_tolerant_span(func_text, old)
                    if not _match:
                        return source, False
                    if func_text.count(_match) > 1:
                        return source, False
                    new_func = func_text.replace(_match, new, 1)
                    result = "".join(lines[:s]) + new_func + "".join(lines[e:])
                    return result, True
            except Exception:
                pass  # fall through to file-level

        _match = old if old in source else self._ws_tolerant_span(source, old)
        if not _match:
            return source, False
        if source.count(_match) > 1:
            return source, False
        return source.replace(_match, new, 1), True

    def _add_import(self, source: str, op: dict[str, Any]) -> tuple[str, bool]:
        """Insert ``import`` after the last existing import line if not present."""
        import_stmt: str = (op.get("import") or "").strip()
        if not import_stmt:
            return source, False
        # Idempotent: AST module-scope check via shared helper.  The previous
        # `if import_stmt in source` substring check matched function-local
        try:
            _stmt_tree = ast.parse(import_stmt)
            _stmt_node = _stmt_tree.body[0] if _stmt_tree.body else None
        except SyntaxError:
            _stmt_node = None
        try:
            _src_tree = ast.parse(source)
        except SyntaxError:
            _src_tree = None
        if _src_tree is not None and isinstance(_stmt_node, ast.ImportFrom):
            _module = "." * (_stmt_node.level or 0) + (_stmt_node.module or "")
            _names = [a.name for a in _stmt_node.names]
            if _names and all(
                is_module_level_import_present(_src_tree, _module, _n) for _n in _names
            ):
                return source, True
        elif _src_tree is not None and isinstance(_stmt_node, ast.Import):
            _names = [a.name for a in _stmt_node.names]
            if _names and all(
                is_module_level_import_present(_src_tree, _n) for _n in _names
            ):
                return source, True

        # ── Merging (AST-based) ──────────────────────────────────────────
        # For ``from X import Y``: if ``from X import Z`` already exists at
        # module level, merge Y into the existing line instead of adding a new one.
        # Handles ALL import forms: single-line, parenthesized multi-line
        # (``from X import (\n    A,\n)``), backslash continuation, and nested
        # in version-gate wrappers (``if TYPE_CHECKING:``).
        #
        # Uses AST to read existing names (ast.ImportFrom.names gives clean,
        # parsed name list) and node.lineno/end_lineno to determine source
        # range.  Replaces the entire import block with a single merged line.
        #
        # This eliminates the old comma-splitting approach that corrupted
        # parenthesized imports by treating ``(`` as an import name.
        if isinstance(_stmt_node, ast.ImportFrom) and _stmt_node.module:
            _merge_module = _stmt_node.module
            _merge_new_names = [a.name for a in _stmt_node.names]
            _merge_level = getattr(_stmt_node, "level", 0) or 0
            if _src_tree is not None:
                for _scope_node in iter_module_scope_nodes(_src_tree):
                    if (
                        isinstance(_scope_node, ast.ImportFrom)
                        and _scope_node.module == _merge_module
                        and (getattr(_scope_node, "level", 0) or 0) == _merge_level
                    ):
                        _existing_names = [a.name for a in _scope_node.names]
                        # Skip ``from X import *`` — cannot merge with star imports
                        if "*" not in _existing_names and "*" not in _merge_new_names:
                            _merged = sorted(set(_existing_names + _merge_new_names))
                            _merge_lines = source.splitlines(keepends=True)
                            _start = _scope_node.lineno - 1  # AST 1-based → 0-based
                            _end = getattr(_scope_node, "end_lineno", _scope_node.lineno)
                            _first_line = _merge_lines[_start]
                            _indent = _first_line[:len(_first_line) - len(_first_line.lstrip())]
                            # Preserve relative import prefix ('.' for level=1, '..' for level=2, etc.)
                            _dot_prefix = "." * _merge_level if _merge_level > 0 else ""
                            _new_line = f"{_indent}from {_dot_prefix}{_merge_module} import {', '.join(n for n in _merged)}\n"
                            _merge_lines[_start:_end] = [_new_line]
                            return "".join(_merge_lines), True
        # ---- End merging --------------------------------------------------

        # ── Module-grouped placement ──────────────────────────────────────
        # For ``from X import Y``: if standalone ``import X`` exists at module
        # level, insert the new import right after it (same-module grouping)
        # instead of appending at the end of all imports.
        if isinstance(_stmt_node, ast.ImportFrom) and _stmt_node.module:
            _group_module = _stmt_node.module
            _group_lines = source.splitlines(keepends=True)
            for _gi, _gl in enumerate(_group_lines):
                _gstripped = _gl.strip()
                if (
                    _gl[:1] not in (" ", "\t")
                    and _gstripped.startswith("import ")
                    and " from " not in _gstripped
                    and len(_gstripped.split()) >= 2
                    and _gstripped.split()[1] == _group_module
                ):
                    # Found: ``import X`` — insert ``from X import Y`` right after
                    _group_lines.insert(_gi + 1, import_stmt + "\n")
                    return "".join(_group_lines), True
        # ---- End module-grouped placement ----------------------------------

        lines = source.splitlines(keepends=True)
        last_import_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Only count module-level imports (line must NOT start with whitespace).
            # Local (indented) imports inside functions must NOT anchor insertion --
            # inserting after them would place code inside a function body.
            if line[:1] not in (" ", "\t") and stripped.startswith(("import ", "from ")) and "import" in stripped:
                # Determine the true end of this import block (handles multi-line imports
                # like ``from X import (\n    A,\n    B,\n)`` where only the first line
                # is detected by the header check).
                # Single-line: end = i. Multi-line (parenthesized): find matching ``)``.
                import_end = i
                _paren_count = stripped.count("(") - stripped.count(")")
                if _paren_count > 0:
                    # Multi-line parenthesized import — scan forward for matching ``)``
                    for j in range(i + 1, len(lines)):
                        _js = lines[j]
                        _paren_count += _js.count("(") - _js.count(")")
                        if _paren_count <= 0:
                            import_end = j
                            break
                elif stripped.endswith("\\"):
                    # Backslash continuation — scan forward until no backslash
                    for j in range(i + 1, len(lines)):
                        if lines[j].rstrip().endswith("\\"):
                            import_end = j
                        else:
                            import_end = j
                            break
                # Place after the actual end of the import block (0-based → +1)
                last_import_idx = import_end + 1
        lines.insert(last_import_idx, import_stmt + "\n")
        return "".join(lines), True

    def _add_guard(
        self, source: str, op: dict[str, Any], symbol: str, parent_class: str = ""
    ) -> tuple[str, bool]:
        """Insert a guard statement into a specific scope of ``symbol``.

        op keys:
          statement   (str)  — the guard code to insert
          insert_scope (str) — where to insert:
            "function_body" (default) — after docstring, at function entry
            "for_loop"                — first line of a for loop body
            "while_loop"              — first line of a while loop body
          loop_variable (str, optional) — for "for_loop": match "for VAR in ..."
            If omitted, derived from Name nodes in the guard statement.
            Ambiguous (≥2 candidates) → returns (source, False) → LLM fallback.

        Idempotent: if guard statement already present verbatim, returns True.
        """
        # If a pre-parsed GuardIR dict is provided, use its compact form and
        # pre-computed scope/loop_variable directly (avoids redundant parsing).
        _ir_dict: Optional[dict[str, Any]] = op.get("ir") if isinstance(op.get("ir"), dict) else None
        if _ir_dict:
            _raw_stmt = _ir_dict.get("compact") or _ir_dict.get("statement") or op.get("statement") or ""
            stmt = _raw_stmt.strip()
            insert_scope = (_ir_dict.get("insert_scope") or op.get("insert_scope") or "function_body").strip()
            loop_variable = (_ir_dict.get("loop_variable") or op.get("loop_variable") or "").strip()
        else:
            stmt = (op.get("statement") or "").strip()
            insert_scope = (op.get("insert_scope") or "function_body").strip()
            loop_variable = (op.get("loop_variable") or "").strip()
        loop_iterable_src = (op.get("loop_iterable_src") or "").strip()

        if not stmt or not symbol:
            return source, False

        # Idempotent: guard already present — check verbatim first (fast path),
        # then fall back to AST-based condition equivalence (handles format/quote
        # differences, e.g. one-liner vs multi-line, single vs double quotes).
        if stmt in source:
            return source, True
        if _guard_already_present(source, stmt, symbol):
            return source, True

        try:
            tree = ast.parse(source)
            lines = source.splitlines(keepends=True)
            func_node = self._find_func_node(tree, symbol, parent_class)
            if func_node is None:
                return source, False

            if insert_scope == "function_body":
                return self._insert_at_function_body(lines, func_node, stmt)

            if insert_scope in ("for_loop", "while_loop"):
                return self._insert_at_loop_body(
                    lines, func_node, stmt, insert_scope, loop_variable,
                    loop_iterable_src=loop_iterable_src,
                )

        except Exception:
            pass
        return source, False

    def _insert_at_function_body(
        self,
        lines: list[str],
        func_node: ast.AST,
        stmt: str,
    ) -> tuple[str, bool]:
        """Insert guard after the function docstring (or after def line).

        Uses name-safety: if the guard references names defined later in the
        function, the insertion point is adjusted to after their definitions.
        """
        body = func_node.body  # type: ignore[attr-defined]
        if not body:
            return "".join(lines), False

        source = "".join(lines)
        safe_line = self._find_safe_insertion_point(stmt, func_node, source)

        if safe_line == -1:
            return "".join(lines), False

        if safe_line == func_node.lineno:  # type: ignore[attr-defined]
            # All names available from start — use docstring-aware placement
            if (
                isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                insert_after = body[0].end_lineno  # 1-based, inclusive
            else:
                insert_after = func_node.lineno  # type: ignore[attr-defined]
        else:
            insert_after = safe_line

        first_body_line = lines[body[0].lineno - 1]
        indent_str = first_body_line[:len(first_body_line) - len(first_body_line.lstrip())]

        guard_lines = [indent_str + ln + "\n" for ln in stmt.splitlines()]
        insert_idx = insert_after  # 0-based position (insert *after* line)
        new_lines = lines[:insert_idx] + guard_lines + lines[insert_idx:]
        return "".join(new_lines), True

    def _insert_at_loop_body(
        self,
        lines: list[str],
        func_node: ast.AST,
        stmt: str,
        insert_scope: str,
        loop_variable: str,
        *,
        loop_iterable_src: str = "",
    ) -> tuple[str, bool]:
        """Insert guard as first statement of a loop body inside ``func_node``.

        Uniqueness contract: exactly one loop must match.  Zero or ≥2 matches
        → returns (original_source, False) so the caller can fall back to LLM.

        loop_iterable_src: when provided, used as a secondary discriminator to
        resolve ambiguous cases where multiple loops share the same loop variable.
        Loops whose ``ast.unparse(iter)`` equals this string are kept; others
        are dropped before the uniqueness check.
        """
        loop_type: type = ast.For if insert_scope == "for_loop" else ast.While

        # Derive loop_variable from guard statement when not supplied.
        if insert_scope == "for_loop" and not loop_variable:
            loop_variable = self._derive_loop_variable(stmt, func_node) or ""
            # If derivation is ambiguous (returned ""), we fall through to
            # the candidate-count check which will catch 0-match or ≥2-match.

        # Collect loop candidates inside the function (direct + nested).
        candidates: list[ast.AST] = [
            node for node in ast.walk(func_node)
            if isinstance(node, loop_type)
            and (
                insert_scope != "for_loop"
                or not loop_variable
                or self._loop_target_matches(node.target, loop_variable)  # type: ignore[attr-defined]
            )
        ]

        # Secondary filter: when multiple loops share the same variable,
        # use iterable_src to select exactly the right one.
        if len(candidates) > 1 and loop_iterable_src and insert_scope == "for_loop":
            _filtered = [
                c for c in candidates
                if _safe_unparse_iter(c) == loop_iterable_src  # type: ignore[attr-defined]
            ]
            if len(_filtered) >= 1:
                candidates = _filtered

        if len(candidates) != 1:
            # 0 → loop not found; ≥2 → ambiguous.  Executor must not choose.
            return "".join(lines), False

        loop_node = candidates[0]
        body = loop_node.body  # type: ignore[attr-defined]
        if not body:
            return "".join(lines), False

        # Name-safety: check guard references are all available before this loop.
        source = "".join(lines)
        if not self._is_safe_for_loop_body(stmt, func_node, source, loop_node):
            return "".join(lines), False

        # Insert before the first statement of the loop body.
        insert_before = body[0].lineno - 1  # 0-based
        first_body_line = lines[insert_before]
        indent_str = first_body_line[:len(first_body_line) - len(first_body_line.lstrip())]

        guard_lines = [indent_str + ln + "\n" for ln in stmt.splitlines()]
        new_lines = lines[:insert_before] + guard_lines + lines[insert_before:]
        return "".join(new_lines), True

    @staticmethod
    def _compute_name_safety_info(
        guard_stmt: str,
        func_node: ast.AST,
        source: str,
        *,
        extra_always_avail: set[str] | None = None,
    ) -> tuple[set[str], dict[str, int], set[str]]:
        """Extract name-safety data for a guard statement.

        Returns:
            (guard_names, first_def, always_avail):
            - guard_names: Name(Load) ids extracted from guard_stmt
            - first_def: mapping name → first definition line in func_node scope
            - always_avail: names available from any point (builtins,
              module-level names, function parameters, extras)
        """
        # 1. Guard names
        _gs = guard_stmt.strip()
        try:
            _gt = ast.parse(_gs, mode="exec")
        except SyntaxError:
            # Guard may be an incomplete statement (e.g., bare "if condition:")
            # — add a dummy body to make it parseable.
            try:
                _gt = ast.parse(_gs + "\n    pass", mode="exec")
            except SyntaxError:
                return set(), {}, set()

        guard_names: set[str] = set()
        for _n in ast.walk(_gt):
            if isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Load):
                guard_names.add(_n.id)

        # 2. Always-available set
        always_avail: set[str] = set(dir(_builtins))
        always_avail.update({
            "self", "cls", "_", "None", "True", "False",
            "NotImplemented", "Ellipsis", "__name__",
            "__file__", "__doc__", "__package__",
        })

        # Module-level names
        try:
            _ft = ast.parse(source)
        except SyntaxError:
            _ft = None

        if _ft is not None:
            for _n in ast.walk(_ft):
                if isinstance(_n, ast.Import):
                    for _a in _n.names:
                        always_avail.add(_a.asname or _a.name.split(".")[0])
                elif isinstance(_n, ast.ImportFrom):
                    for _a in _n.names:
                        always_avail.add(_a.asname or _a.name)
                elif isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if getattr(_n, "col_offset", 99) == 0:
                        always_avail.add(_n.name)
                elif isinstance(_n, ast.Assign) and getattr(_n, "col_offset", 99) == 0:
                    for _t in _n.targets:
                        if isinstance(_t, ast.Name):
                            always_avail.add(_t.id)

        # Function parameters
        for _a in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
            always_avail.add(_a.arg)
        if func_node.args.vararg:
            always_avail.add(func_node.args.vararg.arg)
        if func_node.args.kwarg:
            always_avail.add(func_node.args.kwarg.arg)

        # Extra always-available (e.g., loop variable)
        if extra_always_avail:
            always_avail.update(extra_always_avail)

        # 3. First-def timeline within function scope
        first_def: dict[str, int] = {}

        def _record(n: str, ln: int) -> None:
            if n not in first_def:
                first_def[n] = ln

        _func_lineno = func_node.lineno
        for _a in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
            _record(_a.arg, _func_lineno)
        if func_node.args.vararg:
            _record(func_node.args.vararg.arg, _func_lineno)
        if func_node.args.kwarg:
            _record(func_node.args.kwarg.arg, _func_lineno)

        for _n in ast.walk(func_node):
            if isinstance(_n, ast.Assign):
                for _t in _n.targets:
                    if isinstance(_t, ast.Name):
                        _record(_t.id, _n.lineno)
            elif isinstance(_n, ast.AnnAssign) and isinstance(_n.target, ast.Name):
                _record(_n.target.id, _n.lineno)
            elif isinstance(_n, (ast.For, ast.AsyncFor)) and isinstance(_n.target, ast.Name):
                _record(_n.target.id, _n.lineno)
            elif isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _n is not func_node:
                _record(_n.name, _n.lineno)
            elif isinstance(_n, ast.NamedExpr) and isinstance(_n.target, ast.Name):
                _record(_n.target.id, _n.lineno)
            elif isinstance(_n, ast.With):
                for _item in _n.items:
                    if isinstance(_item.optional_vars, ast.Name):
                        _record(_item.optional_vars.id, _n.lineno)

        return guard_names, first_def, always_avail

    def _find_safe_insertion_point(
        self,
        guard_stmt: str,
        func_node: ast.AST,
        source: str,
    ) -> int:
        """Find the first safe line (1-based) to insert guard_stmt after.

        Safety: all Name(Load) in guard_stmt (excluding always-available names)
        must have at least one definition/assignment before the insertion point.

        Returns: 1-based line number to insert AFTER (use as 0-based slice
        index into the lines list).  Returns -1 if no safe point exists.
        """
        guard_names, first_def, always_avail = self._compute_name_safety_info(
            guard_stmt, func_node, source,
        )
        if not guard_names:
            return func_node.lineno

        need_def = guard_names - always_avail
        if not need_def:
            return func_node.lineno  # safe from function start

        # Check all needed names have definitions in scope
        for name in need_def:
            if name not in first_def:
                return -1  # undefined name → can't safely insert

        # Find the last definition line among needed names
        last_def_line = max(first_def[name] for name in need_def)

        # Walk body statements to find the first safe insertion boundary.
        # Inserting after a body statement's end_lineno is always a valid
        # function-body-level boundary (never splits a compound statement).
        body = func_node.body
        for stmt in body:
            stmt_end = getattr(stmt, "end_lineno", stmt.lineno)
            if stmt_end >= last_def_line:
                return stmt_end

        return -1

    @staticmethod
    def _is_safe_for_loop_body(
        stmt: str,
        func_node: ast.AST,
        source: str,
        loop_node: ast.AST,
    ) -> bool:
        """Check if guard_stmt can be safely inserted at the start of loop body.

        All guard-referenced names must be defined before the loop node to
        avoid forward-reference runtime errors.
        """
        # Add loop variable to always-available for for-loops
        extra: set[str] = set()
        if isinstance(loop_node, (ast.For, ast.AsyncFor)) and isinstance(loop_node.target, ast.Name):
            extra.add(loop_node.target.id)

        guard_names, first_def, always_avail = ASTOpExecutor._compute_name_safety_info(
            stmt, func_node, source, extra_always_avail=extra,
        )
        if not guard_names:
            return True

        need_def = guard_names - always_avail
        if not need_def:
            return True

        loop_lineno = loop_node.lineno
        for name in need_def:
            def_line = first_def.get(name)
            if def_line is None or def_line > loop_lineno:
                return False
        return True

    @staticmethod
    def _derive_loop_variable(stmt: str, func_node: ast.AST) -> str:
        """Derive loop variable by intersecting Name nodes in guard stmt with
        for-loop target variables in the function.

        Returns the unique variable name, or "" if 0 or ≥2 candidates.
        """
        try:
            stmt_names: set = {
                n.id
                for n in ast.walk(ast.parse(stmt, mode="exec"))
                if isinstance(n, ast.Name)
            }
        except SyntaxError:
            return ""

        loop_target_names: set = set()
        for node in ast.walk(func_node):
            if isinstance(node, ast.For):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        loop_target_names.add(n.id)

        matches = stmt_names & loop_target_names
        return matches.pop() if len(matches) == 1 else ""

    @staticmethod
    def _loop_target_matches(target_node: ast.AST, variable_name: str) -> bool:
        """Return True if ``variable_name`` appears as a target in the loop."""
        return any(
            isinstance(n, ast.Name) and n.id == variable_name
            for n in ast.walk(target_node)
        )

    def _delete_stmt(
        self, source: str, op: dict[str, Any], symbol: str, parent_class: str = ""
    ) -> tuple[str, bool]:
        """Delete lines containing ``pattern``, optionally scoped to ``symbol``.

        IMPORTANT: when ``symbol`` is provided, deletion is strictly limited to
        that symbol's AST range.  If the symbol cannot be resolved the operation
        fails loudly (returns False + sets op["_error"]) instead of silently
        falling back to a file-wide delete.
        """
        pattern: str = (op.get("pattern") or "").strip()
        if not pattern:
            return source, False

        lines = source.splitlines(keepends=True)

        if symbol:
            try:
                tree = ast.parse(source)
                node = self._find_func_node(tree, symbol, parent_class)
                if node is None:
                    # Symbol not found — fail loudly instead of falling back to
                    # file-wide deletion, which risks removing unrelated lines.
                    op["_error"] = (
                        f"delete_stmt: symbol '{symbol}' not found in file; "
                        "operation aborted to prevent file-wide deletion."
                    )
                    return source, False
                s = node.lineno - 1
                e = node.end_lineno
                inner = [_item_ for _item_ in lines[s:e] if pattern not in _item_]
                if len(inner) == e - s:
                    return source, False  # nothing removed
                new_lines = lines[:s] + inner + lines[e:]
                return "".join(new_lines), True
            except Exception as exc:
                # AST parse error — fail loudly instead of silent file-wide fallback.
                op["_error"] = (
                    f"delete_stmt: AST parse failed ({exc}); "
                    "operation aborted to prevent file-wide deletion."
                )
                return source, False

        # No symbol specified — file-wide deletion (intentional, use with care).
        new_lines = [_item_ for _item_ in lines if pattern not in _item_]
        if len(new_lines) == len(lines):
            return source, False
        return "".join(new_lines), True

    def _remove_import_name(self, source: str, op: dict[str, Any]) -> tuple[str, bool]:
        """Remove a specific name from a 'from MODULE import A, B, C' statement.

        If the removed name is the only one, the entire import line is deleted.
        If other names remain, the line is rewritten with only the remaining names.
        Formatting (indentation, trailing newline) is preserved.

        op keys:
          module (str, optional): module to match. Accepts dotted relative form
            (e.g. ".pkg", "..mod") OR plain absolute form (e.g. "dataclasses");
            both are normalized against the import's node.level + node.module.
          name   (str, required): the specific name to remove (e.g. "field")
        """
        name: str = (op.get("name") or "").strip()
        module: str = (op.get("module") or "").strip()
        if not name:
            return source, False

        # Use ast.parse() to handle parenthesized imports, backslash
        # continuations, and docstring false positives (SL50/substring risk).
        lines = source.splitlines(keepends=True)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, False

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            # Normalize module to include leading dots (relative imports): AST stores
            # the leading dots in node.level, not node.module. Without this, callers
            # that pass a dotted module such as ".pkg" (e.g. symbol_handlers_apply)
            # would fail to match (".pkg" != "pkg"), and the rewrite below would drop
            # the dots, corrupting the import into a broken absolute form (F821).
            node_module = "." * (node.level or 0) + (node.module or "")
            if module and node_module != module:
                continue
            if not any(a.name == name or (a.asname and a.asname == name)
                       for a in node.names):
                continue

            start_line = node.lineno      # 1-based
            end_line = node.end_lineno    # 1-based, inclusive

            # Keep only names that are NOT the target
            remaining = []
            for a in node.names:
                if a.name == name or (a.asname and a.asname == name):
                    continue
                if a.asname:
                    remaining.append(f"{a.name} as {a.asname}")
                else:
                    remaining.append(a.name)

            # Preserve indentation from the import's first line
            first_line = lines[start_line - 1]
            indent_str = first_line[:len(first_line) - len(first_line.lstrip())]

            if remaining:
                trailing_newline = "\n" if lines[end_line - 1].endswith("\n") else ""
                new_line = (
                    f"{indent_str}from {node_module} import "
                    f"{', '.join(remaining)}{trailing_newline}"
                )
                # Replace entire multi-line import block with compact single line
                lines[start_line - 1:end_line] = [new_line]
            else:
                # Delete entire import block (removes all lines)
                del lines[start_line - 1:end_line]

            return "".join(lines), True

        return source, False

    def _add_class_field(self, source: str, op: dict[str, Any]) -> tuple[str, bool]:
        """Add an annotated field to a class body (idempotent).

        Inserts after the last existing annotated field (AnnAssign), or after
        the class docstring if no fields exist yet.  Preserves file formatting —
        does NOT use ast.unparse (which would lose comments and style).

        op keys:
          class_name    (str, required): name of the class to modify
          field_name    (str, required): new attribute name
          field_type    (str, required): type annotation string (e.g. "float")
          field_default (str, optional): default value string (e.g. "1.0")
                                         if omitted, field has no default
        """
        class_name: str = (op.get("class_name") or "").strip()
        field_name: str = (op.get("field_name") or "").strip()
        field_type: str = (op.get("field_type") or "").strip()
        field_default: str = (op.get("field_default") or "").strip()
        if not class_name or not field_name or not field_type:
            # Specific error (not the generic "no match found") so the caller can
            # see it's a parameter problem, not a missing target in the file.
            op["_error"] = (
                "add_class_field requires 'class_name', 'field_name', and "
                f"'field_type' (got class_name={class_name!r}, "
                f"field_name={field_name!r}, field_type={field_type!r})"
            )
            return source, False

        try:
            tree = ast.parse(source)
            lines = source.splitlines(keepends=True)

            # Find the class node
            cls_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    cls_node = node
                    break
            if cls_node is None:
                op["_error"] = f"add_class_field: class {class_name!r} not found in file"
                return source, False

            # Idempotent: check if field already defined in this class
            for stmt in cls_node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == field_name:
                        return source, True  # already present

            # Find insertion point: after last AnnAssign/Assign in class body
            last_field_lineno = None  # 0-based
            for stmt in cls_node.body:
                if isinstance(stmt, (ast.AnnAssign, ast.Assign)):
                    last_field_lineno = stmt.end_lineno - 1  # 0-based

            if last_field_lineno is None:
                # No fields yet — insert after docstring or after class def line
                body = cls_node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                ):
                    last_field_lineno = body[0].end_lineno - 1
                else:
                    last_field_lineno = cls_node.lineno - 1  # after class def

            # Detect indentation from first body statement
            if cls_node.body:
                first_body_line = lines[cls_node.body[0].lineno - 1]
                indent_str = first_body_line[:len(first_body_line) - len(first_body_line.lstrip())]
            else:
                indent_str = "    "

            # Build the field line
            if field_default:
                field_line = f"{indent_str}{field_name}: {field_type} = {field_default}\n"
            else:
                field_line = f"{indent_str}{field_name}: {field_type}\n"

            insert_pos = last_field_lineno + 1  # insert AFTER last field line (0-based)
            new_lines = [*lines[:insert_pos], field_line, *lines[insert_pos:]]
            new_source = "".join(new_lines)

            # Validate — roll back on syntax error
            ast.parse(new_source)
            return new_source, True
        except (SyntaxError, TypeError, AttributeError):
            return source, False

    def _list_append(self, source: str, op: dict[str, Any]) -> tuple[str, bool]:
        """Append a string value to a module-level list variable (idempotent).

        op keys:
          list_name  (str, required): name of the list variable (e.g. "__all__")
          value      (str, required): string literal to append (e.g. "my_func")

        Finds the assignment ``list_name = [...]`` or ``list_name = (...)`` and
        appends the quoted value if not already present.  Preserves surrounding
        formatting — does NOT use ast.unparse.
        """
        list_name: str = (op.get("list_name") or "").strip()
        value: str = (op.get("value") or "").strip()
        if not list_name or not value:
            return source, False

        quoted = f'"{value}"'

        try:
            tree = ast.parse(source)
            lines = source.splitlines(keepends=True)

            # Find the module-level assignment: list_name = [...]
            target_node = None
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if len(node.targets) != 1:
                    continue
                t = node.targets[0]
                if not (isinstance(t, ast.Name) and t.id == list_name):
                    continue
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    target_node = node
                    break

            if target_node is None:
                return source, False

            # Idempotent: check if value is already in the list
            for elt in target_node.value.elts:
                if isinstance(elt, ast.Constant) and str(elt.value) == value:
                    return source, True  # already present

            # Find the closing bracket/paren line
            end_lineno = target_node.end_lineno - 1  # 0-based
            end_line = lines[end_lineno]

            # Detect bracket type and closing character
            bracket_close = "]" if isinstance(target_node.value, ast.List) else ")"
            close_idx = end_line.rfind(bracket_close)
            if close_idx == -1:
                return source, False

            # Detect indentation from existing elements or use list indent + 4
            existing_elts = target_node.value.elts
            if existing_elts:
                first_elt_line = lines[existing_elts[0].lineno - 1]
                indent_str = first_elt_line[:len(first_elt_line) - len(first_elt_line.lstrip())]
            else:
                list_line = lines[target_node.lineno - 1]
                indent_str = list_line[:len(list_line) - len(list_line.lstrip())] + "    "

            # Detect trailing comma convention from last element
            trailing_comma = ""
            if existing_elts:
                last_elt = existing_elts[-1]
                last_elt_line = lines[last_elt.end_lineno - 1]
                if "," in last_elt_line[last_elt.col_offset:last_elt.end_col_offset + 5]:
                    trailing_comma = ","

            # Single-line vs multi-line list
            is_multiline = target_node.value.lineno != target_node.value.end_lineno

            if is_multiline:
                # Insert new element line before the closing bracket
                new_element_line = f"{indent_str}{quoted},{trailing_comma}\n"
                new_lines = [*lines[:end_lineno], new_element_line, *lines[end_lineno:]]
            else:
                # Inline: insert before closing bracket
                # e.g. ["a", "b"] → ["a", "b", "c"]
                prefix = end_line[:close_idx]
                suffix = end_line[close_idx:]
                sep = ", " if prefix.rstrip().rstrip(",").strip() else ""
                new_end_line = prefix.rstrip().rstrip(",") + f'{sep}{quoted}' + suffix
                new_lines = [*lines[:end_lineno], new_end_line, *lines[end_lineno + 1:]]

            new_source = "".join(new_lines)
            ast.parse(new_source)  # validate
            return new_source, True
        except (SyntaxError, TypeError, AttributeError):
            return source, False

    def _list_remove(self, source: str, op: dict[str, Any]) -> tuple[str, bool]:
        """Remove a string value from a module-level list variable (idempotent).

        op keys:
          list_name  (str, required): name of the list variable (e.g. "__all__")
          value      (str, required): string literal to remove (e.g. "old_func")
        """
        list_name: str = (op.get("list_name") or "").strip()
        value: str = (op.get("value") or "").strip()
        if not list_name or not value:
            return source, False

        try:
            tree = ast.parse(source)
            lines = source.splitlines(keepends=True)

            target_node = None
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if len(node.targets) != 1:
                    continue
                t = node.targets[0]
                if not (isinstance(t, ast.Name) and t.id == list_name):
                    continue
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    target_node = node
                    break

            if target_node is None:
                return source, False

            # Find the element to remove
            elt_to_remove = None
            for elt in target_node.value.elts:
                if isinstance(elt, ast.Constant) and str(elt.value) == value:
                    elt_to_remove = elt
                    break

            if elt_to_remove is None:
                return source, True  # not present → idempotent success

            # Remove the element's line (handles both inline and multi-line lists)
            elt_lineno = elt_to_remove.lineno - 1  # 0-based
            elt_line = lines[elt_lineno]

            # Check if element is alone on its line
            stripped = elt_line.strip().rstrip(",")
            quoted_variants = (f'"{value}"', f"'{value}'")
            if stripped in quoted_variants:
                # Remove entire line
                new_lines = lines[:elt_lineno] + lines[elt_lineno + 1:]
            else:
                # Inline: remove just the element and its comma
                for q in quoted_variants:
                    for pattern in (f", {q}", f",{q}", f"{q}, ", f"{q},"):
                        if pattern in elt_line:
                            new_line = elt_line.replace(pattern, "", 1)
                            new_lines = [*lines[:elt_lineno], new_line, *lines[elt_lineno + 1:]]
                            break
                    else:
                        continue
                    break
                else:
                    return source, False

            new_source = "".join(new_lines)
            ast.parse(new_source)
            return new_source, True
        except (SyntaxError, TypeError, AttributeError):
            return source, False

