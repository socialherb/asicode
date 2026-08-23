"""code_structure_utils.py — Python code structure detection (line + tree-sitter).

Two layers:
- **Line-level heuristics** (is_python_definition, extract_symbol_name, etc.)
  For hot-path line scanning in patch engine and repair loops.
- **tree-sitter functions** (parse_definitions, find_definition_at_line, etc.)
  For accurate detection of multi-line defs, nested classes, decorator stacks.
  Falls back to Python AST when tree-sitter is unavailable.

Callers should prefer AST functions when full source is available.
Fall back to line heuristics only when scanning line-by-line in tight loops.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import re
from dataclasses import dataclass, field

from .languages import LanguageId as _LanguageId
from .languages.base import find_brace_block_end_offset
from .languages.tree_sitter_utils import (
    find_all_symbols as _ts_find_all_symbols,
)
from .languages.tree_sitter_utils import (
    grammar_key_for_path as _ts_grammar_key_for_path,
)
from .languages.tree_sitter_utils import (
    parse_to_tree as _ts_parse_to_tree,
)

# tree_sitter_utils guards its own optional tree-sitter import, so this
# module always imports — the old try/except ImportError was dead code.
_HAS_TS = True

logger = logging.getLogger("asicode.code_structure_utils")
__all__ = [
    "DefinitionInfo",
    "FunctionSignature",
    "collect_defined_names",
    "extract_function_signature",
    "extract_function_signature_detailed",
    "extract_symbol_name",
    "find_definition_at_line",
    "find_import_boundary_ast",
    "find_last_top_level_def",
    "is_class_def",
    "is_decorator",
    "is_function_def",
    "is_import_boundary",
    "is_python_definition",
    "parse_definitions",
    "symbol_defined_anywhere",
    "symbol_exists_in_module",
]
_RE_FUNC_DEF = re.compile("^(\\s*)(?:async\\s+)?def\\s+(\\w+)\\s*\\(")
_RE_CLASS_DEF = re.compile("^(\\s*)class\\s+(\\w+)\\s*[\\(\\[:]")
_RE_DECORATOR = re.compile("^(\\s*)@")

# ── Static regex fallback patterns (non-Python symbol checks) ────────────────
# Hoisted module-level: per-call re.compile of an identical string only churns
# the re module's internal cache (maxsize 512) — hoisting keeps these resident
# and immune to eviction under heavy dynamic-pattern load.
# NOTE (P11-3): the class-header list parts use [^\s,]+ — \S includes commas,
# so \S+ and (?:\s*,\s*\S+)* had an ambiguous split that backtracks
# exponentially when the closing '{' is absent (realistic in .d.ts:
# 'declare class Foo extends Mixin<T0,...,Tn>;'). [^\s,]+ gives the
# comma-delimited list a unique parse (linear).
_RE_TS_CLASS_BODY = re.compile(
    r"(?:export\s+)?(?:abstract\s+)?class\s+\w+\s*"
    r"(?:extends\s+[^\s,]+(?:\s*,\s*[^\s,]+)*\s*)?"
    r"(?:implements\s+[^\s,]+(?:\s*,\s*[^\s,]+)*\s*)?\{"
)
_RE_JVM_CLASS_HEADER = re.compile(r"\b(?:class|interface|enum|object)\s+\w+")
_RE_TS_LAST_TOP_LEVEL_DEF = re.compile(
    r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?"
    r"(?:(?:function|class)\s+(\w+)"
    r"|const\s+(\w+)\s*=)",
    re.MULTILINE,
)


# ── tree-sitter helpers for non-Python symbol detection ───────────────
# ``find_all_symbols`` already supports TS/JS/Go (and more) accurately and
# ignores comment/string occurrences — unlike the regex patterns below which
# false-positive on symbols mentioned inside comments. These helpers prefer
# tree-sitter and fall back to regex only when tree-sitter is unavailable or
# the source fails to parse.


def _collect_symbols_via_ts(content: str, lang_id: _LanguageId) -> list | None:
    """Return ``[(name, kind, start_line, end_line), ...]`` via tree-sitter.

    Returns ``None`` (not empty list) when tree-sitter is unavailable or the
    source failed to parse, so callers can distinguish "definitely empty"
    from "couldn't tell — fall back to regex".
    """
    if not _HAS_TS:
        return None
    # LanguageId.value IS the tree-sitter language string
    # (e.g. TYPESCRIPT.value == "typescript", GO.value == "go").
    try:
        syms = _ts_find_all_symbols(content, lang_id.value)
    except Exception:
        logger.debug("_collect_symbols_via_ts: tree-sitter failed", exc_info=True)
        return None
    return syms if syms else None


def _go_module_level_symbol_exists(content: str, symbol: str) -> bool | None:
    """Go-specific module-level check via direct tree walk.

    ``find_all_symbols`` collapses Go ``method_declaration`` and
    ``function_declaration`` into the same ``function`` kind, so it cannot
    tell a top-level ``func Foo()`` from a receiver method
    ``func (r T) Foo()``. This helper walks the raw AST and returns True only
    for non-receiver (module-level) declarations matching *symbol*:

    - ``function_declaration`` (not ``method_declaration``)
    - ``type_declaration`` (``type Foo struct/interface``)
    - ``var_declaration`` / ``const_declaration``

    Returns ``None`` when tree-sitter is unavailable (caller falls back to
    the pre-existing conservative behaviour).
    """
    if not _HAS_TS:
        return None
    try:
        tree = _ts_parse_to_tree(content, "go")
    except Exception:
        logger.debug("_go_module_level_symbol_exists: tree-sitter parse failed", exc_info=True)
        return None
    if tree is None:
        return None

    code_bytes = content.encode("utf-8")
    found = False
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        ntype = node.type
        if ntype == "function_declaration":
            # Top-level func Foo() — method_declaration has a receiver and
            # is NOT module-level, so we deliberately skip that node type.
            nm = node.child_by_field_name("name")
            if nm is not None and code_bytes[nm.start_byte : nm.end_byte].decode("utf-8", "replace") == symbol:
                found = True
                break
        elif ntype == "type_declaration":
            # type ( Foo struct/interface ) — may declare multiple types.
            for child in node.children:
                if child.type == "type_spec":
                    nm = child.child_by_field_name("name")
                    if nm is not None and code_bytes[nm.start_byte : nm.end_byte].decode("utf-8", "replace") == symbol:
                        found = True
                        break
            if found:
                break
        elif ntype in ("var_declaration", "const_declaration"):
            # var/const ( Foo, Bar ) = ... or var Foo = ...
            for spec in node.children:
                if spec.type in ("var_spec", "const_spec"):
                    for nm_node in spec.children:
                        if (
                            nm_node.type == "identifier"
                            and code_bytes[nm_node.start_byte : nm_node.end_byte].decode("utf-8", "replace") == symbol
                        ):
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        stack.extend(reversed(node.children))
    return found


def _ts_member_is_defined(syms: list, symbol: str) -> bool | None:
    """Containment-based dotted/bare member search over a flat symbol list.

    ``syms`` is the output of ``find_all_symbols``: ``(name, kind, start, end)``
    tuples where classes and their methods appear in one flat list (methods'
    line ranges are nested inside their enclosing class range in TS/JS, but
    NOT in Go where methods are top-level ``func (r T) M()``).

    Returns:
    - ``True`` — the symbol is definitely defined.
    - ``False`` — the symbol is definitely NOT defined (top-level name missing,
      or dotted member whose class/type does not exist).
    - ``None`` — inconclusive for a dotted member whose class/type EXISTS but
      the member isn't structurally verifiable (e.g. Go methods are not nested
      inside their type's range). Caller should fall back to regex.

    Bare names and TS/JS dotted names are fully resolvable (True/False).
    Go dotted names may be inconclusive (None) → caller falls back to regex.
    """
    if not syms:
        return False
    if "." in symbol:
        cls, member = symbol.split(".", 1)
        cls_ranges = [(s, e) for (n, _k, s, e) in syms if n == cls]
        if not cls_ranges:
            return False  # class/type doesn't exist at all → definitely not
        for mname, _mk, ms, me in syms:
            if mname != member:
                continue
            for cs, ce in cls_ranges:
                # strict containment, exclude the class node itself
                if cs <= ms and me <= ce and (ms, me) != (cs, ce):
                    return True
        # Member not found inside the class range. For languages whose
        # methods are nested (TS/JS), this means it genuinely doesn't exist.
        # For Go where methods are top-level and NOT nested, we can't tell —
        # let the caller try regex on the receiver syntax instead.
        # Heuristic: if ANY symbol's range is strictly contained in a class
        # range, the language nests members (TS/JS) and absence is definitive.
        nests_members = any(
            cs < ms2 and me2 < ce for (cs, ce) in cls_ranges for (_n2, _k2, ms2, me2) in syms if (ms2, me2) != (cs, ce)
        )
        return False if nests_members else None
    return any(n == symbol for (n, _k, _s, _e) in syms)


# Regex patterns for TS/JS symbol definitions (top-level only).
# Each pattern factory takes a symbol name and returns a pattern string.
_TS_SYMBOL_PATTERNS = [
    lambda s: rf"(?:export\s+)?(?:async\s+)?function\s+{re.escape(s)}\s*[\(<]",
    lambda s: rf"(?:export\s+)?(?:const|let|var)\s+{re.escape(s)}\s*[=:]",
    lambda s: rf"(?:export\s+)?(?:abstract\s+)?class\s+{re.escape(s)}\s*(?:extends|implements|<|\{{)",
    lambda s: rf"(?:export\s+)?interface\s+{re.escape(s)}\s*(?:extends|<|\{{)",
    lambda s: rf"(?:export\s+)?type\s+{re.escape(s)}\s*(?:=|<)",
    lambda s: rf"(?:export\s+)?(?:const\s+)?enum\s+{re.escape(s)}\s*\{{",
]

# Regex patterns for Go symbol definitions (top-level and methods).
# Supports: func, type (struct/interface), var/const, and methods (func (r T) Name).
_GO_SYMBOL_PATTERNS = [
    # Top-level function: func Foo(   or   func Foo[T](   or   func Foo(
    lambda s: rf"\bfunc\s+{re.escape(s)}\s*[\[\(]",
    # Method with any receiver: func (r *Type) Foo(   or   func (r Type) Foo(
    lambda s: rf"\bfunc\s+\([^)]*\)\s+{re.escape(s)}\s*[\[\(]",
    # Type definition: type Foo struct   or   type Foo interface   or   type Foo [
    lambda s: rf"\btype\s+{re.escape(s)}\s+(?:struct|interface|\[)",
    # Var declaration: var Foo =   or   var Foo type
    lambda s: rf"\bvar\s+{re.escape(s)}\s+(?:=|\[?\w)",
    # Const declaration: const Foo =   or   const Foo type
    lambda s: rf"\bconst\s+{re.escape(s)}\s+(?:=|\[?\w)",
]


# Regex patterns for Java/Kotlin (JVM) symbol definitions.
# Fallback ONLY — tree-sitter handles Java accurately when the grammar is
# installed; the Kotlin grammar may be absent, so these patterns must be solid
# for Kotlin constructs (fun/class/object/val/var) and Java class/interface/enum.
_JVM_SYMBOL_PATTERNS = [
    # class/interface/enum/object Foo (Java + Kotlin; preceding modifiers ignored)
    lambda s: rf"\b(?:class|interface|enum|object)\s+{re.escape(s)}\b",
    # Kotlin top-level/member function: fun foo( / fun <T> foo(
    lambda s: rf"\bfun\s+{re.escape(s)}\s*[(<]",
    # Kotlin property: val/var foo : | = | , | )
    lambda s: rf"\b(?:val|var)\s+{re.escape(s)}\s*[:=,)]",
]


def _ts_symbol_defined(content: str, symbol: str, lang_id: _LanguageId | None = None) -> bool:
    """Check if a symbol is defined in TS/JS source.

    Primary path: tree-sitter via ``find_all_symbols`` — accurate, ignores
    comment/string occurrences (fixes regex false-positives where a symbol
    mentioned only inside a comment was reported as defined).
    Fallback: regex when tree-sitter is unavailable or parsing failed.

    Supports:
    - Top-level: function, const/let/var, class, interface, type, enum
    - Dotted names: ``ClassName.method`` — structurally confirmed via
      containment of the member's line range inside the class range
    - Bare method names: present anywhere in the flat symbol list
    """
    lid = lang_id if lang_id is not None else _LanguageId.TYPESCRIPT
    syms = _collect_symbols_via_ts(content, lid)
    if syms is not None:
        result = _ts_member_is_defined(syms, symbol)
        # None = inconclusive (shouldn't happen for TS/JS since classes nest
        # methods, but kept defensively in case of exotic inputs). Fall
        # through to regex when inconclusive.
        if result is not None:
            return result

    # ── Regex fallback (tree-sitter unavailable / parse failed / inconclusive) ─
    if "." in symbol:
        # Dotted name: ClassName.method → search inside that class body
        parts = symbol.split(".", 1)
        class_name = parts[0]
        member_name = parts[1]
        _class_header_re = re.compile(
            rf"(?:export\s+)?(?:abstract\s+)?class\s+{re.escape(class_name)}\s*"
            r"(?:extends\s+[^\s,]+(?:\s*,\s*[^\s,]+)*\s*)?"
            r"(?:implements\s+[^\s,]+(?:\s*,\s*[^\s,]+)*\s*)?\{"
        )
        _match = _class_header_re.search(content)
        if not _match:
            return False
        # Class body = the brace-delimited region opened by the header's `{`.
        # Delegated to the literal/comment-aware base scanner: the old
        # hand-rolled depth loop miscounted braces inside strings/comments
        # (e.g. `s = "}"`) and truncated the body, losing methods after it.
        _body_end_abs = find_brace_block_end_offset(content, _match.end())
        _class_body = content[_match.end() : _body_end_abs]
        _method_re = re.compile(
            rf"(?:public|private|protected|static|readonly|async|\s)*\b{re.escape(member_name)}\s*[\(=:<]"
        )
        return bool(_method_re.search(_class_body))

    # Top-level symbols — fast path
    for pat_factory in _TS_SYMBOL_PATTERNS:
        if re.search(pat_factory(symbol), content, re.MULTILINE):
            return True

    # Bare method name fallback: search inside all class bodies.
    # Required because symbol_defined_anywhere is called with bare names
    # (e.g. 'getShape' from op.symbol.split('.')[-1]) for class methods.
    # Class-body extents come from the literal/comment-aware base scanner
    # (same SSOT as the dotted path above); the regex is hoisted out of the
    # loop (it depends only on `symbol`).
    _method_re = re.compile(rf"(?:public|private|protected|static|readonly|async|\s)*\b{re.escape(symbol)}\s*[\(=:<]")
    for _cm in _RE_TS_CLASS_BODY.finditer(content):
        _body_start = _cm.end()
        _body_end_abs = find_brace_block_end_offset(content, _body_start)
        _class_body = content[_body_start:_body_end_abs]
        if _method_re.search(_class_body):
            return True

    return False


def _go_symbol_defined(content: str, symbol: str, lang_id: _LanguageId | None = None) -> bool:
    """Check if a symbol is defined in Go source.

    Primary path: tree-sitter via ``find_all_symbols`` — accurate, ignores
    comment/string occurrences (fixes regex false-positives where a symbol
    mentioned only inside a comment was reported as defined).
    Fallback: regex when tree-sitter is unavailable or parsing failed.

    Supports:
    - Top-level: func Foo, type Foo struct/interface, var Foo, const Foo
    - Methods: func (r Type) MethodName — matched via bare or dotted name
    - Dotted names: TypeName.method — structurally confirmed via containment
      (Go methods are top-level ``func (r T) M()``, but find_all_symbols
      returns them in a flat list whose ranges still let us verify which
      receiver-bearing method belongs to which type).
    """
    lid = lang_id if lang_id is not None else _LanguageId.GO
    syms = _collect_symbols_via_ts(content, lid)
    if syms is not None:
        result = _ts_member_is_defined(syms, symbol)
        # None = inconclusive (Go dotted method not nested in type range).
        # Fall through to regex, which matches the receiver syntax directly.
        if result is not None:
            return result

    # ── Regex fallback (tree-sitter unavailable / parse failed / inconclusive) ─
    if "." in symbol:
        # Dotted name: TypeName.method → search for method with that receiver type
        parts = symbol.split(".", 1)
        type_name = parts[0]
        member_name = parts[1]
        # Go methods are top-level: func (r *TypeName) MethodName(...)
        # or: func (r TypeName) MethodName(...)
        _method_re = re.compile(rf"\bfunc\s+\(\w+\s+\*?{re.escape(type_name)}\s*\)\s+{re.escape(member_name)}\s*[\[\(]")
        return bool(_method_re.search(content))

    # Top-level symbols — check each pattern
    for pat_factory in _GO_SYMBOL_PATTERNS:
        if re.search(pat_factory(symbol), content, re.MULTILINE):
            return True

    # Bare method name fallback: search for func (r *Any) methodName across all receivers
    _method_re = re.compile(rf"\bfunc\s+\([^)]*\)\s+{re.escape(symbol)}\s*[\[\(]")
    return bool(_method_re.search(content))


def _collect_defined_names_via_ts(content: str, lang_id: _LanguageId) -> set | None:
    """Tree-sitter primary path shared by the defined-name collection family.

    Returns the defined-name set, or ``None`` when tree-sitter is unavailable
    or parsing failed (the caller then falls back to language-specific regex).
    """
    syms = _collect_symbols_via_ts(content, lang_id)
    if syms is not None:
        return {n for (n, _k, _s, _e) in syms}
    return None


# ── Defined-name regex fallbacks (tree-sitter unavailable / parse failed) ──
_JVM_DEFINED_NAME_PATTERNS: tuple[str, ...] = (
    # class/interface/enum/object Foo
    r"\b(?:class|interface|enum|object)\s+(\w+)",
    # Kotlin: fun foo( / fun <T> foo(
    r"\bfun\s+(\w+)\s*[(<]",
    # Kotlin property: val/var foo
    r"\b(?:val|var)\s+(\w+)\s*[:=,)]",
)
_TS_DEFINED_NAME_PATTERNS: tuple[str, ...] = (
    # function names (including async, export)
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)",
    # class names (including abstract, export)
    r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
    # const/let/var names (including export)
    r"(?:export\s+)?(?:const|let|var)\s+(\w+)",
    # interface names (including export)
    r"(?:export\s+)?interface\s+(\w+)",
    # type aliases (including export)
    r"(?:export\s+)?type\s+(\w+)\s*=",
    # enum names (including const enum, export)
    r"(?:export\s+)?(?:const\s+)?enum\s+(\w+)",
)
_DEFINED_NAME_REGEX_PATTERNS: dict[_LanguageId, tuple[str, ...]] = {
    # func Foo(...)  /  func (r *T) Method(...) or func (r T) Method(...)
    _LanguageId.GO: (
        r"\bfunc\s+(\w+)\s*[\[\(]",
        r"\bfunc\s+\([^)]*\)\s+(\w+)\s*[\[\(]",
        # type Foo struct / type Foo interface / type Foo [
        r"\btype\s+(\w+)\s+(?:struct|interface|\[)",
        # var Foo = / var Foo type
        r"\bvar\s+(\w+)\s+(?:=|\[?\w)",
        # const Foo = / const Foo type
        r"\bconst\s+(\w+)\s+(?:=|\[?\w)",
    ),
    _LanguageId.JAVA: _JVM_DEFINED_NAME_PATTERNS,
    _LanguageId.KOTLIN: _JVM_DEFINED_NAME_PATTERNS,
    _LanguageId.TYPESCRIPT: _TS_DEFINED_NAME_PATTERNS,
    _LanguageId.JAVASCRIPT: _TS_DEFINED_NAME_PATTERNS,
}


def _collect_defined_names_regex(content: str, lang_id: _LanguageId) -> set:
    """Regex fallback for defined-name collection (tree-sitter unavailable)."""
    names: set = set()
    for pattern in _DEFINED_NAME_REGEX_PATTERNS[lang_id]:
        for m in re.finditer(pattern, content):
            names.add(m.group(1))
    return names


def _collect_defined_names_ts_first(content: str, lang_id: _LanguageId) -> set:
    """Tree-sitter-first defined-name collection for Go/JVM/TS-JS.

    Primary path: tree-sitter via ``find_all_symbols`` — accurate, excludes
    comment/string mentions (fixes regex false-positives).
    Fallback: language-specific regex when tree-sitter is unavailable or
    parsing failed.
    """
    ts_names = _collect_defined_names_via_ts(content, lang_id)
    if ts_names is not None:
        return ts_names
    return _collect_defined_names_regex(content, lang_id)


def _extract_brace_body(content: str, search_from: int) -> str | None:
    """Return the substring of the first balanced ``{...}`` block at/after *search_from*.

    Returns the text BETWEEN the outer braces (exclusive). Returns ``None`` if no
    opening brace is found or the block is unbalanced.
    """
    open_idx = content.find("{", search_from)
    if open_idx < 0:
        return None
    depth = 0
    for i in range(open_idx, len(content)):
        ch = content[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[open_idx + 1 : i]
    return None


def _jvm_class_body(content: str, class_name: str) -> str | None:
    """Return the brace-delimited body of the first JVM type named *class_name*.

    Matches ``class``/``interface``/``enum``/``object`` declarations in both Java
    and Kotlin. Used by the regex fallback of ``_jvm_symbol_defined`` to scope
    dotted member searches to the enclosing type body.
    """
    _header_re = re.compile(rf"\b(?:class|interface|enum|object)\s+{re.escape(class_name)}\b")
    m = _header_re.search(content)
    if not m:
        return None
    return _extract_brace_body(content, m.end())


def _jvm_symbol_defined(content: str, symbol: str, lang_id: _LanguageId | None = None) -> bool:
    """Check if a symbol is defined in Java/Kotlin source.

    Primary path: tree-sitter via ``find_all_symbols`` — accurate, ignores
    comment/string occurrences. Java/Kotlin nest members inside class bodies
    (like TS/JS, unlike Go), so containment-based dotted/bare lookup via
    ``_ts_member_is_defined`` is definitive (no inconclusive ``None`` case).

    Fallback: regex when tree-sitter is unavailable or parsing failed. Notably
    the Kotlin grammar may be absent — these patterns cover Kotlin
    (fun/class/object/val/var) and Java (class/interface/enum) constructs.
    """
    lid = lang_id if lang_id is not None else _LanguageId.JAVA
    syms = _collect_symbols_via_ts(content, lid)
    if syms is not None:
        result = _ts_member_is_defined(syms, symbol)
        if result is not None:
            # Kotlin tree-sitter grammar can mis-parse and miss symbols
            # (e.g. single-line ``object`` declarations).  Only trust
            # positive identification; False → fall through to regex.
            if result is False and lid is _LanguageId.KOTLIN:
                pass  # fall through to regex below
            else:
                return result

    # ── Regex fallback (tree-sitter unavailable / parse failed) ──────────
    # Member regex factory covering Kotlin (fun/val/var) and Java
    # (ReturnType name(). ``fun name(`` also catches Java-style calls loosely,
    # but this path only runs when tree-sitter is unavailable.
    def _member_re(name: str) -> re.Pattern:
        e = re.escape(name)
        return re.compile(
            rf"\bfun\s+{e}\s*[(<]"
            rf"|\b(?:val|var)\s+{e}\s*[:=,)]"
            rf"|\b\w+\s+{e}\s*\("
        )

    if "." in symbol:
        # Dotted name: ClassName.member → search inside that class body
        cls, member = symbol.split(".", 1)
        _body = _jvm_class_body(content, cls)
        if _body is None:
            return False
        return bool(_member_re(member).search(_body))

    # Top-level symbols — check each pattern
    for pat_factory in _JVM_SYMBOL_PATTERNS:
        if re.search(pat_factory(symbol), content, re.MULTILINE):
            return True

    # Bare member name fallback: search inside ALL class bodies (a bare name
    # may be a method/field rather than a top-level def).
    for _cm in _RE_JVM_CLASS_HEADER.finditer(content):
        _body = _extract_brace_body(content, _cm.end())
        if _body and _member_re(symbol).search(_body):
            return True
    return False


def is_python_definition(line: str) -> bool:
    """Check if a line starts a Python function, async function, or class definition."""
    return is_function_def(line) or is_class_def(line)


def is_function_def(line: str) -> bool:
    """Check if a line starts a function or async function definition."""
    stripped = line.lstrip()
    if ":" in stripped:
        before_colon = stripped.split(":")[0].strip()
        if before_colon.isidentifier() and (stripped.startswith(("def ", "async def "))):
            return False
    return stripped.startswith(("def ", "async def "))


def is_class_def(line: str) -> bool:
    """Check if a line starts a class definition."""
    return line.lstrip().startswith("class ")


def is_decorator(line: str) -> bool:
    """Check if a line is a decorator."""
    return bool(_RE_DECORATOR.match(line))


def is_import_boundary(line: str) -> bool:
    """Check if a line marks the end of the import section.

    Returns True for function defs, class defs, and decorators — the
    first non-import, non-blank, non-comment code after module-level imports.
    """
    return is_python_definition(line) or is_decorator(line)


def extract_symbol_name(code_header: str) -> tuple[str | None, str]:
    """Extract symbol name and kind from a code definition header.

    Args:
        code_header: First line of a Python definition (may include whitespace).

    Returns:
        (symbol_name, kind) where kind is "function", "class", or "unknown".
        symbol_name is None if the header doesn't match any pattern.

    Examples:
        >>> extract_symbol_name("def foo(x, y):")
        ('foo', 'function')
        >>> extract_symbol_name("async def bar():")
        ('bar', 'function')
        >>> extract_symbol_name("class MyClass(Base):")
        ('MyClass', 'class')
        >>> extract_symbol_name("x = 42")
        (None, 'unknown')
    """
    header = code_header.lstrip()
    m = _RE_FUNC_DEF.match(header)
    if m:
        return (m.group(2), "function")
    m = _RE_CLASS_DEF.match(header)
    if m:
        return (m.group(2).strip(), "class")
    return (None, "unknown")


@dataclass
class DefinitionInfo:
    """Structured info about a Python definition found via AST."""

    name: str
    kind: str
    start_line: int
    end_line: int
    col_offset: int = 0
    decorators: list = field(default_factory=list)
    parent_class: str = ""
    qualified_name: str = ""  # "Parent.child" for methods, "name" for top-level


def parse_definitions(source: str) -> list[DefinitionInfo]:
    """Parse all top-level and nested definitions from Python source.

    Handles multi-line defs, decorator stacks, nested classes, and async
    functions — all cases that line-level heuristics miss.

    Uses tree-sitter when available (primary path), falls back to AST.

    Args:
        source: Complete Python source code.

    Returns:
        List of DefinitionInfo, sorted by start_line.
        Empty list if source fails to parse.
    """
    # Primary: tree-sitter (accurate, multi-language compatible).
    # Validate results against AST parser — tree-sitter is lenient and may
    # return partial results for syntactically invalid code (e.g. "def f(:").
    # The AST fallback correctly returns [] on SyntaxError, so we match that
    # contract even when tree-sitter "succeeds" on broken source.
    if _HAS_TS:
        try:
            # Quick AST validation: if source has a syntax error, skip
            # tree-sitter results entirely to match AST behavior.
            try:
                ast.parse(source)
            except SyntaxError:
                return []  # Match AST behavior: syntax error → empty list
            ts_results = _walk_definitions_py(source)
            if ts_results:
                return ts_results
        except Exception:
            logger.debug("parse_definitions: tree-sitter failed, falling back to AST", exc_info=True)

    # Fallback: Python AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.debug("parse_definitions: SyntaxError, returning empty")
        return []
    results: list[DefinitionInfo] = []
    _walk_definitions(tree, results, parent_class=None)
    results.sort(key=lambda d: d.start_line)
    return results


# Compound statements that DO NOT introduce new lexical scope.  Function
# bodies inside these are still module-level (or class-level) — Python's
# name resolution treats ``if FLAG: def f(): ...`` exactly like ``def f():
# ...`` for module-level binding.
#
# CANONICAL DEFINITION: this is the single source of truth for the
# wrapper-aware module-scope walker family.  Centralising the helper
# here makes a duplicate occurrence physically impossible (the former
# placement_contract.py copy was removed together with the PCL module).
NON_SCOPE_COMPOUND_STMTS: tuple = (
    ast.If,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
)
# Backwards-compat alias for the in-module callers below; new external
# callers should use the public name above.
_NON_SCOPE_COMPOUND_STMTS = NON_SCOPE_COMPOUND_STMTS


def iter_module_scope_nodes(tree: ast.AST):
    """Yield every node at module scope (preorder DFS).

    "Module scope" = not nested inside any FunctionDef / AsyncFunctionDef /
    ClassDef.  Both stmt-level wrappers (``if``/``try``/``with``/``for``/
    ``while``) AND expression-level constructs (BoolOp, Call args, …) are
    entered, so callers can find walrus operators and other deeply
    nested module-scope bindings.

    Iteration starts from ``tree.body`` (or ``tree``'s direct children
    if it's not a Module) — the root tree itself is NOT yielded, callers
    receive the same view as ``ast.iter_child_nodes`` did pre-refactor
    plus the wrapper-aware descent.

    Stable preorder: source-order children, depth-first.  Two consecutive
    runs over the same source produce identical sequences.

    Used by:
      - code_structure_utils.find_import_boundary_ast,
        symbol_exists_in_module
    """
    initial = list(ast.iter_child_nodes(tree))
    stack = list(reversed(initial))
    while stack:
        node = stack.pop()
        yield node
        # Function/class bodies are NOT module scope — their bindings are
        # local to that scope and should not be reachable through this
        # walker.  Caller is still free to record the def/class name itself
        # (which has been yielded above) before deciding to skip the body.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        children = list(ast.iter_child_nodes(node))
        stack.extend(reversed(children))


def _walk_definitions(node: ast.AST, out: list[DefinitionInfo], parent_class: str | None) -> None:
    """Recursively collect definitions from AST nodes (legacy AST fallback).

    Walks ``ast.iter_child_nodes`` for direct hits AND descends into
    non-scope-creating compound statements (``if``/``try``/``with``/
    ``for``/``while``) so that wrapped definitions are visible.  Without
    that descent, definitions inside ``if TYPE_CHECKING:``, optional-import
    ``try-except``, version-gated branches, and the like are silently
      invisible to the patch engine.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async_function" if isinstance(child, ast.AsyncFunctionDef) else "function"
            decorators = [_decorator_name(d) for d in child.decorator_list]
            start = child.decorator_list[0].lineno if child.decorator_list else child.lineno
            end = getattr(child, "end_lineno", child.lineno)
            _qual_name = f"{parent_class}.{child.name}" if parent_class else child.name
            out.append(
                DefinitionInfo(
                    name=child.name,
                    kind=kind,
                    start_line=start,
                    end_line=end,
                    col_offset=child.col_offset,
                    decorators=decorators,
                    parent_class=parent_class or "",
                    qualified_name=_qual_name,
                )
            )
            _walk_definitions(child, out, parent_class=parent_class)
        elif isinstance(child, ast.ClassDef):
            decorators = [_decorator_name(d) for d in child.decorator_list]
            start = child.decorator_list[0].lineno if child.decorator_list else child.lineno
            end = getattr(child, "end_lineno", child.lineno)
            _qual_name = f"{parent_class}.{child.name}" if parent_class else child.name
            out.append(
                DefinitionInfo(
                    name=child.name,
                    kind="class",
                    start_line=start,
                    end_line=end,
                    col_offset=child.col_offset,
                    decorators=decorators,
                    parent_class=parent_class or "",
                    qualified_name=_qual_name,
                )
            )
            new_parent = child.name if parent_class is None else f"{parent_class}.{child.name}"
            _walk_definitions(child, out, parent_class=new_parent)
        elif isinstance(child, (ast.Assign, ast.AnnAssign)):
            # Module-level variable assignments — collect simple Name targets
            # so SPEC_VALIDATION can find constants like TYPE_CONFIG = {...}
            # that are not wrapped in FunctionDef/ClassDef.
            _v_targets: list[ast.expr] = []
            _targets: list[ast.expr] = []
            if isinstance(child, ast.Assign):
                _targets = [t for t in child.targets if isinstance(t, ast.Name)]
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                _targets = [child.target]
            for _t in _targets:
                _name: str = _t.id  # type: ignore[attr-defined]
                _start: int = child.lineno
                _end: int = getattr(child, "end_lineno", _start)
                _qual: str = f"{parent_class}.{_name}" if parent_class else _name
                out.append(
                    DefinitionInfo(
                        name=_name,
                        kind="variable",
                        start_line=_start,
                        end_line=_end,
                        col_offset=child.col_offset,
                        parent_class=parent_class or "",
                        qualified_name=_qual,
                    )
                )
            # Recurse into compound RHS (e.g. dict literals containing
            # inner def/class — unusual but valid).
            _walk_definitions(child, out, parent_class=parent_class)
        elif isinstance(child, _NON_SCOPE_COMPOUND_STMTS):
            # Wrapper stmt — the wrapper itself is not a definition, but its
            # body still binds at the enclosing scope.  Recurse with the
            # SAME parent_class (wrapper does not change class membership).
            _walk_definitions(child, out, parent_class=parent_class)


def _collect_assign_targets(node, out_set: set, code_bytes: bytes) -> None:
    """Extract assignment target names from a tree-sitter statement node.

    Accepts EITHER an ``expression_statement`` wrapper or a bare
    ``assignment``/``augmented_assignment``. The two Python grammars disagree on
    whether the wrapper exists (standalone ``tree-sitter-python`` emits it, the
    ``tree-sitter-language-pack`` bundle does not), and only the pack is a
    declared dependency — so wrapper-only handling collected nothing on a real
    install.

    Handles:
    - ``X = 42``              (assignment → identifier)
    - ``X: int = 42``         (assignment → identifier + type)
    - ``X = Y = 42``          (chained assignment → nested assignment)
    - ``X: int``              (type-only assignment)
    - ``X += 1``              (augmented assignment)

    Accepting the bare form also fixes chained assignment, which the docstring
    always claimed but the code never did: the recursive call passes the INNER
    ``assignment`` node, and the old wrapper-only loop then searched that node's
    children for another ``assignment`` and found none, dropping ``Y``.
    """
    children = (node,) if node.type in ("assignment", "augmented_assignment") else node.children
    for child in children:
        if child.type == "assignment":
            # Direct target identifier
            for sub in child.children:
                if sub.type == "identifier":
                    out_set.add(sub.text.decode("utf-8"))
                elif sub.type == "assignment":
                    # Chained assignment: X = Y = 42
                    _collect_assign_targets(sub, out_set, code_bytes)
        elif child.type == "augmented_assignment":
            for sub in child.children:
                if sub.type == "identifier":
                    out_set.add(sub.text.decode("utf-8"))


def _walk_definitions_py(source: str) -> list[DefinitionInfo]:
    """Collect Python definitions using tree-sitter (primary path when available).

    Python-only: parses with the ``"python"`` grammar. Despite the legacy
    ``_ts`` suffix this has always been Python-only; renamed to ``_py`` to
    avoid implying multi-language tree-sitter support.

    Returns the same ``List[DefinitionInfo]`` as the legacy AST walker
    so callers see an identical interface.

    Handles decorated definitions: the start_line is set to the first
    decorator line (matching AST behavior for decorator stacks).
    """
    from .languages.tree_sitter_utils import (
        _SYMBOL_NODE_TYPES,
        _extract_name,
        _node_kind,
        parse_to_tree,
    )

    tree = parse_to_tree(source, "python")
    if tree is None:
        return []

    code_bytes = source.encode("utf-8")
    results: list[DefinitionInfo] = []
    seen: set = set()

    def _collect(node, parent_class: str | None = None) -> None:
        if node.type in _SYMBOL_NODE_TYPES:
            # ── kind resolution ──────────────────────────────────────────────
            kind = _node_kind(node)
            # function_definition with async child → "async_function"
            if node.type == "function_definition":
                for _child in node.children:
                    if _child.type == "async":
                        kind = "async_function"
                        break

            name = _extract_name(node)
            if name:
                start = node.start_point.row + 1
                end = node.end_point.row + 1
                _qual_name = f"{parent_class}.{name}" if parent_class else name
                _col_offset = node.start_point.column

                decorators: list[str] = []
                if node.type == "decorated_definition":
                    for child in node.children:
                        if child.type == "decorator":
                            d_text = code_bytes[child.start_byte : child.end_byte].decode("utf-8")
                            decorators.append(d_text.lstrip("@").strip())
                        elif child.type in ("function_definition", "class_definition"):
                            # DON'T overwrite start — node.start_point is already
                            # the first decorator line. Only update end_line from
                            # the inner function/class node.
                            end = child.end_point.row + 1
                            _col_offset = child.start_point.column
                else:
                    # For non-decorated definitions, record col_offset
                    _col_offset = node.start_point.column

                dedup_key = (name, start, end, _qual_name)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    results.append(
                        DefinitionInfo(
                            name=name,
                            kind=kind,
                            start_line=start,
                            end_line=end,
                            col_offset=_col_offset,
                            decorators=decorators,
                            parent_class=parent_class or "",
                            qualified_name=_qual_name,
                        )
                    )

                # Container types (class_definition): descend to find methods
                if node.type in ("class_definition",):
                    new_parent = name if parent_class is None else f"{parent_class}.{name}"
                    for child in node.children:
                        if child.type == "block":
                            for block_child in child.children:
                                _collect(block_child, parent_class=new_parent)
                    return

                # function_definition: continue descending to find nested
                # functions/classes inside the body.
                if node.type == "function_definition":
                    pass  # fall through to generic child walk
                elif node.type == "decorated_definition":
                    return  # handled above; don't double-record inner def
                else:
                    return

        for child in node.children:
            _collect(child, parent_class=parent_class)

    _collect(tree.root_node)
    results.sort(key=lambda d: d.start_line)
    return results


def _decorator_name(node: ast.expr) -> str:
    """Extract decorator name from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.dump(node)
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return "<unknown>"


def find_definition_at_line(source: str, line: int) -> DefinitionInfo | None:
    """Find the definition that spans a given line number.

    Args:
        source: Complete Python source code.
        line: 1-based line number.

    Returns:
        The innermost DefinitionInfo containing that line, or None.
    """
    defs = parse_definitions(source)
    best: DefinitionInfo | None = None
    for d in defs:
        if d.start_line <= line <= d.end_line and (best is None or d.start_line >= best.start_line):
            best = d
    return best


def find_import_boundary_ast(source: str) -> int:
    """Find the line number where module-level imports end.

    Returns the 1-based line of the first non-import statement at module
    level.  Returns 1 if there are no imports, or len(lines)+1 if the
    entire file is imports.

    Walks both direct module children AND non-scope-creating compound
    statements (``try``/``if``/``with``/``for``/``while``) so that
    optional-dep imports (``try: import yaml; except: yaml = None``)
    and version-gated imports (``if sys.version_info: from x import Y``)
    are counted toward the import region.  Without this, the boundary
    truncated above any later regular ``import`` line below the wrapper.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 1

    last_import_end = 0
    for node in iter_module_scope_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = getattr(node, "end_lineno", node.lineno)
            last_import_end = max(last_import_end, end)
        elif isinstance(node, ast.Expr) and isinstance(
            getattr(node, "value", None),
            ast.Constant,
        ):
            # Module-level docstring (or other string-constant expression
            # at module top).  ``ast.Str`` was removed in Python 3.12; the
            # forward-compat path is ``ast.Constant``.  Treat as part of
            # the import region so a docstring above imports doesn't
            # short-circuit the boundary calculation.
            last_import_end = max(last_import_end, getattr(node, "end_lineno", node.lineno))
    return last_import_end + 1 if last_import_end > 0 else 1


def _same_import_relativity(query_module: str, node: ast.ImportFrom) -> bool:
    """Return True if *query_module* and *node* have the same relative-ness.

    A relative import (``from .foo import X``, level > 0) is semantically
    different from an absolute import (``from foo import X``, level == 0).
    This helper enforces they are NOT treated as matching during
    ``is_module_level_import_present``, regardless of module-name overlap
    after dot-stripping.
    """
    query_is_relative = query_module.startswith(".")
    node_is_relative = (node.level or 0) > 0
    return query_is_relative == node_is_relative


def _relative_depth_matches(query_module: str, node: ast.ImportFrom) -> bool:
    """Check that relative-import depth is compatible.

    ``from .foo import X`` (depth=1) matches ``from .foo import X`` (depth=1)
    but NOT ``from ..foo import X`` (depth=2).

    When the query is absolute (no leading dots), this check is strict:
    the node must also be absolute (level == 0).  The leaf-name fallback
    (in ``is_module_level_import_present``) handles the case where an
    absolute dotted path should match a relative import's leaf name.
    """
    query_depth = sum(1 for ch in query_module if ch == ".") if query_module.startswith(".") else 0
    node_depth = node.level or 0
    if query_depth == 0 and node_depth == 0:
        return True
    if query_depth > 0 and node_depth > 0:
        return query_depth == node_depth
    return False


def is_module_level_import_present(
    source_or_tree: str | ast.AST,
    module: str,
    name: str | None = None,
) -> bool:
    """Return True if `from {module} import {name}` is present at module scope.

    Single source of truth for "is this import already available at module
    level?" — used by both insert_import idempotency and the import_exists
    intent assertion. Routing both checks through the same AST walk
    prevents the two from disagreeing (raw-text substring matched a
    function-local import or docstring → handler said already_satisfied
    while the AST-based assertion correctly blocked).

    - module: dotted module path (e.g. "collections", "typing").
      Relative imports MUST use leading dots (e.g. ".foo", "..bar.baz")
      and will ONLY match source imports with the same relative depth.
      Absolute imports (no leading dot) match absolute imports only.
    - name: optional imported symbol.  When omitted, matches either a
      whole-module ``import {module}`` or any ``from {module} import …``.

    Function-internal imports, docstrings, comments, and string literals
    are intentionally NOT counted — they do not make the symbol available
    at module scope.
    """
    if isinstance(source_or_tree, str):
        try:
            tree = ast.parse(source_or_tree)
        except SyntaxError:
            return False
    else:
        tree = source_or_tree

    module_normalized = module.lstrip(".")
    # Leaf-name fallback (used below): when assertion uses an absolute dotted path
    # (e.g. "playground.galaga.constants") while the source uses a relative
    # import (e.g. ".constants"), compare the last component.
    module_leaf = module_normalized.split(".")[-1] if "." in module_normalized else ""

    for node in iter_module_scope_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Alias-aware: ``import X as Y`` binds Y, not X — only a
                # same-name binding (or an explicit ``as X``) satisfies the query.
                if (alias.name == module and alias.asname is None) or alias.asname == module:
                    return True
        elif isinstance(node, ast.ImportFrom):
            node_module = node.module or ""
            node_module_leaf = node_module.split(".")[-1] if "." in node_module else node_module
            # ── Three-tier matching ───────────────────────────────
            # 1. Exact: node_module == module (e.g. "foo" == "foo")
            # 2. Normalized: after stripping dots, but only when
            #    relative-ness matches (Galaga bug fix: ".foo" must
            #    NOT match absolute "foo").
            # 3. Leaf-name: dotted path leaf match (e.g. "constants"
            #    in "playground.galaga.constants" matches ".constants").
            #    This intentionally crosses absolute↔relative boundary.
            #
            # NOTE: Tier 3 (leaf-name) deliberately skips the
            # _same_import_relativity check because it's designed
            # to match absolute dotted paths against relative imports.
            # NOTE: Tiers 1 and 2 DO require relative-ness to match
            # (e.g. query "foo" must NOT match source ".foo" even though node_module == "foo").
            if (
                (node_module == module and _same_import_relativity(module, node))
                or (
                    _same_import_relativity(module, node)
                    and _relative_depth_matches(module, node)
                    and node_module == module_normalized
                )
                or (module_leaf and node_module_leaf == module_leaf)
            ):
                if not name:
                    return True
                for alias in node.names:
                    # Alias-aware: ``from X import A as B`` binds B, not A — the
                    # requested name must actually be bound at module scope.
                    if (alias.name == name and alias.asname is None) or alias.asname == name:
                        return True
    return False


def symbol_exists_in_module(source: str, symbol: str, file_path: str | None = None) -> bool:
    """Return True if *symbol* is defined or assigned at module level in *source*.

    Covers:
    - ``def symbol(...)`` / ``async def symbol(...)``
    - ``class symbol(...)``
    - Module-level assignments: ``symbol = ...`` or ``symbol: type = ...``
    - Same set of definitions wrapped in non-scope-creating compound
      statements (``if TYPE_CHECKING:``, ``try-except`` for optional
      deps, ``if sys.version_info >= ...:`` version gates, etc.) — these
      bindings are still module-scope at runtime.

    For TS/JS/Go files, pass ``file_path`` to enable tree-sitter-based
    module-level detection (regex fallback). Dotted names like
    ``ClassName.method`` are NOT module-level and return False.

    Uses tree-sitter when available, falls back to AST.
    """
    if not symbol or not source:
        return False

    lang_id = _LanguageId.from_path(file_path) if file_path else None

    # ── TS/JS/Go: tree-sitter-first module-level detection ──────────────
    # find_all_symbols returns a flat (name, kind, start, end) list where
    # methods/fields are nested inside their enclosing class range.
    # "Module-level" = the symbol exists but is NOT contained inside any
    # class/type range, AND is not a dotted member name.
    if lang_id == _LanguageId.GO:
        # Go needs special handling: find_all_symbols collapses methods and
        # functions into one "function" kind, so a receiver method would be
        # mistaken for module-level. Walk the raw tree to exclude methods.
        if "." in symbol:
            return False
        go_result = _go_module_level_symbol_exists(source, symbol)
        if go_result is not None:
            return go_result
        # tree-sitter unavailable → fall through to AST (conservative False).
    elif lang_id in (_LanguageId.TYPESCRIPT, _LanguageId.JAVASCRIPT):
        if "." in symbol:
            # Dotted names are class members, never module-level.
            return False
        syms = _collect_symbols_via_ts(source, lang_id)
        if syms is not None:
            # A symbol is module-level iff no OTHER symbol range strictly
            # contains its own (start, end) — nested methods/fields are
            # contained inside their class/type range, top-level symbols are not.
            for n, _k, s, e in syms:
                if n != symbol:
                    continue
                contained = any(cs < s and e < ce for (_n2, _k2, cs, ce) in syms if (_n2, cs, ce) != (n, s, e))
                if not contained:
                    return True
            return False
        # tree-sitter unavailable → fall through to AST (returns False for
        # non-Python, which is the pre-existing conservative behaviour).

    # Pre-validate: if source has a syntax error, skip tree-sitter results
    # (tree-sitter may return partial definitions for broken code).
    _has_syntax_error = False
    try:
        ast.parse(source)
    except SyntaxError:
        _has_syntax_error = True

    # Primary: tree-sitter (fast path for valid Python only).
    if _HAS_TS and not _has_syntax_error:
        with contextlib.suppress(RecursionError, TypeError, ValueError, IndexError):  # TS walk → AST fallback
            ts_defs = _walk_definitions_py(source)
            for d in ts_defs:
                if d.name == symbol and (not d.parent_class):
                    return True
            # Tree-sitter didn't find the symbol — fall through to AST
            # which catches assignments and handles syntax errors correctly.
    # Fallback: AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in iter_module_scope_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return True
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
            return True
    return False


def symbol_defined_anywhere(source: str, symbol: str, file_path: str | None = None) -> bool:
    """Return True if *symbol* is defined anywhere in *source* (any nesting level).

    For TS/JS/Go files, pass ``file_path`` to enable tree-sitter-based
    detection (regex fallback). Tree-sitter is accurate and ignores
    comment/string occurrences, fixing false-positives where a symbol
    mentioned only inside a comment was reported as defined.

    Unlike ``symbol_exists_in_module`` which only checks module-level nodes,
    this function walks the full tree to find functions, async functions, and
    classes defined at any depth — including class methods and nested functions.

    Uses tree-sitter for Python when available, AST fallback.

    Use this when you need to check whether a symbol exists in a file regardless
    of whether it is a module-level function or a class method.
    """
    if not symbol or not source:
        return False

    lang_id = _LanguageId.from_path(file_path) if file_path else None

    # TS/JS: tree-sitter-first detection (find_all_symbols supports both,
    # and distinguishes the grammar via LanguageId.value).
    if lang_id in (_LanguageId.TYPESCRIPT, _LanguageId.JAVASCRIPT):
        return _ts_symbol_defined(source, symbol, lang_id)

    # Go: tree-sitter-first detection
    if lang_id == _LanguageId.GO:
        return _go_symbol_defined(source, symbol, lang_id)

    # Java/Kotlin (JVM): tree-sitter-first detection. Members nest inside class
    # bodies (like TS/JS), so containment-based lookup is definitive.
    if lang_id in (_LanguageId.JAVA, _LanguageId.KOTLIN):
        return _jvm_symbol_defined(source, symbol, lang_id)

    # Primary: tree-sitter (catches all nesting levels)
    # Only trust tree-sitter for Python; for unknown languages, treat empty
    # results as inconclusive rather than definitive "not found".
    if _HAS_TS:
        try:
            ts_defs = _walk_definitions_py(source)
            if any(d.name == symbol for d in ts_defs):
                return True
            # Tree-sitter collects only function/class/decorated nodes, so a
            # miss is NOT definitive: module-level or nested assignments
            # (``X = 1`` / ``X: int = 2``) are invisible to it. Fall through
            # to the AST walk — the reference implementation — which treats
            # Assign/AnnAssign as definitions (and, for unknown non-Python
            # languages, degrades to the conservative True on SyntaxError).
        except Exception:
            logger.debug("symbol_defined_anywhere: tree-sitter failed, falling back to AST", exc_info=True)

    # Fallback: AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Non-Python source that tree-sitter couldn't handle.
        # Conservative: we cannot verify the symbol exists, but we also
        # cannot prove it doesn't — assume it does to avoid false-positive
        # blocking (e.g., "all required symbols must remain present" for Go).
        return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return True
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == symbol:
                    return True
        elif isinstance(node, ast.AnnAssign):
            t = getattr(node, "target", None)
            if isinstance(t, ast.Name) and t.id == symbol:
                return True
    return False


def collect_defined_names(source: str, file_path: str | None = None) -> set:
    """Collect all names accessible at module scope from source.

    For TS/JS files, pass ``file_path`` to enable tree-sitter-based detection
    (regex fallback). Tree-sitter is accurate and ignores comment/string
    occurrences.
    For Go files, pass ``file_path`` to enable tree-sitter-based detection.

    Includes:
    - FunctionDef / AsyncFunctionDef names
    - ClassDef names
    - Assignment target names (Assign, AnnAssign)
    - Imported names (from X import a, b  →  {a, b};  import X  →  {X})

    Uses tree-sitter for Python when available, AST fallback.

    Import names are included so callers (e.g. DecompositionGuard) treat
    already-imported names as "accessible" rather than "missing", preventing
    spurious INSERT_AFTER_SYMBOL injection for stdlib/third-party names.

    Returns empty set on parse error.
    """
    # Go / JVM / TS-JS: tree-sitter-first detection (regex fallback)
    if file_path:
        lang_id = _LanguageId.from_path(file_path)
        if lang_id in _DEFINED_NAME_REGEX_PATTERNS:
            return _collect_defined_names_ts_first(source, lang_id)

    names: set = set()

    # Primary: tree-sitter (Python grammar only — skip for non-Python)
    lang_id = _LanguageId.from_path(file_path) if file_path else None
    if lang_id in (None, _LanguageId.PYTHON) and _HAS_TS:
        try:
            from .languages.tree_sitter_utils import (
                _SYMBOL_NODE_TYPES,
                _extract_name,
                parse_to_tree,
            )

            tree = parse_to_tree(source, "python")
            if tree is not None:
                code_bytes = source.encode("utf-8")

                def _collect(node) -> None:
                    if node.type in _SYMBOL_NODE_TYPES:
                        name = _extract_name(node)
                        if name:
                            names.add(name)
                            if node.type not in ("class_definition", "decorated_definition"):
                                return
                    # Module-level assignments: X = 42, X: int = 42
                    #
                    # Two node shapes, one meaning: the standalone
                    # tree-sitter-python grammar wraps a statement-position
                    # assignment in `expression_statement`, the grammar bundled in
                    # tree-sitter-language-pack does not. Only the pack is a
                    # declared dependency, so matching the wrapper alone dropped
                    # every module-level name on a real install.
                    # `augmented_assignment` is listed for the same reason as
                    # `assignment`: under the wrapper grammar it is reached THROUGH
                    # `expression_statement`, but the pack grammar hangs it straight
                    # off `module`, where it would otherwise be walked past.
                    elif node.type in (
                        "expression_statement",
                        "assignment",
                        "augmented_assignment",
                    ):
                        _collect_assign_targets(node, names, code_bytes)
                        return
                    for child in node.children:
                        _collect(child)

                _collect(tree.root_node)
                # Also return via AST fallback for import names, which
                # tree-sitter doesn't track. But if tree-sitter found
                # something, return early for performance.
                if names:
                    # Still missing import names — merge from AST
                    with contextlib.suppress(SyntaxError):
                        _tree = ast.parse(source)
                        for _node in ast.walk(_tree):
                            if isinstance(_node, (ast.ImportFrom, ast.Import)):
                                for alias in _node.names:
                                    names.add(alias.asname if alias.asname else alias.name.split(".")[0])
                    return names
                # Nothing collected — e.g. an imports-only source: ``import os``
                # is not a symbol node, so returning the empty set here would
                # LOSE the imported names the contract promises (``import X``
                # → ``{X}``). Fall through to the AST fallback — the reference
                # implementation — instead of returning the empty set early.
        except Exception:
            logger.debug("collect_defined_names: tree-sitter failed, falling back to AST", exc_info=True)

    # Fallback: AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign):
            _ann_target = getattr(node, "target", None)
            if isinstance(_ann_target, ast.Name):
                names.add(_ann_target.id)
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
    return names


# ── FunctionSignature (param_sig / return_type separation) ─────────────────


@dataclass
class FunctionSignature:
    """Separated function signature: param_sig + return_type.

    ``param_sig`` is the parameter part (comma-separated with annotations).
    ``return_type`` is the return annotation string (empty string if none).

    This separation allows signature-change consumers to distinguish
    param-only changes from return-type-only changes.
    """

    param_sig: str = ""
    return_type: str = ""

    @property
    def canonical(self) -> str:
        """Return the combined canonical form ``(param_sig)->return_type``.

        This is identical to the legacy ``extract_function_signature`` output
        so that existing callers continue to work unchanged.
        """
        return f"({self.param_sig})->{self.return_type}"

    def __bool__(self) -> bool:
        """A non-empty signature has at least a param_sig or return_type."""
        return bool(self.param_sig) or bool(self.return_type)


def extract_function_signature_detailed(source: str, symbol_name: str) -> FunctionSignature | None:
    """Return a ``FunctionSignature`` with separate ``param_sig`` / ``return_type``.

    Like ``extract_function_signature`` but returns structured data so that
    callers can distinguish param-only changes from return-type-only changes.

    Uses Python AST (tree-sitter query path removed — QueryCapture objects
    don't expose tree-sitter node children, making parameter extraction
    impossible without re-parsing).

    Returns ``None`` if the symbol is not found or *source* cannot be parsed.
    """
    bare = symbol_name.rsplit(".", maxsplit=1)[-1]

    # AST-based extraction (primary path — always correct)
    with contextlib.suppress(SyntaxError, ValueError, RecursionError):  # broken source → None
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == bare:
                parts: list[str] = []
                all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                # Index at which keyword-only args begin (after posonly + args).
                kwonly_start = len(node.args.posonlyargs) + len(node.args.args)
                for i, arg in enumerate(all_args):
                    ann = ast.unparse(arg.annotation) if arg.annotation else ""
                    # A bare '*' separator marks the start of keyword-only args.
                    # Omitting it collapses ``def f(a, *, b)`` and ``def f(a, b)``
                    # into the same signature, hiding API-changing edits.  When a
                    # ``*vararg`` is present its leading '*' already serves as the
                    # separator, so the bare marker is only needed otherwise.
                    if i == kwonly_start and node.args.kwonlyargs and not node.args.vararg:
                        parts.append("*")
                    parts.append(f"{arg.arg}:{ann}")
                if node.args.vararg:
                    ann = ast.unparse(node.args.vararg.annotation) if node.args.vararg.annotation else ""
                    parts.append(f"*{node.args.vararg.arg}:{ann}")
                if node.args.kwarg:
                    ann = ast.unparse(node.args.kwarg.annotation) if node.args.kwarg.annotation else ""
                    parts.append(f"**{node.args.kwarg.arg}:{ann}")
                ret = ast.unparse(node.returns) if node.returns else ""
                return FunctionSignature(param_sig=",".join(parts), return_type=ret)
    return None


def extract_function_signature(source: str, symbol_name: str) -> str | None:
    """Return a canonical signature string for *symbol_name* in *source*.

    Delegates to ``extract_function_signature_detailed`` and returns the
    combined canonical form ``(param_sig)->return_type`` for backward
    compatibility.

    Used to detect whether a function's public interface changed between plan
    and execution time.  Only parameter names + annotations + return annotation
    are included; defaults and body are ignored so pure-refactor changes that
    keep the same public API produce an identical signature string.

    Returns None if the symbol is not found or *source* cannot be parsed.
    """
    detailed = extract_function_signature_detailed(source, symbol_name)
    if detailed is None:
        return None
    return detailed.canonical


def find_last_top_level_def(path: str) -> str | None:
    """Return the name of the last top-level function/class in *path*.

    Used as the DPB anchor principle: when an INSERT_AFTER_SYMBOL anchor is
    absent or invalid, redirect to the last top-level definition in the file.

    Supports Python (via AST) and TS/JS (via regex).
    Returns None if the file cannot be read/parsed or has no top-level defs.
    """
    import os as _os

    if not path or not _os.path.isfile(path):
        return None
    # Read separately from the analysis: a failed open previously left ``_src``
    # undefined inside the suppress, surfacing a NameError instead of None.
    try:
        with open(path, encoding="utf-8", errors="replace") as _f:
            _src = _f.read()
    except OSError:
        logger.debug("find_last_top_level_def: cannot read %s", path)
        return None
    # TS/JS: tree-sitter first (accurate multi-construct detection),
    # regex fallback when JS/TS grammars are unavailable.
    if path.endswith((".ts", ".tsx", ".js", ".jsx")):
        # _lang is only consumed on the tree-sitter path (regex fallback
        # ignores it), and _ts_grammar_key_for_path is only defined when
        # tree-sitter imported cleanly — guard to preserve the no-TS path.
        _lang = (_ts_grammar_key_for_path(path) if _HAS_TS else "typescript") or "typescript"
        _last = _find_last_top_level_ts(_src, _lang)
        if _last is not None:
            return _last
        # fall through only if tree-sitter returned nothing AND there was
        # no parseable symbol; the regex fallback inside the helper covers
        # the grammar-missing case, so reaching here means truly empty.
        return None
    # Python: use AST
    _defs = parse_definitions(_src)
    _top = [d for d in _defs if d.col_offset == 0]
    return _top[-1].name if _top else None


def _find_last_top_level_ts(_src: str, _lang: str) -> str | None:
    """Return the last top-level TS/JS symbol name (tree-sitter, regex fallback).

    ``find_all_symbols`` returns ALL symbols — top-level AND nested (class
    methods, inner functions) — because it descends into container nodes.
    We therefore filter to those whose start line has no leading indentation
    in the source, which selects true top-level definitions. Decorators,
    ``export``/``async`` modifiers and generics all sit at column 0 on the
    def line (or its decorator line), so they are retained correctly.

    Falls back to a single-pass regex when tree-sitter or the JS/TS grammar
    is unavailable. The regex only catches ``function``/``class``/``const``
    declarations; tree-sitter additionally covers ``enum``, ``interface``,
    ``type``, generators, and decorated exports.
    """
    # Primary: tree-sitter.
    if _HAS_TS:
        try:
            _syms = _ts_find_all_symbols(_src, _lang)
            if _syms:
                _lines = _src.splitlines()
                _top: list[tuple[int, str]] = []
                for _name, _kind, _start, _end in _syms:
                    _idx = _start - 1  # 1-indexed → 0-indexed
                    if 0 <= _idx < len(_lines):
                        _line = _lines[_idx]
                        # Top-level: the def line (or decorator line) starts at
                        # column 0. Empty lines never appear as a symbol start.
                        if not _line[:1].isspace():
                            _top.append((_start, _name))
                if _top:
                    _top.sort()
                    return _top[-1][1]
        except Exception:
            logger.debug(
                "find_last_top_level_def: tree-sitter failed, falling back to regex",
                exc_info=True,
            )

    # Fallback: regex (single pass). Group 1 = named function/class,
    # group 2 = const arrow/expression assignment.
    _last_name: str | None = None
    for _m in _RE_TS_LAST_TOP_LEVEL_DEF.finditer(_src):
        _last_name = _m.group(1) or _m.group(2)
    return _last_name
