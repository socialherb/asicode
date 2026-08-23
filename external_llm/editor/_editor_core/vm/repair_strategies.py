"""repair_strategies.py — Deterministic repair strategies per language.

Each strategy:
- Takes (code, error, classification) -> List[PrimitiveOp] or None
- Is fully deterministic (no LLM)
- Returns primitive ops that the VM can execute via ASTRewriter

Strategies are dispatched by FailureType via the RepairRegistry.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from external_llm.editor._editor_core.vm.classification import Classification
from external_llm.editor._editor_core.vm.failure_classifier import FailureType
from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor.primitives.models import PrimitiveKind, PrimitiveOp

# ── Utility ───────────────────────────────────────────────────────────


def _make_raw_replacement(code: str) -> list[PrimitiveOp]:
    """Wrap a full code replacement as a raw-replacement op."""
    return [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_STATEMENT,
            payload={"__raw_code__": code},
        )
    ]


def _get_indent(line: str) -> str:
    indent = ""
    for ch in line:
        if ch in (" ", "\t"):
            indent += ch
        else:
            break
    return indent


def _trim_call_arguments(
    lines: list[str],
    idx: int,
    *,
    keep_first: bool = False,
) -> str | None:
    """Trim the argument list of the call on line *idx*; return new source or None.

    Returns None when the call cannot be trimmed (no parens on the line, empty
    argument list, or a single argument).  *keep_first=True* keeps only the
    first argument (Python's "takes 1 positional argument" flavor — self plus
    the intended args); the default drops the last argument.
    """
    line = lines[idx]
    paren_open = line.find("(")
    if paren_open == -1:
        return None
    paren_close = line.find(")", paren_open)
    if paren_close == -1:
        return None
    inner = line[paren_open + 1 : paren_close].strip()
    if not inner:
        return None
    args = [a.strip() for a in inner.split(",")]
    if len(args) < 2:
        return None
    new_inner = args[0] if keep_first else ", ".join(args[:-1])
    lines[idx] = line[: paren_open + 1] + new_inner + line[paren_close:]
    return "\n".join(lines)


def _repair_argument_mismatch(
    code: str,
    error: VerifyError,
    markers: tuple[str, ...],
    *,
    keep_first: bool = False,
) -> list[PrimitiveOp] | None:
    """Trim the call's argument list when the message carries a *marker*.

    Shared by the py/java/kotlin/go argument-mismatch strategies — the only
    language difference is the diagnostic wording and whether to keep the
    first argument (Python's self-flavored "takes 1 positional argument") or
    drop the last (the brace-language "too many arguments" flavor).
    """
    if error.line is None:
        return None
    msg = error.message.lower()
    if not any(marker in msg for marker in markers):
        return None
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    trimmed = _trim_call_arguments(lines, idx, keep_first=keep_first)
    if trimmed is not None:
        return _make_raw_replacement(trimmed)
    return None


# ═══════════════════════════════════════════════════════════════════════
# Python Repair Strategies
# ═══════════════════════════════════════════════════════════════════════

# Known Python import map for auto-resolution
_PY_IMPORT_MAP: dict[str, tuple] = {
    "List": ("typing", True),
    "Dict": ("typing", True),
    "Tuple": ("typing", True),
    "Set": ("typing", True),
    "Optional": ("typing", True),
    "Union": ("typing", True),
    "Any": ("typing", True),
    "Callable": ("typing", True),
    "Iterable": ("typing", True),
    "Iterator": ("typing", True),
    "Generator": ("typing", True),
    "TypeVar": ("typing", True),
    "Generic": ("typing", True),
    "Protocol": ("typing", True),
    "dataclass": ("dataclasses", True),
    "field": ("dataclasses", True),
    "dataclasses": ("dataclasses", False),
    "ABC": ("abc", True),
    "abstractmethod": ("abc", True),
    "defaultdict": ("collections", True),
    "OrderedDict": ("collections", True),
    "Counter": ("collections", True),
    "deque": ("collections", True),
    "namedtuple": ("collections", True),
    "partial": ("functools", True),
    "wraps": ("functools", True),
    "lru_cache": ("functools", True),
    "Path": ("pathlib", True),
    "os": ("os", False),
    "sys": ("sys", False),
    "re": ("re", False),
    "json": ("json", False),
    "math": ("math", False),
    "datetime": ("datetime", False),
    "typing": ("typing", False),
    "Enum": ("enum", True),
    "IntEnum": ("enum", True),
    "pytest": ("pytest", False),
}


def py_repair_missing_variable(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Add import for undefined variable if it's a known stdlib/typing symbol."""
    symbol = classification.symbol
    if not symbol:
        return None
    entry = _PY_IMPORT_MAP.get(symbol)
    if not entry:
        return None
    module, is_name = entry
    stmt = f"from {module} import {symbol}" if is_name else f"import {module}"
    return [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": stmt},
        )
    ]


def py_repair_syntax_error(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix common Python syntax errors (missing colon, indent issues)."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    # Missing colon after def/class/if/for/while/try/except/with/elif/else
    if "expected ':'" in msg or "expected :" in msg:
        line = lines[idx].rstrip()
        if not line.endswith(":"):
            # Check if ending with a keyword that expects colon
            stripped = line.strip()
            if any(
                stripped.startswith(kw)
                for kw in (
                    "def ",
                    "class ",
                    "if ",
                    "elif ",
                    "else",
                    "for ",
                    "while ",
                    "try",
                    "except",
                    "with ",
                    "finally",
                )
            ):
                lines[idx] = line + ":"
                return _make_raw_replacement("\n".join(lines))
        return None

    return None


# Shared by the Python/Java missing-return repairs — the only language
# differences are the header-line recognizer, comment handling, the body-end
# marker and the inserted statement. Same pattern as _repair_unknown_symbol.
def _repair_missing_return(
    code: str,
    error: VerifyError,
    *,
    is_header: Callable[[str], bool],
    skip_comments: bool,
    end_marker: str | None,
    return_stmt_factory: Callable[[str], str],
    extra_indent: bool = False,
) -> list[PrimitiveOp] | None:
    """Insert a return statement into the body containing ``error.line``.

    Walks backward from the error line to find the enclosing function/method
    header (matched by *is_header*), scans the body for its last line, and
    inserts the statement returned by *return_stmt_factory* (which receives
    the stripped header line) at body indent.

    *skip_comments* makes comment lines transparent to the body scan.
    *end_marker* is the line that ends the body: ``"}"`` for brace languages;
    ``None`` ends the body on any line that does not start with the body
    indent (Python's dedent rule).

    The statement is inserted at the body's own indent (the indent of the
    first body line).  *extra_indent=True* inserts at body indent + 4 — the
    Kotlin/Go flavor, whose tests pin the deeper offset.  NOTE: the original
    py/java implementations used the deeper offset too, which produced invalid
    Python (IndentationError) for plain function bodies — fixed for py/java
    here; no test pinned the old offset for them.
    """
    if error.line is None:
        return None
    lines = code.split("\n")
    # Walk backward from error line to find the function/method header
    for i in range(error.line - 1, -1, -1):
        stripped = lines[i].strip()
        if not is_header(stripped):
            continue
        body_start = i + 1
        if body_start >= len(lines):
            return None
        body_indent = _get_indent(lines[body_start])
        if not body_indent:
            return None
        # Find the last non-empty, non-comment line in the body
        last_body_line = body_start
        for j in range(body_start, len(lines)):
            s = lines[j].strip()
            if not s or (skip_comments and s.startswith("#")):
                continue
            if _get_indent(lines[j]).startswith(body_indent):
                last_body_line = j
            elif end_marker is None or s == end_marker:
                # Decreased indent (Python) or closing brace (Java): body ended
                break
        stmt_indent = body_indent + ("    " if extra_indent else "")
        lines.insert(last_body_line + 1, stmt_indent + return_stmt_factory(stripped))
        return _make_raw_replacement("\n".join(lines))
    return None


def py_repair_missing_return(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Add return None to a function missing a return."""
    return _repair_missing_return(
        code,
        error,
        is_header=lambda s: s.startswith("def ") and s.endswith(":"),
        skip_comments=True,
        end_marker=None,
        return_stmt_factory=lambda _s: "return None",
    )


def py_repair_argument_mismatch(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix argument count mismatch — limited case: add/remove self."""
    msg = error.message.lower()
    if "missing 1 required positional argument" in msg:
        # Could be missing 'self' in a method call → not fixable deterministically
        return None
    if "takes 1 positional argument but" in msg:
        # Keep only the first argument (self plus the intended args).
        return _repair_argument_mismatch(
            code,
            error,
            ("takes 1 positional argument but",),
            keep_first=True,
        )
    return None


# ═══════════════════════════════════════════════════════════════════════
# Java Repair Strategies
# ═══════════════════════════════════════════════════════════════════════

_JAVA_IMPORT_MAP: dict[str, str] = {
    "List": "java.util.List",
    "ArrayList": "java.util.ArrayList",
    "Map": "java.util.Map",
    "HashMap": "java.util.HashMap",
    "Set": "java.util.Set",
    "HashSet": "java.util.HashSet",
    "Optional": "java.util.Optional",
    "Date": "java.util.Date",
    "Calendar": "java.util.Calendar",
    "File": "java.io.File",
    "IOException": "java.io.IOException",
    "InputStream": "java.io.InputStream",
    "OutputStream": "java.io.OutputStream",
    "BufferedReader": "java.io.BufferedReader",
    "BufferedWriter": "java.io.BufferedWriter",
    "Path": "java.nio.file.Path",  # nio not io
    "Paths": "java.nio.file.Paths",
    "Stream": "java.util.stream.Stream",
    "Collectors": "java.util.stream.Collectors",
    "Function": "java.util.function.Function",
    "Consumer": "java.util.function.Consumer",
    "Predicate": "java.util.function.Predicate",
    "Supplier": "java.util.function.Supplier",
    "Collections": "java.util.Collections",
    "Arrays": "java.util.Arrays",
    "StringBuilder": "java.lang.StringBuilder",  # auto-imported, but safe
}


# Shared by the Java/Kotlin repair families — the only language difference is
# the import statement's trailing semicolon and the diagnostic wording.
def _repair_unknown_symbol(
    code: str,
    classification: Classification,
    import_map: dict[str, str],
    *,
    stmt_fmt: str = "import {}",
) -> list[PrimitiveOp] | None:
    """Add an import for an unknown symbol if it maps to a known FQN.

    *stmt_fmt* formats the import statement from the FQN — Java
    ``"import {};"``, Kotlin ``"import {}"``, Go ``'import "{}"'``.  The
    existence check matches the same statement form so an already-present
    import is never re-inserted.
    """
    symbol = classification.symbol
    if not symbol:
        return None
    fqn = import_map.get(symbol)
    if not fqn:
        return None
    stmt = stmt_fmt.format(fqn)
    if stmt in code:
        return None
    return [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": stmt},
        )
    ]


def _repair_missing_semicolon(
    code: str,
    error: VerifyError,
    markers: tuple[str, ...],
    *,
    skip_ending: tuple[str, ...] = (),
) -> list[PrimitiveOp] | None:
    """Append ``;`` to the error line when the message carries a *marker*.

    Lines ending with any *skip_ending* suffix are left untouched (Go's
    ``{``-terminated headers must not receive a semicolon).
    """
    if error.line is None:
        return None
    msg = error.message.lower()
    if not any(marker in msg for marker in markers):
        return None
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    line = lines[idx].rstrip()
    if not line.endswith(";") and not any(line.endswith(s) for s in skip_ending):
        lines[idx] = line + ";"
        return _make_raw_replacement("\n".join(lines))
    return None


def java_repair_unknown_symbol(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Add import for unknown symbol if it's a known Java type."""
    return _repair_unknown_symbol(code, classification, _JAVA_IMPORT_MAP, stmt_fmt="import {};")


def java_repair_syntax_error(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix common Java syntax errors (missing semicolons, braces)."""
    return _repair_missing_semicolon(code, error, ("';' expected", "expected ';'"))


def java_repair_missing_return(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Add return null to method missing a return statement."""
    return _repair_missing_return(
        code,
        error,
        is_header=lambda s: any(s.startswith(kw) for kw in ("public ", "private ", "protected ")) and "{" in s,
        skip_comments=False,
        end_marker="}",
        return_stmt_factory=lambda _s: "return null;",
    )


# ═══════════════════════════════════════════════════════════════════════
# Kotlin Repair Strategies
# ═══════════════════════════════════════════════════════════════════════

_KOTLIN_IMPORT_MAP: dict[str, str] = {
    "List": "kotlin.collections.List",
    "MutableList": "kotlin.collections.MutableList",
    "Map": "kotlin.collections.Map",
    "MutableMap": "kotlin.collections.MutableMap",
    "Set": "kotlin.collections.Set",
    "MutableSet": "kotlin.collections.MutableSet",
    "ArrayList": "kotlin.collections.ArrayList",
    "HashMap": "kotlin.collections.HashMap",
    "HashSet": "kotlin.collections.HashSet",
    "Optional": "java.util.Optional",  # Kotlin uses nullable types, but Optional exists
    "File": "java.io.File",
    "Path": "java.nio.file.Path",
    "Paths": "java.nio.file.Paths",
    "BigDecimal": "java.math.BigDecimal",
    "BigInteger": "java.math.BigInteger",
    "LocalDate": "java.time.LocalDate",
    "LocalDateTime": "java.time.LocalDateTime",
    "Duration": "kotlin.time.Duration",
    "CoroutineScope": "kotlinx.coroutines.CoroutineScope",
    "launch": "kotlinx.coroutines.launch",
    "async": "kotlinx.coroutines.async",
    "Dispatchers": "kotlinx.coroutines.Dispatchers",
}


def kotlin_repair_unknown_symbol(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Add import for unknown symbol if known (Kotlin imports carry no semicolon)."""
    return _repair_unknown_symbol(code, classification, _KOTLIN_IMPORT_MAP, stmt_fmt="import {}")


def kotlin_repair_syntax_error(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix common Kotlin syntax errors (missing semicolons)."""
    return _repair_missing_semicolon(code, error, ("expecting ';'", "expected ';'"))


# ═══════════════════════════════════════════════════════════════════════
# Go Repair Strategies
# ═══════════════════════════════════════════════════════════════════════

_GO_IMPORT_MAP: dict[str, str] = {
    "fmt": "fmt",
    "os": "os",
    "io": "io",
    "strings": "strings",
    "strconv": "strconv",
    "math": "math",
    "time": "time",
    "json": "encoding/json",
    "xml": "encoding/xml",
    "csv": "encoding/csv",
    "http": "net/http",
    "url": "net/url",
    "regexp": "regexp",
    "sort": "sort",
    "sync": "sync",
    "errors": "errors",
    "log": "log",
    "flag": "flag",
    "context": "context",
    "bytes": "bytes",
    "bufio": "bufio",
    "ioutil": "io/ioutil",  # deprecated but still common
    "filepath": "path/filepath",
    "path": "path",
    "atomic": "sync/atomic",
    "rand": "math/rand",
    "testing": "testing",
}


def go_repair_unknown_symbol(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix undefined symbol: try import first, then case-correction.

    Go undefined errors come in two flavors:
    1. Missing import (e.g. "undefined: fmt") → add import
    2. Local variable / field name typo (e.g. "undefined: dueDate") →
       try case-correction (capitalize/lowercase first letter).
    """
    symbol = classification.symbol
    if not symbol:
        return None

    # ── Path 1: Known stdlib package → add import (shared helper) ──
    ops = _repair_unknown_symbol(
        code,
        classification,
        _GO_IMPORT_MAP,
        stmt_fmt='import "{}"',
    )
    if ops is not None:
        return ops
    if symbol in _GO_IMPORT_MAP:
        return None  # already imported — something else is wrong

    # ── Path 2: Try case-correction for local symbols ───────────────
    if error.line is None:
        return None

    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    # Collect candidate corrections
    candidates: list[str] = []
    if symbol and symbol[0].islower():
        # Try capitalizing first letter (struct field, exported name)
        candidates.append(symbol[0].upper() + symbol[1:])
    if symbol and symbol[0].isupper():
        # Try lowercasing first letter (local variable, parameter)
        candidates.append(symbol[0].lower() + symbol[1:])

    # Also try the reverse: if symbol is already mixed-case (e.g. "dueDate"),
    # try both extremes
    if len(symbol) > 1:
        cap_first = symbol[0].upper() + symbol[1:]
        low_first = symbol[0].lower() + symbol[1:]
        if cap_first not in candidates:
            candidates.append(cap_first)
        if low_first not in candidates:
            candidates.append(low_first)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_candidates = []
    for c in candidates:
        if c != symbol and c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    # Check each candidate against the code (word-boundary match)
    for candidate in unique_candidates:
        # Use word-boundary regex to avoid substring matches
        _pat = re.compile(r"\b" + re.escape(candidate) + r"\b")
        if _pat.search(code):
            # Replace the undefined symbol in the error line only
            lines[idx] = re.sub(r"\b" + re.escape(symbol) + r"\b", candidate, lines[idx])
            return _make_raw_replacement("\n".join(lines))

    return None


def go_repair_unused_import(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Remove unused import (Go compiler error)."""
    symbol = classification.symbol
    if not symbol:
        return None
    # Find the import line and remove it
    lines = code.split("\n")
    new_lines = []
    removed = False
    in_import_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import (") and not removed:
            in_import_block = True
            new_lines.append(line)
        elif in_import_block:
            if stripped == ")":
                in_import_block = False
                new_lines.append(line)
            elif f'"{symbol}"' in stripped:
                removed = True
                continue
            else:
                new_lines.append(line)
        elif stripped.startswith('import "') and not removed:
            if symbol in stripped:
                removed = True
                continue
            new_lines.append(line)
        else:
            new_lines.append(line)
    if removed:
        return _make_raw_replacement("\n".join(new_lines))
    return None


def go_repair_syntax_error(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix common Go syntax errors (missing semicolons, braces)."""
    semicolon_fix = _repair_missing_semicolon(
        code,
        error,
        ("expected ';'", "expected newline"),
        skip_ending=("{",),
    )
    if semicolon_fix is not None:
        return semicolon_fix
    if error.line is None:
        return None
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    if "expected '{'" in error.message.lower():
        line = lines[idx].rstrip()
        if not line.endswith("{"):
            lines[idx] = line + " {"
            return _make_raw_replacement("\n".join(lines))
    return None


# ═══════════════════════════════════════════════════════════════════════
# Java — Additional Strategies
# ═══════════════════════════════════════════════════════════════════════


def java_repair_argument_mismatch(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix Java argument count mismatch by removing extra arguments."""
    return _repair_argument_mismatch(
        code,
        error,
        ("actual and formal argument lists differ in length",),
    )


def _repair_duplicate_identifier(
    code: str,
    error: VerifyError,
    markers: tuple[str, ...],
    patterns: tuple[tuple[str, str], ...],
) -> list[PrimitiveOp] | None:
    """Rename a duplicate identifier on the error line by appending a suffix.

    *patterns* is a sequence of ``(regex, suffix)`` pairs tried in order on
    the stripped error line; the first match wins.  The shared guard block
    (line bounds, marker check) mirrors the other ``_repair_*`` helpers.
    """
    if error.line is None:
        return None
    msg = error.message.lower()
    if not any(marker in msg for marker in markers):
        return None
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    line = lines[idx]
    stripped = line.strip()
    for rx, suffix in patterns:
        m = re.search(rx, stripped)
        if m:
            orig = m.group(1)
            new_line = stripped.replace(orig, orig + suffix, 1)
            lines[idx] = _get_indent(line) + new_line
            return _make_raw_replacement("\n".join(lines))
    return None


def java_repair_duplicate_identifier(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix duplicate class/local identifier by appending a suffix."""
    return _repair_duplicate_identifier(
        code,
        error,
        markers=("duplicate class",),
        patterns=((r"\bclass\s+(\w+)", "Dup"),),
    )


# ═══════════════════════════════════════════════════════════════════════
# Kotlin — Additional Strategies
# ═══════════════════════════════════════════════════════════════════════


def _kotlin_return_stmt(header: str) -> str:
    """Pick the Kotlin return statement from a ``fun`` header line."""
    returns_value = ":" in header and not header.rstrip().endswith("Unit")
    return "return null" if returns_value else "return"


def kotlin_repair_missing_return(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Add return (Unit/null) to a Kotlin function missing a return."""
    return _repair_missing_return(
        code,
        error,
        is_header=lambda s: s.startswith("fun "),
        skip_comments=False,
        end_marker="}",
        return_stmt_factory=_kotlin_return_stmt,
        extra_indent=True,
    )


def kotlin_repair_argument_mismatch(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix Kotlin argument count mismatch — remove extra arguments."""
    return _repair_argument_mismatch(code, error, ("too many", "required"))


def kotlin_repair_duplicate_identifier(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix duplicate Kotlin identifier by appending a suffix."""
    return _repair_duplicate_identifier(
        code,
        error,
        markers=("duplicate",),
        patterns=((r"\bfun\s+(\w+)", "Dup"), (r"\bclass\s+(\w+)", "Dup")),
    )


# ═══════════════════════════════════════════════════════════════════════
# Go — Additional Strategies
# ═══════════════════════════════════════════════════════════════════════


def go_repair_argument_mismatch(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix Go argument count mismatch — remove extra or add zero-value args."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    if "too many" in msg:
        too_many_fix = _repair_argument_mismatch(code, error, ("too many",))
        if too_many_fix is not None:
            return too_many_fix

    if "not enough" in msg or "not sufficient" in msg:
        line = lines[idx]
        paren_open = line.find("(")
        if paren_open == -1:
            return None
        paren_close = line.find(")", paren_open)
        if paren_close == -1:
            return None
        inner = line[paren_open + 1 : paren_close].strip()

        # Try to extract expected parameter types from the error message
        # Go error: "not enough arguments in call to ...\n    have (type1)\n    want (type1, type2, type3)"
        # NOTE: search in *original* error.message (preserves type casing like time.Time)
        #        while 'msg' (lowercased) is used for keyword checks only.
        _want_types: list[str] = []
        _want_m = re.search(r"want\s+\(([^)]*)\)", error.message, re.IGNORECASE)
        if _want_m:
            _want_types = [t.strip() for t in _want_m.group(1).split(",")]

        _existing = [a.strip() for a in inner.split(",") if a.strip()] if inner else []
        _missing_count = len(_want_types) - len(_existing)

        if _missing_count > 0:
            _fill_args = []
            for i in range(_missing_count):
                _tidx = len(_existing) + i
                if _tidx < len(_want_types):
                    _fill_args.append(_go_zero_value(_want_types[_tidx]))
                else:
                    # Unreachable: _missing_count = len(_want_types) - len(_existing),
                    # so _tidx = len(_existing) + i < len(_want_types) for all
                    # i in range(_missing_count). Kept as a defensive guard.
                    _fill_args.append("nil")  # pragma: no cover
            new_inner = ", ".join(_existing + _fill_args) if _existing else ", ".join(_fill_args)
        # Fallback: add zero value for the missing arg type
        # (can't use nil for value types like time.Time)
        elif not inner:
            new_inner = "nil"  # empty args → use nil as safe default
        else:
            new_inner = inner + ", nil"  # unknown type → nil (will be caught by TYPE_MISMATCH repair)
        lines[idx] = line[: paren_open + 1] + new_inner + line[paren_close:]
        return _make_raw_replacement("\n".join(lines))
    return None


def _go_return_stmt(header: str) -> str:
    """Pick the Go return statement (zero value) from a ``func`` header line."""
    ret_type = None
    paren_close = header.rfind(")")
    if paren_close != -1:
        after_parens = header[paren_close + 1 :].strip()
        if after_parens and "{" not in after_parens:
            ret_type = after_parens.split("{")[0].strip()
    if ret_type:
        return "return " + _go_zero_value(ret_type)
    return "return nil"


def go_repair_missing_return(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Add return with zero-value to a Go function missing a return statement."""
    if "missing return" not in error.message.lower():
        return None
    return _repair_missing_return(
        code,
        error,
        is_header=lambda s: bool(re.match(r"^func\s+\w+", s)),
        skip_comments=False,
        end_marker="}",
        return_stmt_factory=_go_return_stmt,
        extra_indent=True,
    )


def _go_zero_value(type_name: str) -> str:
    """Return the Go zero value literal for a type."""
    t = type_name.strip()
    t_lower = t.lower()
    if t_lower in ("int", "int8", "int16", "int32", "int64"):
        return "0"
    if t_lower in ("uint", "uint8", "uint16", "uint32", "uint64"):
        return "0"
    if t_lower in ("float32", "float64"):
        return "0.0"
    if t_lower in ("bool", "boolean"):
        return "false"
    if t_lower in ("string",):
        return '""'
    if t_lower in ("error",):
        return "nil"
    if t_lower.startswith("[]"):
        return "nil"
    if t_lower.startswith("map["):
        return "nil"
    if t_lower.startswith("*"):
        return "nil"
    if t_lower.startswith("func"):
        return "nil"
    if t_lower.startswith("chan"):
        return "nil"
    if t_lower.startswith("interface"):
        return "nil"
    # ── Unknown types: check if it's a struct/value type ──────────────
    # Go value types (structs, arrays, named types) can't be nil.
    # Capitalized names are exported types (likely structs/interfaces).
    # Use Type{} syntax which works for all composite types.
    # Handle qualified names like "time.Time" → "time.Time{}"
    return t + "{}"


def go_repair_type_mismatch(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix Go type mismatch — add explicit type conversion when possible."""
    if error.line is None:
        return None
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    # Match both current (Go >= 1.21) and legacy (<= 1.20) compiler formats:
    #   current: "cannot use t (variable of struct type time.Time) as string value in ..."
    #            "cannot use nil as time.Time value in ..."
    #   legacy:  "cannot use t (type time.Time) as type string in ..."
    # Search the *original* message (like `go_repair_argument_mismatch`) so
    # qualified type names keep their casing ("time.Time", "sql.NullString").
    m = re.search(
        r"cannot use\s+(?P<expr>nil|\S+?)"
        r"(?:\s*\(\s*type\s+(?P<from_legacy>[^)]+)\s*\))?"
        r"(?:\s*\(\s*variable of\s+(?:struct\s+)?type\s+(?P<from_modern>[^)]+)\s*\))?"
        r"\s+as\s+(?:type\s+)?(?P<to>[^\s]+?)(?:\s+value)?\s+in\b",
        error.message,
        re.IGNORECASE,
    )
    if m:
        from_type = m.group("from_modern") or m.group("from_legacy")
        to_type = m.group("to")
        expr = m.group("expr")
        line = lines[idx]

        # Special case: cannot use nil as type X (X is a value type)
        # Replace nil with Type{} zero-value literal
        if expr == "nil" and to_type != "nil":
            _zero = _go_zero_value(to_type)
            line = line.replace("nil", _zero, 1)
            lines[idx] = line
            return _make_raw_replacement("\n".join(lines))
        # Try wrapping the offending expression in Type(expr)
        # For simple numeric conversions like int, float64 etc.
        numeric_types = {
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float32",
            "float64",
            "byte",
            "rune",
        }
        if from_type in numeric_types and to_type in numeric_types:
            # Wrap the expression in Type(...)
            # Find what to wrap: the assignment or comparison value
            eq_pos = line.find("=")
            if eq_pos != -1:
                rhs = line[eq_pos + 1 :].strip()
                rhs_clean = rhs.rstrip(" {") if "{" in rhs else rhs.rstrip()
                new_rhs = f"{to_type}({rhs_clean})"
                lines[idx] = line[: eq_pos + 1] + " " + new_rhs
                return _make_raw_replacement("\n".join(lines))
    return None


# ═══════════════════════════════════════════════════════════════════════
# Python — Additional Strategies
# ═══════════════════════════════════════════════════════════════════════


def py_repair_duplicate_identifier(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fix duplicate function/class identifier by appending a suffix."""
    return _repair_duplicate_identifier(
        code,
        error,
        markers=("redefined", "duplicate"),
        patterns=((r"\bdef\s+(\w+)", "_dup"), (r"\bclass\s+(\w+)", "Dup")),
    )


# ═══════════════════════════════════════════════════════════════════════
# Strategy map per language
# ═══════════════════════════════════════════════════════════════════════


def _build_python_strategies() -> dict[FailureType, Callable]:
    return {
        FailureType.MISSING_VARIABLE: py_repair_missing_variable,
        FailureType.UNKNOWN_SYMBOL: py_repair_missing_variable,
        FailureType.MISSING_IMPORT: py_repair_missing_variable,
        FailureType.SYNTAX_ERROR: py_repair_syntax_error,
        FailureType.MISSING_RETURN: py_repair_missing_return,
        FailureType.ARGUMENT_MISMATCH: py_repair_argument_mismatch,
        FailureType.DUPLICATE_IDENTIFIER: py_repair_duplicate_identifier,
    }


def _build_java_strategies() -> dict[FailureType, Callable]:
    return {
        FailureType.UNKNOWN_SYMBOL: java_repair_unknown_symbol,
        FailureType.SYNTAX_ERROR: java_repair_syntax_error,
        FailureType.MISSING_RETURN: java_repair_missing_return,
        FailureType.ARGUMENT_MISMATCH: java_repair_argument_mismatch,
        FailureType.DUPLICATE_IDENTIFIER: java_repair_duplicate_identifier,
    }


def _build_kotlin_strategies() -> dict[FailureType, Callable]:
    return {
        FailureType.UNKNOWN_SYMBOL: kotlin_repair_unknown_symbol,
        FailureType.SYNTAX_ERROR: kotlin_repair_syntax_error,
        FailureType.MISSING_RETURN: kotlin_repair_missing_return,
        FailureType.ARGUMENT_MISMATCH: kotlin_repair_argument_mismatch,
        FailureType.DUPLICATE_IDENTIFIER: kotlin_repair_duplicate_identifier,
    }


def _build_go_strategies() -> dict[FailureType, Callable]:
    return {
        FailureType.UNKNOWN_SYMBOL: go_repair_unknown_symbol,
        FailureType.UNUSED_IMPORT: go_repair_unused_import,
        FailureType.SYNTAX_ERROR: go_repair_syntax_error,
        FailureType.ARGUMENT_MISMATCH: go_repair_argument_mismatch,
        FailureType.MISSING_RETURN: go_repair_missing_return,
        FailureType.TYPE_MISMATCH: go_repair_type_mismatch,
    }


_STRATEGY_MAP: dict[str, dict[FailureType, Callable]] = {
    "python": _build_python_strategies(),
    "java": _build_java_strategies(),
    "kotlin": _build_kotlin_strategies(),
    "go": _build_go_strategies(),
}


def get_strategies(language: str) -> dict[FailureType, Callable]:
    """Return the strategy map for *language*."""
    strategies = _STRATEGY_MAP.get(language)
    if strategies is None:
        raise ValueError(f"No repair strategies for language: {language}")
    return strategies


def repair_unknown_symbol(
    code: str,
    error: VerifyError,
    classification: Classification,
) -> list[PrimitiveOp] | None:
    """Fallback: delegate to language-specific unknown symbol repair."""
    get_strategies("python")  # overridden by registry
    return None
