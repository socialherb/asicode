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
from typing import Optional

from external_llm.editor._editor_core.vm.classification import Classification
from external_llm.editor._editor_core.vm.failure_classifier import FailureType
from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor.primitives.models import PrimitiveKind, PrimitiveOp

# ── Utility ───────────────────────────────────────────────────────────

def _make_raw_replacement(code: str) -> list[PrimitiveOp]:
    """Wrap a full code replacement as a raw-replacement op."""
    return [PrimitiveOp(
        kind=PrimitiveKind.INSERT_STATEMENT,
        payload={"__raw_code__": code},
    )]


def _get_indent(line: str) -> str:
    indent = ""
    for ch in line:
        if ch in (" ", "\t"):
            indent += ch
        else:
            break
    return indent


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
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Add import for undefined variable if it's a known stdlib/typing symbol."""
    symbol = classification.symbol
    if not symbol:
        return None
    entry = _PY_IMPORT_MAP.get(symbol)
    if not entry:
        return None
    module, is_name = entry
    if is_name:
        stmt = f"from {module} import {symbol}"
    else:
        stmt = f"import {module}"
    return [PrimitiveOp(
        kind=PrimitiveKind.INSERT_IMPORT,
        payload={"statement": stmt},
    )]


def py_repair_syntax_error(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
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
            if any(stripped.startswith(kw) for kw in
                   ("def ", "class ", "if ", "elif ", "else", "for ",
                    "while ", "try", "except", "with ", "finally")):
                lines[idx] = line + ":"
                return _make_raw_replacement("\n".join(lines))
        return None

    return None


def py_repair_missing_return(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Add return None to a function missing a return."""
    if error.line is None:
        return None
    lines = code.split("\n")
    # Walk backward from error line to find function def
    for i in range(error.line - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            # Find the last line of the function body
            body_start = i + 1
            if body_start >= len(lines):
                return None
            body_indent = _get_indent(lines[body_start])
            if not body_indent:
                return None
            # Find last non-empty, non-comment line in the body
            last_body_line = body_start
            for j in range(body_start, len(lines)):
                if lines[j].strip() and not lines[j].strip().startswith("#"):
                    if _get_indent(lines[j]).startswith(body_indent):
                        last_body_line = j
                    else:
                        # Decreased indent = we left the function body
                        break
            return_stmt_indent = body_indent + "    "
            lines.insert(last_body_line + 1, return_stmt_indent + "return None")
            return _make_raw_replacement("\n".join(lines))
    return None


def py_repair_argument_mismatch(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix argument count mismatch — limited case: add/remove self."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    if "missing 1 required positional argument" in msg:
        # Could be missing 'self' in a method call → not fixable deterministically
        return None
    if "takes 1 positional argument but" in msg:
        line = lines[idx]
        # Try removing extra arguments
        paren_open = line.find("(")
        if paren_open == -1:
            return None
        paren_close = line.find(")", paren_open)
        if paren_close == -1:
            return None
        # Simple case: only one argument (self) expected, multiple given
        inner = line[paren_open + 1:paren_close].strip()
        if inner:
            # Keep only first argument
            first_arg = inner.split(",")[0].strip()
            lines[idx] = line[:paren_open + 1] + first_arg + line[paren_close:]
            return _make_raw_replacement("\n".join(lines))
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


def java_repair_unknown_symbol(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Add import for unknown symbol if it's a known Java type."""
    symbol = classification.symbol
    if not symbol:
        return None
    fqn = _JAVA_IMPORT_MAP.get(symbol)
    if not fqn:
        return None
    stmt = f"import {fqn};"
    # Check if import already exists
    if f"import {fqn};" in code:
        return None
    return [PrimitiveOp(
        kind=PrimitiveKind.INSERT_IMPORT,
        payload={"statement": stmt},
    )]


def java_repair_syntax_error(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix common Java syntax errors (missing semicolons, braces)."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    if "';' expected" in msg or "expected ';'" in msg:
        line = lines[idx].rstrip()
        if not line.endswith(";"):
            lines[idx] = line + ";"
            return _make_raw_replacement("\n".join(lines))
    return None


def java_repair_missing_return(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Add return null to method missing a return statement."""
    if error.line is None:
        return None
    lines = code.split("\n")
    for i in range(error.line - 1, -1, -1):
        stripped = lines[i].strip()
        if any(stripped.startswith(kw) for kw in
               ("public ", "private ", "protected ")):
            if "{" in stripped:
                body_start = i + 1
                if body_start >= len(lines):
                    return None
                body_indent = _get_indent(lines[body_start])
                if not body_indent:
                    return None
                # Find last line of method body
                last_body = body_start
                for j in range(body_start, len(lines)):
                    if lines[j].strip():
                        if _get_indent(lines[j]).startswith(body_indent):
                            last_body = j
                        elif lines[j].strip() == "}":
                            break
                return_indent = body_indent + "    "
                lines.insert(last_body + 1, return_indent + "return null;")
                return _make_raw_replacement("\n".join(lines))
    return None


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
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Add import for unknown symbol if known."""
    symbol = classification.symbol
    if not symbol:
        return None
    fqn = _KOTLIN_IMPORT_MAP.get(symbol)
    if not fqn:
        return None
    stmt = f"import {fqn}"
    if f"import {fqn}" in code:
        return None
    return [PrimitiveOp(
        kind=PrimitiveKind.INSERT_IMPORT,
        payload={"statement": stmt},
    )]


def kotlin_repair_syntax_error(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix common Kotlin syntax errors."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    if "expecting ';'" in msg or "expected ';'" in msg:
        line = lines[idx].rstrip()
        if not line.endswith(";"):
            lines[idx] = line + ";"
            return _make_raw_replacement("\n".join(lines))
    return None


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
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix undefined symbol: try import first, then case-correction.

    Go undefined errors come in two flavors:
    1. Missing import (e.g. "undefined: fmt") → add import
    2. Local variable / field name typo (e.g. "undefined: dueDate") →
       try case-correction (capitalize/lowercase first letter).
    """
    symbol = classification.symbol
    if not symbol:
        return None

    # ── Path 1: Known stdlib package → add import ──────────────────
    pkg = _GO_IMPORT_MAP.get(symbol)
    if pkg:
        stmt = f'import "{pkg}"'
        if f'import "{pkg}"' not in code:
            return [PrimitiveOp(
                kind=PrimitiveKind.INSERT_IMPORT,
                payload={"statement": stmt},
            )]
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
        _pat = re.compile(r'\b' + re.escape(candidate) + r'\b')
        if _pat.search(code):
            # Replace the undefined symbol in the error line only
            lines[idx] = re.sub(r'\b' + re.escape(symbol) + r'\b', candidate, lines[idx])
            return _make_raw_replacement("\n".join(lines))

    return None


def go_repair_unused_import(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
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
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix common Go syntax errors (missing braces, semicolons)."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    if "expected ';'" in msg or "expected newline" in msg:
        # Go uses implicit semicolons — usually a newline issue
        # Add semicolon or newline as appropriate
        line = lines[idx].rstrip()
        if not line.endswith(";") and not line.endswith("{"):
            lines[idx] = line + ";"
            return _make_raw_replacement("\n".join(lines))
    if "expected '{'" in msg:
        line = lines[idx].rstrip()
        if not line.endswith("{"):
            lines[idx] = line + " {"
            return _make_raw_replacement("\n".join(lines))
    return None


# ═══════════════════════════════════════════════════════════════════════
# Java — Additional Strategies
# ═══════════════════════════════════════════════════════════════════════


def java_repair_argument_mismatch(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix Java argument count mismatch by removing extra or adding placeholder args."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    actual_and_formal = re.search(
        r"actual and formal argument lists differ in length",
        msg,
    )
    if actual_and_formal:
        # Count args in the call
        line = lines[idx]
        paren_open = line.find("(")
        if paren_open == -1:
            return None
        paren_close = line.find(")", paren_open)
        if paren_close == -1:
            return None
        inner = line[paren_open + 1:paren_close].strip()
        if not inner:
            return None
        args = [a.strip() for a in inner.split(",")]
        # Try removing last argument if too many
        if len(args) > 1:
            new_inner = ", ".join(args[:-1])
            lines[idx] = line[:paren_open + 1] + new_inner + line[paren_close:]
            return _make_raw_replacement("\n".join(lines))
    return None


def java_repair_duplicate_identifier(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix duplicate class/local identifier by appending a suffix."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    if "duplicate class" in msg:
        line = lines[idx]
        stripped = line.strip()
        # Find class name after "class " keyword
        m = re.search(r"\bclass\s+(\w+)", stripped)
        if m:
            orig = m.group(1)
            new_name = orig + "Dup"
            new_line = stripped.replace(orig, new_name, 1)
            lines[idx] = _get_indent(line) + new_line
            return _make_raw_replacement("\n".join(lines))
    return None


# ═══════════════════════════════════════════════════════════════════════
# Kotlin — Additional Strategies
# ═══════════════════════════════════════════════════════════════════════


def kotlin_repair_missing_return(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Add return (Unit/null) to a Kotlin function missing a return."""
    if error.line is None:
        return None
    lines = code.split("\n")
    for i in range(error.line - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("fun "):
            body_start = i + 1
            if body_start >= len(lines):
                return None
            body_indent = _get_indent(lines[body_start])
            if not body_indent:
                return None
            # Determine if function returns a value (has explicit return type)
            returns_value = ":" in stripped and not stripped.rstrip().endswith("Unit")
            # Find last line of function body
            last_body = body_start
            for j in range(body_start, len(lines)):
                if lines[j].strip():
                    if _get_indent(lines[j]).startswith(body_indent):
                        last_body = j
                    elif lines[j].strip() == "}":
                        break
            return_indent = body_indent + "    "
            stmt = "return null" if returns_value else "return"
            lines.insert(last_body + 1, return_indent + stmt)
            return _make_raw_replacement("\n".join(lines))
    return None


def kotlin_repair_argument_mismatch(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix Kotlin argument count mismatch — remove extra arguments."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    # Kotlin: "too many arguments" or "required: X, found: Y"
    if "too many" in msg or "required" in msg:
        line = lines[idx]
        paren_open = line.find("(")
        if paren_open == -1:
            return None
        paren_close = line.find(")", paren_open)
        if paren_close == -1:
            return None
        inner = line[paren_open + 1:paren_close].strip()
        if not inner:
            return None
        args = [a.strip() for a in inner.split(",")]
        if len(args) > 1:
            new_inner = ", ".join(args[:-1])
            lines[idx] = line[:paren_open + 1] + new_inner + line[paren_close:]
            return _make_raw_replacement("\n".join(lines))
    return None


def kotlin_repair_duplicate_identifier(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix duplicate Kotlin identifier by appending a suffix."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    if "duplicate" in msg:
        line = lines[idx]
        stripped = line.strip()
        # Find function name after "fun " keyword
        m = re.search(r"\bfun\s+(\w+)", stripped)
        if m:
            orig = m.group(1)
            new_name = orig + "Dup"
            new_line = stripped.replace(orig, new_name, 1)
            lines[idx] = _get_indent(line) + new_line
            return _make_raw_replacement("\n".join(lines))
        # Find class name after "class " keyword
        m = re.search(r"\bclass\s+(\w+)", stripped)
        if m:
            orig = m.group(1)
            new_name = orig + "Dup"
            new_line = stripped.replace(orig, new_name, 1)
            lines[idx] = _get_indent(line) + new_line
            return _make_raw_replacement("\n".join(lines))
    return None


# ═══════════════════════════════════════════════════════════════════════
# Go — Additional Strategies
# ═══════════════════════════════════════════════════════════════════════


def go_repair_argument_mismatch(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix Go argument count mismatch — remove extra or add zero-value args."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    if "too many" in msg:
        line = lines[idx]
        paren_open = line.find("(")
        if paren_open == -1:
            return None
        paren_close = line.find(")", paren_open)
        if paren_close == -1:
            return None
        inner = line[paren_open + 1:paren_close].strip()
        if not inner:
            return None
        args = [a.strip() for a in inner.split(",")]
        if len(args) > 1:
            new_inner = ", ".join(args[:-1])
            lines[idx] = line[:paren_open + 1] + new_inner + line[paren_close:]
            return _make_raw_replacement("\n".join(lines))

    if "not enough" in msg or "not sufficient" in msg:
        line = lines[idx]
        paren_open = line.find("(")
        if paren_open == -1:
            return None
        paren_close = line.find(")", paren_open)
        if paren_close == -1:
            return None
        inner = line[paren_open + 1:paren_close].strip()

        # Try to extract expected parameter types from the error message
        # Go error: "not enough arguments in call to ...\n    have (type1)\n    want (type1, type2, type3)"
        # NOTE: search in *original* error.message (preserves type casing like time.Time)
        #        while 'msg' (lowercased) is used for keyword checks only.
        _want_types: list[str] = []
        _want_m = re.search(r'want\s+\(([^)]*)\)', error.message, re.IGNORECASE)
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
                    _fill_args.append("nil")
            if _existing:
                new_inner = ", ".join(_existing + _fill_args)
            else:
                new_inner = ", ".join(_fill_args)
        else:
            # Fallback: add zero value for the missing arg type
            # (can't use nil for value types like time.Time)
            if not inner:
                new_inner = "nil"  # empty args → use nil as safe default
            else:
                new_inner = inner + ", nil"  # unknown type → nil (will be caught by TYPE_MISMATCH repair)
        lines[idx] = line[:paren_open + 1] + new_inner + line[paren_close:]
        return _make_raw_replacement("\n".join(lines))
    return None


def go_repair_missing_return(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Add return with zero-value to a Go function missing a return statement."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    error.line - 1

    if "missing return" not in msg:
        return None

    # Walk backward to find the function signature
    for i in range(error.line - 1, -1, -1):
        stripped = lines[i].strip()
        if re.match(r'^func\s+\w+', stripped):
            func_line = stripped
            # Determine return type for zero value
            ret_type = None
            paren_close = func_line.rfind(")")
            if paren_close != -1:
                after_parens = func_line[paren_close + 1:].strip()
                if after_parens and "{" not in after_parens:
                    ret_type = after_parens.split("{")[0].strip()

            body_start = i + 1
            if body_start >= len(lines):
                return None
            body_indent = _get_indent(lines[body_start])
            if not body_indent:
                return None

            # Find last line of function body
            last_body = body_start
            for j in range(body_start, len(lines)):
                if lines[j].strip():
                    if _get_indent(lines[j]).startswith(body_indent):
                        last_body = j
                    elif lines[j].strip() == "}":
                        break

            return_indent = body_indent + "    "
            if ret_type:
                # Choose zero-value based on type name
                zero_val = _go_zero_value(ret_type)
                lines.insert(last_body + 1, return_indent + "return " + zero_val)
            else:
                lines.insert(last_body + 1, return_indent + "return nil")
            return _make_raw_replacement("\n".join(lines))
    return None


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
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix Go type mismatch — add explicit type conversion when possible."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    # Pattern: "cannot use X (type T) as type U in ..."
    # T and U can be qualified names like "time.Time" or "sql.NullString"
    m = re.search(
        r'cannot use\s+.+?\s+\(type\s+([\w.]+)\)\s+as\s+type\s+([\w.]+)',
        msg,
    )
    if m:
        from_type = m.group(1)
        to_type = m.group(2)
        line = lines[idx]

        # Special case: cannot use nil as type X (X is a value type)
        # Replace nil with Type{} zero-value literal
        if from_type == "nil" and to_type != "nil":
            _zero = _go_zero_value(to_type)
            line = line.replace("nil", _zero, 1)
            lines[idx] = line
            return _make_raw_replacement("\n".join(lines))
        # Try wrapping the offending expression in Type(expr)
        # For simple numeric conversions like int, float64 etc.
        numeric_types = {
            "int", "int8", "int16", "int32", "int64",
            "uint", "uint8", "uint16", "uint32", "uint64",
            "float32", "float64",
            "byte", "rune",
        }
        if from_type in numeric_types and to_type in numeric_types:
            # Wrap the expression in Type(...)
            # Find what to wrap: the assignment or comparison value
            eq_pos = line.find("=")
            if eq_pos != -1:
                rhs = line[eq_pos + 1:].strip()
                rhs_clean = rhs.rstrip(" {") if "{" in rhs else rhs.rstrip()
                new_rhs = f"{to_type}({rhs_clean})"
                lines[idx] = line[:eq_pos + 1] + " " + new_rhs
                return _make_raw_replacement("\n".join(lines))
    return None


# ═══════════════════════════════════════════════════════════════════════
# Python — Additional Strategies
# ═══════════════════════════════════════════════════════════════════════


def py_repair_duplicate_identifier(
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fix duplicate function/class identifier by appending a suffix."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split("\n")
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None

    if "redefined" in msg or "duplicate" in msg:
        line = lines[idx]
        stripped = line.strip()
        # Find function name after "def "
        m = re.search(r"\bdef\s+(\w+)", stripped)
        if m:
            orig = m.group(1)
            new_name = orig + "_dup"
            new_line = stripped.replace(orig, new_name, 1)
            lines[idx] = _get_indent(line) + new_line
            return _make_raw_replacement("\n".join(lines))
        # Find class name after "class "
        m = re.search(r"\bclass\s+(\w+)", stripped)
        if m:
            orig = m.group(1)
            new_name = orig + "Dup"
            new_line = stripped.replace(orig, new_name, 1)
            lines[idx] = _get_indent(line) + new_line
            return _make_raw_replacement("\n".join(lines))
    return None


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
    code: str, error: VerifyError, classification: Classification,
) -> Optional[list[PrimitiveOp]]:
    """Fallback: delegate to language-specific unknown symbol repair."""
    get_strategies("python")  # overridden by registry
    return None
