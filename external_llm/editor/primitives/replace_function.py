"""replace_function.py — Generic REPLACE_FUNCTION_BODY primitive.

Language-agnostic version that works on CodeContext instead of TSModule.
Same logic as ts_vm/primitives/replace_function.py but without
TS-specific dependencies.
"""
from __future__ import annotations

import logging
import re

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import PrimitiveOp, PrimitiveResult
from external_llm.languages.models import LanguageId

logger = logging.getLogger(__name__)


def _strip_full_function_def(body: str, func_name: str, language: str = "") -> str:
    """Strip leading full function/method definition, extracting just the body.

    Detects and strips for all supported languages:
      * Python:  def method_name(...) -> Type:
      * Java:    [public|private|protected] [static] Type methodName(...) [{]
      * Kotlin:  fun methodName(...): Type
      * Go:      func methodName(...) Type { ... }
      * TS/JS:   [async] function name(...) { ... }

    Falls back gracefully (returns original body) if no full definition detected.
    """
    stripped = body.strip()
    if not stripped:
        return body

    # Use the short name (without class prefix) for matching
    short_name = func_name.split(".")[-1]
    _fn = re.escape(short_name)

    # ── Language-agnostic patterns ──────────────────────────────────────

    # Pattern: function name(...) {  (JS/TS declaration)
    pat_fn_decl = re.compile(
        r'^(?:async\s+)?function\s+' + _fn + r'\s*\([^)]*\)\s*(?::\s*[^{]+)?\s*\{'
    )
    # Pattern: name = [async] function(...) {  (JS/TS expr)
    pat_fn_expr = re.compile(
        r'^(?:\S+\s+)?' + _fn
        + r'\s*=\s*(?:async\s+)?function\s*\([^)]*\)\s*(?::\s*[^{]+)?\s*\{'
    )
    # Pattern: name = (...) => {  (JS/TS arrow)
    pat_arrow = re.compile(
        r'^(?:\S+\s+)?' + _fn
        + r'\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^{]+)?\s*=>\s*\{'
    )

    # Pattern: def name(...) [-> Type]:  (Python)
    pat_py_def = re.compile(
        r'^\s*def\s+' + _fn + r'\s*\([^)]*\)\s*(?:\s*->\s*[^:]+)?\s*:\s*$'
    )

    # Pattern: [modifiers] Type name(...) [throws Type] [{]  (Java)
    pat_java = re.compile(
        r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?'
        + r'(?:\w+(?:<[^>]+>)?\s+)*' + _fn
        + r'\s*\([^)]*\)\s*(?:\s*throws\s+\w+(?:,\s*\w+)*)?\s*(?:\{)?\s*$'
    )

    # Pattern: fun name(...) [: Type] [= expr | {]  (Kotlin)
    pat_kotlin = re.compile(
        r'^\s*fun\s+' + _fn + r'\s*\([^)]*\)\s*(?::\s*[^{=]+)?\s*(?:\{|=)?\s*$'
    )

    # Pattern: func name(...) [Type] [{]  (Go)
    pat_go = re.compile(
        r'^\s*func\s+' + _fn + r'\s*\([^)]*\)\s*(?:\w+(?:\s*\{)?)?\s*$'
    )

    # ── Check for brace-delimited function (JS/TS/Java/Kotlin/Go) ──────
    brace_match = None
    for pat in (pat_fn_decl, pat_fn_expr, pat_arrow, pat_java, pat_go, pat_kotlin):
        m = pat.match(stripped)
        if m:
            brace_match = m
            break

    if brace_match:
        open_pos = brace_match.end()
        # Find the opening brace (could be on same line or next line)
        brace_start = stripped.find("{", brace_match.start())
        if brace_start == -1:
            return body  # No brace found, can't strip
        if brace_start < brace_match.end():
            open_pos = brace_start + 1
        else:
            # Brace is on the same line as signature
            open_pos = brace_start + 1

        depth = 1
        i = open_pos
        while i < len(stripped) and depth > 0:
            c = stripped[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            if depth > 0:
                i += 1

        if depth != 0:
            logger.warning(
                "_strip_full_function_def: mismatched braces for '%s', falling back",
                short_name,
            )
            return body

        inner = stripped[open_pos:i]
        logger.info(
            "_strip_full_function_def: stripped brace-delimited definition for '%s'",
            short_name,
        )
        return inner

    # ── Check for colon-delimited function (Python) ─────────────────────
    py_match = pat_py_def.match(stripped)
    if py_match:
        # Find the colon and take everything after (with proper indent)
        colon_pos = stripped.find(":")
        if colon_pos == -1:
            return body
        inner = stripped[colon_pos + 1:].strip()
        if inner:
            logger.info(
                "_strip_full_function_def: stripped Python def for '%s'",
                short_name,
            )
            return inner
        return body

    return body  # No full definition detected


def replace_function_body(op: PrimitiveOp, ctx: CodeContext) -> PrimitiveResult:
    """Replace the body of a named function/class/interface.

    Payload:
        name: str — function/class/interface name
        body: str — new body content
    """
    name = op.payload.get("name")
    new_body = op.payload.get("body", "")

    if not name:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="REPLACE_FUNCTION_BODY: missing 'name' in payload",
        )

    # Find symbol
    sym = ctx.get_symbol(name)
    if sym is None:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message=f"REPLACE_FUNCTION_BODY: '{name}' not found",
        )

    body_range = ctx.find_body_range(name)

    # ── Body pre-processing: strip full function definition ───────────
    _lang = ctx.language.value if hasattr(ctx, 'language') else ""
    _pre_stripped = _strip_full_function_def(new_body, name, language=_lang)
    new_body = _pre_stripped

    if body_range is None:
        # For non-brace-delimited symbols (e.g., variables), replace entire range
        new_code = ctx.replace_range(sym.start_byte, sym.end_byte, new_body)
        if new_code == ctx.code:
            return PrimitiveResult(
                success=True, code=ctx.code,
                message=f"REPLACE_FUNCTION_BODY: no-op for '{name}'",
            )
        return PrimitiveResult(
            success=True, code=new_code,
            message=f"Replaced variable/content of '{name}'",
            affected_range=(sym.start_byte, sym.start_byte + len(new_body)),
        )

    start, end = body_range

    # ── Detect actual body indentation from existing code ────────────────
    # Instead of hardcoding "  ", measure the indentation of the first
    # non-empty line in the existing body. This respects the file's actual
    # convention (4-space Python, 2-space TS, etc.).
    indent = ctx.symbol_indent(name)
    existing_body_text = ctx.slice(start, end)
    detected_indent = ""
    for line in existing_body_text.split("\n"):
        if line.strip():
            detected_indent = line[:len(line) - len(line.lstrip())]
            break
    body_indent = detected_indent if detected_indent else indent + "  "
    body_lines = new_body.strip().split("\n")
    indented_body = "\n".join(body_indent + line for line in body_lines)

    # For Python, body_range.start points to the first content line (after
    # the `:` and any blank lines), so replacement should be clean body
    # content without extra newline prefix that would create blank lines.
    # For brace-delimited languages, the range includes the leading newline
    # and trailing whitespace before `}`, so prefix/suffix are needed.
    if ctx.language == LanguageId.PYTHON:
        new_body_content = indented_body
        suffix = ""
    else:
        new_body_content = "\n" + indented_body
        suffix = "\n" + indent

    # ── Structural class member guard ────────────────────────────────
    _is_class = sym.kind == "class"
    if _is_class:
        lang_str = ctx.language.value
        from external_llm.languages.tree_sitter_utils import count_unique_class_members

        _old_counts = count_unique_class_members(ctx.code, name, lang_str)
        if _old_counts is not None:
            _old_methods, _old_fields = _old_counts
            _synth = f"class _ {{\n{new_body}\n}}"
            _new_counts = count_unique_class_members(_synth, "_", lang_str)
        else:
            _new_counts = None

        if _old_counts is not None and _new_counts is not None:
            _new_methods, _new_fields = _new_counts
            if _new_methods < _old_methods or _new_fields < _old_fields:
                return PrimitiveResult(
                    success=False, code=ctx.code,
                    message=(
                        f"REPLACE_FUNCTION_BODY: class '{name}' lost members — "
                        f"old had {_old_methods}m/{_old_fields}f, "
                        f"new has {_new_methods}m/{_new_fields}f. "
                        "Structural member count violation."
                    ),
                )

    # ── Build new code ─────────────────────────────────────────────────
    # For brace-delimited: replace between { and } with \nbody\nindent
    # For Python: replace first_content_line..last_content_line with indented body
    replacement = new_body_content + suffix
    new_code = ctx.replace_range(start, end, replacement)

    # ── No-op guard ──────────────────────────────────────────────────
    if new_code == ctx.code:
        return PrimitiveResult(
            success=True, code=ctx.code,
            message=f"REPLACE_FUNCTION_BODY: no-op — body for '{name}' unchanged",
        )

    return PrimitiveResult(
        success=True, code=new_code,
        message=f"Replaced body of '{name}'",
        affected_range=(start, start + len(indented_body)),
    )
