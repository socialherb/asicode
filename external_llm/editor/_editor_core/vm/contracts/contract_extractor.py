"""contract_extractor.py — Extract function contracts from source code.

Reads source code via CodeContext (tree-sitter backed) and produces
FunctionContract for each function. Language-agnostic.

Ported from ts_vm/contract/contract_extractor.py with TSModule → CodeContext.
"""
from __future__ import annotations

from typing import Optional

from external_llm.editor._editor_core.vm.contracts.contract_models import FunctionContract, ParamContract
from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import SymbolDef
from external_llm.languages.models import LanguageId


def _guess_language(file_path: str) -> Optional[LanguageId]:
    """Guess language from file extension."""
    import os
    ext_map = {
        ".py": LanguageId.PYTHON,
        ".java": LanguageId.JAVA,
        ".kt": LanguageId.KOTLIN,
        ".kts": LanguageId.KOTLIN,
        ".go": LanguageId.GO,
        ".ts": LanguageId.TYPESCRIPT,
        ".js": LanguageId.JAVASCRIPT,
    }
    _, ext = os.path.splitext(file_path)
    return ext_map.get(ext.lower())



def _find_matching_paren(text: str, open_pos: int) -> int:
    """Find the matching closing parenthesis for the opening at open_pos.

    Handles nested parentheses correctly, unlike rfind().
    """
    depth = 0
    for i in range(open_pos, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


class ContractExtractor:
    """Extracts function contracts from source code.

    Uses CodeContext (tree-sitter) to find symbol definitions
    and parse parameter/return-type information.
    """

    def __init__(self, language: Optional[LanguageId] = None):
        self._language = language

    def extract_all(
        self, code: str, file_path: str = "",
    ) -> list[FunctionContract]:
        """Extract contracts for all functions in code."""
        lang = self._language or _guess_language(file_path)
        if lang is None:
            return []
        ctx = CodeContext(code, file_path, lang)
        functions = ctx.get_symbols_by_kind("function")
        methods = ctx.get_symbols_by_kind("method")
        symbols = functions + methods
        contracts = []
        for sym in symbols:
                contract = self._extract_from_symbol(ctx, sym, file_path)
                if contract:
                    contracts.append(contract)
        return contracts

    def _extract_from_symbol(
        self, ctx: CodeContext, sym: SymbolDef, file_path: str,
    ) -> Optional[FunctionContract]:
        """Extract a FunctionContract from a single symbol definition."""
        # Get the function signature (first line up to :)
        name = sym.name
        code = sym.body_start_byte is not None
        if code:
            sig_text = ctx.slice(sym.start_byte, sym.body_start_byte)
        else:
            sig_text = ctx.slice(sym.start_byte, sym.end_byte)

        # Parse params from signature
        params = self._parse_params(sig_text)

        return FunctionContract(
            name=name,
            params=params,
            return_type=self._extract_return_type(sig_text),
            file_path=file_path,
        )

    def _parse_params(self, sig_text: str) -> list[ParamContract]:
        """Parse parameters from a function signature text.

        Handles Python, Java, Kotlin, Go, TS/JS style signatures.
        Uses _find_matching_paren to avoid matching ')' in the body.
        """
        paren_open = sig_text.find("(")
        if paren_open == -1:
            return []
        paren_close = _find_matching_paren(sig_text, paren_open)
        if paren_close <= paren_open:
            return []

        params_str = sig_text[paren_open + 1:paren_close]
        if not params_str.strip():
            return []

        params: list[ParamContract] = []
        # Split by comma, respecting generics <>
        parts = self._split_params(params_str)

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Detect default value (=)
            has_default = "=" in part
            # Detect vararg (... , *args)
            is_vararg = part.startswith("*") or part.startswith("...")
            # Detect optional type annotation (? in TS/Kotlin)
            is_optional = "?" in part

            # Extract name and type
            name, type_name = self._extract_name_type(part)

            if name:
                # Strip * and ... prefix for vararg name
                clean_name = name.lstrip("*").lstrip(".")
                params.append(ParamContract(
                    name=clean_name,
                    type_name=type_name,
                    has_default=has_default or is_vararg,
                    is_optional=is_optional or is_vararg,
                    position=i,
                ))

        return params

    def _find_matching_paren(self, text: str, open_pos: int) -> int:
        """Find the matching closing parenthesis for the opening at open_pos.

        Handles nested parentheses correctly, unlike rfind().
        """
        depth = 0
        for i in range(open_pos, len(text)):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _split_params(self, params_str: str) -> list[str]:
        """Split by comma, respecting angle brackets and parens."""
        parts = []
        depth_angle = 0
        depth_paren = 0
        current = []

        for ch in params_str:
            if ch == "<":
                depth_angle += 1
            elif ch == ">":
                depth_angle -= 1
            elif ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren -= 1
            elif ch == "," and depth_angle == 0 and depth_paren == 0:
                parts.append("".join(current))
                current = []
                continue
            current.append(ch)

        if current:
            parts.append("".join(current))
        return parts

    def _extract_name_type(self, param_str: str) -> tuple:
        """Extract name and optional type from a single parameter string.

        Supports:
        - name: type  (Python/TS/Kotlin)
        - name type  (Go)
        - name       (Python/JS without type)
        - name: type = default (TS/Kotlin with default)
        - name=default (Python)
        - type name  (Java/Kotlin without colon)
        """
        _TYPE_KEYWORDS = {
            "int", "long", "double", "float", "boolean", "char",
            "byte", "short", "void", "String", "Int", "Long",
            "Double", "Float", "Boolean", "Char", "Byte", "Short",
            "Unit", "Any", "Nothing", "Any?", "Number", "BigInt",
            "never", "unknown", "undefined", "null", "symbol",
            "object", "Object", "Integer", "Character",
            "string", "bool", "rune", "uint", "uint8",
            "uint16", "uint32", "uint64", "int8", "int16", "int32",
            "int64", "float32", "float64", "complex64", "complex128",
            "error",
        }

        param_str = param_str.strip()

        # Remove default value if present
        default_idx = param_str.find("=")
        if default_idx > 0:
            param_str = param_str[:default_idx].strip()

        # If there's a colon, it's definitively name: type (Python/TS/Kotlin)
        if ":" in param_str:
            parts = param_str.split(":", 1)
            name = parts[0].strip()
            type_name = parts[1].strip() if len(parts) > 1 else None
            return name, type_name

        # No colon — split by whitespace to get parts
        parts = param_str.split(None, 1)
        if len(parts) == 2:
            first, second = parts
            # Heuristic: if first is a known type keyword or starts with
            # uppercase and second doesn't, it's Java style "type name"
            first_is_type = (
                first in _TYPE_KEYWORDS
                or (first[0].isupper() and not second[0].isupper())
            )
            if first_is_type:
                return second, first  # type name → name, type
            else:
                return first, second  # name type → name, type
        elif len(parts) == 1:
            return parts[0], None

        return "", None

    def _extract_return_type(self, sig_text: str) -> Optional[str]:
        """Extract return type from function signature."""
        # Python: "def foo() -> Type:"
        arrow_idx = sig_text.rfind("->")
        if arrow_idx > 0:
            after = sig_text[arrow_idx + 2:].strip()
            colon_idx = after.find(":")
            if colon_idx > 0:
                return after[:colon_idx].strip()
            return after.strip() or None

        # Java/Kotlin/Go/TS: "Type name(params)" or "name(params) Type"
        # Check if there's text after ) before the closing brace/:
        paren_close = sig_text.rfind(")")
        if paren_close > 0:
            after = sig_text[paren_close + 1:].strip()
            # Filter out punctuation (:, {, etc.)
            after = after.rstrip(":{")
            if after and not after.startswith(":"):
                return after.strip()

        return None
