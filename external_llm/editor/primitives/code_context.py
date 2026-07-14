"""code_context.py — Language-agnostic CodeContext for primitives.

CodeContext wraps source code and provides language-agnostic access to
symbols, imports, call sites, and byte-level string operations.
It replaces TSModule+ExecutionContext dependency with a generic interface
backed by tree_sitter_utils (or regex fallback).
"""
from __future__ import annotations

import logging
from typing import Optional

from external_llm.editor.primitives.models import CallSite, ImportInfo, SymbolDef
from external_llm.languages.models import LanguageId

logger = logging.getLogger(__name__)


class CodeContext:
    """Generic code context for primitive execution.

    Provides the same byte-range operations as ts_vm's ExecutionContext
    but without TSModule dependency. Backed by tree_sitter_utils for
    multi-language symbol/import/call extraction.
    """

    def __init__(self, code: str, file_path: str, language: LanguageId):
        self.code = code
        self._code_bytes = code.encode("utf-8")
        self.file_path = file_path
        self.language = language

        # ── Lazy-populated caches ────────────────────────────────────
        self._symbols: Optional[dict[str, SymbolDef]] = None
        self._symbols_by_kind: Optional[dict[str, list[SymbolDef]]] = None
        self._imports: Optional[list[ImportInfo]] = None
        self._call_sites: Optional[list[CallSite]] = None
        # Dotted name resolution cache: "ClassName.methodName" → SymbolDef
        # Built alongside _symbols by scanning class bodies for methods.
        self._class_methods: Optional[dict[str, SymbolDef]] = None
        # Line number cache for byte-offset conversion
        self._line_starts: Optional[list[int]] = None

    # ── Language id helper ───────────────────────────────────────────

    def _lang_str(self) -> str:
        """Return tree-sitter-compatible language string."""
        return self.language.value

    # ── Byte-range operations (same interface as ts_vm ExecutionContext) ──

    def slice(self, start_byte: int, end_byte: int) -> str:
        """Extract source text at byte range."""
        return self._code_bytes[start_byte:end_byte].decode("utf-8")

    def replace_range(self, start_byte: int, end_byte: int, new_text: str) -> str:
        """Replace byte range with new_text. Returns updated full code."""
        new_bytes = new_text.encode("utf-8")
        result = (
            self._code_bytes[:start_byte]
            + new_bytes
            + self._code_bytes[end_byte:]
        )
        return result.decode("utf-8")

    def insert_at(self, byte_offset: int, text: str) -> str:
        """Insert text at byte offset. Returns updated full code."""
        text_bytes = text.encode("utf-8")
        result = (
            self._code_bytes[:byte_offset]
            + text_bytes
            + self._code_bytes[byte_offset:]
        )
        return result.decode("utf-8")

    def delete_range(self, start_byte: int, end_byte: int) -> str:
        """Delete byte range. Returns updated full code."""
        result = (
            self._code_bytes[:start_byte]
            + self._code_bytes[end_byte:]
        )
        return result.decode("utf-8")

    # ── Symbol access ───────────────────────────────────────────────

    def _ensure_symbols(self) -> None:
        """Lazy-build symbol index."""
        if self._symbols is not None:
            return
        self._symbols = {}
        self._symbols_by_kind = {}

        from external_llm.languages.tree_sitter_utils import find_all_symbols

        lang = self._lang_str()
        all_syms = find_all_symbols(self.code, lang)
        if not all_syms:
            # tree-sitter unavailable — try regex fallback
            self._extract_symbols_regex()
            # Still build dotted name map via provider (tree-sitter is unavailable
            # so _build_class_methods will be a no-op, but provider fallback works)
            self._class_methods = {}
            self._build_class_methods()
            self._build_class_methods_from_provider()
            return

        code_bytes = self._code_bytes

        for name, kind, start_line, end_line in all_syms:
            # Convert line ranges to byte ranges
            # start_line: start of the definition (1-indexed)
            # end_line: last line of the definition (1-indexed)
            start_byte = self._line_to_byte(start_line)
            # end_byte: byte after the last character on end_line.
            # _line_to_byte(end_line) gives the start of end_line; add the
            # byte length of end_line's content (excluding the newline).
            end_line_start = self._line_to_byte(end_line)
            # Byte length of end_line's content = distance to the next line
            # start (or EOF), minus 1 for the '\n' when one is present.
            next_start = self._line_to_byte(end_line + 1)
            if next_start <= len(code_bytes) and next_start > end_line_start \
                    and code_bytes[next_start - 1] == 0x0A:
                end_line_bytes = next_start - end_line_start - 1
            else:
                end_line_bytes = next_start - end_line_start
            end_byte = end_line_start + end_line_bytes
            if end_byte < start_byte:
                end_byte = start_byte + 1

            sd = SymbolDef(
                name=name,
                kind=kind,
                start_byte=start_byte,
                end_byte=end_byte,
            )

            # Find body range (brace-delimited body)
            body_range = self._find_brace_body_range(start_byte, end_byte)
            if body_range:
                sd.body_start_byte, sd.body_end_byte = body_range

            self._symbols[name] = sd
            self._symbols_by_kind.setdefault(kind, []).append(sd)

        # ── Build dotted name map (ClassName.methodName) ────────────
        self._class_methods = {}
        # Tree-sitter based: walks class_declaration/class_definition nodes
        self._build_class_methods()
        # Provider-based fallback: handles languages without class syntax
        # (Go receiver methods, Rust impl blocks, etc.) via provider.find_class_methods()
        self._build_class_methods_from_provider()

    def _build_class_methods_from_provider(self) -> None:
        """Use language provider's find_class_methods to fill dotted name gaps.

        Tree-sitter's _build_class_methods only handles languages with
        ``class_definition`` / ``class_declaration`` AST nodes (Python, TS/JS,
        Java, Kotlin). Languages that use other syntax (Go receiver methods,
        Rust impl blocks) are not covered.

        This fallback asks the provider for methods of each type/class/struct
        found in ``_symbols`` and registers ``TypeName.methodName`` → SymbolDef
        entries. Uses the provider's batch ``find_all_class_methods`` so the
        source is parsed at most once instead of once per class.
        """
        from external_llm.languages import LanguageRegistry

        provider = LanguageRegistry.instance().get(self.file_path)
        if provider is None:
            return

        # Collect type names from already-indexed symbols
        type_names = set()
        for name, sd in self._symbols.items():
            if sd.kind in ("type", "class", "struct", "interface"):
                type_names.add(name)

        if not type_names:
            return

        # Single batched call: parse the source once (per provider) and receive
        # methods for every class. Falls back to per-class calls only if the
        # provider does not provide a batch result.
        try:
            all_methods = provider.find_all_class_methods(self.code)
        except Exception:
            all_methods = None

        for type_name in type_names:
            # Skip types that already have methods via _build_class_methods
            # (avoid redundant work for class-based languages)
            prefix = type_name + "."
            if any(k.startswith(prefix) for k in self._class_methods):
                continue

            if all_methods is not None:
                methods = all_methods.get(type_name, [])
            else:
                try:
                    methods = provider.find_class_methods(self.code, type_name)
                except Exception:
                    continue
            if methods:
                self._index_methods_from_provider(type_name, methods)

    def _index_methods_from_provider(
        self, type_name: str, methods: list,
    ) -> None:
        """Register provider-found methods as ``TypeName.methodName`` → SymbolDef.

        Args:
            type_name: The receiver/parent type name (e.g. ``"TodoList"``).
            methods: List of ``(method_name, start_line, end_line)`` tuples
                     from ``provider.find_class_methods()``.
        """
        lines = self.code.split("\n")
        for method_name, start_line, end_line in methods:
            dotted = f"{type_name}.{method_name}"

            # Convert 1-indexed line numbers to byte offsets
            start_byte = self._line_to_byte(start_line)
            end_line_idx = end_line - 1  # 0-indexed
            if end_line_idx < len(lines):
                end_line_bytes = len(lines[end_line_idx].encode("utf-8"))
                end_byte = self._line_to_byte(end_line) + end_line_bytes
            else:
                end_byte = self._line_to_byte(end_line + 1) - 1
            if end_byte < start_byte:
                end_byte = start_byte + 1

            sd = SymbolDef(
                name=method_name,
                kind="method",
                start_byte=start_byte,
                end_byte=end_byte,
            )

            # Find body range (brace-delimited body)
            body_range = self._find_brace_body_range(start_byte, end_byte)
            if body_range:
                sd.body_start_byte, sd.body_end_byte = body_range

            # Register dotted name
            self._class_methods[dotted] = sd

            # Also register simple name (if not shadowed by another symbol)
            if method_name not in self._symbols:
                self._symbols[method_name] = sd
                self._symbols_by_kind.setdefault("method", []).append(sd)
            else:
                # Simple name already exists — log for debugging
                existing = self._symbols[method_name]
                if existing.kind != "method":
                    logger.debug(
                        "_index_methods_from_provider: %s.%s shadowed by "
                        "existing %s symbol '%s'",
                        type_name, method_name, existing.kind, method_name,
                    )

    def _build_class_methods(self) -> None:
        """Build a map of \"ClassName.methodName\" → SymbolDef for class methods."""
        from external_llm.languages.tree_sitter_utils import parse_to_tree

        lang = self._lang_str()
        tree = parse_to_tree(self.code, lang)
        if tree is None:
            return

        def _walk(node, current_class: str = "") -> None:
            # Python uses "class_definition", TS/Java uses "class_declaration"
            if node.type in ("class_definition", "class_declaration"):
                _name_node = node.child_by_field_name("name")
                if _name_node is not None:
                    name_text = _name_node.text
                    if name_text is not None:
                        current_class = name_text.decode("utf-8")
                else:
                    current_class = ""
                # Walk class body for methods
                for child in node.named_children:
                    _walk(child, current_class)
                return

            # Python uses "function_definition" for class methods;
            # TS/Java/Kotlin use "method_definition" or "method_declaration"
            if current_class and node.type in (
                "method_definition",         # TS/JS
                "method_declaration",         # Java/Kotlin
                "function_definition",        # Python (inside class body)
            ):
                _prop = node.child_by_field_name("name")
                if _prop is not None:
                        name_bytes = _prop.text
                        if name_bytes is not None:
                            method_name = name_bytes.decode("utf-8")
                            dotted = f"{current_class}.{method_name}"
                            # Create SymbolDef directly from tree-sitter node
                            # (find_all_symbols only returns top-level symbols,
                            # so class methods are NOT in self._symbols)
                            start_byte = node.start_byte
                            end_byte = node.end_byte
                            sd = SymbolDef(
                                name=method_name,
                                kind="method" if lang != "python" else "function",
                                start_byte=start_byte,
                                end_byte=end_byte,
                            )
                            self._class_methods[dotted] = sd
                            # Also index by simple name (only if not shadowed)
                            if method_name not in self._symbols:
                                self._symbols[method_name] = sd

            for child in node.named_children:
                _walk(child, current_class)

        _walk(tree.root_node)

    def _extract_symbols_regex(self) -> None:
        """Fallback: extract symbols using language provider's regex patterns."""
        from external_llm.languages import LanguageRegistry

        provider = LanguageRegistry.instance().get(self.file_path)
        if provider is None:
            return

        try:
            defs = provider.find_top_level_definitions(self.code)
        except Exception:
            return

        lines = self.code.split("\n")
        for name, kind, start_line, end_line in defs:
            start_byte = self._line_to_byte(start_line)
            end_line_idx = end_line - 1
            if end_line_idx < len(lines):
                end_line_bytes = len(lines[end_line_idx].encode("utf-8"))
                end_byte = self._line_to_byte(end_line) + end_line_bytes
            else:
                end_byte = self._line_to_byte(end_line + 1) - 1
            if end_byte < start_byte:
                end_byte = start_byte + 1

            sd = SymbolDef(
                name=name,
                kind=kind,
                start_byte=start_byte,
                end_byte=end_byte,
            )
            body_range = self._find_brace_body_range(start_byte, end_byte)
            if body_range:
                sd.body_start_byte, sd.body_end_byte = body_range

            self._symbols[name] = sd
            self._symbols_by_kind.setdefault(kind, []).append(sd)

    def _line_to_byte(self, line: int) -> int:
        """Convert 1-indexed line number to byte offset.

        Memoised: the cumulative per-line byte offsets (``_line_starts``) are
        computed once and reused. Previously this re-split the source and
        walked it from the start on every call — O(L) per call, called
        ~2× per symbol during indexing, which dominated large-file analysis.
        """
        starts = self._line_starts
        if starts is None:
            starts = self._build_line_starts()
        # line is 1-indexed; starts[0] is byte 0 (start of line 1).
        idx = line - 1
        if idx < len(starts):
            return starts[idx]
        return len(self._code_bytes)

    def _build_line_starts(self) -> list[int]:
        """Compute and cache the byte offset of the start of every line.

        Returns a list where element ``i`` is the byte offset of the start of
        the (0-indexed) i-th line. Built from the UTF-8 byte view so multibyte
        characters are handled correctly (matching the original
        ``len(l.encode('utf-8'))`` semantics).
        """
        # Scan the byte view once: every '\n' (0x0A) marks the end of a line,
        # so the following byte is the start of the next line. This is O(len)
        # and run exactly once per CodeContext instance.
        starts = [0]
        data = self._code_bytes
        append = starts.append
        for i, b in enumerate(data):
            if b == 0x0A:  # '\n'
                append(i + 1)
        self._line_starts = starts
        return starts

    def _find_brace_body_range(self, start_byte: int, end_byte: int) -> Optional[tuple[int, int]]:
        """Find the body byte range for a symbol.

        For brace-delimited languages (Java, Kotlin, Go, TS/JS): finds
        content between outer ``{`` and ``}``.
        For Python: uses indentation-based detection.
        """
        # Python uses indentation, not braces
        if self.language == LanguageId.PYTHON:
            return self._find_python_body_range(start_byte, end_byte)

        byte_slice = self._code_bytes[start_byte:end_byte]
        open_idx = byte_slice.find(b"{")
        close_idx = byte_slice.rfind(b"}")
        if open_idx == -1 or close_idx == -1 or open_idx >= close_idx:
            return None
        abs_open = start_byte + open_idx + 1
        abs_close = start_byte + close_idx
        return (abs_open, abs_close)

    def _find_python_body_range(self, start_byte: int, end_byte: int) -> Optional[tuple[int, int]]:
        """Find body byte range for Python (indentation-based).

        Python functions/classes have no braces — the body is the indented
        block after the ``:`` on the def/class line.
        Returns ``(body_start_byte, body_end_byte)`` suitable for replacement.
        """
        byte_slice = self._code_bytes[start_byte:end_byte]
        text = byte_slice.decode("utf-8")
        lines = text.split("\n")
        if not lines:
            return None

        # Find the signature line (def/class/async def ending with :)
        sig_line_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.endswith(":"):
                continue
            if (stripped.startswith("def ") or stripped.startswith("class ")
                    or stripped.startswith("async def ")):
                sig_line_idx = i
                break

        if sig_line_idx is None:
            return None

        sig_indent = len(lines[sig_line_idx]) - len(lines[sig_line_idx].lstrip())

        # Compute cumulative byte offsets for each line (relative to slice start)
        line_starts = []
        offset = 0
        for line in lines:
            line_starts.append(offset)
            offset += len(line.encode("utf-8")) + 1  # +1 for newline

        # Find first indented non-empty line (body start) and last indented line (body end)
        body_start_byte = None
        body_end_byte = None

        for i in range(sig_line_idx + 1, len(lines)):
            line = lines[i]
            raw = line.rstrip()
            if not raw.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= sig_indent:
                break  # dedent to/above signature level → end of body
            if body_start_byte is None:
                body_start_byte = start_byte + line_starts[i]
            body_end_byte = start_byte + line_starts[i] + len(raw.encode("utf-8"))

        if body_start_byte is None or body_end_byte is None:
            return None

        if body_end_byte < body_start_byte:
            body_end_byte = body_start_byte + 1

        return (body_start_byte, body_end_byte)

    def _resolve_dotted_name(self, name: str) -> Optional[SymbolDef]:
        """Resolve a dotted name like \"ClassName.methodName\".

        Returns the SymbolDef of the class method, or None.
        """
        if "." not in name:
            return None
        self._ensure_symbols()
        if self._class_methods:
            return self._class_methods.get(name)
        return None

    def get_symbol(self, name: str) -> Optional[SymbolDef]:
        """Find a symbol by name (supports dotted names like \"Class.method\")."""
        self._ensure_symbols()
        if name in self._symbols:
            return self._symbols[name]
        # Try dotted name resolution
        return self._resolve_dotted_name(name)

    def get_symbols_by_kind(self, kind: str) -> list[SymbolDef]:
        """Find all symbols of a given kind."""
        self._ensure_symbols()
        return self._symbols_by_kind.get(kind, [])

    def get_function(self, name: str) -> Optional[SymbolDef]:
        """Find a function symbol by name (supports dotted names)."""
        sym = self.get_symbol(name)
        if sym and sym.kind in ("function", "method"):
            return sym
        # Fallback: check all symbols (could be misclassified)
        return sym

    def get_class(self, name: str) -> Optional[SymbolDef]:
        """Find a class symbol by name."""
        sym = self.get_symbol(name)
        if sym and sym.kind == "class":
            return sym
        return None

    def find_body_range(self, name: str) -> Optional[tuple[int, int]]:
        """Find (body_start_byte, body_end_byte) for a named symbol."""
        sym = self.get_symbol(name)
        if sym is None:
            return None
        if sym.body_start_byte is not None and sym.body_end_byte is not None:
            return (sym.body_start_byte, sym.body_end_byte)
        # Fallback: search directly
        return self._find_brace_body_range(sym.start_byte, sym.end_byte)

    def symbol_indent(self, name: str) -> str:
        """Get the indentation string of a symbol's definition."""
        sym = self.get_symbol(name)
        if sym is None:
            return ""
        text = self.slice(sym.start_byte, sym.end_byte)
        indent = ""
        for ch in text:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break
        return indent

    # ── Import access ────────────────────────────────────────────────

    def _ensure_imports(self) -> None:
        """Lazy-build import index."""
        if self._imports is not None:
            return
        self._imports = []

        from external_llm.languages.tree_sitter_utils import query_matches

        lang = self._lang_str()
        from external_llm.languages.tree_sitter_utils import _IMPORT_QUERIES

        q = _IMPORT_QUERIES.get(lang)
        if q is None:
            return

        matches = query_matches(self.code, lang, q)
        for match_group in matches:
            source_caps = match_group.get("source", [])
            import_caps = match_group.get("import", [])
            if not source_caps:
                continue
            source_text = source_caps[0].text.strip().strip('"\';')
            start_byte = import_caps[0].start_byte if import_caps else source_caps[0].start_byte
            end_byte = import_caps[0].end_byte if import_caps else source_caps[0].end_byte

            self._imports.append(ImportInfo(
                source=source_text,
                start_byte=start_byte,
                end_byte=end_byte,
                statement=self.slice(start_byte, end_byte),
            ))

        # Sort by byte position
        self._imports.sort(key=lambda i: i.start_byte)

    def get_imports(self) -> list[ImportInfo]:
        """Get all import statements."""
        self._ensure_imports()
        return list(self._imports)

    def get_import_insertion_point(self) -> int:
        """Get byte offset after the last import statement."""
        self._ensure_imports()
        if not self._imports:
            return 0
        last = self._imports[-1]
        offset = last.end_byte
        # Skip past trailing newline
        while offset < len(self._code_bytes) and self._code_bytes[offset] in (ord("\n"), ord("\r")):
            offset += 1
        return offset

    # ── Call site access ─────────────────────────────────────────────

    def _ensure_call_sites(self) -> None:
        """Lazy-build call site index."""
        if self._call_sites is not None:
            return
        self._call_sites = []

        from external_llm.languages.tree_sitter_utils import query_matches

        lang = self._lang_str()
        from external_llm.languages.tree_sitter_utils import _CALL_QUERIES

        q = _CALL_QUERIES.get(lang)
        if q is None:
            return

        matches = query_matches(self.code, lang, q)
        for match_group in matches:
            callee_caps = match_group.get("callee", [])
            call_caps = match_group.get("call", [])
            if not callee_caps or not call_caps:
                continue
            cs = CallSite(
                callee=callee_caps[0].text,
                start_byte=call_caps[0].start_byte,
                end_byte=call_caps[0].end_byte,
            )
            # Resolve caller: find the enclosing function/class/variable
            self._resolve_caller(cs)
            self._call_sites.append(cs)

    def _resolve_caller(self, cs: CallSite) -> None:
        """Find the enclosing symbol name for a call site."""
        self._ensure_symbols()
        if not self._symbols:
            return
        caller = ""
        caller_range = 0
        for name, sym in self._symbols.items():
            if sym.start_byte <= cs.start_byte <= sym.end_byte:
                span = sym.end_byte - sym.start_byte
                if not caller or span < caller_range:
                    caller = name
                    caller_range = span
        cs.caller = caller

    def get_call_sites(self, callee: Optional[str] = None) -> list[CallSite]:
        """Get call sites, optionally filtered by callee name."""
        self._ensure_call_sites()
        if callee is None:
            return list(self._call_sites)
        return [cs for cs in self._call_sites if cs.callee == callee]

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def from_file_path(cls, code: str, file_path: str) -> "CodeContext":
        """Create a CodeContext from source code and file path.

        Language is inferred from file extension.
        """
        lang = LanguageId.from_path(file_path)
        return cls(code, file_path, lang)
