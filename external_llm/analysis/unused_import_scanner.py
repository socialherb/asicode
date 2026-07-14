"""Unused import scanner — finds import statements whose names are never referenced.

AST-based deterministic analysis (Python-only — import semantics are
language-specific, and the stdlib AST is strictly more precise here than the
tree-sitter CST walk this module previously used):
  - Parses each .py file, collects all import names
  - Walks the AST for Load-context Name usages (excluding the import lines themselves)
  - Reports import names that never appear in a Load context

Exclusions (false positive prevention):
  - ``__all__`` re-exports (name is part of public API)
  - imports inside ``if TYPE_CHECKING:`` guard blocks (conditional, intentional)
  - ``from __future__ import ...`` (compile-time directives)
  - typing module symbols (``import_normalizer`` owns this domain)
  - ``import x as x`` / ``from m import x as x`` (PEP 484 explicit re-export)
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.languages import LanguageId as _LanguageId

from . import parse_cache

logger = logging.getLogger(__name__)


# Typing symbols owned by import_normalizer — unused import scanner must not
# flag these as dead; normalizer handles them deterministically.  Only applies
# to imports actually coming from the typing family (see scan loop) — a
# same-named symbol from another module is still a regular candidate.
_TYPING_MODULE_SYMBOLS: frozenset = frozenset({
    "Any", "Callable", "ClassVar", "Dict", "Final", "FrozenSet",
    "Generator", "Generic", "Iterable", "Iterator", "List", "Literal",
    "Mapping", "MutableMapping", "MutableSequence", "Optional", "Protocol",
    "Sequence", "Set", "Tuple", "Type", "TypeVar", "Union",
    "Annotated", "TypedDict", "NamedTuple", "cast", "overload",
    "runtime_checkable", "TYPE_CHECKING", "get_type_hints",
    "ParamSpec", "Concatenate", "TypeAlias", "Never",
})

_TYPING_MODULES: frozenset = frozenset({"typing", "typing_extensions"})


@dataclass
class UnusedImportCandidate:
    """One unused import statement."""
    file: str
    symbol_name: str           # imported name (alias-resolved)
    lineno: int                # line number of the import statement
    import_line_text: str      # raw import line for display
    kind: str = "unused_import"
    is_test_file: bool = False  # True when the import lives in a test file.
                                # Test fixtures/conftests carry a high false-positive
                                # rate (magic imports, pytest plugins), so callers can
                                # down-weight or bucket these separately.

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "symbol_name": self.symbol_name,
            "lineno": self.lineno,
            "import_line_text": self.import_line_text,
            "kind": self.kind,
            "is_test_file": self.is_test_file,
        }


def _is_test_path(file_path: str) -> bool:
    """Heuristic: is this a test file?

    Mirrors ``file_handlers._is_test_file`` but kept local to avoid an
    analysis→editor reverse dependency.  Criteria are intentionally simple
    (basename prefix + path segment) — this only drives a low-signal flag for
    callers to down-weight or bucket test-file candidates, never a correctness
    decision.
    """
    _basename = os.path.basename(file_path)
    if _basename.startswith(("test_", "Test")):
        return True
    _norm = str(file_path).replace("\\", "/")
    return "/test/" in _norm or "/tests/" in _norm


def _extract_all_names(tree: ast.Module) -> set:
    """Return set of names registered in ``__all__`` literal, if any.

    Identical logic to dead_block_scanner._extract_all_list.  Returns a
    sentinel ``{"*__dynamic__*"}`` when ``__all__`` is non-literal.
    """
    names: set = set()
    sentinel_dynamic = False
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "__all__":
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                value = node.value
        if value is None:
            continue
        for n in ast.walk(value):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                names.add(n.value)
            elif isinstance(n, ast.Name):
                sentinel_dynamic = True
    if sentinel_dynamic:
        names.add("*__dynamic__*")
    return names


def _is_type_checking_test(test: ast.expr) -> bool:
    """True for ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING`` conditions."""
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _collect_type_checking_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """Line ranges of ``if TYPE_CHECKING:`` guard blocks.

    Imports inside these ranges are intentionally conditional (often used only
    in quoted annotations the AST walk cannot see) and are never flagged.
    Imports OUTSIDE the guard are scanned normally.
    """
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            end = getattr(node, "end_lineno", node.lineno)
            ranges.append((node.lineno, end))
    return ranges


def _collect_import_info(tree: ast.Module, lines: list[str]) -> list:
    """Collect (local_name, line_text, lineno, module) for each import.

    ``module`` is the module name for ``ImportFrom`` (e.g. ``"os"``,
    ``"__future__"``), or ``None`` for ``import X`` statements.

    Skips PEP 484 explicit re-exports (``import x as x`` /
    ``from m import x as x``) — the redundant alias is the documented idiom
    for "this name is intentionally re-exported".
    """
    import_info: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname is not None and alias.asname == alias.name:
                    continue  # PEP 484 re-export idiom
                local_name = alias.asname or alias.name
                line_text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                import_info.append((local_name, line_text, node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None and alias.asname == alias.name:
                    continue  # PEP 484 re-export idiom
                # `import a.b.c` binds the first component (`a`) in the namespace.
                # Using split(".")[-1] was wrong: `import importlib.util` would yield
                # "util", which never appears as a standalone Name → false unused report.
                local_name = alias.asname or alias.name.split(".")[0]
                line_text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                import_info.append((local_name, line_text, node.lineno, None))
    return import_info


_ANNOTATION_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _collect_type_names_from_strings(tree: ast.Module) -> set:
    """Single-pass walk: extract type names from string annotations AND type-bearing call sites.

    Operates on two branches during a single tree traversal:

    1. **Annotation positions** — function returns, argument annotations,
       variable annotations, and PEP 695 type-alias statements that carry
       forward-reference string constants.
    2. **Call-site type names** — ``typing.cast("Foo", ...)`` first argument
       and ``TypeVar("T", ..., bound="Bar")`` positional/bound constraints.
    """
    names: set = set()
    _TypeAliasNode = getattr(ast, "TypeAlias", ())  # PEP 695 (3.12+)

    for node in ast.walk(tree):
        # ── Annotation positions (forward references as string literals) ──
        ann = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ann = node.returns
        elif isinstance(node, ast.AnnAssign):
            ann = node.annotation
            if _call_func_name(node.annotation) == "TypeAlias" and node.value is not None:
                _add_identifiers_from_annotation(node.value, names)
        elif isinstance(node, ast.arg):
            ann = node.annotation
        elif _TypeAliasNode and isinstance(node, _TypeAliasNode):
            _add_identifiers_from_annotation(node.value, names)
        _add_identifiers_from_annotation(ann, names)

        # ── Call-site type names (cast / TypeVar string args) ──
        if isinstance(node, ast.Call):
            _fname = _call_func_name(node.func)
            if _fname == "cast":
                if node.args:
                    _add_identifiers_from_annotation(node.args[0], names)
            elif _fname == "TypeVar":
                for _arg in node.args[1:]:
                    _add_identifiers_from_annotation(_arg, names)
                for _kw in node.keywords:
                    if _kw.arg == "bound":
                        _add_identifiers_from_annotation(_kw.value, names)

    return names




def _add_identifiers_from_annotation(ann: ast.AST | None, names: set) -> None:
    """Parse identifiers from *ann*, including nested string constants.

    Forward references appear as ``ast.Constant`` string literals, but they may
    be nested inside composite annotations such as ``list["Operation"]``
    (``ast.Subscript``) or ``dict[str, "Callable"]``. The outer annotation node
    is then a ``Subscript``, not a ``Constant``, so a naive ``isinstance``
    guard would miss the inner string. Walk the entire annotation subtree to
    find every string constant and extract the identifiers it binds, so a
    forward reference inside a generic is not mistaken for an unused import.
    """
    if ann is None:
        return
    for sub in ast.walk(ann):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            for m in _ANNOTATION_IDENTIFIER_RE.finditer(sub.value):
                name = m.group()
                # Dotted names (e.g. ``collections.abc.Callable``) in string
                # annotations are missed by ``_collect_load_names``. Add the first
                # component (``collections``) so the import isn't falsely flagged
                # as unused.
                names.add(name.split(".", 1)[0])


def _collect_load_names(tree: ast.Module) -> set:
    """Collect all Name nodes used in Load context (excluding import lines)."""
    used: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            val = node.value
            if isinstance(val, ast.Name):
                used.add(val.id)
    return used


def _call_func_name(func: ast.AST) -> str:
    """Return the trailing name of a call's func node.

    ``cast(...)`` → ``"cast"``; ``typing.cast(...)`` → ``"cast"``.  Used to
    recognise type-bearing runtime helpers regardless of whether the name is
    referenced directly or via a module attribute.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""




def scan_unused_imports(
    *,
    repo_root: str,
    file_paths: list[str],
    max_per_file: int = _cfg.counts.SCANNER_UNUSED_IMPORT_MAX,
) -> list[UnusedImportCandidate]:
    """Scan files for unused import statements.

    Args:
        repo_root: Repository root path (for resolving relative paths).
        file_paths: List of file paths (relative or absolute) to scan.
        max_per_file: Maximum unused imports to report per file.

    Returns:
        List of ``UnusedImportCandidate``, one per unused import name.
    """
    candidates: list[UnusedImportCandidate] = []
    truncated_files: list[str] = []

    for rel_path in file_paths or []:
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(repo_root or "", rel_path)

        # ── Determine language — import analysis is Python-specific ──
        _lang_id = _LanguageId.from_path(rel_path)
        _lang = _lang_id.value if _lang_id is not None else "python"
        if _lang != "python":
            continue

        source = parse_cache.read_source(abs_path)
        if source is None:
            continue
        tree = parse_cache.parse_ast(abs_path)
        if tree is None:
            logger.debug("[UNUSED_IMPORT] SyntaxError in %s — skipping", rel_path)
            continue
        lines = source.splitlines()
        all_names = _extract_all_names(tree)
        if "*__dynamic__*" in all_names:
            logger.debug("[UNUSED_IMPORT] %s has dynamic __all__ — conservative skip", rel_path)
            continue
        type_checking_ranges = _collect_type_checking_ranges(tree)
        import_info = _collect_import_info(tree, lines)
        if not import_info:
            continue
        load_names = _collect_load_names(tree)
        # Lazy string-annotation scan: only run the expensive full-tree walk when
        # there are imported names NOT already covered by load_names.  In most
        # files all imports resolve to Name nodes, making this scan pure waste.
        _import_names = {n for n, *_ in import_info}
        if _import_names - load_names - {"*"}:
            used_names = (
                load_names
                | _collect_type_names_from_strings(tree)
            )
        else:
            used_names = load_names

        emitted = 0
        for local_name, line_text, lineno, module in import_info:
            if local_name not in used_names and local_name != "*":
                # Skip __all__ re-exports
                if local_name in all_names:
                    continue
                # Skip imports inside TYPE_CHECKING guard blocks — they may be
                # used only in quoted annotations invisible to the Name walk.
                if any(start <= lineno <= end for start, end in type_checking_ranges):
                    continue
                # Skip from __future__ imports — compile-time directives
                if module == "__future__":
                    continue
                # Skip typing-family symbols — import_normalizer owns this domain
                if local_name in _TYPING_MODULE_SYMBOLS and module in _TYPING_MODULES:
                    logger.debug(
                        "[UNUSED_IMPORT] skipping typing symbol '%s' in %s (normalizer domain)",
                        local_name, rel_path,
                    )
                    continue

                candidates.append(UnusedImportCandidate(
                    file=rel_path,
                    symbol_name=local_name,
                    lineno=lineno,
                    import_line_text=line_text,
                    is_test_file=_is_test_path(rel_path),
                ))
                emitted += 1
                if emitted >= max_per_file:
                    # Hitting the configured limit is expected — keep DEBUG-only per file,
                    # then aggregate one line after scan to reduce terminal noise.
                    logger.debug(
                        "[UNUSED_IMPORT] %s: hit max_per_file=%d at name %r, truncating remaining",
                        rel_path, max_per_file, local_name,
                    )
                    truncated_files.append(rel_path)
                    break

    if truncated_files:
        logger.warning(
            "[UNUSED_IMPORT] truncated %d file(s) at max_per_file=%d (first: %s)",
            len(truncated_files), max_per_file, truncated_files[0],
        )
    if candidates:
        logger.info(
            "[UNUSED_IMPORT] %d unused import(s) across %d file(s)",
            len(candidates), len({c.file for c in candidates}),
        )

    return candidates
