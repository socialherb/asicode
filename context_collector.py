"""
Context collector for asicode.

Collects related source files for context expansion when sending prompts to the LLM.
Supports Python (import-based) and Kotlin (symbol-based) expansion.

FIXED: Uses common.unique_keep_order instead of local duplicate.
"""
from __future__ import annotations

import itertools
import logging
import re
from pathlib import Path
from typing import Optional

from common import normalize_rel_path_fast, unique_keep_order  # FIXED: use shared utility
from config import (
    CTX_MAX_FILES,
    KOTLIN_MAX_SYMBOLS,
    KOTLIN_SYMBOL_SCAN_LIMIT,
)

logger = logging.getLogger(__name__)


# ============================================================
# Text decoding / truncation hardening
# ============================================================
def _decode_bytes_best_effort(raw: bytes) -> tuple[str, str]:
    """
    Best-effort decode that avoids *silent* corruption:
    - Try strict decodes first (utf-8/utf-8-sig/cp949/euc-kr)
    - Fallback to utf-8 with replacement to keep visibility (no silent drop)
    Returns: (text, encoding_used)
    """
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return raw.decode(enc, errors="strict"), enc
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def _read_text_best_effort(fp: Path) -> tuple[str, str]:
    raw = fp.read_bytes()
    return _decode_bytes_best_effort(raw)


def _truncate_utf8_bytes_safe(s: str, max_bytes: int) -> tuple[str, bool]:
    """
    Truncate to max_bytes in UTF-8 *without* cutting multi-byte characters.
    Returns: (prefix, truncated?)
    """
    if max_bytes <= 0:
        return "", True
    b = s.encode("utf-8", errors="strict")
    if len(b) <= max_bytes:
        return s, False

    cut = b[:max_bytes]
    try:
        return cut.decode("utf-8", errors="strict"), True
    except UnicodeDecodeError as e:
        safe = cut[: max(0, int(e.start))]
        return safe.decode("utf-8", errors="strict"), True


# ============================================================
# Python import parsing
# ============================================================
# NOTE:
# - Support `import a, b.c as cc` (comma-separated) in addition to simple `import a.b`.
# - Keep the existing relative-import resolver for `from .x import y` and `from ..x import y`.
def _parse_python_imports(text: str, rel_path: str | None = None) -> list[str]:
    """
    Shallow parser: extract module strings from import/from-import.

    Improvement:
    - Resolve relative imports like `from .pkg import x` or `from ..pkg import x`
      using the current file's repo-relative path.
    - Support comma-separated `import a, b.c as cc` lines (best-effort).
    """
    mods: list[str] = []
    base_dir: Path | None = Path(rel_path).parent if rel_path else None

    def _normalize_one_module(mod0: str) -> str:
        mod = (mod0 or "").strip()
        if not mod:
            return ""
        # drop trailing "as alias"
        mod = mod.split()[0].strip()
        if not mod:
            return ""

        # Resolve relative module prefixes (e.g., .foo, ..bar, .)
        if mod.startswith(".") and base_dir is not None:
            dots = len(mod) - len(mod.lstrip("."))
            suffix = mod.lstrip(".")

            # Python semantics:
            #   from .x import ...  => current package (no ascent)
            #   from ..x import ... => parent package (ascent by 1)
            asc = max(0, dots - 1)
            bd = base_dir
            for _ in range(asc):
                # safety: don't loop forever
                if bd == bd.parent:
                    break
                bd = bd.parent

            # Convert filesystem path to module-like (pkg.sub)
            base_mod = str(bd).replace("\\", "/").strip("/")
            base_mod = base_mod.replace("/", ".") if base_mod else ""
            if suffix:
                mod = (base_mod + "." + suffix) if base_mod else suffix
            else:
                mod = base_mod

        # Normalize leading/trailing dots
        mod = mod.strip().strip(".")
        return mod

    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("from ") and " import " in stripped:
            raw_from = stripped[5:].split(" import ", 1)[0].strip()
            if raw_from:
                mod = _normalize_one_module(raw_from)
                if mod:
                    mods.append(mod)
        elif stripped.startswith("import "):
            raw_imp = stripped[7:].strip()
            if raw_imp:
                parts = [p.strip() for p in raw_imp.split(",") if p.strip()]
                for p in parts:
                    mod = _normalize_one_module(p)
                    if mod:
                        mods.append(mod)

    return unique_keep_order(mods)


def _module_to_repo_paths(repo_root: str, module: str) -> list[str]:
    """Convert Python module (pkg.mod) to candidate repo-relative file paths."""
    if not module:
        return []
    repo = Path(repo_root).resolve()
    m = module.strip().strip(".")
    if not m:
        return []
    parts = m.split(".")
    candidates = [
        str(Path(*parts).with_suffix(".py")).replace("\\", "/"),
        str(Path(*parts) / "__init__.py").replace("\\", "/"),
    ]
    if len(parts) == 1:
        candidates.append(str(Path(parts[0]) / "__init__.py").replace("\\", "/"))

    return unique_keep_order([
        r for r in candidates if (repo / r).exists() and (repo / r).is_file()
    ])


# ============================================================
# Kotlin import parsing
# ============================================================
# _parse_kotlin_import_symbols: line.startswith("import ") + split


def _parse_kotlin_import_symbols(text: str) -> list[str]:
    """Extract UpperCamelCase symbol names from Kotlin imports."""
    out: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("import "):
            continue
        rest = stripped[7:]  # remove "import "
        # Handle "as alias"
        as_idx = rest.rfind(" as ")
        if as_idx != -1:
            rest = rest[:as_idx]
        path = rest.strip()
        if not path or path.endswith(".*"):
            continue
        sym = path.split(".")[-1].strip()
        if sym and sym[0].isupper():
            out.append(sym)
    return unique_keep_order(out)


def _find_kotlin_files_for_symbol(
    repo_root: str, target_rel: str, symbol: str, limit: int = 8
) -> list[str]:
    """Find Kotlin files matching **/<symbol>.kt, preferring nearby paths."""
    if not symbol:
        return []
    repo = Path(repo_root).resolve()
    matches = list(itertools.islice(repo.rglob(f"{symbol}.kt"), limit * 4))
    if not matches:
        return []

    target_parts = Path(target_rel).parts if target_rel else ()
    scored: list[tuple[int, str]] = []
    for fp in matches[:max(1, int(limit))]:
        try:
            rel = fp.relative_to(repo)
        except ValueError:
            continue
        rels = str(rel).replace("\\", "/")
        common = sum(1 for a, b in zip(target_parts, rel.parts, strict=False) if a == b)
        scored.append((common, rels))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return unique_keep_order([r for _, r in scored])


# ============================================================
# Shallow multi-file context collector
# ============================================================
def collect_related_files_shallow(
    repo_root: str, target_rel: Optional[str]
) -> tuple[list[str], dict]:
    """
    Collect related files for context expansion.
    - Always includes target_rel
    - Python: follows imports
    - Kotlin: follows imported symbols to **.kt files
    """
    meta: dict = {
        "target": target_rel or "",
        "max_files": CTX_MAX_FILES,
        "kind": "",
        "candidates": [],
        "selected": [],
        "reason": "",
    }
    if not repo_root or not target_rel:
        meta["reason"] = "no_target"
        return [], meta

    # removeprefix("./") is exact-prefix; lstrip("./") is a character-SET
    # {'.','/'} that would strip a dotfile's leading dot (".config.py" ->
    # "config.py"), making the file appear missing.  lstrip("/") handles a
    # leading slash.  See go_provider.py for the same fix.
    rel = normalize_rel_path_fast(target_rel)
    if not rel:
        meta["reason"] = "empty_target"
        return [], meta

    repo = Path(repo_root).resolve()
    fp = repo / rel
    if not fp.exists() or not fp.is_file():
        meta["reason"] = "target_missing"
        return [], meta

    selected: list[str] = [rel]
    is_py = rel.endswith(".py")
    is_kt = rel.endswith(".kt")

    if not (is_py or is_kt):
        meta.update(kind="other", reason="unsupported_target_ext", selected=selected)
        return selected, meta

    try:
        text, enc_used = _read_text_best_effort(fp)
        meta["encoding_used"] = enc_used
    except Exception:
        # Keep contract: return (list[str], meta)
        meta["reason"] = "read_error"
        meta["selected"] = selected
        return selected, meta

    limit = max(1, CTX_MAX_FILES)

    if is_py:
        meta["kind"] = "py"
        mods = _parse_python_imports(text, rel)
        cand_paths: list[str] = []
        for mm in mods:
            cand_paths.extend(_module_to_repo_paths(repo_root, mm))
        uniq = [x for x in unique_keep_order(cand_paths) if x != rel]
        meta["candidates"] = uniq
        for x in uniq:
            if len(selected) >= limit:
                break
            selected.append(x)
    else:
        meta["kind"] = "kt"
        syms = _parse_kotlin_import_symbols(text)[:KOTLIN_MAX_SYMBOLS]
        cand: list[str] = []
        for s in syms:
            cand.extend(_find_kotlin_files_for_symbol(repo_root, rel, s, limit=KOTLIN_SYMBOL_SCAN_LIMIT))
        uniq = [x for x in unique_keep_order(cand) if x != rel]
        meta["candidates"] = uniq
        for x in uniq:
            if len(selected) >= limit:
                break
            selected.append(x)

    meta["selected"] = selected
    meta["reason"] = "ok"
    return selected, meta


def read_file_snippet_context(
    repo_root: str,
    rel_path: str,
    *,
    around_regex: str,
    window_lines: int = 120,
    max_bytes: int = 20000,
) -> tuple[str, dict]:
    """Read a small snippet of ONE file centered around a regex match."""
    meta: dict = {
        "included": False,
        "path": rel_path or "",
        "mode": "snippet_single",
        "around_regex": around_regex,
        "window_lines": int(window_lines),
        "max_bytes": int(max_bytes),
        "bytes_total": 0,
        "reason": "",
        "range": None,
    }

    reln = normalize_rel_path_fast(rel_path)
    if not repo_root or not reln:
        meta["reason"] = "missing_args"
        return "", meta

    fp = Path(repo_root).resolve() / reln
    if not fp.exists() or not fp.is_file():
        meta["reason"] = "missing_file"
        return "", meta

    try:
        text, enc_used = _read_text_best_effort(fp)
        meta["encoding_used"] = enc_used
    except Exception:
        meta["reason"] = "read_error"
        return "", meta

    lines = text.splitlines()
    hit = None
    try:
        rx = re.compile(around_regex)
        for i, ln in enumerate(lines):
            if rx.search(ln):
                hit = i
                break
    except Exception:
        pass

    if hit is None:
        hit = 0
        meta["reason"] = "regex_not_found_fallback_to_start"
    else:
        meta["reason"] = "ok"

    w = max(20, int(window_lines))
    lo = max(0, hit - w)
    hi = min(len(lines), hit + w + 1)
    body = "\n".join(lines[lo:hi]) + "\n"

    body2, truncated = _truncate_utf8_bytes_safe(body, int(max_bytes))
    if truncated:
        body = body2 + "\n\n[...TRUNCATED...]\n"
        meta["truncated"] = True
    else:
        body = body2
        meta["truncated"] = False

    meta["included"] = True
    meta["bytes_total"] = len(body.encode("utf-8", errors="strict"))
    meta["range"] = {"lo": lo + 1, "hi": hi}

    ctx = (
        "CONTEXT (read-only)\nMULTI_FILE: false\n-----\n"
        "CONTEXT FILE (read-only)\n"
        f"FILE: {reln}\n"
        "NOTE: Do NOT modify context text. Produce ONLY unified diff.\n"
        f"SNIPPET_LINES: {lo + 1}-{hi}\n"
        "----- BEGIN FILE -----\n"
        f"{body}"
        "----- END FILE -----\n"
    )
    return ctx, meta
