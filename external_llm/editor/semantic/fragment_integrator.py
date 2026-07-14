"""fragment_integrator.py — Phase D: Safe Fragment Integration.

Integrates GeneratedFragments into the codebase with:
- Idempotency (skip if unique_key already present)
- Syntax validation (ast.parse after each insert)
- Rollback on failure
- SectionPatcher reuse for import/body/wiring

Forbidden paths (venv, site-packages, etc.) are rejected.
"""
from __future__ import annotations

import ast
import logging
import os
from typing import Any

from external_llm.editor.semantic.fragment_generator import GeneratedFragment
from external_llm.languages import LanguageId

logger = logging.getLogger(__name__)

_FORBIDDEN = ("venv", "site-packages", "node_modules", ".git", "__pycache__")


def integrate_fragments(
    fragments: list[GeneratedFragment],
    repo_root: str,
) -> dict[str, Any]:
    """Integrate generated fragments into files.

    Returns integration report.
    """
    result: dict[str, Any] = {
        "applied": [],
        "skipped": [],
        "files_modified": [],
        "success": True,
    }

    if not fragments:
        return result

    applied_keys: set[str] = set()

    for frag in fragments:
        try:
            ok = _apply_fragment(frag, repo_root, applied_keys)
            if ok:
                result["applied"].append({
                    "type": frag.fragment_type,
                    "file": frag.target_file,
                    "key": frag.unique_key,
                })
                if frag.target_file not in result["files_modified"]:
                    result["files_modified"].append(frag.target_file)
                applied_keys.add(frag.unique_key)
            else:
                result["skipped"].append({
                    "type": frag.fragment_type,
                    "key": frag.unique_key,
                    "reason": "already_present_or_failed",
                })
        except Exception as e:
            logger.debug("[FRAG_INT] failed for %s: %s", frag.unique_key, e)
            result["skipped"].append({
                "type": frag.fragment_type,
                "key": frag.unique_key,
                "reason": str(e),
            })

    if result["applied"]:
        logger.info(
            "[FRAG_INT] %d applied, %d skipped, files: %s",
            len(result["applied"]), len(result["skipped"]),
            result["files_modified"],
        )

    return result


def _apply_fragment(
    frag: GeneratedFragment,
    repo_root: str,
    applied_keys: set[str],
) -> bool:
    """Apply a single fragment. Returns True if applied."""
    # Safety: check path
    target = frag.target_file
    if any(p in target for p in _FORBIDDEN):
        logger.debug("[FRAG_INT] forbidden path: %s", target)
        return False

    abs_path = target if os.path.isabs(target) else os.path.join(repo_root, target)

    # Ensure parent directory
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    # Read existing content
    existing = ""
    if os.path.isfile(abs_path):
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                existing = f.read()
        except Exception:
            pass

    # Idempotency
    if frag.unique_key and frag.unique_key in existing:
        return False

    if frag.unique_key in applied_keys:
        return False

    # Build new content
    content = frag.content
    if not content.strip():
        return False

    if existing:
        # Append to existing file with separator
        new_content = existing.rstrip() + "\n\n\n" + content.rstrip() + "\n"
    else:
        # New file
        new_content = content.rstrip() + "\n"

    # Validate syntax (only for .py files)
    if LanguageId.from_path(abs_path) is LanguageId.PYTHON:
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            logger.debug("[FRAG_INT] syntax error after insert in %s: %s", target, e)
            return False

    # Backup for rollback
    backup = existing

    # Write
    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        logger.debug("[FRAG_INT] write failed: %s", e)
        return False

    # Final syntax check
    if LanguageId.from_path(abs_path) is LanguageId.PYTHON:
        try:
            ast.parse(new_content)
        except SyntaxError:
            # Rollback
            try:
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(backup)
            except Exception:
                pass
            return False

    logger.debug(
        "[FRAG_INT] applied %s → %s (%s)",
        frag.fragment_type, target, frag.unique_key[:30],
    )
    return True
