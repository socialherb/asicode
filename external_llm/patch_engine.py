"""
Unified patch intelligence engine for both service and agent paths.

This module centralizes all patch application, synthesis, and repair logic
that was previously duplicated between service.py (deterministic path) and
tool_registry.py (agent path).

Design principles:
1. Single source of truth for patch intelligence
2. Standardized metadata across both paths
3. Fallback ladder: git apply → AST rewrite → symbol search → semantic patch → file-block synthesis
4. Output mode synthesis: converts LLM outputs (UNIFIED_DIFF, FULL_FILE, ASICODE_BLOCK, TARGETED_BLOCK, PLAN_JSON) to unified diff
"""

import difflib
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from common import normalize_rel_path_fast

from .code_structure_utils import extract_symbol_name, is_python_definition

logger = logging.getLogger(__name__)

# Output mode enumeration
try:
    from .output_modes import OutputMode
except ImportError:
    logger.warning("output_modes module not found, OutputMode enum unavailable")
    OutputMode = None  # type: ignore

# File block parsing
try:
    from .output_parser import parse_file_blocks
except ImportError:
    logger.warning("output_parser module not found, parse_file_blocks unavailable")
    parse_file_blocks = None  # type: ignore

# ─── Patch Context ────────────────────────────────────────────────────────────

@dataclass
class PatchContext:
    """Optional context for patch application."""
    original_request: Optional[str] = None
    file_content: Optional[str] = None
    llm_output: Optional[str] = None
    output_mode: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

# ─── Patch Result ─────────────────────────────────────────────────────────────

@dataclass
class PatchResult:
    """Standardized result from patch application."""
    success: bool
    patch_applied: Optional[str] = None  # final unified diff applied
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Standardized metadata fields
    # reason: str - high-level success/failure reason
    # mode: str - which method succeeded (git_apply, ast_rewrite, symbol_search, semantic_patch, file_block_synth)
    # fallback_used: List[str] - sequence of fallbacks attempted
    # first_fail_reason: str - why git apply failed
    # second_fail_reason: str - why AST rewrite failed (if applicable)
    # synth_reason: str - why synthesis was needed
    # execution_steps: List[Dict] - detailed step-by-step log
    # normalized_patch: str - patch after normalization

# ─── Patch Engine ─────────────────────────────────────────────────────────────

class PatchEngine:
    """
    Unified patch intelligence engine.

    Provides:
    1. Patch application with full fallback ladder
    2. LLM output synthesis (multiple formats → unified diff)
    3. Patch normalization and validation
    4. Standardized metadata reporting
    """

    # Safety caps for auto-mode FILE rewrites (MVP)
    _MAX_FILE_CHARS = 250_000
    _MAX_PATCH_CHARS = 350_000
    _MAX_FILE_REWRITE_CHANGE_RATIO = 0.8  # reject if >80% of file changed
    _MAX_FILE_REWRITE_CHANGED_LINES = 1000  # reject if >1000 lines changed
    _MAX_FILE_RETRY_FILE_CHARS = 10_000  # maximum file size for FILE retry

    # Regex for parsing FILE blocks (legacy fallback)
    _RE_FILE_BLOCK = re.compile(
        r"(?ims)"
        r"(?:^|\n)\s*(?:FILE|Path|Target file)\s*:\s*(?P<path>[^\n\r]+?)\s*\r?\n"
        r"(?:```[^\n\r]*\r?\n(?P<code1>[\s\S]*?)\r?\n```|"
        r"(?P<code2>(?:(?!^\s*(?:FILE|Path|Target file)\s*:).*\r?\n)+))"
    )

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self._setup_components()

    def _setup_components(self):
        """Lazy import and setup of component modules."""
        # Diff applier (shared with agent path)
        try:
            from diff_apply import apply_patch as diff_apply_patch
            self._diff_apply = diff_apply_patch
        except ImportError:
            logger.warning("diff_apply module not found, patch application may be limited")
            self._diff_apply = None

        # AST rewriter
        try:
            from .ast_rewrite import ASTRewriter
            self.ast_rewriter = ASTRewriter(self.repo_root)
        except ImportError:
            logger.warning("ast_rewrite module not found, AST rewrite fallback disabled")
            self.ast_rewriter = None

        # Semantic patcher
        try:
            from .semantic_patch import SemanticPatchEngine
            self.semantic_patcher = SemanticPatchEngine(self.repo_root)
        except ImportError:
            logger.warning("semantic_patch module not found, semantic patch fallback disabled")
            self.semantic_patcher = None

        # Symbol searcher (already used by agent tools)
        try:
            from .agent.symbol_search import SymbolSearcher
            self.symbol_searcher = SymbolSearcher(self.repo_root)
        except ImportError:
            logger.warning("symbol_search module not found, symbol search fallback disabled")
            self.symbol_searcher = None

        # Patch synthesizer
        try:
            from .patch_synthesizer import PatchSynthesizer
            self.patch_synthesizer = PatchSynthesizer(self.repo_root)
        except ImportError:
            logger.warning("patch_synthesizer module not found, synthesis disabled")
            self.patch_synthesizer = None

        # Hybrid parser
        try:
            from .hybrid_parser import HybridOutputParser
            self.hybrid_parser = HybridOutputParser()
        except ImportError:
            logger.warning("hybrid_parser module not found, output mode parsing disabled")
            self.hybrid_parser = None

    def _output_mode_to_enum(self, output_mode: str) -> Optional[OutputMode]:
        """Convert output mode string to OutputMode enum."""
        if OutputMode is None:
            return None

        mode_map = {
            "auto": OutputMode.UNIFIED_DIFF,  # Default to unified diff for auto
            "diff": OutputMode.UNIFIED_DIFF,
            "full_file": OutputMode.FULL_FILE,
            "edit_blocks": OutputMode.ASICODE_BLOCK,
            "plan": OutputMode.PLAN_JSON,
        }
        return mode_map.get(output_mode, OutputMode.UNIFIED_DIFF)

    def apply_patch(self, patch_text: str, target_file: Optional[str] = None,
                    context: Optional[PatchContext] = None) -> PatchResult:
        """
        Apply a patch with full intelligence.

        Args:
            patch_text: Unified diff text (or patch candidate)
            target_file: Optional target file path (for validation)
            context: Optional patch context (original request, file content, etc.)

        Returns:
            PatchResult with success/failure and standardized metadata
        """
        # Initialize metadata
        metadata = {
            "reason": "",
            "mode": "",
            "fallback_used": [],
            "first_fail_reason": "",
            "second_fail_reason": "",
            "synth_reason": "",
            "execution_steps": [],
            "normalized_patch": patch_text,
        }

        # Step 1: Normalize patch (do NOT abort on check failure — try tolerant path)
        self._add_step(metadata, "normalize", "Normalizing patch candidate")
        normalized, norm_error = self.normalize_and_validate(patch_text, target_file)
        if norm_error:
            metadata["first_fail_reason"] = f"normalization: {norm_error}"
        metadata["normalized_patch"] = normalized

        # ALWAYS use the sanitized version — normalize_and_validate returns
        # the cleaned text even on preflight failure.  Using raw text would
        # discard BOM/fence/indent cleanup and make tolerant paths fail too.
        # This guard is critical: without it, tolerant/reanchor paths operate
        # on contaminated text, causing cascading failures (P1, CRITICAL).
        if not normalized and patch_text:
            # Sanitization stripped everything — log a warning but keep
            # the (empty) normalized value rather than falling back to raw.
            # Empty patch will fail fast in the apply step, which is safer
            # than applying unsanitized raw text.
            logger.warning(
                "[PATCH_ENGINE_P1] normalize_and_validate returned empty result "
                "for non-empty input (len=%d). Using sanitized (empty) rather "
                "than raw input to avoid contaminating tolerant/reanchor paths.",
                len(patch_text),
            )
        work_patch = normalized

        # Step 1b: Early exit — target file doesn't exist and patch is not a new-file creation.
        # No amount of reanchoring / fallback can fix a missing file; bail fast so the LLM
        # can immediately switch to create_file instead of exhausting all repair attempts.
        _patch_is_new_file = (
            "--- /dev/null" in work_patch
            or "@@ -0,0 " in work_patch
        )
        # Resolve check path: explicit target_file or parsed from patch header
        _check_target = target_file
        if not _check_target and not _patch_is_new_file:
            for _pl in work_patch.splitlines():
                if _pl.startswith("+++ b/"):
                    _check_target = _pl[6:].strip()
                    break
                elif _pl.startswith("+++ ") and not _pl.startswith("+++ /dev/null"):
                    _check_target = _pl[4:].strip()
                    break
        if not _patch_is_new_file and _check_target:
            _tf_full = Path(self.repo_root) / _check_target
            if not _tf_full.exists():
                metadata["reason"] = "file_not_found"
                metadata["mode"] = "early_exit"
                return PatchResult(
                    success=False,
                    patch_applied=None,
                    error=(
                        f"Target file does not exist: {_check_target}. "
                        f"Use the 'create_file' tool to create it first, "
                        f"or use '--- /dev/null' as the patch source header for new-file creation."
                    ),
                    metadata=metadata,
                )

        # Step 1c: Pre-apply git-state gate — classify the target's tracking state so we can
        # skip the guaranteed-to-fail 3-way merge for files that have no pre-image blob.
        # `git apply --3way` needs the patch's pre-image blob in the object store, which an
        # untracked/gitignored file never has and a freshly-edited file has only staled.
        # Plain `git apply` (non-3way) still works for these files, so we only skip 3-way,
        # not the whole pipeline. This avoids the "repository lacks the necessary blob"
        # failure *before* it happens and saves a wasted subprocess.
        _skip_3way = False
        _target_git_state = "unknown"
        if not _patch_is_new_file and _check_target:
            _target_git_state = self._classify_target_git_state(_check_target)
            metadata["target_git_state"] = _target_git_state
            if _target_git_state in ("untracked", "gitignored", "freshly_edited"):
                _skip_3way = True
            elif _target_git_state == "tracked":
                # Mode B: tracked+clean, but the patch's `index` line carries a
                # fabricated SHA (LLM cannot compute real git blob hashes). This
                # gate has NO effect on correctness: when the patch context
                # matches, `git apply --check` passes and the 3-way branch is
                # never even reached, so skip_3way is never consulted. Its only
                # value is in the *drift* case — where --check fails — it skips
                # a wasted `git apply --3way` subprocess that is guaranteed to
                # die with "repository lacks the necessary blob" because the
                # fabricated old-SHA is absent from the object store. Non-3way
                # variants patch purely by context-line matching.
                if self._patch_index_shas_are_fake(work_patch):
                    _skip_3way = True
                    metadata["skip_3way_reason"] = "fake_index_sha"

        # Step 2: Try git apply (primary path)
        self._add_step(metadata, "git_apply", "Attempting git apply")
        if not norm_error and self._diff_apply:
            try:
                _ga_ok, _ga_msg, _ga_reason, _ga_details = self._diff_apply(
                    self.repo_root, work_patch,
                    file_path_hint=target_file,
                    skip_3way=_skip_3way,
                )
                if _ga_ok:
                    metadata["reason"] = "git_apply_success"
                    metadata["mode"] = "git_apply"
                    return PatchResult(
                        success=True,
                        patch_applied=work_patch,
                        metadata=metadata
                    )
                else:
                    metadata["first_fail_reason"] = _ga_msg or _ga_reason or "git apply failed"
            except Exception as e:
                metadata["first_fail_reason"] = f"git apply exception: {e}"
        elif not self._diff_apply:
            metadata["first_fail_reason"] = "diff_apply module not available"

        # Step 2b: Tolerant git apply variants (fix hunk counts, whitespace, etc.)
        self._add_step(metadata, "tolerant_apply", "Trying tolerant git apply variants")
        tol_ok, _tol_err, tol_mode = self._tolerant_git_apply(work_patch, target_file, allow_3way=not _skip_3way)
        if tol_ok:
            metadata["reason"] = f"tolerant_apply_success:{tol_mode}"
            metadata["mode"] = f"tolerant_{tol_mode}"
            metadata["fallback_used"] = [tol_mode]
            return PatchResult(success=True, patch_applied=work_patch, metadata=metadata)

        # Step 2c: Exact-line re-anchoring (search for removed lines in actual file)
        self._add_step(metadata, "exact_reanchor", "Attempting exact-line re-anchor")
        reanchored = self._exact_reanchor_patch(work_patch, target_file)
        if not reanchored:
            # Fallback to fuzzy SequenceMatcher re-anchoring
            self._add_step(metadata, "reanchor", "Attempting fuzzy context re-anchor")
            reanchored = self._reanchor_patch(work_patch, target_file)
        if reanchored:
            tol_ok2, _tol_err2, tol_mode2 = self._tolerant_git_apply(reanchored, target_file, allow_3way=not _skip_3way)
            if tol_ok2:
                metadata["reason"] = f"reanchor_success:{tol_mode2}"
                metadata["mode"] = f"reanchor_{tol_mode2}"
                metadata["fallback_used"] = ["reanchor", tol_mode2]
                return PatchResult(success=True, patch_applied=reanchored, metadata=metadata)
            # Also try primary diff_apply on reanchored patch
            if self._diff_apply:
                try:
                    _ra_ok, _ra_msg, _ra_reason, _ra_details = self._diff_apply(
                        self.repo_root, reanchored,
                        skip_3way=_skip_3way,
                    )
                    if _ra_ok:
                        metadata["reason"] = "reanchor_git_apply_success"
                        metadata["mode"] = "reanchor_git_apply"
                        metadata["fallback_used"] = ["reanchor"]
                        return PatchResult(success=True, patch_applied=reanchored, metadata=metadata)
                except Exception as e:
                    logger.debug("PatchEngine: reanchored patch diff_apply failed: %s", e)

        # Step 3: Fallback ladder (AST / symbol / semantic / file-block)
        self._add_step(metadata, "repair", "Attempting repair ladder")
        llm_output = None
        if context and context.llm_output:
            llm_output = context.llm_output

        # Try repair ladder
        repair_result = self.repair_patch(
            patch_text=normalized,
            target_file=target_file,
            failure_reason=metadata["first_fail_reason"],
            llm_output=llm_output
        )

        # Merge metadata
        if repair_result.success:
            # Combine metadata
            merged_metadata = {**metadata, **repair_result.metadata}
            merged_metadata["reason"] = repair_result.metadata.get("reason", "repair_success")
            merged_metadata["mode"] = repair_result.metadata.get("mode", "unknown_repair")
            merged_metadata["fallback_used"] = repair_result.metadata.get("fallback_used", [])

            # Actually apply the repaired patch
            if repair_result.patch_applied:
                apply_ok, apply_err = self._apply_diff_once(repair_result.patch_applied, target_file)
                if apply_ok:
                    return PatchResult(
                        success=True,
                        patch_applied=repair_result.patch_applied,
                        metadata=merged_metadata
                    )
                else:
                    # Repaired patch failed to apply
                    merged_metadata["reason"] = "repaired_patch_apply_failed"
                    merged_metadata["second_fail_reason"] = apply_err
                    return PatchResult(
                        success=False,
                        error=f"Repaired patch failed to apply: {apply_err}",
                        metadata=merged_metadata
                    )
            else:
                # No patch produced (should not happen)
                merged_metadata["reason"] = "repaired_patch_missing"
                return PatchResult(
                    success=False,
                    error="Repair succeeded but no patch produced",
                    metadata=merged_metadata
                )
        else:
            metadata["reason"] = "repair_failed"
            metadata["fallback_used"] = repair_result.metadata.get("fallback_used", [])
            metadata["second_fail_reason"] = repair_result.metadata.get("error", "repair failed")
            _final_err = f"Patch application failed and repair attempts exhausted: {metadata['first_fail_reason']}"
            # Actionable guidance for the known blob-deficient states. The 3-way merge
            # path (which the repair ladder would otherwise rely on) cannot find a
            # pre-image blob for these, so steer the caller to a tool that works without
            # one. This turns a confusing "lacks the necessary blob" into a concrete
            # next step.
            if _target_git_state == "freshly_edited":
                _final_err += (
                    f" (target '{_check_target}' is freshly-edited: its working tree differs "
                    f"from git, so 3-way merge has no usable pre-image blob. Re-read the file "
                    f"and regenerate the patch against its CURRENT content, or use "
                    f"'modify_symbol'/'edit_text' for a single-symbol change.)"
                )
            elif _target_git_state in ("untracked", "gitignored"):
                _final_err += (
                    f" (target '{_check_target}' is {_target_git_state}: git has no pre-image "
                    f"blob for it, so 3-way merge cannot work. Use 'modify_symbol' or "
                    f"'edit_text' instead, or stage the file with 'git add' first.)"
                )
            return PatchResult(
                success=False,
                error=_final_err,
                metadata=metadata
            )

    def synthesize_and_apply(self, llm_output: str, target_file: str,
                             output_mode: str = "auto") -> PatchResult:
        """
        Parse LLM output, synthesize diff, then apply.

        Args:
            llm_output: Raw LLM output (could be diff, full file, edit blocks, etc.)
            target_file: Target file path
            output_mode: Output mode hint ("auto", "diff", "full_file", "edit_blocks", "plan")

        Returns:
            PatchResult with success/failure
        """
        metadata = {
            "reason": "",
            "mode": "",
            "fallback_used": [],
            "first_fail_reason": "",
            "second_fail_reason": "",
            "synth_reason": "",
            "execution_steps": [],
            "normalized_patch": "",
        }

        # Step 1: Parse LLM output
        self._add_step(metadata, "parse", f"Parsing LLM output with mode={output_mode}")
        if self.hybrid_parser and self.patch_synthesizer and OutputMode is not None:
            try:
                # Convert output_mode string to enum
                expected_mode = self._output_mode_to_enum(output_mode)
                if expected_mode is None:
                    metadata["first_fail_reason"] = "output_mode enum not available"
                    return PatchResult(
                        success=False,
                        error="Output mode enumeration not available",
                        metadata=metadata
                    )

                # Parse LLM output using hybrid parser
                parsed = self.hybrid_parser.parse(llm_output, expected_mode)
                if not parsed.success:
                    metadata["first_fail_reason"] = f"parse failed: {parsed.error}"
                    return PatchResult(
                        success=False,
                        error=f"Failed to parse LLM output: {parsed.error}",
                        metadata=metadata
                    )

                # Check for NEEDS_DISAMBIGUATION
                if parsed.mode is None:
                    # This indicates NEEDS_DISAMBIGUATION
                    metadata["synth_reason"] = "needs_disambiguation"
                    return PatchResult(
                        success=False,
                        error="LLM output requires disambiguation",
                        metadata=metadata
                    )

                # Synthesize unified diff
                synthesized = self.patch_synthesizer.synthesize(parsed, target_file)
                metadata["synth_reason"] = f"parsed mode={parsed.mode.value}"
                metadata["normalized_patch"] = synthesized

                # Step 2: Apply the synthesized patch
                context = PatchContext(
                    original_request=None,
                    file_content=None,
                    llm_output=llm_output,
                    output_mode=output_mode,
                    metadata={"parsed_mode": parsed.mode.value}
                )
                return self.apply_patch(synthesized, target_file, context)
            except Exception as e:
                metadata["first_fail_reason"] = f"synthesis failed: {e}"
                return PatchResult(
                    success=False,
                    error=f"LLM output synthesis failed: {e}",
                    metadata=metadata
                )
        else:
            missing = []
            if not self.hybrid_parser:
                missing.append("hybrid_parser")
            if not self.patch_synthesizer:
                missing.append("patch_synthesizer")
            if OutputMode is None:
                missing.append("output_modes")
            metadata["first_fail_reason"] = f"components not available: {', '.join(missing)}"
            return PatchResult(
                success=False,
                error=f"LLM output synthesis components not available: {', '.join(missing)}",
                metadata=metadata
            )

    def _auto_repair_patch(self, patch: str, target_file: str) -> Optional[str]:
        """
        Attempt AST-based repair of a failing patch.
        """
        try:
            from external_llm.ast_rewrite import ASTRewriter

            rewriter = ASTRewriter(self.repo_root)

            file_path = target_file
            if not file_path:
                return None

            path = Path(self.repo_root) / file_path
            if not path.exists():
                return None

            path.read_text(encoding="utf-8")

            # Improved diff parsing: handle unified diff format properly
            lines = patch.splitlines()
            new_lines = []
            in_hunk = False
            for line in lines:
                if line.startswith("@@"):
                    in_hunk = True
                    continue
                if in_hunk and line.startswith("+"):
                    # Skip header lines (+++ b/file)
                    if not line.startswith("+++"):
                        new_lines.append(line[1:])  # Remove leading '+'

            new_code = "\n".join(new_lines).strip()

            if not new_code:
                return None

            header = new_code.lstrip()

            symbol_name, symbol_kind = extract_symbol_name(header)
            if symbol_name:
                if symbol_kind == "function":
                    result = rewriter.replace_function(
                        file_path,
                        symbol_name,
                        new_code
                    )
                elif symbol_kind == "class":
                    result = rewriter.replace_class(
                        file_path,
                        symbol_name,
                        new_code
                    )
                else:
                    result = None

                if result is not None:
                    return rewriter.generate_patch(file_path, result)

        except Exception as e:
            logger.debug("AST repair failed for %s: %s", target_file, e)
            return None

        return None


    def _try_synthesize_diff_from_file_blocks(
        self,
        repo_root: str,
        target_file: str,
        llm_text: str,
    ) -> tuple[str, str]:
        """
        Parse full-file rewrite blocks from LLM output and synthesize a unified diff.

        MVP guardrails:
          - target file must exist
          - only ONE file is allowed (must match target_file; basename match allowed as fallback)
          - cap file/patched size
          - output is still validated later via validate_diff(..., target_file=target_file)
        """
        rr = Path(repo_root).resolve()
        tgt_rel = normalize_rel_path_fast(str(target_file))
        tgt_path = (rr / tgt_rel).resolve()

        if not tgt_path.exists() or not tgt_path.is_file():
            return ("", "target_missing")

        try:
            old_text = tgt_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ("", "read_failed")

        # Parse blocks via the canonical parser
        parsed_blocks: list[dict[str, str]] = []
        try:
            parsed_blocks = parse_file_blocks(llm_text or "")
        except Exception:
            parsed_blocks = []

        # Normalize blocks to (rel_path, code)
        blocks: list[tuple[str, str]] = []
        for b in parsed_blocks or []:
            p = str(b.get("path") or "").strip().strip('"').strip("'")
            if not p:
                continue
            rel = normalize_rel_path_fast(p)
            code = b.get("text") or b.get("content") or ""
            code = self._strip_trailing_fences(str(code))
            blocks.append((rel, code))

        # (optional) fallback parse using legacy regex if canonical parser returns nothing
        if not blocks:
            for m in self._RE_FILE_BLOCK.finditer(llm_text or ""):
                p = (m.group("path") or "").strip().strip('"').strip("'")
                code = m.group("code1")
                if code is None:
                    code = m.group("code2") or ""
                code = self._strip_trailing_fences(str(code))
                rel = normalize_rel_path_fast(p)
                if rel:
                    blocks.append((rel, code))

        if not blocks:
            return ("", "no_file_block")

        # Pick best match: exact target path, else basename match
        chosen_rel: Optional[str] = None
        new_text: Optional[str] = None

        for rel, code in blocks:
            if rel == tgt_rel:
                chosen_rel, new_text = rel, code
                break

        if new_text is None:
            tgt_base = Path(tgt_rel).name
            for rel, code in blocks:
                if Path(rel).name == tgt_base:
                    chosen_rel, new_text = rel, code
                    break

        if new_text is None:
            return ("", "no_target_file_block")

        # Guard: reject if there are other file blocks besides the chosen one
        other_files = [rel for (rel, _c) in blocks if rel != chosen_rel]
        if other_files:
            return ("", "multi_file_block")

        # Normalize new text to ensure trailing newline (git-style)
        if not new_text.endswith("\n"):
            new_text = new_text + "\n"

        if len(new_text) > int(self._MAX_FILE_CHARS):
            return ("", "file_too_large")

        if old_text == new_text:
            return ("", "no_changes")

        # Safety valve: reject over-large rewrites.
        # FILE blocks are full rewrites; if the model drifts and rewrites most of the file,
        # we would rather fail than apply a huge, hard-to-review patch.
        try:
            sm = difflib.SequenceMatcher(a=old_text, b=new_text)
            change_ratio = 1.0 - float(sm.ratio())
        except Exception:
            change_ratio = 1.0

        try:
            old_lines = old_text.splitlines()
            new_lines = new_text.splitlines()
            max_lines = max(len(old_lines), len(new_lines), 1)
            # Approx: changed lines ~= (1 - similarity) * max(lines)
            changed_lines_est = round(change_ratio * max_lines)
        except Exception:
            changed_lines_est = 10**9

        if (change_ratio > float(self._MAX_FILE_REWRITE_CHANGE_RATIO)) or (changed_lines_est > int(self._MAX_FILE_REWRITE_CHANGED_LINES)):
            return ("", "file_rewrite_too_large")

        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(True),
                new_text.splitlines(True),
                fromfile=f"a/{tgt_rel}",
                tofile=f"b/{tgt_rel}",
                lineterm="",
            )
        )
        patch_body = "\n".join(diff_lines).rstrip() + "\n"

        # Ensure synthesized diffs ALWAYS include `diff --git` header (stability + tooling friendliness).
        # difflib.unified_diff() emits only ---/+++ headers by default.
        patch = patch_body
        if patch_body and (not patch_body.startswith("diff --git ")):
            patch = f"diff --git a/{tgt_rel} b/{tgt_rel}\n" + patch_body

        if len(patch) > int(self._MAX_PATCH_CHARS):
            return ("", "patch_too_large")

        return (patch, "file_block_synth")

    @staticmethod
    def _strip_trailing_fences(s: str) -> str:
        t = str(s or "")
        # If the model accidentally included an ending fence inside unfenced capture, trim it.
        t = re.sub(r"\n```[\s\S]*\Z", "\n", t)
        return t

    def _salvage_small_model_output(self, patch_text: str, target_file: str) -> Optional[str]:
        """
        Enhanced patch synthesis for small/local LLMs with malformed outputs.
        Moved from tool_registry.py _synthesize_simple_diff.

        Supported cases:
        1) Simple "-old / +new" replacement (single or multi-line)
        2) Ed-like / malformed insert style (1c1, 2a, etc.)
        3) Partial before/after blocks with fuzzy matching
        4) Code blocks with markdown fences
        5) Symbol-aware line number correction
        6) Guided editing suggestions for common failure patterns
        """
        if not target_file:
            return None

        try:
            import difflib
            import re
            from pathlib import Path

            rel = str(target_file).strip().lstrip("/")
            if not rel:
                return None

            abs_path = Path(self.repo_root) / rel
            if not abs_path.exists() or not abs_path.is_file():
                return None

            old_text = abs_path.read_text(encoding="utf-8", errors="replace")
            old_lines = old_text.splitlines()

            # --- Phase 1: Parse patch text for various patterns ---

            # Pattern 1: Simple +/- lines (original logic, extended)
            added_lines = []
            removed_lines = []

            # Pattern 2: Before/after blocks (common in malformed LLM output)
            before_blocks = []
            after_blocks = []

            # Pattern 3: Ed-style commands (1c1, 2a, 3d, etc.)
            ed_commands = []

            # Pattern 4: Code blocks with markdown fences
            code_blocks = []

            lines = patch_text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].rstrip()

                # Simple +/- lines (skip diff headers)
                if line.startswith("+") and not line.startswith("+++"):
                    added_lines.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    removed_lines.append(line[1:])

                # Before/after block detection (case-insensitive)
                elif re.match(r'^before\s*:', line, re.IGNORECASE):
                    before_content = []
                    i += 1
                    while i < len(lines) and not re.match(r'^after\s*:', lines[i], re.IGNORECASE):
                        before_content.append(lines[i])
                        i += 1
                    if i < len(lines) and re.match(r'^after\s*:', lines[i], re.IGNORECASE):
                        i += 1
                        after_content = []
                        while i < len(lines) and lines[i].strip() and not re.match(r'^before\s*:|^after\s*:', lines[i], re.IGNORECASE):
                            after_content.append(lines[i])
                            i += 1
                        before_blocks.append("\n".join(before_content))
                        after_blocks.append("\n".join(after_content))
                        continue

                # Ed-style commands (e.g., "1c1", "2a", "3d")
                elif re.match(r'^\d+[acd]\d*$', line):
                    ed_commands.append(line)

                # Markdown code fences
                elif line.strip().startswith("```"):
                    i += 1
                    code_content = []
                    while i < len(lines) and not lines[i].strip().startswith("```"):
                        code_content.append(lines[i])
                        i += 1
                    code_blocks.append("\n".join(code_content))

                i += 1

            # --- Phase 2: Try each synthesis strategy in order of reliability ---

            # Strategy 1: Multi-line replacement with fuzzy matching (line-level)
            if removed_lines and added_lines:
                n_removed = len(removed_lines)
                if len(old_lines) >= n_removed:
                    # Sliding window: find best line-level match for removed_lines in old_lines.
                    # A full-block match is required (no partial char-level matches) to avoid
                    # silent corruption (e.g. "return 99" + leftover "1" = "return 991").
                    best_idx = None
                    best_ratio = 0.0
                    for start_idx in range(len(old_lines) - n_removed + 1):
                        window = old_lines[start_idx:start_idx + n_removed]
                        sm = difflib.SequenceMatcher(None, window, removed_lines)
                        ratio = sm.ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_idx = start_idx

                    # Require at least 60% line-level match (prevents partial corruption)
                    if best_idx is not None and best_ratio >= 0.6:
                        new_lines = list(old_lines)
                        new_lines[best_idx:best_idx + n_removed] = added_lines

                        diff_lines = list(difflib.unified_diff(
                            old_lines,
                            new_lines,
                            fromfile=f"a/{rel}",
                            tofile=f"b/{rel}",
                            lineterm="",
                        ))

                        if diff_lines:
                            diff_text = "\n".join(diff_lines)
                            if not diff_text.startswith("diff --git "):
                                diff_text = f"diff --git a/{rel} b/{rel}\n" + diff_text
                            if not diff_text.endswith("\n"):
                                diff_text += "\n"
                            return diff_text

            # Strategy 2: Before/after blocks with fuzzy matching (line-level)
            if before_blocks and after_blocks and len(before_blocks) == len(after_blocks):
                new_lines = list(old_lines)
                all_succeeded = True

                for before_block, after_block in zip(before_blocks, after_blocks, strict=False):
                    before_lines = before_block.splitlines()
                    after_lines = after_block.splitlines()
                    n_before = len(before_lines)
                    if n_before == 0:
                        all_succeeded = False
                        break

                    # Exact match first (line-level)
                    found = False
                    for start_idx in range(len(new_lines) - n_before + 1):
                        if new_lines[start_idx:start_idx + n_before] == before_lines:
                            new_lines[start_idx:start_idx + n_before] = after_lines
                            found = True
                            break

                    if not found:
                        # Fuzzy match using line-level SequenceMatcher (full-block only)
                        best_idx = None
                        best_ratio = 0.0
                        for start_idx in range(len(new_lines) - n_before + 1):
                            window = new_lines[start_idx:start_idx + n_before]
                            sm = difflib.SequenceMatcher(None, window, before_lines)
                            ratio = sm.ratio()
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_idx = start_idx

                        if best_idx is not None and best_ratio >= 0.7:
                            new_lines[best_idx:best_idx + n_before] = after_lines
                        else:
                            all_succeeded = False
                            break

                if all_succeeded:
                    diff_lines = list(difflib.unified_diff(
                        old_lines,
                        new_lines,
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                        lineterm="",
                    ))

                    if diff_lines:
                        diff_text = "\n".join(diff_lines)
                        if not diff_text.startswith("diff --git "):
                            diff_text = f"diff --git a/{rel} b/{rel}\n" + diff_text
                        if not diff_text.endswith("\n"):
                            diff_text += "\n"
                        return diff_text

            # Strategy 3: Symbol-aware insertion for added lines
            if added_lines:
                # Filter out lines already present
                old_lines_set = set(old_lines)  # O(n²) → O(n) : set lookup is O(1)
                insert_lines = [ln for ln in added_lines if ln.strip() and ln not in old_lines_set]
                if not insert_lines:
                    return None

                # Try to find insertion point using symbol search if available
                anchor_index = None
                try:
                    from .agent.symbol_search import SymbolSearcher
                    searcher = SymbolSearcher(self.repo_root)

                    # Look for function/class definitions in added lines
                    for line in insert_lines:
                        # Simple heuristic: look for "def " or "class " in Python
                        if "def " in line or "class " in line:
                            symbol = line.split("def ")[-1].split("class ")[-1].split("(")[0].strip()
                            results = searcher.search(symbol, limit=5)
                            if results:
                                # Find the line after the last occurrence
                                for result in results:
                                    if result.get("file_path") == rel:
                                        line_num = result.get("line_number", 0)
                                        if line_num > 0:
                                            anchor_index = line_num - 1  # Convert to 0-index
                                            break
                                if anchor_index is not None:
                                    break
                except ImportError:
                    pass  # Symbol search not available, fall back to heuristics

                # Fallback heuristics (original logic)
                if anchor_index is None:
                    # Heuristics for HTML/UI files
                    for i, ln in enumerate(old_lines):
                        stripped = ln.strip().lower()
                        if stripped.startswith("<html"):
                            anchor_index = i
                            break

                if anchor_index is None:
                    for i, ln in enumerate(old_lines):
                        if ln.strip().lower().startswith("<!doctype"):
                            anchor_index = i + 1
                            break

                if anchor_index is None:
                    anchor_index = 0

                # Insert lines
                new_lines = list(old_lines)
                for offset, ins in enumerate(insert_lines):
                    new_lines.insert(anchor_index + offset, ins)

                new_text = "\n".join(new_lines)
                if old_text.endswith("\n"):
                    new_text += "\n"

                diff_lines = list(difflib.unified_diff(
                    old_lines,
                    new_text.splitlines(),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                    lineterm="",
                ))

                if not diff_lines:
                    return None

                diff_text = "\n".join(diff_lines)
                if not diff_text.startswith("diff --git "):
                    diff_text = f"diff --git a/{rel} b/{rel}\n" + diff_text
                if not diff_text.endswith("\n"):
                    diff_text += "\n"
                return diff_text

            # Strategy 4: Code blocks from markdown fences
            if code_blocks:
                # Try to interpret code block as replacement for entire file or function
                # This is a simple heuristic: if code block looks like a complete function/class,
                # try to replace the existing one
                for code_block in code_blocks:
                    # Check if code block contains function/class definition
                    lines = code_block.splitlines()
                    for i, line in enumerate(lines):
                        if is_python_definition(line):
                            # Try to find matching function/class in original file
                            _sym_name, _ = extract_symbol_name(line)
                            symbol_name = _sym_name or line.split("(")[0].strip()
                            # Simple line-by-line search for the symbol
                            for _j, old_line in enumerate(old_lines):
                                if symbol_name in old_line and is_python_definition(old_line):
                                    # Found a match, try replacement
                                    # This is a simplistic approach - in reality would need AST parsing
                                    # For now, fall back to insertion strategy
                                    pass

            # Strategy 5: Ed-style command parsing (basic implementation)
            if ed_commands:
                # Simple ed command interpreter (very basic)
                # Format: <line_number><command><optional_parameter>
                # c = change, a = append after, d = delete
                new_lines = list(old_lines)
                changes_made = False

                for cmd in ed_commands:
                    match = re.match(r'^(\d+)([acd])(\d*)$', cmd)
                    if match:
                        line_num = int(match.group(1))
                        command = match.group(2)
                        # Convert 1-indexed to 0-indexed
                        idx = line_num - 1 if line_num > 0 else 0

                        if command == 'd' and 0 <= idx < len(new_lines):
                            del new_lines[idx]
                            changes_made = True
                        # Note: 'c' and 'a' would need content from following lines
                        # This is a simplified implementation

                if changes_made:
                    new_text = "\n".join(new_lines)
                    if old_text.endswith("\n"):
                        new_text += "\n"

                    diff_lines = list(difflib.unified_diff(
                        old_lines,
                        new_text.splitlines(),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                        lineterm="",
                    ))

                    if diff_lines:
                        diff_text = "\n".join(diff_lines)
                        if not diff_text.startswith("diff --git "):
                            diff_text = f"diff --git a/{rel} b/{rel}\n" + diff_text
                        if not diff_text.endswith("\n"):
                            diff_text += "\n"
                        return diff_text

            return None
        except Exception:
            # Any unexpected error → salvage failed
            return None

    def convert_patch_to_edit_blocks(self, patch: str, target_file: Optional[str] = None) -> Optional[dict]:
        """
        Convert a unified diff patch to edit_blocks structure.
        Moved from agent_loop.py _convert_patch_to_edit_blocks.

        Returns a dictionary with 'file_path' and 'blocks' if successful,
        otherwise None.
        """

        if not patch or not patch.strip():
            return None

        # Extract target file from patch headers or target_file
        file_path = target_file
        if not file_path:
            for line in patch.splitlines():
                m = re.match(r'^\+\+\+ b/(.+)$', line)
                if m:
                    file_path = m.group(1).strip()
                    break
                m2 = re.match(r'^diff --git a/\S+ b/(.+)$', line)
                if m2:
                    file_path = m2.group(1).strip()
                    break
        if not file_path:
            return None

        # Helper function to extract before/after from hunk body
        def _hunk_to_before_after(hunk_lines: list) -> tuple:
            """Extract (before_text, after_text) from a hunk body (list of lines).

            Returns (None, None) if extraction fails.
            """
            before_lines = []
            after_lines = []
            for hl in hunk_lines:
                if not hl:
                    continue
                stripped = hl.rstrip("\n")
                if stripped.startswith(" "):
                    before_lines.append(stripped[1:])
                    after_lines.append(stripped[1:])
                elif stripped.startswith("-"):
                    before_lines.append(stripped[1:])
                elif stripped.startswith("+"):
                    after_lines.append(stripped[1:])
                # skip \\ No newline at end of file, etc.

            before = "\n".join(before_lines)
            after = "\n".join(after_lines)
            if not before.strip() and not after.strip():
                return None, None
            return before, after

        # Parse hunks → (before_lines, after_lines) pairs
        blocks = []
        hunk_body: list = []
        in_hunk = False

        for line in patch.splitlines(keepends=True):
            if line.startswith("@@"):
                if in_hunk and hunk_body:
                    b, a = _hunk_to_before_after(hunk_body)
                    if b is not None:
                        blocks.append({"before": b, "after": a})
                hunk_body = []
                in_hunk = True
                continue
            if in_hunk:
                hunk_body.append(line)

        # Last hunk
        if in_hunk and hunk_body:
            b, a = _hunk_to_before_after(hunk_body)
            if b is not None:
                blocks.append({"before": b, "after": a})

        if not blocks:
            return None

        return {"file_path": file_path, "blocks": blocks}

    def repair_patch(self, patch_text: str, target_file: str,
                                 failure_reason: str, llm_output: Optional[str] = None) -> PatchResult:
                    """
                    Attempt repair using fallback ladder.

                    Args:
                        patch_text: Original patch that failed
                        target_file: Target file path
                        failure_reason: Why the patch failed
                        llm_output: Optional original LLM output for context

                    Returns:
                        PatchResult with repair attempt outcome
                    """
                    metadata = {
                        "reason": "",
                        "mode": "",
                        "fallback_used": [],
                        "first_fail_reason": failure_reason,
                        "second_fail_reason": "",
                        "synth_reason": "",
                        "execution_steps": [],
                        "normalized_patch": patch_text,
                    }

                    self._add_step(metadata, "repair_start", f"Starting repair ladder for {target_file}")

                    # If no LLM output (agent path), extract new_code from the patch text
                    # and try _auto_repair_patch as fallback (P8 fix)
                    if not llm_output:
                        self._add_step(metadata, "no_llm_fallback",
                                       "No LLM output — trying auto-repair from patch text")
                        auto_fix = self._auto_repair_patch(patch_text, target_file)
                        if auto_fix:
                            metadata["reason"] = "repair_success:auto_repair"
                            metadata["mode"] = "auto_repair"
                            metadata["fallback_used"] = ["auto_repair"]
                            return PatchResult(
                                success=True,
                                patch_applied=auto_fix,
                                metadata=metadata
                            )
                        metadata["reason"] = "no_llm_fallback_all_failed"
                        return PatchResult(
                            success=False,
                            error="Cannot repair patch without original LLM output "
                                  "and auto-repair from patch text failed",
                            metadata=metadata
                        )

                    # Parse file blocks from LLM output (llm_output guaranteed non-None here)
                    parsed_blocks = []
                    if parse_file_blocks:
                        try:
                            parsed_blocks = parse_file_blocks(llm_output or "")
                        except Exception as e:
                            logger.debug("parse_file_blocks failed: %s", e)
                            parsed_blocks = []
                    else:
                        logger.warning("parse_file_blocks not available")

                    if not parsed_blocks:
                        metadata["reason"] = "no_parsed_blocks"
                        return PatchResult(
                            success=False,
                            error="No parseable file blocks found in LLM output",
                            metadata=metadata
                        )

                    # Prefer the block that matches target_file; fall back to first block
                    block = parsed_blocks[0]
                    try:
                        normalized_target = normalize_rel_path_fast(str(target_file))
                        for candidate in parsed_blocks:
                            candidate_path = normalize_rel_path_fast(str(
                                candidate.get("path")
                                or candidate.get("file")
                                or candidate.get("filename")
                                or ""
                            ))
                            if candidate_path and normalized_target and candidate_path == normalized_target:
                                block = candidate
                                break
                    except Exception:
                        block = parsed_blocks[0]

                    new_code = block.get("text") or block.get("content") or ""
                    if not new_code.strip():
                        metadata["reason"] = "empty_new_code"
                        return PatchResult(
                            success=False,
                            error="Empty code block in LLM output",
                            metadata=metadata
                        )

                    # Track which fallbacks we attempt
                    fallback_attempted = []
                    fallback_succeeded = False
                    result_patch = None
                    result_mode = None

                    # 1. AST rewrite fallback
                    if self.ast_rewriter:
                        self._add_step(metadata, "ast_rewrite", "Attempting AST rewrite")
                        fallback_attempted.append("ast_rewrite")
                        try:
                            llm_header = (llm_output or "").strip().splitlines()[0].strip()
                            if llm_header.startswith("FUNCTION:"):
                                func_name = llm_header.split("FUNCTION:")[1].strip()
                                result = self.ast_rewriter.replace_function(
                                    target_file,
                                    func_name,
                                    new_code
                                )
                                result_patch = self.ast_rewriter.generate_patch(target_file, result)
                                result_mode = "ast_function"
                                fallback_succeeded = True
                            elif llm_header.startswith("CLASS:"):
                                class_name = llm_header.split("CLASS:")[1].strip()
                                result = self.ast_rewriter.replace_class(
                                    target_file,
                                    class_name,
                                    new_code
                                )
                                result_patch = self.ast_rewriter.generate_patch(target_file, result)
                                result_mode = "ast_class"
                                fallback_succeeded = True
                            elif llm_header.startswith("METHOD:"):
                                path = llm_header.split("METHOD:")[1].strip()
                                class_name, method_name = path.split(".")
                                result = self.ast_rewriter.replace_method(
                                    target_file,
                                    class_name,
                                    method_name,
                                    new_code
                                )
                                result_patch = self.ast_rewriter.generate_patch(target_file, result)
                                result_mode = "ast_method"
                                fallback_succeeded = True
                            elif is_python_definition(new_code):
                                func_name, _ = extract_symbol_name(new_code)
                                result = self.ast_rewriter.replace_function(
                                    target_file,
                                    func_name,
                                    new_code
                                )
                                result_patch = self.ast_rewriter.generate_patch(target_file, result)
                                result_mode = "ast_autodetect"
                                fallback_succeeded = True
                        except Exception as e:
                            logger.debug("AST rewrite attempt failed: %s", e)
                            metadata["second_fail_reason"] = f"ast_rewrite_failed: {e}"

                    # 2. Symbol search fallback
                    if not fallback_succeeded and self.symbol_searcher and self.ast_rewriter:
                        self._add_step(metadata, "symbol_search", "Attempting symbol search fallback")
                        fallback_attempted.append("symbol_search")
                        try:
                            header = new_code.strip().splitlines()[0].strip()
                            symbol_name, symbol_kind = extract_symbol_name(header)

                            if symbol_name:
                                results = self.symbol_searcher.find_symbol(symbol_name, kind=symbol_kind if symbol_kind != "function" else "any")
                            else:
                                results = self.symbol_searcher.find_symbol(header)

                            if not results:
                                sym = self.symbol_searcher.fuzzy_find_symbol(symbol_name or header)
                                if sym:
                                    results = [sym]

                            if results:
                                sym = results[0]
                                if sym.kind in ("function", "async_function", "method"):
                                    result = self.ast_rewriter.replace_function(
                                        sym.file,  # was sym.file_path
                                        sym.name,
                                        new_code
                                    )
                                    result_patch = self.ast_rewriter.generate_patch(sym.file, result)
                                    result_mode = "ast_symbol_function"
                                    fallback_succeeded = True
                                elif sym.kind == "class":
                                    result = self.ast_rewriter.replace_class(
                                        sym.file,
                                        sym.name,
                                        new_code
                                    )
                                    result_patch = self.ast_rewriter.generate_patch(sym.file, result)
                                    result_mode = "ast_symbol_class"
                                    fallback_succeeded = True
                        except Exception as e:
                            logger.debug("Symbol search fallback failed: %s", e)
                            metadata["second_fail_reason"] = f"symbol_search_failed: {e}"

                    # 3. Semantic patch fallback
                    if not fallback_succeeded and self.semantic_patcher:
                        self._add_step(metadata, "semantic_patch", "Attempting semantic patch fallback")
                        fallback_attempted.append("semantic_patch")
                        try:
                            sem_result = self.semantic_patcher.apply_semantic_patch(
                                file_path=target_file,
                                new_code=new_code,
                            )
                            if sem_result:
                                result_patch = self.semantic_patcher.generate_patch(target_file, sem_result)
                                if sem_result.kind == "class":
                                    result_mode = "semantic_class"
                                else:
                                    result_mode = "semantic_function"
                                fallback_succeeded = True
                        except Exception as e:
                            logger.debug("Semantic patch fallback failed: %s", e)
                            metadata["second_fail_reason"] = f"semantic_patch_failed: {e}"

                    # 4. File-block diff synthesis (using patch_synthesizer for FULL_FILE mode)
                    if not fallback_succeeded and self.patch_synthesizer and self.hybrid_parser and OutputMode is not None:
                        self._add_step(metadata, "file_block_synth", "Attempting file-block synthesis")
                        fallback_attempted.append("file_block_synth")
                        try:
                            # Try to parse as FULL_FILE mode
                            parse_result = self.hybrid_parser.parse(llm_output, OutputMode.FULL_FILE)
                            if parse_result.success and parse_result.mode == OutputMode.FULL_FILE:
                                synthesized = self.patch_synthesizer.synthesize(parse_result, target_file)
                                if synthesized.strip():
                                    result_patch = synthesized
                                    result_mode = "file_block_synth"
                                    fallback_succeeded = True
                        except Exception as e:
                            logger.debug("File-block synthesis failed: %s", e)
                            metadata["second_fail_reason"] = f"file_block_synth_failed: {e}"

                    # Return result
                    if fallback_succeeded:
                        metadata["reason"] = f"repair_success_{result_mode}"
                        metadata["mode"] = result_mode
                        metadata["fallback_used"] = fallback_attempted
                        metadata["synth_reason"] = f"repaired via {result_mode}"
                        return PatchResult(
                            success=True,
                            patch_applied=result_patch,
                            metadata=metadata
                        )
                    else:
                        metadata["reason"] = "all_repair_failed"
                        metadata["fallback_used"] = fallback_attempted
                        metadata["second_fail_reason"] = metadata.get("second_fail_reason", "all fallbacks failed")
                        return PatchResult(
                            success=False,
                            error=f"All repair attempts failed: {failure_reason}",
                            metadata=metadata
                        )

    @staticmethod
    def _trim_patch_to_first_header(patch: str) -> str:
        """
        Some LLMs prepend junk (or emit a hunk header before file headers), which makes
        `git apply` fail with errors like: "patch fragment without header".

        Keep only from the first real diff header line.
        Acceptable starts:
          - "diff --git ..."
          - "--- ..."
          - "+++ ..."
        """
        if not patch:
            return ""

        lines = str(patch).replace("\r\n", "\n").split("\n")
        start_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if line.startswith("diff --git ") or line.startswith("--- ") or line.startswith("+++ "):
                start_idx = i
                break

        if start_idx is None:
            return str(patch).strip()

        trimmed = "\n".join(lines[start_idx:]).strip()
        if trimmed and not trimmed.endswith("\n"):
            trimmed += "\n"
        return trimmed

    @staticmethod
    def _sanitize_patch_lines(patch: str) -> str:
        """
        External LLMs sometimes:
          - prepend BOM (\\ufeff)
          - indent diff markers with spaces/tabs
          - wrap diffs inside markdown fences (```diff ... ```)
        Any of these can make `git apply` fail to recognize headers/hunks properly.

        We normalize by:
          - removing BOM at line starts
          - dropping markdown fence lines
          - left-stripping lines that *look like* diff markers
        """
        if not patch:
            return ""

        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")
        out: list[str] = []

        for line in lines:
            if not line:
                out.append(line)
                continue

            # Remove BOM at the start of a line
            if line and line[0] == "\ufeff":
                line = line.lstrip("\ufeff")

            stripped = line.lstrip()

            # Drop markdown fences that often leak into patch output
            if stripped.startswith("```"):
                continue

            if (
                stripped.startswith("diff --git ")
                or stripped.startswith("--- ")
                or stripped.startswith("+++ ")
                or stripped.startswith("@@ ")
                or stripped.startswith("index ")
                or stripped.startswith("new file mode ")
                or stripped.startswith("deleted file mode ")
                or stripped.startswith("similarity index ")
                or stripped.startswith("rename from ")
                or stripped.startswith("rename to ")
            ):
                out.append(stripped)
            else:
                out.append(line)

        normalized = "\n".join(out).strip()
        if normalized and not normalized.endswith("\n"):
            normalized += "\n"
        return normalized

    @staticmethod
    def _keep_only_target_file_section(patch: str, target_file: Optional[str]) -> str:
        """
        Best-effort: keep ONLY the diff section for the target file.

        Why:
          Some models append extra junk after a valid diff, including orphan hunks
          or a second partial diff. Even if the first part is valid, the tail breaks
          `git apply` with "patch fragment without header".

        Behavior:
          - If `diff --git` sections exist: keep the first section whose a/ or b/ path matches target_file.
            If no match found, keep the first section only.
          - If no `diff --git` lines exist: keep from the first '---'/'+++' header up to end,
            but stop if we detect a second file header for a different file.
        """
        if not patch:
            return ""

        # This function assumes input has already been sanitized (no BOM/indent on markers).
        # Call _sanitize_patch_lines() first.
        tf = normalize_rel_path_fast(target_file or "")
        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")

        # Locate diff --git boundaries
        diff_idxs = [i for i, _item_ in enumerate(lines) if _item_.startswith("diff --git ")]
        if diff_idxs:
            diff_git_re = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")
            sections: list[tuple[int, int, str, str]] = []
            for s_i, start in enumerate(diff_idxs):
                end = diff_idxs[s_i + 1] if (s_i + 1) < len(diff_idxs) else len(lines)
                m = diff_git_re.match(lines[start] or "")
                a_path = m.group(1) if m else ""
                b_path = m.group(2) if m else ""
                sections.append((start, end, a_path, b_path))

            chosen = sections[0]
            if tf:
                for sec in sections:
                    _s, _e, a_path, b_path = sec
                    if a_path == tf or b_path == tf:
                        chosen = sec
                        break
                    if Path(a_path).name == Path(tf).name or Path(b_path).name == Path(tf).name:
                        chosen = sec
                        break

            s, e, _a, _b = chosen
            kept = "\n".join(lines[s:e]).strip()
            if kept and not kept.endswith("\n"):
                kept += "\n"
            return kept

        # No diff --git sections. Keep a single ---/+++ file section.
        start_idx: Optional[int] = None
        for i, _item_ in enumerate(lines):
            if _item_.startswith("--- "):
                start_idx = i
                break
        if start_idx is None:
            return txt.strip() + ("\n" if txt.strip() else "")

        # determine the file path from the first header, if possible
        first_file: Optional[str] = None
        m0 = re.match(r"^---\s+a/(.+?)\s*$", lines[start_idx] or "")
        if m0:
            first_file = m0.group(1)

        end_idx = len(lines)
        # stop if we see another file header for a different file (second '--- a/...' later)
        for j in range(start_idx + 1, len(lines)):
            if lines[j].startswith("--- "):
                m1 = re.match(r"^---\s+a/(.+?)\s*$", lines[j] or "")
                f1 = m1.group(1) if m1 else None
                if first_file is None:
                    end_idx = j
                    break
                if f1 and f1 != first_file:
                    end_idx = j
                    break

        kept = "\n".join(lines[start_idx:end_idx]).strip()
        if kept and not kept.endswith("\n"):
            kept += "\n"
        return kept

    @staticmethod
    def _force_target_file_paths(patch: str, target_file: Optional[str]) -> str:
        """
        If the model emits headers for a basename (e.g., service.py) instead of the full rel path
        (external_llm/service.py), `git apply` fails with "No such file or directory".

        If basename matches target_file basename, rewrite:
          - diff --git a/<base> b/<base>
          - --- a/<base>
          - +++ b/<base>
        into the full target rel path.
        """
        if not patch:
            return ""
        tf = normalize_rel_path_fast(target_file or "")
        if not tf:
            return patch

        base = Path(tf).name
        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")
        out: list[str] = []

        for line in lines:
            s = line or ""

            # diff --git a/x b/x  OR  diff --git x y
            if s.startswith("diff --git "):
                parts = s.split()
                if len(parts) >= 4:
                    a_raw = (parts[2] or "").strip()
                    b_raw = (parts[3] or "").strip()
                    a_path = a_raw[2:] if a_raw.startswith("a/") else a_raw
                    b_path = b_raw[2:] if b_raw.startswith("b/") else b_raw
                    if (
                        (a_path == base and b_path == base)
                        or (Path(a_path).name == base and Path(b_path).name == base)
                    ):
                        out.append(f"diff --git a/{tf} b/{tf}")
                        continue

            # --- a/x  OR  --- x   (but never rewrite /dev/null)
            if s.startswith("--- "):
                p = s[4:].strip()
                if p != "/dev/null":
                    a_path = p[2:] if p.startswith("a/") else p
                    if a_path == base or Path(a_path).name == base:
                        out.append(f"--- a/{tf}")
                        continue

            # +++ b/x  OR  +++ x   (but never rewrite /dev/null)
            if s.startswith("+++ "):
                p = s[4:].strip()
                if p != "/dev/null":
                    b_path = p[2:] if p.startswith("b/") else p
                    if b_path == base or Path(b_path).name == base:
                        out.append(f"+++ b/{tf}")
                        continue

            out.append(line)

        fixed = "\n".join(out).strip()
        if fixed and not fixed.endswith("\n"):
            fixed += "\n"
        return fixed

    @staticmethod
    def _ensure_headers_before_any_hunk(patch: str, target_file: Optional[str]) -> str:
        """
        Strong best-effort guardrail for the most common git-apply failure:
          "patch fragment without header at line N: @@ ..."

        If the patch contains a hunk (@@ ...) but there is NO '---'/'+++' header
        anywhere before the FIRST hunk, we inject:
          --- a/<target>
          +++ b/<target>

        This is intentionally simple and global (whole-patch) to catch cases where
        the model outputs an orphan hunk fragment or omits headers entirely.
       """
        if not patch:
            return ""

        tf = normalize_rel_path_fast(target_file or "")
        if not tf:
            return patch

        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")

        first_hunk_idx: Optional[int] = None
        for i, _item_ in enumerate(lines):
            if _item_.startswith("@@ "):
                first_hunk_idx = i
                break
        if first_hunk_idx is None:
            return patch

        has_minus = any(_item_.startswith("--- ") for _item_ in lines[:first_hunk_idx])
        has_plus = any(_item_.startswith("+++ ") for _item_ in lines[:first_hunk_idx])
        if has_minus and has_plus:
            return patch

        # Try to place headers right before the first hunk.
        injected: list[str] = []
        injected.extend(lines[:first_hunk_idx])
        injected.append(f"diff --git a/{tf} b/{tf}")
        injected.append(f"--- a/{tf}")
        injected.append(f"+++ b/{tf}")
        injected.extend(lines[first_hunk_idx:])

        out = "\n".join(injected).strip()
        if out and not out.endswith("\n"):
            out += "\n"
        return out

    @staticmethod
    def _normalize_patch_headers(patch: str, target_file: Optional[str]) -> str:
        """
        Best-effort repair for external-LLM patch corruption that causes:
          "patch fragment without header at line N: @@ ..."

        Common failure patterns:
          - hunks (@@ ...) appear before any file headers (---/+++)
          - patch contains `diff --git a/X b/X` but is missing `---/+++` lines before the first hunk
          - later in the patch, a new orphan hunk appears without headers (often due to truncation)

        Strategy:
          - Walk line-by-line
          - Track current file paths from `diff --git a/... b/...`
          - Track whether current section has seen `---` and `+++`
          - If we hit a hunk without both headers, inject headers inferred from:
              1) the most recent diff --git line, else
              2) target_file (if provided), else
              3) leave as-is (git apply will reject; validation is the guardrail)
        """
        if not patch:
            return ""

        tf = normalize_rel_path_fast(target_file or "")
        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")

        out: list[str] = []
        have_minus = False
        have_plus = False
        cur_a: Optional[str] = None
        cur_b: Optional[str] = None

        diff_git_re = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")

        for line in lines:
            m = diff_git_re.match(line or "")
            if m:
                # New section
                cur_a = f"--- a/{m.group(1)}"
                cur_b = f"+++ b/{m.group(2)}"
                have_minus = False
                have_plus = False
                out.append(line)
                continue

            if line.startswith("--- "):
                have_minus = True
                out.append(line)
                continue

            if line.startswith("+++ "):
                have_plus = True
                out.append(line)
                continue

            if line.startswith("@@ "):
                if not (have_minus and have_plus):
                    # Inject headers before the hunk.
                    if cur_a and cur_b:
                        out.append(cur_a)
                        out.append(cur_b)
                        have_minus = True
                        have_plus = True
                    elif tf:
                        out.append(f"--- a/{tf}")
                        out.append(f"+++ b/{tf}")
                        have_minus = True
                        have_plus = True
                out.append(line)
                continue

            out.append(line)

        normalized = "\n".join(out).strip()
        if normalized and not normalized.endswith("\n"):
            normalized += "\n"
        return normalized

    # ---------------------------------------------------------------------
    # Auto-mode: file block parsing + diff synthesis
    # ---------------------------------------------------------------------

    # Legacy regex fallback (kept for compatibility / debugging)
    _RE_FILE_BLOCK = re.compile(
        r"(?ims)"
        r"(?:^|\n)\s*(?:FILE|Path|Target file)\s*:\s*(?P<path>[^\n\r]+?)\s*\r?\n"
        r"(?:```[^\n\r]*\r?\n(?P<code1>[\s\S]*?)\r?\n```|"
        r"(?P<code2>(?:(?!^\s*(?:FILE|Path|Target file)\s*:).*\r?\n)+))"
    )

    def normalize_and_validate(self, patch_text: str, target_file: Optional[str]) -> tuple[str, Optional[str]]:
        """
        Normalize patch candidate and validate with git apply --check.
        Applies the same sanitation/repair steps across diff/auto/fast paths.
        """
        if not patch_text:
            return "", "Empty patch text"
        p = self._trim_patch_to_first_header(str(patch_text))
        p = self._sanitize_patch_lines(p)
        p = self._keep_only_target_file_section(p, target_file)
        p = self._force_target_file_paths(p, target_file)
        p = self._ensure_headers_before_any_hunk(p, target_file)
        p = self._normalize_patch_headers(p, target_file)

        # Add trailing newline if missing
        if p and not p.endswith("\n"):
            p += "\n"

        # Simple validation: check if it looks like a unified diff
        if self._looks_like_unified_diff(p):
            # Run git apply --check preflight (non-fatal: tolerant flags may succeed)
            ok, err = self._git_apply_check_best_effort(p)
            if not ok:
                logger.debug(
                    "normalize_and_validate: git apply precheck failed "
                    "(non-fatal, tolerant path may still succeed): %s", err,
                )
            return p, None
        else:
            return p, "Patch does not look like a unified diff"


    def _looks_like_unified_diff(self, text: str) -> bool:
        """Heuristic check for unified diff format."""
        t = str(text or "")
        if not t.strip():
            return False
        # Heuristic: any real diff header + at least one hunk marker
        has_header = any(s in t for s in ("diff --git ", "--- a/", "+++ b/")) or t.lstrip().startswith("--- ")
        has_hunk = ("@@ " in t)
        # Allow hunk-only patches (starting with @@, no header) — git apply handles them
        return bool(has_hunk and (has_header or t.lstrip().startswith("@@")))

    def _git_apply_check_best_effort(self, patch_text: str) -> tuple[bool, Optional[str]]:
        """
        Run git apply --check --recount --whitespace=nowarn - in repo_root.
        Returns (success, error_message).
        """
        try:
            import subprocess
            result = subprocess.run(
                ["git", "apply", "--check", "--recount", "--whitespace=nowarn", "-"],
                cwd=self.repo_root,
                input=patch_text.encode("utf-8"),
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True, None
            else:
                error = result.stderr.decode("utf-8", errors="ignore").strip()
                if not error:
                    error = result.stdout.decode("utf-8", errors="ignore").strip()
                if not error:
                    error = f"git apply --check failed with exit code {result.returncode}"
                return False, error
        except Exception as e:
            logger.debug("git apply --check failed with exception: %s", e)
            return False, f"git apply --check exception: {e}"

    def _classify_target_git_state(self, target_file: Optional[str]) -> str:
        """Classify the git tracking state of a target file (pre-apply gate).

        Returns one of:
          - "tracked":        file is committed in git (3-way merge pre-image blob available)
          - "freshly_edited": file exists but working tree differs from index
                              (pre-image blob likely stale → 3-way will mismatch)
          - "untracked":      file exists but is not tracked by git (no pre-image blob at all)
          - "gitignored":     file is excluded by .gitignore (no pre-image blob at all)
          - "unknown":        classification failed (e.g. not a git repo, or file missing)

        Why this matters: `git apply --3way` requires the patch's pre-image blob in the
        git object store. For untracked/gitignored files that blob never exists, and for
        freshly-edited files it is stale — so 3-way fails with
        "repository lacks the necessary blob to perform a 3-way merge". Plain `git apply`
        (non-3way) still works for these files, so the caller should pass skip_3way=True
        to avoid the wasted, guaranteed-to-fail 3-way subprocess.
        """
        if not target_file:
            return "unknown"
        tf = Path(self.repo_root) / target_file
        if not tf.exists():
            return "unknown"  # file_not_found is handled by the earlier early-exit gate
        try:
            # 1) gitignored check (cheapest, unambiguous)
            chk = subprocess.run(
                ["git", "check-ignore", "-q", target_file],
                cwd=self.repo_root,
                capture_output=True,
                timeout=5,
            )
            if chk.returncode == 0:
                return "gitignored"

            # 2) git status --porcelain for this path: tracked? modified?
            #    XY format: X=index status, Y=worktree status. We only care about a few.
            #    "?? file"  → untracked
            #    " M file"   → worktree modified (freshly-edited, blob is HEAD's = may mismatch)
            #    "A  file"/"AM" → staged-but-uncommitted (intent-to-add / staged new file)
            #    (absent)   → fully tracked & clean
            st = subprocess.run(
                ["git", "status", "--porcelain", "--ignore-submodules", "-z", "--", target_file],
                cwd=self.repo_root,
                capture_output=True,
                timeout=5,
            )
            if st.returncode != 0:
                return "unknown"
            out = (st.stdout or b"").decode("utf-8", errors="ignore")
            if not out.strip():
                return "tracked"  # tracked AND clean
            # Parse first record's XY
            first = out.split("\0", 1)[0]
            xy = first[:2] if len(first) >= 2 else "  "
            x, y = xy[0], xy[1]
            if x == "?" and y == "?":
                return "untracked"
            if x in (" ", "A", "M", "D", "R", "C") and y == "M":
                # staged-or-tracked but worktree-modified → pre-image blob (HEAD or index)
                # may not match the patch's pre-image
                return "freshly_edited"
            return "tracked"
        except Exception as e:
            logger.debug("_classify_target_git_state failed: %s", e)
            return "unknown"

    def _patch_index_shas_are_fake(self, patch_text: str) -> bool:
        """Detect fabricated `index <sha>..<sha>` lines (Mode B).

        Scope: this is a *minor performance/noise optimization*, NOT a
        correctness fix. When the patch context matches the working tree,
        `git apply --check` passes and the 3-way branch is never reached —
        so `skip_3way` is never even consulted. The detector only matters in
        the *drift* case: there `--check` fails (CONFLICT), the patch's
        fabricated old-SHA (the model cannot run `git hash-object`) is absent
        from the object store, and `git apply --3way` is guaranteed to die
        with "repository lacks the necessary blob to perform 3-way merge".
        Skipping 3-way there avoids one wasted subprocess and keeps the
        repair ladder (reanchor / AST / symbol) the actual recovery path.

        Returns True  — an `index` line names a SHA absent from the store.
        Returns False — no index line, all SHAs resolve, or the probe itself
                        failed (conservative: keep prior 3-way behavior).
        """
        for m in re.finditer(
            r'^index ([0-9a-f]{7,40})\.\.([0-9a-f]{7,40})',
            patch_text, re.MULTILINE,
        ):
            for sha in (m.group(1), m.group(2)):
                # All-zero SHA = legitimate placeholder for file creation
                # (old side) or deletion (new side); skip it.
                if sha.strip("0") == "":
                    continue
                try:
                    chk = subprocess.run(
                        ["git", "cat-file", "-e", sha],
                        cwd=self.repo_root, capture_output=True, timeout=5,
                    )
                    if chk.returncode != 0:
                        return True
                except Exception as e:
                    logger.debug("cat-file probe failed for %s: %s", sha, e)
                    return False
        return False

    def _apply_diff_once(self, patch_text: str, target_file: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Apply a unified diff using the underlying diff_apply module.
        Returns (success, error_message).

        Also accepts fragment-only unified diffs that contain a hunk header
        (`@@ ... @@`) but omit the required file headers (`---` / `+++`).
        When possible, synthesize minimal headers from the patch body before
        handing off to diff_apply.
        """
        if not self._diff_apply:
            return False, "diff_apply module not available"

        normalized = patch_text or ""
        stripped = normalized.lstrip()

        has_hunk = "@@ " in normalized
        has_old_header = ("\n--- " in ("\n" + normalized)) or stripped.startswith("--- ")
        has_new_header = ("\n+++ " in ("\n" + normalized)) or stripped.startswith("+++ ")

        if has_hunk and not (has_old_header and has_new_header):
            inferred_path = None

            for line in normalized.splitlines():
                s = line.strip()

                if s.startswith("+++ b/"):
                    inferred_path = s[len("+++ b/"):].strip()
                    break
                if s.startswith("--- a/"):
                    inferred_path = s[len("--- a/"):].strip()
                    break
                if s.startswith("+++ "):
                    candidate = s[len("+++ "):].strip()
                    if candidate and candidate != "/dev/null":
                        inferred_path = candidate.removeprefix("b/")
                        break
                if s.startswith("--- "):
                    candidate = s[len("--- "):].strip()
                    if candidate and candidate != "/dev/null":
                        inferred_path = candidate.removeprefix("a/")
                        break
                if s.startswith("diff --git "):
                    parts = s.split()
                    if len(parts) >= 4 and parts[3].startswith("b/"):
                        inferred_path = parts[3][2:]
                        break

            if inferred_path:
                normalized = (
                    f"--- a/{inferred_path}\n"
                    f"+++ b/{inferred_path}\n"
                    f"{stripped}"
                )

        try:
            _ado_ok, _ado_msg, _ado_reason, _ado_details = self._diff_apply(
                self.repo_root, normalized,
                file_path_hint=target_file,
            )
            if _ado_ok:
                return True, None
            else:
                return False, _ado_msg or _ado_reason or "git apply failed"
        except Exception as e:
            return False, f"diff_apply exception: {e}"

    # ── Hunk header count fix ─────────────────────────────────────────────────

    def _fix_hunk_counts(self, patch_text: str) -> str:
        """Recompute @@ hunk header line counts from actual hunk body.

        Small models often produce wrong counts in @@ -a,b +c,d @@ lines
        (e.g., @@ -6,3 +6,3 @@ when the body only has 2 lines each side).
        Git treats this as a "corrupt patch" and refuses to apply even with --recount.

        This method counts the actual context/remove/add lines and rewrites the header.
        """
        hunk_header_re = re.compile(
            r'^(@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@)(.*)',
            re.DOTALL,
        )
        lines = patch_text.splitlines(keepends=True)
        output = []
        i = 0
        while i < len(lines):
            line = lines[i]
            m = hunk_header_re.match(line)
            if not m:
                output.append(line)
                i += 1
                continue
            old_start = m.group(2)
            new_start = m.group(3)
            suffix = m.group(4)

            # Collect hunk body (until next @@ or end of patch header block)
            i += 1
            hunk_body = []
            while i < len(lines) and not lines[i].startswith("@@"):
                hunk_body.append(lines[i])
                i += 1

            # Count lines
            old_count = 0
            new_count = 0
            for hl in hunk_body:
                s = hl.rstrip("\n")
                if s.startswith(" "):
                    old_count += 1
                    new_count += 1
                elif s.startswith("-"):
                    old_count += 1
                elif s.startswith("+"):
                    new_count += 1
                # lines starting with \ (no newline at end) are skipped

            # Rebuild header with correct counts
            new_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}"
            output.append(new_header)
            output.extend(hunk_body)

        return "".join(output)

    def _add_diff_headers(self, patch_text: str, target_file: Optional[str]) -> str:
        """Add missing diff headers (diff --git, ---, +++) to a patch.

        Handles:
        - hunk-only patches (starting with @@)
        - patches with --- a/ but missing diff --git line
        """
        text = patch_text.strip()
        if not target_file:
            return patch_text

        # Hunk-only: starts with @@ but no headers at all
        if text.lstrip().startswith("@@") and "--- " not in text and "+++ " not in text:
            fp = target_file.lstrip("/")
            header = (
                f"diff --git a/{fp} b/{fp}\n"
                f"--- a/{fp}\n"
                f"+++ b/{fp}\n"
            )
            return header + text + "\n"

        # Has --- a/ but missing diff --git
        if "--- a/" in text and "diff --git" not in text:
            fp = target_file.lstrip("/")
            return f"diff --git a/{fp} b/{fp}\n" + text + "\n"

        return patch_text

    # ── Tolerant apply: try multiple git apply flag combinations ─────────────

    def _tolerant_git_apply(self, patch_text: str, target_file: Optional[str] = None,
                            allow_3way: bool = True) -> tuple[bool, Optional[str], str]:
        """Try multiple git apply flag combinations for tolerant (small model) mode.

        Returns (success, error_message, mode_used).
        Pipeline:
          0. Fix hunk counts + add missing headers (preprocessing)
          1. Preprocessed patch + --ignore-whitespace
          2. Preprocessed patch + --ignore-space-change
          3. Preprocessed patch (no extra flags, just correct counts)
          4. --3way (creates merge markers if needed — last resort)

        Args:
            allow_3way: When False, drop the ``--3way`` variant. The caller sets this for
                targets known to lack a pre-image blob (untracked / gitignored /
                freshly-edited files), where ``--3way`` is a guaranteed failure
                ("repository lacks the necessary blob"). The non-3way variants above
                remain and still patch such files correctly.
        """
        # Step 0: Preprocess — fix hunk counts and add missing headers
        fixed = self._fix_hunk_counts(patch_text)
        if target_file:
            fixed = self._add_diff_headers(fixed, target_file)
        # Strip CRLF
        if "\r\n" in fixed:
            fixed = fixed.replace("\r\n", "\n")

        # Try the preprocessed patch variants
        patches_to_try = [
            (fixed, ["--ignore-whitespace"], "fixed_ignore_ws"),
            (fixed, ["-C0", "--ignore-whitespace"], "fixed_C0_ignore_ws"),  # C0 = no context required
            (fixed, ["-C0"], "fixed_C0"),                                    # pure line-number matching
            (fixed, ["--ignore-space-change"], "fixed_ignore_sc"),
            (fixed, [], "fixed_plain"),
        ]
        if allow_3way:
            patches_to_try.append(
                (patch_text, ["--3way"], "3way_merge"),  # fallback to raw + 3way
            )
        for try_patch, flags, mode_name in patches_to_try:
            encoded = try_patch.encode("utf-8")
            try:
                # --check first
                check = subprocess.run(
                    ["git", "apply", "--check", *flags, "-"],
                    cwd=self.repo_root,
                    input=encoded,
                    capture_output=True,
                    timeout=10,
                )
                if check.returncode == 0:
                    # Actually apply
                    apply_r = subprocess.run(
                        ["git", "apply", *flags, "-"],
                        cwd=self.repo_root,
                        input=encoded,
                        capture_output=True,
                        timeout=30,
                    )
                    if apply_r.returncode == 0:
                        logger.info("tolerant_git_apply succeeded mode=%s flags=%s", mode_name, flags)
                        return True, None, mode_name
                    else:
                        err = apply_r.stderr.decode("utf-8", errors="ignore").strip()
                        logger.warning(
                            "tolerant_git_apply: --check passed but apply failed "
                            "mode=%s flags=%s error=%s",
                            mode_name, flags, err,
                        )
                        return False, f"check OK but apply failed ({mode_name}): {err}", mode_name
            except Exception as exc:
                logger.debug("tolerant apply(%s) exception: %s", mode_name, exc)

        # NOTE: no index-level fallback for "untracked/gitignored" files here.
        # Plain `git apply` (the non-3way variants above) already patches AND
        # creates working-tree files regardless of git-tracking or .gitignore
        # status, and tolerates moderate line-number drift — verified directly.
        # A previous `git add -N` (intent-to-add) retry was removed because it
        # could never succeed for this case:
        #   • `git add -N` rejects gitignored paths without -f, so the retry was
        #     skipped for exactly the files it claimed to handle;
        #   • even forced, `git apply --3way` needs the patch's pre-image blob,
        #     which an untracked file (or a model-generated diff) never has
        #     → "does not exist in index" / "does not match index".
        # If every variant above fails, the patch itself is malformed — that is
        # the repair ladder's job (AST / symbol / semantic / file-block), not the
        # git index's. Mutating the index here only risked leaving the file staged.
        return False, "All tolerant git apply variants failed", "none"

    # ── Fuzzy context re-anchoring: fix wrong @@ line numbers ────────────────

    def _exact_reanchor_patch(self, patch_text: str, target_file: Optional[str]) -> Optional[str]:
        """Re-anchor a unified diff by finding exact removed-line content in the file.

        Faster and more reliable than SequenceMatcher for small line offsets.
        Searches for the first `-` line's exact content in the actual file,
        then rewrites @@ headers if the offset is within ±50 lines.
        """
        if not target_file:
            return None

        file_path = os.path.join(self.repo_root, target_file) if self.repo_root else target_file
        try:
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                file_lines = [_item_.rstrip("\n\r") for _item_ in fh.readlines()]
        except OSError:
            return None
        if not file_lines:
            return None

        lines = patch_text.splitlines(keepends=True)
        output = []
        i = 0
        changed = False

        hunk_header_re = re.compile(
            r'^(@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@)(.*)', re.DOTALL
        )

        # Copy header lines
        while i < len(lines) and not lines[i].startswith("@@"):
            output.append(lines[i])
            i += 1

        while i < len(lines):
            line = lines[i]
            m = hunk_header_re.match(line)
            if not m:
                output.append(line)
                i += 1
                continue

            old_start = int(m.group(2))
            old_count = int(m.group(3)) if m.group(3) is not None else 1
            new_count = int(m.group(5)) if m.group(5) is not None else 1
            suffix = m.group(6)

            # Collect hunk body
            i += 1
            hunk_body = []
            while i < len(lines) and not lines[i].startswith("@@"):
                hunk_body.append(lines[i])
                i += 1

            # Extract removed lines (stripped of diff prefix)
            removed_lines = []
            for hl in hunk_body:
                if hl.startswith("-"):
                    removed_lines.append(hl[1:].rstrip("\n\r"))

            if not removed_lines:
                output.append(line)
                output.extend(hunk_body)
                continue

            # Search for the first removed line in the actual file
            search_text = removed_lines[0].strip()
            if not search_text or len(search_text) < 5:
                output.append(line)
                output.extend(hunk_body)
                continue

            # Find all matching positions
            matches = []
            for idx, fl in enumerate(file_lines):
                if fl.strip() == search_text:
                    matches.append(idx)

            if not matches:
                output.append(line)
                output.extend(hunk_body)
                continue

            # Pick the match closest to the original position (within ±50 lines)
            original_pos = old_start - 1  # 0-indexed
            best_match = min(matches, key=lambda x: abs(x - original_pos))
            offset_diff = abs(best_match - original_pos)

            if offset_diff == 0 or offset_diff > 50:
                output.append(line)
                output.extend(hunk_body)
                continue

            # Verify: check if all removed lines match at this position
            _all_match = True
            for j, rl in enumerate(removed_lines):
                _file_idx = best_match + j
                if _file_idx >= len(file_lines) or file_lines[_file_idx].strip() != rl.strip():
                    _all_match = False
                    break
            if not _all_match:
                output.append(line)
                output.extend(hunk_body)
                continue

            # Calculate the line offset for the hunk header
            # The first context line should start at (best_match - context_before_count + 1)
            context_before_count = 0
            for hl in hunk_body:
                if hl.startswith(" "):
                    context_before_count += 1
                elif hl.startswith("-") or hl.startswith("+"):
                    break

            new_start = best_match - context_before_count + 1  # 1-indexed
            delta = new_start - old_start
            new_new_start = int(m.group(4)) + delta

            new_header = f"@@ -{new_start},{old_count} +{new_new_start},{new_count} @@{suffix}"
            logger.info(
                "exact_reanchor: hunk @@ -%d → -%d (offset=%+d, match='%s') file=%s",
                old_start, new_start, delta, search_text[:40], target_file,
            )
            output.append(new_header)
            output.extend(hunk_body)
            changed = True

        if not changed:
            return None
        return "".join(output)

    def _reanchor_patch(self, patch_text: str, target_file: Optional[str]) -> Optional[str]:
        """Re-anchor a unified diff patch to the correct line numbers.

        Small models often generate patches with wrong @@ line numbers.
        This method:
        1. Parses each hunk's context + removed lines
        2. Searches the actual file for the best SequenceMatcher match
        3. Rewrites @@ headers with the correct line numbers
        4. Returns the repaired patch text, or None if re-anchoring fails.
        """
        if not target_file:
            return None
        import difflib
        import re

        # Resolve file path
        file_path = os.path.join(self.repo_root, target_file) if self.repo_root else target_file
        try:
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                file_lines = fh.readlines()
        except OSError:
            return None

        if not file_lines:
            return None

        lines = patch_text.splitlines(keepends=True)
        output = []
        i = 0
        changed = False

        hunk_header_re = re.compile(
            r'^(@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@)(.*)', re.DOTALL
        )

        # Collect header lines (diff --git, ---, +++) verbatim
        while i < len(lines) and not lines[i].startswith("@@"):
            output.append(lines[i])
            i += 1

        while i < len(lines):
            line = lines[i]
            m = hunk_header_re.match(line)
            if not m:
                output.append(line)
                i += 1
                continue

            old_start = int(m.group(2))
            old_count = int(m.group(3)) if m.group(3) is not None else 1
            new_count = int(m.group(5)) if m.group(5) is not None else 1
            suffix = m.group(6)

            # Collect hunk body
            i += 1
            hunk_body = []
            while i < len(lines) and not lines[i].startswith("@@"):
                hunk_body.append(lines[i])
                i += 1

            # Extract context (unchanged) + removed lines to use as search key
            search_lines = []
            for hl in hunk_body:
                if hl.startswith(" ") or hl.startswith("-"):
                    search_lines.append(hl[1:])  # strip prefix

            if not search_lines:
                # Cannot anchor empty search; keep as-is
                output.append(line)
                output.extend(hunk_body)
                continue

            # Build a normalized search string
            search_str = "".join(search_lines).strip()
            if not search_str:
                output.append(line)
                output.extend(hunk_body)
                continue

            # Try exact line match first (fast path)
            best_score = 0.0
            best_offset = old_start - 1  # default: use original

            # Sliding window search with SequenceMatcher
            # Optimization (P5): reuse SequenceMatcher with set_seqs(),
            # limit search window to ±200 lines, skip files > 2000 lines
            window = len(search_lines)
            if window <= 0:
                output.append(line)
                output.extend(hunk_body)
                continue

            # Skip fuzzy matching for very large files — too expensive
            file_len = len(file_lines)
            if file_len > 2000:
                logger.debug(
                    "reanchor_patch: skipping fuzzy match for large file "
                    "(%d lines, target=%s)", file_len, target_file,
                )
                output.append(line)
                output.extend(hunk_body)
                continue

            # Restrict search to ±200 lines around original position
            search_start = max(0, old_start - 1 - 200)
            search_end = min(file_len - window + 1, old_start - 1 + 200)
            if search_start >= search_end:
                search_start = 0
                search_end = file_len - window + 1

            matcher = difflib.SequenceMatcher(None)
            for start_idx in range(search_start, search_end):
                chunk = file_lines[start_idx:start_idx + window]
                chunk_str = "".join(chunk).strip()
                matcher.set_seqs(search_str, chunk_str)
                ratio = matcher.ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_offset = start_idx
                    if best_score > 0.95:
                        break  # near-perfect match — no need to scan further

            # Only re-anchor if we found a significantly better match than the original
            original_pos = old_start - 1  # 0-indexed
            if best_score >= 0.7 and best_offset != original_pos:
                new_start = best_offset + 1  # 1-indexed
                # Recalculate +N (new file start = old file start adjusted by delta so far)
                delta = new_start - old_start
                new_new_start = int(m.group(4)) + delta
                new_header = f"@@ -{new_start},{old_count} +{new_new_start},{new_count} @@{suffix}"
                logger.info(
                    "reanchor_patch: hunk @@ -%d → -%d (score=%.2f) file=%s",
                    old_start, new_start, best_score, target_file,
                )
                output.append(new_header)
                changed = True
            else:
                output.append(line)  # Keep original header

            output.extend(hunk_body)

        if not changed:
            return None  # Nothing was improved
        return "".join(output)

    def _add_step(self, metadata: dict[str, Any], step: str, description: str):
        """Add execution step to metadata."""
        metadata["execution_steps"].append({
            "step": step,
            "description": description,
            "timestamp": self._current_timestamp()
        })

    def _current_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().isoformat()
