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

import contextlib
import difflib
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from path_security import normalize_rel_path, resolve_inside_repo

from .code_structure_utils import extract_symbol_name, is_python_definition
from .common.atomic_io import atomic_write_text
from .output_modes import OutputMode
from .output_parser import _fence_tail_is_lang_tag, parse_file_blocks

logger = logging.getLogger(__name__)

# ── Precompiled patch-parse regexes (hot apply path) ─────────────────────────
# Hoisted module-level: these static patterns are compiled once instead of per
# patch parse/repair call (the re module's internal cache alone is not a
# guarantee — heavy dynamic-pattern load elsewhere can evict them).
_RE_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")
_RE_HUNK_HEADER_RECOUNT = re.compile(
    r"^(@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@)(.*)",
    re.DOTALL,
)
_RE_HUNK_HEADER_FIX = re.compile(r"^(@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@)(.*)", re.DOTALL)

# ─── Patch Context ────────────────────────────────────────────────────────────


@dataclass
class PatchContext:
    """Optional context for patch application."""

    original_request: str | None = None
    file_content: str | None = None
    llm_output: str | None = None
    output_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Patch Result ─────────────────────────────────────────────────────────────


@dataclass
class PatchResult:
    """Standardized result from patch application."""

    success: bool
    patch_applied: str | None = None  # final unified diff applied
    error: str | None = None
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
    # P22-4: bytes proxy for the >2000-line salvage skip — avoids loading a file
    # we are about to skip anyway (~2000 lines x ~128 bytes/line).
    _SALVAGE_SKIP_MAX_BYTES = 256 * 1024

    # Regex for parsing FILE blocks (legacy fallback)
    _RE_FILE_BLOCK = re.compile(
        r"(?ims)"
        r"(?:^|\n)\s*(?:FILE|Path|Target file)\s*:\s*(?P<path>[^\n\r]+?)\s*\r?\n"
        r"(?:```(?P<fencetail1>[^\n\r]+)\r?\n```|"
        r"```(?P<fencetail2>[^\n\r]*)\r?\n(?P<code1>[\s\S]*?)\r?\n```|"
        r"(?P<code2>(?:(?!^\s*(?:FILE|Path|Target file)\s*:).*\r?\n)+))"
    )

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self._setup_components()

    def _setup_components(self):
        """Import and set up component modules (all first-party — imports
        cannot fail, so the old try/except ImportError fallbacks were dead
        code; the ``is None`` defensive checks in callers are retained)."""
        # Diff applier (shared with agent path)
        from diff_apply import apply_patch as diff_apply_patch

        self._diff_apply = diff_apply_patch

        # AST rewriter
        from .ast_rewrite import ASTRewriter

        self.ast_rewriter = ASTRewriter(self.repo_root)

        # Semantic patcher
        from .semantic_patch import SemanticPatchEngine

        self.semantic_patcher = SemanticPatchEngine(self.repo_root)

        # Symbol searcher (already used by agent tools)
        from .agent.symbol_search import get_symbol_searcher

        self.symbol_searcher = get_symbol_searcher(self.repo_root)

        # Patch synthesizer
        from .patch_synthesizer import PatchSynthesizer

        self.patch_synthesizer = PatchSynthesizer(self.repo_root)

        # Hybrid parser
        from .hybrid_parser import HybridOutputParser

        self.hybrid_parser = HybridOutputParser()

    def _output_mode_to_enum(self, output_mode: str) -> OutputMode | None:
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

    def apply_patch(
        self, patch_text: str, target_file: str | None = None, context: PatchContext | None = None
    ) -> PatchResult:
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
        _patch_is_new_file = "--- /dev/null" in work_patch or "@@ -0,0 " in work_patch
        # Resolve check path: explicit target_file or parsed from patch header
        _check_target = target_file
        if not _check_target and not _patch_is_new_file:
            # split('\t') because `diff -u` and `git diff --no-prefix` put a
            # tab-separated timestamp after the path. Without it the target
            # resolved to "app.py\t2020-01-02 …" and the engine rejected an
            # ordinary `diff -u` patch with "Target file does not exist",
            # naming a file nobody asked for.
            for _pl in work_patch.splitlines():
                if _pl.startswith("+++ b/"):
                    _check_target = _pl[6:].split("\t")[0].strip()
                    break
                if _pl.startswith("+++ ") and not _pl.startswith("+++ /dev/null"):
                    _check_target = _pl[4:].split("\t")[0].strip()
                    break
        if not _patch_is_new_file and _check_target:
            _tf_full = Path(self.repo_root) / _check_target
            # Path-traversal guard: `../outside.py`-style targets must never
            # resolve outside the repo, even when such a file happens to exist
            # (a sibling temp dir or parallel-session artifact would otherwise
            # let a no-op/context-only patch report success against it).
            try:
                _tf_resolved = _tf_full.resolve()
                _repo_resolved = Path(self.repo_root).resolve()
                _tf_escapes = not _tf_resolved.is_relative_to(_repo_resolved)
            except Exception:
                _tf_escapes = False
            if _tf_escapes:
                metadata["reason"] = "path_escapes_repo"
                metadata["mode"] = "early_exit"
                return PatchResult(
                    success=False,
                    patch_applied=None,
                    error=(
                        f"Target file escapes the repository root: {_check_target}. "
                        f"Refusing to apply a patch outside the repo."
                    ),
                    metadata=metadata,
                )
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
            elif _target_git_state == "tracked" and self._patch_index_shas_are_fake(work_patch):
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
                _skip_3way = True
                metadata["skip_3way_reason"] = "fake_index_sha"

        # Step 1d: No-op detection — hunks with only context lines change
        # nothing. A context-only patch (e.g. a `diff -u` snapshot) is
        # trivially "applied": the tree is already in the desired state. This
        # also sidesteps Apple git 2.39.5's SIGBUS on whitespace near-miss
        # context under -C0 (see _tolerant_git_apply).
        if self._patch_is_noop(work_patch):
            if not _check_target:
                # Context-only (no-op) patch with NO resolvable target path:
                # there is nothing to verify the no-op against, and a success
                # here would make `apply_patch {"patch": "@@...", "path": ""}`
                # silently succeed. Refuse with a missing-path failure.
                metadata["reason"] = "noop_missing_path"
                metadata["mode"] = "noop_rejected"
                return PatchResult(
                    success=False,
                    patch_applied=None,
                    error=(
                        "Context-only (no-op) patch carries no target file "
                        "path; cannot apply. Provide the 'path' argument or "
                        "a file header."
                    ),
                    metadata=metadata,
                )
            metadata["reason"] = "noop_success"
            metadata["mode"] = "noop"
            return PatchResult(success=True, patch_applied=work_patch, metadata=metadata)

        # Step 2: Try git apply (primary path)
        self._add_step(metadata, "git_apply", "Attempting git apply")
        _primary_broken = False
        if not norm_error and self._diff_apply:
            try:
                _ga_ok, _ga_msg, _ga_reason, _ga_details = self._diff_apply(
                    self.repo_root,
                    work_patch,
                    file_path_hint=target_file,
                    skip_3way=_skip_3way,
                )
                if _ga_ok:
                    metadata["reason"] = "git_apply_success"
                    metadata["mode"] = "git_apply"
                    return PatchResult(success=True, patch_applied=work_patch, metadata=metadata)
                metadata["first_fail_reason"] = _ga_msg or _ga_reason or "git apply failed"
            except Exception as e:
                metadata["first_fail_reason"] = f"git apply exception: {e}"
                # The applier itself is broken — do not spend tolerant/reanchor
                # attempts (they bypass _diff_apply and could mask the failure);
                # fall straight to the repair ladder.
                _primary_broken = True
        elif not self._diff_apply:
            metadata["first_fail_reason"] = "diff_apply module not available"
            _primary_broken = True

        # Step 2b/2c: Tolerant git apply variants + line re-anchoring.
        # Skipped when the primary applier is broken (exception/unavailable) —
        # the tolerant path bypasses _diff_apply and could mask the failure;
        # fall straight to the repair ladder instead.
        if not _primary_broken:
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
                tol_ok2, _tol_err2, tol_mode2 = self._tolerant_git_apply(
                    reanchored, target_file, allow_3way=not _skip_3way
                )
                if tol_ok2:
                    metadata["reason"] = f"reanchor_success:{tol_mode2}"
                    metadata["mode"] = f"reanchor_{tol_mode2}"
                    metadata["fallback_used"] = ["reanchor", tol_mode2]
                    return PatchResult(success=True, patch_applied=reanchored, metadata=metadata)
                # Also try primary diff_apply on reanchored patch
                if self._diff_apply:
                    try:
                        _ra_ok, _ra_msg, _ra_reason, _ra_details = self._diff_apply(
                            self.repo_root,
                            reanchored,
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
            target_file=target_file or "",
            failure_reason=metadata["first_fail_reason"],
            llm_output=llm_output,
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
                if merged_metadata.get("applied"):
                    # repair_patch already applied the repaired patch (direct
                    # callers observe an applied tree) — skip the re-apply,
                    # which would fail on the already-consumed pre-image lines.
                    return PatchResult(
                        success=True, patch_applied=repair_result.patch_applied, metadata=merged_metadata
                    )
                apply_ok, apply_err = self._apply_diff_once(repair_result.patch_applied, target_file)
                if apply_ok:
                    return PatchResult(
                        success=True, patch_applied=repair_result.patch_applied, metadata=merged_metadata
                    )
                # Repaired patch failed to apply
                merged_metadata["reason"] = "repaired_patch_apply_failed"
                merged_metadata["second_fail_reason"] = apply_err
                return PatchResult(
                    success=False, error=f"Repaired patch failed to apply: {apply_err}", metadata=merged_metadata
                )
            # No patch produced (should not happen)
            merged_metadata["reason"] = "repaired_patch_missing"
            return PatchResult(success=False, error="Repair succeeded but no patch produced", metadata=merged_metadata)
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
        return PatchResult(success=False, error=_final_err, metadata=metadata)

    def synthesize_and_apply(self, llm_output: str, target_file: str, output_mode: str = "auto") -> PatchResult:
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
                    return PatchResult(success=False, error="Output mode enumeration not available", metadata=metadata)

                # Parse LLM output using hybrid parser
                parsed = self.hybrid_parser.parse(llm_output, expected_mode)
                if not parsed.success:
                    metadata["first_fail_reason"] = f"parse failed: {parsed.error}"
                    return PatchResult(
                        success=False, error=f"Failed to parse LLM output: {parsed.error}", metadata=metadata
                    )

                # Check for NEEDS_DISAMBIGUATION
                if parsed.mode is None:
                    # This indicates NEEDS_DISAMBIGUATION
                    metadata["synth_reason"] = "needs_disambiguation"
                    return PatchResult(success=False, error="LLM output requires disambiguation", metadata=metadata)

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
                    metadata={"parsed_mode": parsed.mode.value},
                )
                return self.apply_patch(synthesized, target_file, context)
            except Exception as e:
                metadata["first_fail_reason"] = f"synthesis failed: {e}"
                return PatchResult(success=False, error=f"LLM output synthesis failed: {e}", metadata=metadata)
        else:
            missing = []
            if not self.hybrid_parser:
                missing.append("hybrid_parser")
            if not self.patch_synthesizer:
                missing.append("patch_synthesizer")
            if OutputMode is None:
                missing.append("output_modes")
            if missing == ["output_modes"]:
                # Only the mode enumeration is unavailable (components are
                # fine) — report the enum specifically so callers can react
                # to "cannot enumerate output modes" rather than a generic
                # component failure.
                metadata["first_fail_reason"] = "output_mode enum not available"
                return PatchResult(success=False, error="Output mode enumeration not available", metadata=metadata)
            metadata["first_fail_reason"] = f"components not available: {', '.join(missing)}"
            return PatchResult(
                success=False,
                error=f"LLM output synthesis components not available: {', '.join(missing)}",
                metadata=metadata,
            )

    def _auto_repair_patch(self, patch: str, target_file: str) -> str | None:
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

            # Improved diff parsing: handle unified diff format properly
            lines = patch.splitlines()
            new_lines = []
            in_hunk = False
            for line in lines:
                if line.startswith("@@"):
                    in_hunk = True
                    continue
                if in_hunk and line.startswith("+") and not line.startswith("+++"):
                    # Skip header lines (+++ b/file)
                    new_lines.append(line[1:])  # Remove leading '+'

            new_code = "\n".join(new_lines).strip()

            if not new_code:
                return None

            header = new_code.lstrip()

            symbol_name, symbol_kind = extract_symbol_name(header)
            if symbol_name:
                if symbol_kind == "function":
                    result = rewriter.replace_function(file_path, symbol_name, new_code)
                elif symbol_kind == "class":
                    result = rewriter.replace_class(file_path, symbol_name, new_code)
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
        tgt_rel = normalize_rel_path(str(target_file))
        # P22-2: containment guard — prompt-derived paths must not escape the repo.
        try:
            tgt_path = resolve_inside_repo(repo_root, tgt_rel)
        except ValueError:
            return ("", "target_missing")

        if not tgt_path.exists() or not tgt_path.is_file():
            return ("", "target_missing")

        # P22-4: reject over-size targets before reading — old_text was previously
        # read unbounded (only new_text was capped); the bytes proxy is fail-safe.
        try:
            if tgt_path.stat().st_size > int(self._MAX_FILE_CHARS):
                return ("", "file_too_large")
        except OSError:
            return ("", "read_failed")

        try:
            old_text = tgt_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ("", "read_failed")

        # Multi-file guard: more than one FILE:/Path:/Target file: marker means
        # the LLM emitted several rewrite blocks — refuse instead of silently
        # applying the first (the block parsers may surface only one).
        if len(re.findall(r"(?im)^\s*(?:FILE|Path|Target file)\s*:", llm_text or "")) > 1:
            return ("", "multi_file_block")

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
            rel = normalize_rel_path(p)
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
                tail = m.group("fencetail2")
                if tail is None:
                    tail = m.group("fencetail1")
                tail = (tail or "").strip()
                if tail and not _fence_tail_is_lang_tag(tail):
                    code = tail + "\n" + str(code)
                code = self._strip_trailing_fences(str(code))
                rel = normalize_rel_path(p)
                if rel:
                    blocks.append((rel, code))

        if not blocks:
            return ("", "no_file_block")

        # Pick best match: exact target path, else basename match
        chosen_rel: str | None = None
        new_text: str | None = None

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

        # Shared synthesis tail: valve + diff generation for the chosen block.
        return self._synthesize_full_file_diff(repo_root, tgt_rel, new_text, old_text=old_text)

    _SYMBOL_HEAD_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+")

    @staticmethod
    def _single_symbol_snippet(text: str) -> bool:
        """True when *text* is a SINGLE top-level def/class block — the
        function-level snippet case. Multi-symbol text (a full-file rewrite)
        must not be treated as a snippet."""
        if not text:
            return False
        heads = [ln for ln in text.splitlines() if PatchEngine._SYMBOL_HEAD_RE.match(ln)]
        return len(heads) == 1

    @classmethod
    def _replace_symbol_block(cls, old_text: str, new_text: str) -> str | None:
        """Replace the single top-level def/class block named by the first line
        of *new_text* inside *old_text*. Returns the merged text, or None when
        the symbol's block cannot be located."""
        new_lines = new_text.splitlines()
        if not new_lines:
            return None
        first = new_lines[0].rstrip()
        if not cls._SYMBOL_HEAD_RE.match(first):
            return None
        old_lines = old_text.splitlines()
        for i, ln in enumerate(old_lines):
            if ln.rstrip() != first:
                continue
            j = i + 1
            while j < len(old_lines) and (not old_lines[j].strip() or old_lines[j][:1] in (" ", "\t")):
                j += 1
            merged = old_lines[:i] + new_lines + old_lines[j:]
            out = "\n".join(merged)
            if old_text.endswith("\n"):
                out += "\n"
            return out
        return None

    def _synthesize_full_file_diff(
        self, repo_root: str, target_file: str, new_text: str, old_text: str | None = None
    ) -> tuple[str, str]:
        """Synthesize a unified diff for a full-file rewrite candidate.

        Shared by ``_try_synthesize_diff_from_file_blocks`` and the repair
        ladder's file-block fallback. Returns ``(patch, reason)``; reason is
        ``"file_block_synth"`` on success or one of target_missing /
        read_failed / file_too_large / no_changes / file_rewrite_too_large /
        patch_too_large.
        """
        tgt_rel = normalize_rel_path(str(target_file))
        try:
            tgt_path = resolve_inside_repo(repo_root, tgt_rel)
        except ValueError:
            return ("", "target_missing")
        if not tgt_path.exists() or not tgt_path.is_file():
            return ("", "target_missing")
        try:
            if tgt_path.stat().st_size > int(self._MAX_FILE_CHARS):
                return ("", "file_too_large")
        except OSError:
            return ("", "read_failed")
        if old_text is None:
            try:
                old_text = tgt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ("", "read_failed")

        # Normalize new text to ensure trailing newline (git-style)
        if not new_text.endswith("\n"):
            new_text = new_text + "\n"

        if len(new_text) > int(self._MAX_FILE_CHARS):
            return ("", "file_too_large")

        if old_text == new_text:
            return ("", "no_changes")

        # Partial-file python snippet: a single top-level def/class block that
        # exists in the file is a function-level edit, not a full-file rewrite
        # (the rewrite valve below would reject it as file_rewrite_too_large).
        if self._single_symbol_snippet(new_text):
            replaced = self._replace_symbol_block(old_text, new_text)
            if replaced is not None and replaced != old_text:
                new_text = replaced

        # Safety valve: reject over-large rewrites.
        # FILE blocks are full rewrites; if the model drifts and rewrites most of the file,
        # we would rather fail than apply a huge, hard-to-review patch.
        # NOTE (P22-1): compare LINE sequences with autojunk disabled. A
        # character-level SequenceMatcher autojunk treats any element appearing
        # in >1% of b (for b >= 200 chars) as junk — for text that means \n,
        # space, '(', etc. — so ratio() collapses toward 0 and a single-line
        # edit reads as a ~100% rewrite (file_rewrite_too_large for every file
        # beyond ~40-50 KB). changed_lines_est below already assumes a line
        # ratio, so the unit must be lines here too.
        try:
            old_lines = old_text.splitlines()
            new_lines = new_text.splitlines()
            sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
            change_ratio = 1.0 - float(sm.ratio())
        except Exception:
            old_lines = []
            new_lines = []
            change_ratio = 1.0

        try:
            max_lines = max(len(old_lines), len(new_lines), 1)
            # Approx: changed lines ~= (1 - similarity) * max(lines)
            changed_lines_est = round(change_ratio * max_lines)
        except Exception:
            changed_lines_est = 10**9

        if (change_ratio > float(self._MAX_FILE_REWRITE_CHANGE_RATIO)) or (
            changed_lines_est > int(self._MAX_FILE_REWRITE_CHANGED_LINES)
        ):
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
        return re.sub(r"\n```[\s\S]*\Z", "\n", t)

    def _salvage_small_model_output(self, patch_text: str, target_file: str) -> str | None:
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

            rel = str(target_file).strip().lstrip("/")
            if not rel:
                return None

            # P22-2: containment guard — prompt-derived paths must not escape the repo.
            try:
                abs_path = resolve_inside_repo(self.repo_root, rel)
            except ValueError:
                logger.debug("salvage_small_model_output: escape path rejected (%s)", rel)
                return None
            if not abs_path.exists() or not abs_path.is_file():
                return None

            # P22-4: skip huge files on size alone — do not load a file we are
            # about to skip anyway (the >2000-line check below is O(NxMxL);
            # 256 KiB is a conservative bytes proxy for ~2000+ lines).
            try:
                st_size = abs_path.stat().st_size
            except OSError:
                logger.debug("salvage_small_model_output: stat failed for %s", rel)
                return None
            if st_size > int(self._SALVAGE_SKIP_MAX_BYTES):
                logger.debug(
                    "salvage_small_model_output: skipping large file by size (st_size=%d, target=%s)",
                    st_size,
                    rel,
                )
                return None

            old_text = abs_path.read_text(encoding="utf-8", errors="replace")
            old_lines = old_text.splitlines()

            # Skip expensive fuzzy-salvage for very large files. Strategy 1/2
            # below use an O(NxMxL) sliding-window SequenceMatcher.ratio() over
            # the WHOLE file with no window limit, unlike _reanchor_patch()
            # which caps at >2000 lines (see `_reanchor_patch`). Salvage is best-effort
            # recovery for malformed small-model output; large files are
            # handled by the tolerant-apply + reanchor ladder, which already
            # degrades gracefully past 2000 lines.
            if len(old_lines) > 2000:
                logger.debug(
                    "salvage_small_model_output: skipping fuzzy salvage for large file (%d lines, target=%s)",
                    len(old_lines),
                    rel,
                )
                return None

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
                elif re.match(r"^before\s*:", line, re.IGNORECASE):
                    before_content = []
                    i += 1
                    while i < len(lines) and not re.match(r"^after\s*:", lines[i], re.IGNORECASE):
                        before_content.append(lines[i])
                        i += 1
                    if i < len(lines) and re.match(r"^after\s*:", lines[i], re.IGNORECASE):
                        i += 1
                        after_content = []
                        while (
                            i < len(lines)
                            and lines[i].strip()
                            and not re.match(r"^before\s*:|^after\s*:", lines[i], re.IGNORECASE)
                        ):
                            after_content.append(lines[i])
                            i += 1
                        before_blocks.append("\n".join(before_content))
                        after_blocks.append("\n".join(after_content))
                        continue

                # Ed-style commands (e.g., "1c1", "2a", "3d")
                elif re.match(r"^\d+[acd]\d*$", line):
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
                        window = old_lines[start_idx : start_idx + n_removed]
                        # P24-B: autojunk=False — repeated lines (blank,
                        # common tokens) in >=200-line windows would be
                        # purged, skewing the 0.6 threshold below.
                        sm = difflib.SequenceMatcher(None, window, removed_lines, autojunk=False)
                        ratio = sm.ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_idx = start_idx

                    # Require at least 60% line-level match (prevents partial corruption)
                    if best_idx is not None and best_ratio >= 0.6:
                        # Replacement-consistency guard: refuse to collapse more
                        # removed lines into fewer added lines (a truncated /
                        # corrupt output silently deleting code). The exception
                        # is a match covering the ENTIRE file, which is a full
                        # rewrite rather than a partial edit.
                        whole_file_match = best_idx == 0 and n_removed == len(old_lines)
                        if len(added_lines) < n_removed and not whole_file_match:
                            logger.debug(
                                "salvage_small_model_output: strategy 1 rejected — "
                                "replacement collapses %d removed lines into %d added",
                                n_removed,
                                len(added_lines),
                            )
                        else:
                            new_lines = list(old_lines)
                            new_lines[best_idx : best_idx + n_removed] = added_lines

                            diff_lines = list(
                                difflib.unified_diff(
                                    old_lines,
                                    new_lines,
                                    fromfile=f"a/{rel}",
                                    tofile=f"b/{rel}",
                                    lineterm="",
                                )
                            )

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
                        if new_lines[start_idx : start_idx + n_before] == before_lines:
                            new_lines[start_idx : start_idx + n_before] = after_lines
                            found = True
                            break

                    if not found:
                        # Fuzzy match using line-level SequenceMatcher (full-block only)
                        best_idx = None
                        best_ratio = 0.0
                        for start_idx in range(len(new_lines) - n_before + 1):
                            window = new_lines[start_idx : start_idx + n_before]
                            # P24-B: autojunk=False (see above).
                            sm = difflib.SequenceMatcher(None, window, before_lines, autojunk=False)
                            ratio = sm.ratio()
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_idx = start_idx

                        if best_idx is not None and best_ratio >= 0.7:
                            new_lines[best_idx : best_idx + n_before] = after_lines
                        else:
                            all_succeeded = False
                            break

                if all_succeeded:
                    diff_lines = list(
                        difflib.unified_diff(
                            old_lines,
                            new_lines,
                            fromfile=f"a/{rel}",
                            tofile=f"b/{rel}",
                            lineterm="",
                        )
                    )

                    if diff_lines:
                        diff_text = "\n".join(diff_lines)
                        if not diff_text.startswith("diff --git "):
                            diff_text = f"diff --git a/{rel} b/{rel}\n" + diff_text
                        if not diff_text.endswith("\n"):
                            diff_text += "\n"
                        return diff_text

            # Strategy 3: Symbol-aware insertion for added lines.
            # An output carrying BOTH - and + lines expresses a REPLACEMENT,
            # not an insertion: once strategy 1 rejected it (e.g. a collapse
            # into unrelated content), treating the + lines as free insertions
            # would silently apply the rejected garbage.
            if added_lines and not removed_lines:
                # Filter out lines already present
                old_lines_set = set(old_lines)  # O(n²) → O(n) : set lookup is O(1)
                insert_lines = [ln for ln in added_lines if ln.strip() and ln not in old_lines_set]
                if not insert_lines:
                    return None

                # Try to find insertion point using symbol search if available
                anchor_index = None
                with contextlib.suppress(ImportError):
                    from .agent.symbol_search import get_symbol_searcher

                    searcher = get_symbol_searcher(self.repo_root)

                    # Look for function/class definitions in added lines
                    for line in insert_lines:
                        # Simple heuristic: look for "def " or "class " in Python
                        if "def " in line or "class " in line:
                            symbol = line.split("def ")[-1].split("class ")[-1].split("(")[0].strip()
                            if not symbol:
                                continue
                            # NOTE: the previous ``search(symbol, limit=5)``
                            # call did not exist on SymbolSearcher — it raised
                            # AttributeError every time, the function-level
                            # except swallowed it, and symbol-aware anchoring
                            # silently never ran (salvage aborted to None).
                            # find_symbol is the real API (SymbolDef.file/.line,
                            # repo-relative file).
                            results = searcher.find_symbol(symbol)
                            if results:
                                # Find the line after the last occurrence
                                for result in results:
                                    if result.file == rel:
                                        line_num = result.line
                                        if line_num > 0:
                                            anchor_index = line_num - 1  # Convert to 0-index
                                            break
                                if anchor_index is not None:
                                    break
                # Symbol search not available, fall back to heuristics

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

                diff_lines = list(
                    difflib.unified_diff(
                        old_lines,
                        new_text.splitlines(),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                        lineterm="",
                    )
                )

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
                    for _, line in enumerate(lines):
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
                    match = re.match(r"^(\d+)([acd])(\d*)$", cmd)
                    if match:
                        line_num = int(match.group(1))
                        command = match.group(2)
                        # Convert 1-indexed to 0-indexed
                        idx = line_num - 1 if line_num > 0 else 0

                        if command == "d" and 0 <= idx < len(new_lines):
                            del new_lines[idx]
                            changes_made = True
                        # Note: 'c' and 'a' would need content from following lines
                        # This is a simplified implementation

                if changes_made:
                    new_text = "\n".join(new_lines)
                    if old_text.endswith("\n"):
                        new_text += "\n"

                    diff_lines = list(
                        difflib.unified_diff(
                            old_lines,
                            new_text.splitlines(),
                            fromfile=f"a/{rel}",
                            tofile=f"b/{rel}",
                            lineterm="",
                        )
                    )

                    if diff_lines:
                        diff_text = "\n".join(diff_lines)
                        if not diff_text.startswith("diff --git "):
                            diff_text = f"diff --git a/{rel} b/{rel}\n" + diff_text
                        if not diff_text.endswith("\n"):
                            diff_text += "\n"
                        return diff_text

        except Exception:
            # Any unexpected error → salvage failed
            logger.debug("_salvage_small_model_output: salvage failed", exc_info=True)
            return None
        else:
            return None

    def convert_patch_to_edit_blocks(self, patch: str, target_file: str | None = None) -> dict | None:
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
                m = re.match(r"^\+\+\+ b/(.+)$", line)
                if m:
                    file_path = m.group(1).strip()
                    break
                m2 = re.match(r"^diff --git a/\S+ b/(.+)$", line)
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

    def repair_patch(
        self, patch_text: str, target_file: str, failure_reason: str, llm_output: str | None = None
    ) -> PatchResult:
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
            self._add_step(metadata, "no_llm_fallback", "No LLM output — trying auto-repair from patch text")
            auto_fix = self._auto_repair_patch(patch_text, target_file)
            if auto_fix:
                metadata["reason"] = "repair_success:auto_repair"
                metadata["mode"] = "auto_repair"
                metadata["fallback_used"] = ["auto_repair"]
                metadata["applied"] = self._apply_repair_patch(auto_fix, target_file)
                return PatchResult(success=True, patch_applied=auto_fix, metadata=metadata)
            metadata["reason"] = "no_llm_fallback_all_failed"
            return PatchResult(
                success=False,
                error="Cannot repair patch without original LLM output and auto-repair from patch text failed",
                metadata=metadata,
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

        new_code = ""
        if parsed_blocks:
            # Prefer the block that matches target_file; fall back to first block
            block = parsed_blocks[0]
            try:
                normalized_target = normalize_rel_path(str(target_file))
                for candidate in parsed_blocks:
                    candidate_path = normalize_rel_path(
                        str(candidate.get("path") or candidate.get("file") or candidate.get("filename") or "")
                    )
                    if candidate_path and normalized_target and candidate_path == normalized_target:
                        block = candidate
                        break
            except Exception:
                block = parsed_blocks[0]
            new_code = block.get("text") or block.get("content") or ""

        if not new_code.strip():
            # No usable FILE: block — header/fence forms (FUNCTION:/CLASS:/
            # METHOD: or a bare fenced def) still drive the AST/symbol/semantic
            # ladder below, so extract the fenced body directly instead of
            # bailing out (the header dispatch in step 1 needs new_code).
            new_code = self._extract_code_from_llm_output(llm_output or "") or ""

        if not new_code.strip():
            if self._llm_output_has_code_shape(llm_output or ""):
                metadata["reason"] = "empty_new_code"
                return PatchResult(success=False, error="Empty code block in LLM output", metadata=metadata)
            metadata["reason"] = "no_parsed_blocks"
            return PatchResult(success=False, error="No parseable file blocks found in LLM output", metadata=metadata)

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
                    result = self.ast_rewriter.replace_function(target_file, func_name, new_code)
                    result_patch = self.ast_rewriter.generate_patch(target_file, result)
                    result_mode = "ast_function"
                    fallback_succeeded = True
                elif llm_header.startswith("CLASS:"):
                    class_name = llm_header.split("CLASS:")[1].strip()
                    result = self.ast_rewriter.replace_class(target_file, class_name, new_code)
                    result_patch = self.ast_rewriter.generate_patch(target_file, result)
                    result_mode = "ast_class"
                    fallback_succeeded = True
                elif llm_header.startswith("METHOD:"):
                    path = llm_header.split("METHOD:")[1].strip()
                    # rsplit on the LAST dot so class_name may be a nested-class
                    # chain (e.g. "A.B.method" -> class_name="A.B", method_name="method"),
                    # matching replace_method's documented nested-class support.
                    # rsplit on the LAST dot so class_name may be a nested-class
                    # chain (e.g. "A.B.method" -> class_name="A.B", method_name="method"),
                    # matching replace_method's documented nested-class support.
                    class_name, method_name = path.rsplit(".", 1)
                    result = self.ast_rewriter.replace_method(target_file, class_name, method_name, new_code)
                    result_patch = self.ast_rewriter.generate_patch(target_file, result)
                    result_mode = "ast_method"
                    fallback_succeeded = True
                elif is_python_definition(new_code):
                    func_name, _ = extract_symbol_name(new_code)
                    if func_name is not None:
                        result = self.ast_rewriter.replace_function(target_file, func_name, new_code)
                        result_patch = self.ast_rewriter.generate_patch(target_file, result)
                        result_mode = "ast_autodetect"
                        fallback_succeeded = True
            except Exception as e:
                logger.debug("AST rewrite attempt failed: %s", e)
                metadata["second_fail_reason"] = f"ast_rewrite_failed: {e}"
                # A crash inside the header-dispatched AST rewrite is a tooling
                # failure, not a "symbol not found" — stop the ladder rather
                # than masking it with a weaker fallback.
                metadata["reason"] = "all_repair_failed"
                metadata["fallback_used"] = fallback_attempted
                return PatchResult(
                    success=False, error=f"All repair attempts failed: {failure_reason}", metadata=metadata
                )

        # 2. Symbol search fallback
        if not fallback_succeeded and self.symbol_searcher and self.ast_rewriter:
            self._add_step(metadata, "symbol_search", "Attempting symbol search fallback")
            fallback_attempted.append("symbol_search")
            try:
                header = new_code.strip().splitlines()[0].strip()
                symbol_name, symbol_kind = extract_symbol_name(header)

                if symbol_name:
                    results = self.symbol_searcher.find_symbol(
                        symbol_name, kind=symbol_kind if symbol_kind != "function" else "any"
                    )
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
                            new_code,
                        )
                        result_patch = self.ast_rewriter.generate_patch(sym.file, result)
                        result_mode = "ast_symbol_function"
                        fallback_succeeded = True
                    elif sym.kind == "class":
                        result = self.ast_rewriter.replace_class(sym.file, sym.name, new_code)
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
                    result_mode = "semantic_class" if sem_result.kind == "class" else "semantic_function"
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
                if not fallback_succeeded:
                    # Bare-fence full-file output (no FILE: block): synthesize
                    # the diff directly from the extracted code.
                    synth_patch, synth_reason = self._synthesize_full_file_diff(self.repo_root, target_file, new_code)
                    if synth_reason == "file_block_synth":
                        result_patch = synth_patch
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
            # Apply the repaired patch immediately so direct callers observe an
            # applied tree; apply_patch honors the "applied" flag and skips its
            # own re-apply (the pre-image lines are already gone by then).
            metadata["applied"] = bool(result_patch) and self._apply_repair_patch(result_patch, target_file)
            return PatchResult(success=True, patch_applied=result_patch, metadata=metadata)
        metadata["reason"] = "all_repair_failed"
        metadata["fallback_used"] = fallback_attempted
        metadata["second_fail_reason"] = metadata.get("second_fail_reason", "all fallbacks failed")
        return PatchResult(success=False, error=f"All repair attempts failed: {failure_reason}", metadata=metadata)

    def _apply_repair_patch(self, patch_text: str, target_file: str) -> bool:
        """Apply a successfully repaired patch immediately so direct callers
        observe an applied tree. Returns whether the apply succeeded;
        ``apply_patch`` re-applies only when this is False."""
        try:
            ok, _err = self._apply_diff_once(patch_text, target_file)
            return bool(ok)
        except Exception as e:
            logger.debug("apply repaired patch failed for %s: %s", target_file, e)
            return False

    _FENCE_BLOCK_RE = re.compile(r"```(?P<tail>[^\n\r]*)\r?\n(?P<code>[\s\S]*?)\r?\n```")

    @classmethod
    def _extract_code_from_llm_output(cls, llm_output: str) -> str | None:
        """First fenced body in header-style LLM output (no FILE: block).

        Handles `````code`` fence-line tails like FILE-block parsing: a plain
        language tag is dropped, anything else becomes the first code line.
        Returns the code (possibly empty for an empty fence) or None when the
        output carries no fenced block.
        """
        text = (llm_output or "").strip()
        if not text:
            return None
        m = cls._FENCE_BLOCK_RE.search(text)
        if not m:
            return None
        code = m.group("code")
        tail = (m.group("tail") or "").strip()
        if tail and not _fence_tail_is_lang_tag(tail):
            code = tail + "\n" + code
        code = code.replace("\r\n", "\n").strip("\n") + "\n"
        return code if code.strip() else ""

    @staticmethod
    def _llm_output_has_code_shape(llm_output: str) -> bool:
        """True when the output looks like a code block/header form even if
        nothing parseable was extracted (distinguishes "empty_new_code" from
        "no_parsed_blocks")."""
        text = (llm_output or "").strip()
        return "```" in text or bool(re.match(r"^(?:FUNCTION|CLASS|METHOD)\s*:", text))

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
        start_idx: int | None = None
        for i, line in enumerate(lines):
            if line.startswith(("diff --git ", "--- ", "+++ ")):
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
        Normalize LLM-generated patch text by cleaning up common formatting issues.

        External LLMs sometimes:
          - prepend BOM (\\ufeff)
          - indent diff markers with spaces/tabs
          - wrap diffs inside markdown fences (```diff ... ```)
        Any of these can make `git apply` fail to recognize headers/hunks properly.

        We normalize by:
          - removing BOM at line starts
          - dropping markdown fence lines (header region only)
          - left-stripping lines that *look like* diff markers (header region only)

        Region tracking: once an ``@@`` hunk header is seen we enter the *body*
        region. Body lines carry a significant leading char (' ' context, '+'
        add, '-' remove, '\\\\' no-newline) and must be preserved verbatim — even
        if their *content* looks like a diff marker. A context line such as
        ``' +++ b/other.py'`` is legitimate file content (e.g. when the edited
        file itself contains patch/markdown text), not a header; de-indenting it
        would corrupt the patch. Only the header region (between sections) is
        de-indented. A new ``diff --git`` line returns us to the header region.
        """
        if not patch:
            return ""

        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")
        out: list[str] = []

        in_hunk = False  # True while inside a hunk body (after an '@@' header)

        for line in lines:
            if not line:
                out.append(line)
                continue

            # Remove BOM at the start of a line (safe in both regions)
            if line and line[0] == "\ufeff":
                line = line.lstrip("\ufeff")

            stripped = line.lstrip()

            if in_hunk:
                # Body region: a new section header returns to header mode.
                if stripped.startswith("diff --git "):
                    out.append(stripped)
                    in_hunk = False
                else:
                    # Preserve body lines verbatim — leading ' '/'+'/'-'/'\\' is
                    # significant. Do not lstrip or drop fences here.
                    out.append(line)
                continue

            # Header region
            # Drop markdown fences that often leak into patch output
            if stripped.startswith("```"):
                continue

            if stripped.startswith(
                (
                    "diff --git ",
                    "--- ",
                    "+++ ",
                    "@@ ",
                    "index ",
                    "new file mode ",
                    "deleted file mode ",
                    "similarity index ",
                    "rename from ",
                    "rename to ",
                )
            ):
                # An '@@' hunk header transitions us into the body region.
                if stripped.startswith("@@ "):
                    in_hunk = True
                out.append(stripped)
            else:
                out.append(line)

        normalized = "\n".join(out).strip()
        if normalized and not normalized.endswith("\n"):
            normalized += "\n"
        return normalized

    @staticmethod
    def _keep_only_target_file_section(patch: str, target_file: str | None) -> str:
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
        tf = normalize_rel_path(target_file or "")
        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")

        # Locate diff --git boundaries
        diff_idxs = [i for i, _item_ in enumerate(lines) if _item_.startswith("diff --git ")]
        if diff_idxs:
            sections: list[tuple[int, int, str, str]] = []
            for s_i, start in enumerate(diff_idxs):
                end = diff_idxs[s_i + 1] if (s_i + 1) < len(diff_idxs) else len(lines)
                m = _RE_DIFF_GIT.match(lines[start] or "")
                a_path = m.group(1) if m else ""
                b_path = m.group(2) if m else ""
                sections.append((start, end, a_path, b_path))

            chosen = sections[0]
            if tf:
                # Two-pass: prefer an EXACT path match across ALL sections
                # before falling back to basename equality. A single-pass
                # loop lets a basename collision win over a later exact match
                # (e.g. choosing src/utils.py when the target is
                # tests/utils.py), after which _force_target_file_paths would
                # rewrite the wrong section's headers onto the target and
                # apply an unrelated file's changes.
                for sec in sections:
                    _s, _e, a_path, b_path = sec
                    if tf in (a_path, b_path):
                        chosen = sec
                        break
                else:
                    for sec in sections:
                        _s, _e, a_path, b_path = sec
                        if Path(a_path).name == Path(tf).name or Path(b_path).name == Path(tf).name:
                            chosen = sec
                            break

            s, e, _a, _b = chosen
            kept = "\n".join(lines[s:e]).strip()
            if kept and not kept.endswith("\n"):
                kept += "\n"
            return kept

        # No diff --git sections. Keep a single ---/+++ file section.
        start_idx: int | None = None
        for i, _item_ in enumerate(lines):
            if _item_.startswith("--- "):
                start_idx = i
                break
        if start_idx is None:
            return txt.strip() + ("\n" if txt.strip() else "")

        # determine the file path from the first header, if possible
        first_file: str | None = None
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
    def _force_target_file_paths(patch: str, target_file: str | None) -> str:
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
        tf = normalize_rel_path(target_file or "")
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
                    if (a_path == base and b_path == base) or (Path(a_path).name == base and Path(b_path).name == base):
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
    def _ensure_headers_before_any_hunk(patch: str, target_file: str | None) -> str:
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

        tf = normalize_rel_path(target_file or "")
        if not tf:
            return patch

        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")

        first_hunk_idx: int | None = None
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
    def _normalize_patch_headers(patch: str, target_file: str | None) -> str:
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

        tf = normalize_rel_path(target_file or "")
        txt = str(patch).replace("\r\n", "\n")
        lines = txt.split("\n")

        out: list[str] = []
        have_minus = False
        have_plus = False
        cur_a: str | None = None
        cur_b: str | None = None

        for line in lines:
            m = _RE_DIFF_GIT.match(line or "")
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

    def normalize_and_validate(self, patch_text: str, target_file: str | None) -> tuple[str, str | None]:
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
                    "(non-fatal, tolerant path may still succeed): %s",
                    err,
                )
            return p, None
        return p, "Patch does not look like a unified diff"

    def _looks_like_unified_diff(self, text: str) -> bool:
        """Heuristic check for unified diff format."""
        t = str(text or "")
        if not t.strip():
            return False
        # Heuristic: any real diff header + at least one hunk marker
        has_header = any(s in t for s in ("diff --git ", "--- a/", "+++ b/")) or t.lstrip().startswith("--- ")
        has_hunk = "@@ " in t
        # Allow hunk-only patches (starting with @@, no header) — git apply handles them
        return bool(has_hunk and (has_header or t.lstrip().startswith("@@")))

    def _git_apply_check_best_effort(self, patch_text: str) -> tuple[bool, str | None]:
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
                check=False,
            )
            if result.returncode == 0:
                return True, None
            error = result.stderr.decode("utf-8", errors="ignore").strip()
            if not error:
                error = result.stdout.decode("utf-8", errors="ignore").strip()
            if not error:
                error = f"git apply --check failed with exit code {result.returncode}"
        except Exception as e:
            logger.debug("git apply --check failed with exception: %s", e)
            return False, f"git apply --check exception: {e}"
        else:
            return False, error

    def _classify_target_git_state(self, target_file: str | None) -> str:
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
                check=False,
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
                check=False,
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
        except Exception as e:
            logger.debug("_classify_target_git_state failed: %s", e)
            return "unknown"
        else:
            return "tracked"

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

        All SHAs are probed in ONE `git cat-file --batch-check` subprocess
        (stdin line protocol) instead of one `git cat-file -e` per SHA — the
        batch form keeps N spawns → 1 while preserving every observable
        behavior: first missing SHA trips the gate, probe failure is
        conservative (returns False, keeps prior 3-way behavior).

        Returns True  — an `index` line names a SHA absent from the store.
        Returns False — no index line, all SHAs resolve, or the probe itself
                        failed (conservative: keep prior 3-way behavior).
        """
        shas: list[str] = []
        for m in re.finditer(
            r"^index ([0-9a-f]{7,40})\.\.([0-9a-f]{7,40})",
            patch_text,
            re.MULTILINE,
        ):
            for sha in (m.group(1), m.group(2)):
                # All-zero SHA = legitimate placeholder for file creation
                # (old side) or deletion (new side); skip it.
                if sha.strip("0") == "":
                    continue
                shas.append(sha)
        if not shas:
            return False
        try:
            chk = subprocess.run(
                ["git", "cat-file", "--batch-check"],
                cwd=self.repo_root,
                input=("".join(sha + "\n" for sha in shas)).encode("utf-8"),
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception as e:
            logger.debug("cat-file batch probe failed for %s: %s", shas, e)
            return False
        if chk.returncode != 0:
            return False
        # --batch-check echoes one line per input SHA, in order:
        # "<sha> <type> <size>" for existing objects, "<sha> missing" otherwise
        # (abbrev SHAs resolve via the same disambiguation as `cat-file -e`).
        for line in chk.stdout.decode("utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "missing":
                return True
        return False

    def _apply_diff_once(self, patch_text: str, target_file: str | None = None) -> tuple[bool, str | None]:
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
                    inferred_path = s[len("+++ b/") :].strip()
                    break
                if s.startswith("--- a/"):
                    inferred_path = s[len("--- a/") :].strip()
                    break
                if s.startswith("+++ "):
                    candidate = s[len("+++ ") :].strip()
                    if candidate and candidate != "/dev/null":
                        inferred_path = candidate.removeprefix("b/")
                        break
                if s.startswith("--- "):
                    candidate = s[len("--- ") :].strip()
                    if candidate and candidate != "/dev/null":
                        inferred_path = candidate.removeprefix("a/")
                        break
                if s.startswith("diff --git "):
                    parts = s.split()
                    if len(parts) >= 4 and parts[3].startswith("b/"):
                        inferred_path = parts[3][2:]
                        break

            if inferred_path:
                # Strip any pre-hunk --- /+++ fragments from the body so we
                # never emit TWO old/new headers. This branch is entered when
                # at least one header is missing (has_hunk and not (old and
                # new)); if exactly one was present it still lives in `stripped`,
                # so a naive prepend of both would duplicate it and produce a
                # "corrupt patch". Only column-0 header fragments before the
                # first '@@' are dropped; indented context lines are untouched.
                body_lines = stripped.split("\n")
                cleaned: list[str] = []
                seen_hunk = False
                for ln in body_lines:
                    if ln.startswith("@@"):
                        seen_hunk = True
                    if not seen_hunk and (ln.startswith(("--- ", "+++ "))):
                        continue
                    cleaned.append(ln)
                rejoined = "\n".join(cleaned)
                normalized = f"--- a/{inferred_path}\n+++ b/{inferred_path}\n{rejoined}"

        try:
            _ado_ok, _ado_msg, _ado_reason, _ado_details = self._diff_apply(
                self.repo_root,
                normalized,
                file_path_hint=target_file,
            )
            if _ado_ok:
                return True, None
        except Exception as e:
            return False, f"diff_apply exception: {e}"
        else:
            # Tolerant fallback: the strict path rejects zero-context and
            # trailing-'+' hunks that `git apply --unidiff-zero` accepts (and
            # content-verifies via its own offset search), so a well-formed
            # patch that the strict path refuses still applies here.
            tol_ok, _tol_err, _tol_mode = self._tolerant_git_apply(normalized, target_file)
            if tol_ok:
                return True, None
            return False, _ado_msg or _ado_reason or "git apply failed"

    # ── Hunk header count fix ─────────────────────────────────────────────────

    def _fix_hunk_counts(self, patch_text: str) -> str:
        """Recompute @@ hunk header line counts from actual hunk body.

        Small models often produce wrong counts in @@ -a,b +c,d @@ lines
        (e.g., @@ -6,3 +6,3 @@ when the body only has 2 lines each side).
        Git treats this as a "corrupt patch" and refuses to apply even with --recount.

        This method counts the actual context/remove/add lines and rewrites the header.
        """
        lines = patch_text.splitlines(keepends=True)
        output = []
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _RE_HUNK_HEADER_RECOUNT.match(line)
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

            # Count lines. A completely empty body line is a blank CONTEXT line
            # whose leading space was stripped (LLMs and some editors do this) —
            # git apply counts it as context, so we must too, or the rewritten
            # header undercounts and git silently truncates the hunk tail
            # (applying a half-hunk "successfully" = silent corruption).
            # Normalize it back to " " so the emitted patch is well-formed.
            old_count = 0
            new_count = 0
            normalized_body = []
            for hl in hunk_body:
                s = hl.rstrip("\r\n")
                if s == "":
                    old_count += 1
                    new_count += 1
                    normalized_body.append(" " + hl[len(s) :])
                    continue
                if s.startswith(" "):
                    old_count += 1
                    new_count += 1
                elif s.startswith("-"):
                    old_count += 1
                elif s.startswith("+"):
                    new_count += 1
                # lines starting with \ (no newline at end) are skipped
                normalized_body.append(hl)
            hunk_body = normalized_body

            # Rebuild header with correct counts
            new_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}"
            output.append(new_header)
            output.extend(hunk_body)

        return "".join(output)

    @staticmethod
    def _add_diff_headers(patch_text: str, target_file: str | None) -> str:
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
            header = f"diff --git a/{fp} b/{fp}\n--- a/{fp}\n+++ b/{fp}\n"
            return header + text + "\n"

        # Has --- a/ but missing diff --git
        if "--- a/" in text and "diff --git" not in text:
            fp = target_file.lstrip("/")
            return f"diff --git a/{fp} b/{fp}\n" + text + "\n"

        return patch_text

    @staticmethod
    def _patch_is_noop(patch_text: str) -> bool:
        """True when every hunk in the patch is context-only (no ``-``/``+``
        body lines) — applying it would change nothing."""
        in_hunk = False
        saw_hunk = False
        for line in (patch_text or "").splitlines():
            if line.startswith("@@"):
                in_hunk = True
                saw_hunk = True
                continue
            if not in_hunk:
                continue
            if line[:1] in ("-", "+") and not line.startswith(("--- ", "+++ ")):
                return False
        return saw_hunk

    # ── Tolerant apply: try multiple git apply flag combinations ─────────────

    def _tolerant_git_apply(
        self, patch_text: str, target_file: str | None = None, allow_3way: bool = True
    ) -> tuple[bool, str | None, str]:
        """Try multiple git apply flag combinations for tolerant (small model) mode.

        Returns (success, error_message, mode_used).
        Pipeline:
          0. Fix hunk counts + add missing headers (preprocessing)
          1. Preprocessed patch + --ignore-whitespace
          2. Preprocessed patch + --ignore-space-change
          3. Preprocessed patch (no extra flags, just correct counts)
          4. --3way (creates merge markers if needed — last resort)

        The ``-C0`` variants apply hunks purely by (recounted) line numbers —
        git checks ZERO context lines, so a stale header silently lands the
        hunk at the wrong location with rc=0 (real case 2026-07-18: an insert
        intended after a Kotlin function's closing brace landed 17 lines
        below, outside the enclosing ``object``). Every ``-C0`` success is
        therefore content-verified via :meth:`_verify_c0_placement` and rolled
        back on mismatch, letting the caller fall through to re-anchoring.

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
        # --recount everywhere: never trust hunk-header counts (ours included) —
        # an undercounted header makes git consume only part of the hunk body and
        # apply the truncated prefix with rc=0 (silent corruption; see
        # test_patch_tolerant.py::TestFixHunkCountsBlankContext).
        patches_to_try = [
            (fixed, ["--recount", "--ignore-whitespace"], "fixed_ignore_ws"),
            # --unidiff-zero accepts zero-context and trailing-'+' hunks that
            # plain apply rejects mid-file; git still content-verifies the
            # removed lines via its own offset search (unlike -C0, which is
            # blind), so a misplaced hunk cannot land silently.
            (fixed, ["--recount", "--unidiff-zero"], "fixed_unidiff_zero"),
            # NOTE: no -C0 + whitespace-flag combination here — Apple git
            # 2.39.5 SIGBUSes (rc=-10) when a -C0 hunk's context is a
            # whitespace NEAR-MISS of the file content (e.g. a context line
            # missing its indentation) under --ignore-whitespace /
            # --ignore-space-change. -C0 alone fails cleanly instead.
            (fixed, ["--recount", "-C0"], "fixed_C0"),  # pure line-number matching
            (fixed, ["--recount", "--ignore-space-change"], "fixed_ignore_sc"),
            (fixed, ["--recount"], "fixed_plain"),
        ]
        if allow_3way:
            patches_to_try.append(
                (patch_text, ["--3way"], "3way_merge"),  # fallback to raw + 3way
            )
        _c0_verify_fail: str | None = None
        for try_patch, flags, mode_name in patches_to_try:
            _is_c0 = "-C0" in flags or "--unidiff-zero" in flags
            # A -C0 sibling already applied at these line numbers and failed
            # placement verification — another -C0 variant lands identically.
            if _is_c0 and _c0_verify_fail is not None:
                continue
            encoded = try_patch.encode("utf-8")
            try:
                # PERF: no separate `git apply --check` pre-spawn — the apply
                # itself is the decider. git apply is atomic on failure: a
                # non-3way variant exits rc!=0 with the tree untouched
                # (measured: context mismatch -> rc=1, tree byte-identical),
                # so a pre-check was pure diagnostic and cost one extra
                # subprocess per variant (up to 12 spawns per patch in the
                # drift case). The apply below either succeeds, or fails
                # cleanly and the next variant is tried.
                #
                # Snapshot ONLY for the mutating-on-failure variants:
                #   • -C0 / --unidiff-zero: a WRONG-LINE apply returns rc=0 —
                #     the placement check below catches it and needs the
                #     pre-image to roll back.
                #   • --3way: a CONFLICT writes merge markers + stages the
                #     unmerged result and exits rc!=0; rollback needs both
                #     content and index snapshots.
                # Plain variants (--recount ± whitespace flags) are atomic on
                # failure (measured: rc!=0, tree byte-identical), so snapshot
                # for them would be pure waste (2 subprocesses per variant).
                _pre_snapshot: dict[str, str | None] | None = None
                _pre_index: dict[str, tuple[str, str]] | None = None
                if _is_c0 or "--3way" in flags:
                    _pre_snapshot = self._snapshot_patch_targets(try_patch)
                    _pre_index = self._snapshot_index_entries(try_patch)
                # Actually apply
                apply_r = subprocess.run(
                    ["git", "apply", *flags, "-"],
                    cwd=self.repo_root,
                    input=encoded,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if apply_r.returncode == 0:
                    if _is_c0:
                        _place_ok, _place_detail = self._verify_c0_placement(try_patch)
                        if not _place_ok:
                            self._restore_patch_targets(_pre_snapshot)
                            _c0_verify_fail = _place_detail
                            logger.warning(
                                "tolerant_git_apply: mode=%s applied but placement "
                                "verification failed — rolled back: %s",
                                mode_name,
                                _place_detail,
                            )
                            continue
                    logger.info("tolerant_git_apply succeeded mode=%s flags=%s", mode_name, flags)
                    return True, None, mode_name
                err = apply_r.stderr.decode("utf-8", errors="ignore").strip()
                # Non-3way failures leave the tree untouched (measured) — no
                # rollback needed. Only the --3way failure path (CONFLICT)
                # mutates (markers + staging) and must be undone here:
                # "failed" must mean the tree is untouched, otherwise every
                # downstream repair step reads a file this attempt corrupted.
                # Content first, then the index (restoring bytes does NOT
                # clear --3way's unmerged stages — measured).
                if "--3way" in flags:
                    self._restore_patch_targets(_pre_snapshot)
                    self._restore_index_entries(_pre_index)
                    logger.warning(
                        "tolerant_git_apply: --3way apply failed mode=%s — rolled back: error=%s",
                        mode_name,
                        err,
                    )
                    return False, f"3way apply failed — rolled back: {err}", mode_name
                logger.debug(
                    "tolerant_git_apply: mode=%s failed (tree untouched) — next variant: error=%s",
                    mode_name,
                    err,
                )
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
        if _c0_verify_fail is not None:
            return (
                False,
                (
                    "All tolerant git apply variants failed (line-number-only apply "
                    f"was rolled back — hunk context does not match the file at the "
                    f"patch's line numbers, likely stale: {_c0_verify_fail})"
                ),
                "none",
            )
        return False, "All tolerant git apply variants failed", "none"

    @staticmethod
    def _ws_norm_line(line: str) -> str:
        """Whitespace-insensitive normal form of a source line (≥ git's --ignore-whitespace)."""
        return "".join(line.split())

    def _snapshot_patch_targets(self, patch_text: str) -> dict[str, str | None]:
        """Snapshot content of every file a patch touches. None = did not exist."""
        from .agent._shared_utils import extract_files_from_patch

        snap: dict[str, str | None] = {}
        for rel in extract_files_from_patch(patch_text):
            abs_p = os.path.join(self.repo_root, rel) if self.repo_root else rel
            try:
                with open(abs_p, encoding="utf-8", errors="replace") as fh:
                    snap[abs_p] = fh.read()
            except OSError:
                snap[abs_p] = None
        return snap

    def _read_index_entries(self, rels: list[str]) -> dict[str, tuple[str, str]]:
        """``{rel_path: (mode, sha)}`` for the STAGE-0 index entry of each path.

        Paths with no entry (untracked) and paths already unmerged (stages 1-3,
        e.g. a merge the user is in the middle of) are simply absent from the
        result — the caller treats "absent" as "nothing of ours to restore".
        """
        out: dict[str, tuple[str, str]] = {}
        if not rels:
            return out
        try:
            r = subprocess.run(
                ["git", "ls-files", "-s", "-z", "--", *rels],
                cwd=self.repo_root,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("index snapshot failed: %s", exc)
            return out
        if r.returncode != 0:
            return out
        # -z, not the default line output: git C-quotes non-ASCII paths ("\355\225...")
        # in line mode, so a Korean filename would never match the key we look
        # it up by. NUL-terminated records are emitted raw.
        for record in r.stdout.decode("utf-8", errors="replace").split("\0"):
            if not record:
                continue
            meta, _, path = record.partition("\t")
            parts = meta.split()
            if len(parts) != 3 or not path:
                continue
            mode, sha, stage = parts
            if stage == "0":
                out[path] = (mode, sha)
        return out

    def _snapshot_index_entries(self, patch_text: str) -> dict[str, tuple[str, str]]:
        """Snapshot the stage-0 index entry of every file *patch_text* touches.

        ``git apply --3way`` stages what it produces, so a conflicted 3-way
        leaves three unmerged stages behind. Restoring file CONTENT does not
        clear them (measured: ``UU`` survives a byte-identical rewrite), which
        is why the index needs its own snapshot.
        """
        from .agent._shared_utils import extract_files_from_patch

        return self._read_index_entries(extract_files_from_patch(patch_text))

    def _restore_index_entries(self, snapshot: dict[str, tuple[str, str]]) -> None:
        """Re-point the index at the restored working-tree content, for entries
        that moved.

        Only paths captured at stage 0 are restored, and only when the current
        entry differs — a healthy index is never rewritten, so this is a no-op
        for every mode except a conflicted ``--3way``. ``git add`` both clears
        the unmerged stages and reinstates stage 0 in one step. (The previous
        ``git update-index --cacheinfo mode,sha,path`` form is rejected by
        Apple git 2.39.5 with "option `cacheinfo' takes no value".)

        All changed paths are re-added in ONE ``git add -- <paths>``
        subprocess instead of one ``git add <path>`` per path — same
        observable effect (each changed path staged, healthy index untouched),
        N spawns → 1. The per-path error log is preserved via the one
        combined stderr.
        """
        if not snapshot:
            return
        current = self._read_index_entries(list(snapshot))
        changed = [path for path, entry in snapshot.items() if current.get(path) != entry]
        if not changed:
            return
        try:
            r = subprocess.run(
                ["git", "add", "--", *changed],
                cwd=self.repo_root,
                capture_output=True,
                timeout=10,
                check=False,
            )
            if r.returncode != 0:
                logger.error(
                    "tolerant_git_apply: index rollback failed for %s: %s",
                    changed,
                    r.stderr.decode("utf-8", errors="ignore").strip(),
                )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.exception("tolerant_git_apply: index rollback of %s failed: %s", changed, exc)

    def _restore_patch_targets(self, snapshot: dict[str, str | None]) -> None:
        """Undo an applied patch from a _snapshot_patch_targets snapshot."""
        for path, content in snapshot.items():
            try:
                if content is None:
                    if os.path.isfile(path):
                        os.remove(path)
                else:
                    # Atomic like every other repair/rollback write: a SIGKILL
                    # mid-rollback must not leave the target truncated.
                    # realpath keeps symlink semantics — open(path, "w") wrote
                    # through the link, and a non-resolved os.replace would
                    # replace the link itself.
                    atomic_write_text(os.path.realpath(path), content)
            except OSError as exc:
                logger.exception("tolerant_git_apply: rollback of %s failed: %s", path, exc)

    @staticmethod
    def context_free_hunks(patch_text: str) -> list[str]:
        """``"file @@hdr"`` for every hunk carrying NO context line.

        Such a hunk is placed purely by its line number, and nothing downstream
        can check that number. ``_verify_c0_placement`` says so itself — it
        skips them because there is no context to match the file against — but
        the exposure is wider than that method's name suggests: a hunk with no
        context has nothing to match on ANY strategy, so it is placed blind by
        the very first ``git apply`` in the ladder, never reaching the ``-C0``
        path the verifier guards.

        Nothing else catches it either. A stale line number quietly relocates
        the insert, and if the result is still parseable the post-write syntax
        gate passes it: measured, ``@@ -8,0 +9,2 @@`` landed two statements
        after a ``return``, producing unreachable code that compiled fine and
        reported ok=True. So this is reported rather than refused — a
        context-free hunk is legitimate output for ``diff -U0``, and refusing
        would break working callers to catch a malformed minority. The agent
        gets told which hunks were unverifiable and can re-read the file.

        New files are excluded: there is no prior content to misplace against.
        """
        out: list[str] = []
        cur: str | None = None
        cur_is_new = False
        hdr: str = ""
        body: list[str] | None = None

        def _flush() -> None:
            if body is not None and cur and not cur_is_new and not any(ln[:1] == " " for ln in body):
                out.append(f"{cur} {hdr}".strip())

        for line in patch_text.splitlines():
            if line.startswith(("diff --git ", "--- ")):
                _flush()
                body = None
                cur_is_new = line.startswith("--- /dev/null")
                continue
            if line.startswith("+++ "):
                _flush()
                body = None
                p = line[4:].strip()
                if p.startswith("b/"):
                    p = p[2:]
                cur = None if p == "/dev/null" else p
                continue
            if line.startswith("@@"):
                _flush()
                hdr = line.split("@@")[1].join(["@@", "@@"]) if "@@" in line[2:] else line
                body = [] if cur is not None else None
                continue
            if body is None:
                continue
            if line[:1] in (" ", "+", "-"):
                body.append(line)
            elif line == "":
                body.append(" ")
            elif line.startswith("\\"):
                continue
            else:
                _flush()
                body = None
        _flush()
        return out

    def _verify_c0_placement(self, patch_text: str) -> tuple[bool, str]:
        """Content-verify hunk placement after a context-free git apply
        (``-C0`` / ``--unidiff-zero`` blind line-number apply).

        A hunk passes when EITHER:
          * its pre-image (context + removed lines) matches at its CLAIMED
            line position — the case for a not-yet-applied correct patch
            (direct unit-test invocation) and for a blind apply that landed
            exactly where the hunk's old lines live; or
          * its post-image (context + added lines) appears as a consecutive
            block somewhere in the file after apply — the case for a blind
            apply that mutated the tree: at the correct location this holds
            by construction; at a stale-line-number location the surrounding
            lines differ and the block is absent.

        Hunks with no context lines (pure line-number inserts) have nothing to
        verify against and are accepted as-is, as are new files.

        Returns (ok, detail) — detail names the offending file on failure.
        """
        files: dict[str, list[list]] = {}  # rel -> [[hunk_lines, old_start], ...]
        new_files: set[str] = set()
        cur: str | None = None
        cur_is_new = False
        cur_hunk: list | None = None

        for line in patch_text.splitlines():
            if line.startswith(("diff --git ", "--- ")):
                cur_hunk = None
                cur_is_new = line.startswith("--- /dev/null")
                continue
            if line.startswith("+++ "):
                cur_hunk = None
                p = line[4:].strip()
                if p.startswith("b/"):
                    p = p[2:]
                cur = None if p == "/dev/null" else p
                if cur is not None:
                    files.setdefault(cur, [])
                    if cur_is_new:
                        new_files.add(cur)
                continue
            if line.startswith("@@"):
                if cur is None:
                    cur_hunk = None
                else:
                    m = _RE_HUNK_HEADER_FIX.match(line)
                    old_start = int(m.group(2)) if m else None
                    cur_hunk = []
                    files[cur].append([cur_hunk, old_start])
                continue
            if cur_hunk is None:
                continue
            if line[:1] in (" ", "+", "-"):
                cur_hunk.append(line)
            elif line == "":
                # Trailing-whitespace-stripped empty context line.
                cur_hunk.append(" ")
            elif line.startswith("\\"):
                continue  # "\ No newline at end of file"
            else:
                cur_hunk = None  # hunk body ended (junk / next section)

        for rel, hunk_list in files.items():
            if rel in new_files:
                continue
            abs_p = os.path.join(self.repo_root, rel) if self.repo_root else rel
            try:
                with open(abs_p, encoding="utf-8", errors="replace") as fh:
                    file_norm = [self._ws_norm_line(ln) for ln in fh.read().splitlines()]
            except OSError:
                logger.debug("_verify_c0_placement: cannot read %s (deleted by patch?)", abs_p)
                continue  # deleted by the patch — nothing to place-check
            for hunk, old_start in hunk_list:
                if not any(ln[:1] == " " for ln in hunk):
                    continue  # context-free hunk: unverifiable by content
                # Pre-image at the claimed position — but only for hunks that
                # REMOVE lines: a pre-image match then proves the change's
                # target is exactly where the hunk claims. For insert-only
                # hunks the pre-image is just context, whose presence at the
                # position says nothing about where the insertion landed, so
                # they must satisfy the post-image check below.
                pre = [self._ws_norm_line(ln[1:]) for ln in hunk if ln[:1] in (" ", "-")]
                has_removals = any(ln[:1] == "-" for ln in hunk)
                if pre and has_removals and old_start is not None:
                    start = old_start - 1
                    if start >= 0 and file_norm[start : start + len(pre)] == pre:
                        continue
                post = [self._ws_norm_line(ln[1:]) for ln in hunk if ln[:1] in (" ", "+")]
                m = len(post)
                if m == 0:
                    continue
                n = len(file_norm)
                if m > n or not any(file_norm[i : i + m] == post for i in range(n - m + 1)):
                    return False, (
                        f"{rel}: hunk context+content not found as a consecutive block after context-free apply"
                    )
        return True, ""

    # ── Fuzzy context re-anchoring: fix wrong @@ line numbers ────────────────

    def _reanchor_patch_core(self, patch_text, target_file, finder, log_prefix):
        """Shared re-anchor skeleton for ``_exact_reanchor_patch``/``_reanchor_patch``.

        Parses the unified diff into hunks and delegates anchor location to
        *finder*, which receives ``(hunk_body, file_lines, old_start)`` and
        returns ``(best_offset, context_before, log_fragment)`` — the 0-indexed
        anchor position, the number of context lines preceding it (0 when the
        search key already starts at the first hunk line), and the parenthesized
        detail used in the info log — or ``None`` to keep the hunk untouched.
        Returns the re-anchored patch text, or None when nothing changed.
        """
        if not target_file:
            return None

        file_path = os.path.join(self.repo_root, target_file) if self.repo_root else target_file
        try:
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                file_lines = fh.readlines()
        except OSError:
            logger.debug("_reanchor_patch_core: cannot read %s", file_path)
            return None
        if not file_lines:
            return None

        lines = patch_text.splitlines(keepends=True)
        output = []
        i = 0
        changed = False

        # Copy header lines verbatim
        while i < len(lines) and not lines[i].startswith("@@"):
            output.append(lines[i])
            i += 1

        while i < len(lines):
            line = lines[i]
            m = _RE_HUNK_HEADER_FIX.match(line)
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

            hit = finder(hunk_body, file_lines, old_start)
            if hit is None:
                output.append(line)
                output.extend(hunk_body)
                continue

            best_offset, context_before, log_fragment = hit
            new_start = best_offset - context_before + 1  # 1-indexed
            delta = new_start - old_start
            new_new_start = int(m.group(4)) + delta
            new_header = f"@@ -{new_start},{old_count} +{new_new_start},{new_count} @@{suffix}"
            logger.info(
                "%s: hunk @@ -%d → -%d %s file=%s",
                log_prefix,
                old_start,
                new_start,
                log_fragment,
                target_file,
            )
            output.append(new_header)
            output.extend(hunk_body)
            changed = True

        if not changed:
            return None
        return "".join(output)

    def _exact_reanchor_patch(self, patch_text: str, target_file: str | None) -> str | None:
        """Re-anchor a unified diff by finding exact removed-line content in the file.

        Faster and more reliable than SequenceMatcher for small line offsets.
        Searches for the first `-` line's exact content in the actual file,
        then rewrites @@ headers if the offset is within ±50 lines.
        """

        def _find_exact(hunk_body, file_lines, old_start):
            # Extract removed lines (stripped of diff prefix)
            removed_lines = [hl[1:].rstrip("\n\r") for hl in hunk_body if hl.startswith("-")]

            if not removed_lines:
                return None

            # Search for the first removed line in the actual file
            search_text = removed_lines[0].strip()
            if not search_text or len(search_text) < 5:
                return None

            # Find all matching positions
            matches = []
            for idx, fl in enumerate(file_lines):
                if fl.strip() == search_text:
                    matches.append(idx)

            if not matches:
                return None

            # Pick the match closest to the original position (within ±50 lines)
            original_pos = old_start - 1  # 0-indexed
            best_match = min(matches, key=lambda x: abs(x - original_pos))
            offset_diff = abs(best_match - original_pos)

            if offset_diff == 0 or offset_diff > 50:
                return None

            # Verify: check if all removed lines match at this position
            for j, rl in enumerate(removed_lines):
                _file_idx = best_match + j
                if _file_idx >= len(file_lines) or file_lines[_file_idx].strip() != rl.strip():
                    return None

            # The first context line starts at (best_match - context_before_count + 1)
            context_before_count = 0
            for hl in hunk_body:
                if hl.startswith(" "):
                    context_before_count += 1
                elif hl.startswith(("-", "+")):
                    break

            new_start = best_match - context_before_count + 1  # 1-indexed
            delta = new_start - old_start
            return best_match, context_before_count, f"(offset={delta:+d}, match='{search_text[:40]}')"

        return self._reanchor_patch_core(patch_text, target_file, _find_exact, "exact_reanchor")

    def _reanchor_patch(self, patch_text: str, target_file: str | None) -> str | None:
        """Re-anchor a unified diff patch to the correct line numbers.

        Small models often generate patches with wrong @@ line numbers.
        This method:
        1. Parses each hunk's context + removed lines
        2. Searches the actual file for the best SequenceMatcher match
        3. Rewrites @@ headers with the correct line numbers
        4. Returns the repaired patch text, or None if re-anchoring fails.
        """

        def _find_fuzzy(hunk_body, file_lines, old_start):
            # Search key: context + removed lines (the block that must land at
            # the right place).  context_before = leading context lines, so the
            # returned anchor is the 0-indexed position of the FIRST REMOVED
            # line of the matched block — re-anchoring must point the hunk at
            # its change, not at the context that precedes it.
            search_lines = [hl[1:] for hl in hunk_body if hl.startswith((" ", "-"))]  # strip prefix

            if not search_lines:
                return None

            context_before = 0
            for hl in hunk_body:
                if hl.startswith(" "):
                    context_before += 1
                elif hl.startswith(("-", "+")):
                    break

            # Build a normalized search string
            search_str = "".join(search_lines).strip()
            if not search_str:
                return None

            # Sliding window search with SequenceMatcher
            # Optimization (P5): reuse SequenceMatcher with set_seqs(),
            # limit search window to ±200 lines, skip files > 2000 lines
            window = len(search_lines)
            if window <= 0:
                return None

            # Skip fuzzy matching for very large files — too expensive
            file_len = len(file_lines)
            if file_len > 2000:
                logger.debug(
                    "reanchor_patch: skipping fuzzy match for large file (%d lines, target=%s)",
                    file_len,
                    target_file,
                )
                return None

            best_score = 0.0
            best_offset = old_start - 1  # default: use original

            # Restrict search to ±200 lines around original position
            search_start = max(0, old_start - 1 - 200)
            search_end = min(file_len - window + 1, old_start - 1 + 200)
            if search_start >= search_end:
                search_start = 0
                search_end = file_len - window + 1

            # P24-A: autojunk=False — search_str/chunk_str are joined
            # CHARACTER sequences; autojunk would purge '\n', space and other
            # chars appearing in >1% of a >=200-char chunk, collapsing
            # ratio() and silently failing the re-anchor below.
            matcher = difflib.SequenceMatcher(None, autojunk=False)
            for start_idx in range(search_start, search_end):
                chunk = file_lines[start_idx : start_idx + window]
                chunk_str = "".join(chunk).strip()
                matcher.set_seqs(search_str, chunk_str)
                ratio = matcher.ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_offset = start_idx
                    if best_score > 0.95:
                        break  # near-perfect match — no need to scan further

            # Only re-anchor if we found a significantly better match than the
            # original position; a good score AT the claimed position means the
            # header is already right — keep it (None). The claimed start is
            # "already right" when it points at the matched block's first line
            # OR at its first removed line (both are common header conventions).
            original_pos = old_start - 1  # 0-indexed
            anchor = best_offset + context_before  # 0-indexed first removed line
            if best_score >= 0.7 and original_pos not in (anchor, best_offset):
                return anchor, 0, f"(score={best_score:.2f})"
            return None

        return self._reanchor_patch_core(patch_text, target_file, _find_fuzzy, "reanchor_patch")

    def _add_step(self, metadata: dict[str, Any], step: str, description: str):
        """Add execution step to metadata."""
        metadata["execution_steps"].append(
            {"step": step, "description": description, "timestamp": self._current_timestamp()}
        )

    def _current_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime

        return datetime.now().isoformat()
