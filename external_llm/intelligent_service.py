"""
Intelligent External LLM Service for asicode

Enhances ExternalLLMService with smart features:
1. Automatic request analysis
2. Project structure understanding
3. Multi-file planning
4. Intelligent routing

Handles general requests like "create login functionality" by:
- Analyzing the request
- Understanding project structure
- Creating an execution plan
- Generating appropriate code for each file
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from .agent.config.thresholds import config as _cfg
from .client import DEFAULT_LLM_TIMEOUT, OLLAMA_LLM_TIMEOUT
from .languages import LanguageId
from .multi_planner import ExecutionPlan, FileOperation, LLMEnhancedMultiFilePlanner, MultiFilePlanner
from .output_modes import OutputMode
from .project_analyzer import ProjectAnalyzer, ProjectStructure
from .service import ExternalLLMService
from .smart_analyzer import RequestAnalysis, SmartRequestAnalyzer

logger = logging.getLogger(__name__)


class IntelligentLLMService:
    """
    Intelligent wrapper around ExternalLLMService

    Automatically handles:
    - Request analysis ("create login" -> structured plan)
    - Project understanding (Django, FastAPI, etc.)
    - Multi-file operations
    - Context enrichment

    Usage:
        service = IntelligentLLMService("deepseek", api_key)
        result = service.handle_request(
            repo_root="/path/to/project",
            user_request="create login functionality"
        )
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_LLM_TIMEOUT,
    ):
        # Special handling for Ollama: use longer default timeout for model loading.
        # Only override when the caller used the cloud default (not an explicit value).
        provider_stripped = (provider or "").strip()
        if provider_stripped.lower() == "ollama" and timeout == DEFAULT_LLM_TIMEOUT:
            timeout = OLLAMA_LLM_TIMEOUT
            logger.info(f"Using extended timeout for Ollama: {timeout}s")

        # Core LLM service
        self.llm_service = ExternalLLMService(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

        self.provider = provider
        self.model = model or self.llm_service.model

        logger.info(
            f"Initialized IntelligentLLMService: provider={provider}, model={self.model}"
        )

    def _emit_progress(
        self,
        progress_callback: Optional[Callable[[str, str, Optional[int], Optional[int]], None]],
        phase: str,
        message: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        """Safely emit progress events without breaking the run."""
        if not progress_callback:
            return
        try:
            progress_callback(phase, message, current, total)
        except Exception:
            # Never fail the run because UI/log streaming failed.
            return

    def handle_request(
        self,
        repo_root: str,
        user_request: str,
        target_file: Optional[str] = None,
        mode: str = "auto",  # "auto", "single", "multi", "llm_plan", "agent"
        temperature: float = 0.0,
        context_variant: str = "v7",
        max_tokens: int = _cfg.tokens.INTELLIGENT_SERVICE_DEFAULT,
        progress_callback: Optional[Callable[[str, str, Optional[int], Optional[int]], None]] = None,
        # Agent-mode knobs (best-effort; does not affect non-agent modes)
        agent_max_attempts: int = 3,
        agent_run_tests: bool = True,
    ) -> dict[str, Any]:
        """
        Handle a user request intelligently

        Args:
            repo_root: Repository root path
            user_request: User's natural language request
            target_file: Specific target file (optional)
            mode: "auto" (智能判断), "single" (单文件), "multi" (多文件计划), "llm_plan" (LLM-based plan)
            temperature: LLM temperature
            context_variant: Context variant for LLM ("v7", "super", "hybrid")
            max_tokens: Maximum tokens for LLM response (default 4096)

        Returns:
            {
                "success": bool,
                "mode": str,  # "single_file" or "multi_file"
                "patch": str,  # unified diff (for single file)
                "plan": dict,  # execution plan (for multi file)
                "operations": list,  # file operations performed
                "analysis": dict,  # request analysis
                "explanation": str,
            }
        """
        repo_path = Path(repo_root).resolve()

        try:
            # Step 1: Analyze request
            self._emit_progress(progress_callback, "analyzing_request", "Analyzing user request...", 1, 5)
            analyzer = SmartRequestAnalyzer(str(repo_path))
            analysis = analyzer.analyze(user_request)

            logger.info(
                f"Request analysis: intent={analysis.intent}, "
                f"feature={analysis.feature_name}, "
                f"confidence={analysis.confidence:.2f}, "
                f"needs_planning={analysis.needs_planning}"
            )

            # Step 2: Analyze project
            self._emit_progress(progress_callback, "analyzing_project", "Analyzing project structure...", 2, 5)
            project_analyzer = ProjectAnalyzer(str(repo_path))
            project_structure = project_analyzer.analyze()

            frameworks_str = ', '.join(project_structure.frameworks) if project_structure.frameworks else str(project_structure.framework)
            logger.info(f"Project analysis: frameworks=[{frameworks_str}], types={project_structure.project_types}")

            # Step 3: Determine execution mode
            # User wants LLM-driven, hierarchical, step-by-step implementation
            # Always use LLM-enhanced planning for complex features
            llm_planning = True  # Default to LLM-driven planning

            if mode == "auto":
                # Always use multi_file with LLM planning for complex features
                if analysis.needs_planning:
                    exec_mode = "multi_file"
                    llm_planning = True
                else:
                    # Simple requests can be single file
                    exec_mode = "single_file"
                    llm_planning = False

            elif mode == "single":
                exec_mode = "single_file"
                llm_planning = False
            elif mode == "llm_plan":
                exec_mode = "multi_file"
                llm_planning = True
            elif mode == "agent":
                exec_mode = "agent"
                llm_planning = True
            else:
                exec_mode = "multi_file"
                llm_planning = True  # Use LLM planning by default

            logger.info(f"Execution mode: {exec_mode}, LLM planning: {llm_planning}")
            logger.info(f"Analysis: intent={analysis.intent}, needs_planning={analysis.needs_planning}, "
                       f"confidence={analysis.confidence:.2f}")

            # Step 4: Execute based on mode
            self._emit_progress(progress_callback, "executing", f"Executing {exec_mode} operation...", 3, 5)

            if exec_mode == "single_file":
                return self._handle_single_file(
                    repo_path,
                    user_request,
                    analysis,
                    project_structure,
                    target_file,
                    temperature,
                    context_variant,
                    max_tokens,
                    progress_callback=progress_callback,
                )
            elif exec_mode == "agent":
                return self._handle_agent_mode(
                    repo_path,
                    user_request,
                    analysis,
                    project_structure,
                    temperature,
                    context_variant,
                    max_tokens,
                    llm_planning=llm_planning,
                    progress_callback=progress_callback,
                    max_attempts=agent_max_attempts,
                    run_tests=agent_run_tests,
                )
            else:
                # multi_file mode (includes llm_planning flag)
                return self._handle_multi_file(
                    repo_path=repo_path,
                    user_request=user_request,
                    analysis=analysis,
                    project_structure=project_structure,
                    temperature=temperature,
                    context_variant=context_variant,
                    max_tokens=max_tokens,
                    llm_planning=llm_planning,
                    progress_callback=progress_callback,
                )

        except Exception as e:
            logger.exception("Error in intelligent request handling")
            return {
                "success": False,
                "mode": "error",
                "error": str(e),
                "explanation": f"Failed to process request: {e}",
            }

    # ============================================================
    # Agent mode: LLM tool-use loop via AgentLoop
    # ============================================================
    def _handle_agent_mode(
        self,
        repo_path: Path,
        user_request: str,
        analysis: RequestAnalysis,
        project_structure: ProjectStructure,
        temperature: float,
        context_variant: str = "v7",
        max_tokens: int = _cfg.tokens.INTELLIGENT_SERVICE_DEFAULT,
        llm_planning: bool = True,
        progress_callback: Optional[Callable[[str, str, Optional[int], Optional[int]], None]] = None,
        max_attempts: int = 3,
        run_tests: bool = True,
    ) -> dict[str, Any]:
        """
        Agent mode: delegates to AgentLoop for autonomous tool-use execution.

        The LLM autonomously calls tools (bash, write_plan,
        apply_patch, run_tests, run_lint, etc.) to accomplish the task.
        The existing stability infrastructure (git apply --check, plan_compiler,
        TestRunner, path security) is preserved inside ToolRegistry.
        """
        from .agent.agent_loop import AgentLoop
        from .agent.tool_registry import AgentConfig, ToolRegistry

        def _stream_cb(event: str, data: dict) -> None:
            if not progress_callback:
                return
            try:
                turn = data.get("turn", "?")
                tool = data.get("tool", "?")
                result_ok = (data.get("result") or {}).get("ok", True)
                status = "ok" if result_ok else "error"
                progress_callback(
                    "agent_tool_call",
                    f"[Agent turn {turn}] {tool} → {status}",
                    data.get("turn"),
                    self._agent_max_turns,
                )
            except Exception:
                pass

        self._agent_max_turns = max(1, int(max_attempts or 1)) * 10  # turns ≈ 10x legacy attempts

        config = AgentConfig(
            max_turns=self._agent_max_turns,
            max_apply_attempts=max(1, int(max_attempts or 1)),
            run_tests=run_tests,
            run_lint=True,
            context_variant=context_variant,
            stream_callback=_stream_cb,
        )

        registry = ToolRegistry(str(repo_path), config)
        loop = AgentLoop(
            llm_client=self.llm_service.client,
            registry=registry,
            config=config,
            model=self.model,
        )

        self._emit_progress(
            progress_callback,
            "agent_start",
            "[Agent] Starting autonomous tool-use loop...",
            0,
            config.max_turns,
        )

        # Build project context for the agent system prompt
        context = self._build_agent_context(repo_path, analysis, project_structure, context_variant)

        try:
            agent_result = loop.run(user_request, context)
        except Exception as e:
            logger.exception("AgentLoop raised exception")
            return {
                "success": False,
                "mode": "agent",
                "error": f"agent_loop_error: {e}",
                "analysis": self._analysis_to_dict(analysis),
                "agent": {"status": "error", "error": str(e)},
                "explanation": f"Agent loop failed with exception: {e}",
            }

        return self._adapt_agent_result(agent_result, analysis)

    def _build_agent_context(
        self,
        repo_path: Path,
        analysis: RequestAnalysis,
        project_structure: ProjectStructure,
        context_variant: str,
    ) -> str:
        """Build a compact project context string for the agent system prompt."""
        parts = []

        # Project structure
        if project_structure.frameworks:
            parts.append(f"Frameworks: {', '.join(project_structure.frameworks)}")
        elif project_structure.framework:
            parts.append(f"Framework: {project_structure.framework}")
        if project_structure.project_types:
            parts.append(f"Project types: {', '.join(project_structure.project_types)}")
        if project_structure.entry_points:
            parts.append(f"Entry points: {', '.join(project_structure.entry_points[:5])}")
        if project_structure.test_dir:
            parts.append(f"Test directory: {project_structure.test_dir}")

        # Request analysis context
        if analysis.intent:
            parts.append(f"Request intent: {analysis.intent}")
        if analysis.feature_name:
            parts.append(f"Feature: {analysis.feature_name}")
        if analysis.suggested_files:
            parts.append(f"Suggested files: {', '.join(analysis.suggested_files[:5])}")
        if analysis.tech_stack:
            parts.append(f"Tech stack: {', '.join(analysis.tech_stack[:5])}")

        parts.append(f"Repository root: {repo_path}")

        return "\n".join(parts)

    def _adapt_agent_result(
        self,
        agent_result: Any,
        analysis: RequestAnalysis,
    ) -> dict[str, Any]:
        """Convert AgentResult to the existing IntelligentLLMService result dict format."""
        applied_patches = agent_result.applied_patches or []
        turns = agent_result.turns or []
        status = agent_result.status

        # Build combined patch from all applied patches
        combined_patch = "\n".join(applied_patches) if applied_patches else ""

        # Build turn summary
        turn_summary = [
            {
                "turn": t.turn_num,
                "tool": t.tool_name,
                "ok": t.tool_result.ok,
                "content_preview": (t.tool_result.content or "")[:200],
            }
            for t in turns
        ]

        success = status == "success" and bool(applied_patches or agent_result.final_message)

        result: dict[str, Any] = {
            "success": success,
            "mode": "agent",
            "patch": combined_patch,
            "explanation": agent_result.final_message or f"Agent finished with status: {status}",
            "analysis": self._analysis_to_dict(analysis),
            "agent": {
                "status": status,
                "turns_used": len(turns),
                "max_turns": self._agent_max_turns,
                "applied_patches": len(applied_patches),
                "turn_summary": turn_summary,
                **(agent_result.metadata or {}),
            },
        }

        if status == "max_turns":
            result["success"] = bool(applied_patches)
            result["error"] = "agent_max_turns"
            if not applied_patches:
                result["explanation"] = f"Agent reached max turns ({self._agent_max_turns}) without applying any changes."

        elif status == "error":
            result["success"] = False
            result["error"] = agent_result.error or "agent_error"

        return result

    def _determine_output_mode(
        self,
        repo_path: Path,
        target_file: str,
        operation: str = "modify",
        file_exists: Optional[bool] = None,
        change_size_hint: Optional[str] = None,  # 'small', 'medium', 'large', 'rewrite'
    ) -> tuple[OutputMode, str]:
        """
        Determine output mode and context variant based on request type.

        Args:
            repo_path: Repository root path
            target_file: Target file path
            operation: 'create' or 'modify'
            file_exists: Whether file exists (auto-detected if None)
            change_size_hint: Optional hint about change size ('small', 'medium', 'large', 'rewrite')

        Returns:
            Tuple of (output_mode, context_variant)

        Policy:
            1. New file creation → FULL_FILE mode with enhanced prompts
            2. Existing file small modifications → UNIFIED_DIFF
            3. Large changes to existing files → FULL_FILE
            4. Multi-file operations → PLAN_JSON (handled separately)
            5. Gemini provider: Prefer UNIFIED_DIFF even for large changes (reduce FULL_FILE frequency)
        """
        # Determine if file exists
        if file_exists is None:
            file_path = repo_path / target_file.lstrip('/')
            file_exists = file_path.exists()

        # New file creation: use FULL_FILE mode (no existing content for diff)
        if operation == "create" or not file_exists:
            # Ollama provider: try UNIFIED_DIFF even for new files
            # (Ollama seems better at diff generation than FILE block generation)
            if self.provider.lower() in ['ollama']:
                logger.info(f"Ollama provider - using UNIFIED_DIFF for new file {target_file}")
                return OutputMode.UNIFIED_DIFF, "v7"
            else:
                return OutputMode.FULL_FILE, "v7"

        # Existing file: apply heuristics
        file_path = repo_path / target_file.lstrip('/')

        # Heuristic 1: Change size hint
        if change_size_hint in ['large', 'rewrite']:
            # Gemini provider: prefer UNIFIED_DIFF even for large changes
            # (Gemini produces stable unified diffs, reduce FULL_FILE frequency)
            if self.provider.lower() in ['google', 'gemini']:
                logger.info(f"Gemini provider - using UNIFIED_DIFF for {target_file} despite change_size_hint={change_size_hint}")
                return OutputMode.UNIFIED_DIFF, "v7"
            else:
                logger.info(f"Using FULL_FILE mode for {target_file} due to change_size_hint={change_size_hint}")
                return OutputMode.FULL_FILE, "v7"

        # Heuristic 2: File size (lines)
        try:
            if file_path.exists():
                # Count lines in file
                with open(file_path, encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)

                # Large files (over 500 lines) might be better with FULL_FILE for major changes
                if line_count > 500:
                    # For large files, consider using FULL_FILE if change is likely significant
                    # This is a conservative heuristic - we still use UNIFIED_DIFF by default
                    # but log the info for debugging
                    logger.debug(f"Large file detected: {target_file} has {line_count} lines")

                    # If we have a hint that it's a medium or large change, use FULL_FILE
                    if change_size_hint in ['medium', 'large', 'rewrite']:
                        logger.info(f"Using FULL_FILE for large file {target_file} with {change_size_hint} change")
                        return OutputMode.FULL_FILE, "v7"
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Could not read file {target_file} for size check: {e}")

        # Default: use UNIFIED_DIFF mode for modifications
        return OutputMode.UNIFIED_DIFF, "v7"

    def _output_mode_to_string(self, mode: OutputMode) -> str:
        """Convert OutputMode enum to string for generate_patch"""
        if mode == OutputMode.UNIFIED_DIFF:
            return "diff"
        elif mode == OutputMode.FULL_FILE:
            return "full_file"  # dedicated full_file mode (FILE blocks only, no "Prefer unified diff")
        elif mode == OutputMode.ASICODE_BLOCK:
            return "auto"
        elif mode == OutputMode.TARGETED_BLOCK:
            return "auto"
        elif mode == OutputMode.PLAN_JSON:
            return "auto"
        else:
            return "diff"  # fallback

    def _handle_single_file(
        self,
        repo_path: Path,
        user_request: str,
        analysis: RequestAnalysis,
        project_structure: ProjectStructure,
        target_file: Optional[str],
        temperature: float,
        context_variant: str = "v7",
        max_tokens: int = _cfg.tokens.INTELLIGENT_SERVICE_DEFAULT,
        progress_callback: Optional[Callable[[str, str, Optional[int], Optional[int]], None]] = None,
    ) -> dict[str, Any]:
        """Handle single-file operation"""

        # Determine target file if not provided
        if not target_file:
            if analysis.suggested_files:
                target_file = analysis.suggested_files[0]
            else:
                # Cannot proceed without target file
                return {
                    "success": False,
                    "mode": "single_file",
                    "error": "No target file specified and cannot auto-detect",
                    "analysis": self._analysis_to_dict(analysis),
                    "explanation": "Please specify a target file or provide more context",
                }

        # Determine operation type and output mode
        file_path = repo_path / target_file.lstrip('/')
        file_exists = file_path.exists()
        operation = "create" if not file_exists else "modify"

        # Estimate change size for heuristic decisions
        change_size_hint = self._estimate_change_size(user_request, operation, target_file)
        if change_size_hint:
            logger.debug(f"Change size hint for {target_file}: {change_size_hint}")

        # Determine initial output mode with change size hint
        output_mode, context_variant = self._determine_output_mode(
            repo_path, target_file, operation, file_exists, change_size_hint
        )

        # Define retry sequence based on initial mode
        modes_to_try = []
        if output_mode == OutputMode.UNIFIED_DIFF:
            # Try UNIFIED_DIFF first, then FULL_FILE as fallback
            modes_to_try = [OutputMode.UNIFIED_DIFF, OutputMode.FULL_FILE]
        elif output_mode == OutputMode.FULL_FILE:
            # Already in FULL_FILE mode, just try it once then fallback
            modes_to_try = [OutputMode.FULL_FILE]
        else:
            # Other modes (ASICODE_BLOCK, TARGETED_BLOCK, PLAN_JSON)
            modes_to_try = [output_mode]

        result = None
        last_error = None
        fallback_used = False
        used_output_mode = None
        failure_reason = None
        prev_failure_reason = None
        same_failure_repeat = False
        error_feedback_included = False

        # Retry loop with error feedback and failure tracking
        for i, current_mode in enumerate(modes_to_try):
            used_output_mode = current_mode

            # Build enhanced context with current mode
            enhanced_context = self._build_enhanced_context(
                user_request,
                analysis,
                project_structure,
                target_file,
                output_mode=current_mode,
                operation=operation,
            )

            # Add error feedback for correction retry (if this is a retry after failure)
            if i > 0 and last_error:
                error_feedback = self._build_error_feedback(last_error, file_path, repo_path)
                enhanced_context += "\n\n**ERROR FEEDBACK (previous attempt failed)**:\n"
                enhanced_context += error_feedback
                error_feedback_included = True
                logger.info(f"Included error feedback for {target_file} (attempt {i+1})")

            # Convert mode to string for generate_patch
            mode_str = self._output_mode_to_string(current_mode)

            # Call LLM service
            self._emit_progress(
                progress_callback,
                "generating_patch",
                f"Generating patch for {target_file} (mode: {current_mode.value})...",
                4,
                5,
            )

            result = self.llm_service.generate_patch(
                repo_root=str(repo_path),
                user_request=enhanced_context,
                target_file=target_file,
                temperature=temperature,
                context_variant=context_variant,
                output_mode=mode_str,
                max_tokens=max_tokens,
            )

            if result.get("success"):
                logger.info(f"Success with {current_mode.value} mode")
                break  # Success, exit retry loop

            last_error = result.get("error", "")
            logger.warning(f"Failed with {current_mode.value} mode: {last_error}")

            # Extract failure reason from error
            failure_reason = self._extract_failure_reason(last_error)
            # Check if same failure reason repeated
            if prev_failure_reason and failure_reason == prev_failure_reason:
                same_failure_repeat = True
                logger.warning(f"Same failure reason repeated: {failure_reason}, skipping further retries")
                break  # Exit retry loop, will proceed to fallback
            prev_failure_reason = failure_reason

            # Check if we should retry with next mode
            if i < len(modes_to_try) - 1:
                next_mode = modes_to_try[i + 1]
                logger.info(f"Retrying with {next_mode.value} after {current_mode.value} failed")
                # Optional: add a small delay or modify temperature?
            else:
                # No more modes to try
                break

        # If all modes failed, use fallback template
        if not result or not result.get("success"):
            logger.info(f"All output modes failed for {target_file}, using fallback template")
            # Ensure result dict exists
            if not result:
                result = {}
            # Record failure reason and metadata
            result["failure_reason"] = failure_reason or self._extract_failure_reason(last_error) if last_error else "unknown"
            result["output_mode"] = used_output_mode.value if used_output_mode else output_mode.value
            result["retry_count"] = i+1 if 'i' in locals() else len(modes_to_try)
            result["error_feedback_included"] = error_feedback_included
            result["same_failure_repeat"] = same_failure_repeat
            # Create a FileOperation for the target file
            fallback_operation = FileOperation(
                file_path=target_file,
                operation="create" if not file_exists else "modify",
                description=f"Create {target_file}" if not file_exists else f"Modify {target_file}",
                instructions=f"Create {target_file} for {user_request[:100]}..." if not file_exists else f"Modify {target_file} for {user_request[:100]}...",
            )
            default_patch = self._create_default_file_patch(repo_path, fallback_operation)
            if default_patch:
                result["success"] = True
                result["patch"] = default_patch
                result["error"] = None
                result["explanation"] = f"Created default {target_file} (LLM failed)"
                result["fallback_used"] = True
                result["fallback_reason"] = last_error or "all_modes_failed"
                logger.info(f"Created default file for {target_file}")
            else:
                # Fallback also failed
                result["success"] = False
                result["error"] = f"All modes failed and fallback generation failed: {last_error}"
                result["fallback_used"] = False
                result["fallback_reason"] = "fallback_generation_failed"
        else:
            # Success - add metadata about which mode worked
            if used_output_mode:
                result["output_mode_used"] = used_output_mode.value
                result["retry_count"] = i+1 if 'i' in locals() else 1
            result["error_feedback_included"] = error_feedback_included
            result["same_failure_repeat"] = same_failure_repeat
            result["fallback_used"] = False

        # Analyze failure patterns for improvement
        self._analyze_failure_patterns(result)

        # Enhance result
        result["mode"] = "single_file"
        result["analysis"] = self._analysis_to_dict(analysis)
        result["target_file"] = target_file

        return result

    def _handle_multi_file(
        self,
        repo_path: Path,
        user_request: str,
        analysis: RequestAnalysis,
        project_structure: ProjectStructure,
        temperature: float,
        context_variant: str = "v7",
        max_tokens: int = _cfg.tokens.INTELLIGENT_SERVICE_DEFAULT,
        llm_planning: bool = False,
        progress_callback: Optional[Callable[[str, str, Optional[int], Optional[int]], None]] = None,
        force_output_mode: Optional[OutputMode] = None,
        force_context_variant: Optional[str] = None,
        force_files: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Handle multi-file operation with planning"""

        # Debug logging
        logger.info(f"_handle_multi_file: llm_planning={llm_planning}, has_client={hasattr(self.llm_service, 'client')}")
        if hasattr(self.llm_service, 'client'):
            logger.info(f"client type: {type(self.llm_service.client)}, client value: {self.llm_service.client}")

        # Create execution plan
        if force_context_variant:
            # Force context expansion for planning attempt (best-effort; planner may ignore)
            logger.info(f"Agent override: force_context_variant={force_context_variant}")

        if force_output_mode:
            logger.info(f"Agent override: force_output_mode={force_output_mode.value}")

        if llm_planning and hasattr(self.llm_service, 'client'):
            # Use LLM-enhanced planner
            planner = LLMEnhancedMultiFilePlanner(
                repo_root=str(repo_path),
                llm_client=self.llm_service.client,
                llm_model=self.llm_service.model,
                temperature=temperature,
            )
            logger.info(f"Using LLM-enhanced multi-file planner with client: {self.llm_service.client}, model: {self.llm_service.model}")
        else:
            # Use rule-based planner
            planner = MultiFilePlanner(str(repo_path))
            logger.info(f"Using rule-based multi-file planner (llm_planning={llm_planning}, has_client={hasattr(self.llm_service, 'client')})")

        self._emit_progress(progress_callback, "planning", "Creating execution plan...", 3, None)
        plan = planner.create_plan(user_request)

        # Agent targeting: after pytest failures, focus regeneration on likely-relevant files (best-effort).
        if force_files:
            try:
                wanted = set()
                for p in force_files:
                    s = (p or "").strip()
                    if not s:
                        continue
                    wanted.add(s.lstrip("/"))
                if wanted:
                    original_ops = list(plan.operations)
                    filtered_ops = []
                    for op in original_ops:
                        fp = (op.file_path or "").lstrip("/")
                        if fp in wanted:
                            filtered_ops.append(op)
                    if filtered_ops:
                        plan.operations = filtered_ops  # type: ignore[attr-defined]
                        logger.info(
                            f"Agent targeting active: filtered operations {len(original_ops)} -> {len(filtered_ops)}"
                        )
                    else:
                        logger.info("Agent targeting had no matching operations; running full plan.")
            except Exception as e:
                logger.warning(f"Agent targeting filter failed; running full plan: {e}")

        logger.info(
            f"Created plan: {len(plan.operations)} operations, "
            f"complexity={plan.complexity}"
        )
        # Log instructions for each operation
        for i, op in enumerate(plan.operations):
            logger.debug(f"Operation {i+1}: {op.file_path} - instructions length: {len(op.instructions)}, description: {op.description}")

        # Execute plan step by step
        operations_results = []
        all_success = True

        for i, operation in enumerate(plan.operations):
            if progress_callback:
                progress_callback("executing_operation",
                                  f"Processing {operation.file_path} ({operation.operation})...",
                                  i + 1,
                                  len(plan.operations))
            logger.info(
                f"Executing operation {i+1}/{len(plan.operations)}: "
                f"{operation.operation} {operation.file_path}"
            )

            # Determine file path and existence
            file_path = operation.file_path.lstrip('/')
            target_path = repo_path / file_path
            file_exists = target_path.exists()

            # Estimate change size for heuristic decisions
            change_size_hint = self._estimate_change_size(user_request, operation.operation, file_path)
            if change_size_hint:
                logger.debug(f"Change size hint for {file_path}: {change_size_hint}")

            # Determine initial output mode with change size hint
            output_mode, _ = self._determine_output_mode(
                repo_path, file_path, operation.operation, file_exists, change_size_hint
            )

            # Agent override: force output mode for all operations
            if force_output_mode is not None:
                output_mode = force_output_mode

            # Define retry sequence based on initial mode (same as _handle_single_file)
            modes_to_try = []
            if output_mode == OutputMode.UNIFIED_DIFF:
                # Try UNIFIED_DIFF first, then FULL_FILE as fallback
                modes_to_try = [OutputMode.UNIFIED_DIFF, OutputMode.FULL_FILE]
            elif output_mode == OutputMode.FULL_FILE:
                # Already in FULL_FILE mode, just try it once then fallback
                modes_to_try = [OutputMode.FULL_FILE]
            else:
                # Other modes (ASICODE_BLOCK, TARGETED_BLOCK, PLAN_JSON)
                modes_to_try = [output_mode]

            result = None
            last_error = None
            fallback_used = False
            used_output_mode = None
            failure_reason = None
            prev_failure_reason = None
            same_failure_repeat = False
            error_feedback_included = False

            # Retry loop with error feedback and failure tracking
            for retry_idx, current_mode in enumerate(modes_to_try):
                used_output_mode = current_mode

                # Build request with current mode
                operation_request = self._build_operation_request(
                    operation,
                    project_structure,
                    user_request,
                    output_mode=current_mode,
                )

                # Add error feedback for correction retry (if this is a retry after failure)
                if retry_idx > 0 and last_error:
                    error_feedback = self._build_error_feedback(last_error, target_path, repo_path)
                    operation_request += "\n\n**ERROR FEEDBACK (previous attempt failed)**:\n"
                    operation_request += error_feedback
                    error_feedback_included = True
                    logger.info(f"Included error feedback for {file_path} (attempt {retry_idx+1})")

                # Convert mode to string for generate_patch
                mode_str = self._output_mode_to_string(current_mode)

                # Ensure parent directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Call LLM for this file
                logger.debug(f"LLM request for {operation.file_path} (mode: {current_mode.value}): {operation_request[:200]}...")
                effective_cv = force_context_variant or context_variant

                result = self.llm_service.generate_patch(
                    repo_root=str(repo_path),
                    user_request=operation_request,
                    target_file=file_path,
                    temperature=temperature,
                    context_variant=effective_cv,
                    output_mode=mode_str,
                    max_tokens=max_tokens,
                )
                logger.debug(f"LLM result for {operation.file_path}: success={result.get('success')}, error={result.get('error')}, patch_len={len(result.get('patch', ''))}")

                if result.get("success"):
                    logger.info(f"Success with {current_mode.value} mode")
                    break  # Success, exit retry loop

                last_error = result.get("error", "")
                logger.warning(f"Failed with {current_mode.value} mode: {last_error}")

                # Extract failure reason from error
                failure_reason = self._extract_failure_reason(last_error)
                # Check if same failure reason repeated
                if prev_failure_reason and failure_reason == prev_failure_reason:
                    same_failure_repeat = True
                    logger.warning(f"Same failure reason repeated: {failure_reason}, skipping further retries")
                    break  # Exit retry loop, will proceed to fallback
                prev_failure_reason = failure_reason

                # Check if we should retry with next mode
                if retry_idx < len(modes_to_try) - 1:
                    next_mode = modes_to_try[retry_idx + 1]
                    logger.info(f"Retrying with {next_mode.value} after {current_mode.value} failed")
                else:
                    # No more modes to try
                    break

            # If all modes failed, use fallback template (for create operations)
            if not result or not result.get("success"):
                # Ensure result dict exists
                if not result:
                    result = {}
                # Record failure reason and metadata
                result["failure_reason"] = failure_reason or self._extract_failure_reason(last_error) if last_error else "unknown"
                result["output_mode"] = used_output_mode.value if used_output_mode else output_mode.value
                result["retry_count"] = retry_idx+1 if 'retry_idx' in locals() else len(modes_to_try)
                result["error_feedback_included"] = error_feedback_included
                result["same_failure_repeat"] = same_failure_repeat

                # Try to create a default file if it's a "create" operation
                if operation.operation == "create":
                    logger.info(f"Attempting to create default file for {operation.file_path}")
                    default_patch = self._create_default_file_patch(repo_path, operation)
                    if default_patch:
                        result["success"] = True
                        result["patch"] = default_patch
                        result["error"] = None
                        result["explanation"] = f"Created default {operation.file_path} (LLM failed)"
                        result["fallback_used"] = True
                        result["fallback_reason"] = last_error or "all_modes_failed"
                        logger.info(f"Created default file for {operation.file_path}")
                    else:
                        # Fallback also failed
                        result["success"] = False
                        result["error"] = f"All modes failed and fallback generation failed: {last_error}"
                        result["fallback_used"] = False
                        result["fallback_reason"] = "fallback_generation_failed"
                else:
                    # Modify operation failure - no fallback
                    result["success"] = False
                    result["error"] = last_error or "unknown"
                    result["fallback_used"] = False
            else:
                # Success - add metadata about which mode worked
                if used_output_mode:
                    result["output_mode_used"] = used_output_mode.value
                    result["retry_count"] = retry_idx+1 if 'retry_idx' in locals() else 1
                result["error_feedback_included"] = error_feedback_included
                result["same_failure_repeat"] = same_failure_repeat
                result["fallback_used"] = False

            # Analyze failure patterns for this individual operation
            if not result.get("success"):
                # Enhance result dict with additional info for analysis
                analysis_result = dict(result)
                analysis_result["target_file"] = operation.file_path
                analysis_result["output_mode"] = used_output_mode.value if used_output_mode else output_mode.value
                self._analyze_failure_patterns(analysis_result)

            # Add operation result to list with full metadata
            op_result = {
                "file": operation.file_path,
                "operation": operation.operation,
                "success": result.get("success", False),
                "patch": result.get("patch", ""),
                "explanation": result.get("explanation", ""),
                "error": result.get("error"),
                "output_mode_used": result.get("output_mode_used"),
                "retry_count": result.get("retry_count", 0),
                "failure_reason": result.get("failure_reason"),
                "error_feedback_included": result.get("error_feedback_included", False),
                "same_failure_repeat": result.get("same_failure_repeat", False),
                "fallback_used": result.get("fallback_used", False),
                "fallback_reason": result.get("fallback_reason"),
            }
            operations_results.append(op_result)

            if not result.get("success"):
                all_success = False
                logger.warning(f"Operation failed for {operation.file_path}: {result.get('error')}")
                # Continue with next operation (don't stop)

        # Combine all patches
        combined_patch = self._combine_patches(operations_results)

        # Analyze overall failure patterns
        failed_operations = [op for op in operations_results if not op.get("success")]
        if failed_operations:
            logger.info(f"Multi-file execution summary: {len(failed_operations)}/{len(operations_results)} operations failed")
            for op in failed_operations[:3]:  # Log first 3 failures
                logger.info(f"  Failed: {op['file']} - {op.get('error', 'unknown error')}")

        return {
            "success": all_success,
            "mode": "multi_file",
            "patch": combined_patch,
            "plan": self._plan_to_dict(plan),
            "operations": operations_results,
            "analysis": self._analysis_to_dict(analysis),
            "explanation": self._generate_multi_file_explanation(plan, operations_results),
        }


    def _build_project_context_summary(self, project_structure: ProjectStructure) -> str:
        """Build concise project context summary for LLM planning"""
        lines = []

        if project_structure.frameworks:
            lines.append(f"- **Frameworks**: {', '.join(project_structure.frameworks)}")
        elif project_structure.framework:
            lines.append(f"- **Framework**: {project_structure.framework}")

        if project_structure.project_types:
            lines.append(f"- **Project Type**: {', '.join(project_structure.project_types)}")

        if project_structure.directories:
            lines.append("- **Directory Structure**:")
            for purpose, dirs in project_structure.directories.items():
                if purpose != 'other' and dirs:
                    lines.append(f"  - {purpose}: {', '.join(dirs)}")

        if project_structure.naming_style:
            lines.append(f"- **Naming Convention**: {project_structure.naming_style}")

        if project_structure.common_imports:
            lines.append(f"- **Common Imports**: {', '.join(project_structure.common_imports[:5])}")

        if project_structure.example_files:
            lines.append("- **Example Files**:")
            for file_type, path in list(project_structure.example_files.items())[:3]:
                lines.append(f"  - {file_type}: `{path}`")

        return "\n".join(lines) if lines else "No project context available."


    def _build_enhanced_context(
        self,
        user_request: str,
        analysis: RequestAnalysis,
        project_structure: ProjectStructure,
        target_file: str,
        output_mode: OutputMode = OutputMode.UNIFIED_DIFF,
        operation: str = "modify",
    ) -> str:
        """Build enhanced context for single-file operation with mode-specific instructions"""

        parts = []

        # Original request
        parts.append(f"**User Request**: {user_request}")
        parts.append("")

        # Analysis insights
        if analysis.feature_name:
            parts.append(f"**Feature**: {analysis.feature_name}")

        if analysis.intent:
            parts.append(f"**Intent**: {analysis.intent}")

        parts.append("")

        # Project context
        if project_structure.frameworks:
            parts.append(f"**Frameworks**: {', '.join(project_structure.frameworks)}")
        elif project_structure.framework:
            parts.append(f"**Framework**: {project_structure.framework}")
        if project_structure.project_types:
            parts.append(f"**Project Type**: {', '.join(project_structure.project_types)}")

        if project_structure.naming_style:
            parts.append(f"**Naming Convention**: {project_structure.naming_style}")

        # Example files for reference
        if project_structure.example_files:
            parts.append("")
            parts.append("**Reference Examples** (follow similar patterns):")
            for file_type, path in list(project_structure.example_files.items())[:3]:
                parts.append(f"- {file_type}: `{path}`")

        parts.append("")
        parts.append("---")
        parts.append("")

        # Mode-specific instructions
        parts.append("**Instructions**:")
        parts.append(f"- Target file: `{target_file}`")
        parts.append(f"- Operation: {operation}")
        if project_structure.frameworks:
            parts.append(f"- Follow {', '.join(project_structure.frameworks)} conventions")
        else:
            parts.append(f"- Follow {project_structure.framework or 'Python'} conventions")
        parts.append("- Include proper imports and error handling")
        parts.append("- Follow existing code style")

        # Output format instructions based on mode
        if output_mode == OutputMode.FULL_FILE:
            parts.append("")
            parts.append("**Output Format (FULL_FILE Mode)**:")
            parts.append("- Output MUST be a single FILE block with complete file content")
            parts.append("- Format:")
            parts.append(f"  FILE: {target_file}")
            parts.append("  ```<language>")
            parts.append("  // Complete file content here")
            parts.append("  // Every line exactly as it should appear in the file")
            parts.append("  ```")
            parts.append("- Example for new file creation:")
            parts.append(f"  FILE: {target_file}")
            parts.append("  ```javascript")
            parts.append("  // New JavaScript file")
            parts.append("  function example() {")
            parts.append("    return 'Hello';")
            parts.append("  }")
            parts.append("  ```")
        elif output_mode == OutputMode.UNIFIED_DIFF:
            parts.append("")
            parts.append("**Output Format (UNIFIED_DIFF Mode)**:")
            parts.append("- Output MUST be a valid unified diff (git apply compatible)")
            parts.append("- MUST include file headers (--- a/... +++ b/...) AND at least one @@ hunk")
            parts.append("- Example:")
            parts.append("  ```diff")
            parts.append(f"  diff --git a/{target_file} b/{target_file}")
            parts.append("  index 1234567..89abcde 100644")
            parts.append(f"  --- a/{target_file}")
            parts.append(f"  +++ b/{target_file}")
            parts.append("  @@ -1,3 +1,4 @@")
            parts.append("   // Existing content")
            parts.append("  +// New line added")
            parts.append("   // More content")
            parts.append("  ```")
            parts.append("- **CRITICAL**: If you cannot produce a valid diff with @@ hunks, use FILE block format instead")
        else:
            # Other modes (ASICODE_BLOCK, TARGETED_BLOCK, PLAN_JSON, NEEDS_DISAMBIGUATION)
            parts.append("")
            parts.append("**Output Format**: Follow the appropriate output format for the requested mode.")

        return "\n".join(parts)

    def _build_operation_request(
        self,
        operation: FileOperation,
        project_structure: ProjectStructure,
        original_request: str,
        output_mode: OutputMode = OutputMode.UNIFIED_DIFF,
    ) -> str:
        """Build request for a specific file operation with mode-specific instructions"""

        parts = []

        # Original request for context
        parts.append(f"**Overall Goal**: {original_request}")
        parts.append("")

        # Specific task for this file
        parts.append(f"**Current Task**: {operation.description}")
        parts.append(f"**File**: `{operation.file_path}`")
        parts.append(f"**Operation**: {operation.operation}")
        parts.append("")

        # Instructions - use provided or generate enhanced instructions
        instructions = operation.instructions
        if not instructions:
            # Generate enhanced instructions based on file type, operation, and description
            if operation.operation == "create":
                if operation.file_path.endswith('.css'):
                    if 'line number' in operation.description.lower() or 'editor' in operation.description.lower():
                        instructions = f"Create a new CSS file at {operation.file_path} for a code editor with line numbers. Include:\n- Monospace font for code\n- Proper positioning for line numbers column\n- Styling for code editor area\n- Syntax highlighting styles if possible\n- Responsive design"
                    else:
                        instructions = f"Create a new CSS file at {operation.file_path} with appropriate styles. Include proper selectors, properties, and comments."
                elif operation.file_path.endswith('.js'):
                    if 'line number' in operation.description.lower() or 'editor' in operation.description.lower():
                        instructions = f"Create a new JavaScript file at {operation.file_path} for line number functionality. Implement:\n- Function to generate line numbers based on code content\n- Scroll synchronization between code area and line numbers\n- Update line numbers when code is edited\n- Event listeners for user interactions\n- Error handling and debugging"
                    else:
                        instructions = f"Create a new JavaScript file at {operation.file_path} with proper functions and event handlers. Include error handling and comments."
                elif operation.file_path.endswith('.html') or 'templates/' in operation.file_path:
                    if 'editor' in operation.description.lower() or 'line number' in operation.description.lower():
                        instructions = f"Create a new HTML template at {operation.file_path} for a code editor with line numbers. Include:\n- A textarea or contenteditable div for code editing\n- A separate div for displaying line numbers\n- Proper CSS classes for styling\n- References to CSS and JavaScript files\n- JavaScript hooks for line number updates"
                    else:
                        instructions = f"Create a new HTML template at {operation.file_path} with proper structure, elements, and CSS/JS references."
                elif LanguageId.from_path(operation.file_path) is LanguageId.PYTHON:
                    if 'service' in operation.description.lower() or 'endpoint' in operation.description.lower():
                        instructions = f"Create a new Python service file at {operation.file_path}. Include:\n- Route/endpoint definitions\n- Proper request/response models\n- Error handling and logging\n- Business logic for editor functionality"
                    elif 'test' in operation.file_path:
                        instructions = f"Create a new test file at {operation.file_path} for editor functionality. Include:\n- Test cases for line number functionality\n- Setup and teardown methods\n- Assertions and validation\n- Mocking if needed"
                    else:
                        instructions = f"Create a new Python file at {operation.file_path} with proper imports, functions, classes, and docstrings."
                else:
                    instructions = f"Create a new file at {operation.file_path} for {original_request[:100]}..."
            else:  # modify
                if operation.file_path == 'main.py' and 'route' in operation.description.lower():
                    instructions = "Modify main.py to add a new route for the editor page. Include:\n- Import for editor service router\n- Route registration in the app\n- Proper route configuration\n- Update any necessary middleware or dependencies"
                else:
                    instructions = f"Modify the file {operation.file_path} to {operation.description}"

        # Enhance instructions for main.py modifications
        if operation.file_path == 'main.py' and operation.operation == 'modify':
            if '**Important for main.py modifications**' not in instructions:  # Avoid duplication
                instructions += "\n\n**Important for main.py modifications**:"
                instructions += "\n- Find the correct location in the existing main.py file"
                instructions += "\n- Maintain existing imports and structure"
                instructions += "\n- Add imports at the top if needed"
                instructions += "\n- Add routes after other route definitions"
                instructions += "\n- Ensure proper indentation and syntax"

        parts.append("**Specific Instructions**:")
        parts.append(instructions)
        parts.append("")

        # Dependencies
        if operation.dependencies:
            parts.append("**Dependencies** (already created):")
            for dep in operation.dependencies:
                parts.append(f"- `{dep}`")
            parts.append("")

        # Template reference
        if operation.template_file:
            parts.append(f"**Template Reference**: `{operation.template_file}`")
            parts.append("")

        # Project conventions
        if project_structure.frameworks:
            parts.append(f"**Frameworks**: {', '.join(project_structure.frameworks)}")
            parts.append(f"**Follow {', '.join(project_structure.frameworks)} best practices**")
        elif project_structure.framework:
            parts.append(f"**Framework**: {project_structure.framework}")
            parts.append(f"**Follow {project_structure.framework} best practices**")

        if project_structure.project_types:
            parts.append(f"**Project Type**: {', '.join(project_structure.project_types)}")

        # Output format instructions based on mode
        parts.append("")
        parts.append("**Output Format (IMPORTANT)**:")

        if output_mode == OutputMode.UNIFIED_DIFF:
            parts.append("- **Mode**: UNIFIED_DIFF ONLY - Output MUST be a valid unified diff")
            parts.append("- **Requirements**:")
            parts.append("  - Start with 'diff --git a/path b/path' or '--- a/path'")
            parts.append("  - Include proper hunk headers: '@@ -start,count +start,count @@'")
            parts.append("  - Ensure patch can be applied with 'git apply'")
            parts.append("  - MUST include at least one @@ hunk (no header-only diffs)")
            parts.append("- **Example**:")
            parts.append("  ```diff")
            parts.append(f"  diff --git a/{operation.file_path} b/{operation.file_path}")
            parts.append("  index 1234567..89abcde 100644")
            parts.append(f"  --- a/{operation.file_path}")
            parts.append(f"  +++ b/{operation.file_path}")
            parts.append("  @@ -1,3 +1,4 @@")
            parts.append("   // Existing content")
            parts.append("  +// New line added")
            parts.append("   // More content")
            parts.append("  ```")
            parts.append("- **CRITICAL**: If you cannot produce a valid diff with @@ hunks, the request will fail.")
        elif output_mode == OutputMode.FULL_FILE:
            parts.append("- **Mode**: FULL_FILE MODE - Output MUST be a single FILE block")
            parts.append("- **Requirements**:")
            parts.append(f"  FILE: {operation.file_path}")
            parts.append("  ```<language>")
            parts.append("  // Complete file content here")
            parts.append("  // Every line exactly as it should appear in the file")
            parts.append("  ```")
            parts.append("- **Example**:")
            parts.append(f"  FILE: {operation.file_path}")
            parts.append("  ```javascript")
            parts.append("  // New JavaScript file")
            parts.append("  function example() {")
            parts.append("    return 'Hello';")
            parts.append("  }")
            parts.append("  ```")
            parts.append("- **CRITICAL**: For new file creation, output MUST be a FILE block with complete content.")
        else:
            # Other modes (ASICODE_BLOCK, TARGETED_BLOCK, PLAN_JSON, NEEDS_DISAMBIGUATION)
            parts.append(f"- **Mode**: {output_mode.value.upper()} MODE")
            parts.append("- Follow the appropriate output format for this mode.")
            if output_mode == OutputMode.ASICODE_BLOCK:
                parts.append("- Format: ASICODE_BEGIN / BEFORE / AFTER / ASICODE_END blocks")
            elif output_mode == OutputMode.TARGETED_BLOCK:
                parts.append("- Format: FUNCTION: <name> + INSERT_AFTER: <line> + code block")
            elif output_mode == OutputMode.PLAN_JSON:
                parts.append("- Format: JSON plan with operations array")
            elif output_mode == OutputMode.NEEDS_DISAMBIGUATION:
                parts.append("- Output: NEEDS_DISAMBIGUATION with clarification questions")

        return "\n".join(parts)

    def _combine_patches(self, operations_results: list[dict]) -> str:
        """Combine patches from multiple operations"""

        patches = []

        for result in operations_results:
            if result.get("success") and result.get("patch"):
                patches.append(result["patch"])

        return "\n\n".join(patches)

    def _create_default_file_patch(self, repo_path: Path, operation: FileOperation) -> str:
        """Create a default file patch when LLM fails to generate one"""

        file_path = operation.file_path.lstrip('/')
        target_path = repo_path / file_path

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate default content based on file type
        ext = file_path.split('.')[-1].lower() if '.' in file_path else ''

        if ext == 'js':
            content = """// JavaScript file for editor with line numbers
// Created automatically because LLM failed to generate content

function updateLineNumbers(editorId) {
    const editor = document.getElementById(editorId);
    const lineNumbers = document.getElementById(editorId + '-linenumbers');
    if (!editor || !lineNumbers) return;

    const lines = editor.value.split('\\n');
    lineNumbers.innerHTML = lines.map((_, i) => `<div>${i + 1}</div>`).join('');
}

function syncScroll(editorId) {
    const editor = document.getElementById(editorId);
    const lineNumbers = document.getElementById(editorId + '-linenumbers');
    if (editor && lineNumbers) {
        lineNumbers.scrollTop = editor.scrollTop;
    }
}

// Export functions for use
window.EditorUtils = {
    updateLineNumbers,
    syncScroll
};"""
        elif ext == 'css':
            content = """/* CSS for editor with line numbers */
/* Created automatically because LLM failed to generate content */

.editor-container {
    display: flex;
    border: 1px solid #ccc;
    border-radius: 4px;
    overflow: hidden;
    font-family: monospace;
}

.line-numbers {
    background-color: #f5f5f5;
    color: #666;
    padding: 8px;
    text-align: right;
    border-right: 1px solid #ddd;
    user-select: none;
    min-width: 40px;
    overflow-y: hidden;
}

.code-area {
    flex: 1;
    border: none;
    padding: 8px;
    resize: vertical;
    min-height: 200px;
    font-family: monospace;
    white-space: pre;
}"""
        elif ext == 'html' or 'templates/' in file_path:
            content = f"""<!-- HTML template for editor with line numbers -->
<!-- Created automatically because LLM failed to generate content -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Editor</title>
    <link rel="stylesheet" href="/static/css/{file_path.replace('.html', '.css').split('/')[-1].replace('editor.html', 'editor.css')}">
</head>
<body>
    <div class="editor-container">
        <div class="line-numbers" id="editor-linenumbers">1</div>
        <textarea class="code-area" id="editor" oninput="updateLineNumbers('editor')" onscroll="syncScroll('editor')"></textarea>
    </div>
    <script src="/static/js/{file_path.replace('.html', '.js').split('/')[-1].replace('editor.html', 'editor.js')}"></script>
</body>
</html>"""
        elif ext == 'py':
            if 'test' in file_path:
                content = """# Test file for editor functionality
# Created automatically because LLM failed to generate content

import pytest

def test_editor_basic():
    \"\"\"Basic test for editor functionality\"\"\"
    assert True

def test_line_numbers():
    \"\"\"Test line number calculation\"\"\"
    assert 1 + 1 == 2"""
            else:
                content = f"""# File: {file_path}
# Created automatically by asicode. Please implement the actual functionality."""
        else:
            content = f"""# File: {file_path}
# Created automatically because LLM failed to generate content
# This is a placeholder file. Please implement the actual functionality."""

        # Create unified diff patch for new file
        diff = f"""diff --git a/{file_path} b/{file_path}
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/{file_path}
@@ -0,0 +1,{len(content.split(chr(10)))} @@
{chr(10).join('+' + line for line in content.split(chr(10)))}
"""

        # Write the file
        try:
            target_path.write_text(content, encoding='utf-8')
            logger.info(f"Created default file: {file_path}")
            return diff
        except Exception as e:
            logger.error(f"Failed to create default file {file_path}: {e}")
            return ""

    def _generate_multi_file_explanation(
        self,
        plan: ExecutionPlan,
        operations_results: list[dict],
    ) -> str:
        """Generate explanation for multi-file operation"""

        lines = []

        lines.append(f"Executed multi-file plan for: {plan.original_request}")
        lines.append("")
        lines.append(f"Complexity: {plan.complexity}")
        lines.append(f"Operations performed: {len(operations_results)}")
        lines.append("")

        # Summary
        successful = sum(1 for r in operations_results if r.get("success"))
        failed = len(operations_results) - successful

        lines.append(f"✅ Successful: {successful}")
        if failed > 0:
            lines.append(f"❌ Failed: {failed}")

        lines.append("")
        lines.append("Files modified:")
        for result in operations_results:
            status = "✅" if result.get("success") else "❌"
            lines.append(f"{status} {result['file']} ({result['operation']})")

        return "\n".join(lines)

    def _estimate_change_size(self, user_request: str, operation: str, file_path: str) -> Optional[str]:
        """
        Estimate change size based on operation type.

        Keyword-based estimation was removed — unreliable since words like
        "add" could mean anything from adding a line to adding a feature.
        Operation type is the only reliable structural signal available here.
        """
        if operation == 'create':
            return 'medium'
        elif operation == 'modify':
            return 'small'
        return None

    def _analyze_failure_patterns(self, result: dict) -> None:
        """
        Analyze failure patterns and log insights for improvement.

        This can be extended to track statistics over time.
        """
        if not result.get('success'):
            error = result.get('error', '')
            output_mode = result.get('output_mode_used', result.get('output_mode', 'unknown'))
            target_file = result.get('target_file', 'unknown')

            # Extract file extension for pattern analysis
            file_ext = 'unknown'
            if target_file and target_file != 'unknown':
                if '.' in target_file:
                    file_ext = target_file.split('.')[-1]
                else:
                    file_ext = 'no_extension'

            failure_type = 'unknown'
            if 'empty_patch' in error:
                failure_type = 'empty_patch'
            elif 'git_apply_check_failed' in error:
                failure_type = 'git_apply_failed'
            elif 'invalid_diff' in error:
                failure_type = 'invalid_diff'

            logger.info(
                f"Failure analysis - Type: {failure_type}, "
                f"Mode: {output_mode}, File: .{file_ext}, "
                f"Error: {error[:100]}..."
            )

            # Simple recommendations based on failure patterns
            if failure_type == 'empty_patch' and output_mode == 'unified_diff':
                logger.info("  → Recommendation: Try FULL_FILE mode for this file type")
            elif failure_type == 'git_apply_failed':
                logger.info("  → Recommendation: Check if diff context lines match file content")

    def _build_error_feedback(self, error: str, file_path: Path, repo_path: Path) -> str:
        """
        Build error feedback for correction retry.

        Includes:
        - Failed file path
        - Git apply error summary
        - Original context lines (10-30 lines) if file exists
        """
        feedback_lines = []

        # File path
        rel_path = file_path.relative_to(repo_path) if file_path.is_relative_to(repo_path) else file_path
        feedback_lines.append(f"- Failed file: {rel_path}")

        # Error summary
        error_summary = error
        # Extract error code if present
        if "git_apply_check_failed" in error:
            error_summary = "git apply check failed (patch cannot be applied)"
        elif "empty_patch" in error:
            error_summary = "empty patch (no diff content)"
        elif "missing_hunks" in error:
            error_summary = "missing hunks (diff has no @@ hunk)"
        elif "invalid_diff" in error:
            error_summary = "invalid diff format"
        feedback_lines.append(f"- Error: {error_summary}")

        # Original context lines
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                if lines:
                    feedback_lines.append("- Original file context (first 30 lines):")
                    for i, line in enumerate(lines[:30]):
                        feedback_lines.append(f"  {i+1}: {line}")
                        if i >= 29:
                            break
            except Exception as e:
                logger.warning(f"Could not read original file for error feedback: {e}")
        else:
            feedback_lines.append("- Original file: does not exist (new file creation)")

        return "\n".join(feedback_lines)

    def _extract_failure_reason(self, error: str) -> str:
        """
        Extract failure reason code from error string.

        Returns short codes like: 'empty_patch', 'git_apply_failed', 'missing_hunks',
        'invalid_diff', 'no_diff', 'header_only', etc.
        """
        if not error:
            return "unknown"

        error_lower = error.lower()

        if "empty_patch" in error_lower:
            return "empty_patch"
        elif "git_apply_check_failed" in error_lower:
            return "git_apply_failed"
        elif "missing_hunks" in error_lower:
            return "missing_hunks"
        elif "header-only" in error_lower or "header only" in error_lower:
            return "header_only"
        elif "invalid_diff" in error_lower:
            return "invalid_diff"
        elif "no diff found" in error_lower or "no_diff" in error_lower:
            return "no_diff"
        elif "inconsistent hunk line counts" in error_lower:
            return "inconsistent_hunk_lines"
        else:
            # Return first 50 chars as reason
            return error[:50].replace("\n", " ").strip()

    def _analysis_to_dict(self, analysis: RequestAnalysis) -> dict:
        """Convert analysis to dict"""
        return {
            "intent": analysis.intent,
            "feature_name": analysis.feature_name,
            "suggested_files": analysis.suggested_files,
            "tech_stack": analysis.tech_stack,
            "confidence": analysis.confidence,
            "needs_planning": analysis.needs_planning,
        }

    def _plan_to_dict(self, plan: ExecutionPlan) -> dict:
        """Convert plan to dict"""
        return {
            "complexity": plan.complexity,
            "strategy": plan.strategy,
            "operations": [
                {
                    "file": op.file_path,
                    "operation": op.operation,
                    "description": op.description,
                    "dependencies": op.dependencies,
                }
                for op in plan.operations
            ],
            "success_criteria": plan.success_criteria,
        }


def create_intelligent_service_from_env(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[IntelligentLLMService]:
    """
    Create IntelligentLLMService from environment variables.

    Environment variables:
    - EXTERNAL_LLM_PROVIDER: Provider name (openai, anthropic, google, deepseek, ollama, zai, openrouter)
    - EXTERNAL_LLM_MODEL: Model to use (optional)
    - OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY / OLLAMA_API_KEY / ZAI_API_KEY / OPENROUTER_API_KEY
    - EXTERNAL_LLM_BASE_URL: Optional base URL override

    Args:
        api_key: Optional API key override. If provided, skips env var lookup.
    """
    # Get provider from parameter or env var (no implicit default)
    prov = (provider or os.getenv("EXTERNAL_LLM_PROVIDER", "") or "").strip().lower()
    if not prov:
        logger.debug("No external LLM provider configured")
        return None

    # Remove "external_" prefix if present (UI uses "external_google", "external_deepseek")
    if prov.startswith("external_"):
        prov = prov[len("external_"):]

    api_key_env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "zai": "ZAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "opencode": "OPENCODE_API_KEY",
    }
    api_key_var = api_key_env_vars.get(prov)
    if not api_key_var:
        logger.error("Unknown provider: %s", prov)
        return None

    # Use explicit api_key parameter if provided, otherwise fall back to env var
    resolved_key = (api_key or "").strip() or (os.getenv(api_key_var, "") or "").strip()
    if not resolved_key:
        if prov != "ollama":
            logger.warning("No API key found for %s (set %s or pass api_key)", prov, api_key_var)
            return None
        else:
            logger.info("Ollama provider doesn't require API key, using empty string")
    api_key = resolved_key

    m = (model or os.getenv("EXTERNAL_LLM_MODEL", "") or "").strip() or None
    # Strip "external_provider:" prefix from model name (UI sends "external_deepseek:deepseek-chat")
    if m and ":" in m:
        prefix, _, bare_model = m.partition(":")
        if prefix.startswith("external_"):
            m = bare_model.strip() or None
    # Provider-scoped base_url: a foreign provider's global base_url must not
    # leak in (see resolve_provider_base_url). Falls back to the client's
    # DEFAULT_BASE_URL when no override applies.
    from .client import resolve_provider_base_url
    base_url = resolve_provider_base_url(prov)

    try:
        svc = IntelligentLLMService(
            provider=prov,
            api_key=api_key,
            model=m,
            base_url=base_url,
        )
        logger.info("Intelligent LLM service created: %s (%s)", prov, svc.model)
        return svc
    except Exception as e:
        logger.error("Failed to create intelligent LLM service: %s", e)
        return None
