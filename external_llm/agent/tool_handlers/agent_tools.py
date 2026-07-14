"""Agent delegation and memory tool handlers for ToolRegistry."""
from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from .constants import ASK_USER_DEFAULT_TIMEOUT  # leaf module — avoids tool_registry circular import

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)


class AgentToolsMixin:
    """Mixin providing agent-level tool implementations for ToolRegistry."""

    def _tool_update_plan(self, args: dict[str, Any]) -> "ToolResult":
        """Create/replace the work plan for the current goal.

        The plan lives on the registry instance (``self.session_plan``);
        this handler returns the rendered checklist as the tool result so the
        model sees current state inline. The design chat loop gates final
        responses on open items (see design_chat_loop._respond_impl) but does
        NOT re-inject the plan on every iteration.
        """
        from ..plan_state import diff_plans, open_items, render_plan, validate_plan

        plan, err = validate_plan(args.get("goal"), args.get("items"))
        if plan is None:
            return self._make_result(ok=False, content="", error=err)

        # Preserve the goal from the first call when later calls omit it
        prev = getattr(self, "session_plan", None)
        if not plan["goal"] and prev and prev.get("goal"):
            plan["goal"] = prev["goal"]

        summary = diff_plans(prev, plan)
        # Previous plan's title→status map — used by CLI to render per-item changes (new items,
        # status transitions, deletions) as annotations. None for the first plan ("Plan created").
        prev_statuses = (
            {it.get("title", ""): it.get("status", "") for it in prev.get("items", [])}
            if prev else None
        )
        self.session_plan = plan
        n_open = len(open_items(plan))
        return self._make_result(
            ok=True,
            content=f"{summary}\n{render_plan(plan)}",
            metadata={
                "open_items": n_open,
                "total_items": len(plan["items"]),
                "summary": summary,
                "plan": plan,
                "prev_statuses": prev_statuses,
            },
        )

    def _tool_update_memory(self, args: dict[str, Any]) -> "ToolResult":
        """Append a timestamped note to .asicode/memory.md."""
        import os

        note = str(args.get("note", "")).strip()
        if not note:
            return self._make_result(ok=False, content="", error="'note' is required")
        note = note[:1000]

        section = str(args.get("section", "")).strip()
        memory_dir = os.path.join(self.repo_root, ".asicode")
        memory_path = os.path.join(memory_dir, "memory.md")

        try:
            os.makedirs(memory_dir, exist_ok=True)
            self._ensure_asicode_gitignored()
            timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            if section:
                entry = f"\n### {section} ({timestamp})\n{note}\n"
            else:
                entry = f"\n<!-- {timestamp} -->\n{note}\n"
            with open(memory_path, "a", encoding="utf-8") as fh:
                fh.write(entry)
            return self._make_result(
                ok=True,
                content=f"Memory updated: {len(note)} chars appended to .asicode/memory.md",
                metadata={"path": ".asicode/memory.md", "section": section},
            )
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Failed to update memory: {e}")

    def _tool_delegate_to_helper(self, args: dict[str, Any]) -> "ToolResult":
        """Delegate an isolated code generation subtask to the helper model."""
        if self.local_assistant is None:
            return self._make_result(
                ok=False,
                content="",
                error=(
                    "Helper not available. "
                    "Set helper_enabled=True and helper_model in config to enable delegate_to_helper."
                ),
            )

        role = args.get("role", "").strip()
        instruction = args.get("instruction", "").strip()
        file_path = args.get("file_path", "").strip()
        target_symbol = args.get("target_symbol", "").strip()

        function_signature = args.get("function_signature", "").strip()
        context_code = args.get("context_code", "").strip()

        constraints = args.get("constraints", "").strip()

        if not role:
            return self._make_result(ok=False, content="", error="'role' is required")
        if not instruction:
            return self._make_result(ok=False, content="", error="'instruction' is required")

        if not function_signature or not context_code:
            try:
                from external_llm.agent.symbol_search import SymbolSearcher
                from external_llm.context.context_packs import HelperContextBuilder

                builder = HelperContextBuilder(self.repo_root)

                if not function_signature and target_symbol:
                    try:
                        searcher = SymbolSearcher(str(self.repo_root))
                        symbol_info = searcher.get_symbol_info(target_symbol, file_path=file_path)
                        if symbol_info and symbol_info.get("signature"):
                            function_signature = str(symbol_info.get("signature") or "").strip()
                    except (AttributeError, TypeError):
                        pass

                local_snippet = context_code
                if not local_snippet and file_path:
                    try:
                        start_line = 1
                        end_line = 80

                        if target_symbol:
                            try:
                                searcher = SymbolSearcher(str(self.repo_root))
                                symbol_info = searcher.get_symbol_info(target_symbol, file_path=file_path)
                                if symbol_info and symbol_info.get("line"):
                                    line_no = int(symbol_info.get("line"))
                                    start_line = max(1, line_no - 12)
                                    end_line = line_no + 28
                            except (AttributeError, TypeError):
                                pass

                        from pathlib import Path
                        abs_fp = Path(self.repo_root) / file_path if not Path(file_path).is_absolute() else Path(file_path)
                        try:
                            content = abs_fp.read_text(encoding="utf-8", errors="replace")
                            c_lines = content.splitlines()
                            s = max(0, start_line - 1)
                            e = min(end_line, len(c_lines))
                            local_snippet = "\n".join(f"{i}: {line}" for i, line in enumerate(c_lines[s:e], start=s + 1))
                        except Exception:
                            local_snippet = ""
                    except (OSError, AttributeError):
                        pass

                helper_pack = builder.build(
                    task=instruction,
                    function_signature=function_signature or None,
                    local_snippet=local_snippet or None,
                    constraints=constraints or None,
                )

                if helper_pack and getattr(helper_pack, "content", ""):
                    context_code = helper_pack.content

            except Exception as e:
                logger.debug(f"Helper context build failed: {e}")

        try:
            result = self.local_assistant.delegate_single_task(
                role=role,
                instruction=instruction,
                file_path=file_path,
                function_signature=function_signature,
                context_code=context_code,
                constraints=constraints,
            )

            if result.get("success", False):
                content = f"Local LLM generated {role}:\n\n```{result.get('language', 'python')}\n{result.get('code', '')}\n```\n"
                if result.get("issues"):
                    content += "\nIssues noted:\n" + "\n".join(f"- {issue}" for issue in result.get("issues", []))
                if result.get("validation"):
                    content += f"\nValidation: {result.get('validation', {})}"
                return self._make_result(
                    ok=True,
                    content=content,
                    metadata={"role": role, "validation": result.get("validation"), "issues": result.get("issues")},
                )
            else:
                error_msg = result.get("error", "Local model generation failed")
                return self._make_result(
                    ok=False,
                    content="",
                    error=f"Local model delegation failed: {error_msg}",
                )
        except Exception as e:
            logger.exception("Local assistant delegation failed")
            return self._make_result(
                ok=False,
                content="",
                error=f"Local assistant delegation raised an exception: {e}",
            )

    def _tool_ask_user(self, args: dict[str, Any]) -> "ToolResult":
        """Ask the user a clarification question. Blocks until response or timeout."""
        question = str(args.get("question", "")).strip()
        q_type = str(args.get("type", "free_text")).strip()
        options = args.get("options") or []
        reason = str(args.get("reason", "")).strip()
        default = str(args.get("default", "")).strip()

        if not question:
            return self._make_result(ok=False, content="", error="'question' is required")

        # Check if feature is enabled
        config = getattr(self, "config", None)
        if not config or not getattr(config, "user_checkpoint_enabled", False):
            return self._make_result(
                ok=True,
                content=f"User checkpoint disabled. Using default: {default}",
                metadata={"status": "disabled", "answer": default},
            )

        # Check question count limit
        max_q = getattr(config, "user_checkpoint_max_questions", 3)
        current_count = getattr(config, "_user_checkpoint_count", 0)
        if current_count >= max_q:
            return self._make_result(
                ok=True,
                content=f"Question limit reached ({max_q}). Using default: {default}",
                metadata={"status": "limit_reached", "answer": default},
            )

        callback = getattr(config, "user_checkpoint_callback", None)
        if not callback:
            return self._make_result(
                ok=True,
                content=f"No checkpoint callback. Using default: {default}",
                metadata={"status": "no_callback", "answer": default},
            )

        # Increment question counter
        config._user_checkpoint_count = current_count + 1

        # Emit SSE and block
        question_data = {
            "question": question,
            "type": q_type,
            "options": options,
            "reason": reason,
            "default": default,
            "source": "agent",
            "timeout": getattr(config, "user_checkpoint_timeout", ASK_USER_DEFAULT_TIMEOUT),
            "question_id": f"ask_{uuid.uuid4().hex[:8]}",
        }

        try:
            response = callback(question_data)
            status = response.get("status", "timeout")
            answer = response.get("answer", default)
            note = response.get("note", "")

            content = f"User response ({status}): {answer}"
            if note:
                content += f"\nNote: {note}"

            return self._make_result(
                ok=True,
                content=content,
                metadata={"status": status, "answer": answer, "question": question},
            )
        except Exception as e:
            logger.warning("ask_user callback failed: %s", e)
            return self._make_result(
                ok=True,
                content=f"Checkpoint error, using default: {default}",
                metadata={"status": "error", "answer": default},
            )

