"""
Autonomous Agent System for asicode.

Provides proactive, event-driven agent execution independent of user prompts.

Architecture:
  TriggerEngine  — collects events (file change, test fail, agent stall, cron)
  TriggerPolicy  — rule-based decision: event → action (NOTIFY/SUGGEST/AUTO_FIX/ESCALATE)
  TaskQueue      — priority queue with rate limiting and deduplication
  PushManager    — SSE-based server→client push registry
  ProactiveRunner — top-level coordinator wiring all components

Public API:
  get_or_create_runner(repo_root, model_tier, push_manager) → ProactiveRunner
  get_push_manager() → PushManager
  make_stream_callback_interceptor(repo_root, original_cb) → Callable
"""

from external_llm.editor.agent.autonomous.proactive_runner import (
    ProactiveRunner,
    get_or_create_runner,
    make_stream_callback_interceptor,
    update_runner_model_tier,
)
from external_llm.editor.agent.autonomous.push_manager import PushManager, get_push_manager
from external_llm.editor.agent.autonomous.task_queue import AutonomousTask, AutonomousTaskQueue
from external_llm.editor.agent.autonomous.trigger_engine import TriggerEngine, TriggerEvent, TriggerKind
from external_llm.editor.agent.autonomous.trigger_policy import ActionDecision, ActionKind, TriggerPolicy

__all__ = [
    "ActionDecision",
    "ActionKind",
    "AutonomousTask",
    "AutonomousTaskQueue",
    "ProactiveRunner",
    "PushManager",
    "TriggerEngine",
    "TriggerEvent",
    "TriggerKind",
    "TriggerPolicy",
    "get_or_create_runner",
    "get_push_manager",
    "make_stream_callback_interceptor",
    "update_runner_model_tier",
]
