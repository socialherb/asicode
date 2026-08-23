"""repl_impl stage-3 pty coverage: /claude collaboration + auto-continue.

Residual stage-2 misses were: the whole ``/claude`` dispatch branch (77 lines:
arg parsing, SDK-missing install flow, session run, verdict record, cancel and
error paths) plus auto-continue edges (``/auto`` error/cap/NEXT_SUGGEST-off
prints, error-turn stop, checkpoint change summary, suggestion-kick failure,
clipboard image path, think-effort label).

Strategy (same as stage 2): the child driver re-applies fakes; ``--collab-*``
flags swap in a fake ``external_llm.repl.collaborate`` module via
``sys.modules`` so every ``from external_llm.repl.collaborate import ...``
site in repl_impl resolves to it; ``--auto-suggest-text``/``--force-underline``
arm a real countdown so the auto-submit fires through the live prompt.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

_TODAY = datetime.now().strftime("%Y-%m-%d %H:%M +0900")

INSIGHTS_FILE = (
    "# Design Chat Insights (.asr-edit/design_insights.md)\n"
    "\n"
    "> **Principle**: Structural/generic approach.\n"
    "\n"
    f"### [pattern] {_TODAY}\n"
    "Original insight line one.\n"
    "\n"
    f"### [design_decision] {_TODAY}\n"
    "Original insight line two.\n"
)


class _PtyReplBase:
    CHILD = str(Path(__file__).parent / "repl_stage2_child.py")

    def _spawn(self, repo_root, *extra, env=None, timeout=60.0):
        from tests.unit.pty_driver import SpawnPtySession

        argv = [sys.executable, self.CHILD, "--mode", "repl", "--repo", repo_root, *extra]
        return SpawnPtySession(argv, cwd=os.getcwd(), env=env, timeout=timeout)

    def _send_cmd(self, sess, text):
        data = text if isinstance(text, bytes) else text.encode()
        sess.clear()
        sess.send(data)
        sess.wait_for(data, timeout=30)
        time.sleep(0.15)
        sess.send(b"\r")

    def _write_insights(self, repo_root):
        p = Path(repo_root) / ".asicode"
        p.mkdir(parents=True, exist_ok=True)
        (p / "design_insights.md").write_text(INSIGHTS_FILE, encoding="utf-8")

    def _session(self, tmp_path, *extra, env=None):
        repo = str(tmp_path)
        self._write_insights(repo)
        sess = self._spawn(repo, *extra, env=env)
        sess.wait_for(b"asicode", timeout=60)
        return sess


class TestCollabPty(_PtyReplBase):
    """/claude collaboration branches (77 residual misses)."""

    def test_collab_sdk_installed_success(self, tmp_path):
        """Installed SDK: --fresh/--model parsing, handoff skip, run, verdict."""
        sess = self._session(tmp_path)
        try:
            self._send_cmd(sess, "/claude --fresh --model sonnet-4-6 verify the build")
            sess.wait_for(b"verify the build (sonnet-4-6)", timeout=10)
            sess.wait_for(b"[collab summary]", timeout=10)
            sess.wait_for(b"[collab log flushed]", timeout=10)
            # Verdict recorded into the fake session (no error printed).
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_collab_m_short_flag(self, tmp_path):
        """``-m`` short flag parsing."""
        sess = self._session(tmp_path)
        try:
            self._send_cmd(sess, "/claude -m haiku-3 run the tests")
            sess.wait_for(b"run the tests (haiku-3)", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_usage_when_no_task(self, tmp_path):
        """No task -> usage line, no session start."""
        sess = self._session(tmp_path)
        try:
            self._send_cmd(sess, "/claude")
            sess.wait_for(b"usage: /claude [--fresh]", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_sdk_missing_decline_install(self, tmp_path):
        """SDK missing + user declines (n) -> install hint."""
        sess = self._session(tmp_path, "--collab-sdk", "missing")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"claude_agent_sdk", timeout=10)
            sess.wait_for(b"Install it now?", timeout=10)
            sess.clear()
            # The answer feeds builtin input() (canonical mode), NOT a ptk
            # prompt (raw mode): with the harness's ICRNL disabled on the
            # slave (see pty_driver._disable_cr_translation), CR is never
            # translated to NL, so only an explicit "\n" terminates the line.
            sess.send(b"n\n")
            sess.wait_for(b"Install with:  pip install '.[collaborate]'", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_sdk_missing_install_ok(self, tmp_path):
        """SDK missing + y + install ok -> session starts."""
        sess = self._session(tmp_path, "--collab-sdk", "missing", "--collab-install", "ok")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"Install it now?", timeout=10)
            sess.clear()
            sess.send(b"y\n")
            sess.wait_for(b"installed \xe2\x80\x94 starting collaboration session.", timeout=10)
            sess.wait_for(b"verify the build (claude-sonnet-4-6)", timeout=10)
            sess.wait_for(b"[collab summary]", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_sdk_missing_install_fail(self, tmp_path):
        """SDK missing + y + install failure -> manual-install hint."""
        sess = self._session(tmp_path, "--collab-sdk", "missing", "--collab-install", "fail")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"Install it now?", timeout=10)
            sess.clear()
            sess.send(b"y\n")
            sess.wait_for(b"did not complete successfully", timeout=10)
            sess.wait_for(b"pip install '.[collaborate]'", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_sdk_missing_install_keyboardinterrupt(self, tmp_path):
        """SDK missing + y + install KeyboardInterrupt -> cancelled."""
        sess = self._session(tmp_path, "--collab-sdk", "missing", "--collab-install", "kb")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"Install it now?", timeout=10)
            sess.clear()
            sess.send(b"y\n")
            sess.wait_for(b"cancelled.", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_orchestrator_raises(self, tmp_path):
        """CollaborationOrchestrator run() raises -> error print."""
        sess = self._session(tmp_path, "--collab-orch", "raise")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"collaboration error: fake collaboration crash", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_orchestrator_keyboardinterrupt(self, tmp_path):
        """CollaborationOrchestrator run() KeyboardInterrupt -> cancelled."""
        sess = self._session(tmp_path, "--collab-orch", "keyboard")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"cancelled.", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_result_error_skips_verdict(self, tmp_path):
        """Result with .error truthy -> no add_turn (verdict skipped)."""
        sess = self._session(tmp_path, "--collab-result-error")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"[collab summary] error=RuntimeError", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_handoff_failure_does_not_block(self, tmp_path):
        """build_session_handoff raises -> context None, session still runs."""
        sess = self._session(tmp_path, "--collab-handoff-raise")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"verify the build (claude-sonnet-4-6)", timeout=10)
            sess.wait_for(b"[collab summary]", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_collab_verdict_record_failure(self, tmp_path):
        """format_verdict_for_session raises -> debug log, session continues."""
        sess = self._session(tmp_path, "--collab-verdict-raise")
        try:
            self._send_cmd(sess, "/claude verify the build")
            sess.wait_for(b"[collab summary]", timeout=10)
            # The verdict failure is only a debug log — REPL stays alive.
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()


class TestAutoContinuePty(_PtyReplBase):
    """/auto dispatch edges + auto-continue main-loop branches."""

    def test_auto_bad_arg_prints_error(self, tmp_path):
        sess = self._session(tmp_path)
        try:
            self._send_cmd(sess, "/auto badarg")
            sess.wait_for(b"usage: /auto [N | on | off]", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_auto_numeric_sets_cap(self, tmp_path):
        sess = self._session(tmp_path)
        try:
            self._send_cmd(sess, "/auto 3")
            sess.wait_for(b"max 3 consecutive", timeout=10)
            self._send_cmd(sess, "/auto off")
            sess.wait_for(b"auto-continue OFF", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_auto_on_warns_when_next_suggest_disabled(self, tmp_path):
        """display.NEXT_SUGGEST disabled (env at import) -> warning on /auto on."""
        env = dict(os.environ)
        env["ASICODE_NEXT_SUGGEST"] = "0"
        sess = self._session(tmp_path, env=env)
        try:
            self._send_cmd(sess, "/auto on")
            sess.wait_for(b"next-step suggestion is disabled in config", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_error_turn_stops_auto_continue(self, tmp_path):
        """is_error result + auto on -> 'turn ended with an error — stopped'."""
        sess = self._session(tmp_path, "--error-turn")
        try:
            self._send_cmd(sess, "/auto on")
            sess.wait_for(b"auto-continue ON", timeout=10)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"auto-continue: turn ended with an error \xe2\x80\x94 stopped", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_think_effort_label(self, tmp_path):
        """_reasoning_effort set -> 'think ON (high)' status label."""
        sess = self._session(tmp_path)
        try:
            self._send_cmd(sess, "/think high")
            sess.wait_for(b"effort=high", timeout=10)
            # Next prompt's status line shows the effort label.
            self._send_cmd(sess, "hello")
            sess.wait_for(b"think ON (high)", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_checkpoint_change_summary(self, tmp_path):
        """Non-git dir: newer checkpoint + changed files -> summary print."""
        sess = self._session(tmp_path, "--checkpoint-seq")
        try:
            self._send_cmd(sess, "hello")
            sess.wait_for(b"2 file(s) changed", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_checkpoint_summary_failure(self, tmp_path):
        """_newest_checkpoint_id raises at turn end -> debug log only."""
        sess = self._session(tmp_path, "--checkpoint-fail")
        try:
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_suggestion_kick_failure(self, tmp_path):
        """_kick_next_prompt_suggestion raises -> debug log, REPL alive."""
        sess = self._session(tmp_path, "--suggest-kick-fail")
        try:
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=10)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_clipboard_image_path(self, tmp_path):
        """Clipboard image detection -> 📷 hint + base64 size."""
        sess = self._session(tmp_path, "--clipboard-image")
        try:
            self._send_cmd(sess, "hello")
            sess.wait_for(b"\xf0\x9f\x93\xb7 image/png", timeout=10)
            sess.wait_for(b"Here is the plan: done.", timeout=10)
            # _send_cmd's echo-wait for "exit" can false-match the prompt's
            # "Ctrl+C exit" helper line while the turn output is still landing
            # — send the quit directly once the turn has completed.
            time.sleep(0.5)
            sess.clear()
            sess.send(b"exit\r")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()

    def test_auto_submit_fires_countdown(self, tmp_path):
        """Real auto-submit: countdown arms, fires, next turn is auto-input.

        Uses ASICODE_AUTO_CONTINUE_DELAY=2 (module-level parse at child import)
        + --force-underline so the plain branch's _input_underline guard in
        _auto_submit_now is satisfied. The suggestion arrives ~0.6s after the
        turn ends (while the next prompt is live), arms the Timer, and 2s later
        the ghost text is submitted automatically -> "auto-continue step 1/1".
        """
        env = dict(os.environ)
        env["ASICODE_AUTO_CONTINUE_DELAY"] = "2"
        sess = self._session(tmp_path, "--auto-suggest-text", "verify the change", "--force-underline", env=env)
        try:
            self._send_cmd(sess, "/auto 1")
            sess.wait_for(b"auto-continue ON", timeout=10)
            self._send_cmd(sess, "hello")
            # First turn completes normally...
            sess.wait_for(b"Here is the plan: done.", timeout=10)
            # ...then the countdown fires and submits the ghost automatically.
            sess.wait_for(b"auto-continue step 1/1", timeout=15)
            # Cap reached -> paused notice.
            sess.wait_for(b"cap reached", timeout=15)
            # Back to an interactive prompt: manual input still works.
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
        finally:
            sess.close()
