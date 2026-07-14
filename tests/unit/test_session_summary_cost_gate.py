"""Behavioral guard: the CLI exit banner (``_print_session_summary``) honors
``ASICODE_HIDE_COST``.

The shared ``_hide_cost_display()`` gate was wired into every turn/status line
but the *exit summary* forgot it, so ``ASICODE_HIDE_COST=1`` still leaked the
session dollar amount (e.g. ``session 6h 50m · ↑… ↓… tokens · $9.4463``) on
quit. This pins the gate to the exit banner so the on-demand ``/cost`` /
``/status`` reports (intentionally ungated) remain the only surfaces that
show dollars when the switch is on.
"""

import asi


def _capture_print(monkeypatch):
    recorded = []
    monkeypatch.setattr(asi, "_print", lambda *a, **k: recorded.append(a[0]))
    return recorded


class TestSessionSummaryHonorsCostGate:
    def _tokens(self, cost=12.3456, actual=12.3456):
        return {"prompt": 1000, "completion": 500, "cost": cost, "actual_cost": actual}

    def test_hides_dollar_when_switch_on(self, monkeypatch):
        monkeypatch.setenv("ASICODE_HIDE_COST", "1")
        recorded = _capture_print(monkeypatch)

        asi._print_session_summary(self._tokens(), asi.time.monotonic())

        assert recorded, "exit banner should still print tokens/duration"
        assert "$" not in recorded[0], recorded[0]
        assert "tokens" in recorded[0]  # token counts are NOT money → kept

    def test_shows_dollar_when_switch_off(self, monkeypatch):
        monkeypatch.delenv("ASICODE_HIDE_COST", raising=False)
        recorded = _capture_print(monkeypatch)

        asi._print_session_summary(self._tokens(), asi.time.monotonic())

        assert "$" in recorded[0], recorded[0]

    def test_zero_usage_stays_silent(self, monkeypatch):
        monkeypatch.delenv("ASICODE_HIDE_COST", raising=False)
        recorded = _capture_print(monkeypatch)

        asi._print_session_summary({"prompt": 0, "completion": 0, "cost": 0.0}, asi.time.monotonic())

        assert recorded == []

    def test_gate_keys_off_truthy_truth_values(self, monkeypatch):
        # On/off truth set matches _hide_cost_display (1/true/yes/on → hide).
        for val in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("ASICODE_HIDE_COST", val)
            recorded = _capture_print(monkeypatch)
            asi._print_session_summary(self._tokens(), asi.time.monotonic())
            assert "$" not in recorded[0], (val, recorded[0])
        for val in ("", "0", "no", "off", "false"):
            monkeypatch.setenv("ASICODE_HIDE_COST", val)
            recorded = _capture_print(monkeypatch)
            asi._print_session_summary(self._tokens(), asi.time.monotonic())
            assert "$" in recorded[0], (val, recorded[0])
