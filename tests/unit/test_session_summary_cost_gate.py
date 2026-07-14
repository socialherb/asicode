"""Behavioral guard: the CLI exit banner (``_print_session_summary``) never shows
money.

The exit banner is an *ambient* summary printed automatically on quit, so the
dollar amount is noise there — the cost estimate is not surfaced on any CLI
surface (debug _log only). This pins that invariant: tokens + elapsed time
always render when there is usage, money never does, and a zero-usage session
stays silent.
"""

import asi


def _capture_print(monkeypatch):
    recorded = []
    monkeypatch.setattr(asi, "_print", lambda *a, **k: recorded.append(a[0]))
    return recorded


class TestSessionSummaryNeverShowsMoney:
    def _tokens(self, cost=12.3456, actual=12.3456):
        return {"prompt": 1000, "completion": 500, "cost": cost, "actual_cost": actual}

    def test_no_dollar_regardless_of_env(self, monkeypatch):
        # Money is excluded from the exit banner unconditionally — neither the
        # ASICODE_HIDE_COST switch nor a large actual_cost can surface it here.
        for val in (None, "1", "0", "true", "false", "yes", "no"):
            monkeypatch.delenv("ASICODE_HIDE_COST", raising=False)
            if val is not None:
                monkeypatch.setenv("ASICODE_HIDE_COST", val)
            recorded = _capture_print(monkeypatch)
            asi._print_session_summary(self._tokens(), asi.time.monotonic())
            assert recorded, (val, "exit banner should print tokens/duration")
            assert "$" not in recorded[0], (val, recorded[0])
            assert "tokens" in recorded[0], (val, recorded[0])

    def test_zero_usage_stays_silent(self, monkeypatch):
        monkeypatch.delenv("ASICODE_HIDE_COST", raising=False)
        recorded = _capture_print(monkeypatch)

        asi._print_session_summary(
            {"prompt": 0, "completion": 0, "cost": 0.0}, asi.time.monotonic()
        )

        assert recorded == []

    def test_duration_and_tokens_still_render(self, monkeypatch):
        # The ambient summary must keep elapsed time + token counts.
        recorded = _capture_print(monkeypatch)
        asi._print_session_summary(self._tokens(), asi.time.monotonic())
        line = recorded[0]
        assert "session" in line
        assert "↑" in line and "↓" in line
        assert "$" not in line
