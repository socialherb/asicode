"""Behavioral guard: the CLI exit banner (``_print_session_summary``) never shows
money.

The exit banner is an *ambient* summary printed automatically on quit, so the
dollar amount is noise there — the cost estimate is not surfaced on any CLI
surface (debug _log only). This pins that invariant: tokens + elapsed time
always render when there is usage, money never does, and a zero-usage session
stays silent.

Complementary guard: the accumulated session cost IS persisted at session end —
to the log file only (logger "asicode.session", which _TerminalInfoFilter
suppresses on the terminal), because the session total would otherwise be lost
with the session (per-turn _log lines cover turns, not the session total).
"""

import logging

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

        asi._print_session_summary({"prompt": 0, "completion": 0, "cost": 0.0}, asi.time.monotonic())

        assert recorded == []

    def test_duration_and_tokens_still_render(self, monkeypatch):
        # The ambient summary must keep elapsed time + token counts.
        recorded = _capture_print(monkeypatch)
        asi._print_session_summary(self._tokens(), asi.time.monotonic())
        line = recorded[0]
        assert "session" in line
        assert "↑" in line and "↓" in line
        assert "$" not in line


class TestSessionCostPersistedToLogFileOnly:
    """Session cost accumulates across turns but was never written anywhere —
    lost with the session. It must be logged (file-only) at session end."""

    def test_cost_logged_at_session_end(self, monkeypatch, caplog):
        _capture_print(monkeypatch)  # terminal surface stays silent on money
        with caplog.at_level(logging.INFO, logger="asicode.session"):
            asi._print_session_summary(
                {"prompt": 1000, "completion": 500, "cost": 0.1234, "actual_cost": 0.1234},
                asi.time.monotonic(),
            )
        cost_records = [r for r in caplog.records if "session cost" in r.getMessage()]
        assert cost_records, "session cost must be persisted at session end"
        assert "$0.1234" in cost_records[0].getMessage()

    def test_actual_billed_differs_suffix(self, monkeypatch, caplog):
        _capture_print(monkeypatch)
        with caplog.at_level(logging.INFO, logger="asicode.session"):
            asi._print_session_summary(
                {"prompt": 1000, "completion": 500, "cost": 0.1500, "actual_cost": 0.0900},
                asi.time.monotonic(),
            )
        msg = next(r.getMessage() for r in caplog.records if "session cost" in r.getMessage())
        assert "estimated $0.1500" in msg
        assert "actual billed $0.0900" in msg

    def test_zero_usage_logs_nothing(self, monkeypatch, caplog):
        _capture_print(monkeypatch)
        with caplog.at_level(logging.INFO, logger="asicode.session"):
            asi._print_session_summary({}, asi.time.monotonic())
        assert not [r for r in caplog.records if "session cost" in r.getMessage()]

    def test_usage_without_cost_data_logs_nothing(self, monkeypatch, caplog):
        # Local models report tokens but no cost — nothing to record.
        _capture_print(monkeypatch)
        with caplog.at_level(logging.INFO, logger="asicode.session"):
            asi._print_session_summary({"prompt": 100, "completion": 50}, asi.time.monotonic())
        assert not [r for r in caplog.records if "session cost" in r.getMessage()]

    def test_terminal_filter_suppresses_cost_log(self):
        # The file-only guarantee: _TerminalInfoFilter must drop the INFO record
        # on the terminal handler while the file handler (unfiltered) keeps it.
        record = logging.LogRecord(
            "asicode.session", logging.INFO, __file__, 1, "session cost: estimated $0.1234", None, None
        )
        assert asi._TerminalInfoFilter().filter(record) is False
        warn_record = logging.LogRecord(
            "asicode.session", logging.WARNING, __file__, 1, "session cost warning", None, None
        )
        assert asi._TerminalInfoFilter().filter(warn_record) is True
