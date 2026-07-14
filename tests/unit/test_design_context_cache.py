"""Prompt-cache prefix stability for design-chat context assembly.

Verbatim turn labels must be ABSOLUTE (stable across user turns) so the cached
conversation prefix is reused. A relative "turns-from-now" index re-labels every
prior turn whenever a new turn is appended, breaking the cache prefix.

The current request is identified by a trailing STATIC system marker, not a
"[REQUEST]" content prefix — the prefix mutated the previous user message on
every new turn ([REQUEST] X → (turn N) X), breaking the cache prefix there.
"""
import inspect

from external_llm.agent.context_manager import SessionCompressionContext


class _FakeSession:
    def __init__(self, n_turns: int, compressed_up_to: int = 0, summary: str = ""):
        roles = ["user", "assistant"]
        self.turns = [
            {"role": roles[i % 2], "content": f"content-{i}", "model": ""}
            for i in range(n_turns)
        ]
        self.compressed_up_to = compressed_up_to
        self.compressed_summary = summary
        self.session_id = "test"


def _labeled_history(msgs):
    return [
        m["content"]
        for m in msgs
        if m["role"] in ("user", "assistant", "tool")
        and m["content"].startswith("(turn ")
    ]


def _build(session):
    ctx = SessionCompressionContext("/tmp/nonexistent_repo_for_test")
    return ctx.build_context_messages(session, skip_core_prompt=True, mode="code")


class TestVerbatimPrefixStability:
    def test_past_turn_labels_stable_as_conversation_grows(self):
        # The same underlying turns must keep identical labels when more turns
        # are appended — otherwise the cache prefix breaks every user turn.
        # This includes the current request: it gets a normal "(turn N)" label
        # so its bytes never change when the next turn is appended.
        # Only USER turns carry labels (assistant turns are unlabelled so the
        # model does not mimic the "(turn N)" prefix in its own responses), so
        # the odd-indexed assistant turns (content-1, content-3) do not appear.
        m3 = _labeled_history(_build(_FakeSession(3)))
        m5 = _labeled_history(_build(_FakeSession(5)))
        assert m3 == [
            "(turn 1) content-0",
            "(turn 3) content-2",
        ]
        assert m5[: len(m3)] == m3  # prefix preserved byte-for-byte

    def test_labels_are_absolute_not_relative(self):
        # Absolute numbering counts from the session start, not from "now".
        # Assistant turns (content-1, content-3) are unlabelled by design.
        m5 = _labeled_history(_build(_FakeSession(5)))
        assert m5 == [
            "(turn 1) content-0",
            "(turn 3) content-2",
            "(turn 5) content-4",
        ]

    def test_assistant_turns_are_not_labelled(self):
        # Assistant turns must NOT carry a "(turn N)" prefix: when they do, the
        # model copies the surface pattern and prepends "(turn N)" to its own
        # generated responses. User/tool turns keep the label as anchors.
        msgs = _build(_FakeSession(5))
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst, "expected assistant turns"
        assert all(not c.startswith("(turn ") for c in asst)
        # The assistant content itself is preserved verbatim.
        assert "content-1" in asst
        assert "content-3" in asst

    def test_absolute_index_offset_by_compressed_up_to(self):
        # When old turns are folded into the summary, verbatim numbering
        # continues from compressed_up_to (no reuse of turn 1..N labels).
        s = _FakeSession(5, compressed_up_to=2, summary="prior summary")
        labels = _labeled_history(_build(s))
        # verbatim window = turns[2:], absolute numbers start at 3
        assert labels[0].startswith("(turn 3) ")

    def test_recent_header_has_no_volatile_count(self):
        # The RECENT CONVERSATION header must not embed a turn count (it would
        # change every turn and break the prefix at that point).
        msgs = _build(_FakeSession(4))
        headers = [m["content"] for m in msgs if "RECENT CONVERSATION" in m["content"]]
        assert headers, "header missing"
        for h in headers:
            assert not any(ch.isdigit() for ch in h), f"volatile count in header: {h!r}"


class TestCurrentRequestMarker:
    def test_current_request_keeps_stable_turn_label(self):
        msgs = _build(_FakeSession(3))
        conv = [m for m in msgs if m["role"] in ("user", "assistant")]
        assert conv[-1]["content"] == "(turn 3) content-2"

    def test_marker_is_trailing_static_system_message(self):
        # Marker must be byte-identical across turns (no turn number) — on
        # Anthropic it is hoisted into the system block, which must stay stable.
        m3 = [m["content"] for m in _build(_FakeSession(3))
              if "CURRENT REQUEST" in m["content"]]
        m5 = [m["content"] for m in _build(_FakeSession(5))
              if "CURRENT REQUEST" in m["content"]]
        assert len(m3) == 1
        assert m3 == m5
        assert not any(ch.isdigit() for ch in m3[0])

    def test_marker_follows_the_request_message(self):
        msgs = _build(_FakeSession(3))
        contents = [m["content"] for m in msgs]
        i_req = contents.index("(turn 3) content-2")
        i_marker = next(i for i, c in enumerate(contents) if "CURRENT REQUEST" in c)
        assert i_marker > i_req

    def test_no_marker_when_last_turn_is_assistant(self):
        msgs = _build(_FakeSession(4))  # ends with an assistant turn
        assert not any("CURRENT REQUEST" in m["content"] for m in msgs)

    def test_no_request_prefix_anywhere(self):
        msgs = _build(_FakeSession(5))
        assert not any(m["content"].startswith("[REQUEST]") for m in msgs)


class TestPreservedTurns:
    def test_preserve_turn_survives_compression_pointer(self):
        # A preserve=True turn behind compressed_up_to is excluded from the LLM
        # summary, so the builder must re-insert it verbatim — otherwise the
        # "preserve" flag silently DELETES the turn from context.
        s = _FakeSession(6, compressed_up_to=4, summary="prior summary")
        s.turns[1]["preserve"] = True
        msgs = _build(s)
        contents = [m["content"] for m in msgs]
        assert "(turn 2) content-1" in contents
        # Chronological: preserved old turn appears before the verbatim window
        assert contents.index("(turn 2) content-1") < contents.index("(turn 5) content-4")

    def test_preserved_labels_stable_across_rebuilds(self):
        s = _FakeSession(6, compressed_up_to=4, summary="prior summary")
        s.turns[1]["preserve"] = True
        first = [m["content"] for m in _build(s) if m["content"].startswith("(turn 2)")]
        s.turns.append({"role": "user", "content": "next", "model": ""})
        second = [m["content"] for m in _build(s) if m["content"].startswith("(turn 2)")]
        assert first == second

    def test_no_preserved_turns_no_extra_messages(self):
        s = _FakeSession(6, compressed_up_to=4, summary="prior summary")
        msgs = _build(s)
        labels = _labeled_history(msgs)
        assert labels[0].startswith("(turn 5) ")  # nothing re-inserted


class TestTopBlockOrdering:
    def test_insights_below_static_blocks(self, tmp_path):
        # Insights grow when save_insight runs mid-session; any change
        # invalidates the prefix below it. The static blocks (repo root,
        # project.md) must therefore sit ABOVE the insights block.
        # Repo root comes FIRST: it is shared with general mode, so the
        # [core][repo root] prefix stays cached across /code ↔ /general.
        d = tmp_path / ".asicode"
        d.mkdir()
        (d / "project.md").write_text("project ctx body", encoding="utf-8")
        (d / "design_insights.md").write_text("insight body", encoding="utf-8")
        ctx = SessionCompressionContext(str(tmp_path))
        msgs = ctx.build_context_messages(
            _FakeSession(3), skip_core_prompt=True, mode="code",
        )
        contents = [m["content"] for m in msgs]
        i_proj = next(i for i, c in enumerate(contents) if "Project Context" in c)
        i_repo = next(i for i, c in enumerate(contents) if "CURRENT REPOSITORY" in c)
        i_ins = next(i for i, c in enumerate(contents) if "DESIGN INSIGHTS" in c)
        i_summary_or_turn = next(
            i for i, c in enumerate(contents) if c.startswith("(turn ")
        )
        assert i_repo < i_proj < i_ins < i_summary_or_turn

    def test_repo_root_present_in_general_mode(self, tmp_path):
        # General mode still exposes `bash`; without the repo root block the
        # LLM hallucinates working-dir paths (observed: wrong repo path on a
        # commit request in General mode).
        ctx = SessionCompressionContext(str(tmp_path))
        msgs = ctx.build_context_messages(
            _FakeSession(3), skip_core_prompt=True, mode="general",
        )
        contents = [m["content"] for m in msgs]
        assert any("CURRENT REPOSITORY" in c for c in contents)
        # code-only blocks must stay excluded from general mode
        assert not any("Project Context" in c for c in contents)
        assert not any("DESIGN INSIGHTS" in c for c in contents)


class TestInsightsLayer3CachePosition:
    """Layer 3 promotion is turn-volatile: it must be injected AFTER the cached
    verbatim-turns prefix (late position), never in the early 0c insights block.

    Regression guard: an earlier design folded the promoted body into the 0c
    insights block, which invalidated the prompt cache (summary + all turns) on
    every turn whenever the promoted set shifted with the recent user turns.
    """
    _BODY = (
        "FileLockManager uses WeakValueDictionary for lock identity "
        "cross-request UNIQUE-TAIL-MARKER-ZZZ"
    )

    def _repo_with_archive(self, tmp_path):
        d = tmp_path / ".asicode"
        d.mkdir(exist_ok=True)
        (d / "design_insights.md").write_text("active insight body\n", encoding="utf-8")
        (d / "design_insights_archive.md").write_text(
            "### [architecture] 2026-06-30 13:18\n" + self._BODY + "\n\n",
            encoding="utf-8",
        )
        return str(tmp_path)

    def _session_with_user(self, user_text):
        # 3 turns (u/a/u) → last is a user turn, so [CURRENT REQUEST] appears.
        s = _FakeSession(3)
        s.turns[0]["content"] = user_text  # user turn (in last-8 task window)
        s.turns[2]["content"] = user_text  # user turn (the current request)
        return s

    def _build(self, repo, user_text):
        ctx = SessionCompressionContext(repo)
        return ctx.build_context_messages(
            self._session_with_user(user_text),
            skip_core_prompt=True, mode="code",
        )

    def test_promoted_block_not_in_early_insights_block(self, tmp_path):
        repo = self._repo_with_archive(tmp_path)
        msgs = self._build(repo, "FileLockManager lock identity WeakValueDictionary")
        contents = [m["content"] for m in msgs]
        # The EARLY 0c insights block carries active + Layer 2 index only — NOT
        # the Layer 3 promoted body.
        i_ins = next(i for i, c in enumerate(contents) if "DESIGN INSIGHTS" in c)
        assert "PROMOTED FROM ARCHIVE" not in contents[i_ins]
        assert "UNIQUE-TAIL-MARKER-ZZZ" not in contents[i_ins]
        # The promoted body exists as a SEPARATE late system message.
        assert any("PROMOTED FROM ARCHIVE" in c for c in contents)

    def test_promoted_block_after_all_verbatim_turns(self, tmp_path):
        repo = self._repo_with_archive(tmp_path)
        msgs = self._build(repo, "FileLockManager lock identity WeakValueDictionary")
        contents = [m["content"] for m in msgs]
        i_last_turn = max(i for i, c in enumerate(contents) if c.startswith("(turn "))
        i_promoted = next(i for i, c in enumerate(contents) if "PROMOTED FROM ARCHIVE" in c)
        # Late position → only the volatile tail is uncached; summary + turns cached.
        assert i_promoted > i_last_turn

    def test_no_promoted_message_when_irrelevant(self, tmp_path):
        # When the current task matches no archived entry, NOTHING is injected —
        # the message list is byte-identical to the pre-feature shape, so the
        # cache is fully preserved on irrelevant turns (zero-cost guarantee).
        repo = self._repo_with_archive(tmp_path)
        msgs = self._build(repo, "zzzz totally unrelated qwx")
        assert not any("PROMOTED FROM ARCHIVE" in m["content"] for m in msgs)

    def test_prefix_stable_across_turns_despite_promotion_flip(self, tmp_path):
        # The cached history prefix (0c insights + turns 1..N) must be
        # byte-identical across the turn-N → turn-(N+1) boundary even though
        # promotion status FLIPS between the two turns. This is the real
        # regression: an earlier design folded promotion into 0c, so a flip
        # mutated the cached prefix and invalidated summary + all turns.
        repo = self._repo_with_archive(tmp_path)
        match = "FileLockManager lock identity WeakValueDictionary"
        filler = "unrelated filler content qwx"

        # Turn N: the matching turns are the only user turns in the (last-8)
        # task window → promotion fires.
        s_n = _FakeSession(3)
        s_n.turns[0]["content"] = match
        s_n.turns[2]["content"] = match
        msgs_n = SessionCompressionContext(repo).build_context_messages(
            s_n, skip_core_prompt=True, mode="code")

        # Turn N+1: same first 3 turns, then 8 more USER turns of filler push
        # the matching turns OUT of the last-8 user window → task_query no
        # longer overlaps the archive entry → no promotion. (19 turns: user
        # turns at even indices 0,2 = match; 4,6,...,18 = filler.)
        s_n1 = _FakeSession(19)
        s_n1.turns[0]["content"] = match
        s_n1.turns[2]["content"] = match
        for _i in range(4, 19, 2):
            s_n1.turns[_i]["content"] = filler
        msgs_n1 = SessionCompressionContext(repo).build_context_messages(
            s_n1, skip_core_prompt=True, mode="code")

        # Promotion flipped: turn N promotes, turn N+1 does not.
        assert any("PROMOTED FROM ARCHIVE" in m["content"] for m in msgs_n)
        assert not any("PROMOTED FROM ARCHIVE" in m["content"] for m in msgs_n1)

        # Yet the cached prefix through "(turn 3) <match>" is byte-identical —
        # the promotion flip mutated nothing in 0c / summary / early turns.
        anchor = "(turn 3) " + match
        c_n = [m["content"] for m in msgs_n]
        c_n1 = [m["content"] for m in msgs_n1]
        assert c_n[: c_n.index(anchor) + 1] == c_n1[: c_n1.index(anchor) + 1]


class TestProcessTemperature:
    def test_temperature_fixed_per_process(self):
        import external_llm.agent.design_chat_loop as dcl
        assert 0.0 <= dcl._PROCESS_TEMPERATURE <= 0.3
        # No call site may re-randomize per call.
        src = inspect.getsource(dcl)
        assert "temperature=random.uniform" not in src
