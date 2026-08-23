"""Bounded head+tail accumulation for a subprocess output stream.

SSOT for the "never materialise a command's whole output" discipline. Three
tools spawn a process whose output goes to the model — ``bash``, ``grep`` and
``run_tests`` — and each one independently learned that ``communicate()`` (or a
plain ``lines.append()`` loop) holds everything the command printed before any
budget can apply. Measured on this repo: ``cat`` of a 108 MB log cost +360.6 MB
of peak RSS through ``bash``, ripgrep over the same log cost 522 MB, and 40 MB
of pytest output cost +229 MB through ``run_tests``.

Lives here rather than beside any one caller so the next stream-reading tool
inherits the bound instead of rediscovering it.
"""

from __future__ import annotations

from collections import deque as _deque


class _BoundedCapture:
    """Accumulate one stream's text, keeping only its head and its tail.

    ``communicate()`` materialises everything a command prints before any budget
    can apply, and the budget is ``BASH_OUTPUT_MAX_CHARS`` — ~130 KB. Measured on
    ``cat`` of a 108 MB log: +360.6 MB of peak RSS, 0.24 s, to produce that 130 KB.
    Reading into a bounded head plus a tail ring instead costs +0.4 MB and 0.03 s
    for a byte-identical answer, because the decode and the million allocations
    behind it never happen.

    Head AND tail, not just a tail, because ``_truncate_bash_output`` needs both:
    pytest puts its summary at the end and the failing command at the start. Each
    side retains the FULL ``max_chars``, so anything that truncation would have
    kept is still present and the rendered result is unchanged — a stream is only
    ever elided past 2x the cap, where truncation was certain anyway.

    ``total`` counts every character the command produced, so the truncation
    notice can name the real number rather than what survived.
    """

    __slots__ = ("_cap", "_head", "_head_len", "_tail", "_tail_len", "total")

    def __init__(self, cap: int) -> None:
        self._cap = max(1, cap)
        self._head: list[str] = []
        self._head_len = 0
        self._tail: _deque = _deque()
        self._tail_len = 0
        self.total = 0

    def feed(self, text: str) -> None:
        if not text:
            return
        self.total += len(text)
        _room = self._cap - self._head_len
        if _room > 0:
            self._head.append(text[:_room])
            self._head_len += min(_room, len(text))
            text = text[_room:]
            if not text:
                return
        if len(text) >= self._cap:
            # One feed already covers the whole tail window, so everything
            # before it is outside it. Sliced rather than kept whole: the pump
            # reads at most _PIPE_READ_CHUNK at a time so this is unreachable
            # from there, but a class whose bound depends on how its caller
            # happens to chunk is a bound in name only.
            self._tail.clear()
            self._tail_len = 0
            text = text[-self._cap :]
        self._tail.append(text)
        self._tail_len += len(text)
        # Drop whole chunks from the front while the rest still covers the cap,
        # so the retained tail is never shorter than what truncation will slice.
        while self._tail and self._tail_len - len(self._tail[0]) >= self._cap:
            self._tail_len -= len(self._tail.popleft())

    @property
    def dropped(self) -> int:
        return self.total - self._head_len - self._tail_len

    def text(self) -> str:
        """The retained text, with the gap named where one exists.

        The marker matters only on the background-transition path, which hands
        this string straight to the job's buffer. On the normal path the content
        is always longer than the cap when anything was dropped, so
        ``_truncate_bash_output`` cuts the middle out — marker included — and
        the reader sees exactly one truncation notice.
        """
        _head = "".join(self._head)
        _tail = "".join(self._tail)
        _gap = self.dropped
        if not _gap:
            return _head + _tail
        return f"{_head}\n... [{_gap:,} chars dropped (middle)] ...\n{_tail}"
