"""Regression: closed-thinking provider doc anchors must use symbol names, not
line numbers — and model_registry must point ANTHROPIC_DEFAULT at its SSOT.

``reasoning_callback`` is intentionally NOT consumed by the closed-thinking
providers (``anthropic_client.py``, ``openai_client.py``): only ``providers.py``
clients (DeepSeek / ZAI-GLM / Ollama) consume it. The invariant is documented
in prose notes that cross-reference each other across the two provider modules
and between each module's ``chat()`` and its streaming/with-tools methods.

Line-number references (``chat() L344 note``, ``openai_client.py:313,472``)
rot on every edit — a maintainer following ``:689`` lands on an unrelated line
(``cache_control``, retry logic) and may mis-read the invariant as relaxed,
then "fix" it by forwarding ``reasoning_callback`` and breaking parity between
the two closed-thinking providers. Symbol-name anchors (``chat()``) are stable.

Likewise, ``model_registry.py`` historically pointed the ``ANTHROPIC_DEFAULT``
comment at ``anthropic_client.py``, which only *consumes*
``_cfg.tokens.ANTHROPIC_DEFAULT``. The real SSOT is
``agent/config/thresholds.py``.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ANTHROPIC = _ROOT / "external_llm" / "anthropic_client.py"
_OPENAI = _ROOT / "external_llm" / "openai_client.py"
_MODEL_REGISTRY = _ROOT / "external_llm" / "model_registry.py"

# Rotting line-number reference forms caught by this gate:
#   "openai_client.py:313,472"   "anthropic_client.py:344"   "chat() L344 note"
_LINE_REF = re.compile(r"\bL\d+\b|\.py:\d+")


def _comment_blocks(path: Path) -> list[str]:
    """Each maximal run of consecutive ``#`` comment lines, joined into one string."""
    blocks: list[str] = []
    cur: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("#"):
            cur.append(s)
        else:
            if cur:
                blocks.append(" ".join(cur))
            cur = []
    if cur:
        blocks.append(" ".join(cur))
    return blocks


def _reasoning_blocks(path: Path) -> list[str]:
    return [b for b in _comment_blocks(path) if "reasoning_callback" in b]


def test_closed_thinking_notes_have_no_line_number_refs():
    """No reasoning_callback note may use a line-number cross-reference."""
    for path in (_ANTHROPIC, _OPENAI):
        blocks = _reasoning_blocks(path)
        assert blocks, f"{path.name}: no reasoning_callback comment block found"
        for b in blocks:
            assert not _LINE_REF.search(b), (
                f"{path.name}: reasoning_callback note uses a rotting "
                f"line-number reference; use a symbol-name anchor "
                f"(e.g. 'chat()') instead:\n    {b}"
            )


def test_canonical_notes_cross_reference_chat_by_symbol():
    """Each provider's reasoning_callback note must anchor on the other's chat()."""
    ant = _reasoning_blocks(_ANTHROPIC)
    oai = _reasoning_blocks(_OPENAI)
    # Anthropic's canonical note mirrors openai's chat() by symbol name.
    assert any("chat()" in b for b in ant), (
        "anthropic_client.py: reasoning_callback note must reference chat() "
        "(its own canonical note) rather than a sibling line number"
    )
    assert any("chat()" in b for b in oai), (
        "openai_client.py: reasoning_callback note must reference chat() "
        "(its own canonical note) rather than a sibling line number"
    )


def test_model_registry_anthropic_default_points_at_ssot():
    """model_registry.py ANTHROPIC_DEFAULT comment must cite the thresholds.py SSOT."""
    hits = [
        line
        for line in _MODEL_REGISTRY.read_text().splitlines()
        if "ANTHROPIC_DEFAULT" in line and line.strip().startswith("#")
    ]
    assert hits, "model_registry.py: ANTHROPIC_DEFAULT comment moved or removed"
    for line in hits:
        assert "thresholds.py" in line, (
            "model_registry.py ANTHROPIC_DEFAULT comment must point at the "
            "agent/config/thresholds.py SSOT (anthropic_client.py only *consumes* "
            "_cfg.tokens.ANTHROPIC_DEFAULT):\n    " + line
        )
