"""Request semantic analysis — extract code identifiers from edit requests.

Thin normalizer: extracts quoted/backtick code identifiers and bare
snake_case identifiers. No action/role keyword detection.

This module exists for backward compatibility; the semantic scoring
dimension has been removed from the relevance scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestSemantics:
    """Normalized semantic interpretation of an edit request.

    Note: actions and target_roles are no longer populated by
    analyze_request(). They are kept for backward compatibility
    with callers that may read these fields.
    """

    actions: set[str] = field(default_factory=set)
    """Deprecated — always empty. Previously held action classes."""

    target_roles: set[str] = field(default_factory=set)
    """Deprecated — always empty. Previously held target code roles."""

    code_identifiers: set[str] = field(default_factory=set)
    """Quoted or backtick-delimited code identifiers explicitly mentioned."""


def analyze_request(request: str) -> RequestSemantics:
    """Analyze an edit request for code identifiers.

    Extracts quoted/backtick-delimited code identifiers and bare
    snake_case identifiers from the request text. No action/role
    detection — the semantic dimension has been removed.

    Args:
        request: The edit request text.

    Returns:
        RequestSemantics with code_identifiers populated.
    """
    sem = RequestSemantics()
    sem.code_identifiers = _extract_code_identifiers(request)
    return sem


def _extract_code_identifiers(request: str) -> set[str]:
    """Extract code identifiers from the request.

    Sources:
    1. Quoted/backtick-delimited: `code_token`, 'code_token', "code_token"
    2. Snake_case or dotted identifiers in natural language context
    """
    identifiers: set[str] = set()

    # Quoted/backtick identifiers: scan for quote pairs, extract content
    _i = 0
    while _i < len(request):
        _ch = request[_i]
        if _ch in "`'\"":
            _end = request.find(_ch, _i + 1)
            if _end != -1:
                _content = request[_i + 1:_end]
                if _content and (_content[0].isidentifier() or _content[0] == '_'):
                    identifiers.add(_content)
                _i = _end + 1
                continue
        _i += 1

    # Bare snake_case identifiers (must have underscore or dot to distinguish from normal words)
    _cur = []
    for _ch in request:
        if _ch.isalnum() or _ch in '_.':
            _cur.append(_ch)
        else:
            _word = ''.join(_cur)
            if len(_word) >= 3 and ('_' in _word or '.' in _word) and (_word[0].isidentifier() or _word[0] == '_'):
                identifiers.add(_word)
            _cur = []
    _word = ''.join(_cur)
    if len(_word) >= 3 and ('_' in _word or '.' in _word) and (_word[0].isidentifier() or _word[0] == '_'):
        identifiers.add(_word)

    return identifiers
