"""Single source of truth for BM25 scoring (agent-wide).

Consolidates what used to be five parallel copies of the same formula:
``rag_searcher._bm25_score`` (the reference implementation), the inline
idf_map/tf_norm literals in ``insights_manager._promote_matching`` (the
binary tf=1 variant), the per-doc ranking loop in
``design_chat_loop._DesignChatIndex.search``, and the copy-pasted
small-corpus setup twins in ``symbol_search`` and ``read_tools`` (per-doc
counters + document frequency + average length + scores).

Numeric contract
----------------
Every helper is BIT-IDENTICAL to the formula it replaced — same operation
order, hence exactly equal floats — sealed by ``tests/unit/agent/
test_bm25_core.py``, which transcribes the pre-consolidation formulas
verbatim and demands ``==`` (not ``isclose``). Rankings feed persistent
caches (vector-cache keys, promote-from-archive ordering), so even an
epsilon drift would silently reshuffle them.

Import rule
-----------
Stdlib only. This module sits at the bottom of the agent import graph
(everything may import it; it imports nothing local), which is what makes
it a safe shared home — sealed by the stdlib-only AST gate in the tests.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

# BM25 tuning. Previously ``rag_searcher._K1``/``_B`` plus a matching
# literal pair in insights_manager kept in sync by a "(matches
# rag_searcher)" comment — one definition now.
K1 = 1.5
B = 0.75


def bm25_idf(n_docs: int, df: int) -> float:
    """Lucene-style smoothed BM25 IDF for one term."""
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def bm25_idf_map(query_tokens: Iterable[str], df: Mapping[str, int], n_docs: int) -> dict[str, float]:
    """Per-token IDF over the query tokens (dict form).

    IDF depends only on the term and corpus stats (df, n_docs) — never on
    the document — so any per-document scoring loop must precompute this
    once per query instead of recomputing ``math.log`` per doc x term.
    """
    return {qt: bm25_idf(n_docs, df.get(qt, 0)) for qt in query_tokens}


def bm25_idf_pairs(query_tokens: Sequence[str], df: Mapping[str, int], n_docs: int) -> list[tuple[str, float]]:
    """``(token, idf)`` pairs preserving query-token MULTIPLICITY.

    ``CodeTokenizer.tokenize`` does not deduplicate and the reference
    ``_bm25_score`` loop iterated the raw token list, so a duplicated query
    term contributed its weight twice. The fast path must keep that: pairs
    are materialised from the original sequence, not from the deduped dict.
    """
    idf_map = bm25_idf_map(query_tokens, df, n_docs)
    return [(qt, idf_map[qt]) for qt in query_tokens]


def bm25_tf_norm(tf: int, doc_len: int, avgdl: float) -> float:
    """BM25 term-frequency normalisation (length-normalised saturation)."""
    return tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len / avgdl))


def bm25_score_pairs(
    q_pairs: Sequence[tuple[str, float]],
    doc_token_counts: Mapping[str, int],
    doc_len: int,
    avgdl: float,
) -> float:
    """Fast path: IDF precomputed per query, per-doc denominator hoisted.

    Per document x term this costs one dict get plus float mul/add — the
    ``math.log`` and the two ``df`` lookups moved to the once-per-query
    ``bm25_idf_pairs``. For a caller holding a lock around the doc loop
    (``rag_searcher._bm25_search`` takes ``_index_lock``) this directly
    shortens the lock-held section.
    """
    if doc_len == 0 or avgdl == 0:
        return 0.0
    k_denom = K1 * (1 - B + B * doc_len / avgdl)
    score = 0.0
    for qt, idf in q_pairs:
        tf = doc_token_counts.get(qt, 0)
        if tf:
            score += idf * (tf * (K1 + 1) / (tf + k_denom))
    return score


def bm25_score(
    query_tokens: Sequence[str],
    doc_token_counts: Mapping[str, int],
    doc_len: int,
    df: Mapping[str, int],
    n_docs: int,
    avgdl: float,
) -> float:
    """Reference BM25 score for one document (compat signature).

    Same parameters and semantics as the old ``rag_searcher._bm25_score``;
    implemented via the precomputed path so there is exactly one loop body.
    """
    return bm25_score_pairs(
        bm25_idf_pairs(query_tokens, df, n_docs),
        doc_token_counts,
        doc_len,
        avgdl,
    )


def bm25_rank(query_tokens: Sequence[str], tokenized_docs: Sequence[Sequence[str]]) -> list[float]:
    """Score a small in-memory corpus; one score per doc, input order.

    Owns the setup that ``symbol_search`` and ``read_tools`` each carried a
    copy of: per-doc token counters, doc lengths, average length, per-query
    -token document frequency, then the fast-path scoring loop. Returns
    ``[]`` for an empty corpus (the twins would have divided by zero).
    """
    if not tokenized_docs:
        return []
    doc_tc = [dict(Counter(t)) for t in tokenized_docs]
    doc_lens = [len(t) for t in tokenized_docs]
    n_docs = len(tokenized_docs)
    avgdl = sum(doc_lens) / n_docs
    df: dict[str, int] = {}
    for qt in query_tokens:
        df[qt] = sum(1 for tc in doc_tc if qt in tc)
    q_pairs = bm25_idf_pairs(query_tokens, df, n_docs)
    return [bm25_score_pairs(q_pairs, doc_tc[i], doc_lens[i], avgdl) for i in range(n_docs)]
