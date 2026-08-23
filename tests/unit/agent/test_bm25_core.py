"""Bit-identity + behaviour contract for the consolidated BM25 core.

P9-1 (turn 10150): ``agent/bm25.py`` is now the single source of the BM25
formula that previously lived as FIVE parallel copies —
``rag_searcher._bm25_score`` (reference), the inline idf_map/tf_norm in
``insights_manager._promote_matching`` (binary tf=1 variant), the
``design_chat_loop`` ranking loop, and the copy-pasted setup twins in
``symbol_search`` / ``read_tools``. Twin drift is an observed bug class in
this repo (the cancel-scope round found call_graph/rag_searcher twins
fixed on one side only).

Because BM25 rankings feed persistent state (vector-cache ordering,
promote-from-archive selection, result caps), even an epsilon drift would
silently reshuffle them — so these tests transcribe the PRE-consolidation
formulas VERBATIM (from commit 0da2c67e) and demand EXACT ``==`` float
equality, never ``isclose``.
"""

from __future__ import annotations

import ast
import math
import random
from collections import Counter
from pathlib import Path

from external_llm.agent.bm25 import (
    K1,
    B,
    bm25_idf_map,
    bm25_idf_pairs,
    bm25_rank,
    bm25_score,
    bm25_score_pairs,
    bm25_tf_norm,
)

MODULE_PATH = Path("external_llm/agent/bm25.py")


# ── Pre-consolidation formulas, transcribed verbatim ─────────────────────────


def _old_reference_score(query_tokens, doc_token_counts, doc_len, df, n_docs, avgdl):
    """rag_searcher._bm25_score @ 0da2c67e — constants inlined, ops unchanged."""
    _K1 = 1.5
    _B = 0.75
    if doc_len == 0 or avgdl == 0:
        return 0.0
    score = 0.0
    for qt in query_tokens:
        tf = doc_token_counts.get(qt, 0)
        if tf == 0:
            continue
        idf = math.log((n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0)
        tf_norm = tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * doc_len / avgdl))
        score += idf * tf_norm
    return score


def _old_insights_idf_map(qset, df, n_docs):
    """insights_manager idf_map comprehension @ 0da2c67e."""
    return {qt: math.log((n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0) for qt in qset}


def _old_insights_tf_norm(doc_len, avgdl):
    """insights_manager binary tf=1 normaliser @ 0da2c67e."""
    _K1, _B = 1.5, 0.75
    return 1.0 * (_K1 + 1) / (1.0 + _K1 * (1 - _B + _B * doc_len / avgdl))


def _old_twin_scores(query_tokens, tokenized_docs):
    """The symbol_search/read_tools setup twin @ 0da2c67e (verbatim shape)."""
    _doc_tc: list[dict[str, int]] = [dict(Counter(t)) for t in tokenized_docs]
    _doc_lens = [len(t) for t in tokenized_docs]
    _n = len(tokenized_docs)
    _avgdl = sum(_doc_lens) / _n
    _df: dict[str, int] = {}
    for qt in query_tokens:
        _df[qt] = sum(1 for tc in _doc_tc if qt in tc)
    return [_old_reference_score(query_tokens, _doc_tc[i], _doc_lens[i], _df, _n, _avgdl) for i in range(_n)]


# ── Randomised bit-identity ──────────────────────────────────────────────────


def _random_corpus(rng: random.Random):
    vocab = [f"tok{i}" for i in range(12)]
    n_docs = rng.randint(1, 25)
    docs = []
    for _ in range(n_docs):
        ln = rng.choice([0, 1, 2, 5, 10, 30])
        docs.append([rng.choice(vocab) for _ in range(ln)])
    # Duplicate query tokens are possible (tokenize does not dedupe).
    q = [rng.choice([*vocab, "unseen"]) for _ in range(rng.randint(1, 8))]
    return docs, q


def test_bm25_score_bit_identical_to_pre_consolidation_reference():
    rng = random.Random(42)
    for _ in range(300):
        docs, q = _random_corpus(rng)
        n_docs = len(docs)
        df_full = {}
        for t in set(q):
            df_full[t] = rng.randint(0, n_docs)  # includes 0 and n_docs edges
        df = {k: v for k, v in df_full.items() if rng.random() < 0.8}  # missing keys
        for doc in docs:
            tc = dict(Counter(doc))
            avgdl = rng.choice([1.0, 3.7, 25.0])
            got = bm25_score(q, tc, len(doc), df, n_docs, avgdl)
            want = _old_reference_score(q, tc, len(doc), df, n_docs, avgdl)
            assert got == want, (q, doc, df, avgdl, got, want)


def test_fast_path_bit_identical_to_reference():
    rng = random.Random(7)
    for _ in range(300):
        docs, q = _random_corpus(rng)
        n_docs = len(docs)
        df = {}
        for t in set(q):
            df[t] = rng.randint(0, n_docs)
        pairs = bm25_idf_pairs(q, df, n_docs)
        # Pair multiplicity must mirror the raw query sequence.
        assert [p[0] for p in pairs] == q
        for doc in docs:
            tc = dict(Counter(doc))
            avgdl = rng.choice([1.0, 3.7, 25.0])
            fast = bm25_score_pairs(pairs, tc, len(doc), avgdl)
            ref = _old_reference_score(q, tc, len(doc), df, n_docs, avgdl)
            assert fast == ref, (q, doc, avgdl, fast, ref)


def test_zero_guards_preserved():
    tc = {"a": 3}
    assert bm25_score(["a"], tc, 0, {"a": 1}, 5, 4.0) == 0.0  # doc_len == 0
    assert bm25_score(["a"], tc, 7, {"a": 1}, 5, 0.0) == 0.0  # avgdl == 0
    assert bm25_score_pairs([("a", 1.2)], tc, 0, 0.0) == 0.0


def test_duplicate_query_token_contributes_per_occurrence():
    # ["a","a","b"] must score exactly reference — which double-counts "a"
    # because the old loop iterated the raw (un-deduped) token list.
    tc = {"a": 2, "b": 1}
    df = {"a": 3, "b": 5}
    assert bm25_score(["a", "a", "b"], tc, 6, df, 10, 4.0) == _old_reference_score(["a", "a", "b"], tc, 6, df, 10, 4.0)
    doubled = bm25_score(["a", "a", "b"], tc, 6, df, 10, 4.0)
    single = bm25_score(["a", "b"], tc, 6, df, 10, 4.0)
    assert doubled > single  # the duplicate visibly adds weight


def test_bm25_rank_bit_identical_to_twin_setup():
    rng = random.Random(99)
    for _ in range(200):
        docs, q = _random_corpus(rng)
        assert bm25_rank(q, docs) == _old_twin_scores(q, docs)


def test_bm25_rank_empty_corpus_returns_empty():
    # The twins would have raised ZeroDivisionError on the avgdl divide;
    # [] is the strict-improvement contract.
    assert bm25_rank(["a"], []) == []


def test_insights_variants_bit_identical():
    rng = random.Random(1234)
    for _ in range(200):
        qset = {f"t{i}" for i in rng.sample(range(15), rng.randint(1, 6))}
        n_docs = rng.randint(1, 40)
        df = {t: rng.randint(0, n_docs) for t in qset if rng.random() < 0.8}
        assert bm25_idf_map(qset, df, n_docs) == _old_insights_idf_map(qset, df, n_docs)
        doc_len = rng.randint(1, 60)
        avgdl = rng.choice([1.0, 4.2, 33.0])
        assert bm25_tf_norm(1, doc_len, avgdl) == _old_insights_tf_norm(doc_len, avgdl)


def test_module_imports_stdlib_only():
    """bm25.py must stay at the bottom of the agent import graph.

    It is the shared home everyone imports; importing anything local would
    create cycles and re-couple the five former copies.
    """
    allowed = {"__future__", "math", "collections", "typing"}
    tree = ast.parse(MODULE_PATH.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.append(node.module)
    assert found, "imports vanished — gate is stale"
    for mod in found:
        assert mod.split(".")[0] in allowed, f"non-stdlib import: {mod}"


def test_constants_unchanged():
    # Tuning constants are part of the ranking contract with persisted
    # caches — changing them reshuffles every ranking at once.
    assert (K1, B) == (1.5, 0.75)
