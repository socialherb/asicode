"""ident_overlap role gate: coincidental-structure pairs are dropped.

Structural metrics alpha-normalise identifiers away, so two functions from
unrelated domains can score above min_similarity on shape alone (observed:
_build_interrupt_note ↔ _build_agent_interrupt_note, _strip_ansi ↔ _plain
in asi.py).  The ident_tokens Jaccard gate drops those pairs while
keeping genuine copy-paste duplicates — including ones whose parameters
were renamed at copy time.
"""
from __future__ import annotations

import textwrap

from external_llm.analysis.ast_similarity_scanner import (
    scan_similarity_candidates,
)


def _write(tmp_path, name: str, src: str) -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return name


# 같은 제어 흐름(가드 → 루프 → 조건 누적 → 반환)이지만 도메인이 전혀 다른 두 함수.
# 구조 점수는 min_similarity를 넘되 ident_overlap은 바닥이어야 한다.
_COINCIDENTAL = """
    def collect_retry_hosts(hosts, attempts):
        if not hosts:
            return []
        failed = []
        for host in hosts:
            status = ping_host(host, attempts)
            if status.timed_out:
                failed.append(host.address)
        return failed

    def collect_stale_caches(caches, max_age):
        if not caches:
            return []
        stale = []
        for cache in caches:
            entry = inspect_cache(cache, max_age)
            if entry.expired:
                stale.append(cache.key)
        return stale
"""

# 전형적 복붙 중복 — 파라미터/지역변수를 개명했지만 같은 함수들을 호출하고
# 같은 문자열 상수를 쓴다.  게이트를 통과해야 한다.
_COPY_PASTE = """
    def load_user_config(path, defaults):
        if not path:
            return dict(defaults)
        try:
            with open(path) as f:
                data = json.load(f)
        except OSError as e:
            logger.warning("config load failed: %s", e)
            return dict(defaults)
        merged = dict(defaults)
        merged.update(data)
        return merged

    def load_project_config(cfg_path, base):
        if not cfg_path:
            return dict(base)
        try:
            with open(cfg_path) as f:
                payload = json.load(f)
        except OSError as err:
            logger.warning("config load failed: %s", err)
            return dict(base)
        merged = dict(base)
        merged.update(payload)
        return merged
"""


def test_coincidental_structure_pair_is_dropped(tmp_path):
    fname = _write(tmp_path, "coincidental.py", _COINCIDENTAL)
    cands = scan_similarity_candidates(str(tmp_path), [fname])
    pairs = {frozenset([c.symbol_a, c.symbol_b]) for c in cands}
    assert frozenset(["collect_retry_hosts", "collect_stale_caches"]) not in pairs


def test_copy_paste_duplicate_survives_gate(tmp_path):
    fname = _write(tmp_path, "dup.py", _COPY_PASTE)
    cands = scan_similarity_candidates(str(tmp_path), [fname])
    pairs = {frozenset([c.symbol_a, c.symbol_b]) for c in cands}
    assert frozenset(["load_user_config", "load_project_config"]) in pairs
    c = next(c for c in cands
             if {c.symbol_a, c.symbol_b} == {"load_user_config", "load_project_config"})
    assert c.shadow_overlaps["ident_overlap"] >= 0.25


def test_forced_pair_bypasses_gate(tmp_path):
    """사용자가 명시한 forced pair는 게이트와 무관하게 항상 결과에 포함된다."""
    fname = _write(tmp_path, "coincidental.py", _COINCIDENTAL)
    cands = scan_similarity_candidates(
        str(tmp_path), [fname],
        forced_pairs=[("collect_retry_hosts", "collect_stale_caches")],
    )
    forced = [c for c in cands if c.forced]
    assert len(forced) == 1
    assert {forced[0].symbol_a, forced[0].symbol_b} == {
        "collect_retry_hosts", "collect_stale_caches"}
    # 관측 신호로 ident_overlap이 기록된다 (게이트 미적용이어도)
    assert "ident_overlap" in forced[0].shadow_overlaps


def test_candidate_pairs_report_ident_overlap_signal(tmp_path):
    fname = _write(tmp_path, "dup.py", _COPY_PASTE)
    cands = scan_similarity_candidates(str(tmp_path), [fname])
    assert cands
    assert all("ident_overlap" in c.shadow_overlaps for c in cands)
