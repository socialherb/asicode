# 개선 제안서 — 검증 완료 (15항목)

**생성일**: 2026-07-16  
**범위**: 전체 코드베이스 (`asi.py` + `external_llm/` + `tests/`)  
**현재 상태**: `226 passed (analysis), ruff clean`  
**검증**: 2026-07-16 독립 검증 완료 (기각 4, 조건부 2, 채택 9 → 11항목)  
**구현 진행**: Phase 1 ✅ · Phase 2 ✅ (P2-2 완료) · Phase 3 ✅ (P2-3 완료 · P1-1 완료)

---

## 📋 검증 결과 요약

| 판정 | 건수 | 항목 | 사유 |
|:----:|:----:|------|------|
| ❌ 기각 | 4 | P0-1, P0-2, P2-1, P2-4 | 오진 및 실제 해를 끼치는 제안 |
| ⚠️ 조건부 | 2 | P1-3, P2-2 | 방향은 타당하나 내용 결함 — 수정 후 채택 |
| ✅ 채택 | 7 | P1-1, P1-2, P2-3, P3-1, P3-2, P3-3, P4-1 | 검증 통과 |
| ❌ 종결 | 2 | P4-2, P4-3 | P4-2: 채택 표기 후 실측으로 오진 확인 (timeout 불필요 패턴) · P4-3: 착수 전 실측으로 대상 부재 (폴더 폴링 없음) |

**메타 관찰**: 라인 수·파일 위치 등 기계적 사실은 정확했으나, "참조 없음 = dead" 추론과 정량 주장에서 반복적으로 붕괴. 심각도 라벨이 높을수록(P0) 오진율이 높았음. P3-1처럼 코드를 실제로 읽고 쓴 항목은 품질이 높았음. → 제안서 기반 루프에는 "제안 → 독립 검증 → 착수" 게이트가 필수.

**게이트 실제 적용 사례 (P4-3, 2026-08-01)**: "radio.py:160 = file watcher 폴링 → watchdog 전환" 제안을 착수 전 실측한 결과, L160의 `time.sleep(10)`은 서버 연결 재시도 대기이며 폴더 감시 코드는 존재하지 않음 → **종결 (대상 없음)**. 외부 의존성(watchdog) 추가라는 2d/중 리스크가 게이트 없이 착수됐다면 낭비였을 것.

**실행 권장 순서**: P3-1 → P1-2 → P1-3 (가장 안전하고 가시적인 개선)  
**구현 현황**: P3-1 ✅ 완료 | P1-2 ✅ 완료 | P3-3 ✅ 완료 | P3-2 ✅ 완료 | P1-3 ✅ 완료 | P1-1 ✅ 완료 (P6-2, `c52891fd`) | P2-2 ✅ 완료 | P2-3 ✅ 완료

---

## 🟠 P1 — 성능/안정성

---

### P1-1. `asi.py:run_repl` 분할 ✅ 채택 — **✅ 완료 (P6-2, `c52891fd`)**

**파일**: `asi.py` — 원래 `run_repl` L6728–8988 (2261라인, 중첩 depth 19, 중첩 함수 10개)
**현황**: 단일 함수가 REPL 전 생명주기(초기화, 입력 처리, 명령어 dispatch, tool loop, 세션 관리, 출력 렌더링) 포함. 지역변수 100+개, 중첩 depth ~20.

**완료 (2026-08-01)**:
| 단계 | 결과 |
|------|------|
| Phase A | `run_repl` → `_run_repl_impl` rename + `run_repl()` wrapper 위임 (API 호환) |
| Phase B | `_init_repl_engine(args, repo_root)` — LLM engine/provider + design-chat core 초기화 (~118라인) |
| Phase C | `_init_session_state(repo_root, svc, design_config)` — 세션/터미널 상태 초기화 (~110라인) |

**완료 (2026-08-02, P6-2 `c52891fd`)**: `run_repl` 블록 전체(~6,900줄)를 `external_llm/repl/repl_impl.py`로 추출 — asi.py 10,363→3,410줄, barrel re-export 76개 심볼 유지(`import asi; asi.run_repl` / `from asi import run_repl` 호환). `run_repl`/`run_once`/`run_subagent_worker` + `_init_repl_engine`/`_collect_input`/`_run_repl_impl`(중첩 `_dispatch_command`/`_run_chat_turn`)로 분할 완료, REPL 테스트군 183 passed. 메인 루프 내부 세분화는 별도 라운드.

**핵심 제약 (메인 루프 분할의 벽)**: 명령어 dispatch 블록(`/help`·`/diff`…, 612라인)은 `continue`/`break` 17곳, `/claude` 6곳, orchestrator·tool-loop try 5곳 — 루프 제어와 강결합이라 중첩 함수/모듈 함수 추출 불가. 후처리부(8893–9082)는 외부 캡처 변수 40+개. → 메인 루프 분할은 `continue`/`break` → 반환 코드 리팩터링이 선행돼야 하며 별도 라운드로.

**영향**: 수정 시 ripple effect, 단위 테스트 불가능.

**제안**: 5~7개 함수로 분할:

| 함수 | 책임 | 추정 라인 |
|------|------|----------|
| `_init_engine_and_model()` | LLM engine/provider 초기화 | ~200 |
| `_init_session_state()` | 세션/컨텍스트 초기화 | ~150 |
| `_process_input()` | 입력 전처리/검증 | ~200 |
| `_handle_command()` | `/명령어` dispatch | ~300 |
| `_execute_tool_loop()` | tool loop 메인 | ~500 |
| `_render_output()` | 출력 스트리밍/렌더링 | ~300 |
| `_handle_session()` | 세션 저장/로드 | ~200 |

**전략**: 기존 `run_repl`을 `_run_repl_impl()`로 rename → 새 `run_repl()` wrapper가 위임 (API 호환) → 하위 함수 하나씩 추출. 각 단계마다 pytest 회귀 테스트.

**노력**: 3-5일 | **리스크: 상** — 광범위 통합 테스트 필요

---

### P1-2. API response 파싱 DRY ✅ 채택

**파일 분포** (7곳 정확히 실측 일치):
| 파일 | 횟수 |
|------|------|
| `planner_plan_create.py` | 1 |
| `llm_body_generator.py` | 1 |
| `design_chat_loop.py` | 3 |
| `agent_loop.py` | 1 |
| 기타 | 1 |

**패턴** (7곳 동일):
```python
response.get("choices", [{}])[0].get("message", {}).get("content", "")
```

**제안**:
```python
# external_llm/client.py
def extract_llm_content(response: dict, *, default: str = "") -> str:
    """Extract LLM response content from standard OpenAI-format dict."""
    try:
        return str(response["choices"][0]["message"]["content"] or default)
    except (KeyError, IndexError, TypeError):
        return default
```

**노력**: 0.5일 | **리스크: 하** — 1:1 substitution + pytest

---

### P1-3. 정규화 파이프라인 DRY ⚠️ 조건부 (전면 재작성 필요)

**파일**: `external_llm/` 전역  
**현황**: `.lower()`.replace() 계열 정규화 체인이 여러 변이체로 분산.

**검증 결과**:
- `.lower()` 총 사용: 664건 (최초 제안 186건은 누락)
- 동일 줄 `.lower().replace("-", "_")` 패턴: **6건** (최초 제안 ~40건은 과대추정)
- 변이체 A (`.lower().replace("-","_").replace(" ","_").strip()`): 실측 6건

**변이체 분포** (실측):

| 변이체 | 패턴 예 | 실측 건수 | 설명 |
|--------|---------|----------|------|
| A | `.lower().replace("-", "_").replace(" ", "_").strip()` | 6 | 키 정규화 |
| B | `.strip().lower()` | ~15 | 단순 trim + 소문자 |
| C | `.lower().strip("/").split("/")` | ~8 | 경로 정규화 |
| D | `.strip().lower().split("/")[-1]` | ~8 | basename 추출 |
| E | `.strip().lower().replace("_", "-")` | ~5 | 역방향 (dash 복원) |

**문제점**:
- 동일한 정규화 의도가 5가지 방식으로 분산 → 유지보수 시 일관성 깨짐
- 각 변이체가 서로 다른 edge case 처리
- 변이체 B를 단순 `normalize_key()`로 대체 시 내부 공백/대시까지 언더스코어로 바뀌는 **동작 변경** 발생

**제안** (수정):
```python
# external_llm/languages/_normalize.py
_NORMALIZE_TABLE = str.maketrans(" -", "__")


def normalize_key(s: str) -> str:
    """Normalize identifier: lowercase, translate spaces/dashes to underscores, strip.

    NOTE: 기존 .strip().lower()와 달리 내부 공백/대시도 변환하므로
    동작 변경이 예상되는 사이트는 별도 마이그레이션 필요.
    """
    return s.lower().translate(_NORMALIZE_TABLE).strip()


def strip_lower(s: str) -> str:
    """Strip whitespace and lowercase — 변이체 B 전용."""
    return s.strip().lower()
```

**수정된 코드** (최초 제안 버그 수정):
- `strip(strip_chars or "")` → `strip()` (후행 공백 strip 보장)
- 변이체 A→`normalize_key(s)`, 변이체 B→`strip_lower(s)` 분리

**노력**: 1일 (재작성 포함) | **리스크: 중** — 변이체 B 치환 시 동작 변경 주의

---

## 🟡 P2 — 코드 품질

---

### P2-2. `write_tools.py` 분할 ⚠️ 조건부 — **✅ 완료**

**파일**: `external_llm/agent/tool_handlers/write_tools.py` (분할 전 6249라인 실측 일치)  
**현황**: 단일 파일 6249라인, 50+ 함수, 10+ dataclass. 단일 파일 최대 규모 2위 (asi.py 다음).  

**완료 (2026-08-05)**: barrel + 4모듈 분할 — write-safety 3층 gate은 `write_tools_core.py`에 집약 (PARITY 계약 준수), 모든 기존 import는 barrel re-export로 호환 유지.

| 모듈 | 라인 |
|------|------|
| `write_tools.py` (barrel re-export) | 49 |
| `write_tools_core.py` | 743 |
| `write_tools_edit_mixin.py` | 2478 |
| `write_tools_patch_mixin.py` | 3386 |
| `write_tools_ast_mixin.py` | 246 |
**영향**: 탐색/디버깅 어려움, circular import 위험, 병렬 개발 불가.

**⚠️ 중요 — write-safety PARITY 계약**:
5개 write 도구(apply_patch, edit_text, modify_symbol, edit_ast, anchor_edit)는 모두 동일한 3층 post-edit gate을 공유:
1. 구문 검증 (언어 provider)
2. origin-skip (pre-edit 스냅샷과 동일 에러면 soft-fail)
3. rollback (실패 시 원복)

분할 시 이 gate 코드가 중복되지 않도록 `write_tools_core.py`에 집약하고, 각 도구 모듈은 `_safety_manager`/`_verify` 함수만 참조해야 함.

**제안**: 4개 모듈로 분할, 기존 `write_tools.py`는 barrel re-export 유지:

| 모듈 | 책임 | 추정 라인 |
|------|------|----------|
| `write_tools_core.py` | 공통 검증/로깅/atomic-write/brace scanner | ~800 |
| `write_tools_edit.py` | edit_text / modify_symbol | ~1500 |
| `write_tools_patch.py` | apply_patch / diff_apply / anchor_edit | ~2000 |
| `write_tools_ast.py` | edit_ast / AST ops | ~1000 |
| `write_tools.py` | `from .write_tools_edit import ...` re-export | ~50 |

**전략**: 새 모듈 생성 + re-export → 모든 기존 import `from ...write_tools import ...` 호환 → 단계적 직접 import migration.

**노력**: 2-3일 | **리스크: 중** — import cycle + write-safety PARITY 유지 필수

---

### P2-3. `_build_engine` param 축소 (13개 → config object) ✅ 완료

**파일**: `external_llm/repl/repl_impl.py` (`_build_engine` — P6-2로 `asi.py`에서 이동)  
**현황**: `def _build_engine(repo_root, request_text, provider, model, api_key, max_turns, stream_cb, cancel_event, *, svc, route_decision, thinking_mode, reasoning_effort, scoped_verification)` — **13개** keyword param (실측 2026-08-05). 호출부는 모두 keyword 호출. → `EngineConfig` dataclass로 이관, `_build_engine(config)` 1-param (커밋, `tests/unit/test_repl_engine_config.py` 6건).

**제안**:
```python
@dataclass
class EngineConfig:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    # ... 14개 전부 → dataclass field


def _build_engine(config: EngineConfig) -> LLMEngine: ...
```

**노력**: 1일 | **리스크: 하**

---

### P2-4. `time.sleep(N)` / timeout inventory (잔여 3건) ✅ 전항 종결 (P7-2)

P0-1/P0-2 기각으로 제외. HTTP/스레드 경로 잔여 blocking call — 2026-08-01 P7 라운드 실측으로 전항 종결:

| 파일 | 라인 | N | 경로 유형 | 처리 |
|------|------|---|-----------|------|
| `webapp/routes/stats.py` | ~~546~~ → 619 | 2s | webapp SSE stats | → **종결 (이미 수정됨)** — 실측 `await asyncio.sleep(2)` (async SSE 라우트, 문서 라인 stale) |
| `radio.py` | 160 | 10s | 서버 연결 재시도 대기 (파일 폴링 아님 — 폴더 감시 코드 부재) | → **종결 (대상 없음)** |
| `orchestrator.py` | ~~2413, 2496~~ | — | Popen no timeout | → **종결 (오진)** — `.wait()`/`.communicate()` **0건** 실측. 백그라운드 워커는 subagent_ipc 파일 + cancel.json 생명주기 관리, communicate를 붙이면 오케스트레이터 블로킹 |

---

## 🟢 P3 — Scanner 개선 (`# noqa` 인식 + false-positive 제거)

---

### P3-1. `unused_import_scanner` — `# noqa: F401` 인식 지원 ✅ 채택 (최우수)

**파일**: `external_llm/analysis/unused_import_scanner.py` (L336 scan loop)  
**현황**: scanner는 AST 기반 분석으로 import line의 `# noqa: F401` comment를 완전히 무시. Barrel re-export 파일 11곳에서 ~107건의 false positive 발생.

**검증 결과**: `import_info` 튜플에 `line_text` 필드 실재(L144/161/336), noqa 처리 부재 확인, 적용 위치·테스트 계획 모두 정확.

**정량적 검증**:

| barrel re-export 파일 | `# noqa: F401` 라인 | 영향받는 import 수 | 현재 scanner flag 수 |
|----------------------|--------------------|--------------------|---------------------|
| `change_spec_assertions.py` | 1 (L58) | 22 | 10+ (truncated) |
| `symbol_handlers.py` | 1 (L14) | 55 | 10+ (truncated) |
| `intent_verifier.py` | 1 (L22) | 22 | 10+ (truncated) |
| `models.py` (ts_vm) | 1 (L13) | 7 | 7 |
| `deterministic_plan_builder.py` | 1 (L18) | 1 | 1 |
| `planner_agent.py` | 1 (L36) | 1 | 1 |
| `planner_helpers.py` | 1 (L59) | 1 | 1 |
| `operation_executor.py` | 4 (L63-93) | 4 | 5 |
| `agent_phase_manager.py` | 1 | 1 | 1 |
| `symbol_handlers_shared.py` | 1 | 1 | 1 |
| `collaboration_orchestrator.py` | 1 | 1 | 1 |
| **합계** | **14** | **116** | **~107** |

**증상 예** (`change_spec_assertions.py:58`):
```python
from external_llm.editor._editor_core.lane.change_spec_assertions_shared import (  # noqa: F401
    _INTENTIONALLY_UNHANDLED,  # ← scanner flag (false positive)
    _KINDS,  # ← scanner flag
    _TIER1_KINDS,  # ← scanner flag
    # ... 총 22개 이름 모두 flag
)
```

**원인**: scanner는 AST로 분석하므로 `# noqa: F401` 주석에 접근 불가. `import_info` 튜플에 `line_text`가 포함되어 있지만(L144/161/336), 현재 코드는 이 필드를 전혀 검사하지 않음.

**제안** — scan loop (L336)에 `# noqa: F401` 체크 추가:

```python
def _has_noqa_comment(line_text: str, codes: set[str] | None = None) -> bool:
    """Check if *line* carries a # noqa comment, optionally for specific codes."""
    idx = line_text.find("#")
    if idx == -1:
        return False
    rest = line_text[idx + 1 :].strip()
    if not rest.lower().startswith("noqa"):
        return False
    if codes is not None:
        codes_part = rest.partition(":")[2].strip()
        return bool(codes & set(c.strip() for c in codes_part.split(",")))
    return True
```

**적용 위치** (L336-337, 기존 로직보다 먼저):
```python
for local_name, line_text, lineno, module in import_info:
    # ── # noqa: F401 suppression ──
    if _has_noqa_comment(line_text, {"F401"}):
        continue
    if local_name not in used_names and local_name != "*":
        # ... 기존 로직 그대로 ...
```

**단위 테스트**:
```
_has_noqa_comment("# noqa")               → True
_has_noqa_comment("# noqa: F401")         → True (codes={"F401"})
_has_noqa_comment("# noqa: F841")         → False (codes={"F401"})
_has_noqa_comment("# NOQA: F401, F841")   → True (codes={"F401"})
_has_noqa_comment("import x  # noqa: F401") → True
_has_noqa_comment("# comment")            → False
_has_noqa_comment("import os")            → False
```

**기대 효과**:
- **107건 false positive 제거** (11개 파일, 14개 noqa 라인)
- Barrel re-export 파일 대상 scanner 결과: **0건** (현재: 107건)
- 남은 genuine FP 4건은 P3-3에서 `# noqa: F401` 추가

**노력**: 0.5일 | **리스크: 하** — 기존 로직 변경 없음, exclusion만 추가

---

### P3-2. `vulture_scanner` — `# noqa: F841` 인식 지원 ✅ 채택

**파일**: `external_llm/analysis/vulture_scanner.py` (L526-612 filter loop)  
**현황**: Vulture 결과 필터링 시 line text 확인 없음 → `# noqa: F841`가 있어도 무시하고 flag.  
**영향**: variable/attribute 레벨에서 false positive 가능.

**제안**: P3-1의 `_has_noqa_comment()`를 `analysis/_noqa_utils.py` 공유 모듈로 추출:

```python
# analysis/_noqa_utils.py (shared)
def has_noqa_comment(line_text: str, codes: set[str] | None = None) -> bool: ...
def has_noqa_comment_on_line(lines: list[str], lineno: int, codes: set[str] | None = None) -> bool:
    """Check line at 1-indexed *lineno* for # noqa."""
    if 1 <= lineno <= len(lines):
        return has_noqa_comment(lines[lineno - 1], codes)
    return False
```

**vulture_scanner 적용** (L600 전):
```python
# ── # noqa: F841 suppression ──
if source_lines and has_noqa_comment_on_line(source_lines, first_lineno, {"F841"}):
    continue
```

**노력**: 0.5일 | **리스크: 하**

---

### P3-3. Genuine FP에 `# noqa: F401` 후속 태깅 ✅ 채택

P3-1 적용 후에도 남는 genuine false positive 4건 (`operation_executor.py`, noqa 누락):

| 파일 | symbol | 라인 |
|------|--------|------|
| `operation_executor.py` | `_detect_change_event` | 63 |
| `operation_executor.py` | `GuardContext` | 81 |
| `operation_executor.py` | `_PreExecGuardResult` | 81 |
| `operation_executor.py` | `_extract_f821_names` | 93 |

**제안**: 4개 import line에 `# noqa: F401` 추가.  
**노력**: 0.1일 | **리스크: 하**

---

## ⚪ P4 — 보류

| 항목 | 파일 | 라인 | 설명 | 노력 | 리스크 |
|------|------|------|------|------|--------|
| **P4-1** | `webapp/routes/stats.py` | 546 | `time.sleep(2)` → `asyncio.sleep(2)` | 0.25d | 하 |
| **P4-2** | `external_llm/agent/orchestrator.py` | 2413, 2496 | ~~`Popen` without timeout → `communicate(timeout=30)`~~ — 채택 표기됐으나 커밋 부재, 실측 결과 L2413 fire-and-forget 의도(stdout/stderr DEVNULL) / L2496 tracked Popen(이미 비동기 수명주기 관리) → **종결 (오진)** | 0.25d | 하 |
| **P4-3** | `radio.py` | 160 | ~~`time.sleep(10)` polling → `watchdog` inotify/kqueue~~ — L160은 서버 재연결 대기이며 폴더 감시 코드가 없음 → **종결 (대상 없음)** | 2d | 중 |

---

## 🟢 P5 — 2026-08-01 개선 제안 라운드

| 항목 | 대상 | 설명 | 노력 | 리스크 | 상태 |
|------|------|------|------|--------|------|
| **P5-1** | `external_llm/agent/tool_handlers/git_tools.py:1828` | `pip install {' '.join(pkgs)}` + `shell=True` → `shlex.join` 인용 (주입 방어) | 0.1d | 하 | **✅ 완료** — shlex.join 적용, 39 passed |
| **P5-2** | `os.walk` uncached 30+ 사이트 | git-first SSOT(`repo_files.cached_repo_file_list`)/TTL 60s 캐시 통일 — 실측 185ms vs 21ms (9배) | 2-3d | 중 | **✅ 완료** — 12개 사이트 변환, 캐시 중앙화 + atomic_io 쓰기 퍼널 무효화, 테스트 8건 신규. **planner_spec 원복** (파일 세트 변경 즉시 반영 계약 — TTL 스테일 회귀), **symbol_index 제외** (변경 감지 원천) |
| **P5-3** | 대형 함수 16개 | anchor 3종 자립 블록 추출 — `_handle_anchor_edit` 1406→1344 (`_redirect_future_import_anchor` + `_try_import_add_fast_path`), `_handle_insert_after_line` 1295→1257 (`_try_canonical_anchor_fast_path`), `_tool_anchor_edit` 864→847 (`_resolve_ast_anchor_line`) | 항목당 1-3d | 상 | **✅ 1단계 완료** — 598줄 단일 If(`_handle_insert_after_line`) 등 대형 섹션 분할은 조기 return 16개로 자동 변환 불가 → **별도 라운드** |
| **P5-4** | `mcp/server.py` | SSE 위에 Streamable-HTTP 트랜스포트 추가 (2025-11-05 프로토콜) | 2-3d | 중 | **✅ 완료** — `streamable_server.py` 신규 (POST /mcp JSON+SSE 모드, Mcp-Session-Id, GET/DELETE), CLI `--mode http`, 테스트 6건 신규 |
| **P5-5-1** | `test_intent_verifier.py:7` | ~~`_SAMPLE_CODE` 데드 상수~~ → **종결 (오진)**: L33 기본 인자 `code=_SAMPLE_CODE`로 라이브, `_verify([...])` 49+ 호출이 기본값 의존, vulture 스캔 0건 | 0.1d | 하 | **❌ 종결 (오진)** |
| **P5-5-2** | `asi.py` | `_ensure_*_console_imported` 3중 구조 (sim 1.00) → 파라미터화. **저우선** — 20줄 미만, 스캐너도 extraction 비권고 | 0.1d | 하 | ⏳ 보류 |

---

## 🟠 P7 — 2026-08-01 하드닝/문서/게이트 라운드

> 참고: P6 라운드 — P6-1 REPL dispatch 계약 테스트(`140c8331`) · P6-2 asi.py REPL 블록 모듈화(REPL 블록 6,900줄을 `external_llm/repl/repl_impl.py`로 추출, asi.py 10,363→3.4K줄, barrel re-export 76개 심볼 유지) · P6-3 write_task 쓰기 순서 하드닝(항상-mint epoch + sidecar-first, `6ee43ed6`) — **전부 완료**.
>
> 참고: R 라운드 (2026-08-02) — R1+R2+R3 MCP Streamable-HTTP 세션 수명주기 하드닝(`5512d50c`) · R4 git-context SSOT 위임(`20c9da68`) · R5 pytest-xdist 기본 병렬화(`922e0920`) — **전부 완료**.

| 항목 | 대상 | 설명 | 노력 | 리스크 | 상태 |
|------|------|------|------|--------|------|
| **P7-1** | `external_llm/agent/tool_handlers/web_search_tools.py:2214` | `_tool_web_fetch` SSRF 가드 부재 — 호스트 검증 0건 (ipaddress/is_private/is_loopback grep 전무) | 0.5-1d | 하 | ✅ 완료 (`fe542611`) — IP 6종 판정 + getaddrinfo all-public 정책 + redirect 재검증 훅 + `ASI_ALLOW_PRIVATE_URLS=1` 해치 |
| **P7-2** | P2-4 잔여 3건 | 전항 종결 — stats.py 이미 `asyncio.sleep`, orchestrator `.wait()`/`.communicate()` 0건, radio.py 기존 종결 | 0.1d | 없음 | ✅ 종결 |
| **P7-3** | `scripts/check_no_new_silent_except.py` | 훅 `--index-only` 모드 — unstaged 스태시 함정 원천 제거 | 0.5d | 하 | ✅ 완료 |

---

### P7-1. [하드닝 · 채택 권장] `_tool_web_fetch` SSRF 가드 부재

**파일**: `external_llm/agent/tool_handlers/web_search_tools.py:2214` (`_tool_web_fetch`)

**현황**: 임의 http(s) URL을 `follow_redirects=True` + 3회 재시도로 fetch하며 **호스트 검증이 전무** — `ipaddress`/`is_private`/`is_loopback` grep 결과 `browser_tools.py` 포함 **0건**. scheme 기본값 지정·Reddit rewrite·바이너리 content-type 차단만 존재.

**공격 벡터**: 웹 콘텐츠 프롬프트 인젝션 → 에이전트가 `http://127.0.0.1:11434/api/tags`(Ollama)·`:8000`(webapp)·`:8080`(SEARXNG) fetch → **로컬 데이터가 대화로 유출**. redirect 재검증도 필요 (가드가 있어도 우회 가능).

**제안**: URL 호스트 검증 (문자열 파싱 + DNS resolve 후 `is_loopback`/`is_private`/`is_link_local` 재검증, redirect마다 재검증), `ASI_ALLOW_PRIVATE_URLS=1` env 이스케이프 해치. SEARXNG 백엔드는 `_search_searxng` 별도 경로라 default `localhost:8080` 접근은 **영향 없음** — `_tool_web_fetch`의 사용자 URL에만 적용.

**노력**: 0.5–1일 | **리스크: 하**

---

### P7-2. [문서 종결] P2-4 "잔여 3건" → 전항 종결

기계적 사실 확인 (P4-2/P4-3과 동일한 "기계적 사실이 정량적 주장보다 강함" 패턴):

| P2-4 항목 | 현황 판정 |
|---|---|
| `stats.py:546` time.sleep(2) | **이미 수정됨** — 현재 619행 `await asyncio.sleep(2)` (async SSE 라우트, 문서 라인 stale) |
| `orchestrator.py:2413,2496` → `communicate(timeout=30)` | **오진** — orchestrator.py에 `.wait()`/`.communicate()` **0건**. 백그라운드 워커는 subagent_ipc 파일 + cancel.json으로 생명주기 관리, communicate를 붙이면 오케스트레이터가 블로킹됨 |
| `radio.py:160` | 기존 종결 유지 |

→ 상단 P2-4 섹션 전체 "✅ 종결" 처리 완료.

**노력**: 0.1일 | **리스크: 없음**

---

### P7-3. [게이트 UX · 소형] silent-except 훅 `--index-only` 모드

**파일**: `scripts/check_no_new_silent_except.py` (`.pre-commit-config.yaml` `no-new-silent-except` 훅)

**현황**: pre-commit 훅 7종이 `always_run` + **unstaged 스태시 후 워킹트리 전체 스캔** — baseline이 이미 갱신된 다른 파일이 unstaged면 거짓 실패 (P6-1 커밋 라운드에서 1회 실제 겪음: 스태시 → 부분 baseline → 2단 커밋으로 우회).

**제안**: `--index-only` 모드 추가 (`git ls-files` + `git show :<path>`로 **HEAD+인덱스 스냅샷** 스캔) — 훅이 staged 상태만 검사하므로 baseline과 항상 일치, 스태시 함정 원천 제거.

**노력**: 0.5일 | **리스크: 하** | **✅ 완료** — `scripts/check_no_new_silent_except.py`에 `--index-only` 구현 (untracked/unstaged 무시, 비-git 트리는 워킹트리 폴백), `.pre-commit-config.yaml`·`.github/workflows/lint.yml` 훅 엔트리에 적용, 신규 테스트 4건. 라이브 검증: 병렬 세션 untracked 파일 존재 중에도 index-only 스캔 `1323 == 1323` 통과 (워킹트리 스캔은 그 파일 때문에 거짓 실패하던 상태 — 함정 실증).

---

## 🟠 P8 — 2026-08-16 CI 게이트 콜드 부트 개선 라운드

> 컨텍스트: P-I 재측정(`35f52561`, 2026-08-16)에서 게이트 실측 — **웜 15-18s / 콜드 ~69.3s** (`scripts/check_structural_scanners.py --gate-only`). 열거→읽기 통합(B1, `bb641142`)과 원자 쓰기(B2, `73baaf43`)가 캐시 신뢰성을 확보했으나, CI는 캐시가 **아예 없다** — 아래 제안은 그 CI 콜드 부트 비용을 다룬다.

### P8-1. [✅ 완료] `lint.yml` structural-scanner 스텝 `actions/cache`로 `.cache/` 재사용

**파일**: `.github/workflows/lint.yml` — "Check ZERO deterministic structural scanner candidates" 스텝(`run: python scripts/check_structural_scanners.py --gate-only`)

**현황**: `.cache/`는 gitignore(`.gitignore:26`)라 CI checkout에 존재하지 않음 → **매 push마다 콜드 부트**. 실측: 웜 15-18s → 콜드 ~69.3s (**3.9-4.6배**, 게이트 全 스텝 중 최대 단일 비용). `--gate-only`는 8개 스캐너 + 그래프 빌드 전부를 요구하므로 캐시 히트 시 실질 절감은 50s+.

**타당성 (코드로 검증됨)**:
- 캐시 fingerprint는 `(path, mtime_ns, size)` — `actions/cache`(gzip 압축 tar)로 복원하면 mtime/size가 **원본 그대로 보존**되어 스탬프 일치 → 히트.
- 파이썬 파일이 **변경된 경우 mtime_ns/size가 바뀌어 미스** → 그 파일만 재분석(self-heal, 변경 후 첫 실행이 콜드 부트와 동일 결과).
- B2(`73baaf43`) 원자 쓰기 계약: 캐시는 완전 payload 1개 (`atomic_write_json`/streaming temp+replace) — CI 복원 파일이 손상/절단돼도 **fail-open**으로 풀 재분석 (정확성 무영향).
- `CACHE_VERSION`/`_DBX_CACHE_VERSION`/`_CRX_CACHE_VERSION` 등 버전 키가 payload에 내장 — 스캐너 로직 변경 시 자동 무효화 (수동 버전 범프 불필요).

**리스크: 하** | **노력: 0.25d** | **절감: push당 ~50s** (콜드 69s → 웜 ~18s) | **✅ 완료 — `cb0a7fe3`** (구현안 그대로 적용, fingerprint 보존 검증 포함)

**구현안**: `unit-tests` 잡과 별개로, 같은 스텝 앞에

```yaml
- name: Cache gate analyzer results (.cache/)
  id: gate-cache
  uses: actions/cache@v4
  with:
    path: .cache
    key: gate-scanners-${{ runner.os }}-${{ hashFiles('external_llm/**/*.py', 'scripts/*.py') }}
    restore-keys: |
      gate-scanners-${{ runner.os }}-
```

**주의사항**:
1. **`hashFiles`는 파이썬 소스만 키** — 소스 무변경 + 의존성만 바뀐 push도 웜 유지 (의도).
2. **key에 개별 캐시 버전(`CACHE_VERSION` 등)을 넣지 말 것** — 버전은 payload 내장이라 이미 자동 무효화되며, key에 넣으면 불필요한 캐시 분열.
3. **`--gate-only` 스텝은 `python -m pip install -e .` 직후** — 캐시 restore는 그 **앞**에 배치 (별도 스텝).
4. 그래프 캐시가 복원되면 `graph.cache_stats` 로그가 **`hit/total`**으로 찍힘 — CI 로그에서 히트율 확인 가능 (검증 지표).
5. **콜드 ↔ 웜 결과가 같음은 B1 계약상 보장** — 웜 빌드는 캐시-served여도 bit-for-bit 동일 (그래프 walk 순서가 단일 주입 순서, `check_structural_scanners.py:451-454`).

**대안 (기각)**: ① `.cache/` 커밋 — gitignore 계약 위반 + 매 push 100MB 업로드, 기각. ② `--gate-only` 스텝을 release.yml로 이동 — 이 잡이 유일한 push 게이트라 무의미, 기각. ③ pre-commit 훅에 캐시 워밍 — 로컬은 이미 웜이라 무의미, 기각.

### P8-2. [✅ 완료] `check_structural_scanners.py` 게이트 타이밍 주석 갱신

**파일**: `tests/unit/test_check_structural_scanners.py:714-715` (`~21s warm` 주석 → P-I 실측 반영)

**현황**: 웜/콜드 실측값이 문서화된 곳이 없어 CI 캐시 설계자가 기대값을 알 수 없음. P-I(`35f52561`)가 2파일 갱신하며 `tests/unit/test_check_structural_scanners.py:714-715`에 `15-18s warm / 69s cold --gate-only`를 남겼으나, `check_structural_scanners.py` 헤더/도움말에는 콜드 부트 언급이 없음.

**제안**: `scripts/check_structural_scanners.py` docstring/`--help`에 실측 추가 — `--gate-only` 콜드 ~69s (fresh repo/CI, 캐시 부재 시) / 웜 15-18s. "CI 캐시 도입 시 웜에 근접" 기대값 명시.

**리스크: 없음** | **노력: 0.1d** | **✅ 완료 — `f249257f`** (docstring에 콜드/웜 실측 고정 + 스캐너 개수 정정 포함)

---

## 🟢 P9 — 2026-08-20 개선 제안 라운드 (구조 스캐너 8종 0건 이후, 실측 기반)

> 컨텍스트: 구조 스캐너 8종(dead_block/contradictory/broken_contract/duplicate_definition 등) 全검색 0건 — 게이트가 트리를 청결하게 유지 중. 신규 버그 후보 없음을 전제로, **쌍둥이 drift 위험 + 실측 성능** 축으로만 제안 구성.

### P9-1. [✅ 완료] BM25 5중 복제 공식 → `agent/bm25.py` 단일 소스 통합 + idf 호이스트

**대상**: `rag_searcher._bm25_score`(참조 구현) ↔ `insights_manager` 인라인 idf/tf_norm ↔ `design_chat_loop` 랭킹 루프 ↔ `symbol_search`/`read_tools` 셋업 쌍둥이(_doc_tc/_df/_avgdl/_scores 블록 통째 복붙) — 탐색 과정에서 **쌍둥이 2개가 아니라 5형제**로 확인됨.

**근거**: ① 쌍둥이 drift는 실증된 버그 클래스(cancel-scope 2라운드에서 call_graph↔rag_searcher 한쪽만 수정된 사례) — BM25도 이미 insights 쪽만 idf 호이스트가 적용된 불균형 상태였음("matches rag_searcher" 주석으로 수동 동기화). ② 실측: lock 보유 구간(`_index_lock`) 내 스코어 루프 **1.06→0.76 ms/query (1.39x)** — 병렬 서브에이전트 rag_search 경합 시 lock-held 시간 직접 단축.

**구현**: `external_llm/agent/bm25.py` 신설(stdlib-only, agent import graph 최하층 — AST 게이트로 봉인). K1/B/idf/tf_norm/참조 score/fast-path(pairs)/bm25_rank(셋업 쌍둥이 대체) 제공. **비트 동일성 계약**: 이전 공식 전사 대비 `==` 정확 일치(isclose 아님) — 랭킹이 vector-cache/promote 순서를 먹이므로 epsilon drift도 허용 안 함. `tokenize`가 중복 토큰을 보존하므로 fast-path multiplicity 유지(전사 테스트로 봉인). `bm25_rank`는 빈 코퍼스에서 ZeroDivisionError 대신 `[]`(엄격 개선).

**리스크: 하** | **노력: 0.5d** | **✅ 완료 — 신규 테스트 9건(test_bm25_core.py) + 관련 357 테스트 통과, winners 비트 동일 확인**

### P9-2. [✅ 완료] improvement_proposals.md P8 상태 스테일 정리

**현황**: P8-1(`cb0a7fe3`)/P8-2(`f249257f`)는 구현 완료였으나 상태가 "제안"으로 잔류 + 변경 이력 미기록.

**✅ 완료 — 본 커밋에서 P9 섹션 추가와 함께 정리.**

### P9-3. [데이터 확보 — 마이닝 대기] v0.2.27 verify durations 마이닝

**현황**: v0.2.27 릴리스(2026-08-20)에서 첫 full-verify durations 아티팩트 확보 — `.verify_artifacts/verify-durations-full-20260820-204549.txt` (16/16 게이트, 215s). top 실측: ① pty 50k 절단 테스트 13.8s (CI 플레이크 이력 보유 — v0.2.24), ② vector_cache 재사용 12.6s, ③ symbol_search rg 부재 8.5s; top-40 중 stage3 spawned REPL 군집이 18건으로 최대 블록. 후속 라운드에서 xdist 워커 밸런싱/마킹 구성(실측 기반으로만).

**노력: 0.25d** | **리스크: 없음** | **✅ 완료 — P10-1으로 승계 (2026-08-20)**

---

## 🔬 P10 라운드 (2026-08-20) — P9-3 durations 마이닝 기반

> 컨텍스트: v0.2.27 full-verify durations(20260820-204549)의 top-40 중 stage3 spawned 군집 18건 + 13.8s/12.6s/8.5s 헤비급 = **top-heavy 스위트**. 기본 라운드로빈 `--dist=load`는 워커 테일 스트레글러를 남김. 실측 A/B로 검증.

### P10-1. [✅ 완료] xdist `--dist=worksteal` 전환 — 풀스위트 1.24x

**근거** (8워커, 동일 트리·명령, 인터리브 A/B×2 — 병렬 세션 부하 드리프트 상쇄):

| 모드 | round1 | round2 | 중앙값 |
|---|---|---|---|
| `load` (기본) | 199.1s | 202.1s | **200.6s** |
| `worksteal` | 156.1s | 167.1s | **161.6s** |

**→ 1.24x (−39s, −19.5%)**. worksteal(pytest-xdist ≥3.5)은 유휴 워커가 바쁜 워커의 큐에서 테스트를 훔쳐 top-heavy 테일을 자동 평준화. 최종 검증(신규 계약 테스트 6건 포함) **15623 passed / 149.9s — A/B 최속 기록 갱신**.

**구현**: `pyproject.toml` addopts에 `"--dist", "worksteal"` 추가(분리 쌍 — pytest 9 verbatim 규칙). CI(lint.yml)·release verify 전부 addopts 상속으로 일괄 적용. `tests/unit/test_pytest_dist_contract.py` 신규 6건: R1 실제 addopts가 worksteal 핀(해시 아닌 tomllib 구조적 파싱) / R2 `-n` 병렬 플래그 공존(직렬 실행 시 inert 방지) / R3 dev extra `pytest-xdist>=3.6` ≥3.5(도입 버전) / 추출기 공허 가드 3건(플래그 소실·값 오류·토큰 융합 거부).

**부수 관찰**: `load`에서만 2/4 라운드 실패(동일 1건, worksteal 0/2) — 요구 시 재현 안 됨(추가 2회 통과). 사전 존재하는 파일 조합 플레이 클래스(전역 매니저 오염 계열 추정)로 **본 변경과 무관하나**, worksteal 배치가 우연히 문제 조합을 해소한 것으로 기록. 정체 미식별 — 재발 시 `--dist=load` 되돌림 없이 실패 ID 캡처 권고.

**리스크: 없음** | **노력: 0.25d** | **✅ 완료 — A/B 실측 + 풀스위트 GREEN + 계약 테스트 6건**

---

## 🛠 P11 라운드 (2026-08-20) — P9-3 헤비급 마이닝 2차 (top-3 중 2건이 낭비/버그)

> 컨텍스트: 구조 스캐너 8종 0건(유사 19쌍 전부 "extraction not advised"). 수확은 전부 durations top 헤비급 실측에서 — 13.8s/12.6s/8.5s 중 2건이 테스트 자체의 결함이었다.

### P11-1. [✅ 완료] vector_cache 무효화 테스트 patch 스코프 탈출 — 버그

**현황**: `test_vector_cache_invalidation.py`의 `_manager()` 헬퍼가 `with patch(get_global_embedding_model→None)` **안에서 생성만 하고 반환** — 그러나 `__init__`은 모델을 lazy 로드하므로 patch는 아무것도 가로채지 않았고, 실제 로드는 테스트 본문의 `_ensure_index_loaded()`→`_ensure_model_loaded()`(patch 밖)에서 실행됨. cProfile 실측 테스트 7.05s 전부 `_ensure_model_loaded` = **실 SentenceTransformer 로드**. docstring 계약("without loading a real SentenceTransformer") 위반 + HF 캐시 미보유 환경에서 유닛 테스트가 네트워크 다운로드(CI 불안 요인).

**수리**: 형제 파일(`test_vector_cache_lazy_and_migration.py`)과 동일한 autouse fixture `_no_real_model`(함수 스코프 monkeypatch)로 교체, 무의미한 `with patch` 제거. **실측: 7.05s → 1.50s(4 passed, 개별 duration <0.02s), `HF_HUB_OFFLINE=1` + 존재하지 않는 ST 홈에서도 GREEN — 모델 로드 완전 제거 확인.** 나머지 4개 vector_cache 테스트 파일은 전부 올바른 패턴 — 이 파일로 국소.

**리스크: 하** | **노력: 0.1d** | **✅ 완료**

### P11-2. [✅ 완료] pty 50k 절단 테스트 → bracketed paste — ~18x

**현황**: 스위트 최장 테스트(단독 8.4s / in-suite 13.8s, v0.2.24 CI 플레이크 2회 이력)가 50k 바이트를 **raw per-key**로 피딩 — 인간이 타이핑 불가능한 경로로 ptk의 키별 전체 라인 재렌더를 강제(O(n²): 25k=3.51s → 50k=7.98s = 2.27x). 실사용자 대형 입력은 bracketed paste(단일 삽입)로 도착. 절단 검사는 `prompt()` 반환 텍스트 기준(`repl_impl.py:3180`)이므로 전달 경로와 무관 — **제품이 아니라 테스트가 인위적 최악 경로를 걷고 있었음 → 제품 변경 불필요, 테스트만 교체.**

**수리**: `ESC[200~`+50001자+`ESC[201~`+`\r` 단일 paste로 교체. v0.2.24 플레이크 방지 계약(feeder 스레드 + `feed_done` 배리어 + 90s join) 유지. **실측: 0.46s(~18x), 3회 반복 0.63s 안정, 파일 전체 47 passed 12.59s.** 부수: P10 부수 관찰의 `load` 분포 플레이 유력 후보(최장 테스트 + 컨텐션 시 90s join 압박) 사실상 해소.

**리스크: 하** | **노력: 0.25d** | **✅ 완료 — 동등 단언 유지(값 50000 + 절단 공지)**

---

## 🛠 P12 라운드 (2026-08-21) — P9-3 헤비급 마이닝 3차 (P11이 top-2 치운 뒤 드러난 새 top 헤비급)

> 컨텍스트: 구조 스캐너 8종 0건. 수확은 durations + cProfile 귀속에서 — 둘 다 P11 정리 후 새로 노출된 헤비급이며, 하나는 제품 버그를 겸한다.

### P12-1. [✅ 완료] tier-2 문법 게이트의 외부 컴파일러 유입 — 버그+성능

**현황**: `symbol_modify_tool.py`의 pre-write 문법 게이트 tier-2(`_ts_syntax_valid`)는 docstring상 "toolchain-free, tree-sitter(CORE dependency)" 게이트인데, 구현은 `SyntaxValidator.validate_syntax`를 거쳐 **kotlin/java/c 프로바이더로 위임 → kotlinc/javac/gcc 서브프로세스 부팅(~2s/호출)**. cProfile 귀속: kotlin 군집 테스트 3.4~5s의 비용 전부가 이 경로. **버거 성분**: "source가 parse clean해야 reject" 전제에 컴파일러를 쓰므로 post-edit 파일 어디든 **시맨틱 에러 하나(타입/미해결 참조)면 tier-2가 통째로 무력화** — 문법 훼손 에디트가 pre-write에서 잡히지 않고 noisy post-write rollback로 회귀. 또한 로컬(kotlinc 있음)/CI(없음)가 다른 경로를 타던 비결정적 커버리지.

**수리**: `_ts_syntax_valid`를 `tree_sitter_utils.find_error_nodes` 직접 호출(pure tree-sitter)로 전환 — `None`(그래마 부재)=무의견 계약 유지. 프로브로 판정 동일성 사전 검증(INVALID/CLEAN 정합, 시맨틱 에러는 CLEAN=문법 게이트의 올바른 맹점). **실측: kotlin tier 테스트 2.04s→0.02s(102x), 게이트 커버 2파일 236 passed 0.84s(최대 0.19s).** 계약 테스트 3종으로 봉인: ① parser 예외→None ② `SyntaxValidator` 라우트 부재(raise→AssertionError로 증명) ③ 시맨틱 에러는 reject하지 않음. **트레이드오프(명시)**: 컴파일러만 잡는 에러 클래스는 pre-write reject → post-write verify+rollback로 이동 — agent 검증 스택(tool_registry 3곳, tool_safety, write mixin)이 기존대로 담당.

**리스크: 중간(동작 트레이드오프 문서화됨)** | **노력: 0.25d** | **✅ 완료**

### P12-2. [✅ 완료] rg 부재 폴백 캐시 스래싱 — cap 512 < 레포 952파일

**현황**: rg 부재(base `pip install` 기본 환경)에서 `find_symbol`이 매 호출 1.5~1.95s — `_PY_FILE_CACHE_MAX_ENTRIES=512`가 레포 Python 파일 수(~950)보다 작아 **전체 순회마다 캐시 전량 퇴거 → 재파싱, 절대 amortize 안 됨**. `test_symbol_search_rg_absent.py`(신규 #1 헤비급 8.45s)가 대표 피해자.

**수리**: cap 2048(엔트리 평균 ~23KB 실측 → 최악 ~47MB bound, 초과 모노레포는 기존 LRU). + rg_absent 테스트 파일이 인스턴스별 캐시라 매 테스트 콜드 워크를 재지불하던 것을 모듈 레벨 지연 싱글턴으로 공유(patch는 함수 스코프 유지, mtime 시그니처 무효화로 정확성 보장). **실측: 5.51s→2.34s(2.4x, 직렬), 콜드 1회만 지불.** eviction 테스트(cap=2 monkeypatch, 호출 시점 상수 읽기) 무영향 — outline/dataclass/prefilter/pool/multilang 등 패밀리 182 passed.

**리스크: 하** | **노력: 0.25d** | **✅ 완료**

### (보류) P12-3. rg 부재용 name→files 역인덱스 — 2048파일 초과 모노레포까지 amortize하는 구조적 해법. cap 상향으로 이번 라운드 체감 충분 — 1d는 다음 기회로.

---

## 📊 전체 15항목 요약 테이블

*(P9 라운드는 본 테이블 집계 이후 신설이므로 요약 테이블에 미포함 — 위 P9 섹션 참조)*

| # | 항목 | 카테고리 | 파일/범위 | 노력 | 리스크 | **검증 상태** |
|---|------|---------|-----------|------|--------|:----------:|
| 1 | `run_repl` 분할 (2217라인) | 🟠 성능 | `asi.py` | 3-5d | 상 | **✅ 완료 (P6-2, `c52891fd`)** |
| 2 | API response 파싱 DRY (7곳) | 🟠 성능 | planner/agent/design_chat | 0.5d | 하 | **✅ 채택** |
| 3 | 정규화 파이프라인 DRY (6건) | 🟠 성능 | `external_llm/` 전역 | 1d | 중 | **⚠️ 조건부** |
| 4 | `write_tools.py` 분할 (6249라인) | 🟡 품질 | `agent/tool_handlers/` | 2-3d | 중 | **✅ 완료** |
| 5 | `_build_engine` param 축소 (13개) | 🟡 품질 | `repl_impl.py` | 1d | 하 | **✅ 완료 (`f681face`)** |
| 6 | `time.sleep` inventory (3건) | 🟡 품질 | stats/radio/orchestrator | 산발 | — | **✅ 채택** |
| 7 | scanner `# noqa: F401` 인식 | 🟢 scanner | `unused_import_scanner.py` | 0.5d | 하 | **✅ 채택 (최우수)** |
| 8 | vulture `# noqa: F841` 인식 | 🟢 scanner | `vulture_scanner.py` | 0.5d | 하 | **✅ 채택** |
| 9 | Genuine FP `# noqa` 추가 (4건) | 🟢 scanner | `operation_executor.py` | 0.1d | 하 | **✅ 채택** |
| 10 | webapp stats `time.sleep(2)` | ⚪ 보류 | `stats.py` ~~546~~ → 619 | 0.25d | 하 | **✅ 완료 (P7-2)** — `asyncio.sleep(2)` 실측 |
| 11 | Popen timeout 추가 (2건) | ⚪ 보류 | `orchestrator.py` | 0.25d | 하 | **❌ 종결 (오진, P7-2)** — `.wait()`/`.communicate()` 0건 |
| 12 | ~~`radio.py` inotify 전환~~ | ⚪ 보류 | `radio.py:160` | 2d | 중 | **❌ 종결 (대상 없음)** |
| 13 | `_tool_web_fetch` SSRF 가드 | 🟠 하드닝 | `web_search_tools.py:2214` | 0.5-1d | 하 | **✅ 완료 (P7-1, `fe542611`)** |
| 14 | P2-4 잔여 3건 종결 | 📋 문서 | `docs/improvement_proposals.md` | 0.1d | 없음 | **✅ 종결 (P7-2)** |
| 15 | silent-except 훅 `--index-only` | 🟢 게이트 | `check_no_new_silent_except.py` | 0.5d | 하 | **✅ 완료 (P7-3)** |

**합계**: 15항목 (P1: 3, P2: 3, P3: 3, P4: 3, P7: 3) | 채택 6, 조건부 2, 종결 3, 완료 4 | 총 노력 추정: **10.5-14일** (+ P7-1 0.5-1d, P7-3 0.5d 별도)

---

## 🎯 우선순위 실행 로드맵

### Phase 1: Scanner noqa 인식 + DRY (1.5일)

**목표**: 가장 안전하고 가시적인 개선. 기존 로직 변경 없음.

| 순서 | 작업 | 노력 | 기대효과 | 리스크 |
|:----:|------|:----:|---------|:------:|
| 1 | **P3-1** `_has_noqa_comment()` + scan loop noqa skip | 0.5d | **107건 false positive 제거** | 하 | ✅ 완료
| 2 | **P3-2** `_source_line_has_noqa()` + vulture noqa:F841 | 0.5d | 향후 FP 방지 | 하 | ✅ 완료
| 3 | **P3-3** 4개 genuine FP `# noqa: F401` 태깅 | 0.1d | 4건 자동 해소 (scanner가 noqa 인식) | 하 | ✅ 완료
| 4 | **P1-2** `extract_llm_content()` 공유 함수 생성 | 0.5d | 중복 7→1 | 하 | ✅ 완료
| 5 | 검증: pytest + ruff + scanner 0 candidates 확인 | 0.25d | 회귀 방지 | — |

**Phase 1 완료 후 기대 (4/6 완료)**:
```
unused_import_scanner: 11개 barrel 파일 → 0 candidates (기존 107건)
ruff check: 0 error (기존 유지)
pytest: 496 passed (회귀 없음)
```

---

### Phase 2: 조건부 항목 수정 적용 (2-3일)

**목표**: P1-3(정규화)과 P2-2(write_tools 분할)를 검증 피드백 반영하여 안전하게 적용.

| 순서 | 작업 | 노력 | 핵심 주의사항 |
|:----:|------|:----:|--------------|
| 6 | **P1-3** `_normalize.py` 생성 + 7곳 마이그레이션 ✅ 완료 | 1d | `normalize_key()` + `strip_lower()`, 7개 사이트 동등성 검증 완료, 6440 passed |
| 7 | **P2-2** write_tools.py → 4개 모듈 분할 ✅ 완료 | 2-3d | write-safety 3층 gate PARITY 유지, barrel re-export 호환 |

---

### Phase 3: 구조 개선 + 잔여 (6-9일)

| 순서 | 작업 | 노력 | 비고 |
|:----:|------|:----:|------|
| 8 | **P2-3** `EngineConfig` dataclass + `_build_engine` 리팩터 | 1d | 13 param → 1 object |
| 9 | ~~**P4-2** Popen timeout 2건 추가~~ | 0.25d | **종결 — 실측 결과 timeout 불필요 패턴 (fire-and-forget 의도 / tracked Popen)** |
| 10 | **P1-1** `run_repl` → 5-7개 함수 분할 | 3-5d | **✅ 완료 (P6-2, `c52891fd`)** — 모듈화로 대체 (repl_impl.py 추출) |
| 11 | **P4-1** webapp stats async sleep | 0.25d | **✅ 완료 (P7-2)** — `await asyncio.sleep(2)` 확인 (`webapp/routes/stats.py:615`) |
| 12 | ~~**P4-3** radio.py → watchdog inotify/kqueue~~ | 2d | **종결 — 폴더 폴링 부재 (L160은 서버 재연결 대기)** |

---

## 🔍 검증 기준

```bash
# 1. 전체 회귀 테스트
python3 -m pytest tests/ -q                          # → 496 passed 유지

# 2. 린트
ruff check                                           # → 0 new error (N814 pre-existing only)

# 3. PytestCollectionWarning
python3 -m pytest --collect-only tests/ 2>&1 | grep -c Warning  # → 0

# 4. Languages 테스트
python3 -m pytest tests/unit/languages/ -q           # → 339 passed, 4 xfailed 유지
```

**Phase 1 전용 검증**:
```bash
# barrel re-export 11개 파일 scan → 0 candidates
python3 -c "
from external_llm.analysis.unused_import_scanner import scan_unused_imports
for f in ['change_spec_assertions.py', 'symbol_handlers.py', 'intent_verifier.py',
          'deterministic_plan_builder.py', 'planner_agent.py', 'planner_helpers.py',
          'operation_executor.py', 'agent_phase_manager.py', 'symbol_handlers_shared.py',
          'collaboration_orchestrator.py']:
    path = 'external_llm/editor/' + ('_editor_core/lane/' if f in ['change_spec_assertions.py','intent_verifier.py','deterministic_plan_builder.py'] else '')
    res = scan_unused_imports(path + f)
    print(f'{f}: {len(res)} candidates')
"
# → 각 0 candidates
```

---

## 💡 실행 권장사항

**Phase 1 → Phase 2 → Phase 3 순서**로 진행하되, **Phase 2는 P1-3(정규화)과 P2-2(write_tools) 중 하나만 선택**해도 무방:

| 우선순위 | 선택 | 이유 |
|:--------:|------|------|
| 🥇 | P3-1 (scanner noqa) | ✅ 완료 — 11개 barrel 파일 107건 FP → 0 |
| 🥈 | P1-2 (API DRY) | ✅ 완료 — 2개 사이트 치환 + 공유 헬퍼 |
| 🥉 | P3-2 / P3-3 (vulture + 태깅) | ✅ 완료 — vulture noqa:F841 인식 + barrel 0 candidates |
| 그 다음 | P2-2 / P2-3 | 조건부 해결 또는 구조 개선 (P1-1·P1-3은 완료) |

각 Phase 완료 후 검증 기준 통과 필수.

---

## 📋 변경 이력

| 날짜 | 변경 | 작성자 |
|------|------|--------|
| 2026-07-16 | 최초 작성 | AI agent |
| 2026-07-16 | 검증 반영: 기각 4건 제거, 조건부 2건 수정, 상태 컬럼 추가 | AI agent |
| 2026-08-01 | P7 라운드 반영: P7-1 SSRF 제안 · P7-2 P2-4 전항 종결 · P7-3 훅 --index-only 제안, 요약 테이블 상태 갱신 (P6-1 완료 커밋 140c8331) | AI agent |
| 2026-08-01 | P7-1 SSRF 가드 완료 (커밋 fe542611) · P7-3 `--index-only` 완료 — 훅/CI 엔트리에 적용, 신규 테스트 4건 (라이브 검증: untracked 파일 존재 중에도 1323==1323 통과) | AI agent |
| 2026-08-02 | R1+R2+R3 MCP Streamable-HTTP 하드닝 (`5512d50c`) · R4 git-context SSOT 위임 (`20c9da68`) | AI agent |
| 2026-08-02 | P6-2 REPL 블록 모듈화 (`c52891fd`, P1-1 종결) · R5 pytest-xdist 기본 병렬화 (`922e0920`) | AI agent |
| 2026-08-02 | P6-3 write_task 하드닝 (`6ee43ed6`, P6 라운드 종결) · C1 문서 스테일 갱신 | AI agent |
| 2026-08-16 | P8 라운드 신설 — P8-1: lint.yml structural-scanner 스텝 `.cache/` actions/cache 재사용 제안 (타당성: mtime_ns fingerprint 보존 + fail-open + 버전 내장 무효화; 리스크 하, push당 ~50s 절감) · P8-2: 게이트 콜드 부트 타이밍 문서화 제안 (근거: P-I 실측 웜 15-18s / 콜드 ~69s) | AI agent |
| 2026-08-20 | P9 라운드 신설 — P9-1: BM25 5중 복제 공식 `agent/bm25.py` 단일 소스 통합 + idf 호이스트 완료 (lock 구간 실측 1.39x, 비트 동일성 계약 test_bm25_core 9건) · P9-2: P8-1/P8-2 상태 스테일 정리 (cb0a7fe3/f249257f 구현 완료 반영) · P9-3: v0.2.27 durations 마이닝 제안(데이터 대기) | AI agent |
| 2026-08-20 | **v0.2.27 릴리스 완료** (public 0efb9c8, 태그 v0.2.27, CI 2워크플로 success, PyPI 0.2.27 라이브). 게이트가 커밋 전에 카탈로그 drift(muse-spark-1.2 소멸, de1f6633) 적중 — 1차 verify 중단을 사전 차단. P9-3 durations 데이터 확보 | AI agent |
| 2026-08-20 | P11 라운드 신설 — P11-1: vector_cache invalidation 테스트 patch 스코프 탈출 버그 수리(autouse fixture, 7.05s→1.50s, HF 오프라인 GREEN) · P11-2: pty 50k 절단 테스트 bracketed paste 전환(8.4s→0.46s, ~18x, 단언 동등) · P4-1 Phase 3 row 11 스테일 정리(P7-2 완료 반영) | AI agent |
| 2026-08-21 | P12 라운드 신설 — P12-1: tier-2 문법 게이트 pure tree-sitter 전환, SyntaxValidator 경유 컴파일러(kotlinc/javac/gcc ~2s) 유입 제거(kotlin tier 2.04s→0.02s, 시맨틱 에러 무력화 버그 수리, 계약테스트 3종) · P12-2: `_PY_FILE_CACHE_MAX_ENTRIES` 512→2048(rg 부재 스래싱 해소, 5.51s→2.34s) + rg_absent 테스트 싱글턴 공유 · P12-3 보류 | AI agent |
| 2026-08-21 | P13 라운드 신설 — P13-1: untracked gate 2종 헤비급 마이닝(unguarded 1.96s / silent_except ~5.1s / ghost-imports 2.0s, 커밋당 워커마다 dup). **unguarded에 per-file `(mtime_ns,size)` fingerprint 디스크 캐시**(`.cache/unguarded_keys_v1.json`, fail-open, A307 계약): 콜드 2.19s→웜 0.046s (**48x**) · **silent_except `--index-only` git show 248회→ `ls-files -s`+`cat-file --batch` 배치화**(cProfile 66%가 fork/exec): 5.1s→1.15s (**4.4x**, 배치==개별 전체 동일성) · P13-2: ghost-imports 3회 전체 ast.walk→모듈 픽스처 1회 공유(3x 파싱 절약). 계약테스트 5종(캐시 히트/증분/손상 fail-open + 배치 동일성/스코프) | AI agent |
| 2026-08-21 | P14-2 완료 — `394164d0`/`b421e47a`/`6a6eab21`: 구조 스캐너 콜드 부트 진단(cProfile: `_read_ast` 미스 2,194회, 944파일 17.24MB 소스 → AST 비용 276MB > 256MiB 예산 LRU 스래싱) → `parse_cache._MAX_CACHE_BYTES` 256→384MiB: 테스트 full-scan 게이트 48.66s→10.27s (**4.7x**), misses 2,194→997, 판정 불변. docstring 수치 콜드/웜 재실측 정정(콜드 ~50s, 웜 CLI ~15s, fresh 프로세스+웜 .cache ~10s). P14-3 진행중 — 웜 게이트 그래프 재사용: vulture preprocess sync **framework 튜플vs리스트 타입 불일치 버그** 수리(웜마다 192/2956 dirty→41MB 재직렬화 10.1s 강제 → dirty 4로 스킵) + **contradictory per-file fingerprint 디스크 캐시 신설**(`.cache/contradictory_scan_v1.json`, dup_dist 키 포함, fail-open, 계약테스트 4종): 웜 게이트 19.4s→9.3s (**2.1x**) | AI agent |
| 2026-08-22 | P15 라운드 (스캔) — **main fast-forward 5커밋 통합** (`git update-ref`, 0574802c~15ed08a9 — 그간 버그픽스 5건이 main에 미반영, 관례 위반 해소). **비버그 검증 3건**: (1) `_build_turn_digest` ImportError 미포함 — `no-new-first-party-import-fallback` 게이트가 이미 "first-party import는 실패 불가" 계약으로 차단 (수정 시도가 게이트에 정확히 걸림 → **게이트가 옳음 확인**, 수정 폐기) (2) `simple_llm_call` silent-failure — 호출부 3곳 모두 빈 문자열→기본값/에러 처리로 안전 (3) lastfailed 스테일 캐시 — 과거 테스트 이름들이 현재 매칭 안 됨 (실제 실패 아님). **남은 신규 후보**: UP045(Optional→`\|`) 전역 ignore 3,692건 전체 자동 fix 가능 — 단, 전역 ignore라 정리해도 재발 차단 안 됨 → "신규 발생 차단 게이트 + staged 마이그레이션" 조합 제안 (P16 후보). | AI agent |
| 2026-08-22 | P16 완료 — `96ff8aa5`: **UP045 전량 마이그레이션 + zero-tolerance 게이트화**. `ruff check --select UP045 --fix .`: 1,521건 (P15 집계 3,692는 전역 ignore 상태에서의 과거 추정치 중복, 실제 doble-width — 180파일, `Optional[X]`→`X \| None`, requires-python>=3.10이라 PEP 604 런타임 안전) + 새 F401 180건(미사용 `typing.Optional` import) 전량 정리 + I001 13건(import 정렬) fix. **pyproject ignore에서 UP045 제거** — full-lint 게이트(`ruff check .` zero-tolerance)가 신규 `Optional` 주석을 자동 차단 (baseline-diff 불필요, zero-tolerance가 정답). 검증: 풀 스위트 16793 passed, compileall clean, pre-commit 16게이트 green. | AI agent |
2026-08-22 | P17 완료 — UP037 전량 마이그레이션 + zero-tolerance 게이트화. ruff check --select UP037 --fix .: 209건/55파일 (quoted-annotation 전부 fix 가능). 안전성 검증: 17개 TYPE_CHECKING 파일 중 15개는 from __future__ import annotations 보유 — future 없는 2개(intent_resolver.py, performance_metrics.py)는 quoted 참조가 자체 모듈 import(OrderedDict, weakref)라 런타임 정의 확인 후 unquote 안전. **pyproject ignore에서 UP037 제거** — full-lint zero-tolerance 게이트가 신규 quoted annotation 자동 차단. 부수 lint 0건 (209건 순수 quote 제거, F401/I001 신규 0). 검증: ruff check . 0건, compileall clean, 풀 스위트 16793 passed (205s), pre-commit 16게이트 green. 남은 전역 ignore: E501(16,665), N806(449), N802(101), N999(46), ASYNC100. | AI agent |
| 2026-08-22 | P18 완료 — `dcc54797`: **N806 전량 마이그레이션 + zero-tolerance 게이트화**. 함수 내 지역 상수 UPPER_CASE → snake_case 70건/37파일 (repl_impl 11건 최다 — `_RichMD`→`_rich_md`·`_M`/`_M2`/`_SEP_W`, providers 4건, context_manager 4건 등). **수동 rename은 3단계 안전장치**: ①AST 좌표 스크립트 2회 실패(유니코드 라인에서 col_offset drift → 라인 병합/오염) 후 **함수 세그먼트 + 토큰 경계 regex**로 전환 (206+/206- 대칭, 라인 병합 0) ②import문 오염 2건 발견·수리 (`_asi_mod._C` 속성, `from rich.text import Text` import문 — N806 토큰이 attribute/import alias까지 매치) ③회귀 23건 → 수리 후 풀 스위트 16793 passed. **pyproject ignore에서 N806 제거** — zero-tolerance 게이트가 신규 지역 상수 자동 차단 (tests/*, tools/* per-file-ignore는 유지 — 테스트/스크립트 컨벤션 보존). pre-commit 15게이트 green. 남은 전역 ignore: E501(16,665), N802(101), N999(46), ASYNC100. | AI agent |
| 2026-08-22 | P19 완료 — `aa25bc63`: **N802 전량 마이그레이션 + zero-tolerance 게이트화**. 실측 12건 (ignore 주석의 101건은 과거 수치): rename 2건 (`_STOP_ITER_FALLBACK`→`_stop_iter_fallback`, `_collect_I_flags`→`_collect_i_flags` — 호출부/테스트 동반) + **noqa 10건** (stdlib/3rd-party 디스패치 프로토콜: `do_GET/do_POST/do_OPTIONS/do_DELETE` 8건 http.server, `visit_FunctionDef/visit_ClassDef` 2건 libcst — rename하면 런타임 디스패치가 깨짐, `# noqa: N802 — stdlib/3rd-party dispatch protocol (name is fixed by caller)`). **pyproject ignore에서 N802 제거** — zero-tolerance 게이트가 신규 위반 자동 차단 (tests/*, tools/* per-file-ignore는 유지). **부수 발견 — 구조 스캐너 per-file 모드가 변경 파일의 pre-existing vulture 후보를 노출**: `c_provider.py` `get_test_command`의 미사용 파라미터(`repo_root`/`test_args`, `return None`만 하는 C-family 계약)가 P16 이전부터 잠복 → `_` 접두사 rename으로 정당 해소 (vulture가 `_` 파라미터 무시, 위치 인자 호출이라 시그니처 불변). **부수 버그 수리 — `b6ef139e`**: `_run_cmd`의 Popen→`os.getpgid` 레이스 (즉시 종료 명령이 reap되면 ProcessLookupError → 크래시, 풀 스위트 flake 2/2 재현) → getpgid 래핑 + pgid=None fallback + 회귀 테스트 2종. 검증: ruff check . 0건, compileall clean, 풀 스위트 16025 passed, pre-commit 15게이트 green. 남은 전역 ignore: E501(16,665), N999(46), ASYNC100. | AI agent |
| 2026-08-22 | P20 완료 — N999 전역 ignore 제거. **실측 0건 확인** (과거 46건 수치는 P17 시점의 스테일 — `__init__.py` per-file-ignore가 이미 존재했고 전역 remain 0). 전역 ignore 제거만으로 완료 — zero-tolerance 게이트(`ruff check .`)가 신규 위반 자동 차단, `__init__.py` per-file-ignore는 유지. 검증: ruff check . 0건, ruff check --select N999 --no-cache . 0건. 남은 전역 ignore: E501(16,665), ASYNC100. | AI agent |

## 2026-08-22 | P21 완료 — ASYNC100 전역 ignore 해제 (config-only)

- 실측: `asyncio.timeout` / `asyncio.wait_for(timeout=...)` 中 **cancel-scope 컨텍스트 매니저(`async with asyncio.timeout`) 사용 0건** — ASYNC100(cancel-scope-no-checkpoint) 트리거 불가 규칙 (기존 주석의 "async-function-with-no-await"는 오타/오기재).
- 전역 ignore에서 `"ASYNC100"` 제거 — `ruff check .` zero-tolerance 게이트가 신규 위반 자동 차단.
- `tests/*` per-file-ignore는 유지.
- 검증: `ruff check .` 0건, `--select ASYNC100` 0건, compileall clean, 풀 스위트 **16795 passed** (5 skipped, 5 xfailed).
- 남은 전역 ignore: `E501`(16,665) 뿐.
## 2026-08-23 | E501 format 연동 라운드 완료 (5커밋)

- **실측**: E501 633건 (스테일 기록 16,665건 아님) — `ruff format` 적용 후 **238건**으로 감소. 잔여 238건 = 문자열 135 + string-assignment 23 + 코드 54 + code+comment 13 + docstring 11 + f-string 2 (대부분 프롬프트/설명/로그 메시지로 코드 래핑 불가).
- **format 적용**: 848파일 reformat을 경로 그룹 4커밋으로 분할 (core 28 / external_llm 179 / webapp 17 / tests+tools 623) — `3de8992d` `6030191a` `7b0a51d7` `dfa61d1a`.
- **format 게이트 신설**: `check_lint_full.py`가 `ruff format --check --diff`도 검사 (zero-tolerance, no baseline). 로컬 pre-commit per-file + CI lint.yml no-args 전수 스캔 동일 게이트. 검출은 returncode + diff 헤더 기준 (stderr 요약 문구는 단일/전체 모드에서 상이). 커밋 `52862a3f`.
- **수리**: `test_slash_dispatch_gate.py` mutation anchor 2건 — format이 alignment 공백 제거로 needle 불일치 → post-format spacing으로 갱신 (47 passed). `dead_params_baseline.txt`에 `webapp/main.py::FastAPI.exception_handler::exc_class` 추가 (format이 스텁 메서드 사이 빈 줄 삽입으로 scope 분리 → NEW 위반 노출, 인터페이스 균일성 예외).
- **새 테스트**: `tests/unit/test_check_lint_full_format.py` 4건 (미포맷 검출/포맷 통과/repo self-check/비py 필터).
- **검증**: 풀 스위트 **16800 passed** (5 skipped, 5 xfailed), pre-commit 15게이트 green, main 동기화 완료.
- **잔여**: E501 전역 ignore 유지 (238건 — 문자열 리터럴 위주, noqa 부착은 노이즈). 다음 라운드: 문자열 E501 정책 (per-file `# ruff: noqa: E501` vs 유지) 검토.
| 2026-08-23 | E501 format 연동 | AI agent |
## 2026-08-23 | E501 per-file 정책 결정 — 전역 ignore 유지 (A안, P21-2)

- **정밀 실측**: E501 잔여 **238건 / 75파일** — AST 정밀 분류 결과 **래핑 가능한 순수 코드 라인 0건**. 전부 데이터: 멀티라인 문자열 내용 52건(26개 문자열) + 단일 문자열 133건 + f-string/문자열 할당 47건 + 주석 7건.
- **noqa 불가 검증 (중요한 함정 발견)**: ruff noqa는 **물리적 라인 단위** — ① 멀티라인 문자열 **시작 라인 trailing noqa는 내용 라인 E501을 커버하지 않음** (최초 프로브는 내용 라인이 우연히 짧아 false negative였음 — exact repro로 반증) ② **내용 라인에 trailing noqa를 넣으면 문자열 데이터로 오염** (시스템 프롬프트가 `\  # noqa: E501`로 시작하는 코드 오염 사고 — 즉시 복구) ③ **`"""\` 백슬래시 시작 라인 trailing noqa는 문자열 오염 + SyntaxWarning** ④ 백슬래시/내용 라인 모두 noqa 불가.
- **시도 → 롤백**: 자동 noqa 스크립트로 230건 적용 시도 → 멀티라인 오염/무효 확인 후 **전부 롤백** (`git restore .`, 70파일 클린 복귀, E501 238건 원상).
- **결정**: 사용자 확인 — **A안 (전역 `"E501"` ignore 유지)**. 238건이 전부 구조적으로 래핑/noqa 불가 데이터이므로 전역 ignore가 정당. 신규 "순수 코드" E501 미보호 리스크는 실측 plain 0건이라 사실상 없음 (format 게이트가 신규 코드 88자 래핑을 강제하므로 plain E501 원천 차단).
- **교훈**: E501 대량 정리 기계 스크립트는 각 변형(내용 포함 시작/백슬래시/f-string/docstring/내용 라인)을 **실제 파일 exact repro로 검증 후** 적용 — 프로브 단독 신뢰 금지.
- **잔여 전역 ignore: `E501` 1건** (영구 유지 결정).

## 2026-08-24 | fast 전단 SSOT 통일 완료 — external_llm/context_collector (P25)

- **목표**: `common.normalize_rel_path_fast` 잔존 (프로덕션 23곳: service 8, patch_engine 10, context_builder 1, output_parser 1, agent_loop 1, context_collector 2) → `path_security.normalize_rel_path` (SSOT)로 전량 통일.
- **판정**: 모든 사이트가 `resolve_inside_repo`와 짝을 이룸 — fast 결과가 그대로 resolve에 전달되므로 SSOT 단일화가 항상 안전. 빈값 처리(`if not tf: return`)는 fast/SSOT 동일 계약. `a/`/`b/` 접두사는 diff 경로 컨벤션이라 SSOT 기본 제거가 옳음 (output_parser `_norm_rel`은 기존에 자체 제거 후 fast 위임 → SSOT로 단순화).
- **변경** (6파일, +49/−43): import 교체 + 23곳 호출 교체. `agent_loop.py`는 기존 `normalize_rel_path` import와 중복 → 기존 1개 유지 + 정렬 (I001). docstring/주석의 fast 언급을 SSOT로 갱신 (현재형만; 과거형 `did not`은 유지).
- **테스트**: `test_context_builder_unit._norm` → SSOT + SSOT 계약 추가 4건 (따옴표/a·b 접두사/traversal/드라이브). **의도된 동작 변화 2건 갱신**: `test_context_collector_bounds`의 traitversal 기대가 `path_outside_repo`(fast가 통과→resolve가 거부) → `empty_target`/`missing_args`(SSOT가 normalize에서 조기 거부) — 목표인 fail-closed 조기 차단.
- **잔존 fast**: `common.py` 정의 자체 + 테스트 5곳 (test_common 직접 테스트, 과거형 기록). 프로덕션 fast 사용 **0건 달성**.
- **검증**: 관련 643 passed, 풀 스위트 전체 실행 (백그라운드), ruff 0건, compileall clean.

## 2026-08-24 | P26 — 게이트 상위 40 재프로파일: 잔여 병목 전부 "본질 + 간헐적 경합" 확인 (최적화 여지 0)

- **방법**: 풀 게이트 (`tests/unit -m "not slow" --durations=40`, `--dist loadgroup`) 재실행 183.34s (16067 passed). 상위 40의 각 항목을 **단독/파일 단독 병렬**로 재측정해 "진짜 비용" vs "간헐적 경합"을 분리.
- **판정표**:

| 게이트 상위 항목 | 게이트서 관측 | 파일 단독 | 판정 |
|---|---|---|---|
| `test_dump_candidates_writes_raw_json_and_exits_zero@repo_scan` | 14.94s | 콜드 5.15s / 웜 2.58s | **본질** — repo_scan 그룹이 콜드 rebuild 전체를 지불 (그래프 1.5s + 스캐너 2.7s + JSON dump 2.3s) |
| context_manager setup/teardown ×10건 (3.2~4.5s) | ~30-60s 누적 | 전부 0.05-0.12s (파일 병렬 77 tests = 2.40s) | **간헐적 경합** — 순수 메모리 연산 |
| `test_vector_cache_checkpoint_failure` 10.2s | 10.2s | 파일 병렬 5 tests = 3.23s (고정 ~1.65s/test = faiss/numpy 임포트) | **간헐적 경합** |
| `test_release_untracked_import_gate` / `test_release_ignored_py_gate` etc. | 3.8-8.2s | 웜에서 2.26-2.58s | warm 캐시 정상 |

- **콜드 비용 분해 (cProfile, 단독)**: JSON encode 2.31s (6회 atomic_write_json, `_iterencode_dict` 0.97s + `_iterencode` 1.86s) + JSON decode 0.57s + gc.collect 0.43s + tree-sitter 0.07s×55. **JSON I/O가 콜드 5.26s의 ~55%**.
- **최적화 후보 전부 닫힘**:
  - JSON encode 최적화 → 이미 **entry-wise streaming** (P0-2, 2026-08-12, 335MB→7.3MB peak, compact separators). 여지 0.
  - 콜드 rebuild → **repo_scan xdist_group 직렬화가 이미 1워커로 집중** (P3/P5). 콜드 병렬 12-15s는 8워커 CPU 경합 분, 단독 5.15s가 본질 하한.
  - 캐시 재사용 → 게이트 순서상 첫 repo_scan만 12-15s, 이후 웜 2.58s. 이미 최선.
  - `test_dump_candidates`가 스캐너 8개 전부 + 그래프 + JSON dump를 실제로 실행 — 테스트가 하는 일 자체가 비싸고, 그 일이 게이트 계약 (후보 dump 정확성).
- **결론**: P1-P6로 병목 후보 (edit_text tsc 스폰 / 캐시 / repo_scan 그룹 / AST 스캔 / xdist dist 모드 / 콜드 rebuild)가 전부 닫힘. 남은 게이트 비용은 "8개 워커가 각자 1회 지불하는 본질 비용 + 부하 지터" — 게이트 총시간이 183-445s 사이로 출렁이는 건 지터 (Variance는 워커 수·머신 부하 의존), 최적화 여지 0.
- **권장**: 추가 병목 헌팅 중단. 게이트 시간 안정화는 `-n auto` 워커 수 조정 등 인프라 영역 (코드 아님).

## 2026-08-24 | P27 — 게이트 워커 수 A/B: `-n auto`(=8) 유지 확정 (n=4/n=2는 확실히 열등)

- **동기**: P26 결론 "게이트 183-445s 출렁임은 부하 지터"에 대한 후속 — M1 8코어(4P+4E) 8GB에서 swap 4.86GB 사용 중이라 "워커를 줄이면 메모리 압박/스왑이 줄어 오히려 빨라질 것"이라는 가설을 실험 (머신: Apple M1, 8GB, swap used 4.86GB).
- **방법**: 동일 트리 (f375e070a), warm 캐시, Chrome 열려 있던 상태 (측정 전), 게이트 그대로 `tests/unit -m "not slow" --durations=15`, `--dist loadgroup` 고정, `-n 8/4/2`만 변화, 런 사이 20s 안정화.
- **결과**:

| n (워커) | 총시간 | 최대 개별 | 비고 |
|---|---|---|---|
| **8 (auto)** | **177.19s** | 12.93s | 정상 (P26의 183s와 동일 범위 → auto=8 확인) |
| 4 | 309.49s (1.75x) | **77.32s** | `test_vector_cache_model_order` 스톨 — faiss/vector-cache가 워커에 직렬 누적 |
| 2 | 701.26s (3.96x) | **112.15s** | `test_save_fsyncs_staged_index_before_rename` 스톨 + Thread warning |

- **가설 기각**: 워커 감소 → 스왑 감소 → 빨라짐. **정반대**. 스왑 used는 실험 전 4.86GB → 후 4.93GB (불변, 스왑은 원인이 아님). 느려진 이유는 **무거운 테스트 클러스터 (vector-cache faiss/numpy, REPL pty spawn)가 소수 워커에 직렬로 몰리는 것** — n=4에서 77s, n=2에서 112s 스톨은 병렬 분산 효용의 역전증거.
- **결론**: `-n auto`(8 workers) 유지가 실측상 최속 & 재현성 최고. 워커 수 조정으로 얻을 시간 없음. 게이트 변동성은 머신 부하 (다른 사용자 프로세스, Chrome 등)가 지배 — 코드/설정으로 해결 불가.
- **자산**: 세 런 로그 `/tmp/wn_8.log`, `/tmp/wn_4.log`, `/tmp/wn_2.log` (duration 상세 포함).

## 2026-08-24 | P28 — CI 워커 유형 검토: ubuntu-latest(4 vCPU)가 이 계정의 유일+최적 옵션 (larger runner 불가)

- **동기**: P26/P27이 "게이트 변동성은 머신 부하 지배, 코드/설정으로 해결 불가"로 닫은 후, 유일하게 남은 조정 축은 CI 러너 하드웨어. "CI 워커 유형을 올리면 (8/16 vCPU larger runner) 게이트가 빨라지는가"를 실측으로 검토.
- **방법**: (1) GitHub 공식 문서로 러너 스펙 확정 (2) 최근 CI 성공 5건의 job 로그로 실측 수집 (3) 로컬 8코어에서 동일 게이트를 `-n 2/4/8`으로 돌려 CI 4-vCPU 조건 시뮬레이션.
- **실측 (러너 스펙)**:
  - `ubuntu-latest` = **4 vCPU / 16 GB RAM** (GitHub-hosted runners 공식 문서; 2026-08-24 확인). CI 로그도 정확히 `4 workers [13424 items]` — `-n auto`가 4로 해석됨.
  - **larger runner (8/16/32 vCPU)는 Team/Enterprise 플랜 전용 + 유료** (Actions runner pricing). 이 계정은 개인 Free (`type: User`, org 없음) → **선택 자체가 불가능**.
- **실측 (CI job)**: 최근 성공 5건 (2026-08-18~23) unit-tests job:
  - 총 소요: 288 / 333 / 331 / 334 / 343s (mean ~326s) — 매우 일관적 (4-vCPU 수렴값).
  - 순수 pytest: **236s** (14:41:35→14:45:31). 나머지 ~107s는 checkout/setup-python/apt/pip install/coverage combine.
- **실측 (로컬 시뮬레이션, 동일 트리 0829be2c1, warm 캐시)**:

| 시나리오 | 총시간 | vs CI |
|---|---|---|
| CI 4-vCPU `-n auto`(4) | 304-343s (mean ~326s) | 기준 |
| 로컬 8코어 `-n 8` | 177-203s | 0.6x |
| 로컬 8코어 `-n 4` | 255-309s | 0.8-1.0x |
| 로컬 8코어 `-n 2` | 630-701s | 2.0x |

- **결론 1 (워커 유형)**: 4-vCPU ubuntu-latest는 (a) 이 계정에서 유일하게 선택 가능한 호스팅 러너, (b) 이미 그 하드웨어의 수렴값(304s, 4 workers, 실패 0)에서 동작. **8-vCPU larger runner로 가면 pytest가 8 workers가 되어 ~203s (1.5-1.7x) 단축 가능하지만 플랜 제약으로 실행 불가능.** 현행 CI 워커 구성은 이 계정에서 최적 — 변경할 코드/설정 없음.
- **결론 2 (발견된 주석 불일치)**: lint.yml `unit-tests` job 주석은 "checks out the FULL private tree"라 하지만, **실제 CI는 public repo `socialherb/asicode`의 main을 체크아웃** (확정: CI head `480ca0d5c`는 로컬에 없는 커밋; public repo tree에 lane//webapp//tools/ 0개). 그 결과:
  - CI 수집 13424 items vs 로컬 16067 — 차이 2645개 = **private 전용 테스트 84개 파일** (lane/, webapp/, tools/ coupled).
  - **webapp extra 설치의 주석 근거("83 files import webapp")는 public 트리에는 해당 없음** — public 트리에 webapp-coupled tests 부재. 설치는 무해하지만 주석이 실제와 다름.
  - lint.yml과 release.yml이 사실상 **같은(exported) 트리**를 검사 — PR 게이트와 릴리스 게이트가 같은 커버리지. private 전용 테스트는 어느 CI job도 안 돎 (의도된 단일-트리 정책으로 보이나, 명시된 적 없음).
- **권장**: 워커 유형 변경 액션 없음. lint.yml 주석 "FULL private tree" → "public exported tree"로 정정 필요. private 전용 테스트를 CI에서도 돌리려면 별도 private-트리 job이 필요 (그 경우 webapp extra 주석이 비로소 의미를 가짐) — 별도 작업으로 분리.

## 2026-08-24 | P29 — private 전용 테스트 CI job 구성: private-tests.yml 구현 (webapp/tools coupled 103개)

- **동기**: P28 결론 2가 남긴 "private 전용 테스트(webapp/, tools/, export-infra coupled)를 CI에서도 돌리려면 별도 private-트리 job 필요"를 실제 구현. public exported 트리(CI)는 webapp/tools가 없어 이 파일들을 원천 실행 불가 — **단일-트리 정책이 설계상 의도**이므로, "public에 없음"을 바로잡는 대신 private 전체 트리를 도는 전용 job을 추가.
- **실측 (후보 정리, P28 이어서)**:
  - 실제 private-only 테스트 파일: **103개** (P28의 "84개"는 import 기반만; 실제로는 excluded conftest fixture를 request하는 테스트까지 재귀 제외되어 +19개 — `is_excluded()`가 SSOT로 정확히 재현, `/tmp/private_only.txt`).
  - private 트리에서 전부 green: **2441 passed in 34.53s** (단일 프로세스, slow 제외) — 완전 실행 가능.
  - structural gate(private 트리)는 baseline 없이도 0 candidates (private은 baseline이 없어도 green).
  - GitHub에 private mirror repo **없음** (계정 `socialherb`: public asicode + AsRecord(private)만) — 로컬 remote 0개 전용.
- **설계 (구현됨)**:
  - `.github/workflows/private-tests.yml` — 독립 워크플로, **private mirror에서만 실행**:
    - 파일 목록을 `scripts/export_public.py --list`로 **동적 생성** (SSOT: export 규칙이 바뀌면 자동 반영 — hardcoded 목록 결함 방지). `comm -23`으로 shipped 제외.
    - `-m "not slow"` (private-only slow 2개 제외) + `-n auto` 단일 프로세스 (10초대).
    - 빈 목록 guard: public 트리에서 실수 실행 시 pytest가 testpaths로 폴백해 전체를 도는 것을 방지 (exit 1).
    - `on: push(main) / pull_request / workflow_dispatch` — 리소스 절약.
  - `scripts/export_public.py` `EXCLUDE_FILES`에 `.github/workflows/private-tests.yml` 추가 — **public export에서 제외** (public 트리에 존재하면 pytest 0개 선택 → fail; 존재를 원천 차단).
  - lint.yml/release.yml과 동일한 install (`. [dev,webapp]`), 동일 Python 3.12.
- **검증 (모두 통과)**:
  - 워크플로 YAML 파싱 OK (주석의 `SSOT:` 콜론이 YAML 스칼라를 깨는 문제 수정).
  - SSOT 목록: shipped 496 / all 599 / private-only 103 (정확).
  - `python -m pytest $(private-only) -q -m "not slow"` → **2441 passed in 34.53s** (+ 무해한 atexit 서브프로세스 종료 워닝).
  - export dry-run: `.github/workflows/`에 lint.yml/release.yml만 존재, private-tests.yml **없음** (제외 확인).
  - export 기반 테스트 3종 (test_export_structural_baseline / test_release_ignored_py_gate / test_precommit_config) 20 passed.
- **결론**: private 전용 103개 테스트가 이제 private mirror push 시 CI에서 실행됨. public 트리는 이 워크플로 자체가 없어 no-op (설계상). private mirror 생성·push는 사용자 승인 후 `gh repo create asicode-private --private --source=. --push` (선택).
