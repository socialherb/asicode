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
    _KINDS,                    # ← scanner flag
    _TIER1_KINDS,              # ← scanner flag
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
    rest = line_text[idx + 1:].strip()
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

### P8-1. [제안: 채택 검토] `lint.yml` structural-scanner 스텝 `actions/cache`로 `.cache/` 재사용

**파일**: `.github/workflows/lint.yml` — "Check ZERO deterministic structural scanner candidates" 스텝(`run: python scripts/check_structural_scanners.py --gate-only`)

**현황**: `.cache/`는 gitignore(`.gitignore:26`)라 CI checkout에 존재하지 않음 → **매 push마다 콜드 부트**. 실측: 웜 15-18s → 콜드 ~69.3s (**3.9-4.6배**, 게이트 全 스텝 중 최대 단일 비용). `--gate-only`는 8개 스캐너 + 그래프 빌드 전부를 요구하므로 캐시 히트 시 실질 절감은 50s+.

**타당성 (코드로 검증됨)**:
- 캐시 fingerprint는 `(path, mtime_ns, size)` — `actions/cache`(gzip 압축 tar)로 복원하면 mtime/size가 **원본 그대로 보존**되어 스탬프 일치 → 히트.
- 파이썬 파일이 **변경된 경우 mtime_ns/size가 바뀌어 미스** → 그 파일만 재분석(self-heal, 변경 후 첫 실행이 콜드 부트와 동일 결과).
- B2(`73baaf43`) 원자 쓰기 계약: 캐시는 완전 payload 1개 (`atomic_write_json`/streaming temp+replace) — CI 복원 파일이 손상/절단돼도 **fail-open**으로 풀 재분석 (정확성 무영향).
- `CACHE_VERSION`/`_DBX_CACHE_VERSION`/`_CRX_CACHE_VERSION` 등 버전 키가 payload에 내장 — 스캐너 로직 변경 시 자동 무효화 (수동 버전 범프 불필요).

**리스크: 하** | **노력: 0.25d** | **절감: push당 ~50s** (콜드 69s → 웜 ~18s)

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

### P8-2. [문서: 반영] `check_structural_scanners.py` 게이트 타이밍 주석 갱신

**파일**: `tests/unit/test_check_structural_scanners.py:714-715` (`~21s warm` 주석 → P-I 실측 반영)

**현황**: 웜/콜드 실측값이 문서화된 곳이 없어 CI 캐시 설계자가 기대값을 알 수 없음. P-I(`35f52561`)가 2파일 갱신하며 `tests/unit/test_check_structural_scanners.py:714-715`에 `15-18s warm / 69s cold --gate-only`를 남겼으나, `check_structural_scanners.py` 헤더/도움말에는 콜드 부트 언급이 없음.

**제안**: `scripts/check_structural_scanners.py` docstring/`--help`에 실측 추가 — `--gate-only` 콜드 ~69s (fresh repo/CI, 캐시 부재 시) / 웜 15-18s. "CI 캐시 도입 시 웜에 근접" 기대값 명시.

**리스크: 없음** | **노력: 0.1d**

---

## 📊 전체 15항목 요약 테이블

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
| 11 | **P4-1** webapp stats async sleep | 0.25d | `asyncio.sleep(2)` |
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
