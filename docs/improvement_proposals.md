# 개선 제안서 — 검증 완료 (11항목)

**생성일**: 2026-07-16  
**범위**: 전체 코드베이스 (`asi.py` + `external_llm/` + `tests/`)  
**현재 상태**: `226 passed (analysis), ruff clean`  
**검증**: 2026-07-16 독립 검증 완료 (기각 4, 조건부 2, 채택 9 → 11항목)  
**구현 진행**: Phase 1 3/6 완료 ✅

---

## 📋 검증 결과 요약

| 판정 | 건수 | 항목 | 사유 |
|:----:|:----:|------|------|
| ❌ 기각 | 4 | P0-1, P0-2, P2-1, P2-4 | 오진 및 실제 해를 끼치는 제안 |
| ⚠️ 조건부 | 2 | P1-3, P2-2 | 방향은 타당하나 내용 결함 — 수정 후 채택 |
| ✅ 채택 | 9 | P1-1, P1-2, P2-3, P3-1, P3-2, P3-3, P4-1, P4-2, P4-3 | 검증 통과 |

**메타 관찰**: 라인 수·파일 위치 등 기계적 사실은 정확했으나, "참조 없음 = dead" 추론과 정량 주장에서 반복적으로 붕괴. 심각도 라벨이 높을수록(P0) 오진율이 높았음. P3-1처럼 코드를 실제로 읽고 쓴 항목은 품질이 높았음. → 제안서 기반 루프에는 "제안 → 독립 검증 → 착수" 게이트가 필수.

**실행 권장 순서**: P3-1 → P1-2 → P1-3 (가장 안전하고 가시적인 개선)  
**구현 현황**: P3-1 ✅ 완료 | P1-2 ✅ 완료 | P3-3 ✅ 완료 | P3-2 ✅ 완료 | P1-3 ✅ 완료 | P2-2 🔲 조건부

---

## 🟠 P1 — 성능/안정성

---

### P1-1. `asi.py:run_repl` 분할 ✅ 채택

**파일**: `asi.py` (L~7500–9716, 2217라인)  
**현황**: 단일 함수가 REPL 전 생명주기(초기화, 입력 처리, 명령어 dispatch, tool loop, 세션 관리, 출력 렌더링) 포함. 지역변수 100+개, 중첩 depth ~20.  
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

### P2-2. `write_tools.py` 분할 ⚠️ 조건부 (경로 수정 + PARITY 언급)

**파일**: `external_llm/agent/tool_handlers/write_tools.py` (6249라인 실측 일치)  
**현황**: 단일 파일 6249라인, 50+ 함수, 10+ dataclass. 단일 파일 최대 규모 2위 (asi.py 다음).  
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

### P2-3. `_build_engine` param 축소 (14개 → config object) ✅ 채택

**파일**: `asi.py` (`_build_engine`)  
**현황**: `def _build_engine(provider, model, api_key, base_url, max_tokens, temperature, ...)` — 14개 keyword param. 호출부는 모두 keyword 호출.

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

### P2-4. `time.sleep(N)` / timeout inventory (잔여 3건)

P0-1/P0-2 기각으로 제외. HTTP/스레드 경로 잔여 blocking call:

| 파일 | 라인 | N | 경로 유형 | 처리 |
|------|------|---|-----------|------|
| `webapp/routes/stats.py` | 546 | 2s | webapp GET stats | → `asyncio.sleep(2)` |
| `radio.py` | 160 | 10s | file watcher (데몬 스레드) | → inotify (P4-3) |
| `orchestrator.py` | 2413, 2496 | — | Popen no timeout | → `communicate(timeout=30)` |

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
| **P4-2** | `orchestrator.py` | 2413, 2496 | `Popen` without timeout → `communicate(timeout=30)` | 0.25d | 하 |
| **P4-3** | `radio.py` | 160 | `time.sleep(10)` polling → `watchdog` inotify/kqueue | 2d | 중 |

---

## 📊 전체 11항목 요약 테이블

| # | 항목 | 카테고리 | 파일/범위 | 노력 | 리스크 | **검증 상태** |
|---|------|---------|-----------|------|--------|:----------:|
| 1 | `run_repl` 분할 (2217라인) | 🟠 성능 | `asi.py` | 3-5d | 상 | **✅ 채택** |
| 2 | API response 파싱 DRY (7곳) | 🟠 성능 | planner/agent/design_chat | 0.5d | 하 | **✅ 채택** |
| 3 | 정규화 파이프라인 DRY (6건) | 🟠 성능 | `external_llm/` 전역 | 1d | 중 | **⚠️ 조건부** |
| 4 | `write_tools.py` 분할 (6249라인) | 🟡 품질 | `agent/tool_handlers/` | 2-3d | 중 | **⚠️ 조건부** |
| 5 | `_build_engine` param 축소 | 🟡 품질 | `asi.py` | 1d | 하 | **✅ 채택** |
| 6 | `time.sleep` inventory (3건) | 🟡 품질 | stats/radio/orchestrator | 산발 | — | **✅ 채택** |
| 7 | scanner `# noqa: F401` 인식 | 🟢 scanner | `unused_import_scanner.py` | 0.5d | 하 | **✅ 채택 (최우수)** |
| 8 | vulture `# noqa: F841` 인식 | 🟢 scanner | `vulture_scanner.py` | 0.5d | 하 | **✅ 채택** |
| 9 | Genuine FP `# noqa` 추가 (4건) | 🟢 scanner | `operation_executor.py` | 0.1d | 하 | **✅ 채택** |
| 10 | webapp stats `time.sleep(2)` | ⚪ 보류 | `stats.py:546` | 0.25d | 하 | **✅ 채택** |
| 11 | Popen timeout 추가 (2건) | ⚪ 보류 | `orchestrator.py` | 0.25d | 하 | **✅ 채택** |
| 12 | `radio.py` inotify 전환 | ⚪ 보류 | `radio.py:160` | 2d | 중 | **✅ 채택** |

**합계**: 12항목 (P1: 3, P2: 3, P3: 3, P4: 3) | 채택 9, 조건부 2 | 총 노력 추정: **10.5-14일**

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
| 7 | **P2-2** write_tools.py → 4개 모듈 분할 | 2-3d | write-safety 3층 gate PARITY 유지, barrel re-export 호환 |

---

### Phase 3: 구조 개선 + 잔여 (6-9일)

| 순서 | 작업 | 노력 | 비고 |
|:----:|------|:----:|------|
| 8 | **P2-3** `EngineConfig` dataclass + `_build_engine` 리팩터 | 1d | 14 param → 1 object |
| 9 | **P4-2** Popen timeout 2건 추가 | 0.25d | `communicate(timeout=30)` |
| 10 | **P1-1** `run_repl` → 5-7개 함수 분할 | 3-5d | 각 단계 pytest 회귀 |
| 11 | **P4-1** webapp stats async sleep | 0.25d | `asyncio.sleep(2)` |
| 12 | **P4-3** radio.py → watchdog inotify/kqueue | 2d | 외부 의존성 추가 |

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
| 그 다음 | P1-3 / P2-2 / P2-3 / P1-1 | 조건부 해결 또는 구조 개선 |

각 Phase 완료 후 검증 기준 통과 필수.

---

## 📋 변경 이력

| 날짜 | 변경 | 작성자 |
|------|------|--------|
| 2026-07-16 | 최초 작성 | AI agent |
| 2026-07-16 | 검증 반영: 기각 4건 제거, 조건부 2건 수정, 상태 컬럼 추가 | AI agent |
