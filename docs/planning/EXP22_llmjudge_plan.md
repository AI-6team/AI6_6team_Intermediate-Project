# EXP22: 평가 방식 개선 — LLM Judge 다차원 평가 + Oracle 누수 제거

## Context

**현재 상태**: EXP21 P1 = 프로젝트 best (overall=0.9968, 3-run 100% gate pass)

**4가지 핵심 문제**:
1. **Oracle 선택 누수**: P1은 GT로 best answer를 고름 → 실제 운영과 괴리, RAGAS 평가 시 과대추정
2. **단일 지표**: kw_v5b만 사용 → precision/faithfulness 미측정
3. **RAGAS API 리스크**: gpt-5-mini temperature=0.01 미지원 → FixedTempChatOpenAI 필수
4. **긴 context**: top_k=20 평균 12,800chars(max 32,168) → Judge에 전량 전달 시 비용/불안정

**검증 완료**:
- `ragas==0.4.3` 설치 확인
- FixedTempChatOpenAI + Faithfulness/ContextRecall 한국어 정상 동작 확인
- faithful=1.0, hallucinated=0.0 올바르게 판별

## 변경 사항 (5개)

### 1. Oracle 누수 제거: `first_deterministic` 선택 모드

**파일**: `scripts/run_exp19_phase_d_eval.py` (invoke_sc 내 selection 분기)

```python
elif selected_mode == "first_deterministic":
    # temp=0.0 첫 번째 shot을 그대로 사용 (GT 비의존)
    best_answer = answers[0]  # SC config[0] = (0.0, model)
    best_score = individual_scores[0]
    selected_index = 0
    selection_note = "first_deterministic"
```

- EXP22 모드에서 `selection_mode="first_deterministic"` 설정
- **추가 기록**: `kw_v5_oracle` (oracle best score)도 CSV에 함께 저장 → oracle gap 측정용
- 기존 P1 config/결과는 변경 없음

### 2. RAGAS 평가 함수 (Faithfulness + Context Recall)

**파일**: `scripts/run_exp19_phase_d_eval.py` (새 함수)

```python
def evaluate_ragas_single(question, answer, contexts, reference, ragas_llm, ragas_metrics):
    """RAGAS 메트릭으로 단일 문항 평가.

    Args:
        contexts: list[str] — Judge용 context (top_k_judge개, max_chars 제한)
        reference: str — ground truth
        ragas_llm: LangchainLLMWrapper(FixedTempChatOpenAI)
        ragas_metrics: [Faithfulness, ContextRecall] 인스턴스

    Returns:
        dict: {faithfulness, context_recall, ragas_status, ragas_error}
    """
```

핵심:
- `ragas.dataset_schema.SingleTurnSample` + `EvaluationDataset` 사용
- 개별 문항씩 `ragas.evaluate()` 호출 (batch=1, 실패 격리)
- 실패 시 `ragas_status="error"`, `ragas_error=str(e)`, 점수는 NaN

### 3. Judge용 Context Cap

**전략**: top_k_judge=10 + max_chars=15,000

```python
def prepare_judge_contexts(docs, top_k_judge=10, max_chars=15000):
    """Judge용 context 준비: 상위 N개 chunk, 총 문자수 제한"""
    contexts = []
    total = 0
    for doc in docs[:top_k_judge]:
        text = doc.page_content
        if total + len(text) > max_chars:
            break
        contexts.append(text)
        total += len(text)
    return contexts
```

- Retrieval은 기존대로 top_k=20 (kw_v5 평가용)
- Judge에는 상위 10개, 15K자 cap (평균 ~8.5K로 대부분 전량 포함)

### 4. 실패 복구 + 집계 신뢰도

**CSV 추가 컬럼** (6개):
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `kw_v5_oracle` | float | oracle best kw_v5 (기존 방식 비교용) |
| `faithfulness` | float/NaN | RAGAS Faithfulness |
| `context_recall` | float/NaN | RAGAS Context Recall |
| `ragas_status` | str | "ok" / "error" / "skipped" |
| `ragas_error` | str | 에러 메시지 (정상 시 빈 문자열) |
| `judge_context_chars` | int | Judge에 전달된 context 총 문자수 |

**Report 집계** (build_report 수정):
```python
"ragas": {
    "faithfulness_mean": float,
    "context_recall_mean": float,
    "valid_n": int,          # ragas_status=="ok" 건수
    "error_n": int,          # ragas_status=="error" 건수
    "mismatch_cases": [      # kw_v5≥0.9 & faith<0.5 또는 kw_v5<0.7 & faith≥0.8
        {"question": str, "kw_v5": float, "faithfulness": float, ...}
    ]
},
"per_split": {
    "dev": {
        "kw_v5": float,
        "kw_v5_oracle": float,     # oracle 비교
        "faithfulness_mean": float,
        "context_recall_mean": float,
        "ragas_valid_n": int,
    },
    ...
}
```

### 5. EXP22 모드 등록

```python
"e22": {
    "label": "EXP22_LLMJudge",
    "desc": "P1 config + non-oracle selection + RAGAS (Faithfulness, ContextRecall)",
    "prompt_path": Path("scripts/prompts/exp19_phase_d_prompt_v5.txt"),
    "pool_size": 50,
    "query_expansion": False,
    "sc_config": SC_5SHOT_CONFIGS,
    "variant_weights": None,
    "top_k": 20,
    "structure_aware": True,
    "include_sealed": True,
    "answer_postprocess": "stability_v1",
    "selection_mode": "first_deterministic",  # GT 비의존!
    "ragas_enabled": True,                     # RAGAS 평가 활성화
    "csv_path": Path("data/experiments/exp22_llmjudge_metrics.csv"),
    "report_path": Path("data/experiments/exp22_llmjudge_report.json"),
    "config_name": "exp22_llmjudge_nogt",
}
```

## 수정 파일

| 파일 | 변경 |
|------|------|
| `scripts/run_exp19_phase_d_eval.py` | (1) `first_deterministic` 선택모드 (2) `evaluate_ragas_single()` (3) `prepare_judge_contexts()` (4) CSV 6컬럼 추가 (5) report 집계 확장 (6) e22 모드 |

## 실행 계획

```bash
cd bidflow
python -X utf8 scripts/run_exp19_phase_d_eval.py --mode e22 --fresh
```

## 비용/시간 예상

- SC 5-shot: 50Q × 5 = 250 calls (기존과 동일)
- RAGAS: 50Q × ~3 calls(faith 2 + ctx_recall 1) = ~150 추가 calls
- 총: ~400 calls, 예상 ~75분
- Judge context cap 덕에 대형 문서도 안정적

## 검증 방법

1. CSV에 faithfulness/context_recall/ragas_status 컬럼 존재 확인
2. `ragas_status=="ok"` 비율 ≥90% (5건 이하 에러)
3. report.json에 `ragas.mismatch_cases` 리스트 존재
4. `kw_v5` vs `kw_v5_oracle` 차이 = oracle gap 정량화
5. `kw_v5` 높고 `faithfulness` 낮은 케이스 있으면 → hallucination 위험 식별

## 실행 전 체크포인트

### CP1: kw_v5 / kw_v5_oracle 정의 고정
- **kw_v5**: `first_deterministic`로 선택된 답변(temp=0.0, GT 비의존)의 keyword_accuracy_v5 점수
- **kw_v5_oracle**: 동일 SC 5-shot 중 GT 기준 최고 점수 (기존 oracle 방식, 비교 전용)
- CSV에 두 컬럼 모두 기록, report에서 split별 양쪽 평균 집계
- gate 판정은 **kw_v5 (non-oracle) 기준** — oracle은 참고용

### CP2: RAGAS 실패 처리/집계 정책
- **개별 문항 단위 호출**: batch=1로 격리 → 1건 실패가 전체에 전파되지 않음
- **실패 시**: faithfulness=NaN, context_recall=NaN, ragas_status="error", ragas_error=에러 메시지
- **집계**: NaN 제외 평균 (pandas .mean() 기본 동작), valid_n/error_n 별도 기록
- **허용 기준**: error_n ≤ 5 (50건 중 10% 이하)이면 집계 신뢰, 초과 시 경고 출력
- **timeout**: per-question ragas evaluate에 timeout=120s 설정, 초과 시 error 처리

### CP3: Resume 재시도 정책
- 기존 resume 로직 유지: CSV에 이미 처리된 `split::question` 키는 skip
- RAGAS 컬럼이 NaN인 기존 행도 "완료"로 간주 (RAGAS 실패는 재시도하지 않음)
- `--fresh` 플래그 사용 시에만 전량 재실행
- 중간 중단 후 재시작: `python -X utf8 scripts/run_exp19_phase_d_eval.py --mode e22` (fresh 없이)

## 문서화 정책

- **계획 문서**: `docs/planning/EXP22_llmjudge_plan.md` — 이 계획 파일 내용 기록
- **실행 기록**: `docs/planning/EXP22_llmjudge_execution.md` — 실행 후 결과/분석
- **HISTORY 업데이트**: `docs/planning/HISTORY_v2_execution.md` — 요약 테이블 행 추가 + EXP22 섹션
- **MEMORY 업데이트**: 프로젝트 메모리에 최종 상태 반영

## 기대 결과

- kw_v5 (non-oracle) ≈ P3 수준 (0.95~0.97) — oracle 제거로 하락 예상
- kw_v5_oracle ≈ P1 수준 (0.99+) — 기존 성능 유지 확인
- faithfulness ≈ 0.85~0.95 — 대부분 context 기반 답변
- context_recall ≈ 0.80~0.90 — top_k=20 retrieval이 GT 커버하는 정도
- mismatch 0~3건 — 교차 검증으로 약점 식별
