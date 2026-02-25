# RAG 최적화 고도화 브리핑 (EXP01~22)

작성일: 2026-02-25  
작성 목적: 팀 회의 공유용 요약 (의사결정 흐름 + 현재 성능 + 다음 단계)

---

## 1) 한 줄 결론

- **성능 상한(oracle)**: EXP21 P1 기준 `0.9968` (48/50), P1-R 3-run gate pass-rate `100%`
- **실운영 기준(non-oracle)**: EXP22 기준 `kw_v5=0.9783`, `Faithfulness=0.9382`, `Context Recall=0.9767`
- 즉, **“높은 정답률 + 근거성 검증 가능한 평가 체계”로 전환 완료**

---

## 2) 실험 1~22 요약 (무엇을 했고, 왜 그렇게 결정했는가)

## EXP01~09: 단일문서 최적화 → 일반화 실패 확인

| 실험 | 핵심 변경 | 결과/의사결정 |
|---|---|---|
| EXP01 | chunk=500, layout baseline | 기준선 확보 |
| EXP01-v2 | Table+Metadata(T+M) | CR 개선, 구조 정보 활용 방향 채택 |
| EXP02 | hybrid(alpha/top_k) | 일부 개선, 하이브리드 유지 |
| EXP03 | 한국어 zero-shot prompt | faithfulness 우수, 한국어 프롬프트 방향 채택 |
| EXP04 | metadata+verbatim ablation | 효과 제한적 |
| EXP04-v3 | reranker@50 | 단일문서 best 도달 |
| EXP05 | extract prompt + elbow cut | 미세조정 효과 제한 |
| EXP06 | normalization + 3-run 평균 | 변동성 관리 원칙 도입 |
| EXP07 | table-aware 평가 | text-table gap 확인 |
| EXP08 | 100문서 EDA | 코퍼스 특성/분포 파악 |
| EXP09 | 일반화 검증 | 대폭 하락 → **단일문서 최적화 과적합 결론** |

근거 문서: `docs/planning/HANDOFF_v2_next_experiments.md`

## EXP10~12: 일반화 복구 (파싱/리트리벌 재구축)

| 실험 | 핵심 변경 | 결과/의사결정 |
|---|---|---|
| EXP10 (A~E) | 다문서 testset 재구성 + 파싱/청킹/리트리벌 재설계 | `c500_pv2` 계열이 실질 baseline으로 정착 |
| EXP11 | generation 중심 개선 시도(F~J) | 다수 변형이 baseline 미달 → 과도한 prompt 제약 기각 |
| EXP12 | retrieval 파라미터 최적화 | `multi_query`가 best (`kw_v3=0.900`) |

## EXP13~18: 실패 원인 분해 후 정밀 개선

| 실험 | 핵심 변경 | 결과/의사결정 |
|---|---|---|
| EXP13 | contextual retrieval | 전반 하락 → 기각 |
| EXP14 | 오답 진단 (retrieval vs generation) | 개선 우선순위 분리 완료 |
| EXP15 | SC 3-shot generation 개선 | `kw_v3=0.9258`으로 상승 |
| EXP16 | metric v4 + SC5 검증 | `kw_v4=0.9534`, SC5 단독 증분 제한 |
| EXP17 | metric v5 + 0.99 도전 | `0.9547`, 0.99 미달 |
| EXP18 | GT 정제 + targeted prompt | `kw_v5=0.9851` 도달 |

## EXP19~21: 0.99 달성, 일반화/재현성 검증

| 실험 | 핵심 변경 | 결과/의사결정 |
|---|---|---|
| EXP19 | targeted 보강 | `kw_v5=0.9952` (29/30) |
| EXP19 B/C/D | holdout·sealed 포함 일반화 검증 | 과적합 보정, 구조-aware retrieval 정착 |
| EXP20 D9 | metric v5b(슬래시/괄호/공백 보정) | holdout/sealed gate 개선 |
| EXP20v2 D10 | evaluation 후처리 | 단일 run 3/3 gate 통과, 하지만 D10-R 불안정(33.3%) |
| EXP21 P1~P5 | 안정화 실험 | **P1 최종 best** `0.9968` |
| EXP21 P1-R | 3-run 재현성 | **3/3 run 모두 gate 통과(100%)** |

## EXP22: 평가 방법론 고도화 (non-oracle + LLM Judge)

| 실험 | 핵심 변경 | 결과/의사결정 |
|---|---|---|
| EXP22 | `first_deterministic` (GT 비의존) + RAGAS(Faithfulness, Context Recall) | **실운영 기준 성능 확정** + 다차원 품질 검증 체계 도입 |

---

## 3) 지표 의미 (회의용)

| 지표 | 의미 | 해석 포인트 |
|---|---|---|
| `kw_v5` | 실제 선택 답변의 키워드 정합도 (현재 non-oracle 기준) | 최종 사용자 체감 정답률 근사 |
| `kw_v5_oracle` | SC 후보 중 GT 기준 최고 점수 | “후보 안에 정답이 있었는가” 상한 |
| `oracle gap` | `kw_v5_oracle - kw_v5` | 선택전략 손실 크기 |
| `Faithfulness` | 답변이 context 근거에 충실한지 | hallucination 위험 지표 |
| `Context Recall` | 검색 context가 정답 근거를 얼마나 포함하는지 | retriever 커버리지 지표 |
| `Gate` | dev≥0.99, holdout≥0.95, sealed≥0.95 | 운영 통과 기준 |

---

## 4) 현재 최종 성능 (보고용 숫자)

### 4-1. 실운영 기준 (EXP22, non-oracle)

| Split | kw_v5 | kw_v5_oracle | Faithfulness | Context Recall |
|---|---:|---:|---:|---:|
| Dev | 0.9818 | 1.0000 | 0.9302 | 0.9611 |
| Holdout | 0.9620 | 1.0000 | 0.9155 | 1.0000 |
| Sealed | 0.9842 | 1.0000 | 0.9848 | 1.0000 |
| **Overall** | **0.9783** | **1.0000** | **0.9382** | **0.9767** |

- Oracle gap: **2.17pp**
- RAGAS: `valid_n=50`, `error_n=0` (100% 성공)

### 4-2. 성능 상한/안정성 (EXP21 P1)

- EXP21 P1 best: **0.9968 (48/50)**
- P1-R: **3-run gate pass-rate 100%**

---

## 5) Mismatch 3건 검수 결과 (반영 완료)

| # | 케이스 | 원인 | 판정 |
|---|---|---|---|
| 1 | doc_D/하자담보 (faith=0.0) | Judge context에 chapter prefix 미전달 | Judge false negative |
| 2 | doc_D/보안준수사항 (faith=0.41) | chapter prefix 누락 + 일부 문장 직접 인용 한계 | Partial false negative |
| 3 | doc_E/평가방식 (kw=0.45) | temp=0.0 shot이 불완전 답변 선택 | 실질 SC 선택 손실 |

요약:
- **Case 1~2는 Judge 측 context 불일치 이슈**
- **실질 손실은 Case 3 1건**
- 심각한 hallucination으로 단정할 케이스는 없음

---

## 6) 재현성 현황 (회의 시점 안내)

- EXP21(P1): 재현성 검증 완료 (3-run, 100% gate pass)
- EXP22: 재현성 추가 실험 진행 중  
  - 현재 저장된 run1/run2 결과 존재  
  - run3 완료 후 EXP22-R 최종 pass-rate 확정 예정

---

## 7) 차후 방향 (내 담당 고도화 로드맵)

## A. 속도/비용 단축 (Quantization + 호출 최적화)

목표:
- 품질 하락 최소화(<=0.5pp)로 latency/cost 절감

실행안:
1. **Reranker 양자화 실험**
- 대상: `bge-reranker-v2-m3`
- 비교: FP16 vs INT8(우선) vs INT4(선택)
- 지표: `kw_v5`, faithfulness, p95 latency, GPU 메모리

2. **SC 호출 축소**
- 기본 `first_deterministic` 1-shot
- 저신뢰 문항만 추가 샷(선택적 SC)
- 목표: 평균 호출 수 30~50% 절감

3. **Judge 비용 캡 유지**
- `top_k_judge=10`, `max_chars=15k` 기본 유지
- 필요 시 문항 유형별 동적 cap 적용

성공 기준(제안):
- 품질: overall kw_v5 하락 <= 0.5pp
- 신뢰성: faithfulness 하락 <= 0.02
- 속도: end-to-end p95 latency 25% 이상 단축

## B. 다문서 동시 처리 (Multi-Document Pipeline)

목표:
- 여러 문서 업로드 시 병렬 처리 + 문서 간 혼선 최소화

실행안:
1. **Doc Router 계층 추가**
- 질문별 상위 M개 문서 선별 (doc-level retrieval)

2. **문서별 병렬 Retrieval/Rerank**
- 각 문서에서 top-k 추출 후 글로벌 merge(RRF/score fusion)

3. **출처 강제 답변 포맷**
- 답변에 `doc_id/chunk_idx` 근거 태깅

4. **평가셋 확장**
- cross-doc 질문 포함 벤치마크 추가
- 추가 지표: 문서 식별 정확도(doc attribution accuracy)

성공 기준(제안):
- 단일문서 대비 품질 하락 <= 1.0pp
- 문서 식별 정확도 >= 95%
- 동시 3문서 기준 처리시간 선형 악화 방지

---

## 8) 회의에서 바로 말할 핵심 메시지

1. 우리는 이미 **상한 성능(0.9968)**과 **운영 성능(0.9783)**을 분리해 설명할 수 있다.  
2. EXP22로 **평가 신뢰도(oracle 제거 + LLM Judge)**를 확보했다.  
3. 다음 스프린트의 초점은 **품질 유지한 속도 최적화**와 **다문서 동시 처리 아키텍처**다.

