# EXP21 Phase P1~P5 실행 기록 (안정화)

## 개요
- 실행 일시: 2026-02-24
- 목적: EXP20v2 D10의 재현성 불안정(3-run pass-rate 33.3%) 원인 분리 및 안정화 옵션 검증
- 평가 범위: dev 30 + holdout_locked 10 + sealed_holdout 10 = 총 50문항

## 우선순위 5개 (Stabilization Backlog)
1. `answer_postprocess=stability_v1`로 고변동 문항(시점형/보안/공동수급) 후처리 고정
2. `decode_policy=type_v1`로 질문 유형별 temperature 세트 고정
3. 비오라클 선택 전략(`consensus_v1`) 성능/안정성 검증
4. deterministic retrieval + tie-break + `context_hash` 로깅
5. 선택 전략 가드(`consensus_guarded_v1`)로 비오라클 드리프트 완화

## Phase 구성

| Phase | Mode | 핵심 변경 |
|------|------|----------|
| P1 | `e21_p1` | Priority #1 |
| P2 | `e21_p2` | P1 + Priority #2 |
| P3 | `e21_p3` | P2 + Priority #3 + #4 |
| P4 | `e21_p4` | P2 + Priority #4 (oracle 선택 유지) |
| P5 | `e21_p5` | P4 + Priority #5 |

## 성능 결과 (vs D10)

| Split | D10 | P1 | P2 | P3 | P4 | P5 |
|------|-----|----|----|----|----|----|
| Dev | **1.0000** | **1.0000** | **1.0000** | 0.9555 | 0.9737 | 0.9737 |
| Holdout | 0.9549 | **0.9933** | 0.9822 | 0.9711 | 0.9778 | 0.9600 |
| Sealed | 0.9818 | **0.9909** | 0.9818 | **0.9909** | **0.9909** | 0.9842 |
| Overall | 0.9874 | **0.9968** | 0.9928 | 0.9657 | 0.9779 | 0.9731 |
| Perfect | 46/50 | **48/50** | 47/50 | 45/50 | 47/50 | 45/50 |
| Gate(Dev/Hold/Sealed) | ✅/✅/✅ | ✅/✅/✅ | ✅/✅/✅ | ❌/✅/✅ | ❌/✅/✅ | ❌/✅/✅ |

## 실행 이슈
- `e21_p2` 31/50에서 OpenAI timeout 1회 발생
- `--fresh` 재시작 대신 resume(`--mode e21_p2`)로 이어서 완료, 최종 산출물 정상 생성

## 핵심 관찰
- `P1`이 최고 성능이자 3/3 gate 통과: D10 대비 holdout +3.84pp, overall +0.95pp.
- `P2`도 3/3 gate 통과했지만 `P1` 대비 소폭 하락(특히 holdout -1.11pp).
- `P3`(consensus_v1)에서 dev 급락. `doc_D/security(0.211)`, `doc_E/evaluation(0.455)`가 치명적.
- `P4`는 consensus를 제거해도 dev가 0.9737로 미달. `deterministic_retrieval` 적용 시 특정 문항(특히 `doc_D/security`) 회귀가 재현됨.
- `P5`(guarded consensus)도 dev 미달. 현재 데이터 기준 비오라클 선택 전략은 성능 이득보다 손실이 큼.

## 결론
- EXP21 최종 권장안: **`P1 (exp21_p1_postprocess_stability)`**
- 채택: Priority #1
- 보류/기각: Priority #3~#5 조합(`P3/P4/P5`)
- 선택적 적용: Priority #2(`P2`)는 추가 이득이 제한적이어서 기본값 미채택

## P1-R 재현성 검증 (3-run)

### 실행
- 동일 설정 `mode=e21_p1 --fresh`를 3회 독립 실행
- 산출물 분리 저장: `*_run1`, `*_run2`, `*_run3`

### 결과 요약

| Run | Overall | Dev | Holdout | Sealed | Gate (dev/holdout/sealed) | Overall Gate |
|-----|---------|-----|---------|--------|-----------------------------|--------------|
| run1 | 0.9964 | 1.0000 | 0.9822 | 1.0000 | ✅ / ✅ / ✅ | ✅ |
| run2 | 0.9968 | 1.0000 | 1.0000 | 0.9842 | ✅ / ✅ / ✅ | ✅ |
| run3 | 0.9906 | 1.0000 | 0.9711 | 0.9818 | ✅ / ✅ / ✅ | ✅ |

### Pass-rate (3-run)
- dev gate: 3/3 (100%)
- holdout gate: 3/3 (100%)
- sealed gate: 3/3 (100%)
- overall gate(동시 통과): 3/3 (100%)

### 변동 핵심 문항

| Split | doc_key | 문항 | run1 | run2 | run3 | range |
|------|---------|------|------|------|------|-------|
| holdout_locked | hold_H | 상황관리시스템이 제공해야 하는 주요 기능 | 0.889 | 1.000 | 0.778 | **0.222** |
| sealed_holdout | hold_H | 보안 의무/위반 책임 | 1.000 | 0.909 | 0.818 | **0.182** |
| sealed_holdout | hold_J | 저작권/권리분쟁 책임 | 1.000 | 0.933 | 1.000 | 0.067 |
| holdout_locked | hold_F | 주요 업무 프로세스/기능 | 0.933 | 1.000 | 0.933 | 0.067 |

### 판정
- P1은 단일 run뿐 아니라 3-run 기준으로도 **안정적 3/3 gate 통과**.
- D10-R(33.3% pass-rate) 대비 재현성이 구조적으로 개선됨.

## 산출물
- `data/experiments/exp21_phase_p1_metrics.csv`
- `data/experiments/exp21_phase_p1_report.json`
- `data/experiments/exp21_phase_p1_metrics_run1.csv`
- `data/experiments/exp21_phase_p1_report_run1.json`
- `data/experiments/exp21_phase_p1_metrics_run2.csv`
- `data/experiments/exp21_phase_p1_report_run2.json`
- `data/experiments/exp21_phase_p1_metrics_run3.csv`
- `data/experiments/exp21_phase_p1_report_run3.json`
- `data/experiments/exp21_phase_p1_stability_runs.csv`
- `data/experiments/exp21_phase_p1_stability_summary.json`
- `data/experiments/exp21_phase_p1_stability_question_variance.csv`
- `data/experiments/exp21_phase_p2_metrics.csv`
- `data/experiments/exp21_phase_p2_report.json`
- `data/experiments/exp21_phase_p3_metrics.csv`
- `data/experiments/exp21_phase_p3_report.json`
- `data/experiments/exp21_phase_p4_metrics.csv`
- `data/experiments/exp21_phase_p4_report.json`
- `data/experiments/exp21_phase_p5_metrics.csv`
- `data/experiments/exp21_phase_p5_report.json`
- `scripts/run_exp19_phase_d_eval.py` (`e21_p1~p5`, 안정화/선택전략 모드 추가)
