# EXP20v2 Phase D10 실행 기록

## 개요
- 실행 일시: 2026-02-24 13:20 ~ 14:13 (약 53분)
- 실험 모드: `d10` (`EXP20v2_Phase_D10`)
- 목적: EXP20 D9의 dev 잔여 2건(evaluation) 보완으로 dev gate(≥0.99) 재도전
- 평가 범위: dev 30 + holdout_locked 10 + sealed_holdout 10 = 총 50문항

## 변경 사항
- 기준 파이프라인: D9와 동일
  - prompt v5, structure-aware(TOC + chapter prefix), top_k=20, pool_size=50
  - SC 5-shot: `[0.0, 0.1, 0.2, 0.3, 0.5]`
  - metric: kw_v5b
- 추가된 후처리(`answer_postprocess=eval_v1`)
  - `doc_D` 유형: 답변에 `나. 제안서 평가 방법`만 있고 문맥에 `다. 제안서 평가 기준`이 있으면 보완
  - `doc_E` 유형: TOC 파싱 아티팩트 `471./472.`를 `1./2.`로 보정하고 문장 연결어(`다루며`) 보완

## 성능 결과

| Split | D9 | D10 | Delta | Gate |
|-------|----|-----|-------|------|
| Dev | 0.9854 (28/30) | **1.0000 (30/30)** | **+1.46pp** | ✅ (≥0.99) |
| Holdout | 0.9616 (8/10) | 0.9549 (7/10) | -0.67pp | ✅ (≥0.95) |
| Sealed | 0.9818 (9/10) | 0.9818 (9/10) | ±0.00pp | ✅ (≥0.95) |
| Overall | 0.9799 (45/50) | **0.9874 (46/50)** | **+0.75pp** | — |

## 주요 변화 문항
- 개선
  - dev/doc_D evaluation: `0.833 -> 1.000`
  - dev/doc_E evaluation: `0.727 -> 1.000`
  - holdout/hold_H procurement: `0.889 -> 1.000`
- 하락(동일 설정 내 SC 변동)
  - holdout/hold_F technical: `1.000 -> 0.933`
  - holdout/hold_H technical: `1.000 -> 0.889`

## 잔여 Non-perfect (4건)
- holdout_locked / hold_F / technical / 0.933
- holdout_locked / hold_G / compliance / 0.727
- holdout_locked / hold_H / technical / 0.889
- sealed_holdout / hold_H / compliance / 0.818

## 판정
- Dev gate: ✅
- Holdout gate: ✅
- Sealed gate: ✅
- **단일 실행(run1) 기준 3/3 Gate 통과**

## 산출물
- `data/experiments/exp20v2_phase_d10_metrics.csv`
- `data/experiments/exp20v2_phase_d10_report.json`
- `scripts/run_exp19_phase_d_eval.py` (`d10` 모드 + `answer_postprocess=eval_v1`)

## 재현성 검증 (추가 2회, 총 3회)

### 3회 결과 요약

| Run | Overall | Dev | Holdout | Sealed | Gate (dev/holdout/sealed) |
|-----|---------|-----|---------|--------|-----------------------------|
| run1 | 0.9874 | 1.0000 | 0.9549 | 0.9818 | ✅ / ✅ / ✅ |
| run2 | 0.9750 | 0.9733 | 0.9549 | 1.0000 | ❌ / ✅ / ✅ |
| run3 | 0.9789 | 1.0000 | 0.9125 | 0.9818 | ✅ / ❌ / ✅ |

### Pass-rate (3회)
- dev gate pass-rate: 2/3 (66.7%)
- holdout gate pass-rate: 2/3 (66.7%)
- sealed gate pass-rate: 3/3 (100%)
- overall gate(pass all): 1/3 (33.3%)

### 주요 변동 문항
- dev/doc_C: `1.0 -> 0.2 -> 1.0` (range 0.8)
- holdout/hold_H(공동수급): `1.0 -> 1.0 -> 0.556` (range 0.444)
- holdout/hold_G(보안요건): `0.727 -> 0.727 -> 0.636` (range 0.091)

### 결론
- D10은 **단일 run에서 3/3 gate 통과 가능**함을 확인
- 하지만 **재현성은 불충분**(overall gate pass-rate 33.3%)하여 안정적 통과라고 보기 어려움
- 즉, 현재 상태는 **“달성 가능성 확인” 단계**이며, 운영 기준으로는 추가 안정화가 필요

### 재현성 산출물
- `data/experiments/exp20v2_phase_d10_metrics_run1.csv`
- `data/experiments/exp20v2_phase_d10_report_run1.json`
- `data/experiments/exp20v2_phase_d10_metrics_run2.csv`
- `data/experiments/exp20v2_phase_d10_report_run2.json`
- `data/experiments/exp20v2_phase_d10_metrics_run3.csv`
- `data/experiments/exp20v2_phase_d10_report_run3.json`
- `data/experiments/exp20v2_phase_d10_stability_runs.csv`
- `data/experiments/exp20v2_phase_d10_stability_summary.json`
- `data/experiments/exp20v2_phase_d10_stability_question_variance.csv`
