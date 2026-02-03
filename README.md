# BidFlow: 보안 강화형 지능형 입찰 분석 시스템

## 프로젝트 개요
BidFlow는 입찰 공고(RFP)를 분석하여 필수/결격/독소 조항을 추출하고, 회사 프로필과 비교하여 적격 여부를 판정하는 AI 시스템입니다.

## 주요 기능
- **RFP 자동 분석**: 필수 요건 및 제약 사항 구조적 추출
- **적격성 판정**: 회사 스펙과 비교하여 Go/No-Go 판정 (Validator)
- **근거 하이라이트**: 판정 근거가 되는 원문 위치(텍스트/표) 제공
- **보안 강화**: OWASP 기반 프롬프트 인젝션 방어 및 3-Rail 보안 아키텍처

## 설치 및 실행
```bash
# 가상환경 생성 및 의존성 설치
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# 실행 (개발 모드)
uvicorn src.bidflow.apps.api.main:app --reload
streamlit run src/bidflow/apps/ui/Home.py
```
