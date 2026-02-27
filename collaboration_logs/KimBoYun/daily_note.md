# 협업일지

## 0203

GCP VM 세팅 및 계정 권한 부여.

가이드에 맞게 각자 RAG 구현해보고 다음주쯤 종합하기로 결정.

RAG
- parser 구현
- hwp/docs to pdf
hwp, word 등 포맷이 다 다름. -> 어떻게 pdf로 합치지?

-> win32com 사용하면 가능. but window환경만 가능. -> 전처리 모듈은 윈도우에서 처리하도록 해야.

## 0204
- parser 구현 시 메타데이터 작업을 어떻게?
문서와 메타데이터 연결 -> doc_id

멘토님 조언:
langraph 기반
RAG를 외부 라이브러리 사용해야 성능이 잘 나올것

## 0205
ingest.py를 통한 문서 최적화.(hwp, pdf, docx를 pdf로 일괄 변환된 문서를 메타데이터와 연결)
doc + meta -> doc_id

## 0206
index, 규칙기반 retriever 작성.
-> 각 파일간의 연결이 좀 어려운 것 같음.

## 0209
ingest, parse, indexing과 retriever 간의 연결에 진척이 별로 없었다.
기존의 Langchain LCEL기반이 아닌 Retriver로 구현을 했었는데, 구현이 복잡해지고 프로젝트 목표(langchain 기반 RAG 구현)에도 맞지 않는 것으로 판단이 되어서 전체 코드를 바꾸고 있다.

## 0210
클로드를 이용한 바이브 코딩을 위해, 설계서를 다시 작성했다.
이후 베이스 파이프라인을 작성하였다.
작동은 하지만 아직 미흡하고 일부는 아직 제대로 동작되지 않는 부분이 있다. 우선 파이프라인을 완성한 후 수정할 계획이다.

## 0211
현재 검색된 청크의 점수가 좋지 않은 부분이 많다. 따라서 retriever 이전 부분의 흐름을 검토하고 수정해야 한다.
마찬가지로 retriever가 가져온 정보에서 키워드와 관련된 근거인지 확인하는 절차를 더 보강해야할 것 같다.

## 0212
QueryAnalisys 도입
사용자의 질의를 분석해 질의 유형을 분류하고, 어떤 필드를 추출해야 하는지 결정하는     
모듈.

  사용자 쿼리
    → ChatOpenAI (LangChain LCEL chain)
    → JsonOutputParser
    → QueryAnalysis (Pydantic 모델)

+ 중간 발표 준비.

## 0213

Decision 모듈 도입

Generator가 추출한 RFPSummary를 바탕으로 룰 기반 입찰 추천 의사결정을 내리는
모듈입니다.


RFPSummary + DocumentMetadata
    → 3가지 룰 평가 (각 신호: green/yellow/red)
    → 신호 집계
    → DecisionSupport (추천 + 사유)


QueryAnalyzer  →  (query_type 결정)  →  Retriever / Generator
                                                      ↓
                                               RFPSummary
                                                      ↓
  DecisionEngine  →  (룰 평가)  →  DecisionSupport  →  최종 JSON 출력

## 0223
기존 코드를 통합하고 웹화면 구현을 위한 프론트엔드, 벡엔드를 구현을 계획.
임시 streamlit, FastAPI를 활용한 구현.

인증 및 권한 관리 (Authentication & Authorization)
로그인 구현: 간이 로그인 (Streamlit Authenticator) - users.yaml 기반 비밀번호 인증.
Phase 2: SSO 연동 (Google Workspace, LDAP, AD) - 사내 계정 통합. -> SaaS 필수적이므로 불가능
세션 관리:
로그인 상태 유지 및 자동 로그아웃.
2.2 멀티 테넌시 데이터 구조 (Multi-tenancy Data)
현재: data/processed/에 모든 파일 저장, storage.py가 전역 공유. 목표: 사용자별 데이터 격리

스토리지 구조 변경:
data/{user_id}/processed/: 개인별 분석 파일 격리.
data/shared/: 팀 공유/공개 파일 저장소. -> 팀 워크스페이스 개발
DB 도입 검토:
파일(JSON) 기반 관리의 한계(검색/조회/동시성) 극복을 위해 SQLite 도입.
User, RFP, Extraction 테이블 설계.

팀 워크스페이스를 통해 팀 간의 코멘트, 작업 확인 가능. 팀장은 프로필 수정 가능.


## 0224
멀티 테넌시 구조로 변환.
로그인 및 세션 구현. 서버가 실행될 때, 새롭게 세션 쿠키를 생성.

## 0225
SQLite DB 도입 : 기존 json형식으로 저장되던 데이터, 계정정보를 db형식으로 저장하게 마이그레이션 진행.
팀 워크스페이스 개발 : 실제 현장과 유사하게 팀단위로 진행될 것을 고려하여 팀별 워크스페이스를 구성하여 자유롭게 문서를 업로드하고 보고 및 피드백을 받을 수 있도록 구현.
회사 또는 팀이 다르면 당연히 해당 자료를 볼 수 없도록 구현.

## 0226
프로젝트 테스트, npm 오류 분석
Next.js로 프론트를 이전함에 따른 라이브러리 오류가 발생함. 프론트 문제였을 것으로 추측 
-> 창현님이 해결

## 0227
협업 일지 정리, 프로젝트 테스트