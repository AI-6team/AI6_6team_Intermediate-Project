"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { getCurrentUser, getDocuments, getExtractionResult, runExtraction, runValidation, RFPDocument } from '@/lib/api';
import UserHeader from '@/components/UserHeader';
import Modal from '@/components/Modal';
import validationIcon from '../images/validation.png';

const VALIDATION_RUNNING_KEY = "validation_running_doc_hash";

function ValidationResultItem({ res, index }: { res: any, index: number }) {
  const [expanded, setExpanded] = useState(false);
  const [reasonExpanded, setReasonExpanded] = useState(false);
  const evidenceText = res.evidence?.[0]?.text_snippet || "";
  const isLong = evidenceText.length > 150;
  const reasonText = res.reasons?.[0] || "이유 없음";
  const isReasonLong = reasonText.length > 150;
  const isPass = res.decision === 'GREEN';
  const isFail = res.decision === 'RED';

  return (
    <div className={`group bg-white dark:bg-gray-800 rounded-xl p-5 border transition-all duration-200 hover:shadow-md ${
      isPass 
        ? 'border-green-200 dark:border-green-900/50 hover:border-green-300' 
        : isFail 
          ? 'border-red-200 dark:border-red-900/50 hover:border-red-300' 
          : 'border-gray-200 dark:border-gray-700'
    }`}>
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
            #{res.rule_id || index + 1}
          </span>
          <h3 className="font-bold text-lg text-gray-900 dark:text-white">{res.slot_key}</h3>
        </div>
        <span className={`shrink-0 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide border ${
          isPass 
            ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-300 dark:border-green-800' 
            : isFail 
              ? 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800' 
              : 'bg-gray-100 text-gray-600 border-gray-200 dark:bg-gray-700 dark:text-gray-300'
        }`}>
          {res.decision}
        </span>
      </div>
      
      <div className="mb-4">
        <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
          <span className="font-semibold text-gray-900 dark:text-white mr-2">판정 이유:</span>
          {isReasonLong && !reasonExpanded ? `${reasonText.slice(0, 150)}...` : reasonText}
          {isReasonLong && (
            <button
              onClick={() => setReasonExpanded(!reasonExpanded)}
              className="ml-2 text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 text-xs font-medium focus:outline-none"
            >
              {reasonExpanded ? "접기" : "전체 보기"}
            </button>
          )}
        </p>
      </div>

      {evidenceText && (
        <div className="bg-gray-50 dark:bg-gray-700/30 rounded-lg p-3 text-sm border border-gray-100 dark:border-gray-700/50">
          <div className="flex items-start gap-2 text-gray-600 dark:text-gray-400">
            <span className="shrink-0 font-semibold text-xs uppercase text-indigo-500 dark:text-indigo-400 mt-0.5">Evidence</span>
            <div className="flex-1">
              <p className="italic">
                "{isLong && !expanded ? `${evidenceText.slice(0, 150)}...` : evidenceText}"
              </p>
              {isLong && (
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="mt-1 text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 text-xs font-medium focus:outline-none"
                >
                  {expanded ? "접기" : "전체 보기"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ValidationPage() {
  const router = useRouter();

  // localStorage에서 캐시된 값으로 즉시 초기화 (페이지 이동 후 복귀 시 빈 화면 방지)
  const [companyName, setCompanyName] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("cached_team") || "";
    return "";
  });
  const [licenses, setLicenses] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("cached_licenses") || "";
    return "";
  });
  const [region, setRegion] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("cached_region") || "";
    return "";
  });

  const [documents, setDocuments] = useState<RFPDocument[]>(() => {
    if (typeof window !== "undefined") {
      try { return JSON.parse(localStorage.getItem("cached_docs") || "[]"); } catch { return []; }
    }
    return [];
  });
  const [selectedDocId, setSelectedDocId] = useState<string>(() => {
    if (typeof window !== "undefined") return localStorage.getItem("cached_validation_docId") || "";
    return "";
  });
  const [validating, setValidating] = useState(false);
  const [validationResults, setValidationResults] = useState<any[] | null>(() => {
    if (typeof window !== "undefined") {
      const docId = localStorage.getItem("cached_validation_docId") || "";
      if (docId) {
        try { return JSON.parse(localStorage.getItem(`validation_result_${docId}`) || "null"); } catch { return null; }
      }
    }
    return null;
  });
  const [error, setError] = useState<string | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);

  useEffect(() => {
    const fetchUserInfo = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        setShowAuthModal(true);
        return;
      }

      try {
        const data = await getCurrentUser();
        if (!data) return;
        setCompanyName(data.team || "");
        setLicenses(data.licenses || "");
        setRegion(data.region || "");
        // 캐시에 저장
        localStorage.setItem("cached_team", data.team || "");
        localStorage.setItem("cached_licenses", data.licenses || "");
        localStorage.setItem("cached_region", data.region || "");
      } catch (error) {
        console.error("Error fetching user info:", error);
      }
    };
    fetchUserInfo();
  }, [router]);

  useEffect(() => {
    const fetchDocs = async () => {
      const token = localStorage.getItem("token");
      if (!token) return;
      try {
        const docs: RFPDocument[] = await getDocuments();
        setDocuments(docs);
        localStorage.setItem("cached_docs", JSON.stringify(docs));
        // 캐시된 선택이 없거나 유효하지 않을 때만 첫 문서 선택
        const cachedDocId = localStorage.getItem("cached_validation_docId") || "";
        const isCachedValid = cachedDocId && docs.some(d => (d.id || d.doc_hash) === cachedDocId);
        if (!isCachedValid && docs.length > 0) {
          const newDocId = docs[0].id || docs[0].doc_hash;
          setSelectedDocId(newDocId);
          localStorage.setItem("cached_validation_docId", newDocId);
        }
      } catch (error) {
        console.error("Error fetching documents for validation page:", error);
        setError("문서 목록을 불러올 수 없습니다.");
      }
    };
    fetchDocs();
  }, []);

  // 문서 선택 변경 시 자동 검증 실행
  useEffect(() => {
    if (!selectedDocId) return;

    // 캐시된 결과가 있으면 먼저 표시
    const savedValidation = localStorage.getItem(`validation_result_${selectedDocId}`);
    if (savedValidation) {
      try {
        const cached = JSON.parse(savedValidation);
        if (cached && Array.isArray(cached)) {
          setValidationResults(cached);
          return; // 캐시 결과 있으면 재검증 생략
        }
      } catch (e) {
        // 캐시 파싱 실패 시 재검증 진행
      }
    }

    // 캐시 없으면 결과 초기화 후 자동 검증 실행
    setValidationResults(null);
    handleValidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDocId]);

  const stats = validationResults ? {
    total: validationResults.length,
    passed: validationResults.filter((r: any) => r.decision === 'GREEN').length,
    failed: validationResults.filter((r: any) => r.decision === 'RED').length,
    pending: validationResults.filter((r: any) => r.decision !== 'GREEN' && r.decision !== 'RED').length,
  } : { total: 0, passed: 0, failed: 0, pending: 0 };

  const handleValidate = async () => {
    if (!selectedDocId) return;
    setValidating(true);
    setError(null);
    const selectedDoc = documents.find(d => (d.id || d.doc_hash) === selectedDocId);
    const docHash = selectedDoc?.doc_hash || selectedDocId;
    localStorage.setItem(VALIDATION_RUNNING_KEY, docHash);
    // 기존 결과를 유지하면서 로딩 표시 (null로 초기화하지 않음)

    try {
      let analysisData = null;

      // 1. 로컬 스토리지에서 분석 결과 확인
      const savedResult = localStorage.getItem(`analysis_result_${selectedDocId}`) || localStorage.getItem(`analysis_result_${docHash}`);
      if (savedResult) {
        try {
          analysisData = JSON.parse(savedResult);
        } catch (e) {
          console.error("Error parsing saved result:", e);
        }
      }

      // 2. 로컬 스토리지에 없으면 API 호출 (Fallback)
      if (!analysisData) {
        const existingResult = await getExtractionResult(docHash);
        if (existingResult?.data) {
          analysisData = existingResult.data;
          localStorage.setItem(`analysis_result_${selectedDocId}`, JSON.stringify(analysisData));
          localStorage.setItem(`analysis_result_${docHash}`, JSON.stringify(analysisData));
        } else {
          const extractionRes = await runExtraction(docHash);
          if (extractionRes && extractionRes.data) {
            analysisData = extractionRes.data;
            localStorage.setItem(`analysis_result_${selectedDocId}`, JSON.stringify(analysisData));
            localStorage.setItem(`analysis_result_${docHash}`, JSON.stringify(analysisData));
          }
        }
      }

      if (analysisData) {
        const g3 = analysisData.g3 || analysisData.matrix || {}; // g3 또는 matrix 키 확인

        // 2. 검증 요청 페이로드 구성
        const matrix = {
          doc_hash: docHash,
          slots: { ...g3 }
        };

        // 3. 검증 실행
        try {
          const results = await runValidation(matrix);
          if (results) {
            setValidationResults(results);
            localStorage.setItem(`validation_result_${selectedDocId}`, JSON.stringify(results));
          } else {
            setError("검증 결과가 비어있습니다.");
          }
        } catch (e: any) {
          console.error(e);
          setError(e.message || "검증 요청 중 오류가 발생했습니다.");
        }
      } else {
        setError("분석 데이터를 찾을 수 없습니다. 먼저 '분석 결과' 페이지에서 분석을 실행해주세요.");
      }
    } catch (e: any) {
      setError(e?.message || "검증 중 오류가 발생했습니다.");
    } finally {
      if (localStorage.getItem(VALIDATION_RUNNING_KEY) === docHash) {
        localStorage.removeItem(VALIDATION_RUNNING_KEY);
      }
      setValidating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <Modal 
          isOpen={showAuthModal} 
          onClose={() => {
            setShowAuthModal(false);
            router.push("/");
          }}
          title="로그인 필요"
        >
          로그인 후 이용해 주세요.
        </Modal>
        <header className="mb-8 flex justify-between items-end">
          <div className="flex items-center gap-4">
            <div className="bg-white dark:bg-gray-800 p-3 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
              <Image src={validationIcon} alt="Validation" width={40} height={40} className="object-contain" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">자격 검증</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">분석된 RFP 문서에 대해 회사의 자격 요건(면허, 실적 등) 충족 여부를 검증합니다.</p>
            </div>
          </div>
          <UserHeader />
        </header>

        {/* 1. 회사 프로필 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <h2 className="text-lg font-bold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
            <span className="w-1 h-6 bg-indigo-500 rounded-full"></span>
            회사 프로필
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">회사명</label>
              <input
                type="text"
                value={companyName}
                readOnly
                className="block w-full rounded-lg border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 shadow-sm sm:text-sm p-3 border cursor-not-allowed"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">보유 면허 (쉼표로 구분)</label>
              <input
                type="text"
                value={licenses}
                readOnly
                className="block w-full rounded-lg border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 shadow-sm sm:text-sm p-3 border cursor-not-allowed"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">지역</label>
              <input
                type="text"
                value={region}
                readOnly
                className="block w-full rounded-lg border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 shadow-sm sm:text-sm p-3 border cursor-not-allowed"
              />
            </div>
          </div>
        </div>

        {/* 2. 문서 선택 및 검증 실행 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <div className="flex flex-col sm:flex-row gap-4 items-end">
            <div className="flex-1 w-full">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">검증할 문서 선택</label>
              <div className="relative">
                <select
                  value={selectedDocId}
                  onChange={(e) => { setSelectedDocId(e.target.value); localStorage.setItem("cached_validation_docId", e.target.value); }}
                  className="appearance-none block w-full pl-4 pr-10 py-3 text-base border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-lg border shadow-sm"
                >
                  {documents.map(doc => (
                    <option key={doc.id || doc.doc_hash} value={doc.id || doc.doc_hash}>
                      {doc.filename}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-500">
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                </div>
              </div>
            </div>
            <button
              onClick={handleValidate}
              disabled={validating || !selectedDocId}
              className={`w-full sm:w-auto px-8 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white transition-all duration-200
                ${validating || !selectedDocId 
                  ? 'bg-indigo-400 cursor-not-allowed opacity-70' 
                  : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-md focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                }`}
            >
              {validating ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  검증 중...
                </span>
              ) : '재검증'}
            </button>
          </div>
        </div>

        {/* 에러 메시지 */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-center gap-3 text-red-700 dark:text-red-400">
            <svg className="w-6 h-6 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            <span className="font-medium">{error}</span>
          </div>
        )}

        {/* 3. 검증 결과 */}
        {validating && !validationResults && (
          <div className="flex justify-center items-center py-16">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-100 border-t-indigo-600"></div>
          </div>
        )}
        {validationResults ? (
          <div className={`space-y-6 ${validating ? 'opacity-60 pointer-events-none' : ''}`}>
            {/* 요약 카드 */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col items-center justify-center">
                <span className="text-sm font-medium text-gray-500 dark:text-gray-400">총 검증 항목</span>
                <span className="text-3xl font-bold text-gray-900 dark:text-white mt-2">{stats.total}</span>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl shadow-sm border border-green-100 dark:border-green-800 flex flex-col items-center justify-center">
                <span className="text-sm font-medium text-green-600 dark:text-green-400">통과 (Pass)</span>
                <span className="text-3xl font-bold text-green-700 dark:text-green-300 mt-2">{stats.passed}</span>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl shadow-sm border border-red-100 dark:border-red-800 flex flex-col items-center justify-center">
                <span className="text-sm font-medium text-red-600 dark:text-red-400">실패 (Fail)</span>
                <span className="text-3xl font-bold text-red-700 dark:text-red-300 mt-2">{stats.failed}</span>
              </div>
              <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col items-center justify-center">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">보류 (Pending)</span>
                <span className="text-3xl font-bold text-gray-700 dark:text-gray-300 mt-2">{stats.pending}</span>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white">상세 검증 리포트</h3>
              </div>
              <div className="p-6 space-y-4">
                {validationResults.map((res, idx) => (
                  <ValidationResultItem key={idx} res={res} index={idx} />
                ))}
              </div>
            </div>
          </div>
        ) : (
          !validating && !error && (
            <div className="flex flex-col items-center justify-center py-24 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-dashed border-gray-300 dark:border-gray-700 text-center">
              <div className="w-20 h-20 bg-gray-50 dark:bg-gray-700 rounded-full flex items-center justify-center mb-6">
                <span className="text-4xl">✅</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">검증 결과가 없습니다</h3>
              <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
                문서를 선택하면 자동으로 자격 검증이 실행됩니다. 결과가 표시되지 않으면 <span className="font-semibold text-indigo-600 dark:text-indigo-400">재검증</span> 버튼을 클릭하세요.
              </p>
            </div>
          )
        )}
      </div>
    </div>
  );
}
