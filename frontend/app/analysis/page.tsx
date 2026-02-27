"use client";

import { useState, useEffect, useRef, Suspense, type ReactNode } from 'react';
import Image from 'next/image';
import { useSearchParams, useRouter } from 'next/navigation';
import { getDocuments, runExtraction, getExtractionResult, RFPDocument } from '@/lib/api';
import UserHeader from '@/components/UserHeader';
import Modal from '@/components/Modal';
import CommentSection from '@/components/CommentSection';
import analysisIcon from '../images/analysis.png';

const ANALYSIS_RUNNING_KEY = "analysis_running_doc_hash";
type RecordValue = Record<string, unknown>;

interface ExtractionEvidence {
  page_no?: number;
  text_snippet?: string;
  [key: string]: unknown;
}

interface SlotObject {
  value?: unknown;
  evidence?: ExtractionEvidence[];
  [key: string]: unknown;
}

type SlotValue = SlotObject | string | number | boolean | null | undefined;

interface ExtractionViewData {
  g1?: Record<string, SlotValue>;
  g2?: Record<string, SlotValue>;
  g3?: Record<string, SlotValue>;
  g4?: unknown;
  [key: string]: unknown;
}

function isRecord(value: unknown): value is RecordValue {
  return typeof value === "object" && value !== null;
}

function isSlotObject(value: unknown): value is SlotObject {
  return isRecord(value) && ("value" in value || "evidence" in value);
}

export default function AnalysisPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <AnalysisContent />
    </Suspense>
  );
}

function AnalysisContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [documents, setDocuments] = useState<RFPDocument[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<ExtractionViewData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("g1");
  const [showAuthModal, setShowAuthModal] = useState(false);
  const selectedDoc = documents.find((d) => (d.id || d.doc_hash) === selectedDocId);
  const selectedDocHash = selectedDoc?.doc_hash || selectedDocId;
  const isMountedRef = useRef(true);
  useEffect(() => { return () => { isMountedRef.current = false; }; }, []);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthModal(true);
    }
  }, [router]);

  useEffect(() => {
    const fetchDocs = async () => {
      const token = localStorage.getItem("token");
      if (!token) return;

      const applyDocs = (docs: RFPDocument[]) => {
        setDocuments(docs);
        const paramDocId = searchParams.get('docId');
        if (paramDocId && docs.some(d => (d.id || d.doc_hash) === paramDocId)) {
          setSelectedDocId(paramDocId);
        } else if (docs.length > 0) {
          setSelectedDocId(docs[0].id || docs[0].doc_hash);
        }
      };

      try {
        const docs: RFPDocument[] = await getDocuments();
        localStorage.setItem("cached_docs_analysis", JSON.stringify(docs));
        applyDocs(docs);
      } catch (error) {
        console.error("Error fetching documents for analysis page:", error);
        // Backend may be busy with extraction - fall back to cached documents
        const cachedDocs = localStorage.getItem("cached_docs_analysis");
        if (cachedDocs) {
          try {
            applyDocs(JSON.parse(cachedDocs));
          } catch {
            setError("문서 목록을 불러올 수 없습니다.");
          }
        } else {
          setError("문서 목록을 불러올 수 없습니다.");
        }
      }
    };
    fetchDocs();
  }, [searchParams]);

  // 문서 선택 변경 시 서버 → localStorage 순서로 결과 불러오기
  useEffect(() => {
    if (!selectedDocHash) return;

    const loadResult = async () => {
      // 1. 서버에서 저장된 결과 조회
      try {
        const serverResult = await getExtractionResult(selectedDocHash);
        if (serverResult?.data) {
          setResult(serverResult.data);
          localStorage.setItem(`analysis_result_${selectedDocId}`, JSON.stringify(serverResult.data));
          localStorage.setItem(`analysis_result_${selectedDocHash}`, JSON.stringify(serverResult.data));
          if (localStorage.getItem(ANALYSIS_RUNNING_KEY) === selectedDocHash) {
            localStorage.removeItem(ANALYSIS_RUNNING_KEY);
          }
          setAnalyzing(false);
          return;
        }
      } catch {
        // Server may be busy with extraction, fall through to localStorage
      }

      // 2. 서버에 없거나 접근 불가면 localStorage fallback
      const savedResult = localStorage.getItem(`analysis_result_${selectedDocId}`) || localStorage.getItem(`analysis_result_${selectedDocHash}`);
      if (savedResult) {
        try {
          setResult(JSON.parse(savedResult));
        } catch {
          setResult(null);
        }
      } else {
        setResult(null);
      }

      setAnalyzing(localStorage.getItem(ANALYSIS_RUNNING_KEY) === selectedDocHash);
    };
    loadResult();
  }, [selectedDocId, selectedDocHash]);

  useEffect(() => {
    if (!selectedDocHash) return;
    if (localStorage.getItem(ANALYSIS_RUNNING_KEY) !== selectedDocHash) return;

    let cancelled = false;
    let polling = false;
    const poll = async () => {
      if (polling) return false; // Prevent concurrent polls
      polling = true;
      try {
        const serverResult = await getExtractionResult(selectedDocHash);
        if (!cancelled && serverResult?.data) {
          setResult(serverResult.data);
          localStorage.setItem(`analysis_result_${selectedDocId}`, JSON.stringify(serverResult.data));
          localStorage.setItem(`analysis_result_${selectedDocHash}`, JSON.stringify(serverResult.data));
          localStorage.removeItem(ANALYSIS_RUNNING_KEY);
          setAnalyzing(false);
          return true;
        }
      } catch {
        // Server may be busy with extraction, continue polling silently
      } finally {
        polling = false;
      }
      return false;
    };

    setAnalyzing(true);
    const intervalId = setInterval(async () => {
      const done = await poll();
      if (done) clearInterval(intervalId);
    }, 3000);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [selectedDocHash, selectedDocId]);

  const handleAnalyze = async () => {
    if (!selectedDocHash) return;

    setAnalyzing(true);
    setError(null);
    localStorage.setItem(ANALYSIS_RUNNING_KEY, selectedDocHash);
    // 기존 결과를 유지하면서 로딩 표시 (null로 초기화하지 않음)

    try {
      const res = await runExtraction(selectedDocHash);
      if (!isMountedRef.current) return; // Component unmounted during extraction
      if (res && res.data) {
        setResult(res.data);
        localStorage.setItem(`analysis_result_${selectedDocId}`, JSON.stringify(res.data));
        localStorage.setItem(`analysis_result_${selectedDocHash}`, JSON.stringify(res.data));
      } else {
        setError("분석 결과를 가져오지 못했습니다. 서버 로그를 확인해주세요.");
      }
    } catch {
      if (!isMountedRef.current) return;
      setError("분석 중 오류가 발생했습니다.");
    } finally {
      if (isMountedRef.current) {
        if (localStorage.getItem(ANALYSIS_RUNNING_KEY) === selectedDocHash) {
          localStorage.removeItem(ANALYSIS_RUNNING_KEY);
        }
        setAnalyzing(false);
      }
    }
  };

  const renderSlot = (label: string, slotData: SlotValue) => {
    if (!slotData) return null;
    const value = isSlotObject(slotData) ? slotData.value : slotData;
    const evidence = isSlotObject(slotData) && Array.isArray(slotData.evidence) ? slotData.evidence : [];
    const firstEvidence = evidence[0];
    const displayValue =
      value === null || value === undefined
        ? "-"
        : (typeof value === "string" || typeof value === "number" || typeof value === "boolean")
          ? String(value)
          : JSON.stringify(value);

    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-100 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow duration-200 h-full">
        <div className="text-xs font-bold text-indigo-500 dark:text-indigo-400 uppercase tracking-wide mb-2">{label}</div>
        <div className="text-gray-900 dark:text-white text-base font-medium mb-4 leading-relaxed break-words">{displayValue}</div>
        {firstEvidence && (
          <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 text-sm border border-gray-100 dark:border-gray-700">
            <div className="flex items-start gap-2 text-gray-600 dark:text-gray-300">
              <span className="shrink-0 inline-flex items-center justify-center px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300">
                p.{firstEvidence.page_no ?? "-"}
              </span>
              <span className="italic">&quot;{firstEvidence.text_snippet || "-"}&quot;</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCellContent = (content: unknown): ReactNode => {
    if (content === null || content === undefined) return "-";
    
    if (Array.isArray(content)) {
      if (content.length === 0) return <span className="text-gray-400 italic text-xs">Empty</span>;
      return (
        <ul className="list-disc list-inside text-xs space-y-1">
          {content.map((item, i) => (
            <li key={i}>{renderCellContent(item)}</li>
          ))}
        </ul>
      );
    }

    if (isRecord(content)) {
      return (
        <div className="bg-gray-50 dark:bg-gray-800 rounded p-2 text-xs border border-gray-100 dark:border-gray-700 min-w-[150px]">
          {Object.entries(content).map(([key, val]) => (
            <div key={key} className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 mb-1 last:mb-0 border-b border-gray-200/50 dark:border-gray-700/50 last:border-0 pb-1 last:pb-0">
              <span className="font-semibold text-gray-500 dark:text-gray-400">{key}</span>
              <span className="text-gray-700 dark:text-gray-300 break-words">
                {renderCellContent(val)}
              </span>
            </div>
          ))}
        </div>
      );
    }

    return String(content);
  };

  const renderTable = (data: unknown) => {
    const items = Array.isArray(data)
      ? data.filter(isRecord)
      : (isRecord(data) && Array.isArray(data.items) ? data.items.filter(isRecord) : []);
    if (!items || items.length === 0) return <div className="text-gray-500 italic p-4">데이터가 없습니다.</div>;

    const headers = Object.keys(items[0]);

    return (
      <div className="overflow-x-auto border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              {headers.map((header) => (
                <th key={header} className="px-6 py-3 text-left text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wider">
                  {header.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
            {items.map((row, idx) => (
              <tr key={idx} className="even:bg-gray-50/50 dark:even:bg-gray-800/50 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-colors">
                {headers.map((header) => (
                  <td key={`${idx}-${header}`} className="px-6 py-4 text-sm text-gray-700 dark:text-gray-300 align-top">
                    {renderCellContent(row[header])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
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
              <Image src={analysisIcon} alt="Analysis" width={40} height={40} className="object-contain" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">분석 결과 뷰어</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">RFP 문서를 AI로 분석하여 핵심 정보를 추출합니다.</p>
            </div>
          </div>
          <UserHeader />
        </header>

        {/* 문서 선택 및 실행 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <div className="flex flex-col sm:flex-row gap-4 items-end">
            <div className="flex-1 w-full">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">분석할 문서 선택</label>
              <div className="relative">
                <select
                  value={selectedDocId}
                  onChange={(e) => setSelectedDocId(e.target.value)}
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
              onClick={handleAnalyze}
              disabled={analyzing || !selectedDocId}
              className={`w-full sm:w-auto px-8 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white transition-all duration-200
                ${analyzing || !selectedDocId 
                  ? 'bg-indigo-400 cursor-not-allowed opacity-70' 
                  : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-md focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                }`}
            >
              {analyzing ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  분석 중...
                </span>
              ) : '분석 실행'}
            </button>
          </div>
        </div>

        {/* 에러 메시지 표시 */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-center gap-3 text-red-700 dark:text-red-400">
            <svg className="w-6 h-6 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            <span className="font-medium">{error}</span>
          </div>
        )}

        {/* JSON 다운로드 + 결과 뷰어 */}
        {result && (
          <div className="mb-4 flex justify-end">
            <button
              onClick={() => {
                const docHash = selectedDocHash || selectedDocId;
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `extraction_${docHash.slice(0, 12)}.json`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors shadow-sm"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
              JSON 다운로드
            </button>
          </div>
        )}

        {analyzing && (
          <div className="mb-4 p-4 bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-xl flex items-center gap-3 text-indigo-700 dark:text-indigo-300">
            <svg className="animate-spin h-5 w-5 shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            <span className="font-medium">분석 진행 중입니다. 다른 페이지로 이동했다가 돌아와도 결과를 유지합니다.</span>
          </div>
        )}

        {result ? (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
            {/* 탭 헤더 */}
            <div className="flex border-b border-gray-200 dark:border-gray-700">
              {['g1', 'g2', 'g3', 'g4'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 py-4 text-sm font-medium text-center transition-colors relative
                    ${activeTab === tab 
                      ? 'text-indigo-600 dark:text-indigo-400 bg-indigo-50/50 dark:bg-indigo-900/20' 
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                >
                  {tab === 'g1' && '기본 정보 (G1)'}
                  {tab === 'g2' && '일정 (G2)'}
                  {tab === 'g3' && '자격 요건 (G3)'}
                  {tab === 'g4' && '배점표 (G4)'}
                  {activeTab === tab && (
                    <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600 dark:bg-indigo-400"></div>
                  )}
                </button>
              ))}
            </div>

            {/* 탭 내용 */}
            <div className="p-6 bg-gray-50/30 dark:bg-gray-900/30 min-h-[400px]">
              {activeTab === 'g1' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {renderSlot("사업명", result.g1?.project_name)}
                  {renderSlot("발주기관", result.g1?.issuer)}
                  {renderSlot("사업기간", result.g1?.period)}
                  {renderSlot("사업예산", result.g1?.budget)}
                </div>
              )}
              {activeTab === 'g2' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {renderSlot("제출 마감일", result.g2?.submission_deadline)}
                  {renderSlot("설명회 일자", result.g2?.briefing_date)}
                  {renderSlot("질의 응답 기간", result.g2?.qna_period)}
                </div>
              )}
              {activeTab === 'g3' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {renderSlot("필수 면허/자격", result.g3?.required_licenses)}
                  {renderSlot("지역 제한", result.g3?.region_restriction)}
                  {renderSlot("신용평가등급", result.g3?.financial_credit)}
                  {renderSlot("기타 제한 사항", result.g3?.restrictions)}
                </div>
              )}
              {activeTab === 'g4' && (
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-bold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                    <span className="w-1 h-6 bg-indigo-500 rounded-full"></span>
                    배점표 데이터
                  </h3>
                  {renderTable(result.g4)}
                </div>
              )}
            </div>
          </div>
        ) : analyzing ? (
          <div className="flex flex-col items-center justify-center py-20 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-100 border-t-indigo-600 mb-6"></div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">AI 분석 중입니다</h3>
            <p className="text-gray-500 dark:text-gray-400 text-sm">문서의 크기에 따라 시간이 소요될 수 있습니다. 잠시만 기다려주세요.</p>
          </div>
        ) : (
          !error && (
            <div className="flex flex-col items-center justify-center py-24 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-dashed border-gray-300 dark:border-gray-700 text-center">
              <div className="w-20 h-20 bg-gray-50 dark:bg-gray-700 rounded-full flex items-center justify-center mb-6">
                <span className="text-4xl">📊</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">분석 결과가 없습니다</h3>
              <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
                상단의 드롭다운에서 문서를 선택하고 <span className="font-semibold text-indigo-600 dark:text-indigo-400">분석 실행</span> 버튼을 클릭하여 AI 분석 결과를 확인하세요.
              </p>
            </div>
          )
        )}

        {/* 댓글 섹션 */}
        {selectedDocId && (
          <CommentSection docHash={documents.find(d => (d.id || d.doc_hash) === selectedDocId)?.doc_hash || selectedDocId} />
        )}
      </div>
    </div>
  );
}
