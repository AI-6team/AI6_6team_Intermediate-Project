"use client";

import { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useRouter } from "next/navigation";
import dashboardIcon from '../images/dashboard.png';
import UserHeader from '@/components/UserHeader';
import Modal from '@/components/Modal';
import { RFPDocument, DecisionSummary, getDocuments, uploadDocument, getDecisionSummary } from '@/lib/api';

interface BatchResult {
  name: string;
  status: "success" | "error";
  docId?: string;
  error?: string;
}

export default function DashboardPage() {
  const router = useRouter();
  const getErrorMessage = (error: unknown): string =>
    error instanceof Error ? error.message : "알 수 없는 오류";
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState<RFPDocument[]>([]);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);

  // 판정 결과 (doc_hash -> DecisionSummary)
  const [decisions, setDecisions] = useState<Record<string, DecisionSummary>>({});

  // 배치 업로드 상태
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [batchResults, setBatchResults] = useState<BatchResult[] | null>(null);

  const fetchDecisions = async (docs: RFPDocument[]) => {
    const results: Record<string, DecisionSummary> = {};
    await Promise.all(
      docs.map(async (doc) => {
        const hash = doc.doc_hash;
        if (!hash) return;
        try {
          const d = await getDecisionSummary(hash);
          if (d) results[hash] = d;
        } catch {}
      })
    );
    setDecisions(results);
  };

  const fetchDocuments = async () => {
    const token = localStorage.getItem("token");
    if (!token) return;
    try {
      const docs = await getDocuments();
      setDocuments(docs);
      fetchDecisions(docs);
    } catch (error) {
      console.error("Error in fetchDocuments:", error);
      setMessage({ type: 'error', text: '문서 목록을 불러오는데 실패했습니다.' });
    }
  };

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthModal(true);
    }
  }, [router]);

  useEffect(() => {
    fetchDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isBatchMode = files.length > 1;

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setMessage(null);
    setBatchResults(null);

    if (files.length === 1) {
      // 단일 업로드
      try {
        const result = await uploadDocument(files[0]);
        setMessage({ type: 'success', text: `업로드 성공! (ID: ${result.doc_id})` });
        fetchDocuments();
        setFiles([]);
      } catch (e: unknown) {
        setMessage({ type: 'error', text: `업로드 실패: ${getErrorMessage(e)}` });
      } finally {
        setUploading(false);
      }
    } else {
      // 배치 업로드
      const results: BatchResult[] = [];
      setBatchProgress({ current: 0, total: files.length });

      for (let i = 0; i < files.length; i++) {
        setBatchProgress({ current: i + 1, total: files.length });
        try {
          const result = await uploadDocument(files[i]);
          results.push({ name: files[i].name, status: "success", docId: result.doc_id });
        } catch (e: unknown) {
          results.push({ name: files[i].name, status: "error", error: getErrorMessage(e) });
        }
      }

      setBatchResults(results);
      setBatchProgress(null);
      setUploading(false);
      setFiles([]);
      fetchDocuments();
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files).filter(
        (f) => f.name.toLowerCase().endsWith('.pdf') || f.name.toLowerCase().endsWith('.hwp')
      );
      if (droppedFiles.length === 0) {
        setMessage({ type: 'error', text: 'PDF 또는 HWP 파일만 업로드 가능합니다.' });
      } else {
        setFiles(droppedFiles);
        setMessage(null);
        setBatchResults(null);
      }
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const successCount = batchResults?.filter((r) => r.status === "success").length || 0;
  const failCount = batchResults?.filter((r) => r.status === "error").length || 0;

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
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3 mb-2">
            <Image src={dashboardIcon} alt="Dashboard" width={36} height={36} className="object-contain" />
            <span>문서 대시보드</span>
          </h1>
          <p className="text-gray-600 dark:text-gray-400 ml-1">
            RFP 문서를 업로드하고 분석 이력을 관리하세요. 2개 이상 업로드 시 일괄 분석으로 전환됩니다.
          </p>
        </div>
          <UserHeader />
        </header>

        {/* 1. 파일 업로드 섹션 */}
      <div className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 mb-10">
        <h2 className="text-xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <span className="text-blue-600">📤</span> 새로운 RFP 업로드
          {isBatchMode && (
            <span className="ml-2 px-2 py-0.5 bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 text-xs rounded-full font-semibold">
              다문서 일괄 모드
            </span>
          )}
        </h2>

        <div
          className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors group ${
            isDragging
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-300 dark:border-gray-600 hover:border-blue-500 dark:hover:border-blue-400 bg-gray-50 dark:bg-gray-800/50'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-upload"
            accept=".pdf,.hwp"
            multiple
            className="hidden"
            onChange={(e) => {
              const selected = Array.from(e.target.files || []);
              setFiles(selected);
              setMessage(null);
              setBatchResults(null);
            }}
          />
          <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center justify-center w-full h-full">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition-transform">
              📄
            </div>
            <span className="text-lg font-medium text-gray-700 dark:text-gray-200">
              {files.length === 0
                ? "PDF 파일을 선택하거나 이곳에 끌어다 놓으세요"
                : `${files.length}개 파일 선택됨`}
            </span>
            <span className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              최대 50MB, PDF/HWP 형식 지원 (여러 파일 선택 가능)
            </span>

            {files.length === 0 && (
              <div className="mt-6 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 px-6 py-2.5 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors font-medium shadow-sm">
                파일 선택하기
              </div>
            )}
          </label>
        </div>

        {/* 선택된 파일 목록 */}
        {files.length > 0 && (
          <div className="mt-4 space-y-2">
            {files.map((f, i) => (
              <div key={i} className="flex items-center justify-between bg-gray-50 dark:bg-gray-700/50 px-4 py-2 rounded-lg">
                <div className="flex items-center gap-2 text-sm">
                  <span>📄</span>
                  <span className="text-gray-800 dark:text-gray-200">{f.name}</span>
                  <span className="text-gray-400 text-xs">({(f.size / 1024 / 1024).toFixed(2)} MB)</span>
                </div>
                <button
                  onClick={() => removeFile(i)}
                  className="text-gray-400 hover:text-red-500 transition-colors text-sm"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}

        {/* 배치 진행 상태 */}
        {batchProgress && (
          <div className="mt-6">
            <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
              <span>처리 중 {batchProgress.current}/{batchProgress.total}</span>
              <span>{Math.round((batchProgress.current / batchProgress.total) * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
              <div
                className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }}
              />
            </div>
          </div>
        )}

        {files.length > 0 && !uploading && (
          <div className="mt-6 flex justify-end">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="bg-blue-600 text-white px-8 py-3 rounded-xl hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all shadow-md font-bold flex items-center gap-2"
            >
              {uploading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  처리 중...
                </>
              ) : isBatchMode ? (
                <>🚀 다문서 일괄 업로드 시작 ({files.length}개)</>
              ) : (
                <>🚀 업로드 및 분석 시작</>
              )}
            </button>
          </div>
        )}

        {message && (
          <div className={`mt-6 p-4 rounded-xl flex items-center gap-3 ${message.type === 'success' ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>
            <span className="text-xl">{message.type === 'success' ? '✅' : '⚠️'}</span>
            {message.text}
          </div>
        )}

        {/* 배치 결과 요약 */}
        {batchResults && (
          <div className="mt-6">
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">{batchResults.length}</div>
                <div className="text-xs text-blue-600 dark:text-blue-400">전체</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-green-700 dark:text-green-300">{successCount}</div>
                <div className="text-xs text-green-600 dark:text-green-400">성공</div>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-red-700 dark:text-red-300">{failCount}</div>
                <div className="text-xs text-red-600 dark:text-red-400">실패</div>
              </div>
            </div>

            {failCount > 0 && (
              <div className="space-y-2">
                {batchResults
                  .filter((r) => r.status === "error")
                  .map((r, i) => (
                    <div key={i} className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3 text-sm text-red-700 dark:text-red-300">
                      <strong>{r.name}</strong>: {r.error}
                    </div>
                  ))}
              </div>
            )}

            <button
              onClick={() => setBatchResults(null)}
              className="mt-4 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
            >
              결과 닫기
            </button>
          </div>
        )}
      </div>

      {/* 2. 문서 목록 섹션 */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50/50 dark:bg-gray-800">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <span className="text-indigo-600">📋</span> 처리된 문서 목록
          </h2>
          <button
            onClick={fetchDocuments}
            className="text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 flex items-center gap-1 transition-colors bg-white dark:bg-gray-700 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-600 shadow-sm hover:shadow"
          >
            🔄 새로고침
          </button>
        </div>

        {documents.length === 0 ? (
          <div className="text-center py-16 px-6">
            <div className="text-5xl mb-4 opacity-20">📭</div>
            <p className="text-gray-500 dark:text-gray-400 text-lg">아직 처리된 문서가 없습니다.</p>
            <p className="text-gray-400 dark:text-gray-500 text-sm mt-1">위에서 새로운 RFP 파일을 업로드해보세요.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700/50">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap">파일명</th>
                  <th className="px-4 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap min-w-[80px]">작성자</th>
                  <th className="px-4 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap">업로드 일시</th>
                  <th className="px-4 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap">상태</th>
                  <th className="px-4 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap">판정</th>
                  <th className="px-4 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap">ID</th>
                  <th className="px-4 py-4 text-right text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap">작업</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {documents.map((doc) => (
                  <tr key={doc.id || doc.doc_hash} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <span className="text-2xl mr-3">📄</span>
                        <span className="text-sm font-medium text-gray-900 dark:text-white">{doc.filename}</span>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {doc.owner_name || '-'}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {new Date(doc.upload_date).toLocaleString()}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <span className="px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
                        {doc.status || 'Completed'}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      {doc.doc_hash && decisions[doc.doc_hash] ? (
                        <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          decisions[doc.doc_hash].signal === 'green'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                            : decisions[doc.doc_hash].signal === 'red'
                              ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                        }`}>
                          {decisions[doc.doc_hash].signal === 'green' ? 'GO' : decisions[doc.doc_hash].signal === 'red' ? 'NO-GO' : 'REVIEW'}
                        </span>
                      ) : (
                        <span className="text-xs text-gray-400 dark:text-gray-500">-</span>
                      )}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-xs text-gray-400 dark:text-gray-500 font-mono">
                      {(doc.id || doc.doc_hash).substring(0, 8)}...
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-right">
                      <Link
                        href={`/analysis?docId=${doc.id || doc.doc_hash}`}
                        className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 text-sm font-medium transition-colors"
                      >
                        분석 결과 보기 →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      </div>
    </div>
  );
}
