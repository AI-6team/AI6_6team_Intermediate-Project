"use client";

import { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useRouter } from "next/navigation";
import dashboardIcon from '../images/dashboard.png';
import UserHeader from '@/components/UserHeader';
import Modal from '@/components/Modal';
import { RFPDocument } from '@/lib/api';

export default function DashboardPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState<RFPDocument[]>([]);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);

  // 문서 목록 불러오기
  const fetchDocuments = async () => {
    const token = localStorage.getItem("token");
    if (!token) {
      // This should be caught by the other useEffect, but as a safeguard.
      return;
    }
    try {
      const response = await fetch("http://localhost:8000/api/v1/ingest/documents", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!response.ok) {
        if (response.status === 401) {
          // Token is invalid or expired
          localStorage.removeItem("token");
          router.push("/");
        }
        throw new Error(`Failed to fetch documents with status: ${response.status}`);
      }
      const docs = await response.json();
      setDocuments(docs);
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
  }, []); // This effect should run only once on mount

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setMessage(null);
    
    const token = localStorage.getItem("token");
    if (!token) {
        router.push("/");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/v1/ingest/upload", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");
      
      const result = await response.json();
      if (result) {
        setMessage({ type: 'success', text: `업로드 성공! (ID: ${result.doc_id})` });
        fetchDocuments(); // 목록 갱신
        setFile(null);
      }
    } catch (e) {
      setMessage({ type: 'error', text: '업로드 실패: 서버 오류가 발생했습니다.' });
    } finally {
      setUploading(false);
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
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.toLowerCase().endsWith('.pdf') || droppedFile.name.toLowerCase().endsWith('.hwp')) {
        setFile(droppedFile);
        setMessage(null);
      } else {
        setMessage({ type: 'error', text: 'PDF 또는 HWP 파일만 업로드 가능합니다.' });
      }
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
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3 mb-2">
            <Image src={dashboardIcon} alt="Dashboard" width={36} height={36} className="object-contain" />
            <span>문서 대시보드</span>
          </h1>
          <p className="text-gray-600 dark:text-gray-400 ml-1">
            RFP 문서를 업로드하고 분석 이력을 관리하세요.
          </p>
        </div>
          <UserHeader />
        </header>

        {/* 1. 파일 업로드 섹션 */}
      <div className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 mb-10">
        <h2 className="text-xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <span className="text-blue-600">📤</span> 새로운 RFP 업로드
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
            className="hidden"
            onChange={(e) => {
              setFile(e.target.files?.[0] || null);
              setMessage(null);
            }}
          />
          <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center justify-center w-full h-full">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition-transform">
              📄
            </div>
            <span className="text-lg font-medium text-gray-700 dark:text-gray-200">
              {file ? file.name : "PDF 파일을 선택하거나 이곳에 끌어다 놓으세요"}
            </span>
            <span className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "최대 50MB, PDF/HWP 형식 지원"}
            </span>
            
            {!file && (
              <div className="mt-6 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 px-6 py-2.5 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors font-medium shadow-sm">
                파일 선택하기
              </div>
            )}
          </label>
        </div>

        {file && (
          <div className="mt-6 flex justify-end animate-fade-in-up">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="bg-blue-600 text-white px-8 py-3 rounded-xl hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all shadow-md font-bold flex items-center gap-2"
            >
              {uploading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  업로드 및 분석 중...
                </>
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
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">파일명</th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">작성자</th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">업로드 일시</th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">상태</th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">ID</th>
                  <th className="px-6 py-4 text-right text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">작업</th>
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
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {(doc as any).owner_name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {new Date(doc.upload_date).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
                        {doc.status || 'Completed'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-xs text-gray-400 dark:text-gray-500 font-mono">
                      {(doc.id || doc.doc_hash).substring(0, 8)}...
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right">
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
