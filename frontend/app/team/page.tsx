"use client";

import { useState, useEffect } from "react";
import UserHeader from "@/components/UserHeader";
import Modal from "@/components/Modal";
import CommentSection from "@/components/CommentSection";
import {
  RFPDocument,
  TeamMember,
  DecisionSummary,
  getTeamMembers,
  getTeamDocuments,
  getDecisionSummary,
  getCurrentUser,
} from "@/lib/api";

export default function TeamWorkspacePage() {
  const [teamName, setTeamName] = useState<string>("");
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [documents, setDocuments] = useState<RFPDocument[]>([]);
  const [selectedDocHash, setSelectedDocHash] = useState<string>("");
  const [selectedDoc, setSelectedDoc] = useState<RFPDocument | null>(null);
  const [decision, setDecision] = useState<DecisionSummary | null>(null);
  const [loadingDecision, setLoadingDecision] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [noTeam, setNoTeam] = useState(false);

  // 인증 확인 + 팀 정보 로드
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthModal(true);
      return;
    }

    const loadTeamData = async () => {
      try {
        const user = await getCurrentUser();
        if (!user) {
          setLoadError("팀 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.");
          return;
        }
        if (!user.team) {
          setNoTeam(true);
          return;
        }
        setTeamName(user.team);

        const [memberList, docList] = await Promise.all([
          getTeamMembers(),
          getTeamDocuments(),
        ]);
        setMembers(memberList);
        setDocuments(docList);

        if (docList.length > 0) {
          setSelectedDocHash(docList[0].doc_hash);
          setSelectedDoc(docList[0]);
        }
      } catch {
        setLoadError("팀 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.");
      } finally {
        setLoading(false);
      }
    };

    loadTeamData();
  }, []);

  // 선택 문서 변경 시 판정 결과 로드
  useEffect(() => {
    if (!selectedDocHash) return;

    const doc = documents.find((d) => d.doc_hash === selectedDocHash);
    setSelectedDoc(doc || null);

    const loadDecision = async () => {
      setLoadingDecision(true);
      setDecision(null);
      const result = await getDecisionSummary(selectedDocHash);
      setDecision(result);
      setLoadingDecision(false);
    };
    loadDecision();
  }, [selectedDocHash, documents]);

  if (showAuthModal) {
    return (
      <Modal isOpen={true} onClose={() => (window.location.href = "/")}>
        로그인이 필요합니다.
      </Modal>
    );
  }

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-center py-24">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600"></div>
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Team Workspace</h1>
          <UserHeader />
        </div>
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6 text-red-700 dark:text-red-300">
          {loadError}
        </div>
      </div>
    );
  }

  if (noTeam) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-100 dark:bg-indigo-900 rounded-xl flex items-center justify-center text-xl">
              👥
            </div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Team Workspace
            </h1>
          </div>
          <UserHeader />
        </div>
        <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-xl p-6 text-center">
          <p className="text-blue-700 dark:text-blue-300 text-lg">
            현재 소속된 팀이 없습니다.
          </p>
          <p className="text-blue-600 dark:text-blue-400 mt-2">
            프로필에서 팀을 설정하거나 팀이 있는 계정으로 가입하세요.
          </p>
        </div>
      </div>
    );
  }

  const signalBadge = (signal: string) => {
    const styles: Record<string, string> = {
      red: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
      yellow:
        "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-300",
      green:
        "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
    };
    return (
      <span
        className={`px-3 py-1 rounded-full text-sm font-semibold ${styles[signal] || styles.yellow}`}
      >
        {signal === "red"
          ? "NO-GO"
          : signal === "green"
            ? "GO"
            : "REVIEW"}
      </span>
    );
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-100 dark:bg-indigo-900 rounded-xl flex items-center justify-center text-xl">
            👥
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Team Workspace
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              팀: <strong>{teamName}</strong> | 팀원:{" "}
              {members.map((m) => m.name).join(", ")}
            </p>
          </div>
        </div>
        <UserHeader />
      </div>

      {/* 문서 목록이 없을 때 */}
      {documents.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-12 text-center">
          <p className="text-gray-500 dark:text-gray-400 text-lg">
            팀원이 업로드한 RFP 문서가 아직 없습니다.
          </p>
          <a
            href="/dashboard"
            className="inline-block mt-4 text-green-600 dark:text-green-400 hover:underline"
          >
            문서 업로드하러 가기 →
          </a>
        </div>
      ) : (
        <>
          {/* 안건 선택 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              안건 선택
            </label>
            <select
              value={selectedDocHash}
              onChange={(e) => setSelectedDocHash(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-green-500 focus:border-green-500"
            >
              {documents.map((doc) => (
                <option key={doc.doc_hash} value={doc.doc_hash}>
                  {doc.filename}
                  {doc.uploaded_by_name
                    ? ` (by ${doc.uploaded_by_name})`
                    : doc.owner_name
                      ? ` (by ${doc.owner_name})`
                      : ""}{" "}
                  - {(doc.upload_date || "").slice(0, 10)}
                </option>
              ))}
            </select>
          </div>

          {/* 안건 정보 + 판정 결과 */}
          {selectedDoc && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* 안건 정보 */}
              <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  안건 정보
                </h2>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">
                      파일명
                    </span>
                    <span className="text-gray-900 dark:text-white font-medium">
                      {selectedDoc.filename}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">
                      업로더
                    </span>
                    <span className="text-gray-900 dark:text-white">
                      {selectedDoc.uploaded_by_name ||
                        selectedDoc.owner_name ||
                        "-"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">
                      업로드 날짜
                    </span>
                    <span className="text-gray-900 dark:text-white">
                      {(selectedDoc.upload_date || "").slice(0, 10)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">
                      Doc Hash
                    </span>
                    <span className="text-gray-600 dark:text-gray-400 font-mono text-xs">
                      {selectedDoc.doc_hash?.slice(0, 16)}...
                    </span>
                  </div>
                </div>

                {/* 분석 결과 보기 링크 */}
                <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
                  <a
                    href={`/analysis?docId=${selectedDoc.id || selectedDoc.doc_hash}`}
                    className="text-green-600 dark:text-green-400 text-sm hover:underline"
                  >
                    분석 결과 상세 보기 →
                  </a>
                </div>
              </div>

              {/* 판정 결과 */}
              <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  판정 결과
                </h2>
                {loadingDecision ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
                  </div>
                ) : decision ? (
                  <div>
                    {/* 신호 배지 + 추천 */}
                    <div className="flex items-center gap-3 mb-4">
                      {signalBadge(decision.signal)}
                      <span className="text-gray-700 dark:text-gray-300 text-sm font-medium">
                        {decision.recommendation}
                      </span>
                    </div>

                    {/* 카운트 */}
                    <div className="grid grid-cols-3 gap-3">
                      <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                          {decision.n_red}
                        </div>
                        <div className="text-xs text-red-500 dark:text-red-400 mt-1">
                          RED
                        </div>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-gray-600 dark:text-gray-300">
                          {decision.n_gray}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          GRAY
                        </div>
                      </div>
                      <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {decision.n_green}
                        </div>
                        <div className="text-xs text-green-500 dark:text-green-400 mt-1">
                          GREEN
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-6 text-gray-500 dark:text-gray-400">
                    <p>판정 결과가 없습니다.</p>
                    <p className="text-sm mt-1">
                      추출 미완료 또는 프로필 미설정
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 팀 코멘트 */}
          {selectedDocHash && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                팀 코멘트
              </h2>
              <CommentSection docHash={selectedDocHash} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
