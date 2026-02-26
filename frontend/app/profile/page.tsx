"use client";

import { useState, useEffect, type ChangeEvent } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import profileIcon from "../images/validation.png";
import Modal from "@/components/Modal";
import UserHeader from "@/components/UserHeader";
import { getCurrentUser, updateUserProfile } from "@/lib/api";

export default function ProfilePage() {
  const router = useRouter();
  const getErrorMessage = (error: unknown): string =>
    error instanceof Error ? error.message : "프로필 업데이트 중 오류가 발생했습니다.";
  const [companyName, setCompanyName] = useState("");
  const [region, setRegion] = useState("");
  const [companyLogo, setCompanyLogo] = useState<string | null>(null);
  const [licenseList, setLicenseList] = useState<string[]>([""]);
  const [initialCompanyName, setInitialCompanyName] = useState("");
  const [initialRegion, setInitialRegion] = useState("");
  const [initialLicenses, setInitialLicenses] = useState("");
  const [initialCompanyLogo, setInitialCompanyLogo] = useState<string | null>(null);
  const [role, setRole] = useState("");
  const [userName, setUserName] = useState("");

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [modalConfig, setModalConfig] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    redirect?: string;
  }>({ isOpen: false, title: "", message: "", redirect: undefined });

  // 읽기 전용 여부
  const isReadOnly = role === "member";

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setModalConfig({
        isOpen: true,
        title: "로그인 필요",
        message: "로그인 후 이용해 주세요.",
        redirect: "/",
      });
      return;
    }

    setLoading(true);
    getCurrentUser()
      .then((data) => {
        if (!data) throw new Error("Failed to fetch user data");
        setRole(data.role);
        setUserName(data.name);
        setCompanyName(data.team || "");
        setInitialCompanyName(data.team || "");
        setRegion(data.region || "");
        setInitialRegion(data.region || "");
        setInitialLicenses(data.licenses || "");
        setCompanyLogo(data.company_logo || null);
        setInitialCompanyLogo(data.company_logo || null);

        // 면허 목록 파싱
        const licenses = data.licenses
          ? data.licenses
              .split(",")
              .map((l: string) => l.trim())
              .filter(Boolean)
          : [];
        setLicenseList(licenses.length > 0 ? licenses : [""]);
      })
      .catch(() => {
        setError("프로필 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.");
      })
      .finally(() => setLoading(false));
  }, [router]);

  const currentLicensesStr = licenseList.filter(Boolean).join(", ");
  const isChanged =
    companyName !== initialCompanyName ||
    region !== initialRegion ||
    currentLicensesStr !== initialLicenses ||
    (companyLogo || "") !== (initialCompanyLogo || "");

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const updatedUser = await updateUserProfile({
        team: companyName,
        licenses: currentLicensesStr,
        region,
        company_logo: companyLogo || "",
      });
      setCompanyName(updatedUser.team || "");
      setInitialCompanyName(updatedUser.team || "");
      setRegion(updatedUser.region || "");
      setInitialRegion(updatedUser.region || "");
      setInitialLicenses(updatedUser.licenses || "");
      setCompanyLogo(updatedUser.company_logo || null);
      setInitialCompanyLogo(updatedUser.company_logo || null);

      const licenses = updatedUser.licenses
        ? updatedUser.licenses
            .split(",")
            .map((l) => l.trim())
            .filter(Boolean)
        : [];
      setLicenseList(licenses.length > 0 ? licenses : [""]);
      setSuccess("프로필 정보가 성공적으로 업데이트되었습니다.");
    } catch (err: unknown) {
      setError(getErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const addLicense = () => setLicenseList((prev) => [...prev, ""]);
  const removeLicense = (index: number) =>
    setLicenseList((prev) => prev.filter((_, i) => i !== index));
  const updateLicense = (index: number, value: string) =>
    setLicenseList((prev) => prev.map((l, i) => (i === index ? value : l)));

  const handleLogoFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("이미지 파일만 업로드할 수 있습니다.");
      return;
    }
    if (file.size > 2 * 1024 * 1024) {
      setError("이미지는 2MB 이하만 업로드할 수 있습니다.");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setCompanyLogo(reader.result);
        setError(null);
      }
    };
    reader.onerror = () => setError("이미지 읽기에 실패했습니다.");
    reader.readAsDataURL(file);
  };

  const handleModalClose = () => {
    setModalConfig((prev) => ({ ...prev, isOpen: false }));
    if (modalConfig.redirect) {
      router.push(modalConfig.redirect);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-100 border-t-indigo-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-10">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <Modal
          isOpen={modalConfig.isOpen}
          onClose={handleModalClose}
          title={modalConfig.title}
        >
          {modalConfig.message}
        </Modal>

        <header className="flex justify-between items-end mb-8">
          <div className="flex items-center gap-4">
            <div className="bg-white dark:bg-gray-800 p-3 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
              <Image
                src={profileIcon}
                alt="Profile"
                width={40}
                height={40}
                className="object-contain"
              />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                회사 프로필 관리
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {isReadOnly
                  ? "팀장만 회사 프로필을 수정할 수 있습니다. 아래는 현재 팀 프로필입니다."
                  : "팀 이름과 보유 면허를 관리합니다. 이 정보는 자격 검증에 사용됩니다."}
              </p>
            </div>
          </div>
          <UserHeader />
        </header>

        {/* 역할 표시 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-100 dark:bg-indigo-900 rounded-full flex items-center justify-center text-sm font-bold text-indigo-600 dark:text-indigo-300">
              {userName?.charAt(0)?.toUpperCase() || "?"}
            </div>
            <div>
              <div className="font-medium text-gray-900 dark:text-white">
                {userName}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {companyName || "팀 미소속"}
              </div>
            </div>
          </div>
          <span
            className={`px-3 py-1 rounded-full text-xs font-semibold ${
              role === "leader"
                ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300"
                : "bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400"
            }`}
          >
            {role === "leader" ? "팀장 (Leader)" : "팀원 (Member)"}
          </span>
        </div>

        {/* 프로필 폼 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-8 space-y-6">
          {/* 기본 정보 */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <span className="w-1 h-5 bg-indigo-500 rounded-full"></span>
              기본 정보
            </h2>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                회사명 (팀 이름)
              </label>
              <input
                type="text"
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                disabled={isReadOnly}
                className="block w-full rounded-lg border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm sm:text-sm p-3 border focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
              />
            </div>
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                지역
              </label>
              <input
                type="text"
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                disabled={isReadOnly}
                placeholder="예: 서울특별시"
                className="block w-full rounded-lg border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm sm:text-sm p-3 border focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
              />
            </div>
          </div>

          {/* 회사 로고 */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <span className="w-1 h-5 bg-blue-500 rounded-full"></span>
              회사 로고
            </h2>
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
              <div className="w-24 h-24 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 overflow-hidden flex items-center justify-center">
                {companyLogo ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={companyLogo} alt="company logo" className="w-full h-full object-cover" />
                ) : (
                  <span className="text-xs text-gray-400">No Logo</span>
                )}
              </div>

              {!isReadOnly && (
                <div className="flex gap-2">
                  <label className="px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-sm cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700">
                    이미지 업로드
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleLogoFileChange}
                    />
                  </label>
                  <button
                    onClick={() => setCompanyLogo(null)}
                    className="px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-sm hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    제거
                  </button>
                </div>
              )}
            </div>
            <p className="mt-2 text-xs text-gray-500">JPG/PNG 등 이미지, 최대 2MB</p>
          </div>

          {/* 보유 면허/자격 - 동적 리스트 에디터 */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <span className="w-1 h-5 bg-green-500 rounded-full"></span>
              보유 면허 및 자격
            </h2>

            <div className="space-y-3">
              {licenseList.map((license, index) => (
                <div key={index} className="flex items-center gap-2">
                  <input
                    type="text"
                    value={license}
                    onChange={(e) => updateLicense(index, e.target.value)}
                    disabled={isReadOnly}
                    placeholder="면허 명칭 (예: 소프트웨어사업자)"
                    className="flex-1 rounded-lg border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm sm:text-sm p-3 border focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                  />
                  {!isReadOnly && licenseList.length > 1 && (
                    <button
                      onClick={() => removeLicense(index)}
                      className="p-2 text-red-400 hover:text-red-600 dark:text-red-500 dark:hover:text-red-400 transition-colors"
                      title="삭제"
                    >
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                      </svg>
                    </button>
                  )}
                </div>
              ))}
            </div>

            {!isReadOnly && (
              <button
                onClick={addLicense}
                className="mt-3 flex items-center gap-1 text-sm text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300 transition-colors"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                  />
                </svg>
                면허 추가
              </button>
            )}
          </div>

          {/* 저장 */}
          {!isReadOnly && (
            <div className="pt-4 flex justify-end items-center gap-4 border-t border-gray-200 dark:border-gray-700">
              {error && (
                <p className="text-sm text-red-600 mr-auto">{error}</p>
              )}
              {success && (
                <p className="text-sm text-green-600 mr-auto">{success}</p>
              )}
              <button
                onClick={handleSave}
                disabled={saving || !isChanged}
                className={`px-6 py-2.5 border border-transparent text-base font-medium rounded-lg shadow-sm text-white transition-all duration-200 ${
                  saving || !isChanged
                    ? "bg-indigo-400 cursor-not-allowed opacity-70"
                    : "bg-indigo-600 hover:bg-indigo-700 hover:shadow-md focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                }`}
              >
                {saving ? "저장 중..." : "변경사항 저장"}
              </button>
            </div>
          )}

          {/* 읽기 전용 안내 */}
          {isReadOnly && (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 text-sm text-blue-700 dark:text-blue-300">
              이 프로필은 팀장에 의해 관리됩니다. 수정이 필요하면
              팀장에게 요청하세요. 이 정보는 &quot;자격 검증&quot;
              탭에서 입찰 적격성 판정에 사용됩니다.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
