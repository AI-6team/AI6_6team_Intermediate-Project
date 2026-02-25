"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import profileIcon from '../images/validation.png'; // 아이콘은 기존 것을 활용합니다.
import Modal from '@/components/Modal';

export default function ProfilePage() {
  const router = useRouter();
  const [companyName, setCompanyName] = useState("");
  const [licenses, setLicenses] = useState("");
  const [initialCompanyName, setInitialCompanyName] = useState("");
  const [initialLicenses, setInitialLicenses] = useState("");
  
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [modalConfig, setModalConfig] = useState<{isOpen: boolean, title: string, message: string, redirect?: string}>({
    isOpen: false, title: "", message: "", redirect: undefined
  });

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setModalConfig({
        isOpen: true, title: "로그인 필요", message: "로그인 후 이용해 주세요.", redirect: "/"
      });
      return;
    }

    setLoading(true);
    fetch("http://localhost:8000/auth/me", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (res.status === 401) throw new Error("Unauthorized");
        if (!res.ok) throw new Error("Failed to fetch user data");
        return res.json();
      })
      .then((data) => {
        if (data.role !== 'leader') {
          setModalConfig({
            isOpen: true, title: "권한 없음", message: "팀장만 접근할 수 있는 페이지입니다.", redirect: "/dashboard"
          });
          return;
        }
        setCompanyName(data.team || "");
        setLicenses(data.licenses || "");
        setInitialCompanyName(data.team || "");
        setInitialLicenses(data.licenses || "");
      })
      .catch((err) => {
        if (err.message === "Unauthorized") {
          localStorage.removeItem("token");
          router.push("/");
        } else {
          setError("사용자 정보를 불러오는 데 실패했습니다.");
        }
      })
      .finally(() => {
        setLoading(false);
      });
  }, [router]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/");
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/auth/me", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ team: companyName, licenses: licenses }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "저장에 실패했습니다.");
      }

      const updatedUser = await response.json();
      setCompanyName(updatedUser.team || "");
      setLicenses(updatedUser.licenses || "");
      setInitialCompanyName(updatedUser.team || "");
      setInitialLicenses(updatedUser.licenses || "");
      setSuccess("프로필 정보가 성공적으로 업데이트되었습니다.");

    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };
  
  const isChanged = companyName !== initialCompanyName || licenses !== initialLicenses;

  const handleModalClose = () => {
    setModalConfig(prev => ({ ...prev, isOpen: false }));
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
        <div className="flex items-center gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 p-3 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
            <Image src={profileIcon} alt="Profile" width={40} height={40} className="object-contain" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">회사 프로필 관리</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">팀장만 회사 정보(팀 이름, 보유 면허)를 수정할 수 있습니다.</p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-8 space-y-6">
          <div>
            <label htmlFor="companyName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">회사명 (팀 이름)</label>
            <input type="text" id="companyName" value={companyName} onChange={(e) => setCompanyName(e.target.value)} className="block w-full rounded-lg border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm sm:text-sm p-3 border focus:ring-indigo-500 focus:border-indigo-500" />
          </div>
          <div>
            <label htmlFor="licenses" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">보유 면허 (쉼표로 구분)</label>
            <input type="text" id="licenses" value={licenses} onChange={(e) => setLicenses(e.target.value)} placeholder="소프트웨어사업자, 정보통신공사업" className="block w-full rounded-lg border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm sm:text-sm p-3 border focus:ring-indigo-500 focus:border-indigo-500" />
          </div>
          <div className="pt-4 flex justify-end items-center gap-4 border-t border-gray-200 dark:border-gray-700">
            {error && <p className="text-sm text-red-600 mr-auto">{error}</p>}
            {success && <p className="text-sm text-green-600 mr-auto">{success}</p>}
            <button onClick={handleSave} disabled={saving || !isChanged} className={`px-6 py-2.5 border border-transparent text-base font-medium rounded-lg shadow-sm text-white transition-all duration-200 ${saving || !isChanged ? 'bg-indigo-400 cursor-not-allowed opacity-70' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-md focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'}`}>
              {saving ? '저장 중...' : '변경사항 저장'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
