"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function UserHeader() {
  const router = useRouter();
  const [user, setUser] = useState<{ name: string; team: string; role: string } | null>(null);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      // 토큰이 없으면 아무것도 하지 않음. (인증이 필요한 페이지는 각 페이지에서 리디렉션)
      return;
    }

    fetch("http://localhost:8000/auth/me", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (res.ok) return res.json();
        if (res.status === 401) {
          // 토큰이 유효하지 않으면 삭제하고 페이지를 새로고침하여 각 페이지의 인증 로직을 다시 트리거
          localStorage.removeItem("token");
          router.refresh();
        }
        return null;
      })
      .then((data) => {
        if (data) setUser(data);
      })
      .catch((err) => {
        console.error("UserHeader: Failed to fetch user info.", err);
      });
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    router.push("/");
  };

  if (!user) {
    // 로딩 중이거나 로그인하지 않았을 때 레이아웃이 깨지지 않도록 placeholder를 렌더링
    return <div className="w-48 h-12" />;
  }

  return (
    <div className="text-right">
      <div className="text-sm font-bold text-gray-900 dark:text-white">
        {user.name} <span className="font-normal text-gray-500 dark:text-gray-400">({user.role === 'leader' ? '팀장' : '팀원'})</span>
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
        {user.team || "소속 팀 없음"}
      </div>
      <div className="flex items-center justify-end gap-4">
        {user.role === 'leader' && (
          <Link href="/profile" className="text-xs text-indigo-500 hover:text-indigo-700 font-medium transition-colors">
            회사 정보 수정
          </Link>
        )}
        <button onClick={handleLogout} className="text-xs text-red-500 hover:text-red-700 font-medium transition-colors">
          로그아웃
        </button>
      </div>
    </div>
  );
}
