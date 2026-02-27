"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  apiUrl,
  clearAuthToken,
  getAuthTokenServerSnapshot,
  getAuthTokenSnapshot,
  subscribeAuthToken,
} from "@/lib/api";

export default function UserHeader() {
  const router = useRouter();
  const [user, setUser] = useState<{ name: string; team: string; role: string } | null>(null);
  const token = useSyncExternalStore(
    subscribeAuthToken,
    getAuthTokenSnapshot,
    getAuthTokenServerSnapshot
  );

  useEffect(() => {
    if (!token) {
      return;
    }

    const controller = new AbortController();
    fetch(apiUrl("/auth/me"), {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    })
      .then((res) => {
        if (res.ok) return res.json();
        if (res.status === 401) {
          clearAuthToken();
          router.refresh();
        }
        return null;
      })
      .then((data) => {
        setUser(data);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setUser(null);
        console.warn("UserHeader: 사용자 정보를 불러오지 못했습니다.");
      });

    return () => controller.abort();
  }, [router, token]);

  const handleLogout = () => {
    clearAuthToken();
    router.push("/");
  };

  if (!token || !user) {
    // 로딩 중이거나 로그인하지 않았을 때 레이아웃이 깨지지 않도록 placeholder를 렌더링
    return <div className="w-48 h-12" />;
  }

  return (
    <div className="text-right">
      <div className="text-sm font-bold text-gray-900 dark:text-white">
        {user.name} <span className="font-normal text-gray-500 dark:text-gray-400">({user.role === 'leader' ? '팀장' : '팀원'})</span>
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
        {user.team || "회사 미설정"}
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
