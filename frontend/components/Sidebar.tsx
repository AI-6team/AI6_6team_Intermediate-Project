"use client";

import Image, { type StaticImageData } from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import logo from "../app/images/logo_wh.png";
import logoWh from "../app/images/logo_bk.png";
import homeIcon from "../app/images/home.png";
import dashboardIcon from "../app/images/dashboard.png";
import analysisIcon from "../app/images/analysis.png";
import validationIcon from "../app/images/validation.png";

interface SidebarLinkProps {
  href: string;
  icon: StaticImageData;
  label: string;
  isCollapsed: boolean;
}

function getInitialDarkMode(): boolean {
  if (typeof window === "undefined") return false;

  const stored = localStorage.getItem("theme");
  if (stored === "dark") return true;
  if (stored === "light") return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function SidebarLink({ href, icon, label, isCollapsed }: SidebarLinkProps) {
  return (
    <Link
      href={href}
      className={`relative group flex items-center px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-blue-50 dark:hover:bg-gray-700 rounded-md ${
        isCollapsed ? "justify-center" : ""
      }`}
    >
      <div className="w-6 h-6 flex items-center justify-center shrink-0">
        <Image src={icon} alt={label} width={24} height={24} className="object-contain" priority />
      </div>
      {!isCollapsed && <span className="ml-3 whitespace-nowrap">{label}</span>}

      {isCollapsed && (
        <span className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none shadow-md">
          {label}
        </span>
      )}
    </Link>
  );
}

export default function Sidebar() {
  const [darkMode, setDarkMode] = useState<boolean>(getInitialDarkMode);
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  return (
    <aside
      className={`${isCollapsed ? "w-20" : "w-64"} bg-[#FFFFFF] dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col h-full transition-all duration-300 relative`}
    >
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="absolute -right-3 top-9 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full p-1 shadow-sm hover:bg-gray-100 dark:hover:bg-gray-700 z-10 text-gray-500 dark:text-gray-400"
      >
        {isCollapsed ? (
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
          </svg>
        )}
      </button>

      <div className={`p-6 border-b border-gray-100 dark:border-gray-700 ${isCollapsed ? "px-2" : ""}`}>
        <Image src={darkMode ? logoWh : logo} alt="BidFlow Logo" priority className="w-full h-auto object-contain" />
      </div>

      <nav className="flex-1 p-4 space-y-2">
        <SidebarLink href="/" icon={homeIcon} label="홈" isCollapsed={isCollapsed} />
        <SidebarLink href="/dashboard" icon={dashboardIcon} label="문서 대시보드" isCollapsed={isCollapsed} />
        <SidebarLink href="/analysis" icon={analysisIcon} label="분석 결과" isCollapsed={isCollapsed} />
        <SidebarLink href="/validation" icon={validationIcon} label="자격 검증" isCollapsed={isCollapsed} />

        <div className="my-2 border-t border-gray-100 dark:border-gray-700" />

        <Link
          href="/team"
          className={`relative group flex items-center px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-blue-50 dark:hover:bg-gray-700 rounded-md ${
            isCollapsed ? "justify-center" : ""
          }`}
        >
          <div className="w-6 h-6 flex items-center justify-center shrink-0 text-base">👥</div>
          {!isCollapsed && <span className="ml-3 whitespace-nowrap">팀 워크스페이스</span>}
          {isCollapsed && (
            <span className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none shadow-md">
              팀 워크스페이스
            </span>
          )}
        </Link>

        <Link
          href="/profile"
          className={`relative group flex items-center px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-blue-50 dark:hover:bg-gray-700 rounded-md ${
            isCollapsed ? "justify-center" : ""
          }`}
        >
          <div className="w-6 h-6 flex items-center justify-center shrink-0 text-base">🏢</div>
          {!isCollapsed && <span className="ml-3 whitespace-nowrap">회사 프로필</span>}
          {isCollapsed && (
            <span className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none shadow-md">
              회사 프로필
            </span>
          )}
        </Link>
      </nav>

      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setDarkMode((prev) => !prev)}
          className="w-full flex items-center justify-center px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors whitespace-nowrap overflow-hidden relative group"
        >
          {isCollapsed ? (darkMode ? "🌞" : "🌙") : darkMode ? "🌞 라이트 모드" : "🌙 다크 모드"}
          {isCollapsed && (
            <span className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none shadow-md">
              {darkMode ? "라이트 모드" : "다크 모드"}
            </span>
          )}
        </button>
      </div>
    </aside>
  );
}
