"use client";

import Image from 'next/image';
import Link from 'next/link';
import { useState, useEffect } from 'react';
// app 폴더에 있는 이미지를 import (경로: ../app/logo.png)
import logo from '../app/images/logo_wh.png';
import logoWh from '../app/images/logo_bk.png';
import homeIcon from '../app/images/home.png';
import dashboardIcon from '../app/images/dashboard.png';
import analysisIcon from '../app/images/analysis.png';
import validationIcon from '../app/images/validation.png';

export default function Sidebar() {
  const [darkMode, setDarkMode] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    // 초기 테마 설정
    if (localStorage.getItem('theme') === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      setDarkMode(true);
      document.documentElement.classList.add('dark');
    } else {
      setDarkMode(false);
      document.documentElement.classList.remove('dark');
    }
  }, []);

  const toggleDarkMode = () => {
    if (darkMode) {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      setDarkMode(false);
    } else {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      setDarkMode(true);
    }
  };

  // 툴팁이 포함된 사이드바 링크 컴포넌트
  const SidebarLink = ({ href, icon, label }: { href: string; icon: any; label: string }) => (
    <Link
      href={href}
      className={`relative group flex items-center px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-blue-50 dark:hover:bg-gray-700 rounded-md ${
        isCollapsed ? 'justify-center' : ''
      }`}
    >
      <div className="w-6 h-6 flex items-center justify-center shrink-0">
        <Image 
          src={icon} 
          alt={label} 
          width={24} 
          height={24} 
          className="object-contain"
          priority // 이미지 우선 로딩
        />
      </div>
      {!isCollapsed && <span className="ml-3 whitespace-nowrap">{label}</span>}
      
      {/* 커스텀 툴팁 (접혔을 때만 hover 시 표시) */}
      {isCollapsed && (
        <span className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none shadow-md">
          {label}
        </span>
      )}
    </Link>
  );

  return (
    <aside className={`${isCollapsed ? 'w-20' : 'w-64'} bg-[#FFFFFF] dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col h-full transition-all duration-300 relative`}>
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

      <div className={`p-6 border-b border-gray-100 dark:border-gray-700 ${isCollapsed ? 'px-2' : ''}`}>
        <Image src={darkMode ? logoWh : logo} alt="BidFlow Logo" priority className="w-full h-auto object-contain" />
      </div>
      
      {/* 툴팁이 잘리지 않도록 overflow-hidden 제거 */}
      <nav className="flex-1 p-4 space-y-2">
        <SidebarLink href="/" icon={homeIcon} label="홈" />
        <SidebarLink href="/dashboard" icon={dashboardIcon} label="문서 대시보드" />
        <SidebarLink href="/analysis" icon={analysisIcon} label="분석 결과" />
        <SidebarLink href="/validation" icon={validationIcon} label="자격 검증" />
      </nav>

      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={toggleDarkMode}
          className={`w-full flex items-center justify-center px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors whitespace-nowrap overflow-hidden relative group`}
        >
          {isCollapsed ? (darkMode ? '🌞' : '🌙') : (darkMode ? '🌞 라이트 모드' : '🌙 다크 모드')}
           {/* 다크 모드 토글 버튼 툴팁 */}
           {isCollapsed && (
            <span className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none shadow-md">
              {darkMode ? '라이트 모드' : '다크 모드'}
            </span>
          )}
        </button>
      </div>
    </aside>
  );
}
