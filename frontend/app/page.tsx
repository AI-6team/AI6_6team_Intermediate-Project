"use client";

import { useState, useSyncExternalStore } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  apiUrl,
  clearAuthToken,
  getAuthTokenServerSnapshot,
  getAuthTokenSnapshot,
  setAuthToken,
  subscribeAuthToken,
} from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<"login" | "register">("login");
  const authToken = useSyncExternalStore(
    subscribeAuthToken,
    getAuthTokenSnapshot,
    getAuthTokenServerSnapshot
  );
  const isLoggedIn = Boolean(authToken);

  // 로그인 상태
  const [loginId, setLoginId] = useState("");
  const [loginPw, setLoginPw] = useState("");

  // 회원가입 상태
  const [regId, setRegId] = useState("");
  const [regName, setRegName] = useState("");
  const [regEmail, setRegEmail] = useState("");
  const [regTeam, setRegTeam] = useState("");
  const [regRole, setRegRole] = useState<"member" | "leader">("member");
  const [regPw, setRegPw] = useState("");
  const [regPw2, setRegPw2] = useState("");

  const handleLogout = () => {
    clearAuthToken();
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!loginId || !loginPw) {
      alert("아이디와 비밀번호를 입력해주세요.");
      return;
    }

    try {
      const response = await fetch(apiUrl("/auth/login"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: loginId, password: loginPw }),
      });

      if (response.ok) {
        const data = await response.json();
        setAuthToken(data.access_token);
        router.push("/dashboard");
      } else {
        const err = await response.json();
        alert(err.detail || "로그인에 실패했습니다.");
      }
    } catch (error) {
      console.error("Login failed:", error);
      alert("서버 연결 중 오류가 발생했습니다.");
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();

    // Home.py의 유효성 검사 로직 반영
    if (!regId || !regName || !regEmail || !regPw || !regPw2) {
      alert("사용자명, 이름, 이메일, 비밀번호는 필수 입력 항목입니다.");
      return;
    }

    if (!/^[a-zA-Z0-9_]+$/.test(regId)) {
      alert("사용자명은 영문, 숫자, 밑줄(_)만 사용할 수 있습니다.");
      return;
    }

    if (regPw.length < 6) {
      alert("비밀번호는 6자 이상이어야 합니다.");
      return;
    }

    if (regPw !== regPw2) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    if (regTeam && !/^[a-zA-Z0-9_가-힣]+$/.test(regTeam)) {
      alert("회사명은 영문, 숫자, 밑줄, 한글만 사용할 수 있습니다.");
      return;
    }

    try {
      const response = await fetch(apiUrl("/auth/register"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: regId,
          password: regPw,
          name: regName,
          email: regEmail,
          team: regTeam,
          role: regRole,
        }),
      });

      if (response.ok) {
        alert(`'${regName}' 계정이 생성되었습니다! 로그인 탭에서 로그인하세요.`);
        setActiveTab("login");
      } else {
        const err = await response.json();
        alert(err.detail || "회원가입에 실패했습니다.");
      }
    } catch (error) {
      console.error("Registration failed:", error);
      alert("서버 연결 중 오류가 발생했습니다.");
    }
  };

  return (
    <div className="flex flex-col min-h-screen max-w-6xl mx-auto w-full">
      {/* Hero Section */}
      <section className="relative py-20 overflow-hidden bg-white dark:bg-gray-800 rounded-3xl shadow-sm border border-gray-100 dark:border-gray-700 mb-12 mt-4">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-700 dark:to-gray-800 opacity-50" />
        <div className="relative max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Left Column: Text Content */}
            <div className="text-center lg:text-left">
              <div className="inline-block px-4 py-1.5 mb-6 text-sm font-semibold text-blue-600 bg-blue-100 rounded-full dark:bg-blue-900/50 dark:text-blue-300 animate-fade-in-up">
                AI Powered RFP Analysis
              </div>
              <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 dark:text-white mb-6 leading-tight tracking-tight animate-fade-in-up delay-100">
                입찰 성공을 위한 <br className="hidden md:block" />
                <span className="text-blue-600 dark:text-blue-400">가장 확실한 전략, BidFlow</span>
              </h1>
              <p className="text-lg md:text-xl text-gray-600 dark:text-gray-300 mb-10 leading-relaxed animate-fade-in-up delay-200">
                수백 페이지의 제안요청서(RFP)를 AI가 단 몇 초 만에 분석합니다.<br className="hidden sm:block" />
                핵심 요건 추출부터 적격 여부 판정까지, BidFlow와 함께 입찰 경쟁력을 확보하세요.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start animate-fade-in-up delay-300">
                <a
                  href="#features"
                  className="inline-flex items-center justify-center px-8 py-4 text-base font-bold text-gray-700 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-600 transition-all"
                >
                  기능 더 알아보기
                </a>
              </div>
            </div>

            {/* Right Column: Auth Form */}
            <div className="w-full max-w-md mx-auto bg-white dark:bg-gray-900 rounded-2xl shadow-xl overflow-hidden border border-gray-200 dark:border-gray-700 animate-fade-in-up delay-200">
              {isLoggedIn ? (
                <div className="p-8 text-center">
                  <div className="w-20 h-20 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-6 text-4xl">
                    👋
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">환영합니다!</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-8">
                    이미 로그인되어 있습니다.<br />
                    대시보드로 이동하여 작업을 계속하세요.
                  </p>
                  <div className="space-y-3">
                    <Link
                      href="/dashboard"
                      className="block w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg shadow-md transition-colors"
                    >
                      대시보드로 이동
                    </Link>
                    <button
                      onClick={handleLogout}
                      className="block w-full py-3 px-4 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-bold rounded-lg transition-colors"
                    >
                      로그아웃
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex border-b border-gray-200 dark:border-gray-700">
                <button
                  onClick={() => setActiveTab("login")}
                  className={`flex-1 py-4 text-sm font-bold text-center transition-colors ${
                    activeTab === "login"
                      ? "text-blue-600 border-b-2 border-blue-600 bg-blue-50/50 dark:bg-blue-900/20 dark:text-blue-400"
                      : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  }`}
                >
                  로그인
                </button>
                <button
                  onClick={() => setActiveTab("register")}
                  className={`flex-1 py-4 text-sm font-bold text-center transition-colors ${
                    activeTab === "register"
                      ? "text-blue-600 border-b-2 border-blue-600 bg-blue-50/50 dark:bg-blue-900/20 dark:text-blue-400"
                      : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  }`}
                >
                  회원가입
                </button>
              </div>

              <div className="p-8">
                {activeTab === "login" ? (
                  <form onSubmit={handleLogin} className="space-y-5">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">아이디</label>
                      <input type="text" value={loginId} onChange={(e) => setLoginId(e.target.value)} className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent" placeholder="아이디를 입력하세요" />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">비밀번호</label>
                      <input type="password" value={loginPw} onChange={(e) => setLoginPw(e.target.value)} className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent" placeholder="비밀번호를 입력하세요" />
                    </div>
                    <button type="submit" className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg shadow-md transition-colors">
                      로그인
                    </button>
                  </form>
                ) : (
                  <form onSubmit={handleRegister} className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">사용자명 *</label>
                        <input type="text" value={regId} onChange={(e) => setRegId(e.target.value)} className="w-full px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500" placeholder="ID (영문/숫자)" />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">표시 이름 *</label>
                        <input type="text" value={regName} onChange={(e) => setRegName(e.target.value)} className="w-full px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500" placeholder="홍길동" />
                      </div>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">이메일 *</label>
                      <input type="email" value={regEmail} onChange={(e) => setRegEmail(e.target.value)} className="w-full px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500" placeholder="user@example.com" />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">회사명</label>
                      <input type="text" value={regTeam} onChange={(e) => setRegTeam(e.target.value)} className="w-full px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500" placeholder="예: 비드플로우" />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">역할</label>
                      <div className="flex gap-4">
                        <label className="inline-flex items-center">
                          <input type="radio" className="form-radio text-blue-600" name="role" value="member" checked={regRole === "member"} onChange={() => setRegRole("member")} />
                          <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">팀원</span>
                        </label>
                        <label className="inline-flex items-center">
                          <input type="radio" className="form-radio text-blue-600" name="role" value="leader" checked={regRole === "leader"} onChange={() => setRegRole("leader")} />
                          <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">팀장</span>
                        </label>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">비밀번호 *</label>
                        <input type="password" value={regPw} onChange={(e) => setRegPw(e.target.value)} className="w-full px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500" placeholder="6자 이상" />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">비밀번호 확인 *</label>
                        <input type="password" value={regPw2} onChange={(e) => setRegPw2(e.target.value)} className="w-full px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500" />
                      </div>
                    </div>
                    <button type="submit" className="w-full py-3 px-4 bg-green-600 hover:bg-green-700 text-white font-bold rounded-lg shadow-md transition-colors mt-2">
                      가입하기
                    </button>
                  </form>
                )}
              </div>
                </>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            왜 BidFlow인가요?
          </h2>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            복잡한 입찰 과정을 단순화하는 핵심 기능을 제공합니다.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Feature 1 */}
          <div className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:shadow-md transition-all hover:-translate-y-1">
            <div className="w-14 h-14 bg-blue-100 dark:bg-blue-900/30 rounded-2xl flex items-center justify-center mb-6 text-3xl">
              📄
            </div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
              문서 자동 파싱
            </h3>
            <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
              복잡한 표와 서식이 포함된 RFP 문서도 구조를 유지하며 정확하게 텍스트 데이터를 추출합니다.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="px-2.5 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs font-semibold rounded-full">HWP</span>
              <span className="px-2.5 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs font-semibold rounded-full">PDF</span>
            </div>
          </div>

          {/* Feature 2 */}
          <div className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:shadow-md transition-all hover:-translate-y-1">
            <div className="w-14 h-14 bg-indigo-100 dark:bg-indigo-900/30 rounded-2xl flex items-center justify-center mb-6 text-3xl">
              ⚡
            </div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
              지능형 데이터 추출
            </h3>
            <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
              사업명, 예산, 일정, 배점표 등 입찰에 필수적인 핵심 정보를 AI가 자동으로 찾아냅니다.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:shadow-md transition-all hover:-translate-y-1">
            <div className="w-14 h-14 bg-green-100 dark:bg-green-900/30 rounded-2xl flex items-center justify-center mb-6 text-3xl">
              🛡️
            </div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
              자동 자격 검증
            </h3>
            <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
              회사의 보유 면허와 실적을 RFP 요구사항과 비교하여 입찰 가능 여부를 즉시 진단합니다.
            </p>
          </div>
        </div>
      </section>

      {/* Bottom CTA */}
      <section className="mt-12 mb-20 py-16 bg-gray-900 dark:bg-blue-900 rounded-3xl text-center px-6 relative overflow-hidden shadow-xl">
        <div className="absolute top-0 left-0 w-full h-full opacity-20 bg-gradient-to-r from-blue-600 to-purple-600"></div>
        <div className="relative z-10">
          <h2 className="text-3xl font-bold text-white mb-6">
            지금 바로 입찰 분석을 시작하세요
          </h2>
          <p className="text-gray-300 mb-8 max-w-xl mx-auto text-lg">
            더 이상 수작업으로 문서를 검토하며 시간을 낭비하지 마세요.<br />
            BidFlow가 여러분의 입찰 성공 파트너가 되어드립니다.
          </p>
          <Link
            href="/dashboard"
            className="inline-block px-8 py-4 text-base font-bold text-gray-900 bg-white rounded-xl hover:bg-gray-100 transition-colors shadow-lg"
          >
            무료로 시작하기
          </Link>
        </div>
      </section>
    </div>
  );
}
