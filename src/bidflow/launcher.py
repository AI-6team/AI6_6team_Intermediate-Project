"""BidFlow unified launcher.

Starts FastAPI and Next.js together with one command.
"""

from __future__ import annotations

import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from typing import Any

# Langfuse OTEL span 내보내기 실패 로그 억제 (실행에는 영향 없는 경고)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.getenv("BIDFLOW_FRONTEND_DIR", os.path.join(BASE_DIR, "frontend"))

_procs: list[subprocess.Popen[Any]] = []


def _load_env() -> dict[str, str]:
    """프로젝트 루트의 .env 파일을 읽어 환경 변수 딕셔너리로 반환합니다."""
    from dotenv import dotenv_values

    env_path = os.path.join(BASE_DIR, ".env")
    env = {**os.environ}

    if not os.path.exists(env_path):
        print(f"[BidFlow] 경고: .env 파일이 없습니다. ({env_path})")
        return env

    for key, value in dotenv_values(env_path).items():
        if value is not None:
            env.setdefault(key, value)

    print("[BidFlow] .env 로드 완료")
    return env


def _spawn(cmd: list[str], env: dict[str, str], cwd: str | None = None) -> subprocess.Popen[Any]:
    kwargs: dict[str, Any] = {"env": env, "cwd": cwd}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _kill_group(p: subprocess.Popen[Any]) -> None:
    """프로세스 그룹 전체를 종료합니다."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                text=True,
            )
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


def _wait_or_kill(p: subprocess.Popen[Any]) -> None:
    """종료 대기 후 타임아웃 시 강제 종료합니다."""
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if sys.platform == "win32":
                p.kill()
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def _shutdown(sig=None, frame=None) -> None:
    """Ctrl+C / SIGTERM 수신 시 모든 서버를 정상 종료합니다."""
    print("\n[BidFlow] 종료 신호 수신. 서버를 내립니다...")
    for p in _procs:
        _kill_group(p)
    for p in _procs:
        _wait_or_kill(p)
    print("[BidFlow] 모든 서버가 종료되었습니다.")
    sys.exit(0)


def _npm_executable() -> str:
    return "npm.cmd" if sys.platform == "win32" else "npm"


def _is_port_busy(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _find_free_port(start: int, end: int) -> int | None:
    for p in range(start, end + 1):
        if not _is_port_busy(p):
            return p
    return None


def main() -> None:
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    env = _load_env()
    env["BIDFLOW_COOKIE_KEY"] = secrets.token_hex(32)

    api_host = env.get("BIDFLOW_API_HOST", "0.0.0.0")
    api_port = env.get("BIDFLOW_API_PORT", "8000")
    web_port = env.get("BIDFLOW_WEB_PORT", "3000")

    # 이미 사용 중인 포트는 자동 회피 (기존 구버전 서버와 충돌 방지)
    try:
        api_port_int = int(api_port)
    except ValueError:
        api_port_int = 8000

    if _is_port_busy(api_port_int):
        alt_api = _find_free_port(8100, 8199)
        if alt_api is None:
            raise RuntimeError("사용 가능한 API 포트를 찾지 못했습니다. (8100-8199)")
        print(f"[BidFlow] 경고: API 포트 {api_port_int} 사용 중 -> {alt_api}로 변경")
        api_port_int = alt_api

    try:
        web_port_int = int(web_port)
    except ValueError:
        web_port_int = 3000

    if _is_port_busy(web_port_int):
        alt_web = _find_free_port(3100, 3199)
        if alt_web is None:
            raise RuntimeError("사용 가능한 웹 포트를 찾지 못했습니다. (3100-3199)")
        print(f"[BidFlow] 경고: 웹 포트 {web_port_int} 사용 중 -> {alt_web}로 변경")
        web_port_int = alt_web

    api_port = str(api_port_int)
    web_port = str(web_port_int)
    env.setdefault("NEXT_PUBLIC_API_BASE_URL", f"http://localhost:{api_port}")
    configured_origins = [
        origin.strip()
        for origin in env.get("BIDFLOW_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if not configured_origins:
        configured_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    runtime_origins = [f"http://localhost:{web_port}", f"http://127.0.0.1:{web_port}"]
    for origin in runtime_origins:
        if origin not in configured_origins:
            configured_origins.append(origin)
    env["BIDFLOW_CORS_ORIGINS"] = ",".join(configured_origins)

    if not os.path.exists(os.path.join(FRONTEND_DIR, "package.json")):
        raise FileNotFoundError(
            f"Next.js frontend 경로를 찾을 수 없습니다: {os.path.join(FRONTEND_DIR, 'package.json')}"
        )

    fastapi_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "bidflow.main:app",
        "--host",
        api_host,
        "--port",
        api_port,
        "--reload",
    ]
    next_cmd = [_npm_executable(), "run", "dev", "--", "--port", web_port]

    print("[BidFlow] FastAPI 서버 시작 중...")
    _procs.append(_spawn(fastapi_cmd, env=env, cwd=BASE_DIR))
    time.sleep(1)

    print("[BidFlow] Next.js 서버 시작 중...")
    _procs.append(_spawn(next_cmd, env=env, cwd=FRONTEND_DIR))

    print("\n" + "=" * 50)
    print("  BidFlow 실행 중")
    print("=" * 50)
    print(f"  Frontend  → http://localhost:{web_port}")
    print(f"  FastAPI   → http://localhost:{api_port}")
    print(f"  Swagger   → http://localhost:{api_port}/docs")
    print("=" * 50)
    print("  종료: Ctrl+C")
    print("=" * 50 + "\n")

    for p in _procs:
        p.wait()


if __name__ == "__main__":
    main()
