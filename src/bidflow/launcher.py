"""
BidFlow 통합 런처
FastAPI(uvicorn)와 Streamlit을 하나의 명령으로 동시 구동합니다.

실행 방법:
    python run.py          # 프로젝트 루트에서
    bidflow-run            # pip install -e . 후 CLI 명령어
"""
import logging
import os
import secrets
import signal
import subprocess
import sys
import time

# Langfuse OTEL span 내보내기 실패 로그 억제 (실행에는 영향 없는 경고)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.CRITICAL)

# 프로젝트 루트 경로 (src/bidflow/ 기준으로 두 단계 위)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STREAMLIT_ENTRY = os.path.join(BASE_DIR, "src", "bidflow", "apps", "ui", "Home.py")

_procs: list = []


def _load_env() -> dict:
    """프로젝트 루트의 .env 파일을 읽어 환경 변수 딕셔너리로 반환합니다.
    python-dotenv를 사용하여 따옴표·공백을 자동으로 처리합니다.
    """
    from dotenv import dotenv_values

    env_path = os.path.join(BASE_DIR, ".env")
    env = {**os.environ}

    if not os.path.exists(env_path):
        print(f"[BidFlow] 경고: .env 파일이 없습니다. ({env_path})")
        print("[BidFlow] OPENAI_API_KEY 등 환경 변수가 시스템에 설정되어 있어야 합니다.")
        return env

    # dotenv_values: 따옴표 자동 제거, 기존 환경변수는 유지(setdefault)
    for key, value in dotenv_values(env_path).items():
        if value is not None:
            env.setdefault(key, value)

    print("[BidFlow] .env 로드 완료")
    return env


def _kill_group(p: subprocess.Popen) -> None:
    """프로세스 그룹 전체를 종료합니다 (uvicorn reloader 자식까지 포함)."""
    try:
        if sys.platform == "win32":
            p.terminate()
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


def _wait_or_kill(p: subprocess.Popen) -> None:
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


def main():
    # Ctrl+C와 SIGTERM을 직접 처리
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    env = _load_env()

    # 서버 시작마다 새 쿠키 서명 키 생성 → 재시작 시 기존 쿠키 자동 무효화
    env["BIDFLOW_COOKIE_KEY"] = secrets.token_hex(32)
    print("[BidFlow] 쿠키 서명 키가 새로 생성되었습니다.")

    # start_new_session=True: 각 서버를 독립 프로세스 그룹으로 실행
    # → run.py 종료 시 os.killpg()로 그룹 전체(uvicorn reloader 자식 포함) 제거 가능
    popen_kwargs: dict = {"env": env}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True

    fastapi_cmd = [
        sys.executable, "-m", "uvicorn",
        "bidflow.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ]
    streamlit_cmd = [
        sys.executable, "-m", "streamlit",
        "run", STREAMLIT_ENTRY,
        "--server.port", "8501",
        "--server.headless", "true",
    ]

    print("[BidFlow] FastAPI 서버 시작 중...")
    _procs.append(subprocess.Popen(fastapi_cmd, **popen_kwargs))

    time.sleep(1)

    print("[BidFlow] Streamlit 서버 시작 중...")
    _procs.append(subprocess.Popen(streamlit_cmd, **popen_kwargs))

    print("\n" + "=" * 50)
    print("  BidFlow 실행 중")
    print("=" * 50)
    print("  FastAPI   → http://localhost:8000")
    print("  Swagger   → http://localhost:8000/docs")
    print("  Streamlit → http://localhost:8501")
    print("=" * 50)
    print("  종료: Ctrl+C")
    print("=" * 50 + "\n")

    for p in _procs:
        p.wait()


if __name__ == "__main__":
    main()
