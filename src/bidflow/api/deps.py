import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

load_dotenv()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _load_api_keys() -> dict:
    """
    환경변수 BIDFLOW_API_KEYS에서 API 키를 로드합니다.
    형식: "sk-key1:user1,sk-key2:user2"
    """
    raw = os.getenv("BIDFLOW_API_KEYS", "")
    api_keys = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, user = pair.split(":", 1)
            api_keys[key.strip()] = user.strip()
    return api_keys


def get_current_user(api_key: str = Security(api_key_header)) -> str:
    """
    X-API-Key 헤더를 검증하고 user_id를 반환합니다.
    환경변수 BIDFLOW_API_KEYS에서 키:사용자 매핑을 조회합니다.
    """
    api_keys = _load_api_keys()
    if api_key and api_key in api_keys:
        return api_keys[api_key]

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="유효하지 않은 API 키입니다."
    )
