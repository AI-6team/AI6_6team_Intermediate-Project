from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_current_user(api_key_header: str = Security(api_key_header)):
    """
    내부/토큰 접근을 위한 간단한 API 키 검증입니다.
    """
    # MVP에서는 ADMIN_PASSWORD 또는 .env의 특정 API 토큰과 비교합니다.
    expected_key = os.getenv("ADMIN_PASSWORD", "secret") 
    # 또는 별도의 내부 토큰을 사용합니다.
    # 설계서에 Token/Internal로 명시되어 있습니다. 편의상 ADMIN_PASSWORD를 공유 토큰으로 사용합니다.
    
    if api_key_header == expected_key:
        return "admin"
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="자격 증명을 검증할 수 없습니다."
    )
