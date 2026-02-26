from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from bidflow.db import crud
from bidflow.security.jwt_tokens import decode_access_token
import logging

logger = logging.getLogger("uvicorn.error")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # 토큰 정제: 앞뒤 공백 및 따옴표 제거 (클라이언트에서 잘못 보낼 경우 대비)
    token = token.strip().strip('"')

    try:
        payload = decode_access_token(token)
        username = payload["sub"]
    except ValueError as exc:
        logger.warning(f"Auth Failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = crud.get_user(username)
    if user is None:
        logger.warning(f"Auth Failed: User not found. Username: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
