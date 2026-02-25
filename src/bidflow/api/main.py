import hashlib
from typing import Optional
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bidflow.db import crud
from bidflow.db.database import get_connection, get_db_path, init_db
from bidflow.api.deps import get_current_user
from bidflow.api.routers import ingest, extract, validate
# from bidflow.core.config import get_config # 가상의 설정 로더

class Token(BaseModel):
    """API 인증 토큰 응답 모델"""
    access_token: str = Field(..., description="Bearer 토큰. `user:{username}` 형식입니다.")
    token_type: str = Field("bearer", description="토큰 타입")
    username: str = Field(..., description="로그인한 사용자 ID")
    role: str = Field(..., description="사용자 역할 (e.g., 'member', 'leader')")

class UserResponse(BaseModel):
    """사용자 정보 조회 응답 모델"""
    username: str = Field(..., description="사용자 ID")
    name: str = Field(..., description="사용자 이름")
    email: str = Field(..., description="이메일 주소")
    team: Optional[str] = Field(None, description="소속 팀 이름")
    licenses: Optional[str] = Field(None, description="보유 면허 (쉼표로 구분된 문자열)")
    role: str = Field(..., description="사용자 역할 (e.g., 'member', 'leader')")

class MessageResponse(BaseModel):
    """간단한 메시지 응답 모델"""
    message: str
    username: Optional[str] = None

class UserProfileUpdate(BaseModel):
    """사용자 프로필 업데이트 요청 모델"""
    team: Optional[str] = Field(None, description="소속될 팀의 이름")
    licenses: Optional[str] = Field(None, description="회사가 보유한 면허 목록 (쉼표로 구분)")

class CommentCreate(BaseModel):
    """댓글 작성 요청 모델"""
    doc_hash: str
    text: str

class ReplyCreate(BaseModel):
    """답글 작성 요청 모델"""
    text: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str = Field(..., description="로그인 시 사용할 사용자 ID")
    password: str = Field(..., description="비밀번호")
    name: str = Field(..., description="사용자 실명")
    email: str = Field(..., description="이메일 주소")
    team: Optional[str] = Field(None, description="소속될 팀의 이름")
    licenses: Optional[str] = Field(None, description="회사가 보유한 면허 목록 (쉼표로 구분)", example="소프트웨어사업자, 정보통신공사업")
    role: str = Field("member", description="사용자 역할 ('member' 또는 'leader')")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_app() -> FastAPI:
    """FastAPI 애플리케이션 팩토리"""
    app = FastAPI(
        title="BidFlow API",
        description="AI 기반 입찰 제안요청서(RFP) 분석 시스템 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.on_event("startup")
    def on_startup():
        init_db()

    # CORS 설정 (Streamlit이 다른 포트에서 실행되므로 필수)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 보안상 운영 배포 시 구체적인 도메인으로 제한 필요
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])
    app.include_router(extract.router, prefix="/api/v1", tags=["Extraction"])
    app.include_router(validate.router, prefix="/api/v1", tags=["Validation"])

    # Auth 엔드포인트 직접 정의 (auth.py 대체)
    @app.post("/auth/login", response_model=Token, tags=["Authentication"])
    async def login(user_in: UserLogin):
        """
        사용자 로그인 및 인증 토큰 발급.
        - **access_token**: `user:{username}` 형식의 단순 토큰을 사용합니다. 실제 프로덕션에서는 JWT를 사용해야 합니다.
        """
        user = crud.get_user(user_in.username)
        if not user:
            raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
        
        if hash_password(user_in.password) != user["password_hash"]:
            raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
        
        return {
            "access_token": f"user:{user['username']}", 
            "token_type": "bearer",
            "username": user['username'],
            "role": user['role']
        }

    @app.post("/auth/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED, tags=["Authentication"])
    async def register(user_in: UserRegister):
        """
        새로운 사용자를 등록합니다.
        - `licenses` 필드에 회사가 보유한 면허를 쉼표로 구분하여 입력하면, 자격 검증 시 자동으로 사용됩니다.
        """
        existing_user = crud.get_user(user_in.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="이미 존재하는 사용자명입니다.")
        
        crud.upsert_user(
            username=user_in.username,
            name=user_in.name,
            email=user_in.email,
            password_hash=hash_password(user_in.password),
            team=user_in.team or "",
            licenses=user_in.licenses or "",
            role=user_in.role
        )
        
        return {"message": "회원가입이 완료되었습니다.", "username": user_in.username}

    @app.put("/auth/me", response_model=UserResponse, tags=["Authentication"])
    async def update_user_profile(
        profile_update: UserProfileUpdate,
        current_user: dict = Depends(get_current_user)
    ):
        """
        현재 로그인된 사용자의 프로필(팀, 보유 면허)을 수정합니다.
        - **팀장(leader)만 이 API를 호출할 수 있습니다.**
        """
        if current_user.get("role") != "leader":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="팀장 권한이 필요합니다."
            )

        # Get current values to avoid overwriting with None if not provided
        updated_team = profile_update.team if profile_update.team is not None else current_user.get("team", "")
        updated_licenses = profile_update.licenses if profile_update.licenses is not None else current_user.get("licenses", "")

        crud.update_user_profile(
            username=current_user["username"],
            team=updated_team,
            licenses=updated_licenses,
        )

        # Return the updated user info
        return crud.get_user(current_user["username"])

    @app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
    async def read_users_me(current_user: dict = Depends(get_current_user)):
        """
        현재 로그인된 사용자의 정보를 반환합니다.
        """
        return {
            "username": current_user["username"],
            "name": current_user["name"],
            "email": current_user["email"],
            "team": current_user["team"],
            "licenses": current_user.get("licenses", ""),
            "role": current_user["role"]
        }

    @app.get("/api/v1/comments/{doc_hash}", tags=["Comments"])
    def get_document_comments(
        doc_hash: str,
        current_user: dict = Depends(get_current_user)
    ):
        """문서의 댓글 목록을 조회합니다 (팀 단위 공유)."""
        team = current_user.get("team", "")
        return crud.get_comments(team, doc_hash)

    @app.post("/api/v1/comments", tags=["Comments"])
    def create_comment(
        comment_in: CommentCreate,
        current_user: dict = Depends(get_current_user)
    ):
        """새 댓글을 작성합니다."""
        team = current_user.get("team", "")
        if not team:
            raise HTTPException(status_code=400, detail="팀에 소속되어 있어야 댓글을 작성할 수 있습니다.")
        
        comment_id = crud.add_comment(
            team_name=team,
            doc_hash=comment_in.doc_hash,
            author=current_user["username"],
            author_name=current_user["name"],
            text=comment_in.text
        )
        return {"id": comment_id, "message": "Comment added"}

    @app.post("/api/v1/comments/{comment_id}/replies", tags=["Comments"])
    def create_reply(
        comment_id: str,
        reply_in: ReplyCreate,
        current_user: dict = Depends(get_current_user)
    ):
        """답글을 작성합니다."""
        reply_id = crud.add_reply(
            comment_id=comment_id,
            author=current_user["username"],
            author_name=current_user["name"],
            text=reply_in.text
        )
        return {"id": reply_id, "message": "Reply added"}

    @app.delete("/api/v1/comments/{comment_id}", tags=["Comments"])
    def delete_comment(comment_id: str, current_user: dict = Depends(get_current_user)):
        """댓글을 삭제합니다 (작성자 본인만 가능)."""
        if not crud.delete_comment(comment_id, current_user["username"]):
            raise HTTPException(status_code=403, detail="삭제 권한이 없거나 존재하지 않는 댓글입니다.")
        return {"message": "Comment deleted"}

    @app.delete("/api/v1/replies/{reply_id}", tags=["Comments"])
    def delete_reply(reply_id: str, current_user: dict = Depends(get_current_user)):
        """답글을 삭제합니다 (작성자 본인만 가능)."""
        if not crud.delete_reply(reply_id, current_user["username"]):
            raise HTTPException(status_code=403, detail="삭제 권한이 없거나 존재하지 않는 답글입니다.")
        return {"message": "Reply deleted"}

    @app.get("/health", tags=["System"])
    def health_check():
        """
        시스템 및 DB 연결 상태를 확인합니다.
        - DB 파일의 절대 경로를 함께 반환하여 디버깅을 돕습니다.
        """
        try:
            conn = get_connection()
            conn.execute("SELECT 1")
            conn.close()
            return {"status": "ok", "database": "connected", "db_path": get_db_path()}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Database connection failed: {str(e)}"
            )

    @app.get("/")
    def root():
        return {"message": "BidFlow API가 실행 중입니다. 문서: /docs"}

    return app

app = create_app()