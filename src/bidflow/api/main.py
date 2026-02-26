import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bidflow.api.deps import get_current_user
from bidflow.api.rate_limit import RateLimitMiddleware
from bidflow.api.routers import extract, ingest, validate
from bidflow.db import crud
from bidflow.db.database import get_connection, get_db_engine, get_db_path, init_db
from bidflow.domain.models import CompanyProfile, ComplianceMatrix, ExtractionSlot
from bidflow.security.jwt_tokens import create_access_token
from bidflow.security.passwords import hash_password, needs_hash_upgrade, verify_password


class Token(BaseModel):
    """API 인증 토큰 응답 모델"""

    access_token: str = Field(..., description="Bearer JWT 토큰")
    token_type: str = Field("bearer", description="토큰 타입")
    expires_in: int = Field(..., description="토큰 만료까지 남은 시간(초)")
    username: str = Field(..., description="로그인한 사용자 ID")
    role: str = Field(..., description="사용자 역할 (e.g., 'member', 'leader')")


class UserResponse(BaseModel):
    """사용자 정보 조회 응답 모델"""

    username: str = Field(..., description="사용자 ID")
    name: str = Field(..., description="사용자 이름")
    email: str = Field(..., description="이메일 주소")
    team: Optional[str] = Field(None, description="소속 팀 이름")
    licenses: Optional[str] = Field(None, description="보유 면허 (쉼표로 구분된 문자열)")
    region: Optional[str] = Field(None, description="회사 지역 정보")
    company_logo: Optional[str] = Field(None, description="회사 로고(data URL)")
    role: str = Field(..., description="사용자 역할 (e.g., 'member', 'leader')")


class MessageResponse(BaseModel):
    """간단한 메시지 응답 모델"""

    message: str
    username: Optional[str] = None


class UserProfileUpdate(BaseModel):
    """사용자 프로필 업데이트 요청 모델"""

    team: Optional[str] = Field(None, description="소속될 팀의 이름")
    licenses: Optional[str] = Field(None, description="회사가 보유한 면허 목록 (쉼표로 구분)")
    region: Optional[str] = Field(None, description="회사 지역 정보")
    company_logo: Optional[str] = Field(None, description="회사 로고(data URL)")


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
    licenses: Optional[str] = Field(
        None,
        description="회사가 보유한 면허 목록 (쉼표로 구분)",
        example="소프트웨어사업자, 정보통신공사업",
    )
    role: str = Field("member", description="사용자 역할 ('member' 또는 'leader')")


def _parse_licenses(licenses_raw: Optional[str]) -> List[str]:
    if not licenses_raw:
        return []
    return [s.strip() for s in str(licenses_raw).split(",") if s and s.strip()]


def _normalize_company_logo(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None

    value = str(raw).strip()
    if not value:
        return ""

    if not value.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="company_logo는 data:image/* 형식이어야 합니다.")

    # 대략 2MB data URL 제한 (문자열 길이 기준)
    if len(value) > 2_800_000:
        raise HTTPException(status_code=400, detail="company_logo 크기가 너무 큽니다. (2MB 이하)")

    return value


def _load_team_profile(current_user: Dict[str, Any]) -> Optional[CompanyProfile]:
    team_name = (current_user.get("team") or "").strip()
    if not team_name:
        return None

    try:
        from bidflow.ingest.storage import DocumentStore

        store = DocumentStore(user_id=current_user.get("username", "global"), team_name=team_name)
        profile_data = store.load_profile()
        if not profile_data:
            return None
        return CompanyProfile(**profile_data)
    except Exception:
        return None


def _build_effective_profile(current_user: Dict[str, Any]) -> CompanyProfile:
    """팀 공유 프로필이 있으면 우선 사용하고, 없으면 사용자 정보를 fallback으로 사용합니다."""
    team_profile = _load_team_profile(current_user)
    if team_profile:
        data = dict(team_profile.data or {})
        raw_licenses = data.get("licenses", [])
        if isinstance(raw_licenses, list):
            licenses = [str(x).strip() for x in raw_licenses if str(x).strip()]
        elif isinstance(raw_licenses, str):
            licenses = _parse_licenses(raw_licenses)
        else:
            licenses = _parse_licenses(current_user.get("licenses", ""))
        data["licenses"] = licenses
        data["region"] = str(data.get("region", "")).strip()
        return CompanyProfile(
            id=str(team_profile.id),
            name=team_profile.name or (current_user.get("team") or ""),
            data=data,
        )

    licenses_list = _parse_licenses(current_user.get("licenses", ""))
    return CompanyProfile(
        id=current_user.get("username", "unknown"),
        name=current_user.get("team", ""),
        data={"licenses": licenses_list, "region": ""},
    )


def _build_user_response(current_user: Dict[str, Any]) -> Dict[str, Any]:
    """사용자 응답을 팀 공유 프로필 기준(없으면 fallback)으로 구성합니다."""
    region = ""
    company_logo: Optional[str] = None
    team_profile = _load_team_profile(current_user)
    if team_profile:
        raw_licenses = team_profile.data.get("licenses", [])
        if isinstance(raw_licenses, list):
            licenses_str = ", ".join(str(x).strip() for x in raw_licenses if str(x).strip())
        elif isinstance(raw_licenses, str):
            licenses_str = ", ".join(_parse_licenses(raw_licenses))
        else:
            licenses_str = current_user.get("licenses", "")

        profile_data = team_profile.data if isinstance(team_profile.data, dict) else {}
        region = str(profile_data.get("region", "")).strip()
        logo_value = profile_data.get("company_logo")
        if isinstance(logo_value, str) and logo_value.strip():
            company_logo = logo_value.strip()
    else:
        licenses_str = current_user.get("licenses", "")

    return {
        "username": current_user["username"],
        "name": current_user["name"],
        "email": current_user["email"],
        "team": current_user["team"],
        "licenses": licenses_str,
        "region": region,
        "company_logo": company_logo,
        "role": current_user["role"],
    }


def _coerce_extraction_slot(slot_key: str, slot_payload: Dict[str, Any]) -> Optional[ExtractionSlot]:
    """이전/혼합 스키마를 ExtractionSlot 모델로 정규화합니다."""
    if not isinstance(slot_payload, dict):
        return None

    normalized = dict(slot_payload)
    normalized.setdefault("key", slot_key)
    if "status" not in normalized:
        signal = str(normalized.get("signal", "")).upper()
        normalized["status"] = "FOUND" if signal in {"GREEN", "RED"} else "NOT_FOUND"
    if not isinstance(normalized.get("evidence"), list):
        normalized["evidence"] = []

    try:
        return ExtractionSlot(**normalized)
    except Exception:
        return None


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

    cors_origins = [
        origin.strip()
        for origin in os.getenv("BIDFLOW_CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)

    # 라우터 등록
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])
    app.include_router(extract.router, prefix="/api/v1", tags=["Extraction"])
    app.include_router(validate.router, prefix="/api/v1", tags=["Validation"])

    @app.post("/auth/login", response_model=Token, tags=["Authentication"])
    async def login(user_in: UserLogin):
        """
        사용자 로그인 및 인증 토큰 발급.
        - **access_token**: JWT(HS256 기본) 형식 Bearer 토큰
        """
        user = crud.get_user(user_in.username)
        if not user:
            raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

        if not verify_password(user_in.password, user.get("password_hash")):
            raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

        if needs_hash_upgrade(user.get("password_hash")):
            upgraded_hash = hash_password(user_in.password)
            crud.update_user_password_hash(user["username"], upgraded_hash)

        access_token, expires_in = create_access_token(user["username"], user.get("role", "member"))
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": expires_in,
            "username": user["username"],
            "role": user["role"],
        }

    @app.post(
        "/auth/register",
        response_model=MessageResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Authentication"],
    )
    async def register(user_in: UserRegister):
        """
        새로운 사용자를 등록합니다.
        - `licenses` 필드에 회사가 보유한 면허를 쉼표로 구분하여 입력하면, 자격 검증 시 자동으로 사용됩니다.
        """
        role = (user_in.role or "member").strip().lower()
        if role not in {"member", "leader"}:
            raise HTTPException(status_code=400, detail="role은 'member' 또는 'leader'만 허용됩니다.")

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
            role=role,
        )

        return {"message": "회원가입이 완료되었습니다.", "username": user_in.username}

    @app.put("/auth/me", response_model=UserResponse, tags=["Authentication"])
    async def update_user_profile(
        profile_update: UserProfileUpdate,
        current_user: dict = Depends(get_current_user),
    ):
        """
        현재 로그인된 사용자의 프로필(팀, 보유 면허)을 수정합니다.
        - **팀장(leader)만 이 API를 호출할 수 있습니다.**
        """
        if current_user.get("role") != "leader":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="팀장 권한이 필요합니다.",
            )

        current_team = (current_user.get("team") or "").strip()
        updated_team = (
            profile_update.team.strip()
            if profile_update.team is not None and profile_update.team.strip()
            else current_team
        )
        updated_licenses_raw = (
            profile_update.licenses
            if profile_update.licenses is not None
            else current_user.get("licenses", "")
        )
        updated_licenses = ", ".join(_parse_licenses(updated_licenses_raw))
        updated_region = profile_update.region.strip() if profile_update.region is not None else None
        updated_logo = _normalize_company_logo(profile_update.company_logo)

        if current_team and updated_team and current_team != updated_team:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="팀 이름 변경은 지원되지 않습니다. 팀 이름은 동일하게 유지해주세요.",
            )

        crud.update_user_profile(
            username=current_user["username"],
            team=updated_team,
            licenses=updated_licenses,
        )

        # 팀 공유 프로필 1개 정책: 팀 프로필 라이선스를 갱신하고 팀원 fallback 데이터도 동기화
        if updated_team:
            try:
                from bidflow.ingest.storage import DocumentStore

                store = DocumentStore(user_id=current_user["username"], team_name=updated_team)
                existing_profile = store.load_profile() or {}
                profile_payload = existing_profile if isinstance(existing_profile, dict) else {}
                profile_payload.setdefault("id", updated_team)
                profile_payload["name"] = updated_team

                data = profile_payload.get("data", {})
                if not isinstance(data, dict):
                    data = {}
                data["licenses"] = _parse_licenses(updated_licenses_raw)
                if updated_region is not None:
                    data["region"] = updated_region
                else:
                    data["region"] = str(data.get("region", "")).strip()
                if updated_logo is not None:
                    data["company_logo"] = updated_logo
                profile_payload["data"] = data
                store.save_profile(profile_payload)

                crud.update_team_licenses(updated_team, updated_licenses)
            except Exception:
                # 팀 공유 프로필 동기화 실패 시에도 사용자 기본 업데이트는 유지
                pass

        refreshed_user = crud.get_user(current_user["username"])
        if not refreshed_user:
            raise HTTPException(status_code=404, detail="사용자 정보를 찾을 수 없습니다.")
        return _build_user_response(refreshed_user)

    @app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
    async def read_users_me(current_user: dict = Depends(get_current_user)):
        """
        현재 로그인된 사용자의 정보를 반환합니다.
        """
        return _build_user_response(current_user)

    @app.get("/api/v1/comments/{doc_hash}", tags=["Comments"])
    def get_document_comments(doc_hash: str, current_user: dict = Depends(get_current_user)):
        """문서의 댓글 목록을 조회합니다 (팀 단위 공유)."""
        team = current_user.get("team", "")
        return crud.get_comments(team, doc_hash)

    @app.post("/api/v1/comments", tags=["Comments"])
    def create_comment(comment_in: CommentCreate, current_user: dict = Depends(get_current_user)):
        """새 댓글을 작성합니다."""
        team = current_user.get("team", "")
        if not team:
            raise HTTPException(status_code=400, detail="팀에 소속되어 있어야 댓글을 작성할 수 있습니다.")

        comment_id = crud.add_comment(
            team_name=team,
            doc_hash=comment_in.doc_hash,
            author=current_user["username"],
            author_name=current_user["name"],
            text=comment_in.text,
        )
        return {"id": comment_id, "message": "Comment added"}

    @app.post("/api/v1/comments/{comment_id}/replies", tags=["Comments"])
    def create_reply(comment_id: str, reply_in: ReplyCreate, current_user: dict = Depends(get_current_user)):
        """답글을 작성합니다."""
        reply_id = crud.add_reply(
            comment_id=comment_id,
            author=current_user["username"],
            author_name=current_user["name"],
            text=reply_in.text,
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

    @app.get("/api/v1/team/members", tags=["Team"])
    def get_team_members(current_user: dict = Depends(get_current_user)):
        """현재 사용자의 팀원 목록을 반환합니다."""
        team = current_user.get("team", "")
        if not team:
            return []
        return crud.get_team_members(team)

    @app.get("/api/v1/team/documents", tags=["Team"])
    def get_team_documents(current_user: dict = Depends(get_current_user)):
        """팀 전체의 문서 목록을 반환합니다 (업로더 정보 포함)."""
        team = current_user.get("team", "")
        if not team:
            return []

        from bidflow.ingest.storage import DocumentStore

        store = DocumentStore(user_id=current_user["username"])
        docs = store.list_documents()

        deduped: Dict[tuple, Dict[str, Any]] = {}
        for doc in docs:
            key = (doc.get("doc_hash"), doc.get("user_id"))
            if key in deduped:
                continue
            enriched = dict(doc)
            enriched["uploaded_by"] = enriched.get("user_id")
            enriched["uploaded_by_name"] = enriched.get("owner_name") or enriched.get("user_id")
            deduped[key] = enriched

        all_docs = list(deduped.values())
        all_docs.sort(key=lambda d: d.get("upload_date") or "", reverse=True)
        return all_docs

    @app.get("/api/v1/team/decision/{doc_hash}", tags=["Team"])
    def get_decision_summary(doc_hash: str, current_user: dict = Depends(get_current_user)):
        """문서에 대한 판정 결과 요약을 반환합니다."""
        from bidflow.ingest.storage import DocumentStore
        from bidflow.validation.validator import RuleBasedValidator

        user_id = current_user.get("username", "")
        store = DocumentStore(user_id=user_id)
        extraction = store.load_extraction_result(doc_hash)
        if not extraction:
            return None

        try:
            profile = _build_effective_profile(current_user)
            slots_map: Dict[str, ExtractionSlot] = {}
            for group in ["g1", "g2", "g3"]:
                if group in extraction:
                    for key, payload in extraction[group].items():
                        coerced = _coerce_extraction_slot(key, payload)
                        if coerced:
                            slots_map[key] = coerced

            if not slots_map:
                return None

            matrix = ComplianceMatrix(doc_hash=doc_hash, slots=slots_map)
            validator = RuleBasedValidator()
            decisions = validator.validate(matrix, profile)
            rec = validator.get_recommendation(decisions)

            return {
                "signal": rec["signal"],
                "recommendation": rec["recommendation"],
                "n_red": sum(1 for d in decisions if d.decision == "RED"),
                "n_gray": sum(1 for d in decisions if d.decision == "GRAY"),
                "n_green": sum(1 for d in decisions if d.decision == "GREEN"),
            }
        except Exception:
            return None

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
            return {
                "status": "ok",
                "database": "connected",
                "db_engine": get_db_engine(),
                "db_path": get_db_path(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

    @app.get("/")
    def root():
        return {"message": "BidFlow API가 실행 중입니다. 문서: /docs"}

    return app


app = create_app()
