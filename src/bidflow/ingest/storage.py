import os
import shutil
from typing import List, Optional, Dict, Any
from bidflow.domain.models import RFPDocument, ParsedChunk
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LancChainDocument
from bidflow.db import crud


class StorageRegistry:
    """config 기반 스토리지 공간 경로 계산기."""

    def __init__(self, config=None):
        if config is None:
            from bidflow.core.config import get_config
            config = get_config("dev")
        storage = config.storage if config.storage else None
        self.base = (storage.base if storage and storage.base else None) or "data"
        self.accounts_dir = (storage.accounts_dir if storage and storage.accounts_dir else None)

        if storage and storage.user_spaces:
            self._user_spaces = [s["name"] for s in storage.user_spaces]
        else:
            self._user_spaces = ["raw", "processed", "vectordb"]

        if storage and storage.shared_spaces:
            self._shared_spaces = [s["name"] for s in storage.shared_spaces]
        else:
            self._shared_spaces = ["raw", "knowledge", "templates"]

    def user_space(self, user_id: str, space: str) -> str:
        """data/accounts/{user_id}/{space}/"""
        if space not in self._user_spaces:
            raise ValueError(f"알 수 없는 user space: {space}. 가능: {self._user_spaces}")
        return os.path.join(self.base, self.accounts_dir, user_id, space) if self.accounts_dir else os.path.join(self.base, user_id, space)

    def shared_space(self, space: str) -> str:
        """data/shared/{space}/"""
        if space not in self._shared_spaces:
            raise ValueError(f"알 수 없는 shared space: {space}. 가능: {self._shared_spaces}")
        return os.path.join(self.base, "shared", space)

    def user_base(self, user_id: str) -> str:
        """data/accounts/{user_id}/"""
        return os.path.join(self.base, self.accounts_dir, user_id) if self.accounts_dir else os.path.join(self.base, user_id)

    def ensure_spaces(self, user_id: str):
        """사용자 디렉토리 전체 생성."""
        for space in self._user_spaces:
            os.makedirs(self.user_space(user_id, space), exist_ok=True)
        for space in self._shared_spaces:
            os.makedirs(self.shared_space(space), exist_ok=True)

    def team_space(self, team_name: str, space: str) -> str:
        """data/shared/teams/{team_name}/{space}/"""
        return os.path.join(self.base, "shared", "teams", team_name, space)

    def ensure_team_spaces(self, team_name: str):
        """팀 공간 디렉토리 생성."""
        os.makedirs(self.team_space(team_name, "comments"), exist_ok=True)


class DocumentStore:
    """
    파싱된 RFPDocument(JSON)와 원본 파일을 로컬 저장소에서 관리합니다.
    user_id가 주어지면 data/{user_id}/ 하위에, "global"이면 기존 data/ 하위에 저장합니다.
    """
    def __init__(self, user_id: str = "global", registry: StorageRegistry = None, team_name: str = None):
        self.registry = registry or StorageRegistry()
        self.user_id = user_id
        # 프로필 키: 팀 소속이면 team_name, 미소속이면 user_id
        self.profile_key = team_name if team_name else user_id

        if user_id == "global":
            # 하위 호환: 기존 동작 유지
            self.raw_path       = os.path.join(self.registry.base, "raw")
            self.processed_path = os.path.join(self.registry.base, "processed")
            self.profile_path   = os.path.join(self.registry.base, "profile.json")
            self.session_path   = os.path.join(self.registry.base, "session.json")
            os.makedirs(self.raw_path, exist_ok=True)
            os.makedirs(self.processed_path, exist_ok=True)
        else:
            self.registry.ensure_spaces(user_id)
            self.raw_path       = self.registry.user_space(user_id, "raw")
            self.processed_path = self.registry.user_space(user_id, "processed")
            self.profile_path   = os.path.join(self.registry.user_base(user_id), "profile.json")
            self.session_path   = os.path.join(self.registry.user_base(user_id), "session.json")

    def save_document(self, doc: RFPDocument) -> str:
        """
        RFPDocument 메타데이터와 콘텐츠를 SQLite에 저장합니다.
        원본 파일 사본은 raw 디렉토리에 유지합니다.
        """
        # 원본 파일 사본 저장 (없는 경우)
        raw_copy_path = os.path.join(self.raw_path, doc.filename)
        if not os.path.exists(raw_copy_path) and os.path.exists(doc.file_path):
            shutil.copy2(doc.file_path, raw_copy_path)

        upload_date = doc.upload_date.isoformat() if doc.upload_date else None
        crud.upsert_document(
            doc_hash=doc.doc_hash,
            user_id=self.user_id,
            filename=doc.filename,
            file_path=doc.file_path,
            status=doc.status if hasattr(doc, "status") else "READY",
            upload_date=upload_date,
            content=doc.model_dump(mode="json"),
        )
        return f"sqlite:documents/{self.user_id}/{doc.doc_hash}"

    def load_document(self, doc_hash: str) -> Optional[RFPDocument]:
        """SQLite에서 RFPDocument를 로드합니다."""
        data = crud.get_document(doc_hash, self.user_id)
        if data is None:
            return None
        return RFPDocument(**data["content"])

    def list_documents(self) -> List[Dict[str, Any]]:
        """저장된 문서의 메타데이터 목록을 반환합니다."""
        return crud.list_documents(self.user_id)

    def save_extraction_result(self, doc_hash: str, result: Dict[str, Any]) -> str:
        """추출 결과(Matrix)를 SQLite에 저장합니다."""
        data_to_save = result
        if hasattr(result, "model_dump"):
            data_to_save = result.model_dump(mode="json")
        crud.upsert_extraction(doc_hash, self.user_id, data_to_save)
        return f"sqlite:extraction_results/{self.user_id}/{doc_hash}"

    def load_extraction_result(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """저장된 추출 결과를 로드합니다."""
        return crud.get_extraction(doc_hash, self.user_id)

    def save_profile(self, profile: Any) -> str:
        """회사 프로필을 SQLite에 저장합니다. 팀 소속이면 팀 공유 프로필로 저장됩니다."""
        data_to_save = profile
        if hasattr(profile, "model_dump"):
            data_to_save = profile.model_dump(mode="json")
        crud.upsert_profile(self.profile_key, data_to_save)
        return f"sqlite:profiles/{self.profile_key}"

    def load_profile(self) -> Optional[Dict[str, Any]]:
        """저장된 회사 프로필을 로드합니다. 팀 소속이면 팀 공유 프로필을 반환합니다."""
        return crud.get_profile(self.profile_key)

    def save_session_state(self, state_dict: Dict[str, Any]):
        """현재 세션 상태를 SQLite에 저장합니다."""
        crud.upsert_session(self.user_id, state_dict)

    def load_session_state(self) -> Optional[Dict[str, Any]]:
        """저장된 세션 상태를 로드합니다."""
        return crud.get_session(self.user_id)


class VectorStoreManager:
    """ChromaDB 수집 및 검색을 관리합니다."""

    def __init__(self, user_id: str = "global", registry: StorageRegistry = None):
        self.registry = registry or StorageRegistry()

        if user_id == "global":
            actual_dir = os.path.join(self.registry.base, "vectordb")
        else:
            actual_dir = self.registry.user_space(user_id, "vectordb")

        collection_name = f"bidflow_rfp_{user_id}"
        self.persist_directory = actual_dir
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )

    def ingest_document(self, doc: RFPDocument):
        """파싱된 청크를 LangChain 문서로 변환하고 Chroma에 추가합니다."""
        lc_docs = []
        for chunk in doc.chunks:
            metadata = {
                "doc_hash": doc.doc_hash,
                "filename": doc.filename,
                "page_no": chunk.page_no,
                "chunk_id": chunk.chunk_id,
                "type": "text"
            }
            metadata.update(chunk.metadata)

            flat_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (list, dict)):
                    flat_metadata[k] = str(v)
                else:
                    flat_metadata[k] = v

            lc_doc = LancChainDocument(
                page_content=chunk.text,
                metadata=flat_metadata
            )
            lc_docs.append(lc_doc)

        if lc_docs:
            self.vector_db.add_documents(lc_docs)

    def get_retriever(self, search_kwargs: dict = None):
        kwargs = search_kwargs or {"k": 5}
        return self.vector_db.as_retriever(search_kwargs=kwargs)

    def clear(self):
        """벡터 DB 내용을 초기화합니다."""
        try:
            self.vector_db.delete_collection()
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=f"bidflow_rfp_{getattr(self, '_user_id', 'global')}"
            )
            print("Message: Vector DB cleared successfully.")
        except Exception as e:
            print(f"Warning: Failed to clear Vector DB: {e}")
