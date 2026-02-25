import os
import json
import shutil
from typing import List, Optional, Dict, Any, Tuple
from bidflow.domain.models import RFPDocument
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LancChainDocument


class StorageRegistry:
    """config 기반 스토리지 공간 경로 계산기."""

    def __init__(self, config=None):
        if config is None:
            from bidflow.core.config import get_config

            config = get_config("dev")

        storage = config.storage if config and getattr(config, "storage", None) else None
        self.base = (storage.base if storage and getattr(storage, "base", None) else None) or "data"
        self.accounts_dir = (storage.accounts_dir if storage and getattr(storage, "accounts_dir", None) else None) or "accounts"

        if storage and getattr(storage, "user_spaces", None):
            self._user_spaces = [s["name"] for s in storage.user_spaces]
        else:
            self._user_spaces = ["raw", "processed", "vectordb"]

        if storage and getattr(storage, "shared_spaces", None):
            self._shared_spaces = [s["name"] for s in storage.shared_spaces]
        else:
            self._shared_spaces = ["raw", "knowledge", "templates"]

        os.makedirs(self.base, exist_ok=True)

    def user_base(self, user_id: str) -> str:
        return os.path.join(self.base, self.accounts_dir, user_id)

    def user_space(self, user_id: str, space: str) -> str:
        if space not in self._user_spaces:
            raise ValueError(f"알 수 없는 user space: {space}. 가능: {self._user_spaces}")
        return os.path.join(self.user_base(user_id), space)

    def shared_space(self, space: str) -> str:
        if space not in self._shared_spaces:
            raise ValueError(f"알 수 없는 shared space: {space}. 가능: {self._shared_spaces}")
        return os.path.join(self.base, "shared", space)

    def team_space(self, team_name: str, space: str) -> str:
        return os.path.join(self.base, "shared", "teams", team_name, space)

    def ensure_spaces(self, user_id: str):
        for space in self._user_spaces:
            os.makedirs(self.user_space(user_id, space), exist_ok=True)
        for space in self._shared_spaces:
            os.makedirs(self.shared_space(space), exist_ok=True)

    def ensure_team_spaces(self, team_name: str):
        os.makedirs(self.team_space(team_name, "comments"), exist_ok=True)


class DocumentStore:
    """
    파싱된 RFPDocument(JSON)와 원본 파일을 로컬 저장소에서 관리합니다.
    - user_id를 주면 data/accounts/{user_id}/ 하위 저장소를 사용합니다.
    - team_name을 주면 프로필은 팀 공유 키(team_name) 기준으로 저장/조회합니다.
    """

    def __init__(
        self,
        user_id: str = "global",
        registry: StorageRegistry = None,
        team_name: str = None,
        base_path: str = None,
    ):
        self.registry = registry or StorageRegistry()
        if base_path and base_path != self.registry.base:
            self.registry.base = base_path
            os.makedirs(self.registry.base, exist_ok=True)

        self.user_id = user_id or "global"
        self.team_name = team_name or None

        self.base_path = self.registry.base
        self.legacy_raw_path = os.path.join(self.base_path, "raw")
        self.legacy_processed_path = os.path.join(self.base_path, "processed")
        os.makedirs(self.legacy_raw_path, exist_ok=True)
        os.makedirs(self.legacy_processed_path, exist_ok=True)

        if self.user_id == "global":
            self.raw_path = self.legacy_raw_path
            self.processed_path = self.legacy_processed_path
        else:
            self.registry.ensure_spaces(self.user_id)
            self.raw_path = self.registry.user_space(self.user_id, "raw")
            self.processed_path = self.registry.user_space(self.user_id, "processed")

        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)

    def _crud(self):
        """SQLite crud 모듈을 안전하게 가져옵니다. 실패 시 None."""
        try:
            from bidflow.db.database import init_db

            init_db()
            from bidflow.db import crud

            return crud
        except Exception:
            return None

    def _resolve_user(self, tenant_id: Optional[str]) -> str:
        uid = tenant_id if tenant_id is not None else self.user_id
        return uid or "global"

    def _paths_for_user(self, uid: str) -> Tuple[str, str]:
        if uid == "global":
            return self.legacy_raw_path, self.legacy_processed_path
        self.registry.ensure_spaces(uid)
        raw_path = self.registry.user_space(uid, "raw")
        processed_path = self.registry.user_space(uid, "processed")
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)
        return raw_path, processed_path

    def _profile_owner_key(self, uid: str) -> str:
        return self.team_name if self.team_name else uid

    def _profile_path(self, owner_key: str) -> str:
        if self.team_name and owner_key == self.team_name:
            base = self.registry.team_space(self.team_name, "profile")
            os.makedirs(base, exist_ok=True)
            return os.path.join(base, "profile.json")

        if owner_key == "global":
            return os.path.join(self.base_path, "profile.json")

        user_base = self.registry.user_base(owner_key)
        os.makedirs(user_base, exist_ok=True)
        return os.path.join(user_base, "profile.json")

    def _session_path(self, uid: str) -> str:
        if uid == "global":
            return os.path.join(self.base_path, "session.json")
        user_base = self.registry.user_base(uid)
        os.makedirs(user_base, exist_ok=True)
        return os.path.join(user_base, "session.json")

    def save_document(self, doc: RFPDocument, tenant_id: Optional[str] = None) -> str:
        uid = self._resolve_user(tenant_id)
        raw_path, processed_path = self._paths_for_user(uid)

        raw_copy_path = os.path.join(raw_path, doc.filename)
        if not os.path.exists(raw_copy_path) and os.path.exists(doc.file_path):
            shutil.copy2(doc.file_path, raw_copy_path)

        file_name = f"{doc.doc_hash}.json"
        save_path = os.path.join(processed_path, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        crud = self._crud()
        if crud:
            upload_date = doc.upload_date.isoformat() if doc.upload_date else None
            crud.upsert_document(
                doc_hash=doc.doc_hash,
                user_id=uid,
                filename=doc.filename,
                file_path=doc.file_path,
                status=getattr(doc, "status", "READY"),
                upload_date=upload_date,
                content=doc.model_dump(mode="json"),
            )

        return save_path

    def load_document(self, doc_hash: str, tenant_id: Optional[str] = None) -> Optional[RFPDocument]:
        uid = self._resolve_user(tenant_id)
        _, processed_path = self._paths_for_user(uid)

        file_name = f"{doc_hash}.json"
        load_path = os.path.join(processed_path, file_name)
        if not os.path.exists(load_path) and uid == "global":
            load_path = os.path.join(self.legacy_processed_path, file_name)

        if os.path.exists(load_path):
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RFPDocument(**data)

        crud = self._crud()
        if crud:
            row = crud.get_document(doc_hash, uid)
            if row and "content" in row:
                return RFPDocument(**row["content"])
        return None

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        uid = self._resolve_user(tenant_id)
        _, processed_path = self._paths_for_user(uid)

        results: List[Dict[str, Any]] = []
        target_path = processed_path
        if uid == "global" and not os.listdir(processed_path):
            target_path = self.legacy_processed_path

        if os.path.exists(target_path):
            for file_name in os.listdir(target_path):
                if not file_name.endswith(".json") or file_name.endswith("_result.json"):
                    continue
                doc_hash = file_name.replace(".json", "")
                doc = self.load_document(doc_hash, tenant_id=uid)
                if doc:
                    results.append(
                        {
                            "doc_hash": doc.doc_hash,
                            "filename": doc.filename,
                            "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                        }
                    )

        if results:
            return sorted(results, key=lambda d: d.get("upload_date") or "", reverse=True)

        crud = self._crud()
        if crud:
            return crud.list_documents(uid)
        return []

    def save_extraction_result(self, doc_hash: str, result: Dict[str, Any], tenant_id: Optional[str] = None) -> str:
        uid = self._resolve_user(tenant_id)
        _, processed_path = self._paths_for_user(uid)

        data_to_save = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        file_name = f"{doc_hash}_result.json"
        save_path = os.path.join(processed_path, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        crud = self._crud()
        if crud:
            crud.upsert_extraction(doc_hash, uid, data_to_save)

        return save_path

    def load_extraction_result(self, doc_hash: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        uid = self._resolve_user(tenant_id)
        _, processed_path = self._paths_for_user(uid)

        file_name = f"{doc_hash}_result.json"
        load_path = os.path.join(processed_path, file_name)
        if not os.path.exists(load_path) and uid == "global":
            load_path = os.path.join(self.legacy_processed_path, file_name)

        if os.path.exists(load_path):
            with open(load_path, "r", encoding="utf-8") as f:
                return json.load(f)

        crud = self._crud()
        if crud:
            return crud.get_extraction(doc_hash, uid)
        return None

    def save_profile(self, profile: Any, tenant_id: Optional[str] = None) -> str:
        uid = self._resolve_user(tenant_id)
        owner_key = self._profile_owner_key(uid) if tenant_id is None else tenant_id
        data_to_save = profile.model_dump(mode="json") if hasattr(profile, "model_dump") else profile

        crud = self._crud()
        if crud:
            crud.upsert_profile(owner_key, data_to_save)

        save_path = self._profile_path(owner_key)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        if owner_key == "global":
            legacy_path = os.path.join(self.base_path, "profile.json")
            with open(legacy_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        return save_path

    def load_profile(self, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        uid = self._resolve_user(tenant_id)
        owner_key = self._profile_owner_key(uid) if tenant_id is None else tenant_id

        crud = self._crud()
        if crud:
            data = crud.get_profile(owner_key)
            if data is not None:
                return data

        load_path = self._profile_path(owner_key)
        if not os.path.exists(load_path) and owner_key == "global":
            load_path = os.path.join(self.base_path, "profile.json")
        if not os.path.exists(load_path):
            return None
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_session_state(self, state_dict: Dict[str, Any], tenant_id: Optional[str] = None):
        uid = self._resolve_user(tenant_id)

        crud = self._crud()
        if crud:
            crud.upsert_session(uid, state_dict)

        save_path = self._session_path(uid)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

    def load_session_state(self, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        uid = self._resolve_user(tenant_id)

        crud = self._crud()
        if crud:
            data = crud.get_session(uid)
            if data is not None:
                return data

        load_path = self._session_path(uid)
        if not os.path.exists(load_path):
            return None
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_tenant_config(self, config: Dict[str, Any], tenant_id: str = "default") -> str:
        uid = self._resolve_user(tenant_id)
        if uid == "global":
            tenant_root = self.base_path
        else:
            tenant_root = self.registry.user_base(uid)
        os.makedirs(tenant_root, exist_ok=True)
        save_path = os.path.join(tenant_root, "config.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return save_path

    def load_tenant_config(self, tenant_id: str = "default") -> Dict[str, Any]:
        uid = self._resolve_user(tenant_id)
        tenant_root = self.base_path if uid == "global" else self.registry.user_base(uid)
        load_path = os.path.join(tenant_root, "config.json")
        if not os.path.exists(load_path):
            return {}
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def purge_tenant_data(self, tenant_id: str) -> bool:
        uid = self._resolve_user(tenant_id)
        ok = True

        # 파일 저장소 삭제
        if uid == "global":
            for path in [self.legacy_raw_path, self.legacy_processed_path]:
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        os.makedirs(path, exist_ok=True)
                    except Exception as e:
                        print(f"Error purging legacy path {path}: {e}")
                        ok = False
        else:
            user_root = self.registry.user_base(uid)
            if os.path.exists(user_root):
                try:
                    shutil.rmtree(user_root)
                except Exception as e:
                    print(f"Error purging tenant data for {uid}: {e}")
                    ok = False

        # SQLite 데이터 삭제
        crud = self._crud()
        if crud:
            try:
                # 문서/결과/세션/개인프로필 삭제
                for doc in crud.list_documents(uid):
                    crud.delete_document(doc["doc_hash"], uid)
                    crud.delete_extraction(doc["doc_hash"], uid)
                crud.delete_user(uid)
            except Exception:
                # 사용자 계정 삭제는 선택적이므로 실패해도 purge 자체는 진행
                pass

        return ok


class VectorStoreManager:
    """
    ChromaDB 수집 및 검색을 관리합니다.
    - user_id를 주면 user별 벡터 저장소(data/accounts/{user_id}/vectordb)와 컬렉션을 사용합니다.
    - 기존 호출(collection_name/persist_directory 지정)도 그대로 지원합니다.
    """

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        user_id: str = "global",
        registry: StorageRegistry = None,
    ):
        self.registry = registry or StorageRegistry()
        self.user_id = user_id or "global"

        if persist_directory:
            self.persist_directory = persist_directory
        elif self.user_id == "global":
            self.persist_directory = os.path.join(self.registry.base, "vectordb")
        else:
            self.registry.ensure_spaces(self.user_id)
            self.persist_directory = self.registry.user_space(self.user_id, "vectordb")

        if collection_name:
            self.collection_name = collection_name
        elif self.user_id == "global":
            self.collection_name = "bidflow_rfp"
        else:
            self.collection_name = f"bidflow_rfp_{self.user_id}"

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

    def ingest_document(
        self,
        doc: RFPDocument,
        tenant_id: Optional[str] = None,
        user_id: str = "system",
        group_id: str = "general",
        access_level: int = 1,
    ):
        """
        파싱된 청크를 LangChain 문서로 변환하고 Chroma에 추가합니다.
        ACL 메타데이터(tenant_id, user_id, group_id, access_level)를 포함합니다.
        """
        effective_tenant = tenant_id or (self.user_id if self.user_id != "global" else "default")
        effective_user = user_id if user_id != "system" else effective_tenant

        lc_docs = []
        for chunk in doc.chunks:
            metadata = {
                "doc_hash": doc.doc_hash,
                "filename": doc.filename,
                "page_no": chunk.page_no,
                "chunk_id": chunk.chunk_id,
                "type": "text",
                "tenant_id": effective_tenant,
                "user_id": effective_user,
                "group_id": group_id,
                "access_level": access_level,
            }
            metadata.update(chunk.metadata)

            flat_metadata = {}
            for k, v in metadata.items():
                flat_metadata[k] = str(v) if isinstance(v, (list, dict)) else v

            lc_doc = LancChainDocument(page_content=chunk.text, metadata=flat_metadata)
            lc_docs.append(lc_doc)

        if lc_docs:
            self.vector_db.add_documents(lc_docs)

    def get_retriever(self, search_kwargs: dict = None):
        kwargs = search_kwargs or {"k": 5}
        return self.vector_db.as_retriever(search_kwargs=kwargs)

    def delete_tenant_data(self, tenant_id: str):
        try:
            self.vector_db.delete(where={"tenant_id": tenant_id})
            print(f"Deleted vector data for tenant: {tenant_id}")
        except Exception as e:
            print(f"Error deleting vector data for tenant {tenant_id}: {e}")

    def clear(self):
        """
        벡터 DB 내용을 초기화합니다. (실험용)
        Collection을 삭제하고 재생성합니다.
        """
        try:
            self.vector_db.delete_collection()
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            print("Message: Vector DB cleared successfully.")
        except Exception as e:
            print(f"Warning: Failed to clear Vector DB: {e}")
