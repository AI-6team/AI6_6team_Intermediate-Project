import os
import json
import shutil
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from bidflow.domain.models import RFPDocument, ParsedChunk
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LancChainDocument

class DocumentStore:
    """
    파싱된 RFPDocument(JSON)와 원본 파일을 로컬 저장소에서 관리합니다.
    테넌트 격리(Tenant Isolation)를 지원합니다.
    """
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path

    def _get_paths(self, tenant_id: str):
        """테넌트별 격리된 경로 반환"""
        tenant_root = os.path.join(self.base_path, tenant_id)
        raw_path = os.path.join(tenant_root, "raw")
        processed_path = os.path.join(tenant_root, "processed")
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)
        return raw_path, processed_path

    def save_document(self, doc: RFPDocument, tenant_id: str = "default") -> str:
        """
        RFPDocument 메타데이터와 콘텐츠를 JSON으로 저장합니다.
        저장된 파일 경로를 반환합니다.
        """
        raw_path, processed_path = self._get_paths(tenant_id)
        
        file_name = f"{doc.doc_hash}.json"
        save_path = os.path.join(processed_path, file_name)
        
        # 원본 파일 사본 저장 (없는 경우)
        raw_copy_path = os.path.join(raw_path, doc.filename)
        if not os.path.exists(raw_copy_path) and os.path.exists(doc.file_path):
             shutil.copy2(doc.file_path, raw_copy_path)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            
        return save_path

    def load_document(self, doc_hash: str, tenant_id: str = "default") -> Optional[RFPDocument]:
        """
        해시를 사용하여 RFPDocument를 로드합니다.
        """
        _, processed_path = self._get_paths(tenant_id)
        file_name = f"{doc_hash}.json"
        load_path = os.path.join(processed_path, file_name)
        
        if not os.path.exists(load_path):
            return None
            
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return RFPDocument(**data)

    def list_documents(self, tenant_id: str = "default") -> List[Dict[str, Any]]:
        """
        저장된 문서의 메타데이터 목록을 반환합니다.
        """
        _, processed_path = self._get_paths(tenant_id)
        results = []
        if not os.path.exists(processed_path):
            return results
            
        files = os.listdir(processed_path)
        for f in files:
            if f.endswith(".json") and not f.endswith("_result.json"):
                doc_hash = f.replace(".json", "")
                doc = self.load_document(doc_hash, tenant_id)
                if doc:
                    results.append({
                        "doc_hash": doc.doc_hash,
                        "filename": doc.filename,
                        "upload_date": doc.upload_date.isoformat() if doc.upload_date else None
                    })
        return results

    def save_extraction_result(self, doc_hash: str, result: Dict[str, Any], tenant_id: str = "default") -> str:
        """
        추출 결과(Matrix)를 JSON으로 저장합니다.
        경로: data/processed/{doc_hash}_result.json
        """
        _, processed_path = self._get_paths(tenant_id)
        file_name = f"{doc_hash}_result.json"
        save_path = os.path.join(processed_path, file_name)
        
        # Pydantic 모델이나 dict 모두 처리 가능하도록
        # result가 dict가 아니면 model_dump 사용 (여기선 주로 dict로 옴)
        data_to_save = result
        if hasattr(result, "model_dump"):
            data_to_save = result.model_dump(mode="json")
            
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
        return save_path

    def load_extraction_result(self, doc_hash: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        저장된 추출 결과를 로드합니다.
        """
        _, processed_path = self._get_paths(tenant_id)
        file_name = f"{doc_hash}_result.json"
        load_path = os.path.join(processed_path, file_name)
        
        if not os.path.exists(load_path):
            return None
            
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_profile(self, profile: Any, tenant_id: str = "default") -> str:
        """
        회사 프로필을 JSON으로 저장합니다. (Tenant-specific)
        경로: data/{tenant_id}/profile.json
        """
        tenant_root = os.path.join(self.base_path, tenant_id)
        os.makedirs(tenant_root, exist_ok=True)
        save_path = os.path.join(tenant_root, "profile.json")
        
        data_to_save = profile
        if hasattr(profile, "model_dump"):
            data_to_save = profile.model_dump(mode="json")
            
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
        return save_path

    def load_profile(self, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        저장된 회사 프로필을 로드합니다.
        """
        tenant_root = os.path.join(self.base_path, tenant_id)
        load_path = os.path.join(tenant_root, "profile.json")
        
        if not os.path.exists(load_path):
            return None
            
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_session_state(self, state_dict: Dict[str, Any], tenant_id: str = "default"):
        """
        현재 세션 상태(마지막 문서 ID 등)를 저장합니다.
        """
        tenant_root = os.path.join(self.base_path, tenant_id)
        os.makedirs(tenant_root, exist_ok=True)
        save_path = os.path.join(tenant_root, "session.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

    def load_session_state(self, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        저장된 세션 상태를 로드합니다.
        """
        tenant_root = os.path.join(self.base_path, tenant_id)
        load_path = os.path.join(tenant_root, "session.json")
        if not os.path.exists(load_path):
            return None
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_tenant_config(self, config: Dict[str, Any], tenant_id: str = "default") -> str:
        """
        테넌트별 설정(검색 결과 없음 메시지 등)을 저장합니다.
        """
        tenant_root = os.path.join(self.base_path, tenant_id)
        os.makedirs(tenant_root, exist_ok=True)
        save_path = os.path.join(tenant_root, "config.json")
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        return save_path

    def load_tenant_config(self, tenant_id: str = "default") -> Dict[str, Any]:
        """
        테넌트별 설정을 로드합니다.
        """
        tenant_root = os.path.join(self.base_path, tenant_id)
        load_path = os.path.join(tenant_root, "config.json")
        
        if not os.path.exists(load_path):
            return {}
            
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def purge_tenant_data(self, tenant_id: str) -> bool:
        """
        특정 테넌트의 모든 파일 데이터를 영구 삭제합니다.
        """
        tenant_root = os.path.join(self.base_path, tenant_id)
        if os.path.exists(tenant_root):
            try:
                shutil.rmtree(tenant_root)
                print(f"Purged file storage for tenant: {tenant_id}")
                return True
            except Exception as e:
                print(f"Error purging tenant data for {tenant_id}: {e}")
                return False
        return False


class VectorStoreManager:
    """
    ChromaDB 수집 및 검색을 관리합니다.
    """
    def __init__(self, persist_directory: str = "data/vectordb"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="bidflow_rfp"
        )

    def ingest_document(self, doc: RFPDocument, tenant_id: str = "default", user_id: str = "system", group_id: str = "general", access_level: int = 1):
        """
        파싱된 청크를 LangChain 문서로 변환하고 Chroma에 추가합니다.
        ACL 메타데이터(tenant_id, user_id, group_id, access_level)를 포함합니다.
        """
        lc_docs = []
        for chunk in doc.chunks:
            # 메타데이터 (평면 딕셔너리)
            metadata = {
                "doc_hash": doc.doc_hash,
                "filename": doc.filename,
                "page_no": chunk.page_no,
                "chunk_id": chunk.chunk_id,
                "type": "text",
                "tenant_id": tenant_id,
                "user_id": user_id,
                "group_id": group_id,
                "access_level": access_level
            }
            # 청크 추가 메타데이터 (bbox 등) 포함
            metadata.update(chunk.metadata)
            
            # ChromaDB는 메타데이터 값으로 str, int, float, bool만 허용함. List/Dict는 불가.
            # bbox 등 복잡한 데이터는 문자열로 변환하거나 제거해야 함.
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

    def delete_tenant_data(self, tenant_id: str):
        """
        특정 테넌트의 벡터 데이터를 삭제합니다.
        """
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
            # Re-initialize to recreate collection
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="bidflow_rfp"
            )
            print("Message: Vector DB cleared successfully.")
        except Exception as e:
            print(f"Warning: Failed to clear Vector DB: {e}")
