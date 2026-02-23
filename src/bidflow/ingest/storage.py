import os
import json
import shutil
from typing import List, Optional, Dict, Any
from bidflow.domain.models import RFPDocument, ParsedChunk
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LancChainDocument

class DocumentStore:
    """
    파싱된 RFPDocument(JSON)와 원본 파일을 로컬 저장소에서 관리합니다.
    """
    def __init__(self, base_path: str = "data"):
        self.raw_path = os.path.join(base_path, "raw")
        self.processed_path = os.path.join(base_path, "processed")
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)

    def save_document(self, doc: RFPDocument) -> str:
        """
        RFPDocument 메타데이터와 콘텐츠를 JSON으로 저장합니다.
        저장된 파일 경로를 반환합니다.
        """
        file_name = f"{doc.doc_hash}.json"
        save_path = os.path.join(self.processed_path, file_name)
        
        # 원본 파일 사본 저장 (없는 경우)
        raw_copy_path = os.path.join(self.raw_path, doc.filename)
        if not os.path.exists(raw_copy_path) and os.path.exists(doc.file_path):
             shutil.copy2(doc.file_path, raw_copy_path)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            
        return save_path

    def load_document(self, doc_hash: str) -> Optional[RFPDocument]:
        """
        해시를 사용하여 RFPDocument를 로드합니다.
        """
        file_name = f"{doc_hash}.json"
        load_path = os.path.join(self.processed_path, file_name)
        
        if not os.path.exists(load_path):
            return None
            
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return RFPDocument(**data)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        저장된 문서의 메타데이터 목록을 반환합니다.
        """
        results = []
        files = os.listdir(self.processed_path)
        for f in files:
            if f.endswith(".json"):
                doc_hash = f.replace(".json", "")
                doc = self.load_document(doc_hash)
                if doc:
                    results.append({
                        "doc_hash": doc.doc_hash,
                        "filename": doc.filename,
                        "upload_date": doc.upload_date.isoformat() if doc.upload_date else None
                    })
        return results

    def save_extraction_result(self, doc_hash: str, result: Dict[str, Any]) -> str:
        """
        추출 결과(Matrix)를 JSON으로 저장합니다.
        경로: data/processed/{doc_hash}_result.json
        """
        file_name = f"{doc_hash}_result.json"
        save_path = os.path.join(self.processed_path, file_name)
        
        # Pydantic 모델이나 dict 모두 처리 가능하도록
        # result가 dict가 아니면 model_dump 사용 (여기선 주로 dict로 옴)
        data_to_save = result
        if hasattr(result, "model_dump"):
            data_to_save = result.model_dump(mode="json")
            
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
        return save_path

    def load_extraction_result(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """
        저장된 추출 결과를 로드합니다.
        """
        file_name = f"{doc_hash}_result.json"
        load_path = os.path.join(self.processed_path, file_name)
        
        if not os.path.exists(load_path):
            return None
            
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_profile(self, profile: Any) -> str:
        """
        회사 프로필을 JSON으로 저장합니다. (Global)
        경로: data/profile.json
        """
        save_path = os.path.join("data", "profile.json")
        
        data_to_save = profile
        if hasattr(profile, "model_dump"):
            data_to_save = profile.model_dump(mode="json")
            
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
        return save_path

    def load_profile(self) -> Optional[Dict[str, Any]]:
        """
        저장된 회사 프로필을 로드합니다.
        """
        load_path = os.path.join("data", "profile.json")
        
        if not os.path.exists(load_path):
            return None
            
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_session_state(self, state_dict: Dict[str, Any]):
        """
        현재 세션 상태(마지막 문서 ID 등)를 저장합니다.
        """
        save_path = os.path.join("data", "session.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

    def load_session_state(self) -> Optional[Dict[str, Any]]:
        """
        저장된 세션 상태를 로드합니다.
        """
        load_path = os.path.join("data", "session.json")
        if not os.path.exists(load_path):
            return None
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)


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

    def ingest_document(self, doc: RFPDocument):
        """
        파싱된 청크를 LangChain 문서로 변환하고 Chroma에 추가합니다.
        """
        lc_docs = []
        for chunk in doc.chunks:
            # 메타데이터 (평면 딕셔너리)
            metadata = {
                "doc_hash": doc.doc_hash,
                "filename": doc.filename,
                "page_no": chunk.page_no,
                "chunk_id": chunk.chunk_id,
                "type": "text"
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
