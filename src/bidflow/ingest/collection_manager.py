"""다문서 처리용 ChromaDB 컬렉션 생명주기 관리.

TTL 기반 자동 정리 (last_accessed 기준), 컬렉션 등록/조회/정리.
레지스트리는 JSON 파일로 관리 (간단, DB 불필요).
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

import chromadb


class CollectionManager:
    """다문서 처리용 ChromaDB 컬렉션 생명주기 관리."""

    DEFAULT_TTL_DAYS = 7

    def __init__(self, persist_directory: str = "data/vectordb"):
        self.persist_directory = persist_directory
        self.registry_path = Path(persist_directory) / "collection_registry.json"
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def register(self, collection_name: str, doc_hash: str, doc_name: str):
        """컬렉션 등록 (생성 시각 기록)."""
        now = datetime.now().isoformat()
        self.registry[collection_name] = {
            "doc_hash": doc_hash,
            "doc_name": doc_name,
            "created_at": now,
            "last_accessed": now,
        }
        self._save_registry()

    def find_by_hash(self, doc_hash: str) -> Optional[Dict]:
        """doc_hash로 기존 컬렉션 검색. 있으면 {collection_name, ...} 반환."""
        for name, info in self.registry.items():
            if info.get("doc_hash") == doc_hash:
                return {"collection_name": name, **info}
        return None

    def touch(self, collection_name: str):
        """접근 시각 갱신 -> TTL 연장."""
        if collection_name in self.registry:
            self.registry[collection_name]["last_accessed"] = datetime.now().isoformat()
            self._save_registry()

    def cleanup_expired(self, ttl_days: int = None) -> List[str]:
        """TTL 초과 컬렉션 삭제 (last_accessed 기준)."""
        ttl = ttl_days or self.DEFAULT_TTL_DAYS
        cutoff = datetime.now() - timedelta(days=ttl)
        expired = []

        for name, info in list(self.registry.items()):
            try:
                last = datetime.fromisoformat(info["last_accessed"])
            except (ValueError, KeyError):
                last = datetime.fromisoformat(info.get("created_at", "2000-01-01"))
            if last < cutoff:
                expired.append(name)

        for name in expired:
            self._delete_collection(name)
            del self.registry[name]

        if expired:
            self._save_registry()
            print(f"[CollectionManager] Cleaned up {len(expired)} expired collections")

        return expired

    def list_collections(self) -> Dict[str, Dict]:
        """등록된 컬렉션 목록 반환."""
        return dict(self.registry)

    def _delete_collection(self, collection_name: str):
        """ChromaDB에서 컬렉션 삭제."""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(collection_name)
            print(f"[CollectionManager] Deleted collection: {collection_name}")
        except Exception as e:
            print(f"[CollectionManager] Failed to delete {collection_name}: {e}")

    def _load_registry(self) -> Dict[str, Dict]:
        """레지스트리 JSON 로드."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_registry(self):
        """레지스트리 JSON 저장."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
