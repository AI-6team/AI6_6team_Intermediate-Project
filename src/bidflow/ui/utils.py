import os
import requests
import streamlit as st
from typing import Optional, Dict, Any, List

# API 설정
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("ADMIN_PASSWORD", "secret")

def get_headers():
    return {"X-API-Key": API_KEY}

def health_check() -> bool:
    try:
        # API_BASE_URL에서 호스트 루트(http://localhost:8000)만 추출
        root_url = API_BASE_URL.split("/api")[0]
        response = requests.get(root_url, timeout=1)
        return response.status_code == 200
    except:
        return False

def _get_mime_type(filename: str) -> str:
    if filename.lower().endswith('.pdf'):
        return 'application/pdf'
    if filename.lower().endswith('.hwp'):
        return 'application/x-hwp'
    return 'application/octet-stream'

def upload_file(file) -> Optional[Dict[str, Any]]:
    """
    파일 업로드 API 호출
    """
    mime_type = _get_mime_type(file.name)
    files = {"file": (file.name, file, mime_type)}
    try:
        # main.py의 prefix="/api/v1/ingest" 와 ingest.py의 "/upload" 결합
        response = requests.post(f"{API_BASE_URL}/ingest/upload", files=files, headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_documents() -> List[Dict[str, Any]]:
    """
    문서 목록 조회 API 호출
    """
    try:
        # ingest 라우터 하위로 이동됨
        response = requests.get(f"{API_BASE_URL}/ingest/documents", headers=get_headers())
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching docs: {e}")
        return []

def get_document_detail(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    문서 상세 조회 API 호출
    """
    try:
        # ingest 라우터 하위로 이동됨
        response = requests.get(f"{API_BASE_URL}/ingest/documents/{doc_id}/view", headers=get_headers())
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching doc details: {e}")
        return None

def run_extraction(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    추출(Extract) API 호출
    """
    try:
        response = requests.post(f"{API_BASE_URL}/extract/{doc_id}", headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Extraction failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return None

def run_validation(matrix: Dict, profile: Dict) -> Optional[List[Dict[str, Any]]]:
    """
    검증(Validate) API 호출
    """
    try:
        payload = {
            "matrix": matrix,
            "profile": profile
        }
        response = requests.post(f"{API_BASE_URL}/validate", json=payload, headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Validation failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Validation Error: {e}")
        return None
