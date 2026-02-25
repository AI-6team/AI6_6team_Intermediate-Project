from datetime import datetime
from typing import List, Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field

# --- Core Evidence Models ---

class Evidence(BaseModel):
    source_type: Literal["text", "table"]
    page_no: int
    text_snippet: str  # 원문 스니펫 (요약본 아님) 또는 셀 텍스트
    # 텍스트 증거 필드
    chunk_id: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    # 테이블 증거 필드
    table_id: Optional[str] = None
    row_idx: Optional[int] = None
    col_idx: Optional[int] = None
    # 시각적 좌표
    coords: Optional[List[float]] = None  # [x0, y0, x1, y1] 하이라이팅용 좌표

# --- Extraction Models ---

class ExtractionSlot(BaseModel):
    key: str
    value: Union[str, int, float, List[str], None]
    status: Literal["FOUND", "NOT_FOUND", "AMBIGUOUS"]
    evidence: List[Evidence] = Field(default_factory=list)
    integrity_score: float = 1.0  # table_integrity_score와 연동

class ComplianceMatrix(BaseModel):
    doc_hash: str
    slots: Dict[str, ExtractionSlot] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

# --- Validation Models ---

class ValidationResult(BaseModel):
    slot_key: str
    decision: Literal["GREEN", "RED", "GRAY"]
    reasons: List[str]
    evidence: List[Evidence]  # 검증에 사용된 핵심 근거
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]  # HIGH: 무결성 < 0.65 또는 주장-근거 불일치
    timestamp: datetime = Field(default_factory=datetime.now)

class CompanyProfile(BaseModel):
    id: str
    name: str
    data: Dict[str, Any]  # 유연한 프로필 데이터 스키마
    updated_at: datetime = Field(default_factory=datetime.now)

# --- Ingest/Document Models ---

class ParsedChunk(BaseModel):
    chunk_id: str
    text: str
    page_no: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ParsedTable(BaseModel):
    table_id: str
    page_no: int
    caption: str = ""
    rows: List[List[str]]  # MVP용 단순 직렬화
    metadata: Dict[str, Any] = Field(default_factory=dict) # 좌표 등

class RFPDocument(BaseModel):
    id: str
    filename: str
    file_path: str
    doc_hash: str
    upload_date: datetime = Field(default_factory=datetime.now)
    chunks: List[ParsedChunk] = Field(default_factory=list)
    tables: List[ParsedTable] = Field(default_factory=list)
    status: Literal["PROCESSING", "READY", "ERROR"] = "PROCESSING"

# --- Multi-Document Signal Models ---

class DocumentSignal(BaseModel):
    doc_hash: str
    doc_name: str
    signal: Literal["GREEN", "RED", "GRAY"]
    fit_score: float = Field(ge=0.0, le=1.0)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    extraction_summary: Dict[str, Any] = Field(default_factory=dict)
    signal_reasons: List[str] = Field(default_factory=list)
    collection_name: str = ""
    processing_time_sec: float = 0.0
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    postprocess_log: Optional[List[Dict[str, Any]]] = None

class BatchAnalysisResult(BaseModel):
    results: List[DocumentSignal] = Field(default_factory=list)
    total_docs: int = 0
    green_count: int = 0
    red_count: int = 0
    gray_count: int = 0
    created_at: str = ""
    total_processing_time_sec: float = 0.0
    failed_docs: List[Dict[str, str]] = Field(default_factory=list)
