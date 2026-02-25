from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    message: str

class AnalysisRequest(BaseModel):
    doc_id: str
    profile_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    task_id: str
    status: str