from fastapi import FastAPI
from bidflow.api.routers import upload, extract, validate
import os

app = FastAPI(
    title="BidFlow API",
    description="AI 기반 RFP 분석 시스템",
    version="0.1.0"
)

app.include_router(upload.router, prefix="/api", tags=["수집 (Ingest)"])
app.include_router(extract.router, prefix="/api", tags=["추출 (Extract)"])
app.include_router(validate.router, prefix="/api", tags=["검증 (Validate)"])

@app.get("/")
def root():
    return {"message": "BidFlow API가 실행 중입니다."}

if __name__ == "__main__":
    import uvicorn
    # Allow running directly for debug
    uvicorn.run("bidflow.main:app", host="0.0.0.0", port=8000, reload=True)
