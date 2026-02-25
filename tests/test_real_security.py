import os
import sys
import time
import json
from io import BytesIO
from unittest.mock import MagicMock

# src 디렉토리를 모듈 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bidflow.ingest.loader import RFPLoader
from bidflow.retrieval.rag_chain import RAGChain
from bidflow.ingest.storage import DocumentStore, VectorStoreManager

def create_dummy_pdf(filename, content):
    """테스트용 더미 PDF 파일 생성 (메모리 상)"""
    # 실제 PDF 헤더를 포함해야 Magic Number 체크를 통과함
    pdf_header = b"%PDF-1.4\n"
    file_obj = BytesIO(pdf_header + content.encode('utf-8'))
    file_obj.name = filename
    file_obj.size = len(file_obj.getvalue())
    return file_obj

def test_real_pipeline():
    print("🚀 [Real Test] Starting Security Pipeline Verification...")
    
    tenant_id = "test_real_tenant"
    user_id = "real_user"
    
    # 1. 초기화 및 데이터 정리
    loader = RFPLoader()
    loader.purge_tenant(tenant_id)
    
    # 2. PII가 포함된 파일 업로드 테스트
    print("\n📂 [Step 1] Uploading file with PII...")
    
    # [설정] 테스트할 실제 PDF 파일 경로 (프로젝트 루트 기준 data 폴더 등)
    real_pdf_path = "data/raw/고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"
    
    if os.path.exists(real_pdf_path):
        print(f"📄 Found real PDF: {real_pdf_path}")
        with open(real_pdf_path, "rb") as f:
            file_content = f.read()
        file_obj = BytesIO(file_content)
        file_obj.name = os.path.basename(real_pdf_path)
        file_obj.size = len(file_content)
    else:
        print(f"⚠️ File not found: {real_pdf_path}")
        print("   Creating dummy PDF with PII for testing...")
        pii_content = "이 문서의 담당자는 홍길동이며, 주민번호는 900101-1234567 입니다. 연락처는 010-1234-5678 입니다."
        file_obj = create_dummy_pdf("pii_doc.pdf", pii_content)
    
    # 실제 파서를 사용하므로 Mocking 제거
    doc_hash = loader.process_file(file_obj, file_obj.name, tenant_id=tenant_id)
    
    # 저장된 문서 확인 (마스킹 여부)
    doc = loader.doc_store.load_document(doc_hash, tenant_id=tenant_id)
    
    print(f"\n🔍 [Debug] Inspecting {len(doc.chunks)} chunks for PII masking...")
    masked_count = 0
    
    for i, chunk in enumerate(doc.chunks):
        text = chunk.text
        # PIIFilter에서 사용하는 마스킹 치환 문자열 확인
        # 주민/여권/외국인: *******
        # 전화번호: -****- 또는 공백****공백
        # 이메일: @****
        # IP: ***.***.***.***
        if "*******" in text or "-****-" in text or "@****" in text or "***.***.***.***" in text:
            masked_count += 1
            if masked_count <= 3: # 처음 3개만 상세 출력
                print(f"   ✅ Chunk {i} (Page {chunk.page_no}): Masking detected")
                # 마스킹된 부분 주변 컨텍스트 출력
                # 간단히 첫 번째 발견된 마스킹 패턴 기준
                print(f"      Preview: ...{text[:100].replace(chr(10), ' ')}...")

    if masked_count > 0:
        print(f"   🎉 Total {masked_count} chunks contain masked PII.")
    else:
        print("   ⚠️ No PII masking patterns found in the entire document.")
        print("   ℹ️ First chunk content for inspection (First 500 chars):")
        if doc.chunks:
            print(f"   {doc.chunks[0].text[:500]}")
        else:
            print("   (No chunks found)")

    saved_text = doc.chunks[0].text if doc.chunks else ""

    # 3. RAGChain 실행 (감사 로그 테스트)
    print("\n🤖 [Step 2] Invoking RAGChain (Audit Log Test)...")
    
    # 실제 OpenAI 호출을 피하기 위해 LLM과 Retriever를 Mocking하지만, 
    # RAGChain의 로깅 로직은 그대로 실행됨.
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage
    
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content=saved_text, metadata={"filename": file_obj.name, "page_no": 1, "doc_hash": doc_hash})
    ]
    
    # RAGChain 초기화
    # 실제 환경에서는 API Key가 필요하므로, 없으면 Mocking 처리
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️ OPENAI_API_KEY not found. Mocking ChatOpenAI.")
        from unittest.mock import patch
        with patch("bidflow.retrieval.rag_chain.ChatOpenAI") as MockChat:
            mock_llm = MockChat.return_value
            mock_llm.invoke.return_value = AIMessage(content="담당자 정보는 확인되지 않습니다.")
            # pipe().invoke() 체인을 위해 mock 설정
            mock_llm.bind_tools.return_value = mock_llm
            
            rag_chain = RAGChain(retriever=mock_retriever, tenant_id=tenant_id)
            # LLM Mocking을 위해 내부 객체 교체
            rag_chain.llm = mock_llm
            
            # 실행
            rag_chain.invoke("담당자 누구야?", request_metadata={"ip": "1.2.3.4", "user": user_id})
    else:
        print("   🔑 Using real OpenAI API.")
        rag_chain = RAGChain(retriever=mock_retriever, tenant_id=tenant_id)
        rag_chain.invoke("담당자 누구야?", request_metadata={"ip": "1.2.3.4", "user": user_id})

    # 4. 로그 파일 확인
    print("\n📝 [Step 3] Verifying Security Logs...")
    log_file = "logs/audit.log"
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # 가장 최근 로그 확인
        found_audit = False
        for line in reversed(lines):
            try:
                log = json.loads(line)
                if log.get("event") == "rag_response" and log.get("tenant_id") == tenant_id:
                    print(f"   ✅ Audit Log Found: {json.dumps(log, ensure_ascii=False)}")
                    found_audit = True
                    break
            except:
                continue
        
        if not found_audit:
            print("   ❌ Audit Log NOT found.")
    else:
        print("   ❌ Log file does not exist.")

    # 정리
    loader.purge_tenant(tenant_id)
    print("\n🏁 Test Complete.")

if __name__ == "__main__":
    test_real_pipeline()
