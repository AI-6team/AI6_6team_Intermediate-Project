import unittest
import io
import os
import shutil
import warnings
from unittest.mock import MagicMock, patch
from bidflow.ingest.loader import RFPLoader
from bidflow.ingest.storage import DocumentStore
from bidflow.retrieval.hybrid_search import HybridRetriever
from bidflow.domain.models import ParsedChunk

# Mock Embeddings to avoid OpenAI API calls & costs
class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 1536 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 1536

# Suppress SWIG warning from ChromaDB/hnswlib
warnings.filterwarnings("ignore", message=".*swigvarlink.*")

class TestIsolationAndSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patch OpenAIEmbeddings globally for this test class
        cls.patcher = patch("bidflow.ingest.storage.OpenAIEmbeddings", side_effect=lambda **kwargs: FakeEmbeddings())
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        self.test_tenant = "test_tenant_verify"
        self.test_user = "user_verify"
        self.test_group = "group_verify"
        
        # Initialize Loader
        self.loader = RFPLoader()
        
        # Mock Parsers (실제 파싱 없이 더미 청크 반환)
        self.loader.pdf_parser.parse = MagicMock(return_value=[
            ParsedChunk(chunk_id="c1", text="This is a secret document for Tenant Verify.", page_no=1)
        ])
        self.loader.pdf_parser.extract_tables = MagicMock(return_value=[])
        
        # Mock InputRail (보안 검사 통과 처리)
        self.loader.input_rail.check = MagicMock(return_value=True)

        # Clean start
        self.loader.purge_tenant(self.test_tenant)

    def tearDown(self):
        # Clean up after test
        self.loader.purge_tenant(self.test_tenant)

    def create_mock_file(self, content: bytes, filename: str, size: int = None):
        """테스트용 가짜 파일 객체 생성"""
        file_obj = io.BytesIO(content)
        file_obj.name = filename
        file_obj.size = size if size is not None else len(content)
        return file_obj

    def test_1_security_blocklist(self):
        """[보안] 실행 파일(.exe) 업로드 차단 테스트"""
        print("\n[Test 1] Security: Blocklist (.exe)")
        file_obj = self.create_mock_file(b"MZ...", "malware.exe")
        
        with self.assertRaises(ValueError) as cm:
            self.loader.process_file(file_obj, "malware.exe", tenant_id=self.test_tenant)
        
        self.assertIn("보안 정책 위반", str(cm.exception))
        print("-> Pass: .exe blocked successfully")

    def test_2_security_magic_number(self):
        """[보안] 확장자 위변조(Magic Number) 탐지 테스트"""
        print("\n[Test 2] Security: Magic Number Mismatch")
        # 내용은 텍스트인데 확장자는 .pdf인 경우
        file_obj = self.create_mock_file(b"This is not a pdf", "fake.pdf")
        
        with self.assertRaises(ValueError) as cm:
            self.loader.process_file(file_obj, "fake.pdf", tenant_id=self.test_tenant)
        
        self.assertIn("Magic Number Mismatch", str(cm.exception))
        print("-> Pass: Fake PDF blocked successfully")

    def test_3_security_file_size(self):
        """[보안] 파일 크기 제한 테스트"""
        print("\n[Test 3] Security: File Size Limit")
        # 100MB 크기라고 속성 설정
        file_obj = self.create_mock_file(b"%PDF-1.4...", "large.pdf", size=100 * 1024 * 1024)
        
        with self.assertRaises(ValueError) as cm:
            self.loader.process_file(file_obj, "large.pdf", tenant_id=self.test_tenant, max_file_size_mb=50)
        
        self.assertIn("exceeds limit", str(cm.exception))
        print("-> Pass: Large file blocked successfully")

    def test_4_tenant_isolation_acl(self):
        """[격리] 테넌트 격리 및 ACL 필터링 테스트"""
        print("\n[Test 4] Isolation: Tenant & ACL")
        
        # 1. 문서 수집 (Tenant Verify)
        valid_pdf = self.create_mock_file(b"%PDF-1.4 dummy content", "secret.pdf")
        self.loader.process_file(
            valid_pdf, "secret.pdf", 
            tenant_id=self.test_tenant, 
            user_id=self.test_user, 
            group_id=self.test_group
        )

        # 2. 올바른 테넌트/그룹으로 검색 -> 성공해야 함
        retriever = HybridRetriever(
            tenant_id=self.test_tenant,
            user_id=self.test_user,
            group_id=self.test_group
        )
        docs = retriever.invoke("secret")
        self.assertTrue(len(docs) > 0, "Should find document in correct tenant")
        print(f"-> Search (Correct Tenant): Found {len(docs)} docs")

        # 3. 다른 테넌트로 검색 -> 실패해야 함
        retriever_wrong = HybridRetriever(tenant_id="other_tenant")
        docs_wrong = retriever_wrong.invoke("secret")
        print(f"-> Search (Wrong Tenant): Found {len(docs_wrong)} docs")
        self.assertEqual(len(docs_wrong), 0, "Should NOT find document in other tenant")

        # 4. 다른 그룹으로 검색 -> 실패해야 함
        retriever_wrong_group = HybridRetriever(
            tenant_id=self.test_tenant,
            group_id="wrong_group"
        )
        docs_wrong_group = retriever_wrong_group.invoke("secret")
        print(f"-> Search (Wrong Group): Found {len(docs_wrong_group)} docs")
        self.assertEqual(len(docs_wrong_group), 0, "Should NOT find document with wrong group")

        # 5. 데이터 삭제(Purge) 확인
        self.loader.purge_tenant(self.test_tenant)
        print("-> Pass: Tenant data purged")

    def test_5_pii_masking(self):
        """[보안] 개인정보(PII) 마스킹 테스트"""
        print("\n[Test 5] Security: PII Masking")
        
        # PII가 포함된 텍스트
        pii_text = "주민번호: 990101-1234567, 전화번호: 010-1234-5678, 이메일: user@example.com"
        
        # Mock Parser가 PII 텍스트를 반환하도록 설정
        self.loader.pdf_parser.parse = MagicMock(return_value=[
            ParsedChunk(chunk_id="c_pii", text=pii_text, page_no=1)
        ])
        
        # 파일 처리
        dummy_file = self.create_mock_file(b"%PDF-1.4 PII test", "pii_test.pdf")
        doc_hash = self.loader.process_file(dummy_file, "pii_test.pdf", tenant_id=self.test_tenant)
        
        # 저장된 문서 로드
        doc = self.loader.doc_store.load_document(doc_hash, tenant_id=self.test_tenant)
        masked_text = doc.chunks[0].text
        
        print(f"Original: {pii_text}")
        print(f"Masked:   {masked_text}")
        
        # 검증
        self.assertIn("990101-*******", masked_text)
        self.assertIn("010-1234-****", masked_text)
        self.assertIn("user@****", masked_text)
        
        self.assertNotIn("1234567", masked_text)
        self.assertNotIn("5678", masked_text)
        self.assertNotIn("example.com", masked_text)
        
        print("-> Pass: PII masked successfully")

if __name__ == "__main__":
    unittest.main()
