import sys
import os
import shutil

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from bidflow.ingest.loader import RFPLoader
from bidflow.retrieval.hybrid_search import HybridRetriever
from langchain_core.documents import Document

def test_loader_chunking():
    print("\n[Test 1] RFPLoader Chunking Parameter...")
    loader = RFPLoader()
    
    # Create dummy PDF (requires reportlab or just mocking pdfplumber which is hard)
    # Instead, we will rely on checking the method signature/execution if possible, 
    # but without a real PDF it's hard to integ test. 
    # Let's inspect the process_file signature at run time.
    import inspect
    sig = inspect.signature(loader.process_file)
    print(f"RFPLoader.process_file signature: {sig}")
    
    if "chunk_size" in sig.parameters and "chunk_overlap" in sig.parameters:
        print("✅ RFPLoader accepts chunking parameters.")
    else:
        print("❌ RFPLoader missing parameters!")

def test_hybrid_retriever_interface():
    print("\n[Test 2] HybridRetriever Weighted RRF Interface...")
    
    # Mocking VectorStoreManager is complex, let's just check the class initialization
    try:
        retriever = HybridRetriever(
            vector_store_manager=None, # It will create new one
            top_k=5, 
            weights=[0.7, 0.3]
        )
        print(f"Initialized HybridRetriever with weights: {retriever.weights}")
        print("✅ HybridRetriever initialization successful.")
        
        # Test _rrf_merge logic with dummy docs
        doc1 = Document(page_content="A", metadata={"id": 1})
        doc2 = Document(page_content="B", metadata={"id": 2})
        doc3 = Document(page_content="A", metadata={"id": 1}) # Duplicate content
        
        # list1 (BM25) -> doc1(rank0), doc2(rank1) -> Score: 0.7*(1/60) + 0, 0.7*(1/61)
        # list2 (Vec)  -> doc2(rank0), doc3(rank1) -> Score: 0.3*(1/60), 0.3*(1/61)
        
        # docA score: 0.7(1/60) + 0.3(1/61) (since doc3 is A)
        # docB score: 0.7(1/61) + 0.3(1/60)
        
        merged = retriever._rrf_merge([doc1, doc2], [doc2, doc3])
        print(f"Merged Result: {[d.page_content for d in merged]}")
        print("✅ RRF Merge logic executed without error.")
        
    except Exception as e:
        print(f"❌ Failed to initialize/run HybridRetriever: {e}")

if __name__ == "__main__":
    test_loader_chunking()
    test_hybrid_retriever_interface()
