import pandas as pd
import os
import glob
import sys
import asyncio
import traceback

# Fix for Windows asyncio loop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from bidflow.parsing.pdf_parser import PDFParser
from bidflow.eval.ragas_runner import RagasRunner
from langchain_core.documents import Document

GOLDEN_PATH = "data/experiments/golden_testset.csv"
DATA_DIR = "data/raw/files"

def expand_testset():
    try:
        # 1. Find PDF
        pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
        if not pdf_files:
            print("❌ No PDF found.")
            return
        
        sample_pdf = pdf_files[0]
        print(f"📄 Processing: {sample_pdf}")

        # 2. Parse PDF
        print("🔹 Parsing PDF...")
        parser = PDFParser()
        # Ensure arguments are valid based on codebase knowledge
        # parse(file_path, chunk_size, chunk_overlap, table_strategy)
        chunks = parser.parse(sample_pdf, chunk_size=1000, chunk_overlap=100)
        chunks = chunks[:50] # [Optimization] Use first 50 chunks to speed up (sufficient for 30 questions)
        print(f"🔹 Loaded {len(chunks)} chunks.")
        
        lc_docs = [Document(page_content=c.text, metadata=c.metadata) for c in chunks]

        # 3. Check Counts
        current_count = 0
        existing_df = pd.DataFrame()
        if os.path.exists(GOLDEN_PATH):
            existing_df = pd.read_csv(GOLDEN_PATH)
            current_count = len(existing_df)
        
        target = 30
        needed = target - current_count
        
        if needed <= 0:
            print(f"✅ Target reached ({current_count}/{target}). Stopping.")
            return

        print(f"🚀 Generating {needed} questions...")
        
        # 4. Generate
        runner = RagasRunner()
        # generate_testset(docs, test_size)
        new_testset_df = runner.generate_testset(lc_docs, test_size=needed)
        
        # 5. Map & Save
        mappings = []
        for _, row in new_testset_df.iterrows():
            evo = row.get("evolution_type", "synthetic")
            diff = "hard" if "reasoning" in evo else "easy" if "simple" in evo else "medium"
            
            mappings.append({
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", ""),
                "category": evo,
                "difficulty": diff,
                # "source_page": row.get("metadata", {}).get("page", "") # Try to get page if possible
                "source_page": "" # Keep simple to avoid errors
            })
            
        new_df = pd.DataFrame(mappings)
        final_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
        
        final_df.to_csv(GOLDEN_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ Done! Total questions: {len(final_df)}")
        print(final_df.tail())

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    expand_testset()
