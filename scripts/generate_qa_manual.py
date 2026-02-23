import pandas as pd
import os
import glob
import sys
import asyncio
from typing import List
from pydantic import BaseModel, Field

# Fix for Windows asyncio loop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from bidflow.parsing.pdf_parser import PDFParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

GOLDEN_PATH = "data/experiments/golden_testset.csv"
DATA_DIR = "data/raw/files"

class QAPair(BaseModel):
    question: str = Field(description="The question based on the text")
    ground_truth: str = Field(description="The precise answer from the text")
    category: str = Field(description="Category of the question (e.g., technical, budget, schedule, general)")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard")

class QASet(BaseModel):
    qa_pairs: List[QAPair]

def generate_manual():
    try:
        # 1. Load PDF
        pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
        if not pdf_files:
            print("❌ No PDF found.")
            return
        
        sample_pdf = pdf_files[0]
        print(f"📄 Processing: {sample_pdf}")

        parser = PDFParser()
        chunks = parser.parse(sample_pdf, chunk_size=2000, chunk_overlap=0) # Larger chunks for context
        # Use first 15 chunks (enough for 24 questions)
        selected_chunks = chunks[:15]
        
        combined_text = "\n\n".join([c.text for c in selected_chunks])
        
        # 2. Setup LLM
        llm = ChatOpenAI(model="gpt-5-mini", temperature=1)
        parser = PydanticOutputParser(pydantic_object=QASet)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating evaluation datasets for RAG systems from RFP documents."),
            ("user", "Analyze the following RFP document text and generate exactly 24 Question-Answer pairs.\n\n"
                     "The questions should cover various aspects like budget, schedule, technical requirements, and qualifications.\n"
                     "Format the output as JSON matching the schema.\n\n"
                     "Text:\n{text}\n\n"
                     "{format_instructions}")
        ])
        
        chain = prompt | llm | parser
        
        print("🚀 Generating 24+ QA pairs via direct LLM call...")
        result = chain.invoke({
            "text": combined_text[:30000], # Limit context window if needed, but 15 chunks * 2000 chars ~ 30k chars is fine
            "format_instructions": parser.get_format_instructions()
        })
        
        # 3. Append
        new_data = []
        for item in result.qa_pairs:
            new_data.append({
                "question": item.question,
                "ground_truth": item.ground_truth,
                "category": item.category,
                "difficulty": item.difficulty,
                "source_page": ""
            })
            
        new_df = pd.DataFrame(new_data)
        
        existing_df = pd.DataFrame()
        if os.path.exists(GOLDEN_PATH):
            existing_df = pd.read_csv(GOLDEN_PATH)
            
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_csv(GOLDEN_PATH, index=False, encoding="utf-8-sig")
        
        print(f"✅ Successfully expanded testset. Total: {len(final_df)}")
        print(final_df.tail())

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_manual()
