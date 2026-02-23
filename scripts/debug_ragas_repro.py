
import os
import sys
import pandas as pd
from langchain_core.documents import Document
from dotenv import load_dotenv

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bidflow.eval.ragas_runner import RagasRunner

def main():
    load_dotenv()
    print("Starting Ragas reproduction script...")

    # Mock Documents - More content to ensure node filtering passes
    docs = [
        Document(
            page_content="""
            Reference checking is a vital component of the recruitment process, serving as a tool to validate the information provided by candidates. It involves contacting previous employers to gain insights into a candidate's past performance, reliability, and professional behavior. This process helps mitigating the risk of bad hires, which can be costly for organizations.
            When conducting checks, it is advisable to ask open-ended questions. For example, "Can you describe a challenging project the candidate managed?" is more revealing than "Was the candidate a good manager?".
            Legal considerations are paramount; questions must comply with local labor laws to avoid discrimination.
            """,
            metadata={"filename": "hr_guide.pdf", "chunk_id": 1}
        ),
        Document(
            page_content="""
            BidFlow utilizes advanced Natural Language Processing (NLP) techniques to automate the extraction of critical data points from Request for Proposal (RFP) documents. The system aims to reduce the time spent on manual review by 80%.
            Key features include:
            1. Automated Extraction: Pulls project name, budget, deadlines, and requirements.
            2. Validation: Checks extracted data against predefined constraints.
            3. summarization: Generates executive summaries for quick decision making.
            The architecture is built on a microservices pattern, ensuring scalability and maintainability.
            """,
            metadata={"filename": "bidflow_intro.pdf", "chunk_id": 3}
        ),
         Document(
            page_content="""
            The 'Orchestration Layer' in Deal Maker serves as the central nervous system. It coordinates the flow of data between the User Interface and the Execution Agents.
            When a user submits a request, the Orchestrator breaks it down into sub-tasks.
            For instance, if the user asks to "Analyze this contract and draft a response", the Orchestrator first calls the 'Analyzer Agent' to parse the document, then the 'Drafter Agent' to generate the text.
            Finally, it aggregates the results and presents them to the user.
            """,
            metadata={"filename": "dealmaker_arch.pdf", "chunk_id": 4}
        )
    ]

    runner = RagasRunner()
    
    print("Generating Testset...")
    try:
        # Reduced test_size to 2 to minimize API calls, but documents should be rich enough
        testset = runner.generate_testset(docs, test_size=2)
        print("Testset Generated Successfully!")
        print(testset.head())
    except Exception as e:
        print(f"Error during generate_testset: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nRunning Evaluation...")
    try:
        results = runner.run_eval(testset)
        print("Evaluation Completed Successfully!")
        print(results.head())
    except Exception as e:
        print(f"Error during run_eval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
