from bidflow.ingest.storage import DocumentStore
from bidflow.eval.ragas_runner import RagasRunner
from bidflow.apps.ui.auth import require_login
from langchain_core.documents import Document
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Evaluation", page_icon="📈")

user_id = require_login()

st.title("System Evaluation (RAGAS)")

from bidflow.apps.ui.session import init_app_session
init_app_session(user_id=user_id)

if "current_doc_hash" not in st.session_state:
    st.warning("먼저 'Upload' 탭에서 문서를 분석해주세요.")
    st.stop()

doc_hash = st.session_state["current_doc_hash"]
store = DocumentStore(user_id=user_id)
doc = store.load_document(doc_hash)

if not doc:
    st.error("문서를 찾을 수 없습니다.")
    st.stop()

st.write(f"Target Document: **{doc.filename}**")

if not doc.chunks:
    st.error("분석된 청크가 없습니다. 먼저 문서를 업로드해주세요.")
    st.stop()

st.markdown("""
### RAGAS Metrics
- **Faithfulness**: LLM의 답변이 주어진 문맥(근거)에 충실한지 (환각 여부)
- **Answer Relevancy**: 답변이 질문의 의도와 관련이 있는지
""")

if st.button("Run Evaluation (약 1~2분 소요)"):
    with st.spinner("1. 테스트셋 생성 중... (Synthetic Generation)"):
        lc_docs = [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in doc.chunks
        ]

        runner = RagasRunner()
        testset = runner.generate_testset(lc_docs, test_size=3)
        st.success(f"테스트셋 생성 완료 ({len(testset)} pairs)")

        display_cols = []
        for col in ["user_input", "question", "reference", "ground_truth"]:
            if col in testset.columns:
                display_cols.append(col)
        st.dataframe(testset[display_cols[:2]].head() if display_cols else testset.head())

    with st.spinner("2. 평가 수행 중... (Evaluating)"):
        results = runner.run_eval(testset)
        st.success("평가 완료!")

        st.subheader("Evaluation Scores")

        metric_cols = []
        col_mapping = {}
        if "faithfulness" in results.columns:
            metric_cols.append("faithfulness")
            col_mapping["faithfulness"] = "Faithfulness"
        if "answer_relevancy" in results.columns:
            metric_cols.append("answer_relevancy")
            col_mapping["answer_relevancy"] = "Answer Relevancy"
        if "response_relevancy" in results.columns:
            metric_cols.append("response_relevancy")
            col_mapping["response_relevancy"] = "Response Relevancy"

        if metric_cols:
            avg_scores = results[metric_cols].mean()

            cols = st.columns(len(metric_cols))
            for i, col_name in enumerate(metric_cols):
                score = avg_scores[col_name]
                display_name = col_mapping.get(col_name, col_name)
                cols[i].metric(display_name, f"{score:.2f}" if pd.notna(score) else "N/A")

            st.bar_chart(avg_scores)
        else:
            st.warning("평가 메트릭을 찾을 수 없습니다.")

        st.subheader("Detailed Results")
        st.dataframe(results)
