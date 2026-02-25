import os
import shutil
import tempfile

import streamlit as st

from bidflow.apps.ui.auth import require_login
from bidflow.apps.ui.session import init_app_session
from bidflow.extraction.pipeline import ExtractionPipeline
from bidflow.ingest.loader import RFPLoader
from bidflow.ingest.storage import DocumentStore
from bidflow.security.rails.input_rail import SecurityException

st.set_page_config(page_title="Upload RFP", page_icon="📂", layout="wide")


def _signal_badge(signal: str) -> str:
    if signal == "GREEN":
        return "🟢 GREEN"
    if signal == "RED":
        return "🔴 RED"
    if signal == "GRAY":
        return "⚪ GRAY"
    return signal


def _run_single_analysis(uploaded_file, user_id: str) -> None:
    with st.status("문서 처리 중...", expanded=True) as status:
        try:
            st.write("📂 문서를 서버에 저장하고 파싱합니다...")
            loader = RFPLoader(user_id=user_id)
            doc_hash = loader.process_file(uploaded_file, uploaded_file.name)
            st.write(f"✅ 파싱 완료 (ID: {doc_hash})")

            st.write("🧠 Compliance Matrix 추출을 시작합니다... (LLM)")
            pipeline = ExtractionPipeline(user_id=user_id)
            results = pipeline.run(doc_hash)
            st.write("✅ 추출 완료!")
        except SecurityException as e:
            status.update(label="🚨 보안 위협 탐지!", state="error", expanded=True)
            st.error(f"보안 위협이 탐지되어 처리가 차단되었습니다: {e}")
            return
        except Exception as e:
            status.update(label="오류 발생", state="error")
            st.error(f"처리 중 오류: {e}")
            return

        st.session_state["current_doc_hash"] = doc_hash
        st.session_state["extraction_results"] = results
        st.session_state["analysis_success"] = True

        store = DocumentStore(user_id=user_id)
        store.save_extraction_result(doc_hash, results)
        store.save_session_state({"current_doc_hash": doc_hash})
        st.toast("분석 결과가 저장되었습니다.", icon="💾")

        status.update(label="분석 완료!", state="complete", expanded=False)


def _run_batch_analysis(uploaded_files, ragas_enabled: bool) -> None:
    from bidflow.extraction.batch_pipeline import BatchPipeline

    profile = st.session_state.get("company_profile")
    if not profile:
        st.error("회사 프로필이 없습니다. 프로필을 먼저 설정하세요.")
        return

    pipeline = BatchPipeline(company_profile=profile, ragas_enabled=ragas_enabled)

    tmp_dir = tempfile.mkdtemp(prefix="bidflow_batch_")
    file_paths = []
    try:
        for uf in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())
            file_paths.append(tmp_path)

        progress_bar = st.progress(0, text="준비 중...")
        status_text = st.empty()

        def progress_callback(current, total, last_result):
            doc_name = last_result.doc_name if last_result else "..."
            signal = last_result.signal if last_result else ""
            progress_bar.progress(current / total, text=f"처리 중 {current}/{total}")
            if signal:
                status_text.write(f"최근 완료: **{doc_name}** → {_signal_badge(signal)}")

        batch_result = pipeline.process_batch(file_paths, progress_cb=progress_callback)
        progress_bar.progress(1.0, text="완료!")
        status_text.empty()
        st.session_state["batch_result"] = batch_result
    except Exception as e:
        st.error(f"일괄 분석 중 오류가 발생했습니다: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _render_batch_result() -> None:
    if "batch_result" not in st.session_state:
        return

    batch = st.session_state["batch_result"]
    st.divider()
    st.subheader("다문서 분석 결과 요약")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 문서", batch.total_docs)
    col2.metric("🟢 GREEN", batch.green_count)
    col3.metric("🔴 RED", batch.red_count)
    col4.metric("⚪ GRAY", batch.gray_count)
    st.caption(f"총 처리 시간: {batch.total_processing_time_sec:.1f}초")

    if batch.failed_docs:
        with st.expander(f"⚠️ 실패한 문서 ({len(batch.failed_docs)}건)", expanded=True):
            for fd in batch.failed_docs:
                st.error(f"**{fd['name']}**: {fd['error']}")

    st.divider()
    signal_order = {"RED": 0, "GRAY": 1, "GREEN": 2}
    sorted_results = sorted(batch.results, key=lambda r: signal_order.get(r.signal, 1))
    st.subheader("문서별 상세 결과")

    for doc_signal in sorted_results:
        badge = _signal_badge(doc_signal.signal)
        score_str = f"적합도 {doc_signal.fit_score:.0%}"
        time_str = f"{doc_signal.processing_time_sec:.1f}s"

        with st.expander(f"{badge} **{doc_signal.doc_name}** — {score_str} ({time_str})"):
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("신호", doc_signal.signal)
            mcol2.metric("적합도", f"{doc_signal.fit_score:.2f}")
            mcol3.metric("처리 시간", f"{doc_signal.processing_time_sec:.1f}s")

            if doc_signal.signal_reasons:
                st.write("**판정 사유:**")
                for reason in doc_signal.signal_reasons:
                    st.write(f"- {reason}")

    st.divider()
    if st.button("다문서 결과 초기화"):
        st.session_state.pop("batch_result", None)
        st.rerun()


user_id = require_login()
init_app_session(user_id=user_id)

st.title("RFP 문서 업로드")
st.caption("문서 1개 업로드 시 단일 분석, 2개 이상 업로드 시 다문서 일괄 분석으로 자동 전환됩니다.")

with st.sidebar:
    st.subheader("분석 옵션")
    ragas_enabled = st.toggle(
        "다문서 RAGAS 평가 (선택)",
        value=False,
        help="다문서 분석 시 문서당 약 3분 추가 소요됩니다.",
    )

uploaded_files = st.file_uploader(
    "RFP 문서를 업로드하세요 (PDF, HWP, DOCX, HWPX)",
    type=["pdf", "hwp", "docx", "hwpx"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"{len(uploaded_files)}개 파일 선택됨")
    for uf in uploaded_files:
        st.write(f"- {uf.name} ({uf.size:,} bytes)")

mode_label = "단일 분석" if len(uploaded_files) == 1 else "다문서 일괄 분석"
button_label = f"{mode_label} 시작" if uploaded_files else "분석 시작"

if uploaded_files and st.button(button_label, type="primary"):
    if len(uploaded_files) == 1:
        st.session_state.pop("batch_result", None)
        st.session_state.pop("extraction_results", None)
        st.session_state["analysis_success"] = False
        _run_single_analysis(uploaded_files[0], user_id=user_id)
    else:
        st.session_state.pop("analysis_success", None)
        st.session_state.pop("extraction_results", None)
        _run_batch_analysis(uploaded_files, ragas_enabled=ragas_enabled)
        st.rerun()

if st.session_state.get("analysis_success"):
    st.success("분석이 완료되었습니다!")
    try:
        st.page_link("pages/2_Matrix.py", label="결과 보기 (Go to Matrix)", icon="📊")
    except AttributeError:
        if st.button("결과 보기 (Go to Matrix)"):
            st.switch_page("pages/2_Matrix.py")

_render_batch_result()
