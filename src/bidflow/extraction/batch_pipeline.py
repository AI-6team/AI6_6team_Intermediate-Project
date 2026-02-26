"""다문서 일괄 처리 파이프라인.

문서별 독립 파이프라인: 파싱 -> 인덱싱 -> 추출 -> 검증 -> 신호 판정.
개별 실패 격리: 1개 문서 실패해도 나머지 계속 처리.
"""
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from bidflow.core.config import get_config
from bidflow.domain.models import (
    CompanyProfile, ComplianceMatrix, ExtractionSlot,
    ValidationResult, DocumentSignal, BatchAnalysisResult,
)
from bidflow.ingest.storage import VectorStoreManager
from bidflow.ingest.collection_manager import CollectionManager
from bidflow.extraction.pipeline import ExtractionPipeline
from bidflow.validation.validator import RuleBasedValidator


def compute_content_hash(file_path: str) -> str:
    """파일 내용의 MD5 해시 (RFPLoader/DocumentStore doc_hash 규칙과 정합)."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_fit_score(
    results: List[ValidationResult],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """검증 결과에서 종합 적합도 산출 (0.0~1.0).

    Args:
        results: ValidationResult 목록
        weights: slot_key별 가중치 (None이면 기본값 사용)

    Returns:
        가중 적합도 점수
    """
    if not results:
        return 0.0

    if weights is None:
        weights = {
            "required_licenses": 3.0,
            "financial_credit": 2.0,
            "region_restriction": 2.0,
            "budget_check": 1.0,
            "deadline_check": 1.0,
            "info_completeness": 1.0,
        }
    score_map = {"GREEN": 1.0, "GRAY": 0.5, "RED": 0.0}

    total_weight = 0.0
    weighted_sum = 0.0
    for r in results:
        w = weights.get(r.slot_key, 1.0)
        total_weight += w
        weighted_sum += w * score_map.get(r.decision, 0.0)

    return weighted_sum / total_weight if total_weight > 0 else 0.0


class BatchPipeline:
    """다문서 일괄 처리 파이프라인."""

    def __init__(
        self,
        company_profile: CompanyProfile,
        ragas_enabled: bool = False,
        config=None,
    ):
        self.profile = company_profile
        self.validator = RuleBasedValidator()
        self.ragas_enabled = ragas_enabled
        self.config = config or get_config()
        self.collection_manager = CollectionManager()

        # signal 설정
        signal_cfg = self.config.signal or {}
        self._green_threshold = signal_cfg.get("green_threshold", 0.7) if isinstance(signal_cfg, dict) else 0.7
        self._mandatory_slots = signal_cfg.get("mandatory_slots", [
            "required_licenses", "financial_credit", "region_restriction"
        ]) if isinstance(signal_cfg, dict) else ["required_licenses", "financial_credit", "region_restriction"]
        self._weights = signal_cfg.get("weights", None) if isinstance(signal_cfg, dict) else None

    def process_single(self, file_path: str) -> DocumentSignal:
        """문서별 독립 파이프라인."""
        start_time = time.time()

        # 파일 내용 해시 (P1-2)
        doc_hash = compute_content_hash(file_path)
        collection_name = f"bidflow_{doc_hash[:12]}"

        # 동일 문서 재업로드 시 캐시 확인 (P1-3)
        cached = self.collection_manager.find_by_hash(doc_hash)
        skip_indexing = False
        if cached:
            collection_name = cached["collection_name"]
            self.collection_manager.touch(collection_name)
            skip_indexing = True
            print(f"[BatchPipeline] Cache hit for {Path(file_path).name}, reusing {collection_name}")
        else:
            collection_name = self._ensure_unique_collection(collection_name, doc_hash)

        # 격리된 VectorStoreManager
        vsm = VectorStoreManager(collection_name=collection_name)

        if not skip_indexing:
            # 파싱 + 인덱싱
            self._parse_and_index(file_path, doc_hash, vsm)
            self.collection_manager.register(collection_name, doc_hash, Path(file_path).name)

        # G1-G4 추출
        pipeline = ExtractionPipeline(vector_manager=vsm)
        extraction_result = pipeline.run(doc_hash)

        # ComplianceMatrix 구성 -> 검증
        matrix = self._build_matrix(extraction_result, doc_hash)
        validation_results = self.validator.validate(matrix, self.profile)

        # fit_score + 신호 결정
        fit_score = compute_fit_score(validation_results, self._weights)
        signal = self._determine_signal(validation_results, fit_score)
        reasons = self._build_signal_reasons(validation_results, fit_score, signal)

        return DocumentSignal(
            doc_hash=doc_hash,
            doc_name=Path(file_path).name,
            signal=signal,
            fit_score=fit_score,
            validation_results=validation_results,
            extraction_summary=extraction_result,
            signal_reasons=reasons,
            collection_name=collection_name,
            processing_time_sec=time.time() - start_time,
        )

    def process_batch(
        self,
        file_paths: List[str],
        progress_cb: Optional[Callable] = None,
    ) -> BatchAnalysisResult:
        """전체 문서 순차 처리 -- 개별 실패 격리."""
        # TTL 만료 컬렉션 정리
        ttl_cfg = self.config.signal or {}
        ttl_days = ttl_cfg.get("collection_ttl_days", 7) if isinstance(ttl_cfg, dict) else 7
        self.collection_manager.cleanup_expired(ttl_days)

        results: List[DocumentSignal] = []
        failed: List[Dict[str, str]] = []

        for i, path in enumerate(file_paths):
            try:
                signal = self.process_single(path)
                results.append(signal)
            except Exception as e:
                failed.append({"name": Path(path).name, "error": str(e)})
                print(f"[BatchPipeline] Failed: {Path(path).name} - {e}")

            if progress_cb:
                progress_cb(i + 1, len(file_paths), results[-1] if results else None)

        return BatchAnalysisResult(
            results=results,
            total_docs=len(file_paths),
            green_count=sum(1 for r in results if r.signal == "GREEN"),
            red_count=sum(1 for r in results if r.signal == "RED"),
            gray_count=sum(1 for r in results if r.signal == "GRAY"),
            created_at=datetime.now().isoformat(),
            total_processing_time_sec=sum(r.processing_time_sec for r in results),
            failed_docs=failed,
        )

    def _determine_signal(self, results: List[ValidationResult], fit_score: float) -> str:
        """신호 판정 로직.

        RED: 필수 조건 1개라도 RED, 또는 비필수라도 RED 존재
        GREEN: 전체 결과에 RED/GRAY 없음 + fit_score >= threshold
        GRAY: 나머지 (GRAY 항목 존재, 또는 fit_score 미달)
        """
        mandatory_results = [r for r in results if r.slot_key in self._mandatory_slots]

        # RED: 필수 항목 중 하나라도 RED
        if any(r.decision == "RED" for r in mandatory_results):
            return "RED"

        # RED: 비필수라도 RED 존재 (마감 경과 등)
        if any(r.decision == "RED" for r in results):
            return "RED"

        # GREEN 조건: 전체 결과에 GRAY 없음 + fit_score >= threshold
        all_green = all(r.decision == "GREEN" for r in results)
        if all_green and fit_score >= self._green_threshold:
            return "GREEN"

        return "GRAY"

    def _build_signal_reasons(
        self, results: List[ValidationResult], fit_score: float, signal: str
    ) -> List[str]:
        """판정 사유 목록 생성."""
        reasons = []
        if signal == "RED":
            for r in results:
                if r.decision == "RED":
                    reasons.append(f"[RED] {r.slot_key}: {r.reasons[0] if r.reasons else 'N/A'}")
        elif signal == "GRAY":
            for r in results:
                if r.decision == "GRAY":
                    reasons.append(f"[GRAY] {r.slot_key}: {r.reasons[0] if r.reasons else 'N/A'}")
            if fit_score < self._green_threshold:
                reasons.append(f"[GRAY] fit_score={fit_score:.2f} < {self._green_threshold}")
        else:
            reasons.append(f"모든 조건 충족 (fit_score={fit_score:.2f})")
        return reasons

    def _build_matrix(self, extraction_result: Dict[str, Any], doc_hash: str) -> ComplianceMatrix:
        """추출 결과에서 ComplianceMatrix 구성."""
        slots = {}

        # G1: 기본 정보에서 budget 추출
        g1 = extraction_result.get("g1", {})
        budget_data = g1.get("budget", g1.get("estimated_budget", {}))
        if isinstance(budget_data, dict):
            slots["budget"] = ExtractionSlot(
                key="budget",
                value=budget_data.get("value"),
                status=budget_data.get("status", "NOT_FOUND"),
            )

        # G2: 일정에서 submission_deadline 추출
        g2 = extraction_result.get("g2", {})
        deadline_data = g2.get("submission_deadline", {})
        if isinstance(deadline_data, dict):
            slots["submission_deadline"] = ExtractionSlot(
                key="submission_deadline",
                value=deadline_data.get("value"),
                status=deadline_data.get("status", "NOT_FOUND"),
            )

        # G3: 자격 요건
        g3 = extraction_result.get("g3", {})
        for key in ["required_licenses", "financial_credit", "region_restriction"]:
            data = g3.get(key, {})
            if isinstance(data, dict):
                slots[key] = ExtractionSlot(
                    key=key,
                    value=data.get("value"),
                    status=data.get("status", "NOT_FOUND"),
                )

        return ComplianceMatrix(doc_hash=doc_hash, slots=slots)

    def _parse_and_index(self, file_path: str, doc_hash: str, vsm: VectorStoreManager):
        """문서 파싱 후 ChromaDB 인덱싱."""
        from bidflow.domain.models import RFPDocument
        from bidflow.parsing.pdf_parser import PDFParser
        from bidflow.parsing.hwp_parser import HWPParser
        from bidflow.parsing.docx_parser import DOCXParser
        from bidflow.parsing.hwpx_parser import HWPXParser

        ext = Path(file_path).suffix.lower()
        chunks = []
        tables = []
        if ext == ".pdf":
            parser = PDFParser()
            chunks = parser.parse(file_path)
            tables = parser.extract_tables(file_path)
        elif ext == ".hwp":
            parser = HWPParser()
            chunks = parser.parse(file_path)
        elif ext == ".docx":
            parser = DOCXParser()
            chunks = parser.parse(file_path)
            tables = parser.extract_tables(file_path)
        elif ext == ".hwpx":
            parser = HWPXParser()
            chunks = parser.parse(file_path)
            tables = parser.extract_tables(file_path)
        else:
            raise ValueError(f"Unsupported file format for batch processing: {ext}")

        if not chunks:
            raise ValueError("문서 파싱 실패: 추출된 청크가 없습니다.")

        rfp_doc = RFPDocument(
            id=doc_hash,
            filename=Path(file_path).name,
            file_path=file_path,
            doc_hash=doc_hash,
            chunks=chunks,
            tables=tables,
            status="READY",
        )
        vsm.ingest_document(rfp_doc)
        print(f"[BatchPipeline] Indexed {Path(file_path).name} -> {vsm.collection_name}")

    def _ensure_unique_collection(self, collection_name: str, doc_hash: str) -> str:
        """컬렉션명 충돌 방지. 존재하면 suffix 추가."""
        existing = self.collection_manager.find_by_hash(doc_hash)
        if existing:
            return existing["collection_name"]

        registry = self.collection_manager.list_collections()
        if collection_name not in registry:
            return collection_name

        # 충돌: suffix 추가
        for i in range(1, 100):
            candidate = f"{collection_name}_{i}"
            if candidate not in registry:
                return candidate

        return f"{collection_name}_{doc_hash[:16]}"
