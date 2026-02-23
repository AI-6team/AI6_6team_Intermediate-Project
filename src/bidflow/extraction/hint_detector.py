"""Regex-based hint detection for money and date patterns.

Ported from 김슬기's extraction pipeline.
Detects numeric/date patterns in context text and formats them as LLM hints
to improve extraction accuracy.
"""
import re
from typing import List


class HintDetector:
    """정규식 기반 금액/날짜 패턴 감지 및 LLM 힌트 생성"""

    MONEY_PATTERNS = [
        r"\b\d{1,3}(?:,\d{3})+(?:원)\b",          # 100,000,000원
        r"금\s?[\d,]+원",                           # 금 100,000,000원
        r"금\s?[가-힣\s]+원",                        # 금 일억 오천만 원
        r"\b\d{1,3}(?:,\d{3})+\s?천원\b",           # 493,763천원
        r"추정\s?가격\s?[\d,]+",                     # 추정가격 493,763,000
    ]

    DATE_PATTERNS = [
        r"\d{4}년\s?\d{1,2}월\s?\d{1,2}일",          # 2024년 5월 1일
        r"\d{4}\.\s?\d{1,2}\.\s?\d{1,2}",           # 2024. 5. 1
        r"\d{4}-\d{1,2}-\d{1,2}",                   # 2024-05-01
    ]

    def detect_money(self, text: str) -> List[str]:
        """텍스트에서 금액 패턴을 감지하여 반환"""
        candidates = set()
        for p in self.MONEY_PATTERNS:
            candidates.update(re.findall(p, text))
        return sorted(list(candidates), key=len, reverse=True)[:10]

    def detect_dates(self, text: str) -> List[str]:
        """텍스트에서 날짜 패턴을 감지하여 반환"""
        candidates = set()
        for p in self.DATE_PATTERNS:
            candidates.update(re.findall(p, text))
        return sorted(list(candidates), reverse=True)[:10]

    def format_hints(self, text: str) -> str:
        """금액/날짜 힌트를 포맷팅된 문자열로 반환. 힌트가 없으면 빈 문자열."""
        money = self.detect_money(text)
        dates = self.detect_dates(text)

        parts = []
        if money:
            parts.append(f"금액: {', '.join(money)}")
        if dates:
            parts.append(f"날짜: {', '.join(dates)}")

        if not parts:
            return ""

        return "[AI 힌트: 텍스트에서 감지된 주요 패턴]\n" + " / ".join(parts)
