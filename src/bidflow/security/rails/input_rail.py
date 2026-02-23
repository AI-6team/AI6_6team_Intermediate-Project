import re
from typing import List, Optional

class SecurityException(Exception):
    pass

class InputRail:
    """
    사용자 입력 및 문서 내용에서 프롬프트 인젝션/Jailbreak 시도를 탐지하고 차단합니다.
    Ref: OWASP Prompt Injection
    """
    
    # MVP: 정규식 기반 패턴 매칭 (가벼운 방어)
    # 실제로는 BERT 기반 분류 모델이나 전용 Security LLM을 병행해야 함
    # \s+를 사용하여 공백/줄바꿈 변종에 대응
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+the\s+above\s+instructions",
        r"system\s+prompt",
        r"you\s+are\s+a\s+(chat)?gpt",
        r"jailbreak",
        r"DAN\s+mode",
        r"do\s+anything\s+now",
        r"act\s+as",
        r"simulate",
        r"never\s+reveal",
        r"delete\s+your\s+system\s+prompt",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def check(self, text: str) -> bool:
        """
        텍스트에 위험 패턴이 포함되어 있는지 검사합니다.
        탐지 시 SecurityException을 발생시킵니다.
        """
        if not text:
            return True

        # Debug: 검사 대상 확인
        # print(f"[InputRail] Checking: {text[:20]}...")

        # 1. 정규식 검사
        for pattern in self.patterns:
            if pattern.search(text):
                print(f"[SECURITY] Injection attempt detected (Regex): {pattern.pattern}")
                raise SecurityException(f"Banned pattern detected in input: {pattern.pattern}")
        
        # 2. 확실한 Fallback (유저 테스트용)
        # 정규식 실수 방지를 위해 문자열이 그대로 있으면 무조건 차단
        test_phrases = ["ignore previous instructions", "system prompt", "jailbreak"]
        text_lower = text.lower()
        for phrase in test_phrases:
            if phrase in text_lower:
                print(f"[SECURITY] Injection attempt detected (Exact): {phrase}")
                raise SecurityException(f"Banned pattern detected: {phrase}")

        return True

    def sanitize(self, text: str) -> str:
        """
        (선택) 위험한 부분을 제거하거나 이스케이프 처리하여 반환합니다.
        현재는 탐지 우선이므로 원본 반환.
        """
        self.check(text)
        return text
