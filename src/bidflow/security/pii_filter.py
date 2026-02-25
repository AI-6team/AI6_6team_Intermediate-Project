import re
from typing import Optional

class PIIFilter:
    """
    개인정보(PII) 탐지 및 마스킹을 담당하는 보안 필터.
    Input Rail(Sanitization)과 Output Rail(Validation)에서 공통으로 사용됩니다.
    """
    def __init__(self):
        # 탐지용 패턴 (Output Validation용) - 미리 컴파일하여 성능 최적화
        self.detection_patterns = {
            "Resident Registration Number": re.compile(r'\d{6}[-][1-4]\d{6}'),
            "Foreigner Registration Number": re.compile(r'\d{6}[-][5-8]\d{6}'),
            "Credit Card Number": re.compile(r'(\d{4}[-\s]?){3}\d{4}'),
            "Email Address": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            "Driver License Number": re.compile(r'\d{2}[-]\d{2}[-]\d{6}[-]\d{2}'),
            "Phone Number": re.compile(r'(01[016789]|02|0[3-9]\d)[-\s]?\d{3,4}[-\s]?\d{4}'),
            "Passport Number": re.compile(r'[a-zA-Z]\d{8}'),
            "IP Address": re.compile(r'(?<!\d)(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?!\d)')
        }
        
        # 마스킹용 패턴 (Input Sanitization용) - 미리 컴파일하여 성능 최적화
        self.masking_patterns = [
            (re.compile(r'(?<!\d)(\d{6})([-\s]?)[1-4]\d{6}(?!\d)'), r'\1\2*******'),  # 주민번호 (내국인)
            (re.compile(r'(?<!\d)(\d{6})([-\s]?)[5-8]\d{6}(?!\d)'), r'\1\2*******'),  # 외국인등록번호
            (re.compile(r'(\d{4}[-\s]?){3}\d{4}'), r'****-****-****-****'),      # 카드번호
            (re.compile(r'([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), r'\1@****'), # 이메일
            (re.compile(r'(?<!\d)(\d{2})([-\s]?)\d{2}([-\s]?)\d{6}([-\s]?)\d{2}(?!\d)'), r'\1\2**\3******\4**'), # 운전면허
            (re.compile(r'(01[016789]|02|0[3-9]\d)([-\s]?)\d{3,4}([-\s]?)(\d{4})'), r'\1\2****\3\4'), # 전화번호
            (re.compile(r'([a-zA-Z])\d{8}'), r'\1********'), # 여권번호
            (re.compile(r'(?<!\d)(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?!\d)'), r'***.***.***.***') # IP 주소
        ]

    def sanitize(self, text: str, max_len: int = 2000) -> str:
        """
        입력 텍스트 길이 제한 및 PII 마스킹 (Input Sanitization)
        """
        # 1. 길이 제한 (DoS 방지)
        if len(text) > max_len:
            text = text[:max_len]
        
        # 2. PII 마스킹
        for pattern, replacement in self.masking_patterns:
            text = pattern.sub(replacement, text)
        
        return text

    def detect(self, text: str) -> Optional[str]:
        """
        텍스트 내 PII 포함 여부 확인 (Output Validation).
        발견된 첫 번째 PII 유형을 반환하며, 없으면 None을 반환합니다.
        """
        for pii_type, pattern in self.detection_patterns.items():
            if pattern.search(text):
                return pii_type
        return None
