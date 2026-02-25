from bidflow.security.pii_filter import PIIFilter


class PIIMasker:
    """
    기존 호출부와 호환되는 래퍼.
    내부 구현은 중앙화된 PIIFilter를 사용합니다.
    """

    def __init__(self):
        self._pii_filter = PIIFilter()

    def mask(self, text: str) -> str:
        if not text:
            return ""
        # 기존 PIIMasker 동작과 호환되도록 길이 제한 없이 마스킹만 적용합니다.
        return self._pii_filter.sanitize(text, max_len=max(len(text), 2000))
