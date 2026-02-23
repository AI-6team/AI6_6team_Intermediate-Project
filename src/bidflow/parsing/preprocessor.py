import unicodedata
import re

class TextPreprocessor:
    """
    보수적 정규화 전처리기 (Conservative Text Preprocessor)
    - Unicode NFKC (Ligature 해제)
    - Control Character Removal (개행/탭 보존)
    - Whitespace Normalization (문단 구분 보존)
    """
    
    from typing import Union, Tuple, Dict, List

    def normalize(self, text: str, return_report: bool = False) -> Union[str, Tuple[str, Dict]]:
        if not text:
            return ("" if not return_report else ("", {"removed_chars": [], "nfkc_changed": False}))
            
        report = {"removed_chars": [], "nfkc_changed": False}

        # 1. Unicode NFKC (Ligature 해제: ﬁ->fi, 전각->반각 등)
        nfkc_text = unicodedata.normalize('NFKC', text)
        if nfkc_text != text:
            report["nfkc_changed"] = True
        
        # 2. 제어 문자 제거 (개행/탭 보존) & 로깅
        # isprintable()이 False인 문자 중 \n, \t는 살림
        filtered_chars = []
        for c in nfkc_text:
            if c.isprintable() or c in ['\n', '\t']:
                filtered_chars.append(c)
            else:
                report["removed_chars"].append(ord(c))
        
        text = "".join(filtered_chars)
        
        # 3. 공백 정규화
        # 3.1. 개행이 아닌 공백문자(탭, 수직탭, 2칸 공백 등)를 단일 space로 정규화
        # 주의: 표 구조가 탭으로 구분된 경우라도, 텍스트 인덱싱을 위해 공백으로 치환함 (Raw Table 별도 추출 전제)
        text = re.sub(r'[^\S\n]+', ' ', text)
        
        # 3.2. 3개 이상의 연속 개행은 2개(문단 구분)로 축소
        # \n\n\n -> \n\n (빈 페이지나 과도한 여백 제거)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        text = text.strip()
        
        if return_report:
            return text, report
        return text
