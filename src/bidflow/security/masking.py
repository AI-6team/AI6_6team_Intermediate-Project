import re

class PIIMasker:
    """
    개인식별정보(PII)를 비식별화(Masking) 합니다.
    대상: 주민등록번호, 전화번호, 이메일
    """
    
    PATTERNS = {
        # 주민등록번호/외국인번호 (YYMMDD-XXXXXXX)
        "RRN": r"\b(\d{6})[- ]?(\d{7})\b",
        
        # 전화번호 (010-XXXX-XXXX, 02-XXX-XXXX 등 다양한 포맷 단순화)
        "PHONE": r"\b(01[016789]|02|0[3-9][0-9])[- )]?(\d{3,4})[- ]?(\d{4})\b",
        
        # 이메일
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    }

    def mask(self, text: str) -> str:
        if not text:
            return ""

        masked_text = text
        
        # 1. RRN -> YYMMDD-*******
        masked_text = re.sub(self.PATTERNS["RRN"], r"\1-*******", masked_text)
        
        # 2. PHONE -> 010-****-1234 (중간 마스킹)
        def mask_phone(match):
            # 그룹 1(지역/식별), 2(국번), 3(번호)
            g1, g2, g3 = match.groups()
            return f"{g1}-****-{g3}"
            
        masked_text = re.sub(self.PATTERNS["PHONE"], mask_phone, masked_text)
        
        # 3. EMAIL -> a***@domain.com
        def mask_email(match):
            email = match.group()
            try:
                user, domain = email.split('@')
                if len(user) > 3:
                    masked_user = user[:3] + "***"
                else:
                    masked_user = user[0] + "***"
                return f"{masked_user}@{domain}"
            except:
                return email # Fallback

        masked_text = re.sub(self.PATTERNS["EMAIL"], mask_email, masked_text)
        
        return masked_text
