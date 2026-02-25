import unittest
import sys
import os

# src 디렉토리를 모듈 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bidflow.security.pii_filter import PIIFilter

class TestPIIFilter(unittest.TestCase):
    def setUp(self):
        self.pii_filter = PIIFilter()

    def test_sanitize_resident_number(self):
        """주민등록번호 마스킹 테스트"""
        text = "주민번호는 900101-1234567 입니다."
        expected = "주민번호는 900101-******* 입니다."
        result = self.pii_filter.sanitize(text)
        print(f"\n[Resident] {text} -> {result}")
        self.assertEqual(result, expected)

    def test_sanitize_foreigner_registration_number(self):
        """외국인등록번호 마스킹 테스트"""
        text = "외국인번호는 990101-5123456 입니다."
        expected = "외국인번호는 990101-******* 입니다."
        result = self.pii_filter.sanitize(text)
        print(f"\n[Foreigner] {text} -> {result}")
        self.assertEqual(result, expected)

    def test_sanitize_credit_card(self):
        """신용카드 번호 마스킹 테스트"""
        text = "결제 카드: 1234-5678-1234-5678"
        expected = "결제 카드: ****-****-****-****"
        result = self.pii_filter.sanitize(text)
        print(f"\n[CreditCard] {text} -> {result}")
        self.assertEqual(result, expected)

    def test_sanitize_email(self):
        """이메일 주소 마스킹 테스트"""
        text = "문의: support@example.com"
        expected = "문의: support@****"
        result = self.pii_filter.sanitize(text)
        print(f"\n[Email] {text} -> {result}")
        self.assertEqual(result, expected)

    def test_sanitize_driver_license(self):
        """운전면허번호 마스킹 테스트"""
        text = "면허번호: 11-22-333333-44"
        expected = "면허번호: 11-**-******-**"
        result = self.pii_filter.sanitize(text)
        print(f"\n[DriverLicense] {text} -> {result}")
        self.assertEqual(result, expected)

    def test_sanitize_phone_number(self):
        """전화번호 마스킹 테스트"""
        text = "연락처: 010-1234-5678, 02-123-4567"
        expected = "연락처: 010-****-5678, 02-****-4567"
        result = self.pii_filter.sanitize(text)
        print(f"\n[Phone] {text} -> {result}")
        self.assertEqual(result, expected)

    def test_sanitize_passport(self):
        """여권번호 마스킹 테스트"""
        text = "여권: M12345678"
        expected = "여권: M********"
        result = self.pii_filter.sanitize(text)
        print(f"\n[Passport] {text} -> {result}")
        self.assertEqual(result, expected)

    def test_sanitize_ip_address(self):
        """IP 주소 마스킹 테스트"""
        text = "서버 IP는 192.168.0.1 입니다."
        expected = "서버 IP는 ***.***.***.*** 입니다."
        result = self.pii_filter.sanitize(text)
        print(f"\n[IP] {text} -> {result}")
        self.assertEqual(result, expected)

    def test_length_limit(self):
        """길이 제한 테스트"""
        long_text = "A" * 3000
        sanitized = self.pii_filter.sanitize(long_text, max_len=100)
        print(f"\n[LengthLimit] Input len: {len(long_text)} -> Output len: {len(sanitized)}")
        self.assertEqual(len(sanitized), 100)

    def test_detect(self):
        """PII 탐지 테스트"""
        res1 = self.pii_filter.detect("010-1234-5678")
        print(f"\n[Detect] 010-1234-5678 -> {res1}")
        self.assertEqual(res1, "Phone Number")

        res_foreigner = self.pii_filter.detect("990101-5123456")
        print(f"\n[Detect] 990101-5123456 -> {res_foreigner}")
        self.assertEqual(res_foreigner, "Foreigner Registration Number")

        res2 = self.pii_filter.detect("user@example.com")
        print(f"\n[Detect] user@example.com -> {res2}")
        self.assertEqual(res2, "Email Address")

        res_ip = self.pii_filter.detect("192.168.0.1")
        print(f"\n[Detect] 192.168.0.1 -> {res_ip}")
        self.assertEqual(res_ip, "IP Address")

        res3 = self.pii_filter.detect("안녕하세요. 개인정보가 없는 문장입니다.")
        print(f"\n[Detect] Clean text -> {res3}")
        self.assertIsNone(res3)

if __name__ == '__main__':
    unittest.main()
