import unittest

from bidflow.security.pii_filter import PIIFilter


class TestPIIFilter(unittest.TestCase):
    def setUp(self):
        self.pii_filter = PIIFilter()

    def test_sanitize_masks_sensitive_fields(self):
        text = "주민번호 900101-1234567, 이메일 user@example.com, 연락처 010-1234-5678"
        sanitized = self.pii_filter.sanitize(text)
        self.assertIn("900101-*******", sanitized)
        self.assertIn("user@****", sanitized)
        self.assertIn("010-****-5678", sanitized)

    def test_sanitize_applies_length_limit(self):
        long_text = "A" * 3000
        sanitized = self.pii_filter.sanitize(long_text, max_len=100)
        self.assertEqual(len(sanitized), 100)

    def test_detect_returns_first_detected_type(self):
        pii_type = self.pii_filter.detect("연락처는 010-1234-5678 입니다.")
        self.assertEqual(pii_type, "Phone Number")

    def test_detect_none_for_clean_text(self):
        pii_type = self.pii_filter.detect("개인정보가 없는 일반 문장입니다.")
        self.assertIsNone(pii_type)


if __name__ == "__main__":
    unittest.main()
