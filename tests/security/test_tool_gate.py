import unittest

from bidflow.security.tool_gate import ToolExecutionGate


class TestToolExecutionGate(unittest.TestCase):
    def setUp(self):
        self.gate = ToolExecutionGate(allowed_tools={"search_rfp"})

    def test_blocks_unauthorized_tool(self):
        allowed = self.gate.validate_tool_call("delete_db", {"query": "test"})
        self.assertFalse(allowed)

    def test_blocks_private_network_ssrf(self):
        allowed = self.gate.validate_tool_call("search_rfp", {"query": "ok", "url": "http://127.0.0.1/admin"})
        self.assertFalse(allowed)

    def test_blocks_masked_host(self):
        allowed = self.gate.validate_tool_call("search_rfp", {"query": "ok", "url": "http://***.***.***.***/x"})
        self.assertFalse(allowed)

    def test_passes_safe_query(self):
        allowed = self.gate.validate_tool_call("search_rfp", {"query": "입찰 공고 요약"})
        self.assertTrue(allowed)

    def test_blocks_invalid_schema(self):
        allowed = self.gate.validate_tool_call("search_rfp", {"query": 1234})
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
