"""
Tool Execution Gate: LLM이 도구를 실행하거나 외부 리소스에 접근할 때
보안 정책(Allowlist, SSRF, Schema)을 검증하는 모듈입니다.
"""
import ipaddress
import logging
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger("bidflow.security.gate")

class ToolExecutionGate:
    def __init__(self, allowed_tools: Optional[Set[str]] = None):
        # 허용된 도구 이름 목록 (Allowlist)
        self.allowed_tools = allowed_tools or {"search_rfp", "get_company_profile"}
        
        # 내부망 IP 대역 (SSRF 방지용)
        self.private_networks = [
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("169.254.0.0/16"),  # Cloud metadata
            ipaddress.ip_network("::1/128"),         # IPv6 localhost
        ]

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """도구 실행 전 유효성 검사"""
        
        # 1. Allowlist 검사
        if tool_name not in self.allowed_tools:
            logger.warning(f"[Security] Blocked unauthorized tool execution: {tool_name}")
            return False

        # 2. 인자 내 URL 검사 (SSRF 방지)
        for key, value in arguments.items():
            if isinstance(value, str) and (value.startswith("http://") or value.startswith("https://")):
                if not self._is_safe_url(value):
                    logger.warning(f"[Security] Blocked SSRF attempt in argument '{key}': {value}")
                    return False

        # 3. 스키마 검증 (여기서는 간단한 타입 체크 예시)
        # 실제로는 Pydantic 모델을 사용하여 엄격하게 검증 권장
        if not self._validate_schema(tool_name, arguments):
            return False

        return True

    def _is_safe_url(self, url: str) -> bool:
        """URL이 내부망 IP나 허용되지 않은 호스트를 가리키는지 확인"""
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            
            if not hostname:
                return False

            # [Security] 마스킹된 호스트(***)는 차단
            if "*" in hostname:
                logger.warning(f"[Security] Blocked masked hostname: {hostname}")
                return False

            # IP 주소인 경우 직접 확인
            try:
                ip = ipaddress.ip_address(hostname)
                for network in self.private_networks:
                    if ip in network:
                        return False
            except ValueError:
                # 도메인 이름인 경우 (DNS Rebinding 공격 방지를 위해선 실제 resolve 필요하지만, 여기선 생략)
                if hostname in ["localhost", "metadata.google.internal"]:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"URL validation failed: {e}")
            return False

    def _validate_schema(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """도구별 필수 파라미터 및 타입 검증"""
        if tool_name == "search_rfp":
            # 예: query는 필수이며 문자열이어야 함
            if "query" not in args or not isinstance(args["query"], str):
                logger.warning(f"[Security] Schema validation failed for {tool_name}: invalid 'query'")
                return False
        return True

    def apply_to_langchain_tool(self, tool: Any) -> Any:
        """
        LangChain 도구(BaseTool)에 보안 게이트를 적용합니다.
        도구 실행(_run) 전에 validate_tool_call을 수행하도록 래핑합니다.
        """
        # 원본 _run 메서드 보존
        original_run = tool._run

        def guarded_run(*args, **kwargs):
            # 검증용 인자 구성
            check_args = kwargs.copy()
            
            # 위치 인자가 있는 경우 (예: tool("query"))
            if args:
                for i, arg in enumerate(args):
                    check_args[f"arg_{i}"] = arg

            # 보안 검사 실행
            if not self.validate_tool_call(tool.name, check_args):
                raise ValueError(f"[Security] Tool execution blocked by gate: {tool.name}")
            
            return original_run(*args, **kwargs)
        
        # 래핑된 메서드로 교체 (Monkey Patching)
        tool._run = guarded_run
        return tool
