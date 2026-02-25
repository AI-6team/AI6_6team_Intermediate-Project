"""프롬프트 버전 레지스트리.

registry.yaml에서 메타데이터를 읽고 해당 프롬프트 파일을 로드합니다.
단일 소스(rag_v5.txt)를 유지하여 소스 갈라짐을 방지합니다.
"""
from pathlib import Path
import yaml

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(version: str = None) -> str:
    """레지스트리에서 프롬프트 로드. version=None이면 default 사용."""
    registry_path = _PROMPTS_DIR / "registry.yaml"
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    if version is None:
        version = registry["default"]
    versions = registry.get("versions", {})
    if version not in versions:
        raise KeyError(f"Prompt version '{version}' not found. Available: {list(versions.keys())}")
    entry = versions[version]
    prompt_path = _PROMPTS_DIR / entry["file"]
    return prompt_path.read_text(encoding="utf-8")
