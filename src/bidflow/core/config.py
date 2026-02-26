import os
import yaml
from pathlib import Path


class Config(dict):
    def __init__(self, config_dict):
        super().__init__(config_dict)

    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return Config(value)
        return value


def _load_yaml(path: Path) -> dict:
    """YAML 파일 로드. 파일이 없으면 빈 dict 반환."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """중첩 딕셔너리 병합: override가 base를 덮어씀."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_project_root() -> Path:
    """프로젝트 루트 디렉토리(bidflow/)의 절대 경로를 반환합니다."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_config(env="dev"):
    root_dir = get_project_root()
    # 1. base.yaml 로드 (공통 기본값)
    base_path = root_dir / "configs" / "base.yaml"
    config = _load_yaml(base_path)
    # 2. 환경별 yaml 병합 (deep merge)
    env_path = root_dir / "configs" / f"{env}.yaml"
    env_config = _load_yaml(env_path)
    if not env_config:
        print(f"Warning: Config file {env_path} not found or empty.")
    config = _deep_merge(config, env_config)
    return Config(config)
