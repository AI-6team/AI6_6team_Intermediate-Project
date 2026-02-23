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

def get_config(env="dev"):
    # Find project root (assumed to be 3 levels up from this file: src/bidflow/core)
    root_dir = Path(__file__).parent.parent.parent.parent
    config_path = root_dir / "configs" / f"{env}.yaml"
    
    if not config_path.exists():
        # Fallback to defaults or raise error
        print(f"Warning: Config file {config_path} not found.")
        return Config({})
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        
    return Config(config_dict)
