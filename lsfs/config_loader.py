import os
from .config import LSFSConfig


def load_config_from_yaml(yaml_path: str = "config.yaml") -> LSFSConfig:
    """Load configuration from YAML file"""
    if not os.path.exists(yaml_path):
        print(f"Config file not found: {yaml_path}")
        print("Using default configuration from code")
        return LSFSConfig()

    try:
        config = LSFSConfig.from_yaml(yaml_path)
        print(f"Configuration loaded from {yaml_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration from code")
        return LSFSConfig()
