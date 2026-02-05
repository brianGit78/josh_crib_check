import yaml
import os
from typing import Any


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load and parse a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_service_config() -> dict[str, Any]:
    """Load service configuration for real-time video processing."""
    return load_yaml_config('service_config.yaml')


def load_train_config() -> dict[str, Any]:
    """Load training configuration."""
    return load_yaml_config('train_config.yaml')
