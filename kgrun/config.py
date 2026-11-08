"""
Configuration loader and validator for kgrun projects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import tomllib

CONFIG_FILE_NAME = "kgrun.toml"

DEFAULT_INCLUDES = [
    "*.py",
    "src/**",
    "configs/**",
    "requirements*.txt",
]

DEFAULT_EXCLUDES = [
    ".git/**",
    "venv/**",
    ".venv/**",
    "data/**",
    "runs/**",
    "**/__pycache__/**",
    "*.pyc",
    ".kgrun/**",
]


def find_config(start_dir: Path | None = None) -> Path | None:
    """
    Search upwards from the specified directory for kgrun.toml.
    Returns the Path if found, otherwise None.
    """
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        config_path = parent / CONFIG_FILE_NAME
        if config_path.is_file():
            return config_path
    return None


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Parse and validate kgrun.toml configuration.
    """
    with open(config_path, "rb") as fp:
        raw = tomllib.load(fp)

    kernel = raw.get("kernel")
    if not kernel or not isinstance(kernel, str):
        raise ValueError(f"Missing or invalid 'kernel' field in {config_path}")

    config = {
        "kernel": kernel.strip(),
        "accelerator": raw.get("accelerator", "NvidiaTeslaT4"),
        "timeout_h": int(raw.get("timeout_h", 11)),
        "internet": bool(raw.get("internet", True)),
        "datasets": list(raw.get("datasets", [])),
        "include": list(raw.get("include", DEFAULT_INCLUDES)),
        "exclude": list(raw.get("exclude", DEFAULT_EXCLUDES)),
        "pip": str(raw.get("pip", "")).strip(),
    }
    return config


def create_default_config(
    target_path: Path,
    kernel_slug: str,
    accelerator: str = "NvidiaTeslaT4",
    timeout_h: int = 11,
) -> None:
    """
    Write a starter kgrun.toml file.
    """
    content = f'''# kgrun configuration file
kernel      = "{kernel_slug}"
accelerator = "{accelerator}"
timeout_h   = {timeout_h}
internet    = true
datasets    = []
include     = ["*.py", "src/**", "configs/**", "requirements*.txt"]
exclude     = ["venv/**", ".venv/**", "data/**", "runs/**", "**/__pycache__/**", ".git/**", ".kgrun/**"]
# pip       = "requirements-kaggle.txt"
'''
    target_path.write_text(content, encoding="utf-8")

