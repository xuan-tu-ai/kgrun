"""
Unit tests for configuration loading and validation.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from kgrun.config import CONFIG_FILE_NAME, create_default_config, find_config, load_config


def test_create_and_load_config(tmp_path: Path) -> None:
    config_file = tmp_path / CONFIG_FILE_NAME
    create_default_config(config_file, kernel_slug="testuser/test-project", accelerator="NvidiaTeslaT4")

    assert config_file.is_file()
    cfg = load_config(config_file)

    assert cfg["kernel"] == "testuser/test-project"
    assert cfg["accelerator"] == "NvidiaTeslaT4"
    assert cfg["timeout_h"] == 11
    assert cfg["internet"] is True
    assert isinstance(cfg["include"], list)
    assert isinstance(cfg["exclude"], list)


def test_find_config_upwards(tmp_path: Path) -> None:
    config_file = tmp_path / CONFIG_FILE_NAME
    create_default_config(config_file, kernel_slug="testuser/parent-project")

    nested = tmp_path / "src" / "deep" / "nested"
    nested.mkdir(parents=True)

    found = find_config(start_dir=nested)
    assert found is not None
    assert found.resolve() == config_file.resolve()


def test_invalid_config_raises(tmp_path: Path) -> None:
    config_file = tmp_path / CONFIG_FILE_NAME
    config_file.write_text("invalid_toml = [", encoding="utf-8")

    with pytest.raises(Exception):
        load_config(config_file)
