"""
Unit tests for code packaging and pattern matching.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import tarfile
import pytest

from kgrun.pack import collect_project_files, create_code_bundle


def test_collect_project_files(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "utils.py").write_text("def helper(): pass", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")

    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "model.py").write_text("class Net: pass", encoding="utf-8")

    ignored = tmp_path / "venv"
    ignored.mkdir()
    (ignored / "lib.py").write_text("pass", encoding="utf-8")

    include = ["*.py", "src/**"]
    exclude = ["venv/**", "data.csv"]

    matched = collect_project_files(tmp_path, include, exclude)
    rel_names = [p.relative_to(tmp_path).as_posix() for p in matched]

    assert "main.py" in rel_names
    assert "utils.py" in rel_names
    assert "src/model.py" in rel_names
    assert "data.csv" not in rel_names
    assert "venv/lib.py" not in rel_names


def test_create_code_bundle(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('train')", encoding="utf-8")

    b64_payload, sha256_hash, files = create_code_bundle(
        tmp_path,
        include_patterns=["*.py"],
        exclude_patterns=[],
    )

    assert len(files) == 1
    assert files[0] == "train.py"

    raw_tar = base64.b64decode(b64_payload)
    assert hashlib.sha256(raw_tar).hexdigest() == sha256_hash


def test_bundle_size_exceeded(tmp_path: Path) -> None:
    import os
    # Create an uncompressible random binary file (> 800 KB)
    large_file = tmp_path / "big_weights.py"
    large_file.write_bytes(os.urandom(800 * 1024))

    with pytest.raises(ValueError, match="exceeds safe limit"):
        create_code_bundle(
            tmp_path,
            include_patterns=["*.py"],
            exclude_patterns=[],
        )
