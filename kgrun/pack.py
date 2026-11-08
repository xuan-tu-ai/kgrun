"""
Packaging utilities for archiving project files into an in-memory bundle.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import io
import os
from pathlib import Path
import tarfile

# Maximum compressed tarball size to stay under Kaggle 1 MiB inline script limit
MAX_BUNDLE_BYTES = 750 * 1024


def _matches_any(rel_posix: str, patterns: list[str]) -> bool:
    """
    Check if a relative POSIX path matches any glob pattern.
    """
    basename = os.path.basename(rel_posix)
    for pattern in patterns:
        pat = pattern.replace("\\", "/").strip()
        if not pat:
            continue
        # Handle prefix wildcards like venv/**
        if pat.endswith("/**"):
            prefix = pat[:-3].rstrip("/")
            if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
                return True
        # Handle **/name/**
        if pat.startswith("**/") and pat.endswith("/**"):
            mid = pat[3:-3].strip("/")
            parts = rel_posix.split("/")
            if mid in parts:
                return True
        # Standard fnmatch
        if fnmatch.fnmatch(rel_posix, pat):
            return True
        # Match basename if pattern contains no directory separator
        if "/" not in pat and fnmatch.fnmatch(basename, pat):
            return True
    return False


def collect_project_files(
    root_dir: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[Path]:
    """
    Discover all files in root_dir matching inclusion rules and not excluded.
    """
    matched: list[Path] = []
    root = root_dir.resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        rel_dir = current_dir.relative_to(root).as_posix()

        # Prune known ignored directories during tree traversal for performance
        if rel_dir != ".":
            if _matches_any(rel_dir, exclude_patterns):
                dirnames.clear()
                continue

        for filename in filenames:
            file_path = current_dir / filename
            rel_file = file_path.relative_to(root).as_posix()

            if _matches_any(rel_file, exclude_patterns):
                continue
            if _matches_any(rel_file, include_patterns):
                matched.append(file_path)

    matched.sort()
    return matched


def create_code_bundle(
    root_dir: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> tuple[str, str, list[str]]:
    """
    Archive matching files into a base64-encoded compressed tarball.
    Returns:
        tuple of (base64_payload, sha256_hexdigest, list_of_relative_paths)
    """
    root = root_dir.resolve()
    files = collect_project_files(root, include_patterns, exclude_patterns)

    if not files:
        raise ValueError(
            f"No files matched include patterns {include_patterns} in {root_dir}"
        )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for file_path in files:
            rel_name = file_path.relative_to(root).as_posix()
            tar.add(file_path, arcname=rel_name)

    tar_bytes = buf.getvalue()
    bundle_size = len(tar_bytes)

    if bundle_size > MAX_BUNDLE_BYTES:
        raise ValueError(
            f"Code bundle ({bundle_size / 1024:.1f} KB) exceeds safe limit of {MAX_BUNDLE_BYTES / 1024:.0f} KB. "
            "Please exclude large files or data in kgrun.toml and upload them as Kaggle Datasets."
        )

    bundle_sha256 = hashlib.sha256(tar_bytes).hexdigest()
    b64_payload = base64.b64encode(tar_bytes).decode("ascii")
    relative_names = [f.relative_to(root).as_posix() for f in files]

    return b64_payload, bundle_sha256, relative_names

