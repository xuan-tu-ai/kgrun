"""
Dataset management utilities for Kaggle dataset creation, versioning, and status monitoring.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib

from kgrun.config import find_config
from kgrun.runner import get_authenticated_username


def update_config_datasets(config_path: Path, dataset_slug: str) -> None:
    """
    Append dataset slug to datasets array in kgrun.toml without removing comments.
    """
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    found = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("datasets") and "=" in stripped:
            found = True
            try:
                parsed = tomllib.loads(line)
                ds_list = list(parsed.get("datasets", []))
            except Exception:
                ds_list = []
            if dataset_slug not in ds_list:
                ds_list.append(dataset_slug)
            formatted = json.dumps(ds_list)
            lines[i] = f"datasets    = {formatted}"
            break

    if not found:
        lines.append(f'datasets    = ["{dataset_slug}"]')

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_for_dataset_ready(dataset_slug: str, max_wait_sec: int = 120) -> bool:
    """
    Poll Kaggle dataset status until it reaches 'ready' state.
    """
    start = time.time()
    last_status = "unknown"

    while time.time() - start < max_wait_sec:
        res = subprocess.run(
            ["kaggle", "datasets", "status", dataset_slug],
            capture_output=True,
            text=True,
            check=False,
        )
        status_text = (res.stdout + res.stderr).strip().lower()
        if res.returncode == 0:
            last_status = status_text
            if "ready" in status_text:
                return True
        time.sleep(3)

    print(f"[kgrun] Warning: Dataset status did not report 'ready' within {max_wait_sec}s (last: {last_status}).", file=sys.stderr)
    return False


def push_dataset(
    dir_path: str | Path,
    message: str | None = None,
    slug: str | None = None,
    update_config: bool = True,
) -> int:
    """
    Create or version a Kaggle dataset from a local directory.
    """
    target_dir = Path(dir_path).resolve()
    if not target_dir.is_dir():
        print(f"[kgrun] Error: '{dir_path}' is not an existing directory.", file=sys.stderr)
        return 1

    meta_file = target_dir / "dataset-metadata.json"
    username = get_authenticated_username() or "username"

    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            current_id = meta.get("id", "")
        except Exception:
            meta = {}
            current_id = ""
    else:
        meta = {}
        current_id = ""

    # Resolve dataset slug
    if slug:
        final_slug = slug if "/" in slug else f"{username}/{slug}"
    elif current_id:
        final_slug = current_id
    else:
        folder_slug = target_dir.name.lower().replace(" ", "-").replace("_", "-")
        final_slug = f"{username}/{folder_slug}"

    title = meta.get("title") or final_slug.split("/")[-1].replace("-", " ").title()

    meta["id"] = final_slug
    meta["title"] = title
    if "licenses" not in meta:
        meta["licenses"] = [{"name": "CC0-1.0"}]

    # Ensure dataset-metadata.json is written without UTF-8 BOM
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Check if dataset already exists
    check_res = subprocess.run(
        ["kaggle", "datasets", "status", final_slug],
        capture_output=True,
        text=True,
        check=False,
    )
    dataset_exists = (check_res.returncode == 0)

    if dataset_exists:
        version_msg = message or f"Updated dataset version at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"[kgrun] Dataset '{final_slug}' exists. Uploading new version (mode: tar)...")
        cmd = ["kaggle", "datasets", "version", "-p", str(target_dir), "-m", version_msg, "-r", "tar"]
    else:
        print(f"[kgrun] Creating new private dataset '{final_slug}' (mode: tar)...")
        cmd = ["kaggle", "datasets", "create", "-p", str(target_dir), "-r", "tar"]

    upload_res = subprocess.run(cmd)
    if upload_res.returncode != 0:
        print("[kgrun] Error: Kaggle dataset upload command failed.", file=sys.stderr)
        return upload_res.returncode

    print(f"[kgrun] Waiting for dataset '{final_slug}' to become ready...")
    ready = wait_for_dataset_ready(final_slug)
    if ready:
        print(f"[kgrun] Dataset '{final_slug}' is ready on Kaggle!")

    if update_config:
        config_path = find_config()
        if config_path:
            update_config_datasets(config_path, final_slug)
            print(f"[kgrun] Added '{final_slug}' to datasets in {config_path.name}")

    return 0

