"""
Core orchestrator for running scripts, tracking history, streaming logs, and pulling outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from kgrun.bootstrap import generate_bootstrap_script
from kgrun.config import find_config, load_config
from kgrun.pack import create_code_bundle


def get_authenticated_username() -> str:
    """
    Detect the authenticated Kaggle username from local credentials.
    """
    home = Path.home()
    for fname in ["credentials.json", "kaggle.json"]:
        p = home / ".kaggle" / fname
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                uname = data.get("username")
                if uname:
                    return str(uname).strip()
            except Exception:
                pass
    return ""


def append_history(root_dir: Path, record: dict[str, Any]) -> None:
    """
    Append an execution entry to .kgrun/history.jsonl.
    """
    dot_kgrun = root_dir / ".kgrun"
    dot_kgrun.mkdir(parents=True, exist_ok=True)
    history_file = dot_kgrun / "history.jsonl"
    with open(history_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_command_interactive(cmd: list[str]) -> int:
    """
    Execute a subprocess command, streaming stdout and stderr to the console.
    Handles Ctrl+C gracefully without failing unhandled.
    """
    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[kgrun] Interrupted by user (Ctrl+C). The Kaggle remote job continues running in background.", flush=True)
        return 130


def run_script(
    script: str,
    script_args: list[str],
    gpu: str | None = None,
    no_follow: bool = False,
    pull: bool = False,
    timeout_h: float | None = None,
) -> int:
    """
    Package code, push kernel to Kaggle, stream logs, and optionally pull outputs.
    """
    config_path = find_config()
    if not config_path:
        print("[kgrun] Error: kgrun.toml not found. Run 'kgrun init' first.", file=sys.stderr)
        return 1

    root_dir = config_path.parent
    config = load_config(config_path)

    script_path = (root_dir / script).resolve()
    if not script_path.is_file():
        print(f"[kgrun] Error: Target script '{script}' does not exist in {root_dir}", file=sys.stderr)
        return 1

    rel_script = script_path.relative_to(root_dir).as_posix()

    # Determine accelerator
    accelerator = config["accelerator"]
    enable_gpu = True
    if gpu:
        gpu_lower = gpu.lower()
        if gpu_lower in ("t4", "teslat4", "nvidiateslat4"):
            accelerator = "NvidiaTeslaT4"
            enable_gpu = True
        elif gpu_lower in ("p100", "teslap100", "nvidiateslap100"):
            accelerator = "NvidiaTeslaP100"
            enable_gpu = True
        elif gpu_lower == "none":
            accelerator = None
            enable_gpu = False
        else:
            accelerator = gpu
            enable_gpu = True
    elif not accelerator or accelerator.lower() == "none":
        accelerator = None
        enable_gpu = False

    # Determine timeout in seconds
    effective_timeout_h = timeout_h if timeout_h is not None else config["timeout_h"]
    timeout_seconds = int(effective_timeout_h * 3600)

    # Package files
    print(f"[kgrun] Packaging files from {root_dir}...")
    try:
        b64_payload, sha256_hash, files = create_code_bundle(
            root_dir,
            config["include"],
            config["exclude"],
        )
    except Exception as exc:
        print(f"[kgrun] Packaging error: {exc}", file=sys.stderr)
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[kgrun] Packaged {len(files)} files (SHA256: {sha256_hash[:8]}...) for run {run_id}")

    # Generate bootstrap script
    pip_req = config.get("pip", "")
    bootstrap_code = generate_bootstrap_script(
        payload_b64=b64_payload,
        expected_sha256=sha256_hash,
        target_script=rel_script,
        target_args=script_args,
        pip_requirements=pip_req,
        run_id=run_id,
    )

    kernel_slug = config["kernel"]
    raw_title = kernel_slug.split("/")[-1].replace("-", " ").title()
    kernel_title = f"{raw_title} Task" if len(raw_title) < 5 else raw_title

    metadata = {
        "id": kernel_slug,
        "title": kernel_title,
        "code_file": "kgrun_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_internet": "true" if config["internet"] else "false",
    }
    if config.get("datasets"):
        metadata["dataset_sources"] = config["datasets"]

    staging_dir = Path(tempfile.mkdtemp(prefix="kgrun_staging_"))
    try:
        entry_file = staging_dir / "kgrun_entry.py"
        entry_file.write_text(bootstrap_code, encoding="utf-8")

        meta_file = staging_dir / "kernel-metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Push to Kaggle
        push_cmd = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
        if accelerator:
            push_cmd.extend(["--accelerator", accelerator])
        if timeout_seconds > 0:
            push_cmd.extend(["-t", str(timeout_seconds)])

        print(f"[kgrun] Pushing kernel {kernel_slug} (GPU: {accelerator or 'None'}, timeout: {effective_timeout_h}h)...")
        push_res = subprocess.run(push_cmd)
        if push_res.returncode != 0:
            print("[kgrun] Error: Failed to push kernel to Kaggle.", file=sys.stderr)
            return push_res.returncode

        history_entry = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kernel": kernel_slug,
            "script": rel_script,
            "args": script_args,
            "accelerator": accelerator,
            "bundle_sha256": sha256_hash,
        }
        append_history(root_dir, history_entry)

        if not no_follow:
            print(f"[kgrun] Streaming live logs for {kernel_slug} (press Ctrl+C to detach)...")
            log_code = run_command_interactive(["kaggle", "kernels", "logs", kernel_slug, "-f"])
            if log_code == 0:
                print(f"[kgrun] Execution log ended. Waiting for status finalization...")
                final_status = wait_for_completion(kernel_slug)
                print(f"[kgrun] Final status: {final_status}")
        else:
            print(f"[kgrun] Kernel pushed. Check progress with 'kgrun logs -f' or 'kgrun status'.")

        if pull:
            print("[kgrun] Pulling execution outputs...")
            pull_outputs(root_dir, kernel_slug)

        return 0

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def wait_for_completion(kernel_slug: str, max_wait: int = 90) -> str:
    """
    Poll kernel status until it is no longer RUNNING or max_wait seconds elapsed.
    """
    start = time.time()
    last_status = "UNKNOWN"
    while time.time() - start < max_wait:
        res = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (res.stdout + res.stderr).strip()
        last_status = out
        if "RUNNING" not in out and "QUEUED" not in out:
            return last_status
        time.sleep(3)
    return last_status


def pull_outputs(
    root_dir: Path | None = None,
    kernel_slug: str | None = None,
    pattern: str | None = None,
    target_dir: Path | None = None,
) -> int:
    """
    Download kernel output files into runs/<timestamp>/ without overwriting.
    """
    if root_dir is None or kernel_slug is None:
        config_path = find_config()
        if not config_path:
            print("[kgrun] Error: kgrun.toml not found.", file=sys.stderr)
            return 1
        root_dir = config_path.parent
        config = load_config(config_path)
        kernel_slug = config["kernel"]

    # Ensure worker has finalized before downloading output
    wait_for_completion(kernel_slug, max_wait=30)

    if target_dir is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        target_dir = root_dir / "runs" / timestamp_str

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kgrun] Downloading outputs from {kernel_slug} to {target_dir}...")

    cmd = ["kaggle", "kernels", "output", kernel_slug, "-p", str(target_dir)]
    if pattern:
        # Convert shell-style glob pattern to regex if necessary
        regex_pattern = pattern
        if pattern.startswith("*") or "?" in pattern or "[" in pattern:
            import fnmatch
            regex_pattern = fnmatch.translate(pattern)
        cmd.extend(["--file-pattern", regex_pattern])

    res = subprocess.run(cmd)
    if res.returncode == 0:
        downloaded = list(target_dir.glob("**/*"))
        file_count = sum(1 for p in downloaded if p.is_file())
        print(f"[kgrun] Successfully downloaded {file_count} file(s) into {target_dir}")
    return res.returncode


def show_logs(follow: bool = False) -> int:
    """
    Show logs for current project kernel.
    """
    config_path = find_config()
    if not config_path:
        print("[kgrun] Error: kgrun.toml not found.", file=sys.stderr)
        return 1

    config = load_config(config_path)
    kernel_slug = config["kernel"]
    cmd = ["kaggle", "kernels", "logs", kernel_slug]
    if follow:
        cmd.append("-f")
    return run_command_interactive(cmd)


def show_status() -> int:
    """
    Show current execution status for project kernel.
    """
    config_path = find_config()
    if not config_path:
        print("[kgrun] Error: kgrun.toml not found.", file=sys.stderr)
        return 1

    config = load_config(config_path)
    kernel_slug = config["kernel"]
    return subprocess.run(["kaggle", "kernels", "status", kernel_slug]).returncode

