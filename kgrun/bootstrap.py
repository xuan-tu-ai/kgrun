"""
Bootstrap template generator for Kaggle execution entrypoint.
"""

from __future__ import annotations

import json
from typing import Any


BOOTSTRAP_CODE = '''"""
kgrun bootstrap entrypoint executed on remote Kaggle container.
Extracts code payload, prepares runtime environment, sets up stamps, and runs target script.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

PAYLOAD_B64 = {payload_b64!r}
EXPECTED_SHA256 = {expected_sha256!r}
TARGET_SCRIPT = {target_script!r}
TARGET_ARGS = {target_args!r}
PIP_REQUIREMENTS = {pip_requirements!r}
RUN_ID = {run_id!r}


def log(msg: str) -> None:
    print(f"[kgrun] {{msg}}", flush=True)


def get_gpu_info() -> dict[str, Any]:
    gpu_data: dict[str, Any] = {{
        "torch_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "nvidia_smi": None,
    }}
    try:
        import torch
        gpu_data["torch_version"] = torch.__version__
        gpu_data["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            gpu_data["device_count"] = count
            gpu_data["device_names"] = [torch.cuda.get_device_name(i) for i in range(count)]
    except Exception as exc:
        log(f"Torch inspection notice: {{exc}}")

    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            gpu_data["nvidia_smi"] = res.stdout
    except Exception:
        pass

    return gpu_data


def extract_payload(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_tar = base64.b64decode(PAYLOAD_B64)
    actual_sha = hashlib.sha256(raw_tar).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(
            f"Payload SHA256 mismatch! Expected {{EXPECTED_SHA256}}, got {{actual_sha}}"
        )

    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=dest_dir, filter="data")
        else:
            tar.extractall(path=dest_dir)


def main() -> int:
    log(f"Starting run {{RUN_ID}} on Kaggle host")
    log(f"Python interpreter: {{sys.version}}")

    gpu_info = get_gpu_info()
    if gpu_info["cuda_available"]:
        log(f"Detected {{gpu_info['device_count']}} GPU(s): {{', '.join(gpu_info['device_names'])}}")
    else:
        log("No CUDA GPUs detected or torch not compiled with CUDA")

    src_dir = Path("/kaggle/tmp/src")
    extract_payload(src_dir)
    log(f"Extracted project files to {{src_dir}}")

    if PIP_REQUIREMENTS:
        req_path = src_dir / PIP_REQUIREMENTS
        if req_path.is_file():
            log(f"Installing dependencies from {{PIP_REQUIREMENTS}}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
                check=True,
            )

    kgrun_out = Path("/kaggle/working")
    kgrun_out.mkdir(parents=True, exist_ok=True)
    os.environ["KGRUN_OUT"] = str(kgrun_out)

    dataset_candidates = []
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        log(f"Inspecting {{input_root}} contents:")
        for entry in input_root.iterdir():
            log(f"  - {{entry}} (is_dir={{entry.is_dir()}})")
            if entry.is_dir():
                for sub in entry.iterdir():
                    log(f"      * {{sub}} (is_dir={{sub.is_dir()}})")

        for dirpath, _, filenames in os.walk(input_root):
            if filenames:
                dataset_candidates.append(dirpath)
                break

        if not dataset_candidates:
            for item in sorted(input_root.iterdir()):
                if item.is_dir():
                    if item.name.lower() == "datasets":
                        for sub in sorted(item.iterdir()):
                            if sub.is_dir():
                                dataset_candidates.append(str(sub))
                    else:
                        dataset_candidates.append(str(item))

    if dataset_candidates:
        os.environ["KGRUN_DATA"] = dataset_candidates[0]
        log(f"Set KGRUN_DATA={{dataset_candidates[0]}}")

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{{src_dir}}{{os.pathsep if current_pythonpath else ''}}{{current_pythonpath}}"

    stamp = {{
        "run_id": RUN_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": TARGET_SCRIPT,
        "args": TARGET_ARGS,
        "bundle_sha256": EXPECTED_SHA256,
        "python_version": sys.version,
        "gpu_info": gpu_info,
        "datasets": dataset_candidates,
    }}
    stamp_path = kgrun_out / "kgrun_stamp.json"
    stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    log(f"Wrote execution stamp to {{stamp_path}}")

    script_path = src_dir / TARGET_SCRIPT
    if not script_path.is_file():
        log(f"Error: Target script {{TARGET_SCRIPT}} not found in extracted bundle!")
        return 1

    cmd = [sys.executable, "-u", str(script_path), *TARGET_ARGS]
    log(f"Executing: {{' '.join(cmd)}} in {{src_dir}}")
    proc = subprocess.run(cmd, cwd=str(src_dir))
    log(f"Script process finished with exit code {{proc.returncode}}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
'''


def generate_bootstrap_script(
    payload_b64: str,
    expected_sha256: str,
    target_script: str,
    target_args: list[str],
    pip_requirements: str = "",
    run_id: str = "",
) -> str:
    """
    Format and return bootstrap Python script code with embedded payload.
    """
    return BOOTSTRAP_CODE.format(
        payload_b64=payload_b64,
        expected_sha256=expected_sha256,
        target_script=target_script,
        target_args=target_args,
        pip_requirements=pip_requirements,
        run_id=run_id,
    )

